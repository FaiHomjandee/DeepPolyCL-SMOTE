"""
utils.py

Core implementation of DeepPolyCL-SMOTE: a class-conditional contrastive
autoencoder (EncoderCL2 / DecoderCL2), a weighted centroid-guided contrastive
loss, latent-space oversampling via the `smote-variants` package, and the
downstream ResNet classifier training/evaluation that produces every
"DeepPolyCL-SMOTE" number reported in the paper's Table 1.

Requires: torch, torchvision, smote-variants, scikit-learn, scipy,
matplotlib (see requirements.txt).
"""

import os
import copy
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score, balanced_accuracy_score, recall_score, accuracy_score
from scipy.stats import gmean
import smote_variants as sv

CLASSIFIER_EPOCHS = 50


# ==============================================================================
# Model / loss definitions (verbatim from utils_DeepPolyCLSMOTE.py)
# ==============================================================================

class EncoderCL2(nn.Module):
    """Shared conv trunk + one linear head per class (class-conditional latent)."""

    def __init__(self, model_args):
        super().__init__()
        self.n_z = model_args['n_z']
        self.dim_h = model_args['dim_h']
        self.num_class = model_args['num_class']

        self.conv = nn.Sequential(
            nn.Conv2d(model_args['n_channel'], self.dim_h, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.dim_h, self.dim_h * 2, 4, 2, 1),
            nn.BatchNorm2d(self.dim_h * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.dim_h * 2, self.dim_h * 4, 3, 1, 0),
            nn.BatchNorm2d(self.dim_h * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.dim_h * 4, self.dim_h * 8, 3, 1, 0),
            nn.BatchNorm2d(self.dim_h * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Conv stack reduces a 28x28 input to a 3x3 spatial map, and a 32x32
        # input (CIFAR-10's native resolution) to 4x4 — same layers, just a
        # different bottleneck size, so this must match the actual input
        # resolution rather than being hardcoded to the 28x28 case.
        self.bottleneck = model_args.get('bottleneck_size', 3)
        flat_dim = self.dim_h * 8 * self.bottleneck * self.bottleneck

        self.fc_main = nn.Linear(flat_dim, self.n_z)

        self.fc_class = nn.ModuleList([
            nn.Linear(flat_dim, self.n_z)
            for _ in range(self.num_class)
        ])
        for c in range(self.num_class):
            nn.init.xavier_uniform_(self.fc_class[c].weight)
            nn.init.zeros_(self.fc_class[c].bias)

    def forward(self, x, labels):
        features = self.conv(x)
        flat_features = torch.flatten(features, start_dim=1)

        latent_z_mixed = self.fc_main(flat_features)

        list_class_latent = []
        for c in range(self.num_class):
            mask = (labels == c)
            class_flat_features = flat_features[mask]
            if class_flat_features.shape[0] > 0:
                class_z = self.fc_class[c](class_flat_features)
            else:
                class_z = torch.tensor([], device=x.device)
            list_class_latent.append(class_z)

        return latent_z_mixed, list_class_latent


class DecoderCL2(nn.Module):
    def __init__(self, model_args):
        super().__init__()
        self.dim_h = model_args['dim_h']
        self.n_z = model_args['n_z']
        self.n_channel = model_args['n_channel']
        self.bottleneck = model_args.get('bottleneck_size', 3)
        self.fc = nn.Sequential(
            nn.Linear(self.n_z, self.dim_h * 8 * self.bottleneck * self.bottleneck), nn.ReLU()
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(self.dim_h * 8, self.dim_h * 4, 3, 1, 0), nn.BatchNorm2d(self.dim_h * 4), nn.ReLU(True),
            nn.ConvTranspose2d(self.dim_h * 4, self.dim_h * 2, 3, 1, 0), nn.BatchNorm2d(self.dim_h * 2), nn.ReLU(True),
            nn.ConvTranspose2d(self.dim_h * 2, self.dim_h, 4, 2, 1), nn.BatchNorm2d(self.dim_h), nn.ReLU(True),
            nn.ConvTranspose2d(self.dim_h, self.n_channel, 4, 2, 1), nn.Tanh(),
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, self.dim_h * 8, self.bottleneck, self.bottleneck)
        return self.deconv(x)


class ProjectionHead(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class WeightedMaskedNTXent(nn.Module):
    """Class-weighted, class-masked NT-Xent loss between the two latent views."""

    def __init__(self):
        super().__init__()

    def forward(self, z_i, z_j, labels, temperature, class_weights):
        device = z_i.device
        labels_orig = labels.long().to(device)
        class_weights = class_weights.to(device)
        B = z_i.size(0)

        z = torch.cat([z_i, z_j], dim=0)
        z = F.normalize(z, dim=1)

        sim = torch.matmul(z, z.T) / temperature
        sim = sim - torch.max(sim, dim=1, keepdim=True)[0].detach()

        labels_all = torch.cat([labels_orig, labels_orig], dim=0)
        mask = torch.eq(labels_all.unsqueeze(1), labels_all.unsqueeze(0)).float().to(device)

        self_mask = torch.eye(2 * B, device=device)
        mask = mask - self_mask

        exp_sim = torch.exp(sim) * (1 - self_mask)
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        pos_count = mask.sum(1).clamp(min=1)
        mean_log_prob_pos = (mask * log_prob).sum(1) / pos_count

        sample_weights = class_weights[labels_all]
        loss = -(sample_weights * mean_log_prob_pos).mean()
        loss = loss / (2 * B)
        return loss


class ResNet18_28x28(nn.Module):
    """torchvision ResNet18 adapted for small (28x28-ish) single/multi-channel images."""

    def __init__(self, num_classes, n_channel, dropout_rate=0.5):
        super().__init__()
        self.resnet = models.resnet18(weights=None)
        self.resnet.conv1 = nn.Conv2d(n_channel, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet.maxpool = nn.Identity()

        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(num_ftrs, num_classes),
        )

    def forward(self, x):
        return self.resnet(x)


# ==============================================================================
# Helpers
# ==============================================================================

def biased_get_class1(class_id, data, target):
    mask = (target == class_id)
    return data[mask], target[mask]


def extract_latents(encoder, loader, device):
    encoder.eval()
    Z_all, Y_all = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            z_hat, _ = encoder(images, labels)
            Z_all.append(z_hat.cpu())
            Y_all.append(labels.cpu())
    return torch.cat(Z_all, dim=0), torch.cat(Y_all, dim=0)


def get_class_weights(labels_list, Z=None, num_classes=10, method="hybrid", beta=0.999, eps=1e-6):
    """
    method: "inverse", "sqrt", "effective", "variance", "hybrid", "effective_variance"
    Z is required for "variance", "hybrid", "effective_variance".
    """
    if isinstance(labels_list, torch.Tensor):
        labels = labels_list.detach().cpu().long().numpy()
    else:
        labels = np.array([int(item[-1]) if isinstance(item, (list, tuple, np.ndarray))
                            else int(item.item()) if isinstance(item, torch.Tensor)
                            else int(item)
                            for item in labels_list])

    counts = Counter(labels)
    counts_tensor = torch.tensor([counts.get(i, 1) for i in range(num_classes)], dtype=torch.float)

    if method == "inverse":
        weights = 1.0 / counts_tensor
    elif method == "sqrt":
        weights = 1.0 / torch.sqrt(counts_tensor)
    elif method == "effective":
        effective_num = (1.0 - torch.pow(beta, counts_tensor)) / (1.0 - beta)
        weights = 1.0 / effective_num
    elif method in ("variance", "hybrid", "effective_variance"):
        if Z is None:
            raise ValueError(f"Z must be provided for '{method}' method")
        Z = Z.detach().cpu() if isinstance(Z, torch.Tensor) else Z
        weights = []
        for c in range(num_classes):
            zc = Z[labels == c]
            if len(zc) < 2:
                weights.append(0.0)
                continue
            var_c = torch.mean((zc - zc.mean(dim=0)) ** 2)
            if method == "variance":
                w = 1.0 / (var_c + eps)
            elif method == "hybrid":
                w = (1.0 / counts_tensor[c]) * (1.0 / (var_c + eps))
            else:  # effective_variance
                eff_num = (1.0 - torch.pow(beta, counts_tensor[c])) / (1.0 - beta)
                w = (1.0 / eff_num) * (1.0 / (var_c + eps))
            weights.append(w.item())
        weights = torch.tensor(weights)
    else:
        raise ValueError(f"Unknown method '{method}'")

    return weights / weights.mean()


def build_smote_sampler(variant, proportion, random_state=42, **extra_kwargs):
    """
    Instantiate a `smote-variants` sampler by class name, e.g.:
        build_smote_sampler('polynom_fit_SMOTE_poly', proportion=0.6)
        build_smote_sampler('polynom_fit_SMOTE_mesh', proportion=0.6)
        build_smote_sampler('SMOTE', proportion=0.6)
        build_smote_sampler('polynom_fit_SMOTE', proportion=0.6, topology='poly_2')

    Raises a clear error (instead of silently falling back to a different
    sampler) if the requested class name doesn't exist in the installed
    smote-variants version.
    """
    if not hasattr(sv, variant):
        candidates = sorted(name for name in dir(sv) if 'SMOTE' in name or 'polynom' in name)
        raise ValueError(
            f"smote_variants has no class '{variant}'. Available SMOTE-like classes: {candidates}"
        )
    sampler_cls = getattr(sv, variant)
    return sampler_cls(proportion=proportion, random_state=random_state, **extra_kwargs)


# ==============================================================================
# Main pipeline: one (hyperparams, fold) -> (f1, asca, g_mean, per_class_results)
# ==============================================================================

def run_deeppoly_clsmote(
    params,
    fold_data_loaders,
    imbalanced_train_list,
    model_args,
    smote_variant,
    smote_kwargs=None,
    output_root="./runs",
):
    """
    params: dict with lr, alpha, temperature, n_z, weight_decay, warmup_epochs,
            dataset_name, epochs, update_weight_every (optional, default 200)
    fold_data_loaders: (train_loader_ae, val_loader, test_loader)
    imbalanced_train_list: list of (image_tensor, label) for the imbalanced train split
    model_args: dict with dim_h, n_channel, num_class, batch_size
    smote_variant: smote-variants class name, e.g. 'polynom_fit_SMOTE_poly'
    smote_kwargs: extra kwargs forwarded to the sampler constructor (e.g. {'topology': 'poly_2'})
    output_root: base directory for generated images / t-SNE plots / tensorboard runs
    """
    smote_kwargs = smote_kwargs or {}

    lr = params['lr']
    alpha = params['alpha']
    temperature = params['temperature']
    weight_decay = params['weight_decay']
    warmup_epochs = params['warmup_epochs']
    dataset_name = params['dataset_name']
    epochs = params.get('epochs', 200)
    update_weight_every = params.get('update_weight_every', 200)
    # Ablation toggle: False -> use uniform (all-ones) class weights instead of
    # the effective-number/variance weighting, isolating the contribution of
    # the weighting mechanism from the centroid-guided contrastive loss itself.
    use_class_weighting = params.get('use_class_weighting', True)

    model_args = dict(model_args)
    model_args['n_z'] = params['n_z']

    train_loader_ae, val_loader, test_loader = fold_data_loaders
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    run_dir = os.path.join(output_root, dataset_name, smote_variant)

    # ---------------- Stage A: train class-conditional contrastive autoencoder ----------------
    print(f"--- Stage A: Training autoencoder (warmup_epochs={warmup_epochs}) ---")

    encoder = EncoderCL2(model_args).to(device)
    decoder = DecoderCL2(model_args).to(device)
    proj_head = ProjectionHead(model_args['n_z'], model_args['n_z'], model_args['n_z']).to(device)

    criterion_mse = nn.MSELoss().to(device)
    criterion_nt = WeightedMaskedNTXent().to(device)

    enc_optim = torch.optim.Adam(list(encoder.parameters()) + list(proj_head.parameters()),
                                  lr=lr, weight_decay=weight_decay)
    dec_optim = torch.optim.Adam(decoder.parameters(), lr=lr, weight_decay=weight_decay)

    class_weights = None
    best_loss = np.inf
    best_encoder_state, best_decoder_state = None, None

    for epoch in range(epochs):
        encoder.train()
        decoder.train()

        train_loss_epoch = mse_loss_epoch = contrastive_loss_epoch = 0.0
        num_batches = 0

        if epoch == warmup_epochs or (epoch > warmup_epochs and epoch % update_weight_every == 0):
            if use_class_weighting:
                print(f"\nComputing class weights at epoch {epoch} ...")
                latent_all, label_all = extract_latents(encoder, train_loader_ae, device)
                class_weights = get_class_weights(
                    labels_list=label_all, Z=latent_all,
                    num_classes=model_args['num_class'], method="effective_variance",
                ).to(device)
            else:
                class_weights = torch.ones(model_args['num_class'], device=device)
            print(f"Class weights: {class_weights}")

        for images, labs in train_loader_ae:
            images, labs = images.to(device), labs.to(device)

            z_hat, list_class_latent = encoder(images, labs)
            x_hat = decoder(z_hat)
            mse = criterion_mse(x_hat, images)

            view1 = F.normalize(z_hat, dim=1)
            view2 = torch.zeros_like(z_hat)
            for c in range(model_args['num_class']):
                mask = (labs == c)
                if mask.sum() > 0:
                    view2[mask] = list_class_latent[c]
            view2 = F.normalize(view2, dim=1)

            h1 = F.normalize(proj_head(view1), dim=1)
            h2 = F.normalize(proj_head(view2), dim=1)

            if epoch < warmup_epochs:
                if use_class_weighting:
                    epoch_weights = get_class_weights(labs, num_classes=model_args['num_class'],
                                                       method="effective", beta=0.99).to(device)
                else:
                    epoch_weights = torch.ones(model_args['num_class'], device=device)
                loss_nt_xent = criterion_nt(h1, h2, labs, temperature, epoch_weights)
            else:
                loss_nt_xent = criterion_nt(h1, h2, labs, temperature, class_weights)

            loss = alpha * mse + (1 - alpha) * loss_nt_xent

            enc_optim.zero_grad()
            dec_optim.zero_grad()
            loss.backward()
            enc_optim.step()
            dec_optim.step()

            train_loss_epoch += loss.item()
            mse_loss_epoch += mse.item()
            contrastive_loss_epoch += loss_nt_xent.item()
            num_batches += 1

        avg_train_loss = train_loss_epoch / num_batches
        avg_mse_loss = mse_loss_epoch / num_batches
        avg_contrastive_loss = contrastive_loss_epoch / max(1, num_batches)

        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == warmup_epochs:
            tag = "Warmup" if epoch < warmup_epochs else "Epoch"
            print(f"[{tag} {epoch + 1}/{epochs}] Loss={avg_train_loss:.4f} | "
                  f"MSE={avg_mse_loss:.4f} | Contrastive={avg_contrastive_loss:.4f}")

        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
            best_encoder_state = copy.deepcopy(encoder.state_dict())
            best_decoder_state = copy.deepcopy(decoder.state_dict())

    print(f"Best autoencoder loss: {best_loss:.4f}")

    if best_encoder_state is not None:
        encoder.load_state_dict(best_encoder_state)
        decoder.load_state_dict(best_decoder_state)
    encoder.eval()
    decoder.eval()

    # ---------------- Stage B: latent-space SMOTE oversampling ----------------
    print(f"--- Stage B: Generating SMOTE data (variant='{smote_variant}') ---")

    imbal_goal = Counter([label for _, label in imbalanced_train_list])
    print(f"Imbalanced distribution: {imbal_goal}")

    data = torch.stack([item[0] for item in imbalanced_train_list])
    target = torch.tensor([item[1] for item in imbalanced_train_list])

    gen_img_dir = os.path.join(run_dir, "gen_img")
    all_synthetic_images, all_synthetic_labels = [], []

    with torch.no_grad():
        latent_all, _ = encoder(data.to(device), target.to(device))
        X_all = latent_all.cpu().numpy()
        y_all = target.numpy()

        for i in range(1, model_args['num_class']):
            _, yclass_labels = biased_get_class1(i, data, target)
            if len(yclass_labels) == 0:
                continue

            y_bin = (y_all == i).astype(int)
            n_min = int(y_bin.sum())
            if n_min < 2:
                continue

            n_to_gen = abs(imbal_goal[0] - n_min)
            if n_to_gen <= 0:
                continue

            proportion = n_to_gen / n_min
            sampler = build_smote_sampler(smote_variant, proportion=proportion, **smote_kwargs)
            X_res, y_res = sampler.sample(X_all, y_bin)

            X_syn = X_res[len(X_all):]
            y_syn = y_res[len(y_bin):]
            xsamp = X_syn[y_syn == 1][:n_to_gen]
            ysamp = np.full(len(xsamp), i)

            if xsamp.shape[0] == 0:
                continue

            generated_images = decoder(torch.tensor(xsamp, dtype=torch.float32).to(device))

            class_dir = os.path.join(gen_img_dir, f'class_{i}')
            os.makedirs(class_dir, exist_ok=True)
            num_to_save = min(10, generated_images.shape[0])
            for img_idx in range(num_to_save):
                single_image = generated_images[img_idx]
                img_min, img_max = single_image.min(), single_image.max()
                normalized_image = (single_image - img_min) / (img_max - img_min)
                np_image = normalized_image.cpu().permute(1, 2, 0).numpy()
                fig, ax = plt.subplots()
                ax.imshow(np_image, cmap='gray' if model_args['n_channel'] == 1 else None)
                ax.axis('off')
                plt.savefig(os.path.join(class_dir, f'generated_img_{img_idx + 1}.png'),
                            dpi=300, bbox_inches='tight', pad_inches=0)
                plt.close(fig)

            print(f"Saved {num_to_save} images for class {i} in '{class_dir}/'")
            all_synthetic_images.append(generated_images.cpu())
            all_synthetic_labels.append(torch.tensor(ysamp))

    if all_synthetic_images:
        synthetic_data = torch.cat(all_synthetic_images, dim=0)
        synthetic_labels = torch.cat(all_synthetic_labels, dim=0)
        balanced_data = torch.cat([data, synthetic_data], dim=0)
        balanced_labels = torch.cat([target, synthetic_labels], dim=0)
    else:
        balanced_data, balanced_labels = data, target

    counts = {i: (balanced_labels == i).sum().item() for i in range(model_args['num_class'])}
    print(f"Balanced distribution: {Counter(counts)}")

    # ---------------- Stage C: t-SNE visualization ----------------
    print("--- Stage C: Creating t-SNE visualization ---")
    tsne_dir = os.path.join(run_dir, f"tsne_warmup{warmup_epochs}")
    os.makedirs(tsne_dir, exist_ok=True)

    X_tsne = balanced_data.view(balanced_data.shape[0], -1).numpy()
    y_tsne = balanced_labels.numpy()
    X_embedded = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(X_tsne)

    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y_tsne, cmap='tab10', alpha=0.6, s=20)
    plt.colorbar(scatter, label='Class')
    plt.title('t-SNE: Balanced Dataset (Original + Synthetic)')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(tsne_dir, 'tsne_balanced_dataset.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"t-SNE plot saved to: {tsne_dir}/tsne_balanced_dataset.png")

    # ---------------- Prepare balanced train loader ----------------
    # Matches the original pipeline: only 80% of the balanced (real + synthetic)
    # data is used for training; the other 20% is not used for anything (model
    # selection uses the real held-out `val_loader` from fold_data_loaders,
    # not a split of the synthetic-inclusive balanced_dataset).
    balanced_dataset = TensorDataset(balanced_data, balanced_labels)
    train_size = int(0.8 * len(balanced_dataset))
    val_size = len(balanced_dataset) - train_size
    train_subset, _ = random_split(
        balanced_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    balanced_train_loader = DataLoader(train_subset, batch_size=model_args['batch_size'], shuffle=True)

    # ---------------- Stage D: train + evaluate final ResNet classifier ----------------
    print("--- Stage D: Training and Evaluating Final Classifier ---")

    final_classifier = ResNet18_28x28(
        num_classes=model_args['num_class'],
        n_channel=model_args['n_channel'],
        dropout_rate=0.5,
    ).to(device)

    # Same classifier training recipe for every dataset, matching the recipe
    # used to evaluate every baseline (BAGAN/GAMO/DeepSMOTE/DeepCLSMOTE/SMOTE
    # in defs_CoSMO.py and main_SMOTE_baseline.py), so the comparison in
    # Table 1 isolates the oversampling method rather than the classifier.
    classifier_criterion = nn.CrossEntropyLoss(label_smoothing=0.0)
    classifier_optimizer = torch.optim.SGD(
        final_classifier.parameters(), lr=0.01, momentum=0.9,
        weight_decay=0.05, nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        classifier_optimizer, mode='min', factor=0.5, patience=5,
    )
    step_scheduler_on_val_loss = True

    log_name = f"lr{params['lr']}_alpha{alpha}_wd{weight_decay}_temp{temperature}_z{model_args['n_z']}"
    writer = SummaryWriter(log_dir=os.path.join(run_dir, f"{log_name}_warmup{warmup_epochs}"))

    val_losses, val_accuracies = [], []
    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(CLASSIFIER_EPOCHS):
        final_classifier.train()
        epoch_train_loss = epoch_train_correct = epoch_train_total = 0

        for imgs, labs in balanced_train_loader:
            imgs, labs = imgs.to(device), labs.to(device)
            classifier_optimizer.zero_grad()
            output = final_classifier(imgs)
            loss = classifier_criterion(output, labs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(final_classifier.parameters(), max_norm=1.0)
            classifier_optimizer.step()

            epoch_train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            epoch_train_total += labs.size(0)
            epoch_train_correct += (predicted == labs).sum().item()

        avg_train_loss = epoch_train_loss / len(balanced_train_loader)
        train_accuracy = 100 * epoch_train_correct / epoch_train_total
        writer.add_scalar("Accuracy/Train", train_accuracy, epoch + 1)

        final_classifier.eval()
        epoch_val_loss = correct = total = 0
        with torch.no_grad():
            for imgs, labs in val_loader:
                imgs, labs = imgs.to(device), labs.to(device)
                output = final_classifier(imgs)
                loss = classifier_criterion(output, labs)
                epoch_val_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += labs.size(0)
                correct += (predicted == labs).sum().item()

        avg_val_loss = epoch_val_loss / len(val_loader)
        val_accuracy = 100 * correct / total
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)

        writer.add_scalar("Loss/Train", avg_train_loss, epoch + 1)
        writer.add_scalar("Loss/Val", avg_val_loss, epoch + 1)
        writer.add_scalar("Accuracy/Val", val_accuracy, epoch + 1)

        scheduler.step(avg_val_loss) if step_scheduler_on_val_loss else scheduler.step()

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = copy.deepcopy(final_classifier.state_dict())

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{CLASSIFIER_EPOCHS}] "
                  f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
                  f"Val Acc: {val_accuracy:.2f}%")

    writer.close()
    final_classifier.load_state_dict(best_model_state)
    best_epoch_idx = val_losses.index(best_val_loss)

    # ---------------- Final evaluation on the held-out test set ----------------
    final_classifier.eval()
    all_true, all_pred = [], []
    with torch.no_grad():
        for imgs, labs in test_loader:
            imgs, labs = imgs.to(device), labs.to(device)
            output = final_classifier(imgs)
            _, predicted = torch.max(output.data, 1)
            all_true.append(labs.cpu())
            all_pred.append(predicted.cpu())

    true_labels = torch.cat(all_true).numpy()
    predicted_labels = torch.cat(all_pred).numpy()

    f1 = f1_score(true_labels, predicted_labels, average='macro', zero_division=0)
    asca = balanced_accuracy_score(true_labels, predicted_labels)
    per_class_recalls = recall_score(true_labels, predicted_labels, average=None, zero_division=0)
    g_mean = gmean(per_class_recalls)

    per_class_results = {}
    for c in np.unique(true_labels):
        idx = true_labels == c
        per_class_results[int(c)] = {
            'accuracy': accuracy_score(true_labels[idx], predicted_labels[idx]),
            'sample_count': int(idx.sum()),
        }

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print(f"Best Val Loss: {best_val_loss:.4f} | Best Val Accuracy: {val_accuracies[best_epoch_idx]:.2f}%")
    print(f"F1={f1:.4f}  ASCA={asca:.4f}  GM={g_mean:.4f}")
    print(per_class_results)
    print("=" * 60)

    return f1, asca, g_mean, per_class_results
