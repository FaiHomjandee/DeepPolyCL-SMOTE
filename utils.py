# utils.py file
# ==============================================================================
# PART 1: IMPORTS and SETUP
# ==============================================================================
from xml.parsers.expat import model
from sklearn.neighbors import NearestNeighbors
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset, random_split,Dataset
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import os
from sklearn.metrics import classification_report
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import recall_score
from scipy.stats import gmean
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from collections import Counter
import numpy as np
from sklearn.model_selection import StratifiedKFold
import random
import torch
import copy
import time
import matplotlib.pyplot as plt
import torchvision.models as models    
from sklearn.decomposition import PCA 
import sys
from torch.utils.data import WeightedRandomSampler
import smote_variants as sv

 # --- Global Args Dictionary ---
args = {}
args['dim_h'] = 64
args['n_channel'] = 3
#args['n_z'] = 200
args['num_class'] = 7
args['epochs'] = 250 # Use fewer epochs for grid search to save time
args['batch_size'] = 256
args['img_size'] = 28

CLSSIFIER_EPOCHS = 50 #100 


# ==============================================================================
# HELPER FUNCTIONS and SelfCLSMOTE MODEL CLASS DEFINITIONS
# ==============================================================================
def calculate_all_metrics(true_labels, predicted_labels, num_classes= args['num_class']):
    """
    Calculates a comprehensive set of metrics from true and predicted labels.

    Args:
        true_labels (list or np.array): The ground truth labels.
        predicted_labels (list or np.array): The labels predicted by the model.
        num_classes (int): The total number of classes in the dataset.

    Returns:
        dict: A dictionary containing the confusion matrix and macro-averaged metrics.
    """
    
    # Ensure all classes are represented in the confusion matrix, even if not present in the batch
    labels = list(range(num_classes))
    cm = confusion_matrix(true_labels, predicted_labels, labels=labels)

    # Calculate TP, FP, FN, TN for each class from the confusion matrix
    tp = np.diag(cm)
    fp = np.sum(cm, axis=0) - tp
    fn = np.sum(cm, axis=1) - tp
    tn = np.sum(cm) - (tp + fp + fn)

    # Calculate per-class metrics, handling division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        class_accuracy = (tp + tn) / (tp + fp + fn + tn)
        class_sensitivity = tp / (tp + fn)  # Also known as Recall
        class_specificity = tn / (tn + fp)
        class_precision = tp / (tp + fp)
        class_f1 = (2 * class_precision * class_sensitivity) / (class_precision + class_sensitivity)

    # Replace NaN values (from division by zero) with 0.0
    class_accuracy = np.nan_to_num(class_accuracy)
    class_sensitivity = np.nan_to_num(class_sensitivity)
    class_precision = np.nan_to_num(class_precision)
    class_f1 = np.nan_to_num(class_f1)
    
    # Calculate macro-averaged metrics by taking the mean of per-class metrics
    macro_accuracy = np.mean(class_accuracy)
    macro_precision = np.mean(class_precision)
    macro_sensitivity = np.mean(class_sensitivity)
    macro_f1 = np.mean(class_f1)
    
    # Store all results in a dictionary
    metrics = {
        'accuracy': macro_accuracy,
        'precision': macro_precision,
        'sensitivity': macro_sensitivity,
        'f1_score': macro_f1,
        'confusion_matrix': cm
    }
    
    return metrics

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

def biased_get_class1(c, data, target):
    mask = (target == c)
    return data[mask], target[mask]


class AugmentedTensorDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels, dataset_name='cifar10', train=True):
        self.data = data
        self.labels = labels
        self.train = train
        self.dataset_name = dataset_name.lower()

        if self.dataset_name == 'cifar10':
            mean = (0.4914, 0.4822, 0.4465)
            std = (0.2470, 0.2435, 0.2616)
        elif self.dataset_name == 'cifar100':
            mean = (0.5071, 0.4867, 0.4408)
            std = (0.2675, 0.2565, 0.2761)
        else:
            raise ValueError(f"Unsupported dataset_name: {dataset_name}")

        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

        if train:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]

        # Convert normalized tensor back to image space before augmentation
        x = x * self.std + self.mean
        x = torch.clamp(x, 0.0, 1.0)

        x = self.transform(x)
        return x, y

     
class EncoderCL(nn.Module):
    def __init__(self, args):
        super(EncoderCL32, self).__init__()
        self.n_z = args['n_z']
        self.dim_h = args['dim_h']
        self.num_class = args['num_class']

        # For 32x32 input:
        # 32 -> 16 -> 8 -> 6 -> 4
        self.conv = nn.Sequential(
            nn.Conv2d(args['n_channel'], self.dim_h, 4, 2, 1),
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

        self.fc_main = nn.Linear(self.dim_h * 8 * 4 * 4, self.n_z)

        self.fc_class = nn.ModuleList([
            nn.Linear(self.dim_h * 8 * 4 * 4, self.n_z)
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
                class_z = torch.empty((0, self.n_z), device=x.device)

            list_class_latent.append(class_z)

        return latent_z_mixed, list_class_latent


class DecoderCL(nn.Module):
    def __init__(self, args):
        super(DecoderCL32, self).__init__()
        self.dim_h = args['dim_h']
        self.n_z = args['n_z']
        self.n_channel = args['n_channel']

        self.fc = nn.Sequential(
            nn.Linear(self.n_z, self.dim_h * 8 * 4 * 4),
            nn.ReLU()
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(self.dim_h * 8, self.dim_h * 4, 3, 1, 0),
            nn.BatchNorm2d(self.dim_h * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.dim_h * 4, self.dim_h * 2, 3, 1, 0),
            nn.BatchNorm2d(self.dim_h * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.dim_h * 2, self.dim_h, 4, 2, 1),
            nn.BatchNorm2d(self.dim_h),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.dim_h, self.n_channel, 4, 2, 1),
            # nn.Tanh()
            # nn.Sigmoid()
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, self.dim_h * 8, 4, 4)
        x = self.deconv(x)
        return x

class ProjectionHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=None, out_dim=None):
        super(ProjectionHead, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)

class masked_nt_xent(nn.Module):
    def __init__(self):
        super(masked_nt_xent, self).__init__()

    def forward(self, z_i, z_j, labels, temperature):
        B = z_i.size(0)

        z = torch.cat([z_i, z_j], dim=0)          # [2B, D]
        z = F.normalize(z, dim=1)

        sim = torch.matmul(z, z.T) / temperature

        # numerical stability
        sim = sim - torch.max(sim, dim=1, keepdim=True)[0].detach()

        labels = labels.view(-1, 1)
        labels = torch.cat([labels, labels], dim=0)

        # positive mask (same class)
        mask = torch.eq(labels, labels.T).float().to(z.device)

        # remove self-contrast
        self_mask = torch.eye(2 * B, device=z.device)
        mask = mask - self_mask

        # remove self from denominator
        exp_sim = torch.exp(sim) * (1 - self_mask)

        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1).clamp(min=1)

        loss = -mean_log_prob_pos.mean()
        
        print(f"Mean log-prob of positives: {mean_log_prob_pos.mean().item():.4f}")
        return loss

class weighted_masked_nt_xent(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z_i, z_j, labels, temperature, class_weights):
        """
        z_i, z_j: [B, D]
        labels: [B]
        class_weights: [num_classes]
        """

        device = z_i.device

        labels_orig = labels.long().to(device)  # [B]
        class_weights = class_weights.to(device)

        B = z_i.size(0)

        # Concatenate views
        z = torch.cat([z_i, z_j], dim=0)        # [2B, D]
        z = F.normalize(z, dim=1)

        # Similarity matrix
        sim = torch.matmul(z, z.T) / temperature
        sim = sim - torch.max(sim, dim=1, keepdim=True)[0].detach()

        # Build labels for 2B
        labels_all = torch.cat([labels_orig, labels_orig], dim=0)  # [2B]

        # Positive mask (same class)
        mask = torch.eq(
            labels_all.unsqueeze(1),
            labels_all.unsqueeze(0)
        ).float().to(device)

        # Remove self-comparisons
        self_mask = torch.eye(2*B, device=device)
        mask = mask - self_mask

        # # Log-softmax denominator
        exp_sim = torch.exp(sim) * (1 - self_mask)
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        # log_prob = sim - torch.logsumexp(sim.masked_fill(self_mask.bool(), -1e9), dim=1, keepdim=True)

        # Mean log-prob over positives
        pos_count = mask.sum(1).clamp(min=1)
        mean_log_prob_pos = (mask * log_prob).sum(1) / pos_count

        # Class weights per sample
        sample_weights = class_weights[labels_all]
        # sample_weights = sample_weights / sample_weights.mean()

        # Final weighted loss
        # sample_weights = 1
        loss = - (sample_weights * mean_log_prob_pos).mean()
        # print(f"Mean log-prob of positives: {mean_log_prob_pos.mean().item():.4f}")

        loss = loss / (2*B) #math.log(2 * B)
        # print(f"loss: {loss.item():.4f}")


def macro_accuracy(true_labels, pred_labels):
    """
    Calculate macro-averaged accuracy using confusion matrix.
    Works with both torch tensors and numpy arrays.
    
    Args:
        true_labels: Ground truth labels (tensor or numpy array)
        pred_labels: Predicted labels (tensor or numpy array)
    
    Returns:
        float: Macro-averaged accuracy across all classes
    """
    # Convert to numpy arrays if tensors
    if isinstance(true_labels, torch.Tensor):
        true_labels = true_labels.detach().cpu().numpy()
    else:
        true_labels = np.array(true_labels)
    
    if isinstance(pred_labels, torch.Tensor):
        pred_labels = pred_labels.detach().cpu().numpy()
    else:
        pred_labels = np.array(pred_labels)
    
    # Calculate confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    class_accuracies = []
    
    # Calculate per-class accuracy
    for i in range(cm.shape[0]):
        tp = cm[i, i]  # True Positives
        fp = np.sum(cm[:, i]) - tp  # False Positives
        fn = np.sum(cm[i, :]) - tp  # False Negatives
        tn = np.sum(cm) - tp - fp - fn  # True Negatives
        
        # Class accuracy = (TP + TN) / (TP + FP + FN + TN)
        class_accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0
        class_accuracies.append(class_accuracy)
    
    # Return macro average
    return np.mean(class_accuracies)
    
def extract_latents(model, loader, device):
    model.eval()
    Z_all, Y_all = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            z_hat , _= model(images,labels)
            Z_all.append(z_hat.cpu())
            Y_all.append(labels.cpu())
            
    return torch.cat(Z_all, dim=0), torch.cat(Y_all, dim=0)

def DeepPolyCL_SMOTE(params, fold_data_loaders, imbalanced_train_list):
    
    """
    Runs the entire experiment for a single fold with a given set of hyperparameters.
    """
    
    lr = params['lr']
    alpha = params['alpha']
    temperature = params['temperature']
    args['n_z'] = params['n_z']
    weight_decay = params['weight_decay']

    warmup_epochs = params['warmup_epochs']  
    dataset_name = params['dataset_name']
    update_weight_every = params.get("update_weight_every", 200)

    train_loader_ae, val_loader, test_loader = fold_data_loaders
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Using warmup_epochs = {warmup_epochs}")

    # ---------------- Models ----------------
    encoder = EncoderCL32(args).to(device)
    decoder = DecoderCL32(args).to(device)

    proj_head = ProjectionHead(
        in_dim=args['n_z'],
        hidden_dim=args['n_z'],
        out_dim=args['n_z']
    ).to(device)

    criterion_mse = nn.MSELoss().to(device)
    criterion_nt = weighted_masked_nt_xent().to(device)

    enc_optim = torch.optim.Adam(
        list(encoder.parameters()) + list(proj_head.parameters()),
        lr=lr,
        weight_decay=weight_decay
    )
    dec_optim = torch.optim.Adam(decoder.parameters(), lr=lr, weight_decay=weight_decay)

    # ---------------- Get labels ----------------
    if hasattr(train_loader_ae.dataset, 'targets'):
        raw_labels = train_loader_ae.dataset.targets
    elif hasattr(train_loader_ae.dataset, 'labels'):
        raw_labels = train_loader_ae.dataset.labels
    else:
        raw_labels = [label for _, label in train_loader_ae.dataset]

    class_weights = None   # initialized later

    best_loss = np.inf
    best_encoder_state = None
    best_decoder_state = None

    
    # =========================================================
    # Training Loop
    # =========================================================
    for epoch in range(args['epochs']):

        encoder.train()
        decoder.train()

        train_loss_epoch = 0.0
        mse_loss_epoch = 0.0
        contrastive_loss_epoch = 0.0
        num_batches = 0

        # ---- Compute class weights AFTER warmup ----
        if epoch == warmup_epochs or (epoch > warmup_epochs and epoch % update_weight_every == 200):

            print(f"\n🔁 Computing class weights at epoch {epoch} ...")

            latent_all, label_all = extract_latents(encoder, train_loader_ae, device)

            class_weights = get_class_weights(
                labels_list=label_all,
                Z=latent_all,
                num_classes=args['num_class'],
                method="effective_variance",   
            ).to(device)

            print(f"Class weights: {class_weights}")

        for images, labs in train_loader_ae:

            images, labs = images.to(device), labs.to(device)
            # Check the range of pixel values in the batch
            # imgs, _ = next(iter(train_loader_ae))
            # print(imgs.min().item(), imgs.max().item())
            z_hat, list_class_latent = encoder(images, labs)
            x_hat = decoder(z_hat)
            # print("images:", images.min().item(), images.max().item())
            # print("x_hat:", x_hat.min().item(), x_hat.max().item())
            # print(f"images mean/std: {images.mean().item():.4f}, {images.std().item():.4f}")
            # print(f"x_hat mean/std: {x_hat.mean().item():.4f}, {x_hat.std().item():.4f}")

            mse = criterion_mse(x_hat, images)
            # print(f"MSE: {mse.item():.4f}")

            # =====================
            # Warm-up phase
            # =====================
            if epoch < warmup_epochs:
                # loss = mse
                # View 1
                view1 = F.normalize(z_hat, dim=1)

                # View 2 (class prototype aligned)
                view2 = torch.zeros_like(z_hat)
                for c in range(args['num_class']):
                    mask = (labs == c)
                    if mask.sum() > 0:
                        view2[mask] = list_class_latent[c]
                view2 = F.normalize(view2, dim=1)

                # Projection head
                h1 = F.normalize(proj_head(view1), dim=1)
                h2 = F.normalize(proj_head(view2), dim=1)

                # print(labs, class_weights)
                
                loss_nt_xent = criterion_nt(h1, h2, labs, temperature, class_weights)

                loss = alpha * mse + (1 - alpha) * loss_nt_xent

            # =====================
            # Main training phase
            # =====================
            else:
                
                # View 1
                view1 = F.normalize(z_hat, dim=1)

                # View 2 (class prototype aligned)
                view2 = torch.zeros_like(z_hat)
                for c in range(args['num_class']):
                    mask = (labs == c)
                    if mask.sum() > 0:
                        view2[mask] = list_class_latent[c]
                view2 = F.normalize(view2, dim=1)

                # Projection head
                h1 = F.normalize(proj_head(view1), dim=1)
                h2 = F.normalize(proj_head(view2), dim=1)
                
                # print(labs, class_weights)
                # print(f"h1{h1.min()} {h1.max()} | h2 {h2.min()} {h2.max()}")
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

        # ---- Logging ----
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == warmup_epochs:
            if epoch < warmup_epochs:
                print(f"[Warmup Epoch {epoch+1}] Loss={avg_train_loss:.4f}  MSE={avg_mse_loss:.4f}")
            else:
                print(f"Epoch [{epoch+1}/{args['epochs']}] "
                      f"Loss={avg_train_loss:.4f} | "
                      f"MSE={avg_mse_loss:.4f} | "
                      f"Contrastive={avg_contrastive_loss:.4f}")

        # ---- Save best model ----
        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
            best_encoder_state = copy.deepcopy(encoder.state_dict())
            best_decoder_state = copy.deepcopy(decoder.state_dict())
            
        
        
    print(f"The best loss :{best_loss:.4f}")
    print(f"\n✓ Autoencoder training complete!")

    # =========================================================
    # Stage B: Generate Data with SMOTE
    # =========================================================
    print("--- Stage B: Generating SMOTE data ---")
    # encoder.load_state_dict(torch.load(path_enc, map_location=device))
    # decoder.load_state_dict(torch.load(path_dec, map_location=device))
    if best_encoder_state is not None:
        encoder.load_state_dict(best_encoder_state)
        decoder.load_state_dict(best_decoder_state)
    else:
        print("Warning: Best model state was never saved. Using the final model state.")
    encoder.eval(); decoder.eval()

    #imbal_goal = [3000, 2000, 1000, 750, 500, 12, 10, 8, 6, 4] # Example
    
    imbal_goal = Counter([label for _, label in imbalanced_train_list])
    print(f'Imb Distri: {imbal_goal}')
    
    # if dataset_name == 'cifar100':
    #     max_count = 4590
    #     class_start = 0
    # else:
    max_count = max(imbal_goal.values())
    class_start = 1

    data = torch.stack([item[0] for item in imbalanced_train_list])
    target = torch.tensor([item[1] for item in imbalanced_train_list])
    
    #######################################################################
    ## SMOTE Generation with Library ######################################
    all_synthetic_images, all_synthetic_labels = [], []
    save_dir = '/home/ubuntu/fai_run/project/{}/cosmo_mesh_size32_IMG'.format(dataset_name)  

    with torch.no_grad():
        # 1) encode full train set once
        latent_all, _ = encoder(data.to(device), target.to(device))
        X_all = latent_all.cpu().numpy()
        y_all = target.numpy()

        for i in range(class_start, args['num_class']):
            xclass_imgs, yclass_labels = biased_get_class1(i, data, target)
            n_to_gen = abs(max_count - len(yclass_labels))
            
            if n_to_gen <= 0 or len(yclass_labels) == 0:
                continue

            # 2) one-vs-rest labels for library sampler
            y_bin = (y_all == i).astype(int)
            n_min = y_bin.sum()
            if n_min < 2:
                continue
            
            n_maj = len(y_bin) - n_min
            gap = n_maj - n_min
            if gap <= 0:
                continue
            proportion = n_to_gen / gap

            # 3) poly-fit SMOTE from library # sv.polynom_fit_SMOTE_poly #or 'bus', or 'poly_2'
            sampler = sv.polynom_fit_SMOTE_mesh(
                proportion=proportion,
                random_state=42
            )

            X_res, y_res = sampler.sample(X_all, y_bin)


            # 4) keep only newly generated samples of class i
            X_syn = X_res[len(X_all):]
            y_syn = y_res[len(y_bin):]
            xsamp = X_syn[y_syn == 1][:n_to_gen]
            ysamp = np.full(len(xsamp), i)

            if xsamp.shape[0] > 0:
                generated_images = decoder(torch.tensor(xsamp, dtype=torch.float32).to(device))

                # keep your existing save-image code unchanged here
                class_dir = os.path.join(save_dir, f'class_{i}')
                os.makedirs(class_dir, exist_ok=True)
                num_to_save = 500 #min(40, generated_images.shape[0])
                for img_idx in range(num_to_save):
                    single_image = generated_images[img_idx]
                    img_min, img_max = single_image.min(), single_image.max()
                    normalized_image = (single_image - img_min) / (img_max - img_min)
                    np_image = normalized_image.cpu().permute(1, 2, 0).numpy()
                    fig, ax = plt.subplots()
                    ax.imshow(np_image, cmap='gray')
                    ax.axis('off')
                    file_path = os.path.join(class_dir, f'generated_img_{img_idx + 1}.png')
                    plt.savefig(file_path, dpi=300, bbox_inches='tight', pad_inches=0)
                    plt.close(fig)

                print(f"Saved {num_to_save} images for class {i} in '{class_dir}/'")
                all_synthetic_images.append(generated_images.cpu())
                all_synthetic_labels.append(torch.tensor(ysamp))

    ######################################################################
    # Create the final balanced dataset
    if all_synthetic_images:
        synthetic_data = torch.cat(all_synthetic_images, dim=0)
        synthetic_labels = torch.cat(all_synthetic_labels, dim=0)
        balanced_data = torch.cat([data, synthetic_data], dim=0)
        balanced_labels = torch.cat([target, synthetic_labels], dim=0)
    else:
        balanced_data, balanced_labels = data, target
    
    print(f"data {data.min()} {data.max()} | synthetic {synthetic_data.min()} {synthetic_data.max()}  | balanced {balanced_data.min()} {balanced_data.max()}")
