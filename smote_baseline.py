#!/usr/bin/env python
"""
smote_baseline.py

Conventional (pixel-space) SMOTE baseline: 5-nearest-neighbor linear
interpolation directly on flattened raw pixels (no learned representation
at all), followed by the same ResNet classifier training/evaluation used
for every other method. Included as a non-learned classical reference point
(see paper Table 1's "SMOTE (conventional)" row).

Usage:
    DPC_DATASET=mnist python smote_baseline.py
    DPC_DATASET=fmnist python smote_baseline.py
    DPC_DATASET=cifar10 python smote_baseline.py
"""
import pandas as pd
import torch
import time
import copy
import numpy as np
import os
from collections import Counter
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import f1_score

from utils import ResNet18_28x28, CLASSIFIER_EPOCHS

time_start = time.time()

dataset_name = os.environ.get('DPC_DATASET', 'mnist')  # 'mnist' | 'fmnist' | 'cifar10'
num_class = 10
n_channel = 3 if dataset_name == 'cifar10' else 1

DATA_DIR  = Path(os.environ.get('DPC_DATA_DIR', './prepared_data'))
SAVE_PATH = Path(os.environ.get('DPC_RESULTS_DIR', './results'))
SAVE_PATH.mkdir(parents=True, exist_ok=True)


def Generate_SMOTE(X, y, n_to_sample, cl):
    """Classic SMOTE: linear interpolation between a random sample and one
    of its 5 nearest neighbors, applied directly on raw flattened pixels."""
    if n_to_sample <= 0:
        return np.array([]), np.array([])
    n_neigh = min(5, len(X) - 1)
    if n_neigh < 1:
        return np.array([]), np.array([])
    nn_model = NearestNeighbors(n_neighbors=n_neigh + 1)
    nn_model.fit(X)
    _, ind = nn_model.kneighbors(X)
    base_indices = np.random.choice(list(range(len(X))), n_to_sample)
    neighbor_indices = np.random.choice(list(range(1, n_neigh + 1)), n_to_sample)
    X_base = X[base_indices]
    X_neighbor = X[ind[base_indices, neighbor_indices]]
    samples = X_base + np.multiply(np.random.rand(n_to_sample, 1), X_neighbor - X_base)
    return samples, [cl] * n_to_sample


def evaluate_on_test(model, test_loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labs in test_loader:
            imgs  = imgs.to(device)
            preds = model(imgs).argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labs.tolist())

    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    per_class = {}
    for c in range(num_class):
        idx     = [i for i, l in enumerate(all_labels) if l == c]
        correct = sum(1 for i in idx if all_preds[i] == c)
        per_class[c] = {
            'accuracy':     correct / len(idx) if idx else 0.0,
            'sample_count': len(idx)
        }

    recalls = [v['accuracy'] for v in per_class.values()]
    acsa = float(np.mean(recalls))  # macro-average recall, NOT macro-F1
    g_mean = float(np.prod(recalls) ** (1.0 / len(recalls)))
    return f1, acsa, g_mean, per_class


def run_single_fold_SMOTE_baseline(fold_data_loaders, imbalanced_train_list):
    train_loader_ae, val_loader, test_loader = fold_data_loaders
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ── Stage A: Conventional SMOTE on raw pixels ─────────────
    print("--- Stage A: Conventional SMOTE on raw pixels ---")

    imbal_goal    = Counter([label for _, label in imbalanced_train_list])
    data_tensor   = torch.stack([item[0] for item in imbalanced_train_list])
    target_tensor = torch.tensor([item[1] for item in imbalanced_train_list])
    print(f'Imb Distri: {imbal_goal}')

    n_samples = data_tensor.shape[0]
    C, H, W   = data_tensor.shape[1], data_tensor.shape[2], data_tensor.shape[3]
    X_flat    = data_tensor.view(n_samples, -1).numpy()
    y_np      = target_tensor.numpy()

    all_syn_X, all_syn_y = [], []
    for cls_id in range(1, num_class):
        cls_mask = (y_np == cls_id)
        X_cls    = X_flat[cls_mask]
        n_to_gen = abs(imbal_goal[0] - int(cls_mask.sum()))

        if n_to_gen > 0 and len(X_cls) >= 2:
            xsamp, ysamp = Generate_SMOTE(X_cls, y_np[cls_mask], n_to_gen, cls_id)
            if len(xsamp) > 0:
                all_syn_X.append(xsamp)
                all_syn_y.extend(ysamp)
                print(f"  Class {cls_id}: generated {len(xsamp)} samples")

    if all_syn_X:
        X_syn_tensor = torch.tensor(
            np.vstack(all_syn_X), dtype=torch.float32
        ).view(-1, C, H, W)
        y_syn_tensor = torch.tensor(all_syn_y, dtype=torch.long)
        balanced_X   = torch.cat([data_tensor, X_syn_tensor], dim=0)
        balanced_y   = torch.cat([target_tensor, y_syn_tensor], dim=0)
    else:
        balanced_X, balanced_y = data_tensor, target_tensor

    print(f'Balanced Distri: {Counter(balanced_y.tolist())}')

    balanced_loader = DataLoader(
        TensorDataset(balanced_X, balanced_y),
        batch_size=256,
        shuffle=True
    )

    # ── Stage B: Train ResNet ──────────────────────────────────
    print("--- Stage B: Training ResNet classifier ---")

    classifier = ResNet18_28x28(
        num_classes=num_class,
        n_channel=n_channel,
        dropout_rate=0.5
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.0)
    optimizer = torch.optim.SGD(
        classifier.parameters(),
        lr=0.01, momentum=0.9,
        weight_decay=0.05, nesterov=True
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    best_val_loss    = float('inf')
    best_model_state = copy.deepcopy(classifier.state_dict())

    for epoch in range(CLASSIFIER_EPOCHS):
        classifier.train()
        train_loss = 0.0
        for imgs, labs in balanced_loader:
            imgs, labs = imgs.to(device), labs.to(device)
            optimizer.zero_grad()
            loss = criterion(classifier(imgs), labs)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        classifier.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labs in val_loader:
                imgs, labs = imgs.to(device), labs.to(device)
                out        = classifier(imgs)
                val_loss  += criterion(out, labs).item()
                val_correct += (out.argmax(1) == labs).sum().item()
                val_total   += labs.size(0)

        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{CLASSIFIER_EPOCHS}] "
                  f"Val Loss: {avg_val_loss:.4f} | "
                  f"Val Acc: {val_correct/val_total*100:.2f}%")

        if avg_val_loss < best_val_loss:
            best_val_loss    = avg_val_loss
            best_model_state = copy.deepcopy(classifier.state_dict())

    # ── Stage C: Test ──────────────────────────────────────────
    print("--- Stage C: Evaluating on test set ---")
    classifier.load_state_dict(best_model_state)
    f1, acsa, g_mean, per_class = evaluate_on_test(classifier, test_loader, device)
    print(f"F1: {f1:.4f} | ACSA: {acsa:.4f} | G-Mean: {g_mean:.4f}")
    print(per_class)
    return f1, acsa, g_mean, per_class


# ── Main ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Loading {dataset_name} data...")

    folds_data = torch.load(
        DATA_DIR / f"{dataset_name}_folds_data.pth",
        weights_only=False
    )
    imbalanced_train_dataset_list = torch.load(
        DATA_DIR / f"{dataset_name}_imbalanced_data.pth",
        weights_only=False
    )
    print("Data loaded successfully.")

    results_filepath = SAVE_PATH / f"Results_SMOTE_conventional_{dataset_name}.csv"
    results_list     = []
    num_folds        = len(folds_data)
    fold_f1, fold_asca, fold_gm, fold_reports = [], [], [], []

    for fold_idx in range(num_folds):
        print(f"\n--- Fold {fold_idx+1}/{num_folds} ---")

        f1, acsa, gm, report = run_single_fold_SMOTE_baseline(
            folds_data[fold_idx],
            imbalanced_train_dataset_list[fold_idx]
        )

        fold_f1.append(f1)
        fold_asca.append(acsa)
        fold_gm.append(gm)
        fold_reports.append(report)

        results_list.append({
            'dataset': dataset_name,
            'Fold':    fold_idx + 1,
            'F1':      f1,
            'ASCA':    acsa,
            'GM':      gm,
            'Perclass_Report': str(report)
        })
        pd.DataFrame(results_list).to_csv(results_filepath, index=False)
        print(f"    ... Saved fold {fold_idx+1}")

    # Average row
    results_list.append({
        'dataset':  dataset_name,
        'Fold':     'Average',
        'F1':       np.mean(fold_f1),
        'Std_F1':   np.std(fold_f1),
        'ASCA':     np.mean(fold_asca),
        'Std_ASCA': np.std(fold_asca),
        'GM':       np.mean(fold_gm),
        'Std_GM':   np.std(fold_gm),
        'Perclass_Report': str(fold_reports)
    })
    pd.DataFrame(results_list).to_csv(results_filepath, index=False)

    print(f"\n{'='*50}")
    print(f"Avg F1:   {np.mean(fold_f1):.4f} (±{np.std(fold_f1):.4f})")
    print(f"Avg ACSA: {np.mean(fold_asca):.4f} (±{np.std(fold_asca):.4f})")
    print(f"Avg GM:   {np.mean(fold_gm):.4f} (±{np.std(fold_gm):.4f})")
    print(f"Results saved to: {results_filepath}")
    print(f"Time: {(time.time()-time_start)/3600:.2f} hr")
