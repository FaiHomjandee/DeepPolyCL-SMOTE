# DeepPolyCL-SMOTE: A Supervised Contrastive Framework for Deep Latent Space Oversampling 📊

This repository provides the implementation of DeepPolyCL-SMOTE, a deep learning-based data augmentation framework designed to address class imbalance in multi-class image classification. The proposed method extends DeepSMOTE by integrating centroid-guided supervised contrastive learning to explicitly structure the latent space, followed by nonlinear latent interpolation (pf-SMOTE with Mesh topology) for generating high-quality synthetic samples.

Unlike conventional linear interpolation methods, DeepPolyCL-SMOTE produces smoother and more class-consistent samples by leveraging a well-structured latent representation, resulting in improved class separability and classification performance.

## Datasets 📦
This implementation was evaluated on the following publicly available image datasets:
* MNIST: The MNIST database of handwritten digits — [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)
* Fashion-MNIST: A dataset of Zalando's article images (Xiao et al., 2017) — [https://github.com/zalandoresearch/fashion-mnist](https://github.com/zalandoresearch/fashion-mnist)
* CIFAR-10: The CIFAR-10 dataset — [https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)

Each dataset's official training partition is reconfigured into a long-tailed distribution using the exponential-decay rule $n_k = \lfloor n_{\max} \cdot \mu^{k} \rfloor$ with $\mu = 0.6$ (see the paper's "Datasets" subsection for the exact per-class counts). 10-fold cross-validation is applied to the official training partition only; within each fold, data is split into an imbalanced training set, a balanced validation set, and a balanced test set, with long-tail subsampling applied to the training subset after the split so validation/test remain balanced.

**Data preparation is not yet included in this repository.** `main.py` and `smote_baseline.py` both expect two pre-generated files per dataset under `./prepared_data/` (or wherever `DPC_DATA_DIR` points):

* `{dataset}_folds_data.pth` — a list of 10 folds, each fold a `(train_loader_ae, val_loader, test_loader)` tuple of `torch.utils.data.DataLoader`.
* `{dataset}_imbalanced_data.pth` — a list of 10 folds, each fold an indexable dataset (e.g. `torch.utils.data.Subset`) of `(image_tensor, label)` pairs, matching the imbalanced training set used to build `train_loader_ae` for that fold.

For `main.py` (DeepPolyCL-SMOTE) specifically, CIFAR-10 is trained at its native 32x32 resolution rather than being downscaled to match MNIST/FMNIST's 28x28, so its two files are named with a `32` suffix instead: `cifar10_folds_data32.pth` and `cifar10_imbalanced_data32.pth`. `smote_baseline.py` uses the plain (no-suffix) filenames for all three datasets, including CIFAR-10.

A script that generates these files from the raw MNIST/FMNIST/CIFAR-10 downloads, applying the long-tail rule above with a fixed, documented random seed per fold, will be added here; until then this is a known gap for anyone trying to reproduce Table 1 from scratch.

## Code Information 💻
```
DeepPolyCL-SMOTE/
├── main.py             # Trains + evaluates DeepPolyCL-SMOTE (the proposed method), all 3 datasets, 10-fold CV
├── smote_baseline.py   # Conventional pixel-space SMOTE reference baseline (Table 1's "SMOTE (conventional)" row)
├── utils.py             # Model architectures, contrastive loss, latent-space oversampling, classifier training/eval
├── requirements.txt
└── README.md
```

## Usage Instructions 📌

```bash
pip install -r requirements.txt

# Place {dataset}_folds_data.pth and {dataset}_imbalanced_data.pth for each
# dataset under ./prepared_data/ (see "Datasets" above), then:

python main.py                       # DeepPolyCL-SMOTE on MNIST (default), all 10 folds
DPC_DATASET=fmnist  python main.py   # switch dataset without editing the file
DPC_DATASET=cifar10 python main.py

DPC_DATASET=mnist python smote_baseline.py   # conventional SMOTE reference baseline
```

Per-dataset hyperparameters ($\alpha$, temperature, latent dimension, weight decay, epochs — grid-searched on fold 1 and fixed for the remaining folds, per the paper's "Implementation Details") are set in `main.py`'s `PAPER_HYPERPARAMS` dict. `DPC_DATA_DIR`, `DPC_RESULTS_DIR`, and `DPC_OUTPUT_ROOT` environment variables override the default `./prepared_data`, `./results`, `./runs` locations.

Both scripts train a downstream ResNet-18 classifier on the balanced dataset produced by their respective oversampling method and evaluate it on the held-out balanced test set using the same recipe (SGD, momentum 0.9, weight decay 0.05, Nesterov, `ReduceLROnPlateau`), so results are directly comparable across methods.

## Pipeline Overview 🖼️

1. **Data Preparation** — long-tailed MNIST / Fashion-MNIST / CIFAR-10 training sets (see "Datasets" above).
2. **Phase I: Representation Learning** — train an autoencoder with a combined reconstruction (MSE) and centroid-guided weighted contrastive loss.
3. **Phase II: Latent-space Oversampling** — generate synthetic minority-class samples via pf-SMOTE (Mesh topology, via the `smote-variants` package) in the trained latent space.
4. **Balanced Training** — train a ResNet-18 classifier on the resulting balanced (real + synthetic) dataset.
5. **Evaluation** — compute F1, ASCA (macro-average recall), and BHR (Balanced Half Recall — mean recall over the bottom half of classes by recall) on the held-out balanced test set.

## Requirements 📚
See `requirements.txt`. Key dependencies: Python ≥3.9, PyTorch, torchvision, NumPy, pandas, scikit-learn, SciPy, matplotlib, tensorboard, `smote-variants`.

## Citations 📍
If you use this code in your research, please cite:
> Homjandee, S. and Sinapiromsaran, K. (2026). DeepPolyCL-SMOTE: A Supervised Contrastive Framework for Deep Latent Space Oversampling. Code archived on Zenodo: https://doi.org/10.5281/zenodo.22822172. 
