# DeepPolyCL-SMOTE: A Supervised Contrastive Framework for Deep Latent Space Oversampling

  This repository provides the implementation of DeepPolyCL-SMOTE, a deep learning-based data augmentation framework designed to address class imbalance in multi-class image classification. The proposed method extends DeepSMOTE by integrating centroid-guided supervised contrastive learning to explicitly structure the latent space, followed by nonlinear latent interpolation (pf-SMOTE with Mesh topology) for generating high-quality synthetic samples.

Unlike conventional linear interpolation methods, DeepPolyCL-SMOTE produces smoother and more class-consistent samples by leveraging a well-structured latent representation, resulting in improved class separability and classification performance.
  
## Datasets
This implementation was evaluated on the following publicly available image datasets:
* MNIST: The MNIST database of handwritten digits - [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)
* Fashion-MNIST: A dataset of Zalando's article images (Xiao et al., 2017) - [https://github.com/zalandoresearch/fashion-mnist]  (https://github.com/zalandoresearch/fashion-mnist)
* CIFAR-10: The CIFAR-10 dataset - [https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)  

## Code Information
The main files in this repository are:
DeepPolyCLSMOTE/
├── main.py        # Main training and evaluation pipeline
├── utils.py       # Utility functions
├── README.md      # Project documentation

## Usage Instructions
The entire pipeline—including data preparation, model training, latent-space oversampling, and evaluation—is implemented in main.py.

To run the experiments:
1.  **Navigate to the project directory:**
    cd DeepPolyCLSMOTE

2.  **Run the main script:**
    python main.py
   
##  The pipeline overview:
* Data Preparation:
Load datasets (via PyTorch datasets) and simulate long-tailed imbalance.
* Baseline Model Training:
Trains a CNN classifier on imbalanced data for comparison.
* Representation Learning (Phase I):
  Train an autoencoder with:
  * Reconstruction loss (MSE)
  * Centroid-guided supervised contrastive loss
* Latent-space Oversampling (Phase II):
Generate synthetic samples using pf-SMOTE (Mesh topology) in latent space.
* Balanced Training:
Train a classifier (e.g., ResNet) on the augmented dataset.
* Evaluation:
Reports performance using metrics such as F1-score, ASCA, and BHR.

## Requirements:
PyTorch   
NumPy  
scikit-learn  

## Citations: 
If you use this code in your research, please cite the following reference:
> Homjandee, S. and Sinapiromsaran, K. (2025). Deepclsmote: Deep class-latent synthetic minority oversampling technique. https://doi.org/10.5281/zenodo.15362173. Code repository archived on Zenodo.
 
