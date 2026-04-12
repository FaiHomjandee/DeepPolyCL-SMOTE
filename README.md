# DeepPolyCL-SMOTE: A Supervised Contrastive Framework for Deep Latent Space Oversampling

  This repository provides the implementation of DeepPolyCL-SMOTE, a deep learning-based data augmentation framework designed to address class imbalance in multi-class image classification. The proposed method extends DeepSMOTE by integrating centroid-guided supervised contrastive learning to explicitly structure the latent space, followed by nonlinear latent interpolation (pf-SMOTE with Mesh topology) for generating high-quality synthetic samples.

Unlike conventional linear interpolation methods, DeepPolyCL-SMOTE produces smoother and more class-consistent samples by leveraging a well-structured latent representation, resulting in improved class separability and classification performance.
  
## Datasets
This implementation was evaluated on the following publicly available image datasets:
* MNIST: The MNIST database of handwritten digits - [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)
* Fashion-MNIST: A dataset of Zalando's article images (Xiao et al., 2017) - [https://github.com/zalandoresearch/fashion-mnist]  
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
    ```bash
    cd DeepPolyCLSMOTE
    ```

3.  **Run the main script:**
    ```bash
    python main.py
    ```
## Pipeline Overview

1. **Data Preparation**
   - Load datasets (MNIST, Fashion-MNIST, CIFAR-10)
   - Create imbalanced (longtail) training sets
2. **Baseline Training**
   - Train a CNN on imbalanced data
3. **Phase I: Representation Learning**
   - Train an autoencoder using MSE loss
   - Apply centroid-guided supervised contrastive loss
4. **Phase II: Latent-space Oversampling**
   - Generate synthetic samples using pf-SMOTE (Mesh topology)
5. **Balanced Training**
   - Train a classifier (e.g., ResNet) on the augmented dataset
6. **Evaluation**
   - Compute performance metrics (F1-score, ASCA, and BHR) 

## Requirements:
PyTorch   
NumPy  
scikit-learn  

## Citations: 
If you use this code in your research, please cite the following reference:
> Homjandee, S. and Sinapiromsaran, K. (2025). Deepclsmote: Deep class-latent synthetic minority oversampling technique. https://doi.org/10.5281/zenodo.15362173. Code repository archived on Zenodo.
 
