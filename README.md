# Replicating "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"

This repository contains my replication of the original Vision Transformer (ViT) model, as described in the paper ["An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"](https://arxiv.org/abs/2010.11929). The project uses the CIFAR-10 dataset to evaluate the performance of the ViT architecture.

## Overview
The Vision Transformer (ViT) revolutionizes image classification by treating images as sequences of patches and applying transformer-based architectures, originally designed for NLP tasks. This implementation replicates the key components of the original ViT and adapts it for the CIFAR-10 dataset.

### Key Features:
- Implementation of the ViT model from scratch.
- Processing CIFAR-10 images as 4x4 patches.
- Training and evaluation scripts with performance metrics.
- Reproducible results matching or approaching the original paper’s performance benchmarks.

---

## Repository Structure
```
vit-replication/
├── notebook.ipynb      # Jupyter Notebook with the full implementation and results
├── README.md           # This file
├── requirements.txt    # Python dependencies
├── data/               # Placeholder for CIFAR-10 dataset (if not downloaded automatically)
└── results/            # Generated plots and metrics
```

---

## Dataset
The [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. For this project, images are resized and divided into 4x4 patches for input to the ViT model.

### Dataset Handling:
- Downloaded using PyTorch’s `torchvision.datasets` module.
- Preprocessed into 4x4 patches.

---

## Model Details
The ViT model is implemented with the following components:
1. **Patch Embedding**: Splits images into 16x16 patches and embeds them.
2. **Transformer Encoder**: Applies self-attention and feed-forward layers to learn patch relationships.
3. **Classification Head**: Maps the transformer’s output to class predictions.

### Model Hyperparameters:
- **Patch size**: 4x4
- **Embedding dimension**: 512
- **Number of heads**: 12
- **Number of transformer layers**: 8
- **MLP size**: 3072

---

## Results
Training the ViT model on CIFAR-10 achieved:
- **Top-1 Accuracy**: ~85% after fine-tuning.
- **Loss Curve**: Included in the notebook.

Plots and metrics are available in the `results/` directory.

---

## Requirements
To reproduce this project, install the dependencies listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```

Main dependencies:
- Python 3.8+
- PyTorch 1.11+
- Torchvision 0.12+
- NumPy
- Matplotlib

---

## Running the Notebook
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/vit-replication.git
   cd vit-replication
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Open and run the Jupyter Notebook:
   ```bash
   jupyter notebook notebook.ipynb
   ```

---

## References
- Dosovitskiy, A., et al. (2020). ["An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"](https://arxiv.org/abs/2010.11929).
- CIFAR-10 Dataset: [https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Acknowledgments
Special thanks to the authors of the ViT paper for their groundbreaking work and to the PyTorch community for providing tools to make this implementation possible.

