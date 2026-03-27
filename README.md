# Deep Learning: Image Classification with CNN & Transfer Learning

A complete deep learning project implementing image classification on the CIFAR-10 dataset using a custom Convolutional Neural Network built from scratch, followed by transfer learning with a pretrained ResNet18. Built with PyTorch and trained on Google Colab T4 GPU.

---

## Results at a Glance

| Model | Test Accuracy | Precision | Recall | F1-Score | Params | Train Time |
|---|---|---|---|---|---|---|
| Custom CNN | **86.6%** | 0.866 | 0.866 | 0.865 | 1,405,226 | ~28 min |
| ResNet18 (TL) | **95.8%** | 0.958 | 0.958 | 0.958 | 11,310,410 | ~128 min |

---

## Project Structure
```
cifar10-cnn-transfer-learning/
├── CNN_TransferLearning.ipynb   # Main notebook — run this
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── sample_images.png            # CIFAR-10 dataset preview
├── class_distribution.png       # Class balance chart
├── cnn_training_curves.png      # Loss & accuracy over epochs
├── cnn_confusion_matrix.png     # CNN confusion matrix
└── cnn_per_class_accuracy.png   # Per-class accuracy bar chart
```

> **Note:** Model weight files (`.pth`) are not included due to file size.
> They are saved to Google Drive during training and can be regenerated
> by running the notebook end to end (~2.5 hours on a T4 GPU).

---

## Dataset

[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) by Krizhevsky, Nair & Hinton.

- 60,000 colour images at 32 × 32 pixels
- 10 mutually exclusive classes
- 50,000 training / 10,000 test images
- Perfectly balanced — 5,000 images per class
- Downloads automatically via `torchvision` — no manual setup needed

**Classes:** airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

---

## Model Architectures

### Custom CNN

Built from scratch with three convolutional blocks followed by a fully connected classifier head.
```
Input (3 × 32 × 32)
    ↓
Conv Block 1 — 32 filters → (32 × 16 × 16)
    Conv2D → BatchNorm2D → ReLU → Conv2D → BatchNorm2D → ReLU → MaxPool → Dropout(0.2)
    ↓
Conv Block 2 — 64 filters → (64 × 8 × 8)
    Conv2D → BatchNorm2D → ReLU → Conv2D → BatchNorm2D → ReLU → MaxPool → Dropout(0.3)
    ↓
Conv Block 3 — 128 filters → (128 × 4 × 4)
    Conv2D → BatchNorm2D → ReLU → Conv2D → BatchNorm2D → ReLU → MaxPool → Dropout(0.4)
    ↓
FC Head
    Flatten(2048) → Linear(512) → BatchNorm1D → ReLU → Dropout(0.5)
                 → Linear(128)  → BatchNorm1D → ReLU → Dropout(0.3)
                 → Linear(10)
    ↓
Output — 10 class logits (no softmax — CrossEntropyLoss handles it)
```

**Total trainable parameters: 1,405,226**

---

### ResNet18 — Transfer Learning

Pretrained on ImageNet (1.2M images, 1000 classes). Final FC layer replaced with a custom classifier for CIFAR-10.

**Two-phase training strategy:**

| Phase | Backbone | Trainable Params | LR | Epochs |
|---|---|---|---|---|
| Phase 1 — Head only | Frozen | 133,898 | 0.001 | 20 |
| Phase 2 — Fine-tuning | Unfrozen | 11,310,410 | 0.0001 | 20 (ES at 19) |

The low learning rate in Phase 2 is critical — a high LR would overwrite the pretrained ImageNet features before they contribute to CIFAR-10 accuracy (catastrophic forgetting).

---

## Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Weight decay | 1e-4 |
| Batch size | 128 |
| LR Scheduler | ReduceLROnPlateau (factor=0.5, patience=3) |
| Early stopping | Patience 7 (CNN) / 5 (ResNet18) |
| Max epochs | 50 (CNN) / 20 per phase (ResNet18) |
| Loss function | CrossEntropyLoss |

---

## Data Preprocessing

**Training only — augmentation pipeline:**
```
RandomHorizontalFlip()
RandomCrop(32, padding=4)
ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
ToTensor()
Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616))
```

**Validation & Test — no augmentation:**
```
ToTensor()
Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616))
```

> Augmentation is applied **only** to training data. Applying it to
> validation or test data constitutes data leakage and inflates metrics.

---

## Per-Class Results

### Custom CNN — Test Accuracy
| Class | Precision | Recall | F1-Score | Accuracy |
|---|---|---|---|---|
| airplane | 0.86 | 0.89 | 0.87 | 88.9% |
| automobile | 0.94 | 0.94 | 0.94 | 94.0% |
| bird | 0.85 | 0.78 | 0.81 | 77.9% |
| cat | 0.77 | 0.68 | 0.72 | 67.6% |
| deer | 0.85 | 0.88 | 0.87 | 88.1% |
| dog | 0.76 | 0.84 | 0.80 | 83.6% |
| frog | 0.90 | 0.91 | 0.90 | 90.8% |
| horse | 0.91 | 0.89 | 0.90 | 89.4% |
| ship | 0.93 | 0.92 | 0.92 | 91.9% |
| truck | 0.89 | 0.94 | 0.91 | 94.0% |
| **macro avg** | **0.87** | **0.87** | **0.87** | **86.6%** |

### ResNet18 — Test Accuracy
| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| airplane | 0.96 | 0.97 | 0.97 |
| automobile | 0.97 | 0.97 | 0.97 |
| bird | 0.98 | 0.93 | 0.95 |
| cat | 0.92 | 0.89 | 0.91 |
| deer | 0.95 | 0.98 | 0.96 |
| dog | 0.91 | 0.94 | 0.93 |
| frog | 0.98 | 0.98 | 0.98 |
| horse | 0.98 | 0.97 | 0.98 |
| ship | 0.97 | 0.98 | 0.98 |
| truck | 0.96 | 0.97 | 0.97 |
| **macro avg** | **0.96** | **0.96** | **0.96** |

---

## Key Findings

**Cat is the hardest class** — 67.6% accuracy with the custom CNN,
improving to 89% with ResNet18. The single largest error in the confusion
matrix is cat being misclassified as dog (147 times out of 1000).

**Transfer learning jumped immediately** — Phase 2 epoch 1 went straight
to 93% validation accuracy. The pretrained ImageNet features transferred
almost instantly to CIFAR-10.

**Hardware matters enormously:**

| Task | Local CPU | Colab T4 GPU |
|---|---|---|
| CNN 50 epochs | ~2.5 hours | ~28 minutes |
| ResNet18 Phase 1 | Never finished | ~61 minutes |
| ResNet18 Phase 2 | ~22 hours (est.) | ~67 minutes |
| **Total** | **30+ hours** | **~2.5 hours** |

---

## How to Run

### Option 1 — Google Colab (recommended)
1. Open `CNN_TransferLearning.ipynb` in [Google Colab](https://colab.research.google.com)
2. Set runtime: **Runtime → Change runtime type → T4 GPU**
3. Run all cells top to bottom
4. Everything saves automatically to Google Drive

### Option 2 — Local
```bash
# Clone the repo
git clone https://github.com/your_username/cifar10-cnn-transfer-learning.git
cd cifar10-cnn-transfer-learning

# Install dependencies
pip install -r requirements.txt

# Open the notebook
jupyter notebook CNN_TransferLearning.ipynb
```

> Local training on CPU is not recommended for the ResNet18 transfer
> learning section — Phase 2 takes approximately 97 minutes per epoch.

---

## Requirements
```
torch
torchvision
matplotlib
scikit-learn
seaborn
numpy
Pillow
```

Install with:
```bash
pip install -r requirements.txt
```

---

## Lessons Learned

- **GPU is non-negotiable for transfer learning** — CPU made ResNet18 training practically impossible at this scale
- **Augment training data only** — applying augmentation to val/test is data leakage
- **Val accuracy > Train accuracy is expected** — augmentation makes training images harder; this is regularisation working, not a bug
- **Checkpoint frequently** — save best weights to Drive after every improvement; kernel crashes are real
- **Lower LR in fine-tuning is critical** — LR=1e-4 in Phase 2 prevents catastrophic forgetting of pretrained features
- **BatchNorm + Dropout together** — highly effective combination; BatchNorm stabilises, Dropout prevents memorisation

---

## Potential Improvements

- Train CNN for more than 50 epochs — model had not fully converged
- Try ResNet50 or EfficientNet for higher accuracy ceiling
- Apply CutMix / MixUp augmentation — proven +1–2% on CIFAR-10
- Use cosine annealing LR schedule instead of ReduceLROnPlateau
- Add learning rate warmup at the start of fine-tuning

---

## References

- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html) — Krizhevsky, Nair & Hinton
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) — He et al. (ResNet)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [torchvision Models](https://pytorch.org/vision/stable/models.html)

---

## License

MIT License — free to use, modify and distribute with attribution.
