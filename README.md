#  Deep-Industrial-Inspector

A Deep Learning-based computer vision system for automated surface defect detection using Transfer Learning and PyTorch.

###  Technical Core
* **Architecture:** ResNet18 (Fine-tuned for binary classification).
* **Framework:** PyTorch & Torchvision.
* **Optimization:** Adam Optimizer with Cross-Entropy Loss.
* **Explainability:** Gradient-based Saliency Maps for visual inspection of model decisions.
  
###  Training Pipeline
* **Data Augmentation:** Random rotations, flips, and normalization to improve model robustness.
* **Transfer Learning:** Leveraging ImageNet pre-trained weights to achieve high accuracy with limited industrial data.

###  How to run
```bash
python train.py --epochs 20 --batch_size 32
