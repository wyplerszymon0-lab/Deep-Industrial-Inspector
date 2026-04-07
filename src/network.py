import torch
import torch.nn as nn
from torchvision import models

class DefectDetector(nn.Module):
    def __init__(self, num_classes=2):
        super(DefectDetector, self).__init__()
        # Load a pre-trained ResNet18
        self.model = models.resnet18(weights='IMAGENET1K_V1')
        
        # Replace the last fully connected layer for our task
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)
