import torch.nn as nn
from torchvision import models

class GenderClassifierResNet18(nn.Module):
    def __init__(self):
        super(GenderClassifierResNet18, self).__init__()
        self.model = models.resnet18(pretrained=True)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 2)  # 2 output classes: male, female

    def forward(self, x):
        return self.model(x)