
from torchvision import models
from torch import nn

class RestNetClassifier(nn.Module):
    _classes = [1, 0]
    def __init__(self):
        super().__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.model.conv1 = nn.Conv2d(
            1, 
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, len(self._classes))
        )
    def forward(self, x):
        x = self.model(x)
        return x