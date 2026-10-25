"""
CNN model architectures for Chest X-ray Classification:
1. ResNet-18 (standard transfer learning backbone)
2. CustomCNN (lightweight CNN optimized for fast CPU/embedded inference)
"""
import torch
import torch.nn as nn
import torchvision.models as models


class CustomCNN(nn.Module):
    """
    Lightweight, modular CNN tailored for chest X-ray classification.
    Comprises 4 Conv-BN-ReLU-Pool feature extraction blocks followed by
    global average pooling and a fully connected classification head with dropout.
    """
    def __init__(self, num_classes: int = 4, in_channels: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 128x128 -> 64x64
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: 64x64 -> 32x32
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: 32x32 -> 16x16
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4: 16x16 -> 8x8
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=0.4),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        logits = self.classifier(x)
        return logits


def build_classifier(
    architecture: str = "resnet18",
    num_classes: int = 4,
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    """
    Build and return the selected CNN classifier model.
    Handles weights download or fallback gracefully.
    """
    arch = architecture.lower().strip()
    if arch == "custom_cnn":
        return CustomCNN(num_classes=num_classes)

    elif arch == "resnet18":
        try:
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            model = models.resnet18(weights=weights)
        except Exception as e:
            print(f"Warning: Could not fetch pretrained ResNet18 weights ({e}). Initializing with random weights.")
            model = models.resnet18(weights=None)

        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False

        # Replace final classification head
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes),
        )
        return model

    else:
        raise ValueError(f"Unsupported CNN architecture: {architecture}. Choose 'resnet18' or 'custom_cnn'.")
