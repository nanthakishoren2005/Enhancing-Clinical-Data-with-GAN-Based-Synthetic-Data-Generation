"""
Unit tests for CNN classifiers (ResNet18 and CustomCNN) output shapes and losses for 4-class classification.
"""
import torch
import torch.nn as nn
from src.cnn.model import build_classifier, CustomCNN


def test_custom_cnn_forward():
    model = CustomCNN(num_classes=4, in_channels=3)
    x = torch.randn(2, 3, 128, 128)
    logits = model(x)
    assert logits.shape == (2, 4)


def test_resnet18_classifier_forward():
    model = build_classifier(architecture="resnet18", num_classes=4, pretrained=False)
    x = torch.randn(2, 3, 128, 128)
    logits = model(x)
    assert logits.shape == (2, 4)

    # Test cross-entropy loss computation
    criterion = nn.CrossEntropyLoss()
    targets = torch.tensor([0, 3])
    loss = criterion(logits, targets)
    assert loss.item() > 0
