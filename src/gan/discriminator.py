"""
Conditional DCGAN Discriminator network for 128x128 chest X-ray patch verification.
Evaluates authenticity of a chest X-ray given its conditioning class label.
"""
import torch
import torch.nn as nn
from .generator import weights_init


class ConditionalDiscriminator(nn.Module):
    """
    Conditional Discriminator network. Takes an RGB chest X-ray (3x128x128)
    and class label, projecting the label into a spatial feature channel,
    and returns unnormalized authenticity logits.
    """
    def __init__(
        self,
        num_classes: int = 4,
        in_channels: int = 3,
        feature_maps: int = 64,
        img_size: int = 128,
    ):
        super().__init__()
        self.img_size = img_size
        self.num_classes = num_classes

        # Class label spatial embedding (1 x img_size x img_size)
        self.label_embedding = nn.Embedding(num_classes, img_size * img_size)

        # Input channels = image channels (3) + conditioned label channel (1) = 4
        total_in_channels = in_channels + 1

        self.conv_blocks = nn.Sequential(
            # 128x128 -> 64x64 (No BatchNorm in the first discriminator layer per DCGAN guidance)
            nn.Conv2d(total_in_channels, feature_maps // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # 64x64 -> 32x32
            nn.Conv2d(feature_maps // 2, feature_maps, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.LeakyReLU(0.2, inplace=True),

            # 32x32 -> 16x16
            nn.Conv2d(feature_maps, feature_maps * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # 16x16 -> 8x8
            nn.Conv2d(feature_maps * 2, feature_maps * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # 8x8 -> 4x4
            nn.Conv2d(feature_maps * 4, feature_maps * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # 4x4 -> 1x1 logit
            nn.Conv2d(feature_maps * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
        )

        self.apply(weights_init)

    def forward(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, 3, 128, 128)
            labels: (B,) class indices [0..num_classes-1]
        Returns:
            (B,) unnormalized logit (real vs fake score)
        """
        batch_size = images.size(0)
        # Embed label and reshape to (B, 1, H, W)
        c = self.label_embedding(labels).view(batch_size, 1, self.img_size, self.img_size)
        x = torch.cat([images, c], dim=1)  # (B, 4, 128, 128)
        logit = self.conv_blocks(x)        # (B, 1, 1, 1)
        return logit.view(batch_size)
