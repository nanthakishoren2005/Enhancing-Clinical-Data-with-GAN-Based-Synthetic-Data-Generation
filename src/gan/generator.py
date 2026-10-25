"""
Conditional DCGAN Generator network for 128x128 synthetic chest X-ray patch synthesis.
Takes a random noise vector z and an integer class label y as inputs.
"""
import torch
import torch.nn as nn


def weights_init(m):
    """Custom weights initialization according to Radford et al. DCGAN paper."""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class ConditionalGenerator(nn.Module):
    """
    Conditional Generator network mapping latent vector z and class embedding
    to synthetic 3x128x128 RGB chest X-ray patches with values scaled to [-1, 1].
    """
    def __init__(
        self,
        latent_dim: int = 100,
        num_classes: int = 4,
        embedding_dim: int = 32,
        feature_maps: int = 64,
        out_channels: int = 3,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.feature_maps = feature_maps

        # Class embedding layer
        self.label_embedding = nn.Embedding(num_classes, embedding_dim)

        # Joint linear projection: (latent_dim + embedding_dim) -> feature_maps*8 * 4 * 4
        in_dim = latent_dim + embedding_dim
        self.project = nn.Sequential(
            nn.Linear(in_dim, feature_maps * 8 * 4 * 4),
            nn.BatchNorm1d(feature_maps * 8 * 4 * 4),
            nn.ReLU(True),
        )

        # Transposed Convolution blocks upsampling from 4x4 -> 128x128
        self.deconv_blocks = nn.Sequential(
            # 4x4 -> 8x8
            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),

            # 8x8 -> 16x16
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),

            # 16x16 -> 32x32
            nn.ConvTranspose2d(feature_maps * 2, feature_maps, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),

            # 32x32 -> 64x64
            nn.ConvTranspose2d(feature_maps, feature_maps // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps // 2),
            nn.ReLU(True),

            # 64x64 -> 128x128
            nn.ConvTranspose2d(feature_maps // 2, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),  # Normalized outputs in range [-1, 1]
        )

        self.apply(weights_init)

    def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            noise: (B, latent_dim) random Gaussian vector
            labels: (B,) class indices [0..num_classes-1]
        Returns:
            (B, 3, 128, 128) synthetic images in [-1, 1]
        """
        c = self.label_embedding(labels)  # (B, embedding_dim)
        x = torch.cat([noise, c], dim=1)  # (B, latent_dim + embedding_dim)
        x = self.project(x)
        x = x.view(-1, self.feature_maps * 8, 4, 4)
        out = self.deconv_blocks(x)
        return out
