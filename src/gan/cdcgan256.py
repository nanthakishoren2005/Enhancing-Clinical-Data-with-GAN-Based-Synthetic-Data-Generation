"""
Conditional DCGAN Generator & Discriminator for 256x256 synthetic chest X-ray synthesis.
Extends the 128x128 cDCGAN with one additional upsampling stage in the Generator
and one additional downsampling stage in the Discriminator to support 256x256 output.
"""
import torch
import torch.nn as nn


def weights_init_256(m):
    """Custom weights initialization according to Radford et al. DCGAN paper."""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class ConditionalGenerator256(nn.Module):
    """
    Conditional Generator mapping latent vector z and class embedding
    to synthetic 3x256x256 RGB chest X-ray slices with values scaled to [-1, 1].

    Upsampling chain: 4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64 -> 128x128 -> 256x256
    (6 transposed conv blocks after the initial 4x4 projection).
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

        # Transposed Convolution blocks upsampling from 4x4 -> 256x256
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
            nn.ConvTranspose2d(feature_maps // 2, feature_maps // 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps // 4),
            nn.ReLU(True),

            # 128x128 -> 256x256
            nn.ConvTranspose2d(feature_maps // 4, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),  # Normalized outputs in range [-1, 1]
        )

        self.apply(weights_init_256)

    def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            noise: (B, latent_dim) random Gaussian vector
            labels: (B,) class indices [0..num_classes-1]
        Returns:
            (B, 3, 256, 256) synthetic images in [-1, 1]
        """
        c = self.label_embedding(labels)  # (B, embedding_dim)
        x = torch.cat([noise, c], dim=1)  # (B, latent_dim + embedding_dim)
        x = self.project(x)
        x = x.view(-1, self.feature_maps * 8, 4, 4)
        out = self.deconv_blocks(x)
        return out


class ConditionalDiscriminator256(nn.Module):
    """
    Conditional Discriminator network. Takes an RGB image (3x256x256)
    and class label, projecting the label into a spatial feature channel,
    and returns unnormalized authenticity logits.

    Downsampling chain: 256x256 -> 128x128 -> 64x64 -> 32x32 -> 16x16 -> 8x8 -> 4x4 -> 1x1
    (7 conv blocks).
    """

    def __init__(
        self,
        num_classes: int = 4,
        in_channels: int = 3,
        feature_maps: int = 64,
        img_size: int = 256,
    ):
        super().__init__()
        self.img_size = img_size
        self.num_classes = num_classes

        # Class label spatial embedding (1 x img_size x img_size)
        self.label_embedding = nn.Embedding(num_classes, img_size * img_size)

        # Input channels = image channels (3) + conditioned label channel (1) = 4
        total_in_channels = in_channels + 1

        self.conv_blocks = nn.Sequential(
            # 256x256 -> 128x128 (No BatchNorm in the first discriminator layer per DCGAN guidance)
            nn.Conv2d(total_in_channels, feature_maps // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # 128x128 -> 64x64
            nn.Conv2d(feature_maps // 2, feature_maps, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.LeakyReLU(0.2, inplace=True),

            # 64x64 -> 32x32
            nn.Conv2d(feature_maps, feature_maps * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # 32x32 -> 16x16
            nn.Conv2d(feature_maps * 2, feature_maps * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # 16x16 -> 8x8
            nn.Conv2d(feature_maps * 4, feature_maps * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # 8x8 -> 4x4
            nn.Conv2d(feature_maps * 8, feature_maps * 16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 16),
            nn.LeakyReLU(0.2, inplace=True),

            # 4x4 -> 1x1 logit
            nn.Conv2d(feature_maps * 16, 1, kernel_size=4, stride=1, padding=0, bias=False),
        )

        self.apply(weights_init_256)

    def forward(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, 3, 256, 256)
            labels: (B,) class indices [0..num_classes-1]
        Returns:
            (B,) unnormalized logit (real vs fake score)
        """
        batch_size = images.size(0)
        # Embed label and reshape to (B, 1, H, W)
        c = self.label_embedding(labels).view(batch_size, 1, self.img_size, self.img_size)
        x = torch.cat([images, c], dim=1)  # (B, 4, 256, 256)
        logit = self.conv_blocks(x)        # (B, 1, 1, 1)
        return logit.view(batch_size)
