"""
Conditional StyleGAN2-ADA for 128x128 chest X-ray synthesis (Phase 2).
Implements a style-based generator with adaptive discriminator augmentation.
Conditional: single model with class conditioning (not one model per class).

Architecture:
- Mapping Network: z -> w (style space) with class conditioning
- Generator: StyleBlocks with adaptive instance norm, progressive to 128x128
- Discriminator: Standard CNN with DiffAugment adaptive augmentation
- Adaptive augmentation: adjusts augmentation strength based on discriminator overfitting
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def stylegan_weights_init(m):
    """Initialize weights for StyleGAN2-ADA."""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 or classname.find("Linear") != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0.2)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class EqualLinear(nn.Module):
    """
    Equalized learning rate linear layer.
    Scales weights at runtime to maintain stable gradients.
    """
    def __init__(self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1.0, activation=None):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim) / lr_mul)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))
        else:
            self.register_parameter("bias", None)
        self.activation = activation
        self.lr_mul = lr_mul
        self.scale = (1 / math.sqrt(in_dim)) * lr_mul

    def forward(self, x):
        out = F.linear(x, self.weight * self.scale, self.bias * self.lr_mul if self.bias is not None else None)
        if self.activation == "lrelu":
            out = F.leaky_relu(out, negative_slope=0.2)
        elif self.activation == "relu":
            out = F.relu(out)
        return out


class StyleConv(nn.Module):
    """
    Style-based convolutional block with adaptive instance normalization.
    Applies style modulation from the style vector w.
    """
    def __init__(self, in_ch, out_ch, kernel_size=3, style_dim=512, upsample=False):
        super().__init__()
        self.upsample = upsample
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size // 2)
        self.modulation = EqualLinear(style_dim, in_ch, bias_init=1)
        self.noise_weight = nn.Parameter(torch.zeros(1))
        self.bias = nn.Parameter(torch.zeros(out_ch))
        self.activate = nn.LeakyReLU(0.2)

        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size // 2)

    def forward(self, x, style, noise=None):
        # Style modulation
        style = self.modulation(style)  # (B, in_ch)
        x = x * (style[:, :, None, None] + 1)  # Modulate

        # Convolution
        out = self.conv(x)

        # Noise injection
        if noise is None:
            noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
        out = out + self.noise_weight * noise

        # Bias and activation
        out = out + self.bias.view(1, -1, 1, 1)
        out = self.activate(out)
        return out


class ToRGB(nn.Module):
    """1x1 convolution to convert feature maps to RGB image."""
    def __init__(self, in_ch, style_dim=512):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, 3, kernel_size=1)
        self.modulation = EqualLinear(style_dim, in_ch, bias_init=1)
        self.bias = nn.Parameter(torch.zeros(3))

    def forward(self, x, style):
        style = self.modulation(style)
        x = x * (style[:, :, None, None] + 1)
        out = self.conv(x)
        out = out + self.bias.view(1, -1, 1, 1)
        return out


class GeneratorBlock(nn.Module):
    """Generator block: upsample + 2 style convs."""
    def __init__(self, in_ch, out_ch, style_dim=512, upsample=True):
        super().__init__()
        self.upsample = upsample
        self.conv1 = StyleConv(in_ch, out_ch, style_dim=style_dim, upsample=upsample)
        self.conv2 = StyleConv(out_ch, out_ch, style_dim=style_dim)

    def forward(self, x, style1, style2, noise1=None, noise2=None):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.conv1(x, style1, noise1)
        x = self.conv2(x, style2, noise2)
        return x


class StyleGAN2Generator(nn.Module):
    """
    Style-based Generator for conditional image synthesis.
    Maps z -> w (mapping network), then uses w for style modulation
    across generator blocks from 4x4 to 128x128.
    """
    def __init__(self, latent_dim=512, style_dim=512, num_classes=4, channel_multiplier=1):
        super().__init__()
        self.latent_dim = latent_dim
        self.style_dim = style_dim
        self.num_classes = num_classes

        # Class embedding (concatenated to z)
        self.class_embed = nn.Embedding(num_classes, latent_dim)

        # Mapping network: z -> w (8 layers)
        self.mapping = nn.Sequential(
            EqualLinear(latent_dim * 2, style_dim, activation="lrelu"),
            EqualLinear(style_dim, style_dim, activation="lrelu"),
            EqualLinear(style_dim, style_dim, activation="lrelu"),
            EqualLinear(style_dim, style_dim, activation="lrelu"),
            EqualLinear(style_dim, style_dim, activation="lrelu"),
            EqualLinear(style_dim, style_dim, activation="lrelu"),
            EqualLinear(style_dim, style_dim, activation="lrelu"),
            EqualLinear(style_dim, style_dim, activation="lrelu"),
        )

        # Initial constant input
        base_ch = int(512 * channel_multiplier)
        self.start_const = nn.Parameter(torch.randn(1, base_ch, 4, 4))
        self.start_conv = StyleConv(base_ch, base_ch, style_dim=style_dim)

        # Generator blocks: 4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64 -> 128x128
        ch_in = base_ch
        self.blocks = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()

        chs = [base_ch, base_ch // 2, base_ch // 4, base_ch // 8, base_ch // 16]
        for i, out_ch in enumerate(chs):
            upsample = i > 0  # No upsample on first block (already at 4x4 via start_conv)
            self.blocks.append(
                GeneratorBlock(ch_in, out_ch, style_dim=style_dim, upsample=upsample)
            )
            self.to_rgbs.append(ToRGB(out_ch, style_dim=style_dim))
            ch_in = out_ch

        self.apply(stylegan_weights_init)

    def get_styles(self, z, labels):
        """Map z and class label to style vector w."""
        c = self.class_embed(labels)
        x = torch.cat([z, c], dim=1)  # (B, latent_dim * 2)
        w = self.mapping(x)
        return w

    def forward(self, z, labels, noise=None):
        """
        Args:
            z: (B, latent_dim) random noise
            labels: (B,) class indices
            noise: optional list of noise tensors for each block
        Returns:
            (B, 3, 128, 128) synthetic image
        """
        w = self.get_styles(z, labels)  # (B, style_dim)
        batch_size = z.size(0)

        # Start from constant
        x = self.start_const.expand(batch_size, -1, -1, -1)
        x = self.start_conv(x, w)

        if noise is None:
            noise = [None] * len(self.blocks)

        # Pass through generator blocks
        for i, (block, to_rgb) in enumerate(zip(self.blocks, self.to_rgbs)):
            x = block(x, w, w, noise[i] if i < len(noise) else None)

        # Final RGB
        img = to_rgb(x, w)
        return torch.tanh(img)


class DiffAugment:
    """
    Differentiable Augmentation for discriminator.
    Adaptively applies augmentation to prevent overfitting.
    Based on StyleGAN2-ADA's adaptive discriminator augmentation.
    """
    def __init__(self, prob=0.0, mode="color,translation,cutout"):
        self.prob = prob
        self.mode = mode
        self.augment_pipe = self._build_augment_pipe()

    def _build_augment_pipe(self):
        """Build augmentation pipeline from mode string."""
        ops = []
        if "color" in self.mode:
            ops.append(self._color_jitter)
        if "translation" in self.mode:
            ops.append(self._translation)
        if "cutout" in self.mode:
            ops.append(self._cutout)
        if "rotate" in self.mode:
            ops.append(self._rotate)
        return ops

    def _color_jitter(self, x):
        """Random brightness/contrast adjustment."""
        batch_size = x.size(0)
        brightness = torch.randn(batch_size, 1, 1, 1, device=x.device) * 0.1
        contrast = torch.randn(batch_size, 1, 1, 1, device=x.device) * 0.1 + 1.0
        return x * contrast + brightness

    def _translation(self, x):
        """Random translation."""
        batch_size, _, h, w = x.shape
        tx = torch.randint(-2, 3, (batch_size,), device=x.device)
        ty = torch.randint(-2, 3, (batch_size,), device=x.device)
        # Simple roll-based translation
        x = torch.roll(x, shifts=tx.tolist(), dims=3)
        x = torch.roll(x, shifts=ty.tolist(), dims=2)
        return x

    def _cutout(self, x):
        """Random cutout."""
        batch_size, _, h, w = x.shape
        for i in range(batch_size):
            if torch.rand(1) < 0.5:
                size = torch.randint(4, 16, (1,)).item()
                x1 = torch.randint(0, max(1, h - size), (1,)).item()
                y1 = torch.randint(0, max(1, w - size), (1,)).item()
                x[i, :, x1:x1 + size, y1:y1 + size] = 0
        return x

    def _rotate(self, x):
        """Random 90-degree rotation."""
        k = torch.randint(0, 4, (1,)).item()
        if k > 0:
            return torch.rot90(x, k, dims=[2, 3])
        return x

    def __call__(self, x):
        if self.prob <= 0 or not self.training:
            return x
        if torch.rand(1).item() < self.prob:
            for op in self.augment_pipe:
                x = op(x)
        return x

    def train(self, mode=True):
        self.training = mode
        return self

    def eval(self):
        return self.train(False)


class AdaptiveAugment:
    """
    Adaptive augmentation controller.
    Adjusts augmentation probability based on discriminator overfitting signal.
    Based on StyleGAN2-ADA's heuristic.
    """
    def __init__(self, initial_prob=0.0, target=0.6, speed=0.1, interval=4):
        self.prob = initial_prob
        self.target = target
        self.speed = speed
        self.interval = interval
        self.steps = 0

    def step(self, discriminator_real_prob):
        """
        Called every `interval` discriminator steps.
        discriminator_real_prob: average probability discriminator assigns to real images.
        If > target, discriminator is too strong -> increase augmentation.
        If < target, discriminator is too weak -> decrease augmentation.
        """
        self.steps += 1
        if self.steps % self.interval != 0:
            return self.prob

        error = discriminator_real_prob - self.target
        self.prob += self.speed * error
        self.prob = max(0.0, min(1.0, self.prob))
        return self.prob


class StyleGAN2Discriminator(nn.Module):
    """
    Discriminator for StyleGAN2-ADA.
    Standard CNN with spectral norm-like conditioning and adaptive augmentation.
    """
    def __init__(self, num_classes=4, channel_multiplier=1, img_size=128):
        super().__init__()
        self.img_size = img_size
        self.num_classes = num_classes

        # Class label embedding as spatial channel
        self.class_embed = nn.Embedding(num_classes, img_size * img_size)

        base_ch = int(64 * channel_multiplier)

        self.blocks = nn.ModuleList([
            # 128x128 -> 64x64
            nn.Conv2d(4, base_ch, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),

            # 64x64 -> 32x32
            nn.Conv2d(base_ch, base_ch * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_ch * 2),
            nn.LeakyReLU(0.2),

            # 32x32 -> 16x16
            nn.Conv2d(base_ch * 2, base_ch * 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_ch * 4),
            nn.LeakyReLU(0.2),

            # 16x16 -> 8x8
            nn.Conv2d(base_ch * 4, base_ch * 8, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_ch * 8),
            nn.LeakyReLU(0.2),

            # 8x8 -> 4x4
            nn.Conv2d(base_ch * 8, base_ch * 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_ch * 16),
            nn.LeakyReLU(0.2),

            # 4x4 -> 1x1
            nn.Conv2d(base_ch * 16, 1, 4, stride=1, padding=0),
        ])

        self.apply(stylegan_weights_init)

    def forward(self, images, labels):
        batch_size = images.size(0)
        # Embed label as spatial channel
        c = self.class_embed(labels).view(batch_size, 1, self.img_size, self.img_size)
        x = torch.cat([images, c], dim=1)  # (B, 4, H, W)

        for block in self.blocks:
            x = block(x)

        return x.view(batch_size)


class StyleGAN2ADA(nn.Module):
    """
    Combined StyleGAN2-ADA model wrapper.
    """
    def __init__(self, latent_dim=512, style_dim=512, num_classes=4, img_size=128):
        super().__init__()
        self.generator = StyleGAN2Generator(
            latent_dim=latent_dim,
            style_dim=style_dim,
            num_classes=num_classes,
        )
        self.discriminator = StyleGAN2Discriminator(
            num_classes=num_classes,
            img_size=img_size,
        )
        self.augment = DiffAugment(prob=0.0)
        self.adaptive_aug = AdaptiveAugment()

    def forward(self, z, labels):
        """Generator forward pass."""
        return self.generator(z, labels)

    def discriminate(self, images, labels, apply_augment=True):
        """Discriminator forward pass with optional augmentation."""
        if apply_augment and self.augment.training:
            images = self.augment(images)
        return self.discriminator(images, labels)
