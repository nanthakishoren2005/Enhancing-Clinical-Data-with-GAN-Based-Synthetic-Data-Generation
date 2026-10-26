"""
Conditional DDPM (Denoising Diffusion Probabilistic Model) for 128x128 chest X-ray synthesis (Phase 3).
Implements a conditional diffusion model with class conditioning for 4-class chest X-ray generation.

Key features:
- U-Net backbone with time embedding and class conditioning
- Forward diffusion: q(x_t | x_0) with linear beta schedule
- Reverse diffusion: p_theta(x_{t-1} | x_t) learned denoising
- DDIM fast sampling (50-100 steps) for rapid generation
- Class conditioning via embedding concatenation to time + image features

Usage:
    python -m scripts.train_ddpm --epochs 50 --batch-size 16
    python -m scripts.train_ddpm --sample-only --num-per-class 100 --ddim-steps 50
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_timestep_embedding(timesteps, embedding_dim):
    """
    Build sinusoidal timestep embeddings (like Transformer positional encoding).
    Args:
        timesteps: (B,) 1-D tensor of timesteps
        embedding_dim: dimension of embeddings
    Returns:
        (B, embedding_dim) timestep embeddings
    """
    assert embedding_dim % 2 == 0, "embedding_dim must be even"
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    return emb


def conv3x3(in_ch, out_ch, stride=1):
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)


def conv1x1(in_ch, out_ch):
    return nn.Conv2d(in_ch, out_ch, kernel_size=1)


class ResidualBlock(nn.Module):
    """ResNet block with time and class conditioning."""
    def __init__(self, in_ch, out_ch, time_emb_dim, class_emb_dim, dropout=0.1):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch),
        )
        self.class_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(class_emb_dim, out_ch),
        )

        self.block1 = nn.Sequential(
            nn.GroupNorm(8, in_ch),
            nn.SiLU(),
            conv3x3(in_ch, out_ch),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Dropout(dropout),
            conv3x3(out_ch, out_ch),
        )

        self.residual_conv = conv1x1(in_ch, out_ch) if in_ch != out_ch else nn.Identity()

    def forward(self, x, time_emb, class_emb):
        h = self.block1(x)
        h = h + self.time_mlp(time_emb)[:, :, None, None]
        h = h + self.class_mlp(class_emb)[:, :, None, None]
        h = self.block2(h)
        return h + self.residual_conv(x)


class AttentionBlock(nn.Module):
    """Self-attention block for capturing long-range dependencies."""
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = conv1x1(channels, channels * 3)
        self.proj = conv1x1(channels, channels)

    def forward(self, x):
        b, c, h, w = x.shape
        h_in = self.norm(x)
        qkv = self.qkv(h_in)
        q, k, v = qkv.chunk(3, dim=1)

        q = q.view(b, self.num_heads, c // self.num_heads, h * w)
        k = k.view(b, self.num_heads, c // self.num_heads, h * w)
        v = v.view(b, self.num_heads, c // self.num_heads, h * w)

        attn = torch.softmax(torch.matmul(q.transpose(-2, -1), k) / math.sqrt(c // self.num_heads), dim=-1)
        out = torch.matmul(attn, v.transpose(-2, -1).transpose(-2, -1))
        out = out.view(b, c, h, w)
        return x + self.proj(out)


class UNet(nn.Module):
    """
    U-Net backbone for conditional diffusion.
    Encoder: downsample with residual blocks + attention
    Bottleneck: residual + attention + residual
    Decoder: upsample with residual blocks + attention + skip connections
    """
    def __init__(self, in_ch=3, out_ch=3, base_ch=64, channel_mult=(1, 2, 4, 8),
                 num_classes=4, time_emb_dim=256, class_emb_dim=128, dropout=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.time_emb_dim = time_emb_dim
        self.class_emb_dim = class_emb_dim

        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        # Class embedding
        self.class_embed = nn.Embedding(num_classes, class_emb_dim)
        self.class_mlp = nn.Sequential(
            nn.Linear(class_emb_dim, class_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(class_emb_dim * 4, class_emb_dim),
        )

        # Initial convolution
        self.init_conv = conv3x3(in_ch, base_ch)

        # Encoder
        self.down_blocks = nn.ModuleList()
        chs = [base_ch]
        in_ch_ = base_ch
        for i, mult in enumerate(channel_mult):
            out_ch = base_ch * mult
            for _ in range(2):
                self.down_blocks.append(ResidualBlock(in_ch_, out_ch, time_emb_dim, class_emb_dim, dropout))
                in_ch_ = out_ch
                chs.append(in_ch_)
            if i != len(channel_mult) - 1:
                self.down_blocks.append(nn.Conv2d(in_ch_, in_ch_, 4, stride=2, padding=1))
                chs.append(in_ch_)

        # Bottleneck
        self.mid_block1 = ResidualBlock(in_ch_, in_ch, time_emb_dim, class_emb_dim, dropout)
        self.mid_attn = AttentionBlock(in_ch)
        self.mid_block2 = ResidualBlock(in_ch, in_ch, time_emb_dim, class_emb_dim, dropout)

        # Decoder
        self.up_blocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(channel_mult))):
            out_ch = base_ch * mult
            for _ in range(3):
                self.up_blocks.append(ResidualBlock(chs.pop() + in_ch_, out_ch, time_emb_dim, class_emb_dim, dropout))
                in_ch_ = out_ch
            if i != 0:
                self.up_blocks.append(nn.ConvTranspose2d(in_ch_, in_ch_, 4, stride=2, padding=1))

        self.out_norm = nn.GroupNorm(8, base_ch)
        self.out_act = nn.SiLU()
        self.out_conv = conv3x3(base_ch, out_ch)

    def forward(self, x, timesteps, labels):
        """
        Args:
            x: (B, 3, H, W) noisy image
            timesteps: (B,) timestep indices
            labels: (B,) class indices
        Returns:
            (B, 3, H, W) predicted noise
        """
        # Time and class embeddings
        t_emb = get_timestep_embedding(timesteps, self.time_emb_dim)
        t_emb = self.time_mlp(t_emb)
        c_emb = self.class_mlp(self.class_embed(labels))

        # Encoder with skip connections
        h = self.init_conv(x)
        skip_connections = [h]
        for layer in self.down_blocks:
            if isinstance(layer, ResidualBlock):
                h = layer(h, t_emb, c_emb)
            else:
                h = layer(h)
            skip_connections.append(h)

        # Bottleneck
        h = self.mid_block1(h, t_emb, c_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb, c_emb)

        # Decoder with skip connections
        for layer in self.up_blocks:
            if isinstance(layer, ResidualBlock):
                skip = skip_connections.pop()
                h = torch.cat([h, skip], dim=1)
                h = layer(h, t_emb, c_emb)
            else:
                h = layer(h)

        h = self.out_norm(h)
        h = self.out_act(h)
        return self.out_conv(h)


class ConditionalDDPM(nn.Module):
    """
    Conditional DDPM for chest X-ray synthesis.
    Supports DDPM training and DDIM fast sampling.
    """
    def __init__(self, num_classes=4, img_size=128, beta_start=1e-4, beta_end=0.02, num_timesteps=1000):
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        self.num_timesteps = num_timesteps

        # Beta schedule (linear)
        self.register_buffer("betas", torch.linspace(beta_start, beta_end, num_timesteps))
        self.register_buffer("alphas", 1.0 - self.betas)
        self.register_buffer("alphas_cumprod", torch.cumprod(self.alphas, dim=0))
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(self.alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - self.alphas_cumprod))

        # U-Net
        self.model = UNet(
            in_ch=3, out_ch=3,
            base_ch=64, channel_mult=(1, 2, 4, 8),
            num_classes=num_classes,
        )

    def q_sample(self, x_0, t, noise=None):
        """
        Forward diffusion: q(x_t | x_0) = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        return sqrt_alpha_bar * x_0 + sqrt_one_minus * noise, noise

    def p_losses(self, x_0, labels, noise=None):
        """
        Compute training loss: predict noise added at timestep t.
        """
        batch_size = x_0.size(0)
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=x_0.device)
        x_t, noise = self.q_sample(x_0, t, noise)
        predicted_noise = self.model(x_t, t, labels)
        return F.mse_loss(noise, predicted_noise)

    @torch.no_grad()
    def p_sample(self, x_t, t, labels, clip_denoised=True):
        """
        Single reverse diffusion step: p_theta(x_{t-1} | x_t)
        """
        betas_t = self.betas[t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        sqrt_recip_alphas_t = torch.sqrt(1.0 / self.alphas[t])[:, None, None, None]

        model_mean = sqrt_recip_alphas_t * (
            x_t - betas_t * self.model(x_t, t, labels) / sqrt_one_minus_alphas_cumprod_t
        )

        if t[0] == 0:
            return model_mean
        else:
            posterior_variance_t = betas_t
            noise = torch.randn_like(x_t)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample(self, labels, num_timesteps=None):
        """
        Full DDPM sampling (slow, num_timesteps steps).
        """
        if num_timesteps is None:
            num_timesteps = self.num_timesteps
        batch_size = labels.size(0)
        device = labels.device
        x = torch.randn(batch_size, 3, self.img_size, self.img_size, device=device)
        for i in reversed(range(num_timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, t, labels)
        return x

    @torch.no_grad()
    def ddim_sample(self, labels, num_steps=50, eta=0.0):
        """
        DDIM fast sampling (num_steps steps, eta=0 for deterministic).
        """
        batch_size = labels.size(0)
        device = labels.device

        # DDIM timestep sequence
        timesteps = torch.linspace(0, self.num_timesteps - 1, num_steps, dtype=torch.long, device=device)
        timesteps = timesteps.flip(0)

        x = torch.randn(batch_size, 3, self.img_size, self.img_size, device=device)

        for i in range(len(timesteps)):
            t = timesteps[i]
            t_tensor = torch.full((batch_size,), t.item(), device=device, dtype=torch.long)

            # Predict noise
            eps_theta = self.model(x, t_tensor, labels)

            # DDIM update
            t_val = int(t.item())
            alpha_bar_t = self.alphas_cumprod[t_val][:, None, None, None]
            if i + 1 < len(timesteps):
                t_prev_val = int(timesteps[i + 1].item())
                alpha_bar_t_prev = self.alphas_cumprod[t_prev_val][:, None, None, None]
            else:
                alpha_bar_t_prev = torch.tensor(1.0, device=device)[:, None, None, None]

            sigma_t = eta * torch.sqrt(
                (1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_t_prev)
            )

            # Predict x_0
            sqrt_alpha_bar = torch.sqrt(alpha_bar_t)
            sqrt_one_minus = torch.sqrt(1 - alpha_bar_t)
            x_0_pred = (x - sqrt_one_minus * eps_theta) / sqrt_alpha_bar

            # Direction pointing to x_t
            dir_xt = torch.sqrt(1 - alpha_bar_t_prev - sigma_t**2) * eps_theta

            # Sample x_{t-1}
            if eta > 0:
                noise = torch.randn_like(x)
            else:
                noise = 0
            x = torch.sqrt(alpha_bar_t_prev) * x_0_pred + dir_xt + sigma_t * noise

        return x


def ddpm_weights_init(m):
    """Initialize DDPM weights."""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 or classname.find("Linear") != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0.0)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif classname.find("GroupNorm") != -1:
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
