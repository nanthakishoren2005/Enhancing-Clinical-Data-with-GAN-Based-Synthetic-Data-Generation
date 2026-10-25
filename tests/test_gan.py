"""
Unit tests for Conditional DCGAN Generator and Discriminator forward passes and shapes.
"""
import torch
from src.gan.generator import ConditionalGenerator
from src.gan.discriminator import ConditionalDiscriminator


def test_generator_forward():
    batch_size = 4
    latent_dim = 100
    num_classes = 3
    embedding_dim = 32

    netG = ConditionalGenerator(
        latent_dim=latent_dim,
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        feature_maps=32,
    )

    noise = torch.randn(batch_size, latent_dim)
    labels = torch.randint(0, num_classes, (batch_size,))

    fake_imgs = netG(noise, labels)
    assert fake_imgs.shape == (batch_size, 3, 128, 128)
    # Check Tanh range [-1, 1]
    assert fake_imgs.min() >= -1.05
    assert fake_imgs.max() <= 1.05


def test_discriminator_forward():
    batch_size = 4
    num_classes = 3
    netD = ConditionalDiscriminator(
        num_classes=num_classes,
        in_channels=3,
        feature_maps=32,
        img_size=128,
    )

    imgs = torch.randn(batch_size, 3, 128, 128)
    labels = torch.randint(0, num_classes, (batch_size,))

    logits = netD(imgs, labels)
    assert logits.shape == (batch_size,)
