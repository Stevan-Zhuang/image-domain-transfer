# Machine learning
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
import pytorch_lightning as pl

# Type hinting documentation
from typing import Tuple
from argparse import Namespace
from torch import Tensor

class ResBlock(nn.Module):
    """A Residual Block that allows input to skip layers."""
    def __init__(self, in_channels: int) -> None:
        super(ResBlock, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels, in_channels, bias=False,
                kernel_size=3, stride=1, padding=1
            ),
            nn.InstanceNorm2d(
                in_channels, affine=True,
                track_running_stats=True
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels, in_channels, bias=False,
                kernel_size=3, stride=1, padding=1
            ),
            nn.InstanceNorm2d(
                in_channels, affine=True,
                track_running_stats=True
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Return network output."""
        identity = x
        # Skip connection
        return self.net(x) + identity

class Generator(nn.Module):
    """GAN Generator Network utilizing residual blocks."""
    def __init__(self, config: Namespace) -> None:
        super(Generator, self).__init__()
        self.config = config

        # Extract shapes and edges from image
        feature_extraction_layer = [
            nn.Conv2d(
                self.config.image_end[0] + self.config.n_labels,
                self.config.g_hidden,
                kernel_size=7, stride=1, padding=3, bias=False
            ),
            nn.InstanceNorm2d(
                self.config.g_hidden, affine=True,
                track_running_stats=True
            ),
            nn.ReLU(inplace=True)
        ]
        # Decrease image spacial data
        downscale_layer = []
        for layer in range(2):
            downscale_layer.extend([
                # Double hidden every new layer
                nn.Conv2d(
                    self.config.g_hidden * (2**layer),
                    self.config.g_hidden * (2**(layer + 1)),
                    kernel_size=4, stride=2, padding=1, bias=False
                ),
                nn.InstanceNorm2d(
                    self.config.g_hidden * (2**(layer + 1)), affine=True,
                    track_running_stats=True
                ),
                nn.ReLU(inplace=True)
            ])
        # Residual network
        bottleneck_layer = [
            ResBlock(self.config.g_hidden * 4) 
            for _ in range(self.config.g_n_blocks)
        ]
        # Increase image spacial data
        upscale_layer = []
        for layer in reversed(range(2)):
            upscale_layer.extend([
                # Halve hidden every new layer
                nn.ConvTranspose2d(
                    self.config.g_hidden * (2**(layer + 1)),
                    self.config.g_hidden * (2**layer),
                    kernel_size=4, stride=2, padding=1, bias=False
                ),
                nn.InstanceNorm2d(
                    self.config.g_hidden * (2**layer), affine=True,
                    track_running_stats=True
                ),
                nn.ReLU(inplace=True)
            ])
        # Final layer that generates image
        fully_connected_layer = nn.Sequential(
            nn.Conv2d(
                self.config.g_hidden, self.config.image_end[0],
                kernel_size=7, stride=1, padding=3, bias=False
            ),
            nn.Tanh() # Force output to range [-1, 1]
        )
        # Combine layers
        self.net = nn.Sequential(
            *feature_extraction_layer,
            *downscale_layer,
            *bottleneck_layer,
            *upscale_layer,
            *fully_connected_layer,
        )

    def forward(self, x: Tensor, labels: Tensor) -> Tensor:
        """Return network output."""
        # For each label, create a a spacial reprensenation with
        # the same width and height as the image
        labels = labels.view(labels.size(0), labels.size(1), 1, 1)
        labels = labels.repeat(1, 1, x.size(2), x.size(3))

        # Stack these labels with the image
        x = torch.cat([x, labels], dim=1)

        out = self.net(x)
        return out

class Discriminator(nn.Module):
    """
    GAN Discriminator Network with WGAN-GP and patchGAN properties,
    along with an auxillary classifier.
    """
    def __init__(self, config: Namespace) -> None:
        super(Discriminator, self).__init__()
        self.config = config

        # Extract shapes and edges from image
        feature_extraction_layer = [
            nn.Conv2d(
                self.config.image_end[0], self.config.d_hidden,
                kernel_size=4, stride=2, padding=1
            ),
            nn.LeakyReLU(inplace=True)
        ]
        # Decrease image spacial data
        downscale_layer = []
        for layer in range(self.config.d_n_blocks):
            downscale_layer.extend([
                # Halve hidden every new layer
                nn.Conv2d(
                    self.config.d_hidden * (2**layer),
                    self.config.d_hidden * (2**(layer + 1)),
                    kernel_size=4, stride=2, padding=1
                ),
                nn.LeakyReLU(inplace=True)
            ])
        # Combine layers
        self.net = nn.Sequential(
            *feature_extraction_layer,
            *downscale_layer
        )

        # PatchGAN, classifies realism of patches on image
        self.patch_fc = nn.Conv2d(
            self.config.d_hidden * (2**self.config.d_n_blocks), 1,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        # Final image width and height after all convolutional layers
        image_size = (self.config.image_end[2]
                      // (2**(self.config.d_n_blocks + 1)))

        # Label classification
        self.class_fc = nn.Conv2d(
            self.config.d_hidden * (2**self.config.d_n_blocks),
            self.config.n_labels,
            kernel_size=image_size, bias=False
        )

    def forward(self, x: Tensor):
        """Return network output."""
        out = self.net(x)

        pred_error = self.patch_fc(out)
        pred_labels = self.class_fc(out).flatten(1)
        return pred_error, pred_labels
