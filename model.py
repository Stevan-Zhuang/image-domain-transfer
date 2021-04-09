# Machine learning
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
import pytorch_lightning as pl
from components import Generator, Discriminator
from utils import gradient_penalty

# Saving image results
import os
import matplotlib.pyplot as plt
from utils import make_image_grid, plot_image

# Type hinting documentation
from typing import Tuple
from argparse import Namespace
from torch import Tensor
from pytorch_lightning import LightningDataModule

class DomainTransferGAN(pl.LightningModule):
    """
    A Generative Adversarial Network that generates images with
    specific attributes changed from an original input image.
    Implementation of the StarGAN model.
    https://arxiv.org/abs/1711.09020
    """
    def __init__(self, config: Namespace) -> None:
        super(DomainTransferGAN, self).__init__()
        self.config = config

        self.generator = Generator(self.config)
        self.discriminator = Discriminator(self.config)

    def forward(self, x: Tensor, labels: Tensor) -> Tensor:
        """Return network output."""
        return self.generator(x, labels)

    def training_step(self, batch: Tuple[Tensor, Tensor],
                      batch_idx: int, optimizer_idx: int) -> Tensor:
        """A single training step on a batch from the dataloader."""
        opt_g, opt_d = self.optimizers()

        real_images, real_labels = batch
        batch_size = real_images.size(0)
        # Assign each image a label from another random image
        fake_labels = real_labels[torch.randperm(batch_size)]

        # Train discriminator
        pred_real_error, pred_real_labels = self.discriminator(real_images)

        # Train discriminator to classify image labels
        class_loss = self.classification_loss(pred_real_labels, real_labels)

        fake_images = self.generator(real_images, fake_labels)
        pred_fake_error, pred_fake_labels = self.discriminator(fake_images)

        # Train discriminator to recognize real patches
        real_loss = -pred_real_error.mean()
        # Train discriminator to recognize fake patches
        fake_loss = pred_fake_error.mean()

        # Create random noise image sample
        alpha = torch.rand(batch_size, 1, 1, 1)
        alpha = alpha.expand_as(real_images)

        # Create images distanced between the real set and the fake set
        # selecting random points between a real image and a fake image
        interpolated = (alpha * real_images.detach()
                        + (1 - alpha) * fake_images.detach())
        interpolated.requires_grad_(True)

        prob_interpolated, _ = self.discriminator(interpolated)
        # Push the gradients towards one to prevent vanishing gradients
        grad_loss = gradient_penalty(prob_interpolated, interpolated)

        # Scale and sum up losses
        class_loss *= self.config.class_weight
        grad_loss *= self.config.grad_weight
        loss = class_loss + real_loss + fake_loss + grad_loss

        self.log('discriminator loss', loss)
        self.log('discriminator classification loss', class_loss)
        self.log('discriminator real images loss', real_loss)
        self.log('discriminator fake images loss', fake_loss)
        self.log('discriminator gradient loss', grad_loss)

        # Update discriminator weights
        opt_d.zero_grad()
        self.manual_backward(loss, opt_d)
        opt_d.step()

        if (batch_idx + 1) % self.config.n_critic == 0:
            # Train generator
            fake_images = self.generator(real_images, fake_labels)
            pred_fake_error, pred_fake_labels = self.discriminator(fake_images)
            
            # Train generator to make images that match target labels
            class_loss = self.classification_loss(pred_fake_labels, real_labels)

            # Train generator to make realistic images
            fake_loss = -pred_fake_error.mean()

            # Reconstruct original image with generated image and original labels
            recon_images = self.generator(fake_images, real_labels)
            # Train generator to make resulting images similar to original
            recon_loss = F.l1_loss(recon_images, real_images)

            # Scale and sum up losses
            class_loss *= self.config.class_weight
            recon_loss *= self.config.recon_weight
            loss = fake_loss + class_loss + recon_loss

            self.log('generator loss', loss)
            self.log('generator realism loss', fake_loss)
            self.log('generator classification loss', class_loss)
            self.log('generator reconstruction loss', recon_loss)

            # Update generator weights
            opt_g.zero_grad()
            self.manual_backward(loss, opt_g)
            opt_g.step()

    def test_step(self, batch: Tuple[Tensor, Tensor],
                        batch_idx: int) -> Tensor:
        """Save predictions on a small set of images from the generator."""
        # Manually put the model into training mode
        # When not in training mode, instance normalization layers
        # seems to be disabled, which leads to bad colourization
        self.train()

        make_image_grid(self, batch, self.config)
        
        # Don't save over existing file
        count = 0
        while os.path.exists(self.config.result_path + f'/{count}.png'):
            count += 1
        plt.savefig(self.config.result_path + f'/{count}.png')

        # Close figures to prevent too much memory usage
        plt.close('all')

    def configure_optimizers(self) -> Tuple[optim.Adam, optim.Adam]:
        """Return optimizers for generator and discriminator."""
        return (
            optim.Adam(self.generator.parameters(), lr=self.config.lr,
                       betas=(self.config.beta1, self.config.beta2)),
            optim.Adam(self.discriminator.parameters(), lr=self.config.lr,
                       betas=(self.config.beta1, self.config.beta2))
        )

    def classification_loss(self, y_pred: Tensor, y: Tensor) -> Tensor:
        """Calculate how close predicted logits are to ground truth."""
        batch_size = y_pred.size(0)
        loss = F.binary_cross_entropy_with_logits(
            y_pred, y, reduction='sum'
        )
        return loss / batch_size

    def load_checkpoint(self) -> None:
        """Load pretrained weights"""
        g_state = torch.load(self.config.g_checkpoint)
        d_state = torch.load(self.config.d_checkpoint)

        self.generator.load_state_dict(g_state)
        self.discriminator.load_state_dict(d_state)

    def save_checkpoint(self) -> None:
        """Save model weights for further training or deployment."""
        torch.save(self.generator.state_dict(), self.config.g_checkpoint)
        torch.save(self.discriminator.state_dict(), self.config.d_checkpoint)

    def checkpoints_exist(self) -> bool:
        """Return whether model checkpoints exist at file paths."""
        return (os.path.exists(self.config.g_checkpoint)
                and os.path.exists(self.config.d_checkpoint))

    def manual_test_step(self, dm: LightningDataModule) -> None:
        """Manually perform the test step. Used for debugging."""
        dm.setup()
        batch = next(iter(dm.test_dataloader()))
        self.test_step(batch, 0)

    @property
    def automatic_optimization(self) -> bool:
        """
        Tell PyTorch Lightning to not automatically optimize weights.
        This is so it can be done manually, with more freedom in training.
        """
        return False
