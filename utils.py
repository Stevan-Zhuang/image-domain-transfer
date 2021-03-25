# Machine learning
import torch
from torch import autograd

# Handling image data
from PIL import Image
from torchvision import transforms as T

# Image plotting
import matplotlib.pyplot as plt

# Type hinting documentation
from typing import Tuple, List
from argparse import Namespace
from torch import Tensor
from pytorch_lightning import LightningModule

def gradient_penalty(prob_interpolated: Tensor, interpolated: Tensor) -> Tensor:
    """Calculates gradient penalty for WGAN-GP"""
    # Gradient target
    weight = torch.ones_like(prob_interpolated)
    gradients = torch.autograd.grad(
        outputs=prob_interpolated,
        inputs=interpolated,
        grad_outputs=weight,
        create_graph=True,
        retain_graph=True,
    )[0]
    gradients = gradients.flatten(1)

    # Gradient penalty is the squared difference between 1 and the norm
    # of gradient of the predictions with respect to the input images.
    grad_norm = gradients.norm(dim=1)
    grad_penalty = (grad_norm - 1) ** 2
    return grad_penalty.mean()

def gen_alt_labels(labels: Tensor, names: List[str]) -> List[Tuple[Tensor, str]]:
    """Generate all possible label sets with a single label flipped."""
    alt_label_sets = []
    for label_idx, label_name in enumerate(names):
        # Ensure labels are not a pointer to the original
        alt_labels = labels.clone()
        
        # Change the label to its opposite, 1. to 0., 0. to 1.
        alt_labels[label_idx] = float(not alt_labels[label_idx])

        alt_label_sets.append((alt_labels, label_name))

    return alt_label_sets

def make_image_grid(model: LightningModule, batch: Tuple[Tensor, Tensor],
                    config: Namespace) -> None:
    """Create a grid of model predictions against original image."""
    images, label_sets = batch

    # Ensure there is no clipping between plots and titles
    fig, ax = plt.subplots(
        config.gen_images, 1 + config.n_labels,
        constrained_layout=True
    )
    for row in range(config.gen_images):
        image = images[row]
        labels = label_sets[row]

        # Plot original image to the left
        plot_image(ax[row, 0], image, 'Original', config)

        alt_label_sets = gen_alt_labels(labels, config.alt_label_names)

        # Plot all generated images for labels to the right
        # with each column representing a different attribute
        for col, (alt_labels, label_name) in enumerate(alt_label_sets):

            # Model expects data in batch format, add batch dimensions of 1
            gen_image = model(image.unsqueeze(0), alt_labels.unsqueeze(0))
            plot_image(ax[row, col + 1], gen_image, label_name, config)

def plot_image(plot, data: Tensor, name: str, config: Namespace) -> None:
    """Draw a tensor image on a subplot."""
    plot.set_title(name)
    plot.imshow(tensor_to_image(data, config))

    # Remove axis ticks
    plot.xaxis.set_visible(False)
    plot.yaxis.set_visible(False)

def image_to_tensor(data: Image, config: Namespace) -> Tensor:
    """Takes in a PIL Image and returns a float tensor."""
    # Crop the image into a square
    preprocess = T.Compose([
        T.CenterCrop(min(data.size)),
        T.Resize(config.image_end[2]),
        T.ToTensor(),
        T.Normalize(config.image_mean,
                    config.image_std)
    ])
    # Model expects a batch dimension, so add a dimension of size 1
    return preprocess(data).unsqueeze(0)

def tensor_to_image(data: Tensor, config: Namespace) -> Image:
    """Takes in a float tensor and returns a PIL Image."""
    # Stop tracking gradients and remove batch dimension
    data = data.detach().squeeze(0)

    # Switch to height, width, channels for PIL Image format
    array = data.numpy().transpose((1, 2, 0))

    # Reverse the normalization process (x - mean(x)) / std(x)
    array = array * config.image_std[0] + config.image_mean[0]

    # Pixels are now floats in range [0, 1]
    # PIL Image expects unsigned 8 bit integers in the range [0, 255]
    return Image.fromarray((array * 255).astype('uint8'))
