# Machine learning
import torch
import pytorch_lightning as pl

# Data tools
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms as T
import pandas as pd

# Type hinting documentation
from typing import Tuple
from argparse import Namespace
from torch import Tensor

class CelebADataset(Dataset):
    """A dataset that combines a set of images and domain labels."""
    def __init__(self, config: Namespace) -> None:
        self.images = ImageFolder(config.image_folder)
        self.n_images = len(self.images)

        labels = pd.read_csv(config.attr_csv)

        # Train only on small selection of domain labels
        labels = labels[config.choice_labels][:self.n_images]

        # Binary cross entropy loss with logits expects floats
        self.labels = torch.FloatTensor(labels.values)

        # Change -1 values to 0
        torch.clamp_min_(self.labels, 0)

        # The boundary between the training set and test set
        self.train_test_split = config.train_size

        # Preprocessing transformations applied to images
        self.transforms = {
            # Crop image to a square, as conv nets will perform better
            # Make the image smaller so training is faster
            # Normalize pixels to the range [-1, 1] for the model

            'augumented': T.Compose([
                # Random transformations can create new examples
                # and increase the size of the dataset
                T.RandomResizedCrop(config.image_end[2]),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(config.image_mean,
                            config.image_std)
            ]),
            'standard': T.Compose([
                T.CenterCrop(config.image_start[1]),
                T.Resize(config.image_end[2]),
                T.ToTensor(),
                T.Normalize(config.image_mean,
                            config.image_std)
            ])
        }

    def __len__(self) -> int:
        """Number of examples in dataset."""
        return self.n_images

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """Get example of image and domain labels by index."""
        # Ignore folder label
        image = self.images[idx][0]

        transform = self.transforms['standard']
        if idx < self.train_test_split:
            # If example is in training set, augument data
            transform = self.transforms['augumented']

        image = transform(image)

        labels = self.labels[idx]
        return image, labels

class CelebADataModule(pl.LightningDataModule):
    """
    A DataModule container for a sample of the CelebA dataset
    with only 2100 images. Prepares data to be used by the model
    (Transforms, split, etc).
    """
    def __init__(self, config: Namespace) -> None:
        super(CelebADataModule, self).__init__()
        self.config = config
        self.image_shape = self.config.image_end

    def setup(self, stage: str=None) -> None:
        """
        Prepare and splits CelebA dataset into
        training and test sets.
        """
        dataset = CelebADataset(self.config)

        # Seperate train and test datasets by index
        train_indices = list(range(self.config.train_size))
        test_indices = list(range(self.config.train_size,
                                 self.config.dataset_size))
                                 
        self.train_dataset = Subset(dataset, train_indices)
        self.test_dataset = Subset(dataset, test_indices)

    def train_dataloader(self) -> DataLoader:
        """Get dataloader of training dataset."""
        return DataLoader(
            self.train_dataset, batch_size=self.config.batch_size,
            shuffle=True, num_workers=self.config.num_workers
        )
    def test_dataloader(self) -> DataLoader:
        """Get dataloader of test dataset."""
        return DataLoader(
            self.test_dataset, batch_size=self.config.test_size,
            shuffle=False, num_workers=self.config.num_workers
        )
