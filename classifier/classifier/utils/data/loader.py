"""
Author: MINDFUL
Purpose: Supervised learning dataset templates
"""


import torch
import numpy as np

from PIL import Image


class Dataset:

    """
    Create general supervised learning dataset template
    """

    def __init__(self, samples, labels):
        """
        Assign dataset parameters

        Parameters:
        - samples (np.ndarray[any]): dataset samples
        - labels (np.ndarray[any]): dataset labels
        """

        self.labels = labels
        self.samples = samples


class Pytorch_Format(torch.utils.data.Dataset):

    """
    Format dataset template for pytorch dataloader
    """

    def __init__(self, dataset, transforms, mode):
        """
        Assign dataset parameters

        Parameters:
        - dataset (Dataset): generic supervised learning dataset
        - transforms (dict[str, any]): data transforms / augmentations
        - mode (str): signifier for dataset augmentations (train or valid)
        """

        self.mode = mode
        self.dataset = dataset
        self.transforms = transforms

    def image_loader(self, data):
        """
        Load image from data file

        Parameters:
        - data: image data file

        Returns:
        - (np.ndarray[float]): loaded image
        """

        return np.asarray(Image.open(data).convert("RGB"))

    def __getitem__(self, index):
        """
        Gather the n-th dataset observation

        Parameters:
        - index (int): current iterator

        Returns:
        - (tuple[any]): supervised dataset observation
        """

        labels = self.dataset.labels[index]
        filenames = self.dataset.samples[index]

        samples = self.image_loader(filenames)
        samples = self.transforms[self.mode](image=samples)["image"].float()

        return (samples, labels, filenames)

    def __len__(self):
        """
        Gather the length of the dataset

        Returns:
        - (int): number of dataset observations
        """

        return len(self.dataset.samples)
