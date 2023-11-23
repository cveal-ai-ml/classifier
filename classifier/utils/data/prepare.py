"""
Author: MINDFUL
Purpose: Load datasets for Deep Learning (DL) model analysis
"""


import os
import numpy as np

from torch.utils.data import DataLoader

from utils.data.transforms import load_data_transforms
from utils.data.loader import Dataset, Pytorch_Format


def gather_subset(dataset, percent):
    """
    Gather subset of dataset
    Subset is a percent number of observations for each class

    Parameters:
    - dataset (Dataset): supervised learning dataset
    - percent (list[int]): percent of dataset for subset
    """

    all_labels = np.asarray(dataset.labels)
    all_samples = np.asarray(dataset.samples)

    new_samples, new_labels = [], []
    for current_label in np.unique(all_labels):

        indices = np.where(current_label == all_labels)

        class_labels = all_labels[indices]
        class_samples = all_samples[indices]
        num_samples = int(class_samples.shape[0] * (percent / 100))

        new_labels.append(class_labels[:num_samples])
        new_samples.append(class_samples[:num_samples])

    new_labels = list(np.hstack(new_labels))
    new_samples = list(np.hstack(new_samples))

    return Dataset(new_samples, new_labels)


def gather_data(path, file_types=[".png", ".jpg"]):
    """
    Gather image dataset files

    Parameters:
    - path (str): path to image dataset folder
    - file_types (list[str]): valid file types

    Returns:
    - (Dataset): supervised learning dataset
    """

    # Gather: All Dataset Files (Folders)

    all_folders = os.listdir(path)
    all_folders.sort()

    # Load: Supervised Learning Dataset

    all_samples, all_labels = [], []

    for i, current_folder in enumerate(all_folders):

        all_files = os.listdir(os.path.join(path, current_folder))
        all_files.sort()

        for current_file in all_files:
            path_file = os.path.join(path, current_folder, current_file)
            all_samples.append(path_file)
            all_labels.append(i)

    return Dataset(all_samples, all_labels)


def load_datasets_train(path_train, path_valid):
    """
    Load datasets for DL training
    This includes a dataset for training, validation, and testing

    Parameters:
    - path_train (str): path to training dataset folder
    - path_valid (str): path to validation dataset folder

    Returns:
    - (tuple[Dataset]): relevant supervised learning datasets
    """

    train = gather_data(path_train)
    valid = gather_data(path_valid)

    return (train, valid)


def load_datasets_test(path_test):
    """
    Load datasets for DL testing
    This includes a dataset for testing only

    Parameters:
    - path_test (str): path to testing dataset folder

    Returns:
    - (Dataset): supervised learning dataset
    """

    return gather_data(path_test)


def load_test(params):
    """
    Load and format DL testing dataset as pytorch dataloader

    Parameters:
    - params (dict[str, any]): dataset parameters

    Returns:
    - (torch.util.data.DataLoaders): testing dataset
    """

    test = load_datasets_test(params["path_test"])

    transforms = load_data_transforms(params["transforms"],
                                      params["interpolate"],
                                      params["sample_shape"])

    test = Pytorch_Format(test, transforms, "valid")

    test = DataLoader(test, batch_size=params["batch_size"],
                      num_workers=params["num_workers"], shuffle=0)

    return test


def load_train(params):
    """
    Load and format DL training datasets as pytorch dataloader

    Parameters:
    - params (dict[str, any]): dataset parameters

    Returns:
    - (tuple[torch.util.data.DataLoaders]): relevant datasets
    """

    train, valid = load_datasets_train(params["path_train"],
                                       params["path_valid"])

    if params["use_subset"]["enable"]:
        train = gather_subset(train, params["use_subset"]["percent"])

    transforms = load_data_transforms(params["transforms"],
                                      params["interpolate"],
                                      params["sample_shape"])

    train = Pytorch_Format(train, transforms, "train")
    valid = Pytorch_Format(valid, transforms, "valid")

    train = DataLoader(train, persistent_workers=True,
                       batch_size=int(params["batch_size"]),
                       num_workers=params["num_workers"], shuffle=1)

    valid = DataLoader(valid, persistent_workers=True,
                       batch_size=int(params["batch_size"]),
                       num_workers=params["num_workers"], shuffle=0)

    return (train, valid)
