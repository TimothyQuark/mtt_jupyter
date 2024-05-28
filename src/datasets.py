from torchvision.datasets import CIFAR10, CIFAR100
from typing import Any, Callable, Tuple, Dict, List, Optional, Union
from pathlib import Path
import numpy as np
from torchvision import transforms
import torch
from torch import Tensor
import torch.nn as nn
from PIL import Image

# TODO: Training wastes 72% of time doing transforms while calling __getitem___,
# and less than 5% doing a forward pass. This obviously needs to be improved.
# Actually, using more DataLoader workers will solve this problem smartly, the
# MTT authors seem to have forgotten to do this for CIFAR, only did it for ImageNet


class FixedCIFAR10(CIFAR10):
    """
    CIFAR dataset with custom `__getitem__` method that allows for slice indexing
    """

    def __init__(
        self,
        root: str | Path,
        train: bool = True,
        transform: Callable[..., Any] | None = None,
        target_transform: Callable[..., Any] | None = None,
        download: bool = False,
        device=None,
    ) -> None:
        super().__init__(root, train, transform, target_transform, download)

        # if device:
        #     self.data.to(device)

    def __getitem__(self, key) -> List[Tuple[Tensor, int]] | Tuple[Tensor, int]:
        if isinstance(key, slice):
            return [self[i] for i in range(*key.indices(len(self)))]
        else:
            return super().__getitem__(key)


class FixedCIFAR100(CIFAR100):
    """
    CIFAR dataset with custom `__getitem__` method that allows for slice indexing
    """

    def __init__(
        self,
        root: str | Path,
        train: bool = True,
        transform: Callable[..., Any] | None = None,
        target_transform: Callable[..., Any] | None = None,
        download: bool = False,
        device=None,
    ) -> None:
        # Initialization does not need to be changed
        super().__init__(root, train, transform, target_transform, download)

        # if device:
        #     self.data = torch.tensor(self.data)
        #     self.data.to(device)

    def __getitem__(self, key) -> List[Tuple[Tensor, int]] | Tuple[Tensor, int]:
        if isinstance(key, slice):
            return [self[i] for i in range(*key.indices(len(self)))]
        else:
            return super().__getitem__(key)


def load_datasets(settings, hparams):
    """Dataset loader that prepares the dataset, as well as appending the hyperparameters dict
    to include relevant parameters that depend on the dataset (e.g. output channels)

    Args:
        settings (dict): User settings
        hparams (dict): Hyper parameters, this function will append this dict with additional parameters

    Returns:
        train_dataset, test_dataset, hparams
    """
    if settings["dataset"] == "CIFAR10":
        if settings["ZCA"]:
            raise NotImplementedError("ZCA not yet implemented")
        else:
            # Normalize CIFAR10 dataset (Note: not same as those used in MTT repo,
            # these were calculated directly from the PyTorch CIFAR10 dataset itself)
            mean = np.array([0.49191375, 0.48235852, 0.44673872])
            std = np.array([0.24706447, 0.24346213, 0.26147554])
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std),
                ]
            )

        print(f"Loading training dataset ({settings["dataset"]})")
        train_dataset = FixedCIFAR10(
            root="data/CIFAR100",
            train=True,
            download=True,
            transform=transform,
            device=hparams["device"],
        )

        print(f"Loading testing dataset ({settings["dataset"]})")
        test_dataset = FixedCIFAR10(
            root="data/CIFAR100",
            train=False,
            download=True,
            transform=transform,
            device=hparams["device"],
        )

        print("Appending hparams with additional parameters")
        hparams["img_shape"] = (3, 32, 32)
        hparams["out_channels"] = 10

    if settings["dataset"] == "CIFAR100":
        if settings["ZCA"]:
            raise NotImplementedError("ZCA not yet implemented")
        else:
            # Normalize CIFAR10 dataset (Note: taken from the MTT repository)
            mean = np.array([0.4914, 0.4822, 0.4465])
            std = np.array([0.2023, 0.1994, 0.2010])
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std),
                ]
            )

        print(f"Loading training dataset ({settings["dataset"]})")
        train_dataset = FixedCIFAR100(
            root="data/CIFAR10",
            train=True,
            download=True,
            transform=transform,
            device=settings["device"],
        )

        print(f"Loading testing dataset ({settings["dataset"]})")
        test_dataset = FixedCIFAR100(
            root="data/CIFAR10",
            train=False,
            download=True,
            transform=transform,
            device=settings["device"],
        )

        print("Appending hparams with additional parameters")
        hparams["img_shape"] = (3, 32, 32)
        hparams["out_channels"] = 100

    else:
        raise ValueError(
            f"Unknown dataset defined by user: {settings["dataset"]}"
        )

    return train_dataset, test_dataset, hparams


def dataset_info(dataset):
    print("Training set size: %i" % len(dataset))
    # print("Validation set size: %i" % len(val_dataset))
    print("Test set size: %i" % len(dataset))

    first_image, first_target = dataset[0]
    print(
        f"Image dimensions: {first_image.shape}"
    )  # C * W * H (C = 3 because RGB)
    print(f"Target type: {type(first_target)}")
    # print(f"First image tensor: \n {first_image}")
