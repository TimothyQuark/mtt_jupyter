from typing import Any, Callable, Iterator, Tuple, Dict, List, Optional, Union
from util import print_line, set_device
import torch
import torch.nn as nn
import numpy as np
import pprint
import matplotlib.pyplot as plt

from util import (
    print_line,
    init_logger,
    set_device,
    create_tqdm_bar,
    get_random_imgs,
    view_tensor_img,
)
from datasets import load_datasets, dataset_info

"""
This file is used to train the student models, which are used to train the synthetic dataset.
Hyper parameters can be set towards the end of the file.
"""


def main():
    pass


if __name__ == "__main__":
    project_name = "mtt-students"

    print(
        """Script for training student models and synthetic dataset of Dataset Distillation
        by Matching Training Trajectories."""
    )
    print_line()

    print("Checking for CUDA, or using defined device")
    device = set_device()
    print(f"device being used: {device}")
    print_line()

    # User settings:
    settings: dict[str, Any] = {
        "device": device,
        "num_workers": 16,  # CPU workers used by PyTorch, this will immensely speed up training (good estimate is number of CPU threads)
        "dataset": "CIFAR100",  # [CIFAR10, CIFAR100]
        "model": "ConvNet",  # [ConvNet, ResNet]
        "ZCA": False,  # [True, False]
        "debug": False,  # [True, False]
        "synthetic_init": "real",  # [real, noise] init synthetic dataset from real images, or noise
        "syn_class_imgs": 3,  # Number of syntnetic images per class
    }

    # Hyperparameters (note that some hparams are defined by the dataset used,
    # and are added by load_datasets)
    hparams: dict[str, Any] = {
        "images_per_class": 1,  # Number of synthetic images to generate per class
        "epochs": 1,  # Epochs per expert model
        "batch_size": 256,  # Batch size
        "learning_rate": 0.01,  # LR for optimizer
        "weight_decay": 0.0,  # Weight decay used by optimizer
        "momentum": 0.0,  # Momentum used by optimizer
        "net_depth": 3,  # Number of Conv layers (excluding helper layers and additional linear layer)
        "net_width": 128,  # Number of kernels in each Conv layer
        "kernel_size": 3,  # Dimensions of the kernels (H * W)
        "padding": 1,  # Conv layer padding pixels
        "activation": nn.ReLU(inplace=True),  # Activation function
        "norm": nn.GroupNorm(
            num_groups=128, num_channels=128, affine=True
        ),  # Normalization technique
        "pooling": nn.AvgPool2d(kernel_size=2, stride=2),  # Pooling method
        "optimizer": "SGD",  # Optimizer to use for training
        "training_steps": 5000,  # How many training steps are done
        "eval_steps": 100,  # How many training steps between evaluating the synthetic dataset
    }

    eval_pool = np.arange(
        0, hparams["training_steps"] + 1, hparams["eval_steps"]
    ).tolist()

    # Load dataset, and add relevant info the hparams
    # Note that the datasets here are not shuffled, same order when script is run
    train_dataset, test_dataset, hparams = load_datasets(settings, hparams)
    dataset_info(train_dataset)
    print_line()

    # Print settings after relevant info appended to hparams
    print("User settings:")
    pprint.pprint(settings)
    print("Hyperparameters:")
    pprint.pprint(hparams)
    print_line()

    # Create the synthetic dataset
    # TODO: Option to use noise init, instead of real images

    # Debug code to visualize data
    # test = train_dataset[0][0].permute(1, 2, 0).numpy()
    # plt.imshow(test)
    # plt.axis('off')  # Turn off axis
    # plt.show()

    # Count how many images of each class, and also their indices
    # used to grab a random image from each class to init synthetic dataset
    print("Counting number of images for each class")
    indices_class = [[] for c in range(hparams["out_channels"])]
    for idx, (img, label) in enumerate(train_dataset):
        indices_class[label].append(idx)

    for idx, c in enumerate(indices_class):
        print(f"Class {idx} ({hparams["class_names"][idx]}): {len(c)} images")

    print_line()

    # Init synthetic dataset with random images from every class
    # This is all concatenated into a big tensor
    print("Initializing synthetic dataset")
    syn_imgs = None
    for indices in indices_class:
        class_imgs = get_random_imgs(
            train_dataset, indices, settings["syn_class_imgs"]
        )
        if syn_imgs == None:
            syn_imgs = class_imgs
        else:
            syn_imgs = torch.cat((syn_imgs, class_imgs))

    # Explicitly make copy so this is not a view
    syn_imgs = syn_imgs.detach().clone()
    # Labels of each synthetic img, [0,0,0, 1,1,1 ... out_channels]
    label_syn = torch.tensor(
        [
            np.ones(settings["syn_class_imgs"], dtype=np.int_) * i
            for i in range(hparams["out_channels"])
        ],
        dtype=torch.long,
        requires_grad=False,
        device=settings["device"],
    ).view(-1)

    # Sanity check
    assert (
        syn_imgs.shape[0]
        == hparams["out_channels"] * settings["syn_class_imgs"]
    )

    print_line()

    # Training of synthetic dataset
