from typing import Any, Callable, Iterator, Tuple, Dict, List, Optional, Union
from util import print_line, set_device

"""
This file is used to train the student models, which are used to train the synthetic dataset.
Hyper parameters can be set towards the end of the file
"""

def main():
    pass

if __name__ == "__main__":
    project_name = "mtt-students"

    print(
        """Script for training student models and synthetic dataset of Dataset Distillation
        by Matching Training Trajectories. This script does the same as the Jupyter
        Notebook, but is faster and can be run via CLI."""
    )
    print_line()

    print("Checking for CUDA, or using defined device")
    device = set_device()
    print(f"device being used: {device}")
    print_line()

    # User settings:
    settings: dict[str, Any] = {
        "dataset": "CIFAR100",  # [CIFAR10, CIFAR100]
        "model": "ConvNet",  # [ConvNet, ResNet]
        "ZCA": False,  # [True, False]
        "expert_folder": "run_100",  # Name of folder to take expert models from
        "debug": False,  # [True, False]
    }

    # Hyperparameters (note that some hparams are defined by the dataset used,
    # and are added by load_datasets)
    hparams: dict[str, Any] = {
        "device": device,
        "num_workers": 16,  # CPU workers used by PyTorch
        "epochs": 1,  # Epochs per expert model
        "batch_size": 256,  # Batch size
        "images_per_class" : 1, # Number of synthetic images to generate per class
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
    }

