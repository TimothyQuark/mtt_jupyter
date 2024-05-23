import sys
import os
import datetime
import time
from cProfile import Profile
from pstats import SortKey, Stats
from typing import Any, Callable, Tuple, Dict, List, Optional, Union
import pprint
from pathlib import Path
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchinfo import summary
from torch.profiler import profile, record_function, ProfilerActivity

from util import print_line, logger, set_device, create_tqdm_bar
from datasets import FixedCIFAR10, load_datasets, dataset_info
from networks import ConvNet


def train_model(
    model,
    train_loader,
    val_loader,
    loss_func,
    tb_logger,
    epochs=3,
    name="mtt-expert",
    debug=False,  # Used for debugging
):

    # In MTT paper they cut LR to 1/10th halfway through training
    scheduler = torch.optim.lr_scheduler.StepLR(
        model.optimizer, step_size=epochs * len(train_loader) // 2, gamma=0.1
    )

    if debug:
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            with_stack=True,
        ) as prof:
            inputs = torch.randn(256, 3, 32, 32).cuda()
            model(inputs)

        print(
            prof.key_averages().table(sort_by="cuda_time_total", row_limit=10)
        )
    else:
        for epoch in range(epochs):

            # Training
            training_loop = create_tqdm_bar(
                train_loader, desc=f"Training Epoch [{epoch + 1}/{epochs}]"
            )
            training_loss = 0

            for train_iter, batch in training_loop:
                loss = model.training_step(batch, loss_func)
                training_loss += loss.item()
                scheduler.step()

                # Update progress bar
                training_loop.set_postfix(
                    train_loss="{:.8f}".format(
                        training_loss / (train_iter + 1)
                    ),
                    lr="{:.8f}".format(model.optimizer.param_groups[0]["lr"]),
                )
                tb_logger.add_scalar(
                    f"{name}/train_loss",
                    loss.item(),
                    epoch * len(train_loader) + train_iter,
                )


def main():
    project_name = "mtt-expert"

    print(
        """Script for generating expert trajectories for Dataset Distillation
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
        "debug": False,  # [True, False]
    }

    # Hyperparameters (note that some hparams are defined by the dataset used,
    # and are added by load_datasets)
    hparams: dict[str, Any] = {
        "device": device,
        "num_workers": 16,
        "epochs": 2,
        "batch_size": 256,
        "max_patience": 3,
        "learning_rate": 0.01,
        "weight_decay": 0.0,
        "momentum": 0.0,
        "net_depth": 3,  # Number of Conv layers (excluding helper layers and additional linear layer)
        "net_width": 128,  # Number of kernels in each Conv layer
        "kernel_size": 3,  # Dimensions of the kernels (H * W)
        "padding": 1,  # Conv layer padding
        "activation": nn.ReLU(inplace=True),  # Activation function
        "norm": nn.GroupNorm(
            num_groups=128, num_channels=128, affine=True
        ),  # Normalization technique
        "pooling": nn.AvgPool2d(kernel_size=2, stride=2),  # Pooling method
        "optimizer": "SGD",  # Optimizer to use for training
    }

    train_dataset, test_dataset, hparams = load_datasets(settings, hparams)
    dataset_info(train_dataset)
    print_line()

    print("User settings:")
    pprint.pprint(settings)
    print("Hyperparameters:")
    pprint.pprint(hparams)
    print_line()

    model = ConvNet(hparams=hparams)
    print("Summary of model (for provided hyperparameters)")
    # Depth needs to be set higher than default to open up Sequential module
    summary(
        model,
        input_size=(
            model.hp["batch_size"],
            model.hp["img_shape"][0],
            model.hp["img_shape"][1],
            model.hp["img_shape"][2],
        ),
        depth=4,
    )
    print_line()

    # It is best to initialize TensorBoard as late as possible, because it interferes with
    # other processes trying to print to the CLI
    print("Starting up TensorBoard")
    tb_process = logger(
        f"logs/{project_name}/"
    )
    print(
        "Note that Tensorboard will continue running even when the script has ended, this does not mean the script is hanging"
    )
    print_line()

    # Create new model and move to device
    model = ConvNet(hparams=hparams).to(device)

    # Create the tb_logger, project_name needed for TensorBoard to find the logs and display them
    path = os.path.join("logs", project_name)
    num_of_runs = len(os.listdir(path)) if os.path.exists(path) else 0
    path = os.path.join(path, f"run_{num_of_runs + 1}")
    tb_logger = SummaryWriter(path)

    # Train the model
    train_loader = DataLoader(
        train_dataset, batch_size=hparams["batch_size"], shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=128, shuffle=False
    )  # Note that batch size is fixed for test dataset in MTT repo

    epochs = hparams.get("epochs", 5)
    loss_func = torch.nn.CrossEntropyLoss().to(model.device)

    if settings["debug"]:
        print("Running the model in debug mode")
        with Profile() as profile:
            train_model(
                model,
                train_loader,
                None,
                loss_func,
                tb_logger,
                epochs=epochs,
                name="mtt-expert",
                debug=False,
            )
            Stats(profile).strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats()
    else:
        train_model(
            model,
            train_loader,
            None,
            loss_func,
            tb_logger,
            epochs=epochs,
            name="mtt-expert",
        )

    # Keep the script running to keep TensorBoard alive
    print(
        "Script finished, CLI is only running Tensorboard (script can be safely terminated now!)"
    )
    try:
        while True:
            pass
    except KeyboardInterrupt:
        # Terminate TensorBoard process on script interruption
        tb_process.terminate()


if __name__ == "__main__":
    main()
