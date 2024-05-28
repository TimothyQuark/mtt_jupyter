import sys
import os
import datetime
import time
from cProfile import Profile
from pstats import SortKey, Stats
from typing import Any, Callable, Iterator, Tuple, Dict, List, Optional, Union
import pprint
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchinfo import summary
from torch.profiler import profile, record_function, ProfilerActivity

from util import print_line, init_logger, set_device, create_tqdm_bar
from datasets import load_datasets, dataset_info
from networks import ConvNet


"""
This file is used to train the expert models that will be used for MTT. Hyper parameters can
be set towards the end of the file.
"""


def train_expert_model(
    hparams,
    settings,
    train_loader,
    test_loader,
    log_path,
    name,
    pt_profiler=False,  # Used for debugging, PyTorch Profiler
) -> List[Iterator[nn.Parameter]]:

    # Place expert logs in subdirectory of the current run, create new summary writer
    current_expert = (
        len(os.listdir(log_path)) if os.path.exists(log_path) else 0
    ) + 1
    log_path = os.path.join(log_path, f"expert_{current_expert}")
    tb_logger = SummaryWriter(log_path)
    print(f"Training Expert {current_expert}")

    # Create new model and move to device
    model = ConvNet(hparams=hparams, settings=settings).to(settings["device"])
    loss_func = torch.nn.CrossEntropyLoss().to(model.device)
    epochs = hparams["epochs"]

    # In MTT paper they cut LR to 1/10th halfway through training
    scheduler = torch.optim.lr_scheduler.StepLR(
        model.optimizer, step_size=epochs * len(train_loader) // 2, gamma=0.1
    )

    # Profiling for debugging performance
    if pt_profiler:
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            with_stack=True,
        ) as prof:
            inputs = torch.randn(
                model.hp["batch_size"],
                model.hp["img_shape"][0],
                model.hp["img_shape"][1],
                model.hp["img_shape"][2],
            ).cuda()
            model(inputs)

        print(
            prof.key_averages().table(sort_by="cuda_time_total", row_limit=10)
        )
    else:

        # Snapshots of model parameters, that will be used by _____
        timestamps = []

        # Take snapshot before training begins, unsure why MTT does this
        timestamps.append([p.detach().cpu() for p in model.parameters()])

        for epoch in range(epochs):

            # Training
            # Running total loss and accuracy for an epoch (will be averaged out)
            train_loss = 0
            train_acc = 0

            # Training iterator (note epoch here has 1 added to it for user clarity)
            training_loop = create_tqdm_bar(
                train_loader, desc=f"Train Epoch [{epoch + 1}/{epochs}]"
            )

            for train_iter, batch in training_loop:
                loss, acc = model.step(batch, loss_func, "train")
                train_loss += loss.item()
                train_acc += acc
                # Running average of loss and accuracy during epoch (+1 so no zero division)
                avg_loss = train_loss / (train_iter + 1)
                avg_acc = train_acc / (train_iter + 1)

                scheduler.step()

                # Update progress bar
                training_loop.set_postfix(
                    acc="{:.8f}".format(avg_acc),
                    loss="{:.8f}".format(avg_loss),
                    lr="{:.8f}".format(model.optimizer.param_groups[0]["lr"]),
                )
                tb_logger.add_scalar(
                    f"{name}/batch_loss",
                    loss.item(),
                    epoch * len(train_loader) + train_iter,
                )
                tb_logger.add_scalar(
                    f"{name}/batch_accuracy",
                    acc,
                    epoch * len(train_loader) + train_iter,
                )

            # Testing
            # Running total loss and accuracy for an epoch (will be averaged out)
            test_loss = 0
            test_acc = 0

            # Testing iterator (note epoch here has 1 added to it for user clarity)
            testing_loop = create_tqdm_bar(
                train_loader, desc=f"Test Epoch [{epoch + 1}/{epochs}]"
            )

            for train_iter, batch in testing_loop:
                loss, acc = model.step(batch, loss_func, "test")
                test_loss += loss.item()
                test_acc += acc
                # Running average of loss and accuracy during epoch (+1 so no zero division)
                avg_loss = test_loss / (train_iter + 1)
                avg_acc = test_acc / (train_iter + 1)

                # Update progress bar
                testing_loop.set_postfix(
                    acc="{:.8f}".format(avg_acc),
                    loss="{:.8f}".format(avg_loss),
                    lr="{:.8f}".format(model.optimizer.param_groups[0]["lr"]),
                )
                tb_logger.add_scalar(
                    f"{name}/batch_loss",
                    loss.item(),
                    epoch * len(train_loader) + train_iter,
                )
                tb_logger.add_scalar(
                    f"{name}/batch_accuracy",
                    acc,
                    epoch * len(train_loader) + train_iter,
                )

            # At end of every epoch, take snapshot of model parameters
            timestamps.append([p.detach().cpu() for p in model.parameters()])

        # Only return timestamps if not in debug mode
        return timestamps

    # For the first expert, save the model graph to visualize
    if current_expert == 1:
        # Make a graph of the model in TensorBoard
        model.eval()
        tb_logger.add_graph(
            model,
            torch.randn(
                model.hp["batch_size"],
                model.hp["img_shape"][0],
                model.hp["img_shape"][1],
                model.hp["img_shape"][2],
            ).cuda(),
        )


def train_many_experts(
    hparams, settings, train_loader, test_loader, project_name
):

    # TODO: Check that expert data and log files are same length, else can conflict

    # Create the tb_logger, current_run is based on what is inside the log files
    log_path = os.path.join("logs", project_name)
    current_run = (
        len(os.listdir(log_path)) if os.path.exists(log_path) else 0
    ) + 1
    log_path = os.path.join(log_path, f"run_{current_run}")
    print(f"Executing run {current_run}")

    save_intervals = hparams["save_interval"]

    # A list of lists of model parameters
    expert_trajectories: List[List[Iterator[nn.Parameter]]] = []

    for it in range(hparams["experts"]):
        timestamps = train_expert_model(
            hparams=hparams,
            settings=settings,
            train_loader=train_loader,
            test_loader=test_loader,
            log_path=log_path,
            name="mtt-expert",
        )

        expert_trajectories.append(timestamps)

        if not ((it + 1) % save_intervals):
            expert_path = os.path.join("data", "experts", project_name)
            expert_path = os.path.join(expert_path, f"run_{current_run}")
            if not os.path.exists(expert_path):
                os.makedirs(expert_path)
            current_buffer = (
                len(os.listdir(expert_path))
                if os.path.exists(expert_path)
                else 0
            ) + 1
            print(f"Saving replay_buffer_{current_buffer}")
            torch.save(
                expert_trajectories,
                os.path.join(expert_path, f"replay_buffer_{current_buffer}.pt"),
            )
            # Reset expert trajectories list
            expert_trajectories = []


def save_experts():
    pass


def main():
    project_name = "mtt-experts"

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
        "device": device,
        "num_workers": 16,  # CPU workers used by PyTorch, this will immensely speed up training (good estimate is number of CPU threads)
        "dataset": "CIFAR100",  # [CIFAR10, CIFAR100]
        "model": "ConvNet",  # [ConvNet, ResNet]
        "ZCA": False,  # [True, False]
        "debug": False,  # [True, False]
    }

    # Hyperparameters (note that some hparams are defined by the dataset used,
    # and are added by load_datasets)
    hparams: dict[str, Any] = {
        "save_interval": 10,  # How many expert trajectories are saved in each file
        "experts": 10,  # How many expert models to train
        "epochs": 50,  # Epochs per expert model
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
    }

    train_dataset, test_dataset, hparams = load_datasets(settings, hparams)
    dataset_info(train_dataset)
    print_line()

    print("User settings:")
    pprint.pprint(settings)
    print("Hyperparameters:")
    pprint.pprint(hparams)
    print_line()

    model = ConvNet(hparams=hparams, settings=settings)
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
    tb_process = init_logger(f"logs/{project_name}/")
    print(
        "Note that Tensorboard will continue running even when the script has ended, this does not mean the script is hanging"
    )
    print(
        "Also note that if you have TensorFlow installed but not the CUDA Toolkit, you may get warnings that CUDA is not enabled, this does not apply to PyTorch!"
    )
    print_line()

    # Training begins before TensorBoard process is loaded, buffered text prints
    # where it isn't supposed to. This sleep stops that from happening, hacky way to do it.
    time.sleep(2)

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=hparams["batch_size"],
        shuffle=True,
        num_workers=settings["num_workers"],
    )
    # Note that batch size is fixed for test dataset in MTT repo
    test_loader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=settings["num_workers"],
    )

    if settings["debug"]:
        print("Running the model in debug mode")
        with Profile() as profile:
            # Only profile a single model run
            train_expert_model(
                hparams=hparams,
                settings=settings,
                train_loader=train_loader,
                test_loader=train_loader,
                log_path="logs/debug",
                name=project_name,
                pt_profiler=False,  # Additional flag to show PyTorch profiler
            )
            Stats(profile).strip_dirs().sort_stats(
                SortKey.CUMULATIVE
            ).print_stats()
    else:
        train_many_experts(
            hparams=hparams,
            settings=settings,
            train_loader=train_loader,
            test_loader=test_loader,
            project_name=project_name,
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
