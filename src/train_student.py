from typing import Any, Callable, Iterator, Tuple, Dict, List, Optional, Union
import torch
import torch.nn as nn
import numpy as np
import pprint
from tqdm import tqdm
import os

from util import (
    print_line,
    init_logger,
    set_device,
    create_tqdm_bar,
    get_random_imgs,
    view_tensor_img,
    get_time,
)
from datasets import load_datasets, dataset_info
from networks import ConvNet
from reparam_module import ReparamModule

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
        "expert_dir": os.path.join("data", "experts", "mtt-experts", "run_2"),
    }

    # Hyperparameters (note that some hparams are defined by the dataset used,
    # and are added by load_datasets)
    hparams: dict[str, Any] = {
        "images_per_class": 1,  # Number of synthetic images to generate per class
        "epochs": 1,  # Epochs per expert model
        "batch_size": 256,  # Batch size
        "learning_rate": 0.0,  # LR used to train the original model, NOT USED
        "lr_init": 0.01,  # Initial synthetic training data LR (trainable parameter)
        "lr_lr": 1e-05,  # LR for the synthetic training LR
        "lr_img": 1000,  # LR for the synthetic training data
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
        "training_steps": 5000,  # How many training steps are done (note syn_steps is done for every training step)
        "eval_steps": 100,  # How many training steps between evaluating the synthetic dataset
        "max_start_epoch": 25,  # Latest epoch to init starting expert trajectory
        "expert_epochs": 3,  # How many epochs target expert params is after starting expert params
        "syn_steps": 8,  # How many steps the synthetic data is trained every iteration
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
    for idx, (img, label) in tqdm(enumerate(train_dataset)):
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

    # Labels of each synthetic img, [0,0,0, 1,1,1 ... out_channels]
    label_syn = None
    for c in range(hparams["out_channels"]):
        # labels = torch.full((settings["syn_class_imgs"]), c)
        labels = torch.ones(settings["syn_class_imgs"], dtype=torch.long) * c
        if label_syn == None:
            label_syn = labels
        else:
            label_syn = torch.cat((label_syn, labels))

    # Cross Entropy Loss, to calculate loss of base model
    criterion = nn.CrossEntropyLoss().to(settings["device"])

    # Explicitly make copy so this is not a view
    syn_imgs = syn_imgs.detach().clone()
    label_syn = label_syn.detach().clone()

    # Sanity check
    assert (
        syn_imgs.shape[0]
        == hparams["out_channels"] * settings["syn_class_imgs"]
    )
    assert label_syn.shape[0] == syn_imgs.shape[0]

    print_line()

    # Training of synthetic dataset
    syn_imgs = (
        syn_imgs.detach().to(settings["device"]).requires_grad_(True)
    )  # Synthetic dataset
    syn_lr = (
        torch.tensor(hparams["lr_init"])
        .detach()
        .to(settings["device"])
        .requires_grad_(True)
    )  # Synthetic LR
    optimizer_img = torch.optim.SGD(
        [syn_imgs], lr=hparams["lr_img"], momentum=0.5
    )
    optimizer_lr = torch.optim.SGD([syn_lr], lr=hparams["lr_lr"], momentum=0.5)
    optimizer_img.zero_grad()  # TODO: not needed I think

    print("Training Synthetic dataset")
    print(f"Start time: {get_time()}")

    expert_dir = settings["expert_dir"]
    print(f"Expert Dir: {expert_dir}")

    # Load all data into memory
    buffer = []
    n = 1
    while os.path.exists(
        os.path.join(expert_dir, "replay_buffer_{}.pt".format(n))
    ):
        buffer = buffer + torch.load(
            os.path.join(expert_dir, "replay_buffer_{}.pt".format(n))
        )
        n += 1
    if n == 1:
        raise AssertionError("No buffers detected at {}".format(expert_dir))

    best_acc = 0
    best_std = 0

    for it in range(hparams["training_steps"]):

        save_iteration = False  # Flag to determine if iteration should be saved

        if it in eval_pool:
            # TODO: evaluate
            pass

        if it in eval_pool and (save_iteration or it % 1000 == 0):
            # TODO: evaluate and save
            pass

        print(f"Synthetic LR: {syn_lr}")

        # This will get the unmodified ConvNet. This is a random model with random parameter inits, it does not have anything to do with the expert networks yet.
        student_net = ConvNet(hparams=hparams, settings=settings).to(
            settings["device"]
        )

        # This will do the MTT black magic that heavily confuses me
        student_net = ReparamModule(student_net)

        # Set to training mode, where gradients and computational graphs are saved
        student_net.train()

        # This is only layers with learnable parameters, does not include ReLU
        num_params = sum(
            [np.prod(p.size()) for p in (student_net.parameters())]
        )

        # Grab random buffer, which contains many expert trajectories
        expert_trajectory = buffer[np.random.randint(0, len(buffer))]

        start_epoch = np.random.randint(0, hparams["max_start_epoch"])
        starting_params = expert_trajectory[
            start_epoch
        ]  # This is a single model

        target_params = expert_trajectory[
            start_epoch + hparams["expert_epochs"]
        ]
        target_params = torch.cat(
            [p.data.to(settings["device"]).reshape(-1) for p in target_params],
            0,
        )

        # Note this is not a copy but a view of the parameters of the student model
        # These are the parameters we train for a few epochs
        student_params = [
            torch.cat(
                [
                    p.data.to(settings["device"]).reshape(-1)
                    for p in starting_params
                ],
                0,
            ).requires_grad_(True)
        ]
        # These are the starting parameters, they should not change
        starting_params = torch.cat(
            [
                p.data.to(settings["device"]).reshape(-1)
                for p in starting_params
            ],
            0,
        )

        # Ground truth labels
        y_hat = label_syn.to(settings["device"])

        param_loss_list = []
        param_dist_list = []
        indices_chunks = []

        # This is the training of the synthetic data, it calculated parameters theta via SGD, but constructs a computational graph to make it function of x
        for step in range(hparams["syn_steps"]):

            # TODO: Fix this dumb statement
            if not indices_chunks:
                indices = torch.randperm(len(syn_imgs))
                # Split into chunks, but in this case we have one chunk with all data in it
                indices_chunks = list(
                    torch.split(
                        indices,
                        hparams["out_channels"] * settings["syn_class_imgs"],
                    )
                )

            current_indices = indices_chunks.pop()

            x = syn_imgs[current_indices]
            this_y = y_hat[current_indices]

            forward_params = student_params[-1]

            x = student_net(x, flat_param=forward_params)
            ce_loss = criterion(x, this_y)

            # Only works because of ReparamModule
            # TODO: I think this creates computational graph of all student_params, so 20 * 192 = 3840 layers wtf
            # This will cause us to run out of memory
            grad = torch.autograd.grad(
                ce_loss, student_params[-1], create_graph=True
            )[0]

            # Each student param iteration is a function of the previous synthetic training step
            # This results in very huge networks
            student_params.append(student_params[-1] - syn_lr * grad)

        param_loss = torch.tensor(0.0).to(settings["device"])
        param_dist = torch.tensor(0.0).to(settings["device"])

        param_loss += torch.nn.functional.mse_loss(
            student_params[-1], target_params, reduction="sum"
        )
        param_dist += torch.nn.functional.mse_loss(
            starting_params, target_params, reduction="sum"
        )

        param_loss_list.append(param_loss)
        param_dist_list.append(param_dist)

        # TODO: Fix this dumb division, doesn't do anything
        param_loss /= num_params
        param_dist /= num_params

        param_loss /= param_dist

        grand_loss = param_loss

        optimizer_img.zero_grad()
        optimizer_lr.zero_grad()

        grand_loss.backward()

        optimizer_img.step()
        optimizer_lr.step()

        print(grand_loss)
        # print(syn_imgs.grad)

        for _ in student_params:
            del _

    print("Done with script")
