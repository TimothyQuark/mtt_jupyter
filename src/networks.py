import sys
import os
import copy
import datetime
from typing import Any, Callable, Tuple, Dict, List, Optional, Union
import pprint
from pathlib import Path
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
from torchvision import transforms

class ConvSandwich(nn.Module):
    """Helper class to create a convolutional layer and related layers"""

    def __init__(self, hparams, in_channels=None, out_channels=None):
        """
        Args:
            hparams (dict[str, Any]): hyperparameters dictionary
            in_channels (int, optional): Manually override the number of input channels of the Conv layer. If left uninitialized, will use hparams["img_shape"][0]
            as default (i.e. input_channel number for the first Conv layer)
            out_channels (int, optional): Manually override the number of output channels of the Conv layer,
            used for deeper layers where input_channels shrink due to pooling. If left uninitialized, will use hparams["net_width"][0]
            as default.
        """
        super().__init__()

        self.sandwich = nn.Sequential(
            nn.Conv2d(
                in_channels=hparams["img_shape"][0] if not in_channels else in_channels,
                out_channels=hparams["net_width"],
                kernel_size=hparams["kernel_size"],
                padding=hparams["padding"],
            ),
            copy.deepcopy(hparams["norm"]),
            copy.deepcopy(hparams["activation"]),
            copy.deepcopy(hparams["pooling"]),
        )

    def forward(self, x):
        x = self.sandwich(x)
        return x


class PrintLayer(nn.Module):
    """
    Helper Class for debugging behaviour of model. Will print out the shape of the layer input,
    and input passed directly to output
    """

    def forward(self, x):
        print(x.shape)
        # print("MODEL DATA ON: ", x.get_device(), "MODEL PARAMS ON: ", self.classifier.weight.data.get_device())
        return x


def weights_init(m: nn.Module):
    """Initialize weights for each layer of model m, based on the type of activation function

    Args:
        m (nn.Module): PyTorch model to initialize weights of
    """
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_uniform_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")

class ConvNet(nn.Module):
    def __init__(self, hparams) -> None:
        super().__init__()

        self.hp = hparams
        kernel_size = self.hp[
            "kernel_size"
        ]  # Dimensions of each kernel, used to calculate input shapes of layers

        self.features = nn.Sequential(
            ConvSandwich(self.hp),
            ConvSandwich(self.hp, self.hp["net_width"]),
            ConvSandwich(self.hp, self.hp["net_width"]),
        )

        # self.features = ConvSandwich(self.hp)

        self.classifier = nn.Linear(
            (self.hp["net_width"])
            * (self.hp["img_shape"][1] // 8)
            * (
                self.hp["img_shape"][2] // 8
            ),  # num. of kernels x dim. of last layer (after 3 pooling layers)
            self.hp["out_channels"],
        )

        # We want the model to be stored on GPU if possible
        self.device = hparams["device"]

        self.set_optimizer()

        # Apply initial weights
        # self.encoder.apply(weights_init)
        # self.decoder.apply(weights_init)
        # self.classifier.apply(weights_init)

    def forward(self, x):
        x = self.features(x)
        x = x.view(
            x.size(0), -1
        )  # Reshape feature net to [batch_size, H*W/ (2 ** pooling layers)]
        x = self.classifier(x)

        return x

    def set_optimizer(self):
        self.optimizer = None

        if self.hp["optimizer"] == "SGD":
            self.optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.hp["learning_rate"],
                momentum=self.hp["momentum"],
                weight_decay=self.hp["weight_decay"],
            )
        else:
            raise ValueError(f"Unexpected Optimizer chosen: {self.hp["optimizer"]}")

    def training_step(self, batch, loss_func):
        self.train()  # Set model to training mode
        self.optimizer.zero_grad()  # Reset gradient every batch

        # N = batch size, C = channels (3 for RGB), H = image height, W = image width
        images = batch[0].to(self.device)  # Input batch, N x C x H x W
        target = batch[1].to(self.device)

        # Model makes prediction (forward pass)
        pred = self.forward(images)  # N x C x H x W (C = num of labels of dataset)

        # Calculate loss, do backward pass to update weights, optimizer takes step
        # torch.nn.CrossEntropyLoss(ignore_index=0, reduction="mean") wants target to be of type long, not float
        loss = loss_func(pred, target)
        loss.backward()
        self.optimizer.step()

        return loss

    # def validation_step(self, batch, loss_func):
    #     loss = 0

    #     # Set model to eval
    #     self.eval()

    #     with torch.no_grad():
    #         images = batch[0].to(self.device)  # Input batch, N x C x H x W
    #         target = (
    #             batch[1][0].to(self.device).squeeze()
    #         )  # Ground truth, each pixel assigned an ID int. N x H x W

    #         pred = self.forward(images)
    #         loss = loss_func(pred, target)

    #     return loss