import subprocess
from typing import Any, Callable, Tuple, Dict, List, Optional, Union
import torch
from torch import Tensor
from tqdm import tqdm
import matplotlib.pyplot as plt


def visualize_imgs(images: Tensor | List[Tensor]) -> None:
    """Function to visualize images. Note that because we are usually
      normalizing images outside the 0-1 range, the colors may be off
      and you may get clipping warnings

    Args:
        images (Tensor | List[Tensor]): images to visualize
    """
    if isinstance(images, Tensor):
        images = [images]

    n = len(images)
    plt.figure(figsize=(20, 3 * n))

    for idx, (image, targets) in enumerate(images):
        # Normalized input image
        plt.subplot(n, 1, idx * 1 + 1)
        plt.imshow(image.numpy().transpose(1, 2, 0))
        plt.axis("off")
        plt.title("Input image")


def print_line(newline=True):
    if newline:
        print("\n")
    print("-" * 80)


def logger(log_dir) -> subprocess.Popen[bytes]:
    tb_process = subprocess.Popen(
        ["tensorboard", "--logdir", log_dir]
    )
    print(
        f"TensorBoard started at http://localhost:6006/ (or the port specified in the terminal)"
    ),

    return tb_process


def set_device(device=None):
    """Set device to use for training model. If no device given, will check for CUDA,
    else default to CPU

    Args:
        device (_type_, optional): Explicitly set a device Defaults to None.

    Returns:
        _type_: _description_
    """
    if device == None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return device


def create_tqdm_bar(iterable, desc):
    return tqdm(enumerate(iterable), total=len(iterable), ncols=150, desc=desc)
