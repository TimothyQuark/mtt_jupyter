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
    """Helper Function that simply prints a straight line to the CLI. Optionally prints
    an empty line before the straight line

    Args:
        newline (bool, optional): Add empty line before straight line. Defaults to True.
    """

    if newline:
        print("\n")
    print("-" * 80)


def init_logger(log_dir) -> subprocess.Popen[bytes]:
    """Initialize TensorBoard logger

    Args:
        log_dir (Path): Path to where logger should read from

    Returns:
        subprocess.Popen[bytes]: Returns the logger process, so it can be terminated later by the user
    """

    tb_process = subprocess.Popen(["tensorboard", "--logdir", log_dir])
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
