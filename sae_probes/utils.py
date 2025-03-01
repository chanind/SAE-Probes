"""Utility functions for SAE probing."""

import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_path(path: Path) -> Path:
    """
    Ensure path exists by creating directories.

    Args:
        path: Path to ensure

    Returns:
        Same path
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_device(device_str: str | None = None) -> torch.device:
    """
    Get PyTorch device.

    Args:
        device_str: Device string (if None, uses CUDA if available, else CPU)

    Returns:
        PyTorch device
    """
    if device_str is not None:
        return torch.device(device_str)

    if torch.cuda.is_available():
        return torch.device("cuda:0")
    else:
        return torch.device("cpu")
