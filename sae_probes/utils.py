"""Utility functions for SAE probing."""

import json
import random
from dataclasses import asdict, is_dataclass
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


def save_config(config: object, path: Path) -> None:
    """
    Save configuration to JSON file.

    Args:
        config: Configuration object or dictionary
        path: Path to save file
    """
    # Create parent directory if it doesn't exist
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert dataclass to dict if needed
    if is_dataclass(config):
        config_dict = asdict(config)  # type: ignore
    else:
        config_dict = config

    # Save as JSON
    with open(path, "w") as f:
        json.dump(config_dict, f, indent=2)


def load_config(path: Path) -> dict:
    """
    Load configuration from JSON file.

    Args:
        path: Path to load file from

    Returns:
        Configuration dictionary
    """
    with open(path) as f:
        return json.load(f)


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


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert PyTorch tensor to NumPy array.

    Args:
        tensor: PyTorch tensor

    Returns:
        NumPy array
    """
    if isinstance(tensor, torch.Tensor):
        return tensor.cpu().numpy()
    return tensor


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


def print_gpu_memory_usage() -> None:
    """Print GPU memory usage if CUDA is available."""
    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    for i in range(torch.cuda.device_count()):
        print(f"GPU {i} - {torch.cuda.get_device_name(i)}:")
        print(
            f"  Total memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB"
        )
        print(f"  Allocated memory: {torch.cuda.memory_allocated(i) / 1e9:.2f} GB")
        print(f"  Cached memory: {torch.cuda.memory_reserved(i) / 1e9:.2f} GB")
        print(
            f"  Free memory: {torch.cuda.get_device_properties(i).total_memory / 1e9 - torch.cuda.memory_allocated(i) / 1e9:.2f} GB"
        )


def sparse_to_dense(tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert sparse tensor to dense if needed.

    Args:
        tensor: PyTorch tensor (sparse or dense)

    Returns:
        Dense PyTorch tensor
    """
    if tensor.is_sparse:
        return tensor.to_dense()
    return tensor
