"""Dataset loading and processing utilities for SAE probing tasks."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


@dataclass
class DatasetInfo:
    """Information about a binary classification dataset."""

    tag: str
    size: int
    description: str = ""


def get_binary_datasets(dataset_path: Path) -> list[DatasetInfo]:
    """
    Get information about available binary classification datasets.

    Args:
        dataset_path: Path to the directory containing dataset files

    Returns:
        List of DatasetInfo objects for binary classification datasets
    """
    # Read master CSV file
    master_df = pd.read_csv(dataset_path / "probing_datasets_MASTER.csv")
    binary_datasets = master_df[master_df["Data type"] == "Binary Classification"]

    datasets = []
    for _, row in binary_datasets.iterrows():
        tag = row["Dataset save name"].split("/")[-1].split(".")[0]

        # Try to get dataset size
        try:
            df = pd.read_csv(dataset_path / row["Dataset save name"])
            size = len(df)
        except Exception:
            size = 0

        datasets.append(
            DatasetInfo(tag=tag, size=size, description=row.get("Description", ""))
        )

    return datasets


def load_dataset(
    dataset_tag: str,
    dataset_path: Path,
    num_train: int = 1024,
    test_size: int | None = None,
    pos_ratio: float = 0.5,
    seed: int = 42,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Load a dataset and prepare train/test splits.

    Args:
        dataset_tag: Tag identifying the dataset
        dataset_path: Path to the directory containing dataset files
        num_train: Number of training examples
        test_size: Number of test examples (if None, uses all remaining)
        pos_ratio: Ratio of positive examples in train set
        seed: Random seed for reproducibility

    Returns:
        Tuple containing (dataset DataFrame, train indices, test indices)
    """
    # Extract dataset tag from numbered format (e.g. "1_dataset" -> "dataset")
    dataset_base_tag = "_".join(dataset_tag.split("_")[1:])

    # Get master dataset info
    master_df = pd.read_csv(dataset_path / "probing_datasets_MASTER.csv")
    dataset_row = master_df[master_df["Dataset Tag"] == dataset_base_tag]

    if dataset_row.empty:
        raise ValueError(f"Dataset with tag {dataset_base_tag} not found")

    # Get dataset path
    dataset_file = dataset_row["Dataset save name"].iloc[0]
    df = pd.read_csv(dataset_path / dataset_file)

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(df["target"].values)

    # Get train/test indices
    train_indices, test_indices = get_train_test_indices(
        y, num_train, test_size, pos_ratio, seed
    )

    return df, train_indices, test_indices


def get_train_test_indices(
    y: np.ndarray,
    num_train: int,
    num_test: int | None = None,
    pos_ratio: float = 0.5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get indices for balanced train/test splits.

    Args:
        y: Target labels (0/1)
        num_train: Number of training examples
        num_test: Number of test examples (if None, uses all remaining)
        pos_ratio: Ratio of positive examples in train set
        seed: Random seed for reproducibility

    Returns:
        Tuple containing (train indices, test indices)
    """
    # Set random seed for reproducibility
    np.random.seed(seed)

    # Split positive and negative samples
    pos_indices = np.where(y == 1)[0]
    neg_indices = np.where(y == 0)[0]

    # Calculate sizes ensuring they sum to num_train
    pos_train_size = int(np.ceil(pos_ratio * num_train))
    neg_train_size = num_train - pos_train_size

    # Handle case where requested sizes exceed available data
    if pos_train_size > len(pos_indices) or neg_train_size > len(neg_indices):
        raise ValueError(
            f"Requested train split requires {pos_train_size} positive and {neg_train_size} "
            f"negative examples, but only have {len(pos_indices)} positive and "
            f"{len(neg_indices)} negative examples available"
        )

    # Sample train indices
    train_pos = np.random.choice(pos_indices, size=pos_train_size, replace=False)
    train_neg = np.random.choice(neg_indices, size=neg_train_size, replace=False)

    # Get remaining indices for test set
    remaining_pos = np.setdiff1d(pos_indices, train_pos)
    remaining_neg = np.setdiff1d(neg_indices, train_neg)

    # Calculate test sizes
    if num_test is None:
        # Use all remaining examples
        test_pos = remaining_pos
        test_neg = remaining_neg
    else:
        # Use specified number of test examples with same pos_ratio
        pos_test_size = int(np.ceil(pos_ratio * num_test))
        neg_test_size = num_test - pos_test_size

        if pos_test_size > len(remaining_pos) or neg_test_size > len(remaining_neg):
            raise ValueError(
                f"Requested test split requires {pos_test_size} positive and {neg_test_size} "
                f"negative examples, but only have {len(remaining_pos)} positive and "
                f"{len(remaining_neg)} negative examples available after train split"
            )

        test_pos = np.random.choice(remaining_pos, size=pos_test_size, replace=False)
        test_neg = np.random.choice(remaining_neg, size=neg_test_size, replace=False)

    # Combine and shuffle indices
    train_indices = np.random.permutation(np.concatenate([train_pos, train_neg]))
    test_indices = np.random.permutation(np.concatenate([test_pos, test_neg]))

    return train_indices, test_indices


def corrupt_labels(y: np.ndarray, noise_fraction: float) -> np.ndarray:
    """
    Corrupt a fraction of labels by flipping them.

    Args:
        y: Target labels (0/1)
        noise_fraction: Fraction of labels to flip (0.0 to 0.5)

    Returns:
        Corrupted labels
    """
    assert 0 <= noise_fraction <= 0.5, "Noise fraction must be between 0 and 0.5"

    np.random.seed(42)
    # Get indices to flip
    num_to_flip = int(len(y) * noise_fraction)
    flip_indices = np.random.choice(len(y), size=num_to_flip, replace=False)

    # Create copy and flip selected labels
    y_corrupted = y.copy()
    y_corrupted[flip_indices] = 1 - y_corrupted[flip_indices]

    return y_corrupted


def get_class_imbalance_values() -> np.ndarray:
    """
    Get standard set of class imbalance values to test.

    Returns:
        Array of class imbalance values from 0.05 to 0.95
    """
    min_size, max_size, num_points = 0.05, 0.95, 19
    points = np.linspace(min_size, max_size, num=num_points)
    return points


def get_data_scarcity_values() -> np.ndarray:
    """
    Get standard set of training sizes for data scarcity experiments.

    Returns:
        Array of training sizes on a log scale
    """
    min_size, max_size, num_points = 1, 10, 20
    points = np.unique(
        np.round(np.logspace(min_size, max_size, num=num_points, base=2)).astype(int)
    )
    return points
