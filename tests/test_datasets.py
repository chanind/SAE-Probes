import numpy as np
import pandas as pd
import pytest

from sae_probes.datasets import (
    DEFAULT_DATASET_PATH,
    DatasetInfo,
    corrupt_labels,
    get_binary_datasets,
    get_class_imbalance_values,
    get_data_scarcity_values,
    get_train_test_indices,
    load_dataset,
)


def test_get_binary_datasets_loads_default_data() -> None:
    datasets = get_binary_datasets()

    # Basic validation of returned data
    assert isinstance(datasets, list)
    assert len(datasets) == 113
    assert datasets[0].tag == "5_hist_fig_ismale"
    assert datasets[0].size == 5000

    # Verify each item is a properly formed DatasetInfo
    for dataset in datasets:
        assert isinstance(dataset, DatasetInfo)
        assert isinstance(dataset.tag, str)
        assert len(dataset.tag) > 0
        assert isinstance(dataset.size, int)
        assert dataset.size >= 0
        assert isinstance(dataset.description, str)


def test_load_dataset() -> None:
    # Test with default parameters
    df, train_indices, test_indices = load_dataset(
        "5_hist_fig_ismale", DEFAULT_DATASET_PATH
    )

    # Verify returned dataframe
    assert isinstance(df, pd.DataFrame)
    assert "prompt" in df.columns
    assert "target" in df.columns

    # Verify train/test indices
    assert isinstance(train_indices, np.ndarray)
    assert isinstance(test_indices, np.ndarray)
    assert len(train_indices) == 1024  # Default num_train
    assert len(np.intersect1d(train_indices, test_indices)) == 0  # No overlap


def test_load_dataset_custom_parameters() -> None:
    # Test with custom parameters
    df, train_indices, test_indices = load_dataset(
        "5_hist_fig_ismale",
        DEFAULT_DATASET_PATH,
        num_train=100,
        test_size=200,
        pos_ratio=0.7,
        seed=123,
    )

    assert len(train_indices) == 100
    assert len(test_indices) == 200


@pytest.mark.parametrize(
    "noise_fraction,expected_changes",
    [
        (0.0, 0),  # No corruption
        (0.5, 4),  # Maximum allowed corruption (50% of 9 labels, rounded down)
    ],
)
def test_corrupt_labels_noise_levels(
    noise_fraction: float, expected_changes: int
) -> None:
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1])
    y_corrupted = corrupt_labels(y, noise_fraction=noise_fraction)
    if noise_fraction == 0.0:
        assert np.array_equal(y, y_corrupted)
    else:
        assert np.sum(y != y_corrupted) == expected_changes


def test_corrupt_labels_raises_on_invalid_noise() -> None:
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1])
    with pytest.raises(AssertionError):
        corrupt_labels(y, noise_fraction=0.6)


def test_get_train_test_indices_sizes_default_parameters() -> None:
    y = np.array([1, 1, 1, 0, 0])
    train_indices, test_indices = get_train_test_indices(y, num_train=3)

    assert len(train_indices) == 3
    assert len(test_indices) == 2
    assert len(np.intersect1d(train_indices, test_indices)) == 0


def test_get_train_test_indices_raises_on_insufficient_data() -> None:
    y = np.array([1, 1, 1, 0, 0])
    with pytest.raises(ValueError):
        get_train_test_indices(y, num_train=10)


@pytest.mark.parametrize(
    "num_train,test_size,pos_ratio,seed",
    [
        (100, 200, 0.7, 123),
    ],
)
def test_load_dataset_with_parameters(
    num_train: int, test_size: int, pos_ratio: float, seed: int
) -> None:
    df, train_indices, test_indices = load_dataset(
        "5_hist_fig_ismale",
        DEFAULT_DATASET_PATH,
        num_train=num_train,
        test_size=test_size,
        pos_ratio=pos_ratio,
        seed=seed,
    )

    assert len(train_indices) == num_train
    assert len(test_indices) == test_size


def test_get_class_imbalance_values() -> None:
    values = get_class_imbalance_values()

    assert isinstance(values, np.ndarray)
    assert len(values) == 19
    assert np.isclose(values[0], 0.05)
    assert np.isclose(values[-1], 0.95)
    assert np.all(np.diff(values) > 0)  # Values are strictly increasing


def test_get_data_scarcity_values() -> None:
    values = get_data_scarcity_values()

    assert isinstance(values, np.ndarray)
    assert len(values) > 0
    assert np.all(np.diff(values) > 0)  # Values are strictly increasing
    assert values[0] >= 2  # First value is at least 2^1
    assert values[-1] <= 1024  # Last value is at most 2^10
