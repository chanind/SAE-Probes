from pathlib import Path

import numpy as np
import torch

from sae_probes.constants import DEFAULT_CACHE_PATH
from sae_probes.utils_data import get_xy_OOD, get_xyvals


def get_xy_OOD_sae(
    dataset: str,
    model_name: str,
    layer: int,
    k: int = 128,
    return_indices: bool = False,
    num_train: int = 1024,
    cache_path: str | Path = DEFAULT_CACHE_PATH,
):
    _, y_test = get_xy_OOD(dataset, model_name, layer)
    _, y_train = get_xyvals(dataset, layer=layer, model_name=model_name, MAX_AMT=1500)
    X_test = (
        torch.load(
            Path(cache_path) / f"sae_activations_{model_name}_OOD/{dataset}_OOD.pt",
            weights_only=False,
        )
        .to_dense()
        .cpu()
    )
    X_train = (
        torch.load(
            Path(cache_path) / f"sae_activations_{model_name}/{dataset}.pt",
            weights_only=True,
        )
        .to_dense()
        .cpu()
    )
    # Get indices for each class
    pos_indices = np.where(y_train == 1)[0]
    neg_indices = np.where(y_train == 0)[0]

    # Take 512 samples from each class
    pos_selected = pos_indices[: num_train // 2]
    neg_selected = neg_indices[: num_train // 2]

    # Combine and shuffle indices
    selected_indices = np.concatenate([pos_selected, neg_selected])
    shuffled_indices = np.random.permutation(selected_indices)

    # Update X_train and y_train with balanced samples
    X_train = X_train[shuffled_indices]
    y_train = y_train[shuffled_indices]  # type: ignore
    X_train_diff = X_train[y_train == 1].mean(dim=0) - X_train[y_train == 0].mean(dim=0)
    sorted_indices = torch.argsort(torch.abs(X_train_diff), descending=True)
    top_by_average_diff = sorted_indices[:k]
    # print(top_by_average_diff)
    X_train_filtered = X_train[:, top_by_average_diff]
    X_test_filtered = X_test[:, top_by_average_diff]
    if return_indices:
        return X_train_filtered, y_train, X_test_filtered, y_test, top_by_average_diff
    return X_train_filtered, y_train, X_test_filtered, y_test
