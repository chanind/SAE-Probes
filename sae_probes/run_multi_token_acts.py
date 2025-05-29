import os

os.environ["OMP_NUM_THREADS"] = "10"

import argparse
import pickle as pkl
import warnings
from pathlib import Path
from typing import Any

import einops
import numpy as np
import torch
from sae_lens import SAE
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from tqdm import tqdm
from transformer_lens import HookedTransformer

from sae_probes.utils_data import (
    get_model_activations_for_dataset,
    get_numbered_binary_tags,
)
from sae_probes.utils_sae import get_sae_features
from sae_probes.utils_training import find_best_reg

warnings.simplefilter("ignore", category=ConvergenceWarning)

device_script_default: str | torch.device = (
    "cuda:0" if torch.cuda.is_available() else "cpu"
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--to_run_list",
    type=str,
    nargs="+",
    default=[],
    choices=["baseline_attn", "sae_aggregated", "attn_probing"],
)
args = parser.parse_args()

datasets_tags_list = get_numbered_binary_tags()

to_run_list = args.to_run_list


def train_concat_baseline_on_model_acts(
    X_train: torch.Tensor,
    X_test: torch.Tensor,
    y_train: torch.Tensor,
    y_test: torch.Tensor,
    number_to_concat: int = 255,
    pca_k: int = 20,
) -> dict[str, Any]:
    """Train baseline probe on original model activations"""
    # Get sequence length and feature dimension
    seq_len = X_train.shape[1]

    # Initialize lists to store PCA transformed data
    train_pca_features = []
    test_pca_features = []

    # For each token position
    for pos in tqdm(range(1, number_to_concat + 1)):
        if pos >= seq_len:
            break

        # Get token features for this position
        X_train_pos = X_train[:, pos, :].cpu().numpy()
        X_test_pos = X_test[:, pos, :].cpu().numpy()

        train_sums = X_train_pos.sum(axis=-1)
        if train_sums.max() == 0:
            continue

        # Fit PCA on training data
        current_pca_k = min(pca_k, X_train_pos.shape[0], X_train_pos.shape[1])
        if current_pca_k == 0:
            continue

        pca = PCA(n_components=current_pca_k)
        try:
            X_train_pca = pca.fit_transform(X_train_pos)
            X_test_pca = pca.transform(X_test_pos)
        except ValueError as e:
            print(f"PCA failed for position {pos} with k={current_pca_k}. Error: {e}")
            print(
                f"X_train_pos shape: {X_train_pos.shape}, X_test_pos shape: {X_test_pos.shape}"
            )
            continue

        train_pca_features.append(X_train_pca)
        test_pca_features.append(X_test_pca)

    if not train_pca_features or not test_pca_features:
        print("No PCA features were generated. Returning empty results.")
        return {"error": "No PCA features generated", "roc_auc": 0.0, "accuracy": 0.0}

    # Concatenate all PCA features
    X_train_concat = np.hstack(train_pca_features)
    X_test_concat = np.hstack(test_pca_features)

    # Ensure y_train and y_test are numpy arrays
    y_train_np = y_train.cpu().numpy() if isinstance(y_train, torch.Tensor) else y_train
    y_test_np = y_test.cpu().numpy() if isinstance(y_test, torch.Tensor) else y_test

    res = find_best_reg(
        X_train=X_train_concat,
        y_train=y_train_np,
        X_test=X_test_concat,
        y_test=y_test_np,
        plot=False,
        n_jobs=-1,
        parallel=False,
        penalty="l1",
    )

    return res


def largest_nonzero_col_per_row(A: torch.Tensor, sentinel: int = -1) -> torch.Tensor:
    """
    Returns a 1D tensor of length A.size(0), where each entry is the
    largest column index of a nonzero element in that row of A.
    If a row is entirely zero, its index is set to `sentinel`.

    This uses the "mask * column-indices + max" approach.

    Args:
        A (torch.Tensor): A 2D tensor of shape (rows, cols).
        sentinel (int): The value to assign for rows with no nonzero entries.

    Returns:
        torch.Tensor: A 1D tensor of length A.size(0).
    """
    # 1. Create a boolean mask for nonzero entries
    mask = A != 0  # shape: [rows, cols]

    # 2. Create an integer range [0, 1, 2, ..., cols-1]
    #    and let it broadcast to [rows, cols] when multiplied by mask.
    cols = torch.arange(A.size(1), device=A.device)

    # 3. Multiply mask by the column indices and take max across each row.
    #    This effectively picks the largest column index where the row is nonzero.
    max_indices_per_row = (mask * cols).max(dim=1).values

    # 4. Identify rows that have no nonzero entries. Their max will be 0,
    #    but that may also be a valid column index. So we set them to sentinel explicitly.
    no_nonzeros = ~mask.any(dim=1)
    max_indices_per_row[no_nonzeros] = sentinel

    return max_indices_per_row


def train_aggregated_probe_on_acts(
    X_train: torch.Tensor,
    X_test: torch.Tensor,
    y_train: torch.Tensor,
    y_test: torch.Tensor,
    aggregation_method: str,
    k: int | None = None,
    binarize: bool = False,
) -> dict[str, Any]:
    """Train probe on aggregated activations"""

    train_sums = X_train.sum(dim=-1)
    last_nonzero_train = largest_nonzero_col_per_row(train_sums)

    test_sums = X_test.sum(dim=-1)
    last_nonzero_test = largest_nonzero_col_per_row(test_sums)

    if aggregation_method == "mean":
        # Create masks for each sequence to only include tokens after first_nonzero, skipping first token
        train_mask = (
            torch.arange(X_train.size(1), device=X_train.device)[None, :]
            <= last_nonzero_train[:, None]
        ) & (torch.arange(X_train.size(1), device=X_train.device)[None, :] > 0)
        test_mask = (
            torch.arange(X_test.size(1), device=X_test.device)[None, :]
            <= last_nonzero_test[:, None]
        ) & (torch.arange(X_test.size(1), device=X_test.device)[None, :] > 0)

        # Apply masks and take mean only over valid tokens
        X_train_agg = (X_train * train_mask[:, :, None]).sum(dim=1) / train_mask.sum(
            dim=1
        )[:, None].clamp(min=1e-6)
        X_test_agg = (X_test * test_mask[:, :, None]).sum(dim=1) / test_mask.sum(dim=1)[
            :, None
        ].clamp(min=1e-6)
    elif aggregation_method == "last":
        # Create masks to select only the last non-zero token
        train_mask = (
            torch.arange(X_train.size(1), device=X_train.device)[None, :]
            == last_nonzero_train[:, None]
        )
        test_mask = (
            torch.arange(X_test.size(1), device=X_test.device)[None, :]
            == last_nonzero_test[:, None]
        )

        X_train_agg = (X_train * train_mask[:, :, None]).sum(dim=1)
        X_test_agg = (X_test * test_mask[:, :, None]).sum(dim=1)
    else:
        raise ValueError(f"Unknown aggregation_method: {aggregation_method}")

    if binarize:
        X_train_agg = X_train_agg > 0
        X_test_agg = X_test_agg > 0

    if k is not None:
        # Ensure y_train is boolean for indexing if it's not already (e.g. if it's float)
        y_train_bool = y_train.bool() if y_train.is_floating_point() else y_train

        # Handle cases with insufficient samples for one class
        if torch.sum(y_train_bool) == 0 or torch.sum(~y_train_bool) == 0:
            print(
                f"Warning: Insufficient samples for one class in dataset. Skipping k-selection for aggregation method {aggregation_method}."
            )
            # Fallback: use all features if k-selection is not possible
        else:
            mean_class1 = X_train_agg[y_train_bool].mean(dim=0)
            mean_class0 = X_train_agg[~y_train_bool].mean(dim=0)
            X_train_diff = mean_class1 - mean_class0

            # Ensure X_train_diff is 1D
            if X_train_diff.ndim == 0:  # if X_train_agg was (N, 1), mean becomes scalar
                X_train_diff = X_train_diff.unsqueeze(0)

            actual_k = min(k, X_train_agg.shape[1])  # Ensure k is not out of bounds
            if actual_k > 0:  # Proceed only if there are features to select
                sorted_indices = torch.argsort(torch.abs(X_train_diff), descending=True)
                top_by_average_diff = sorted_indices[:actual_k]
                X_train_agg = X_train_agg[:, top_by_average_diff]
                X_test_agg = X_test_agg[:, top_by_average_diff]
            elif X_train_agg.shape[1] == 0:  # No features left
                print(
                    f"Warning: No features available for aggregation method {aggregation_method} after potential binarization. Probing might fail."
                )

    res = find_best_reg(
        X_train=X_train_agg.cpu().numpy()
        if isinstance(X_train_agg, torch.Tensor)
        else X_train_agg,
        y_train=y_train.cpu().numpy() if isinstance(y_train, torch.Tensor) else y_train,
        X_test=X_test_agg.cpu().numpy()
        if isinstance(X_test_agg, torch.Tensor)
        else X_test_agg,
        y_test=y_test.cpu().numpy() if isinstance(y_test, torch.Tensor) else y_test,
        plot=False,
        n_jobs=-1,
        parallel=False,
        penalty="l1",
    )
    return res


def run_sae_aggregated_probing_generic(
    model: HookedTransformer,
    sae: SAE,
    dataset_name: str,
    layer_idx: int,
    aggregation_methods: list[str],
    results_dir: str | Path,
    device: str | torch.device,
    hook_point_name: str | None = None,
    pooling_strategy: str = "last",
    k: int | None = None,
    binarize: bool = False,
    max_seq_len_override: int | None = None,
    setting_type: str = "normal",
    expected_activation_dim: int | None = None,
    data_path_base: str | Path = "data",
    seed: int = 42,
) -> dict[str, Any]:
    """
    Run SAE aggregated probing experiments generically.
    """
    results_dir = Path(results_dir)
    os.makedirs(results_dir, exist_ok=True)

    if hook_point_name is None:
        if hasattr(sae, "cfg") and hasattr(sae.cfg, "hook_point"):
            hook_point_name = sae.cfg.hook_point
            print(f"Using hook_point_name from sae.cfg: {hook_point_name}")
        else:
            raise ValueError(
                "hook_point_name must be provided if not available in sae.cfg.hook_point"
            )

    actual_max_seq_len = (
        max_seq_len_override if max_seq_len_override is not None else model.cfg.n_ctx
    )

    print(
        f"Running SAE aggregated probing for {dataset_name}, layer {layer_idx}, SAE: {getattr(sae.cfg, 'sae_name', 'N/A')}"
    )

    # 1. Get model activations
    X_train_model_acts, X_test_model_acts, y_train, y_test = (
        get_model_activations_for_dataset(
            model=model,
            dataset_name=dataset_name,
            layer_idx=layer_idx,
            setting_type=setting_type,
            hook_point_name=hook_point_name,
            pooling_strategy="none",
            expected_activation_dim=expected_activation_dim or model.cfg.d_model,
            max_seq_len=actual_max_seq_len,
            data_path_base=Path(data_path_base),
            device=device,
            seed=seed,
            force_regenerate=False,
            include_padding_in_hook_name=False,
        )
    )

    if X_train_model_acts.ndim == 2:
        raise ValueError(
            f"Model activations for {dataset_name} are 2D (pooled) but multi-token aggregation needs 3D (unpooled). Check pooling_strategy in get_model_activations_for_dataset."
        )

    # Ensure y_train and y_test are on the correct device and are Tensors
    y_train = (
        torch.tensor(y_train, device=device, dtype=torch.long)
        if not isinstance(y_train, torch.Tensor)
        else y_train.to(device, dtype=torch.long)
    )
    y_test = (
        torch.tensor(y_test, device=device, dtype=torch.long)
        if not isinstance(y_test, torch.Tensor)
        else y_test.to(device, dtype=torch.long)
    )

    # 2. Get SAE features
    X_train_sae_acts, _ = get_sae_features(
        sae=sae,
        model_activations=X_train_model_acts,
        device=device,
        use_encoder_if_present=True,
    )
    X_test_sae_acts, _ = get_sae_features(
        sae=sae,
        model_activations=X_test_model_acts,
        device=device,
        use_encoder_if_present=True,
    )

    print(f"Shape of X_train_sae_acts: {X_train_sae_acts.shape}")
    if X_train_sae_acts.ndim != 3 or X_test_sae_acts.ndim != 3:
        raise ValueError(
            f"SAE activations for {dataset_name} are not 3D after get_sae_features. Expected (batch, seq, d_sae), got {X_train_sae_acts.shape}"
        )

    all_results = {}
    for agg_method in aggregation_methods:
        print(f"  Training with aggregation method: {agg_method}")
        res = train_aggregated_probe_on_acts(
            X_train=X_train_sae_acts.to(device),
            X_test=X_test_sae_acts.to(device),
            y_train=y_train.to(device),
            y_test=y_test.to(device),
            aggregation_method=agg_method,
            k=k,
            binarize=binarize,
        )
        all_results[agg_method] = res
        print(f"    AUC for {agg_method}: {res['roc_auc']:.4f}")

    # Save results
    sae_name_for_path = getattr(sae.cfg, "sae_name", "unknown_sae").replace("/", "_")
    results_file = (
        results_dir
        / f"sae_agg_probe_{model.cfg.model_name}_layer{layer_idx}_{sae_name_for_path}_{dataset_name}.pkl"
    )
    with open(results_file, "wb") as f:
        pkl.dump(all_results, f)
    print(f"Saved SAE aggregated probing results to {results_file}")

    return all_results


def run_baseline_concat_probing_generic(
    model: HookedTransformer,
    dataset_name: str,
    layer_idx: int,
    results_dir: str | Path,
    device: str | torch.device,
    hook_point_name: str,
    number_to_concat: int = 255,
    pca_k: int = 20,
    max_seq_len_override: int | None = None,
    setting_type: str = "normal",
    expected_activation_dim: int | None = None,
    data_path_base: str | Path = "data",
    seed: int = 42,
) -> dict[str, Any]:
    """
    Run baseline concat probing experiments generically using model activations.
    """
    results_dir = Path(results_dir)
    os.makedirs(results_dir, exist_ok=True)

    actual_max_seq_len = (
        max_seq_len_override if max_seq_len_override is not None else model.cfg.n_ctx
    )

    print(
        f"Running Baseline Concat probing for {dataset_name}, layer {layer_idx}, hook {hook_point_name}"
    )

    # 1. Get model activations (unpooled)
    X_train_model_acts, X_test_model_acts, y_train, y_test = (
        get_model_activations_for_dataset(
            model=model,
            dataset_name=dataset_name,
            layer_idx=layer_idx,
            setting_type=setting_type,
            hook_point_name=hook_point_name,
            pooling_strategy="none",
            expected_activation_dim=expected_activation_dim or model.cfg.d_model,
            max_seq_len=actual_max_seq_len,
            data_path_base=Path(data_path_base),
            device=device,
            seed=seed,
            force_regenerate=False,
        )
    )

    if X_train_model_acts.ndim == 2 or X_test_model_acts.ndim == 2:
        raise ValueError(
            f"Model activations for {dataset_name} are 2D (pooled) but baseline concat probing needs 3D (unpooled). "
            f"Check pooling_strategy in get_model_activations_for_dataset. Shape: {X_train_model_acts.shape}"
        )

    # Ensure y_train and y_test are Tensors and on the same device as activations for train_concat_baseline
    y_train = (
        torch.tensor(y_train, device=device, dtype=torch.long)
        if not isinstance(y_train, torch.Tensor)
        else y_train.to(device, dtype=torch.long)
    )
    y_test = (
        torch.tensor(y_test, device=device, dtype=torch.long)
        if not isinstance(y_test, torch.Tensor)
        else y_test.to(device, dtype=torch.long)
    )

    # X_train_model_acts and X_test_model_acts are already on `device` from get_model_activations_for_dataset

    # 2. Train the probe
    print(
        f"Shape of X_train_model_acts for concat baseline: {X_train_model_acts.shape}"
    )
    results = train_concat_baseline_on_model_acts(
        X_train=X_train_model_acts,
        X_test=X_test_model_acts,
        y_train=y_train,
        y_test=y_test,
        number_to_concat=number_to_concat,
        pca_k=pca_k,
    )

    # Save results
    results_file = (
        results_dir
        / f"baseline_concat_probe_{model.cfg.model_name}_layer{layer_idx}_{hook_point_name.replace('.', '_')}_{dataset_name}.pkl"
    )
    with open(results_file, "wb") as f:
        pkl.dump(results, f)
    print(f"Saved baseline concat probing results to {results_file}")

    return results


def train_attn_probing(
    X_train: torch.Tensor,
    X_test: torch.Tensor,
    y_train: torch.Tensor,
    y_test: torch.Tensor,
    device: str | torch.device,
    l2_lambda: float = 0.0,
) -> dict[str, Any]:
    """Train attention-based probing"""
    n_train, seq_len, hidden_dim = X_train.shape
    # n_test = X_test.shape[0] # Not explicitly used, but good for verification

    # Initialize attention weights (one per token position)
    attn_weights = torch.randn(seq_len, requires_grad=True, device=device)

    # Logistic regression weights
    log_reg_weights = torch.randn(hidden_dim, requires_grad=True, device=device)
    log_reg_bias = torch.randn(1, requires_grad=True, device=device)

    optimizer = torch.optim.AdamW(
        [attn_weights, log_reg_weights, log_reg_bias], lr=1e-3
    )

    best_test_auc = 0.0
    best_metrics: dict[str, Any] = {}

    patience = 20
    epochs_no_improve = 0

    # Ensure all tensors are on the target device
    X_train_gpu = X_train.to(device)
    X_test_gpu = X_test.to(device)
    y_train_gpu = y_train.to(device).float()
    y_test_gpu = y_test.to(device).float()

    for epoch in range(1000):
        optimizer.zero_grad()

        # Apply attention
        softmax_attn = torch.softmax(attn_weights, dim=0)
        X_train_weighted = einops.einsum(X_train_gpu, softmax_attn, "b s h, s -> b h")

        # Logistic regression
        logits_train = (
            einops.einsum(X_train_weighted, log_reg_weights, "b h, h -> b")
            + log_reg_bias.squeeze()
        )
        loss_train = torch.nn.functional.binary_cross_entropy_with_logits(
            logits_train, y_train_gpu
        )

        # L2 regularization for logistic regression weights
        l2_reg = l2_lambda * (log_reg_weights**2).sum()
        total_loss = loss_train + l2_reg

        total_loss.backward()
        optimizer.step()

        # Evaluation (no grad)
        with torch.no_grad():
            X_test_weighted = einops.einsum(X_test_gpu, softmax_attn, "b s h, s -> b h")
            logits_test = (
                einops.einsum(X_test_weighted, log_reg_weights, "b h, h -> b")
                + log_reg_bias.squeeze()
            )
            probs_test = torch.sigmoid(logits_test)
            preds_test = (probs_test > 0.5).float()

            # Ensure y_test_gpu and preds_test are on CPU for sklearn metrics
            y_test_cpu = y_test_gpu.cpu().numpy()
            probs_test_cpu = probs_test.cpu().numpy()
            preds_test_cpu = preds_test.cpu().numpy()

            try:
                test_auc = roc_auc_score(y_test_cpu, probs_test_cpu)
                test_acc = accuracy_score(y_test_cpu, preds_test_cpu)
            except ValueError as e:
                # This can happen if y_test_cpu contains only one class
                print(
                    f"Warning: ROC AUC/Accuracy calculation failed (epoch {epoch}): {e}. Setting to 0.5/0.0."
                )
                test_auc = 0.5
                test_acc = 0.0

            if test_auc > best_test_auc:
                best_test_auc = test_auc
                best_metrics = {
                    "roc_auc": test_auc,
                    "accuracy": test_acc,
                    "attn_weights": softmax_attn.cpu().numpy(),
                    "log_reg_weights": log_reg_weights.cpu().numpy(),
                    "log_reg_bias": log_reg_bias.cpu().numpy(),
                    "epoch": epoch,
                }
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                break
    if not best_metrics:
        print(
            "Warning: Attn probing did not yield improved metrics. Returning placeholder."
        )
        best_metrics = {
            "roc_auc": 0.5,
            "accuracy": 0.0,
            "epoch": -1,
        }

    return best_metrics


def run_attn_probing_on_model_acts_generic(
    model: HookedTransformer,
    dataset_name: str,
    layer_idx: int,
    hook_point_name: str,
    results_dir: str | Path,
    device: str | torch.device,
    l2_lambda: float = 0.0,
    max_seq_len_override: int | None = None,
    setting_type: str = "normal",
    expected_activation_dim: int | None = None,
    data_path_base: str | Path = "data",
    seed: int = 42,
) -> dict[str, Any]:
    """
    Run attention-based probing generically using model activations from a specified hook point.
    Activations are expected to be 3D (batch, seq_len, feature_dim).
    """
    results_dir = Path(results_dir)
    os.makedirs(results_dir, exist_ok=True)

    actual_max_seq_len = (
        max_seq_len_override if max_seq_len_override is not None else model.cfg.n_ctx
    )

    print(
        f"Running Attention Probing for {dataset_name}, layer {layer_idx}, hook {hook_point_name}"
    )

    # 1. Get model activations (unpooled, 3D)
    X_train_model_acts, X_test_model_acts, y_train, y_test = (
        get_model_activations_for_dataset(
            model=model,
            dataset_name=dataset_name,
            layer_idx=layer_idx,
            setting_type=setting_type,
            hook_point_name=hook_point_name,
            pooling_strategy="none",
            expected_activation_dim=expected_activation_dim or model.cfg.d_model,
            max_seq_len=actual_max_seq_len,
            data_path_base=Path(data_path_base),
            device=device,
            seed=seed,
            force_regenerate=False,
        )
    )

    if X_train_model_acts.ndim != 3 or X_test_model_acts.ndim != 3:
        raise ValueError(
            f"Model activations for {dataset_name} are not 3D (batch, seq, dim) as required for attention probing. "
            f"Shape: {X_train_model_acts.shape}. Check hook_point_name and pooling_strategy."
        )

    if X_train_model_acts.shape[0] == 0 or X_test_model_acts.shape[0] == 0:
        print(
            f"Warning: Empty train ({X_train_model_acts.shape[0]}) or test ({X_test_model_acts.shape[0]}) activations for {dataset_name}. Returning empty metrics."
        )
        return {"roc_auc": 0.5, "accuracy": 0.0, "error": "No data"}

    # Ensure y_train and y_test are Tensors (already handled by get_model_activations_for_dataset before saving, but good practice)
    # They will be moved to the correct device within train_attn_probing
    y_train = (
        torch.tensor(y_train, dtype=torch.long)
        if not isinstance(y_train, torch.Tensor)
        else y_train.long()
    )
    y_test = (
        torch.tensor(y_test, dtype=torch.long)
        if not isinstance(y_test, torch.Tensor)
        else y_test.long()
    )

    # 2. Train the attention probe
    # X_train_model_acts, X_test_model_acts are on CPU after get_model_activations_for_dataset
    # train_attn_probing will move them to the specified `device`
    print(
        f"Shape of X_train_model_acts for attention probing: {X_train_model_acts.shape}"
    )
    results = train_attn_probing(
        X_train=X_train_model_acts,
        X_test=X_test_model_acts,
        y_train=y_train,
        y_test=y_test,
        device=device,
        l2_lambda=l2_lambda,
    )

    # Add metadata to results
    results["dataset"] = dataset_name
    results["layer"] = layer_idx
    results["hook_point_name"] = hook_point_name
    results["l2_lambda"] = l2_lambda

    # Save results
    hook_point_path_str = (
        hook_point_name.replace(".", "_").replace("[", "_").replace("]", "")
    )
    results_file = (
        results_dir
        / f"attn_probe_{model.cfg.model_name}_layer{layer_idx}_{hook_point_path_str}_{dataset_name}.pkl"
    )
    with open(results_file, "wb") as f:
        pkl.dump(results, f)
    print(f"Saved attention probing results to {results_file}")

    return results


if __name__ == "__main__":
    # --- Placeholder Model and SAE Loading ---
    # In a real script, you would load your specific model and SAEs here.
    # These are illustrative placeholders.

    # Try to load a model
    try:
        # Use a small model for faster placeholder execution if script is run
        placeholder_model = HookedTransformer.from_pretrained(
            "gpt2-small", device=device_script_default
        )
        placeholder_model.eval()
        # placeholder_model.cfg.model_name might be like "gpt2-small"
        # placeholder_model.cfg.n_ctx would be its context length
        # placeholder_model.cfg.d_model would be its hidden dimension
        # placeholder_model.cfg.device should be device_script_default
    except Exception as e:
        print(
            f"Could not load placeholder model (gpt2-small): {e}. Some script parts might not run."
        )
        placeholder_model = None

    # Placeholder for SAE loading - this needs a specific SAE from HF or local path
    # Example: placeholder_sae = SAE.load_from_hf("your-hf-org/your-sae-repo-name", device=device_script_default)
    # For the script to run parts requiring an SAE, this would need to be a valid SAE object.
    placeholder_sae: SAE | None = None
    # As a dummy, if you have a local SAE config and want to test the flow:
    # from sae_lens.training.config import LanguageModelSAERunnerConfig
    # from sae_lens.training.sae_group import SAETrainingGroup
    # sae_cfg_dict = {
    #     "hook_point": f"blocks.0.hook_resid_pre", # Example, must match model
    #     "d_in": placeholder_model.cfg.d_model if placeholder_model else 768,
    #     "d_sae": (placeholder_model.cfg.d_model if placeholder_model else 768) * 4, # Example expansion
    #     "sae_group_name": "gpt2-small-test-saes",
    #     "model_name": placeholder_model.cfg.model_name if placeholder_model else "gpt2-small",
    #     "normalize_sae_decoder": False,
    #     "device": device_script_default,
    #     # ... other necessary SAE config fields
    # }
    # try:
    #     # This is a simplification; actual SAE creation might involve more steps / full config
    #     if placeholder_model:
    #        # placeholder_sae = SAE(cfg=SAEConfig.from_dict(sae_cfg_dict)) # Construct dummy SAE if needed
    #        # placeholder_sae.cfg.sae_name = "dummy_sae_layer0" # Give it a name for paths
    #        pass # Keep placeholder_sae = None for now to avoid full SAE setup here.
    # except Exception as e:
    #     print(f"Could not create placeholder SAE: {e}")
    #     placeholder_sae = None

    # Script-level parameters that were previously global, now more explicit
    # These would ideally be arguments to the script or configured externally.
    # For multi-token experiments, often a specific layer's activations/SAEs are used.
    # This `multi_token_target_layer` would typically correspond to the layer the SAE was trained on.

    multi_token_target_layer = 0  # Example layer, adjust as needed for your model/SAE
    # If placeholder_sae were loaded, its config (sae.cfg.hook_point_layer or similar) would be the source of truth.
    # For now, assume `multi_token_target_layer` is the intended layer for all experiments in this script.

    base_results_dir = Path("results/multi_token_probes_generic")

    # Ensure operations are only attempted if the placeholder model is available
    if placeholder_model:
        model_name_for_paths = placeholder_model.cfg.model_name.replace("/", "_")
        specific_model_results_dir = base_results_dir / model_name_for_paths
        os.makedirs(specific_model_results_dir, exist_ok=True)

        # Shared hook point for residual stream if not SAE-specific
        # This should align with how SAEs are typically trained if used for baseline comparison
        # The generic functions will use sae.cfg.hook_point if available for SAEs
        default_resid_hook_point = f"blocks.{multi_token_target_layer}.hook_resid_pre"
        if default_resid_hook_point not in [
            name for name, _ in placeholder_model.hook_points()
        ]:
            default_resid_hook_point = (
                f"blocks.{multi_token_target_layer}.hook_resid_post"  # Fallback
            )

        # --- Baseline Concatenation Probing ---
        if (
            "baseline_concat" in to_run_list
        ):  # Assuming 'baseline_attn' was a typo for 'baseline_concat'
            print("\\n--- Running Baseline Concatenation Probing ---")
            all_results_baseline_concat = []  # If you still want to aggregate results in script
            for dataset_name_str in tqdm(
                datasets_tags_list, desc="Datasets (Baseline Concat)"
            ):
                print(f"  Processing dataset: {dataset_name_str}")
                res = run_baseline_concat_probing_generic(
                    model=placeholder_model,
                    dataset_name=dataset_name_str,
                    layer_idx=multi_token_target_layer,
                    results_dir=specific_model_results_dir / "baseline_concat",
                    device=device_script_default,
                    hook_point_name=default_resid_hook_point,  # Use a standard residual stream hook
                    number_to_concat=min(
                        255, placeholder_model.cfg.n_ctx - 1
                    ),  # Concat up to context length
                    pca_k=20,  # Default from old script
                    max_seq_len_override=placeholder_model.cfg.n_ctx,
                    expected_activation_dim=placeholder_model.cfg.d_model,
                )
                all_results_baseline_concat.append(
                    {"dataset": dataset_name_str, "results": res}
                )
            # Aggregated saving (optional, as generic functions save individually)
            # if all_results_baseline_concat:
            #     agg_save_path = specific_model_results_dir / f"baseline_concat_all_datasets_layer{multi_token_target_layer}.pkl"
            #     with open(agg_save_path, "wb") as f: pkl.dump(all_results_baseline_concat, f)
            #     print(f"Saved aggregated baseline_concat results to {agg_save_path}")

        # --- SAE Aggregated Probing ---
        if "sae_aggregated" in to_run_list:
            print("\\n--- Running SAE Aggregated Probing ---")
            if placeholder_sae:
                all_results_sae_agg = []
                for dataset_name_str in tqdm(
                    datasets_tags_list, desc="Datasets (SAE Aggregated)"
                ):
                    print(f"  Processing dataset: {dataset_name_str}")
                    # Determine hook point from SAE if possible, else default
                    sae_hook_point = getattr(
                        placeholder_sae.cfg, "hook_point", default_resid_hook_point
                    )
                    sae_layer_idx = (
                        multi_token_target_layer  # Assuming this matches SAE's layer
                    )
                    if hasattr(placeholder_sae.cfg, "hook_point_layer"):
                        sae_layer_idx = placeholder_sae.cfg.hook_point_layer
                    elif hasattr(placeholder_sae.cfg, "hook_layer"):
                        sae_layer_idx = placeholder_sae.cfg.hook_layer

                    res = run_sae_aggregated_probing_generic(
                        model=placeholder_model,
                        sae=placeholder_sae,
                        dataset_name=dataset_name_str,
                        layer_idx=sae_layer_idx,  # Use SAE's layer
                        aggregation_methods=["mean", "last"],
                        results_dir=specific_model_results_dir / "sae_aggregated",
                        device=device_script_default,
                        hook_point_name=sae_hook_point,  # Use SAE's hook point
                        k=128,  # Default k from old script globals
                        binarize=False,  # Default
                        max_seq_len_override=placeholder_model.cfg.n_ctx,
                        expected_activation_dim=placeholder_model.cfg.d_model,  # Model acts before SAE
                    )
                    all_results_sae_agg.append(
                        {"dataset": dataset_name_str, "results": res}
                    )
                # Optional aggregated saving
                # if all_results_sae_agg:
                #     sae_id_for_path = getattr(placeholder_sae.cfg, 'sae_name', 'unknown_sae').replace('/','_')
                #     agg_save_path = specific_model_results_dir / f"sae_aggregated_all_datasets_{sae_id_for_path}.pkl"
                #     with open(agg_save_path, "wb") as f: pkl.dump(all_results_sae_agg, f)
                #     print(f"Saved aggregated sae_aggregated results to {agg_save_path}")
            else:
                print(
                    "  Skipping SAE aggregated probing as placeholder_sae is not defined."
                )

        # --- Attention Probing ---
        if "attn_probing" in to_run_list:
            print("\\n--- Running Attention Probing ---")
            # Choose an appropriate attention hook point for the model
            # Common options: hook_z (output of attention heads) or hook_result (output of attention layer)
            attn_hook_point = f"blocks.{multi_token_target_layer}.attn.hook_z"
            if attn_hook_point not in [
                name for name, _ in placeholder_model.hook_points()
            ]:
                attn_hook_point = (
                    f"blocks.{multi_token_target_layer}.attn.hook_result"  # Fallback
                )

            if attn_hook_point not in [
                name for name, _ in placeholder_model.hook_points()
            ]:
                print(
                    f"  Warning: Could not find a suitable attention hook point like {attn_hook_point} in {placeholder_model.cfg.model_name}. Skipping attention probing."
                )
            else:
                print(f"  Using attention hook point: {attn_hook_point}")
                # Determine expected activation dimension for attention hook
                # For hook_z, it's typically d_model (if QKV bias makes it so) or d_head * n_heads
                # For hook_result, it's d_model
                # This is a simplification; a more robust way would be to inspect model.hook_dict[attn_hook_point].shape[-1] after a forward pass.
                # For gpt2-small (d_model=768), hook_z is (batch, seq, n_head, d_head) usually before flatten, or (batch, seq, d_model) after.
                # get_model_activations_for_dataset with pooling='none' should return (batch, seq, feat)
                # Let's assume the hook returns d_model for simplicity here or d_attn_out if that's a field.
                attn_dim = getattr(
                    placeholder_model.cfg, "d_attn_out", placeholder_model.cfg.d_model
                )

                all_results_attn_probing = []
                for dataset_name_str in tqdm(
                    datasets_tags_list, desc="Datasets (Attn Probing)"
                ):
                    print(f"  Processing dataset: {dataset_name_str}")
                    res = run_attn_probing_on_model_acts_generic(
                        model=placeholder_model,
                        dataset_name=dataset_name_str,
                        layer_idx=multi_token_target_layer,
                        hook_point_name=attn_hook_point,
                        results_dir=specific_model_results_dir / "attn_probing",
                        device=device_script_default,
                        l2_lambda=0.001,  # Example
                        max_seq_len_override=placeholder_model.cfg.n_ctx,
                        expected_activation_dim=attn_dim,
                    )
                    all_results_attn_probing.append(
                        {"dataset": dataset_name_str, "results": res}
                    )
                # Optional aggregated saving
                # if all_results_attn_probing:
                #     hook_name_for_file = attn_hook_point.replace('.','_')
                #     agg_save_path = specific_model_results_dir / f"attn_probing_all_datasets_layer{multi_token_target_layer}_{hook_name_for_file}.pkl"
                #     with open(agg_save_path, "wb") as f: pkl.dump(all_results_attn_probing, f)
                #     print(f"Saved aggregated attn_probing results to {agg_save_path}")

        print(
            "\\n--- Multi-token script execution finished (or skipped parts if placeholders were not met) ---"
        )
    else:
        print("Placeholder model not loaded. Cannot run multi-token experiments.")
