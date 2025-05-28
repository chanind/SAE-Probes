import os

os.environ["OMP_NUM_THREADS"] = "10"

import argparse
import os
import pickle as pkl
import warnings

import einops
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm
from utils_data import (
    get_dataset_sizes,
    get_numbered_binary_tags,
    get_train_test_indices,
    get_yvals,
)
from utils_training import find_best_reg

warnings.simplefilter("ignore", category=ConvergenceWarning)

data_dir = "data"
model_name = "gemma-2-9b"
max_seq_len = 256
layer = 20
k = 128
device = "cuda:0"

# Default SAE ID parameters
parser = argparse.ArgumentParser()
parser.add_argument(
    "--l0", type=int, default=68, help="L0 value for the SAE", choices=[408, 68]
)
parser.add_argument(
    "--to_run_list",
    type=str,
    nargs="+",
    default=[],
    choices=["baseline_attn", "sae_aggregated", "attn_probing"],
)
args = parser.parse_args()

# Set SAE ID based on arguments
l0 = args.l0
sae_id = f"layer_20/width_16k/average_l0_{l0}"


baseline_csv = pd.read_csv(
    f"results/baseline_probes_{model_name}/normal_settings/layer{layer}_results.csv"
)
sae_csv = pd.read_csv(f"results/sae_probes_{model_name}/normal_setting/all_metrics.csv")

datasets = get_numbered_binary_tags()
dataset_sizes = get_dataset_sizes()

to_run_list = args.to_run_list


def load_model_acts(dataset):
    """Load the original model activations for a dataset"""
    hook_name = f"blocks.{layer}.hook_resid_post"
    file_path = f"{data_dir}/model_activations_{model_name}_{max_seq_len}/{dataset}_{hook_name}.pt"
    return torch.load(file_path, weights_only=True)


def load_sae_acts(dataset, sae_id):
    """Load the SAE-encoded activations for a dataset"""
    width = sae_id.split("/")[1]
    l0 = sae_id.split("/")[2]
    train_path = f"{data_dir}/sae_activations_{model_name}_{max_seq_len}/{dataset}_{layer}_{width}_{l0}_X_train_sae.pt"
    test_path = f"{data_dir}/sae_activations_{model_name}_{max_seq_len}/{dataset}_{layer}_{width}_{l0}_X_test_sae.pt"
    y_train_path = f"{data_dir}/sae_activations_{model_name}_{max_seq_len}/{dataset}_{layer}_{width}_{l0}_y_train.pt"
    y_test_path = f"{data_dir}/sae_activations_{model_name}_{max_seq_len}/{dataset}_{layer}_{width}_{l0}_y_test.pt"

    return {
        "X_train": torch.load(train_path, weights_only=True).to_dense(),
        "X_test": torch.load(test_path, weights_only=True).to_dense(),
        "y_train": torch.load(y_train_path, weights_only=True).to_dense(),
        "y_test": torch.load(y_test_path, weights_only=True).to_dense(),
    }


def train_concat_baseline_on_model_acts(
    X_train, X_test, y_train, y_test, number_to_concat=255, pca_k=20
):
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
        X_train_pos = X_train[:, pos, :]
        X_test_pos = X_test[:, pos, :]

        train_sums = X_train_pos.sum(dim=-1)
        if train_sums.max() == 0:
            continue

        # Fit PCA on training data
        pca = PCA(n_components=pca_k)
        X_train_pca = pca.fit_transform(X_train_pos)
        X_test_pca = pca.transform(X_test_pos)

        train_pca_features.append(X_train_pca)
        test_pca_features.append(X_test_pca)

    # Concatenate all PCA features
    X_train_concat = np.hstack(train_pca_features)
    X_test_concat = np.hstack(test_pca_features)

    res = find_best_reg(
        X_train=X_train_concat,
        y_train=y_train,
        X_test=X_test_concat,
        y_test=y_test,
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
    X_train, X_test, y_train, y_test, aggregation_method, k=None, binarize=False
):
    """Train probe on aggregated activations"""

    train_sums = X_train.sum(dim=-1)
    last_nonzero_train = largest_nonzero_col_per_row(train_sums)

    test_sums = X_test.sum(dim=-1)
    last_nonzero_test = largest_nonzero_col_per_row(test_sums)

    if aggregation_method == "mean":
        # Create masks for each sequence to only include tokens after first_nonzero, skipping first token
        train_mask = (
            torch.arange(X_train.size(1))[None, :] <= last_nonzero_train[:, None]
        ) & (torch.arange(X_train.size(1))[None, :] > 0)
        test_mask = (
            torch.arange(X_test.size(1))[None, :] <= last_nonzero_test[:, None]
        ) & (torch.arange(X_test.size(1))[None, :] > 0)

        # Apply masks and take mean only over valid tokens
        X_train_agg = (X_train * train_mask[:, :, None]).sum(dim=1) / train_mask.sum(
            dim=1
        )[:, None]
        X_test_agg = (X_test * test_mask[:, :, None]).sum(dim=1) / test_mask.sum(dim=1)[
            :, None
        ]
    elif aggregation_method == "last":
        # Create masks to select only the last non-zero token
        train_mask = (
            torch.arange(X_train.size(1))[None, :] == last_nonzero_train[:, None]
        )
        test_mask = torch.arange(X_test.size(1))[None, :] == last_nonzero_test[:, None]

        X_train_agg = (X_train * train_mask[:, :, None]).sum(dim=1)
        X_test_agg = (X_test * test_mask[:, :, None]).sum(dim=1)

    if binarize:
        X_train_agg = X_train_agg > 0
        X_test_agg = X_test_agg > 0

    if k is not None:
        X_train_diff = X_train_agg[y_train == 1].mean(dim=0) - X_train_agg[
            y_train == 0
        ].mean(dim=0)
        sorted_indices = torch.argsort(torch.abs(X_train_diff), descending=True)
        top_by_average_diff = sorted_indices[:k]
        X_train_agg = X_train_agg[:, top_by_average_diff]
        X_test_agg = X_test_agg[:, top_by_average_diff]

    res = find_best_reg(
        X_train=X_train_agg,
        y_train=y_train,
        X_test=X_test_agg,
        y_test=y_test,
        plot=False,
        n_jobs=-1,
        parallel=False,
        penalty="l1",
    )
    return res


def run_sae_aggregated_probing(dataset, layer, sae_id, k, binarize=False):
    """Run SAE aggregated probing experiments"""
    data = load_sae_acts(dataset, sae_id)
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]

    results = {}
    for aggregation_method in ["mean", "last"]:
        res = train_aggregated_probe_on_acts(
            X_train,
            X_test,
            y_train,
            y_test,
            aggregation_method,
            k=k,
            binarize=binarize,
        )
        res["dataset"] = dataset
        res["layer"] = layer
        res["sae_id"] = sae_id
        res["aggregation_method"] = aggregation_method
        res["k"] = k
        res["binarize"] = binarize
        results[aggregation_method] = res

    return results


def run_baseline_concat_probing(dataset, layer, sae_id, number_to_concat=255, pca_k=20):
    """Run baseline concatenation probing experiments"""
    X_model = load_model_acts(dataset)
    y_vals = get_yvals(dataset)
    train_indices, test_indices = get_train_test_indices(dataset)

    X_train_model = X_model[train_indices]
    X_test_model = X_model[test_indices]
    y_train = y_vals[train_indices]
    y_test = y_vals[test_indices]

    res = train_concat_baseline_on_model_acts(
        X_train_model, X_test_model, y_train, y_test, number_to_concat, pca_k
    )
    res["dataset"] = dataset
    res["layer"] = layer
    res["sae_id"] = sae_id  # Note: sae_id is not used but kept for consistency
    res["number_to_concat"] = number_to_concat
    res["pca_k"] = pca_k
    return res


def train_attn_probing(X_train, X_test, y_train, y_test, l2_lambda=0):
    """Train attention-based probing"""
    n_train, seq_len, hidden_dim = X_train.shape
    n_test = X_test.shape[0]

    # Initialize attention weights (one per token position)
    attn_weights = torch.randn(seq_len, requires_grad=True, device=device)

    # Logistic regression weights
    log_reg_weights = torch.randn(hidden_dim, requires_grad=True, device=device)
    log_reg_bias = torch.randn(1, requires_grad=True, device=device)

    optimizer = torch.optim.AdamW(
        [attn_weights, log_reg_weights, log_reg_bias], lr=1e-3
    )

    best_test_auc = 0
    best_metrics = {}

    patience = 20
    epochs_no_improve = 0

    X_train_gpu = X_train.to(device)
    X_test_gpu = X_test.to(device)
    y_train_gpu = y_train.to(device).float()
    y_test_gpu = y_test.to(device).float()

    for epoch in tqdm(range(1000)):
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

            test_auc = roc_auc_score(y_test_gpu.cpu().numpy(), probs_test.cpu().numpy())
            test_acc = accuracy_score(
                y_test_gpu.cpu().numpy(), preds_test.cpu().numpy()
            )

            if test_auc > best_test_auc:
                best_test_auc = test_auc
                best_metrics = {
                    "test_auc": test_auc,
                    "test_acc": test_acc,
                    "attn_weights": softmax_attn.cpu().numpy(),
                    "log_reg_weights": log_reg_weights.cpu().numpy(),
                    "log_reg_bias": log_reg_bias.cpu().numpy(),
                    "epoch": epoch,
                }
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    print(
        f"Best test AUC: {best_test_auc:.4f} at epoch {best_metrics.get('epoch', -1)}"
    )
    return best_metrics


def train_attn_probing_on_model_acts(dataset, layer):
    """Train attention probing on original model activations"""
    X_model = load_model_acts(dataset)
    y_vals = get_yvals(dataset)
    train_indices, test_indices = get_train_test_indices(dataset)

    X_train_model = X_model[train_indices]
    X_test_model = X_model[test_indices]
    y_train = y_vals[train_indices]
    y_test = y_vals[test_indices]

    best_metrics = train_attn_probing(X_train_model, X_test_model, y_train, y_test)
    best_metrics["dataset"] = dataset
    best_metrics["layer"] = layer
    return best_metrics


if "baseline_attn" in to_run_list:
    all_results_baseline_concat = []
    for dataset in tqdm(datasets):
        print(f"Running baseline concatenation probing for {dataset}")
        res = run_baseline_concat_probing(dataset, layer, sae_id)
        all_results_baseline_concat.append(res)
    os.makedirs(f"results/multi_token_probes_{model_name}", exist_ok=True)
    with open(
        f"results/multi_token_probes_{model_name}/baseline_concat_results_l0_{l0}.pkl",
        "wb",
    ) as f:
        pkl.dump(all_results_baseline_concat, f)

if "sae_aggregated" in to_run_list:
    all_results_sae_agg = []
    for dataset in tqdm(datasets):
        print(f"Running SAE aggregated probing for {dataset}")
        res = run_sae_aggregated_probing(dataset, layer, sae_id, k=k, binarize=False)
        all_results_sae_agg.append(res)
    os.makedirs(f"results/multi_token_probes_{model_name}", exist_ok=True)
    with open(
        f"results/multi_token_probes_{model_name}/sae_aggregated_results_l0_{l0}.pkl",
        "wb",
    ) as f:
        pkl.dump(all_results_sae_agg, f)

if "attn_probing" in to_run_list:
    all_results_attn_probing = []
    for dataset in tqdm(datasets):
        print(f"Running attention probing for {dataset}")
        res = train_attn_probing_on_model_acts(dataset, layer)
        all_results_attn_probing.append(res)
    os.makedirs(f"results/multi_token_probes_{model_name}", exist_ok=True)
    with open(
        f"results/multi_token_probes_{model_name}/attn_probing_results_l0_{l0}.pkl",
        "wb",
    ) as f:
        pkl.dump(all_results_attn_probing, f)
