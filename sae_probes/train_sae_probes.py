import os
import pickle as pkl
import warnings

import torch
from sae_lens import SAE
from sklearn.exceptions import ConvergenceWarning
from tqdm import tqdm
from transformer_lens import HookedTransformer

from sae_probes.utils_data import (
    get_class_imbalance,
    get_corrupt_frac,
    get_dataset_sizes,
    get_model_activations_for_dataset,
    get_numbered_binary_tags,
    get_training_sizes,
)
from sae_probes.utils_sae import get_sae_features
from sae_probes.utils_training import find_best_reg

warnings.simplefilter("ignore", category=ConvergenceWarning)
torch.set_grad_enabled(False)

# Constants and datasets
dataset_sizes = get_dataset_sizes()
datasets: list[str] = get_numbered_binary_tags()
train_sizes: list[int] = get_training_sizes()
corrupt_fracs: list[float] = get_corrupt_frac()
fracs: list[float] = get_class_imbalance()


# Normal setting functions
def get_normal_sae_paths_generic(
    dataset_name: str,
    sae_name: str,
    reg_type: str,
    binarize: bool = False,
    model_name_str: str = "model",
):
    description_string = f"{dataset_name}_{sae_name}"

    if binarize:
        reg_type += "_binarized"

    save_dir = f"results/sae_probes_{model_name_str}/normal_setting/"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{description_string}_{reg_type}.pkl")
    return {"save_path": save_path}


def run_normal_probe_logic(
    X_train_sae_features: torch.Tensor,
    X_test_sae_features: torch.Tensor,
    y_train: torch.Tensor,
    y_test: torch.Tensor,
    dataset_name: str,
    sae_name: str,
    sae_hook_layer: int | str,
    reg_type: str,
    binarize: bool = False,
    save_results_path: str | None = None,
):
    ks = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    all_metrics = []

    if X_train_sae_features.shape[0] == 0 or X_test_sae_features.shape[0] == 0:
        print(
            f"Skipping probe for {dataset_name}, SAE {sae_name}: Empty train or test features."
        )
        return []

    if X_train_sae_features.shape[1] == 0:
        print(
            f"Skipping probe for {dataset_name}, SAE {sae_name}: SAE features have 0 dimension."
        )
        for k_val in ks:
            metrics = {
                "auc_test": 0.5,
                "auc_train": 0.5,
                "acc_test": 0.0,
                "acc_train": 0.0,
                "f1_test": 0.0,
                "f1_train": 0.0,
            }
            metrics["k"] = k_val
            metrics["dataset"] = dataset_name
            metrics["layer"] = sae_hook_layer
            metrics["sae_id"] = sae_name
            metrics["reg_type"] = reg_type
            metrics["binarize"] = binarize
            all_metrics.append(metrics)
    else:
        if len(torch.unique(y_train)) < 2:
            print(
                f"Warning: y_train for {dataset_name} has fewer than 2 unique classes. Skipping diff calculation, using all features or first k."
            )
            sorted_indices = torch.arange(X_train_sae_features.shape[1])
        else:
            X_train_diff = X_train_sae_features[y_train == 1].mean(
                dim=0
            ) - X_train_sae_features[y_train == 0].mean(dim=0)
            if torch.isnan(X_train_diff).any():
                print(
                    f"Warning: NaN in X_train_diff for {dataset_name}, SAE {sae_name}. Using all features or first k."
                )
                sorted_indices = torch.arange(X_train_sae_features.shape[1])
            else:
                sorted_indices = torch.argsort(torch.abs(X_train_diff), descending=True)

        for k_val in tqdm(ks, desc=f"K-loop for {sae_name} on {dataset_name}"):
            if k_val > X_train_sae_features.shape[1]:
                top_by_average_diff = sorted_indices[: X_train_sae_features.shape[1]]
            else:
                top_by_average_diff = sorted_indices[:k_val]

            if top_by_average_diff.shape[0] == 0 and X_train_sae_features.shape[1] > 0:
                print(
                    f"Warning: top_by_average_diff is empty for k={k_val} on {dataset_name}, SAE {sae_name} but features exist. Using first feature if available."
                )
                if X_train_sae_features.shape[1] > 0:
                    X_train_filtered = X_train_sae_features[:, :1]
                    X_test_filtered = X_test_sae_features[:, :1]
                else:
                    X_train_filtered = X_train_sae_features
                    X_test_filtered = X_test_sae_features
            elif X_train_sae_features.shape[1] == 0:
                X_train_filtered = X_train_sae_features
                X_test_filtered = X_test_sae_features
            else:
                X_train_filtered = X_train_sae_features[:, top_by_average_diff]
                X_test_filtered = X_test_sae_features[:, top_by_average_diff]

            if binarize:
                if X_train_filtered.numel() > 0:
                    X_train_filtered = X_train_filtered > 0
                if X_test_filtered.numel() > 0:
                    X_test_filtered = X_test_filtered > 0

            if X_train_filtered.shape[0] == 0:
                print(
                    f"Skipping k={k_val} for {dataset_name}, SAE {sae_name}: No training samples after filtering/binarization."
                )
                metrics = {
                    "auc_test": 0.5,
                    "auc_train": 0.5,
                    "acc_test": 0.0,
                    "acc_train": 0.0,
                    "f1_test": 0.0,
                    "f1_train": 0.0,
                }
            elif X_train_filtered.shape[1] == 0 and k_val > 0:
                print(
                    f"Skipping k={k_val} for {dataset_name}, SAE {sae_name}: No features selected by top_k logic."
                )
                metrics = {
                    "auc_test": 0.5,
                    "auc_train": 0.5,
                    "acc_test": 0.0,
                    "acc_train": 0.0,
                    "f1_test": 0.0,
                    "f1_train": 0.0,
                }
            else:
                metrics = find_best_reg(
                    X_train=X_train_filtered,
                    y_train=y_train,
                    X_test=X_test_filtered,
                    y_test=y_test,
                    plot=False,
                    n_jobs=-1,
                    parallel=False,
                    penalty=reg_type,
                )
            metrics["k"] = k_val
            metrics["dataset"] = dataset_name
            metrics["layer"] = sae_hook_layer
            metrics["sae_id"] = sae_name
            metrics["reg_type"] = reg_type
            metrics["binarize"] = binarize
            all_metrics.append(metrics)

    if save_results_path:
        print(f"Saving results to {save_results_path}")
        os.makedirs(os.path.dirname(save_results_path), exist_ok=True)
        with open(save_results_path, "wb") as f:
            pkl.dump(all_metrics, f)

    return all_metrics


def run_normal_experiments_generic(
    model: HookedTransformer,
    saes_list: list[SAE],
    reg_type: str,
    binarize: bool = False,
    target_dataset_names: list[str] | None = None,
    device: str | torch.device | None = None,
):
    if device is None:
        device = model.cfg.device

    model.to(device)
    model_name_str = model.cfg.model_name.replace("/", "_")

    datasets_to_run = target_dataset_names if target_dataset_names else datasets

    for sae in tqdm(saes_list, desc="Processing SAEs"):
        sae.to(device)
        sae.eval()
        sae_hook_layer: int | str
        if hasattr(sae.cfg, "hook_layer") and isinstance(sae.cfg.hook_layer, int):
            sae_hook_layer = sae.cfg.hook_layer
        elif hasattr(sae.cfg, "hook_point_layer") and isinstance(
            sae.cfg.hook_point_layer, int
        ):
            sae_hook_layer = sae.cfg.hook_point_layer
        else:
            try:
                sae_hook_layer = int(str(sae.cfg.hook_point).split(".")[1])
            except:
                raise ValueError(
                    f"Cannot determine SAE layer from sae.cfg for SAE: {sae.cfg.sae_name}. Need sae.cfg.hook_layer or sae.cfg.hook_point_layer as int, or parsable sae.cfg.hook_point."
                )

        sae_name = getattr(
            sae.cfg, "sae_name", f"sae_layer{sae_hook_layer}_dim{sae.cfg.d_sae}"
        )

        for dataset_name in tqdm(
            datasets_to_run, desc=f"Datasets for SAE {sae_name}", leave=False
        ):
            paths = get_normal_sae_paths_generic(
                dataset_name=dataset_name,
                sae_name=sae_name,
                reg_type=reg_type,
                binarize=binarize,
                model_name_str=model_name_str,
            )

            if os.path.exists(paths["save_path"]):
                print(
                    f"Results already exist for {dataset_name}, SAE {sae_name}, reg {reg_type}. Skipping: {paths['save_path']}"
                )
                continue

            print(
                f"Running probe for dataset {dataset_name}, SAE {sae_name} (Layer {sae_hook_layer}), reg_type {reg_type}"
            )

            try:
                X_train_model_acts, X_test_model_acts, y_train, y_test = (
                    get_model_activations_for_dataset(
                        model=model,
                        dataset_name=dataset_name,
                        layer_idx=sae_hook_layer,
                        device=device,
                        setting_type="normal",
                        max_seq_len=model.cfg.n_ctx,
                    )
                )
            except Exception as e:
                print(
                    f"Failed to get model activations for {dataset_name}, layer {sae_hook_layer}: {e}. Skipping."
                )
                continue

            if X_train_model_acts.shape[0] == 0 or y_train.shape[0] == 0:
                print(
                    f"No training data/labels for {dataset_name}, layer {sae_hook_layer}. Skipping SAE feature computation."
                )
                continue

            if X_train_model_acts.shape[-1] != sae.d_in:
                print(
                    f"Warning: Model activation dimension ({X_train_model_acts.shape[-1]}) does not match SAE input dimension ({sae.d_in}) for SAE {sae_name} on {dataset_name}. Skipping."
                )
                continue

            X_train_sae_features = get_sae_features(
                sae, X_train_model_acts, device=device
            )
            X_test_sae_features = get_sae_features(
                sae, X_test_model_acts, device=device
            )

            run_normal_probe_logic(
                X_train_sae_features=X_train_sae_features,
                X_test_sae_features=X_test_sae_features,
                y_train=y_train,
                y_test=y_test,
                dataset_name=dataset_name,
                sae_name=sae_name,
                sae_hook_layer=sae_hook_layer,
                reg_type=reg_type,
                binarize=binarize,
                save_results_path=paths["save_path"],
            )
    print("All normal experiments run for the provided SAEs.")


# Scarcity setting functions
def get_scarcity_sae_paths_generic(
    dataset_name: str,
    sae_name: str,
    reg_type: str,
    num_train: int,
    model_name_str: str = "model",
):
    description_string = f"{dataset_name}_{sae_name}_{num_train}"
    save_dir = f"results/sae_probes_{model_name_str}/scarcity_setting/"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{description_string}_{reg_type}.pkl")
    return {"save_path": save_path}


def run_scarcity_probe_logic(
    X_train_sae_features: torch.Tensor,
    X_test_sae_features: torch.Tensor,
    y_train: torch.Tensor,
    y_test: torch.Tensor,
    dataset_name: str,
    sae_name: str,
    sae_hook_layer: int | str,
    reg_type: str,
    num_train: int,
    save_results_path: str | None = None,
):
    ks = [16, 128]
    all_metrics = []
    if X_train_sae_features.shape[0] == 0 or X_test_sae_features.shape[0] == 0:
        print(
            f"Skipping scarcity probe for {dataset_name}, SAE {sae_name}, num_train {num_train}: Empty train or test features."
        )
        return []

    if X_train_sae_features.shape[1] == 0:
        print(
            f"Skipping scarcity probe for {dataset_name}, SAE {sae_name}: SAE features have 0 dimension."
        )
        for k_val in ks:
            metrics = {
                "auc_test": 0.5,
                "auc_train": 0.5,
                "acc_test": 0.0,
                "acc_train": 0.0,
                "f1_test": 0.0,
                "f1_train": 0.0,
            }
            metrics["k"] = k_val
            metrics["dataset"] = dataset_name
            metrics["layer"] = sae_hook_layer
            metrics["sae_id"] = sae_name
            metrics["reg_type"] = reg_type
            metrics["num_train"] = X_train_sae_features.shape[0]
            all_metrics.append(metrics)
    else:
        if len(torch.unique(y_train)) < 2:
            print(
                f"Warning: y_train for {dataset_name} (scarcity {num_train}) has fewer than 2 unique classes. Using all features or first k."
            )
            sorted_indices = torch.arange(X_train_sae_features.shape[1])
        else:
            X_train_diff = X_train_sae_features[y_train == 1].mean(
                dim=0
            ) - X_train_sae_features[y_train == 0].mean(dim=0)
            if torch.isnan(X_train_diff).any():
                print(
                    f"Warning: NaN in X_train_diff for {dataset_name} (scarcity {num_train}), SAE {sae_name}. Using all features or first k."
                )
                sorted_indices = torch.arange(X_train_sae_features.shape[1])
            else:
                sorted_indices = torch.argsort(torch.abs(X_train_diff), descending=True)

        for k_val in tqdm(
            ks, desc=f"K-loop for {sae_name} on {dataset_name} (scarcity {num_train})"
        ):
            if k_val > X_train_sae_features.shape[1]:
                top_by_average_diff = sorted_indices[: X_train_sae_features.shape[1]]
            else:
                top_by_average_diff = sorted_indices[:k_val]

            if top_by_average_diff.shape[0] == 0 and X_train_sae_features.shape[1] > 0:
                print(
                    f"Warning: top_by_average_diff is empty for k={k_val} on {dataset_name} (scarcity {num_train}), SAE {sae_name} but features exist. Using first feature."
                )
                if X_train_sae_features.shape[1] > 0:
                    X_train_filtered = X_train_sae_features[:, :1]
                    X_test_filtered = X_test_sae_features[:, :1]
                else:
                    X_train_filtered = X_train_sae_features
                    X_test_filtered = X_test_sae_features
            elif X_train_sae_features.shape[1] == 0:
                X_train_filtered = X_train_sae_features
                X_test_filtered = X_test_sae_features
            else:
                X_train_filtered = X_train_sae_features[:, top_by_average_diff]
                X_test_filtered = X_test_sae_features[:, top_by_average_diff]

            if X_train_filtered.shape[0] == 0:
                metrics = {
                    "auc_test": 0.5,
                    "auc_train": 0.5,
                    "acc_test": 0.0,
                    "acc_train": 0.0,
                    "f1_test": 0.0,
                    "f1_train": 0.0,
                }
            elif X_train_filtered.shape[1] == 0 and k_val > 0:
                metrics = {
                    "auc_test": 0.5,
                    "auc_train": 0.5,
                    "acc_test": 0.0,
                    "acc_train": 0.0,
                    "f1_test": 0.0,
                    "f1_train": 0.0,
                }
            else:
                metrics = find_best_reg(
                    X_train=X_train_filtered,
                    y_train=y_train,
                    X_test=X_test_filtered,
                    y_test=y_test,
                    plot=False,
                    n_jobs=-1,
                    parallel=False,
                    penalty=reg_type,
                )
            metrics["k"] = k_val
            metrics["dataset"] = dataset_name
            metrics["layer"] = sae_hook_layer
            metrics["sae_id"] = sae_name
            metrics["reg_type"] = reg_type
            metrics["num_train"] = X_train_sae_features.shape[0]
            all_metrics.append(metrics)

    if save_results_path:
        print(f"Saving scarcity results to {save_results_path}")
        os.makedirs(os.path.dirname(save_results_path), exist_ok=True)
        with open(save_results_path, "wb") as f:
            pkl.dump(all_metrics, f)
    return all_metrics


def run_scarcity_experiments_generic(
    model: HookedTransformer,
    saes_list: list[SAE],
    reg_type: str,
    target_dataset_names: list[str] | None = None,
    device: str | torch.device | None = None,
):
    if device is None:
        device = model.cfg.device
    model.to(device)
    model_name_str = model.cfg.model_name.replace("/", "_")

    datasets_to_run = target_dataset_names if target_dataset_names else datasets
    current_train_sizes = train_sizes

    for sae in tqdm(saes_list, desc="Processing SAEs for Scarcity"):
        sae.to(device)
        sae.eval()
        sae_hook_layer: int | str
        if hasattr(sae.cfg, "hook_layer") and isinstance(sae.cfg.hook_layer, int):
            sae_hook_layer = sae.cfg.hook_layer
        elif hasattr(sae.cfg, "hook_point_layer") and isinstance(
            sae.cfg.hook_point_layer, int
        ):
            sae_hook_layer = sae.cfg.hook_point_layer
        else:
            try:
                sae_hook_layer = int(str(sae.cfg.hook_point).split(".")[1])
            except:
                raise ValueError(
                    f"Cannot determine SAE layer for scarcity: {sae.cfg.sae_name}"
                )
        sae_name = getattr(
            sae.cfg, "sae_name", f"sae_layer{sae_hook_layer}_dim{sae.cfg.d_sae}"
        )

        for dataset_name in tqdm(
            datasets_to_run, desc=f"Datasets for SAE {sae_name} (Scarcity)", leave=False
        ):
            for num_train_target in tqdm(
                current_train_sizes, desc=f"NumTrain for {dataset_name}", leave=False
            ):
                paths = get_scarcity_sae_paths_generic(
                    dataset_name=dataset_name,
                    sae_name=sae_name,
                    reg_type=reg_type,
                    num_train=num_train_target,
                    model_name_str=model_name_str,
                )
                if os.path.exists(paths["save_path"]):
                    print(
                        f"Scarcity results exist for {dataset_name}, SAE {sae_name}, num_train {num_train_target}, reg {reg_type}. Skipping."
                    )
                    continue

                print(
                    f"Running scarcity probe: DS {dataset_name}, SAE {sae_name} (L{sae_hook_layer}), Reg {reg_type}, NumTrain {num_train_target}"
                )

                try:
                    X_train_model_acts, X_test_model_acts, y_train, y_test = (
                        get_model_activations_for_dataset(
                            model=model,
                            dataset_name=dataset_name,
                            layer_idx=sae_hook_layer,
                            device=device,
                            setting_type="scarcity",
                            num_train_samples=num_train_target,
                            max_seq_len=model.cfg.n_ctx,
                        )
                    )
                except Exception as e:
                    print(
                        f"Failed to get model activations for scarcity: {dataset_name}, layer {sae_hook_layer}, num_train {num_train_target}: {e}. Skipping."
                    )
                    continue

                if X_train_model_acts.shape[0] == 0 or y_train.shape[0] == 0:
                    print(
                        f"No training data/labels for scarcity: {dataset_name}, layer {sae_hook_layer}, num_train {num_train_target}. Skipping."
                    )
                    continue
                if X_train_model_acts.shape[0] != num_train_target:
                    print(
                        f"Warning: Actual num_train {X_train_model_acts.shape[0]} doesn't match target {num_train_target} for {dataset_name}"
                    )

                if X_train_model_acts.shape[-1] != sae.d_in:
                    print(
                        f"Warning: Model activation dim ({X_train_model_acts.shape[-1]}) != SAE input dim ({sae.d_in}) for {sae_name} on {dataset_name} (scarcity). Skipping."
                    )
                    continue

                X_train_sae_features = get_sae_features(
                    sae, X_train_model_acts, device=device
                )
                X_test_sae_features = get_sae_features(
                    sae, X_test_model_acts, device=device
                )

                run_scarcity_probe_logic(
                    X_train_sae_features=X_train_sae_features,
                    X_test_sae_features=X_test_sae_features,
                    y_train=y_train,
                    y_test=y_test,
                    dataset_name=dataset_name,
                    sae_name=sae_name,
                    sae_hook_layer=sae_hook_layer,
                    reg_type=reg_type,
                    num_train=num_train_target,
                    save_results_path=paths["save_path"],
                )
    print("All scarcity experiments run.")


# Noise setting functions
def get_noise_sae_paths_generic(
    dataset_name: str,
    sae_name: str,
    reg_type: str,
    corrupt_frac_val: float,
    model_name_str: str = "model",
):
    corrupt_frac_str = str(corrupt_frac_val).replace(".", "p")
    description_string = f"{dataset_name}_{sae_name}_noise_{corrupt_frac_str}"
    save_dir = f"results/sae_probes_{model_name_str}/noise_setting/"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{description_string}_{reg_type}.pkl")
    return {"save_path": save_path}


def run_noise_probe_logic(
    X_train_sae_features: torch.Tensor,
    X_test_sae_features: torch.Tensor,
    y_train_corrupted: torch.Tensor,
    y_test: torch.Tensor,
    dataset_name: str,
    sae_name: str,
    sae_hook_layer: int | str,
    reg_type: str,
    corrupt_frac_val: float,
    save_results_path: str | None = None,
):
    ks = [128]
    all_metrics = []

    if X_train_sae_features.shape[0] == 0 or X_test_sae_features.shape[0] == 0:
        print(
            f"Skipping noise probe for {dataset_name}, SAE {sae_name}, corrupt_frac {corrupt_frac_val}: Empty train or test features."
        )
        return []

    if X_train_sae_features.shape[1] == 0:
        print(
            f"Skipping noise probe for {dataset_name}, SAE {sae_name}: SAE features have 0 dimension."
        )
        for k_val in ks:
            metrics = {
                "auc_test": 0.5,
                "auc_train": 0.5,
                "acc_test": 0.0,
                "acc_train": 0.0,
                "f1_test": 0.0,
                "f1_train": 0.0,
            }
            metrics["k"] = k_val
            metrics["dataset"] = dataset_name
            metrics["layer"] = sae_hook_layer
            metrics["sae_id"] = sae_name
            metrics["reg_type"] = reg_type
            metrics["corrupt_frac"] = corrupt_frac_val
            all_metrics.append(metrics)
    else:
        if len(torch.unique(y_train_corrupted)) < 2:
            print(
                f"Warning: y_train_corrupted for {dataset_name} (noise {corrupt_frac_val}) has fewer than 2 unique classes. Using all features or first k."
            )
            sorted_indices = torch.arange(X_train_sae_features.shape[1])
        else:
            X_train_diff = X_train_sae_features[y_train_corrupted == 1].mean(
                dim=0
            ) - X_train_sae_features[y_train_corrupted == 0].mean(dim=0)
            if torch.isnan(X_train_diff).any():
                print(
                    f"Warning: NaN in X_train_diff for {dataset_name} (noise {corrupt_frac_val}), SAE {sae_name}. Using all features or first k."
                )
                sorted_indices = torch.arange(X_train_sae_features.shape[1])
            else:
                sorted_indices = torch.argsort(torch.abs(X_train_diff), descending=True)

        for k_val in tqdm(
            ks,
            desc=f"K-loop for {sae_name} on {dataset_name} (noise {corrupt_frac_val})",
        ):
            if k_val > X_train_sae_features.shape[1]:
                top_by_average_diff = sorted_indices[: X_train_sae_features.shape[1]]
            else:
                top_by_average_diff = sorted_indices[:k_val]

            if top_by_average_diff.shape[0] == 0 and X_train_sae_features.shape[1] > 0:
                print(
                    f"Warning: top_by_average_diff is empty for k={k_val} on {dataset_name} (noise {corrupt_frac_val}), SAE {sae_name}. Using first feature."
                )
                if X_train_sae_features.shape[1] > 0:
                    X_train_filtered = X_train_sae_features[:, :1]
                    X_test_filtered = X_test_sae_features[:, :1]
                else:
                    X_train_filtered = X_train_sae_features
                    X_test_filtered = X_test_sae_features
            elif X_train_sae_features.shape[1] == 0:
                X_train_filtered = X_train_sae_features
                X_test_filtered = X_test_sae_features
            else:
                X_train_filtered = X_train_sae_features[:, top_by_average_diff]
                X_test_filtered = X_test_sae_features[:, top_by_average_diff]

            if X_train_filtered.shape[0] == 0:
                metrics = {
                    "auc_test": 0.5,
                    "auc_train": 0.5,
                    "acc_test": 0.0,
                    "acc_train": 0.0,
                    "f1_test": 0.0,
                    "f1_train": 0.0,
                }
            elif X_train_filtered.shape[1] == 0 and k_val > 0:
                metrics = {
                    "auc_test": 0.5,
                    "auc_train": 0.5,
                    "acc_test": 0.0,
                    "acc_train": 0.0,
                    "f1_test": 0.0,
                    "f1_train": 0.0,
                }
            else:
                metrics = find_best_reg(
                    X_train=X_train_filtered,
                    y_train=y_train_corrupted,
                    X_test=X_test_filtered,
                    y_test=y_test,
                    plot=False,
                    n_jobs=-1,
                    parallel=False,
                    penalty=reg_type,
                )
            metrics["k"] = k_val
            metrics["dataset"] = dataset_name
            metrics["layer"] = sae_hook_layer
            metrics["sae_id"] = sae_name
            metrics["reg_type"] = reg_type
            metrics["corrupt_frac"] = corrupt_frac_val
            all_metrics.append(metrics)

    if save_results_path:
        print(f"Saving noise results to {save_results_path}")
        os.makedirs(os.path.dirname(save_results_path), exist_ok=True)
        with open(save_results_path, "wb") as f:
            pkl.dump(all_metrics, f)
    return all_metrics


def run_noise_experiments_generic(
    model: HookedTransformer,
    saes_list: list[SAE],
    reg_type: str,
    target_dataset_names: list[str] | None = None,
    device: str | torch.device | None = None,
):
    if device is None:
        device = model.cfg.device
    model.to(device)
    model_name_str = model.cfg.model_name.replace("/", "_")

    datasets_to_run = target_dataset_names if target_dataset_names else datasets
    current_corrupt_fracs = corrupt_fracs

    for sae in tqdm(saes_list, desc="Processing SAEs for Noise"):
        sae.to(device)
        sae.eval()
        sae_hook_layer: int | str
        if hasattr(sae.cfg, "hook_layer") and isinstance(sae.cfg.hook_layer, int):
            sae_hook_layer = sae.cfg.hook_layer
        elif hasattr(sae.cfg, "hook_point_layer") and isinstance(
            sae.cfg.hook_point_layer, int
        ):
            sae_hook_layer = sae.cfg.hook_point_layer
        else:
            try:
                sae_hook_layer = int(str(sae.cfg.hook_point).split(".")[1])
            except:
                raise ValueError(
                    f"Cannot determine SAE layer for noise: {sae.cfg.sae_name}"
                )
        sae_name = getattr(
            sae.cfg, "sae_name", f"sae_layer{sae_hook_layer}_dim{sae.cfg.d_sae}"
        )

        for dataset_name in tqdm(
            datasets_to_run, desc=f"Datasets for SAE {sae_name} (Noise)", leave=False
        ):
            for corrupt_frac_val in tqdm(
                current_corrupt_fracs,
                desc=f"CorruptFrac for {dataset_name}",
                leave=False,
            ):
                paths = get_noise_sae_paths_generic(
                    dataset_name=dataset_name,
                    sae_name=sae_name,
                    reg_type=reg_type,
                    corrupt_frac_val=corrupt_frac_val,
                    model_name_str=model_name_str,
                )
                if os.path.exists(paths["save_path"]):
                    print(
                        f"Noise results exist for {dataset_name}, SAE {sae_name}, corrupt_frac {corrupt_frac_val}, reg {reg_type}. Skipping."
                    )
                    continue

                print(
                    f"Running noise probe: DS {dataset_name}, SAE {sae_name} (L{sae_hook_layer}), Reg {reg_type}, CorruptFrac {corrupt_frac_val}"
                )

                try:
                    (
                        X_train_model_acts,
                        X_test_model_acts,
                        y_train_corrupted,
                        y_test_clean,
                    ) = get_model_activations_for_dataset(
                        model=model,
                        dataset_name=dataset_name,
                        layer_idx=sae_hook_layer,
                        device=device,
                        setting_type="noise",
                        corrupt_frac_val=corrupt_frac_val,
                        max_seq_len=model.cfg.n_ctx,
                    )
                except Exception as e:
                    print(
                        f"Failed to get model activations for noise: {dataset_name}, layer {sae_hook_layer}, corrupt_frac {corrupt_frac_val}: {e}. Skipping."
                    )
                    continue

                if X_train_model_acts.shape[0] == 0 or y_train_corrupted.shape[0] == 0:
                    print(
                        f"No training data/labels for noise: {dataset_name}, layer {sae_hook_layer}, corrupt_frac {corrupt_frac_val}. Skipping."
                    )
                    continue

                if X_train_model_acts.shape[-1] != sae.d_in:
                    print(
                        f"Warning: Model activation dim ({X_train_model_acts.shape[-1]}) != SAE input dim ({sae.d_in}) for {sae_name} on {dataset_name} (noise). Skipping."
                    )
                    continue

                X_train_sae_features = get_sae_features(
                    sae, X_train_model_acts, device=device
                )
                X_test_sae_features = get_sae_features(
                    sae, X_test_model_acts, device=device
                )

                run_noise_probe_logic(
                    X_train_sae_features=X_train_sae_features,
                    X_test_sae_features=X_test_sae_features,
                    y_train_corrupted=y_train_corrupted,
                    y_test=y_test_clean,
                    dataset_name=dataset_name,
                    sae_name=sae_name,
                    sae_hook_layer=sae_hook_layer,
                    reg_type=reg_type,
                    corrupt_frac_val=corrupt_frac_val,
                    save_results_path=paths["save_path"],
                )
    print("All noise experiments run.")


# Class Imbalance setting functions
def get_imbalance_sae_paths_generic(
    dataset_name: str,
    sae_name: str,
    reg_type: str,
    frac_val: float,
    model_name_str: str = "model",
):
    frac_str = str(frac_val).replace(".", "p")
    description_string = f"{dataset_name}_{sae_name}_imbalance_{frac_str}"
    save_dir = f"results/sae_probes_{model_name_str}/imbalance_setting/"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{description_string}_{reg_type}.pkl")
    return {"save_path": save_path}


def run_imbalance_probe_logic(
    X_train_sae_features: torch.Tensor,
    X_test_sae_features: torch.Tensor,
    y_train_imbalanced: torch.Tensor,
    y_test: torch.Tensor,
    dataset_name: str,
    sae_name: str,
    sae_hook_layer: int | str,
    reg_type: str,
    frac_val: float,
    save_results_path: str | None = None,
):
    ks = [128]
    all_metrics = []

    if X_train_sae_features.shape[0] == 0 or X_test_sae_features.shape[0] == 0:
        print(
            f"Skipping imbalance probe for {dataset_name}, SAE {sae_name}, frac {frac_val}: Empty train or test features."
        )
        return []

    if X_train_sae_features.shape[1] == 0:
        print(
            f"Skipping imbalance probe for {dataset_name}, SAE {sae_name}: SAE features have 0 dimension."
        )
        for k_val in ks:
            metrics = {
                "auc_test": 0.5,
                "auc_train": 0.5,
                "acc_test": 0.0,
                "acc_train": 0.0,
                "f1_test": 0.0,
                "f1_train": 0.0,
            }
            metrics["k"] = k_val
            metrics["dataset"] = dataset_name
            metrics["layer"] = sae_hook_layer
            metrics["sae_id"] = sae_name
            metrics["reg_type"] = reg_type
            metrics["frac"] = frac_val
            all_metrics.append(metrics)
    else:
        if len(torch.unique(y_train_imbalanced)) < 2:
            print(
                f"Warning: y_train_imbalanced for {dataset_name} (imbalance {frac_val}) has fewer than 2 unique classes. Using all features or first k."
            )
            sorted_indices = torch.arange(X_train_sae_features.shape[1])
        else:
            X_train_diff = X_train_sae_features[y_train_imbalanced == 1].mean(
                dim=0
            ) - X_train_sae_features[y_train_imbalanced == 0].mean(dim=0)
            if torch.isnan(X_train_diff).any():
                print(
                    f"Warning: NaN in X_train_diff for {dataset_name} (imbalance {frac_val}), SAE {sae_name}. Using all features or first k."
                )
                sorted_indices = torch.arange(X_train_sae_features.shape[1])
            else:
                sorted_indices = torch.argsort(torch.abs(X_train_diff), descending=True)

        for k_val in tqdm(
            ks, desc=f"K-loop for {sae_name} on {dataset_name} (imbalance {frac_val})"
        ):
            if k_val > X_train_sae_features.shape[1]:
                top_by_average_diff = sorted_indices[: X_train_sae_features.shape[1]]
            else:
                top_by_average_diff = sorted_indices[:k_val]

            if top_by_average_diff.shape[0] == 0 and X_train_sae_features.shape[1] > 0:
                print(
                    f"Warning: top_by_average_diff is empty for k={k_val} on {dataset_name} (imbalance {frac_val}), SAE {sae_name}. Using first feature."
                )
                if X_train_sae_features.shape[1] > 0:
                    X_train_filtered = X_train_sae_features[:, :1]
                    X_test_filtered = X_test_sae_features[:, :1]
                else:
                    X_train_filtered = X_train_sae_features
                    X_test_filtered = X_test_sae_features
            elif X_train_sae_features.shape[1] == 0:
                X_train_filtered = X_train_sae_features
                X_test_filtered = X_test_sae_features
            else:
                X_train_filtered = X_train_sae_features[:, top_by_average_diff]
                X_test_filtered = X_test_sae_features[:, top_by_average_diff]

            if X_train_filtered.shape[0] == 0:
                metrics = {
                    "auc_test": 0.5,
                    "auc_train": 0.5,
                    "acc_test": 0.0,
                    "acc_train": 0.0,
                    "f1_test": 0.0,
                    "f1_train": 0.0,
                }
            elif X_train_filtered.shape[1] == 0 and k_val > 0:
                metrics = {
                    "auc_test": 0.5,
                    "auc_train": 0.5,
                    "acc_test": 0.0,
                    "acc_train": 0.0,
                    "f1_test": 0.0,
                    "f1_train": 0.0,
                }
            else:
                metrics = find_best_reg(
                    X_train=X_train_filtered,
                    y_train=y_train_imbalanced,
                    X_test=X_test_filtered,
                    y_test=y_test,
                    plot=False,
                    n_jobs=-1,
                    parallel=False,
                    penalty=reg_type,
                    class_weight="balanced",
                )
            metrics["k"] = k_val
            metrics["dataset"] = dataset_name
            metrics["layer"] = sae_hook_layer
            metrics["sae_id"] = sae_name
            metrics["reg_type"] = reg_type
            metrics["frac"] = frac_val
            all_metrics.append(metrics)

    if save_results_path:
        print(f"Saving imbalance results to {save_results_path}")
        os.makedirs(os.path.dirname(save_results_path), exist_ok=True)
        with open(save_results_path, "wb") as f:
            pkl.dump(all_metrics, f)
    return all_metrics


def run_imbalance_experiments_generic(
    model: HookedTransformer,
    saes_list: list[SAE],
    reg_type: str,
    target_dataset_names: list[str] | None = None,
    device: str | torch.device | None = None,
):
    if device is None:
        device = model.cfg.device
    model.to(device)
    model_name_str = model.cfg.model_name.replace("/", "_")

    datasets_to_run = target_dataset_names if target_dataset_names else datasets
    current_fracs = fracs

    for sae in tqdm(saes_list, desc="Processing SAEs for Imbalance"):
        sae.to(device)
        sae.eval()
        sae_hook_layer: int | str
        if hasattr(sae.cfg, "hook_layer") and isinstance(sae.cfg.hook_layer, int):
            sae_hook_layer = sae.cfg.hook_layer
        elif hasattr(sae.cfg, "hook_point_layer") and isinstance(
            sae.cfg.hook_point_layer, int
        ):
            sae_hook_layer = sae.cfg.hook_point_layer
        else:
            try:
                sae_hook_layer = int(str(sae.cfg.hook_point).split(".")[1])
            except:
                raise ValueError(
                    f"Cannot determine SAE layer for imbalance: {sae.cfg.sae_name}"
                )
        sae_name = getattr(
            sae.cfg, "sae_name", f"sae_layer{sae_hook_layer}_dim{sae.cfg.d_sae}"
        )

        for dataset_name in tqdm(
            datasets_to_run,
            desc=f"Datasets for SAE {sae_name} (Imbalance)",
            leave=False,
        ):
            for frac_val in tqdm(
                current_fracs, desc=f"ImbalanceFrac for {dataset_name}", leave=False
            ):
                paths = get_imbalance_sae_paths_generic(
                    dataset_name=dataset_name,
                    sae_name=sae_name,
                    reg_type=reg_type,
                    frac_val=frac_val,
                    model_name_str=model_name_str,
                )
                if os.path.exists(paths["save_path"]):
                    print(
                        f"Imbalance results exist for {dataset_name}, SAE {sae_name}, frac {frac_val}, reg {reg_type}. Skipping."
                    )
                    continue

                print(
                    f"Running imbalance probe: DS {dataset_name}, SAE {sae_name} (L{sae_hook_layer}), Reg {reg_type}, Frac {frac_val}"
                )

                try:
                    (
                        X_train_model_acts,
                        X_test_model_acts,
                        y_train_imbalanced,
                        y_test_balanced,
                    ) = get_model_activations_for_dataset(
                        model=model,
                        dataset_name=dataset_name,
                        layer_idx=sae_hook_layer,
                        device=device,
                        setting_type="imbalance",
                        class_imbalance_frac=frac_val,
                        max_seq_len=model.cfg.n_ctx,
                    )
                except Exception as e:
                    print(
                        f"Failed to get model activations for imbalance: {dataset_name}, layer {sae_hook_layer}, frac {frac_val}: {e}. Skipping."
                    )
                    continue

                if X_train_model_acts.shape[0] == 0 or y_train_imbalanced.shape[0] == 0:
                    print(
                        f"No training data/labels for imbalance: {dataset_name}, layer {sae_hook_layer}, frac {frac_val}. Skipping."
                    )
                    continue

                if X_train_model_acts.shape[-1] != sae.d_in:
                    print(
                        f"Warning: Model activation dim ({X_train_model_acts.shape[-1]}) != SAE input dim ({sae.d_in}) for {sae_name} on {dataset_name} (imbalance). Skipping."
                    )
                    continue

                X_train_sae_features = get_sae_features(
                    sae, X_train_model_acts, device=device
                )
                X_test_sae_features = get_sae_features(
                    sae, X_test_model_acts, device=device
                )

                run_imbalance_probe_logic(
                    X_train_sae_features=X_train_sae_features,
                    X_test_sae_features=X_test_sae_features,
                    y_train_imbalanced=y_train_imbalanced,
                    y_test=y_test_balanced,
                    dataset_name=dataset_name,
                    sae_name=sae_name,
                    sae_hook_layer=sae_hook_layer,
                    reg_type=reg_type,
                    frac_val=frac_val,
                    save_results_path=paths["save_path"],
                )
    print("All imbalance experiments run.")
