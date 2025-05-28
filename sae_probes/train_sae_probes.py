import argparse
import os
import pickle as pkl
import random
import warnings

import torch
from sklearn.exceptions import ConvergenceWarning
from tqdm import tqdm
from utils_data import (
    get_class_imbalance,
    get_corrupt_frac,
    get_dataset_sizes,
    get_numbered_binary_tags,
    get_training_sizes,
)
from utils_sae import get_sae_layers, layer_to_sae_ids
from utils_training import find_best_reg

warnings.simplefilter("ignore", category=ConvergenceWarning)
torch.set_grad_enabled(False)

# Constants and datasets
dataset_sizes = get_dataset_sizes()
datasets = get_numbered_binary_tags()
train_sizes = get_training_sizes()
corrupt_fracs = get_corrupt_frac()
fracs = get_class_imbalance()


def load_activations(path):
    return torch.load(path, weights_only=True).to_dense().float()


# Normal setting functions
def get_normal_sae_paths(
    dataset, layer, sae_id, reg_type, binarize=False, model_name="gemma-2-9b"
):
    if model_name == "gemma-2-9b":
        width = sae_id.split("/")[1]
        l0 = sae_id.split("/")[2]
        description_string = f"{dataset}_{layer}_{width}_{l0}"
    elif model_name == "llama-3.1-8b":
        description_string = f"{dataset}_{sae_id}"
    elif model_name == "gemma-2-2b":
        name = "_".join(sae_id[2].split("/")[0].split("_")[1:])
        l0 = sae_id[3]
        rounded_l0 = round(float(l0))
        description_string = f"{dataset}_{name}_{rounded_l0}"
    else:
        raise ValueError(f"Invalid model name: {model_name}")

    if binarize:
        reg_type += "_binarized"

    save_path = f"data/sae_probes_{model_name}/normal_setting/{description_string}_{reg_type}.pkl"
    train_path = f"data/sae_activations_{model_name}/normal_setting/{description_string}_X_train_sae.pt"
    test_path = f"data/sae_activations_{model_name}/normal_setting/{description_string}_X_test_sae.pt"
    y_train_path = f"data/sae_activations_{model_name}/normal_setting/{description_string}_y_train.pt"
    y_test_path = f"data/sae_activations_{model_name}/normal_setting/{description_string}_y_test.pt"
    return {
        "save_path": save_path,
        "train_path": train_path,
        "test_path": test_path,
        "y_train_path": y_train_path,
        "y_test_path": y_test_path,
    }


def run_normal_baseline(
    dataset, layer, sae_id, reg_type, binarize=False, model_name="gemma-2-9b"
):
    paths = get_normal_sae_paths(dataset, layer, sae_id, reg_type, binarize, model_name)
    train_path, test_path, y_train_path, y_test_path = (
        paths["train_path"],
        paths["test_path"],
        paths["y_train_path"],
        paths["y_test_path"],
    )

    # Check if all required files exist
    if not all(
        os.path.exists(p) for p in [train_path, test_path, y_train_path, y_test_path]
    ):
        print(
            f"Missing activation files for dataset {dataset}, layer {layer}, SAE {sae_id}"
        )
        return False

    X_train_sae = load_activations(train_path)
    X_test_sae = load_activations(test_path)
    y_train = load_activations(y_train_path)
    y_test = load_activations(y_test_path)

    ks = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]  # For normal setting
    all_metrics = []
    # For now only implemented for classification
    X_train_diff = X_train_sae[y_train == 1].mean(dim=0) - X_train_sae[
        y_train == 0
    ].mean(dim=0)
    sorted_indices = torch.argsort(torch.abs(X_train_diff), descending=True)

    for k in tqdm(ks):
        top_by_average_diff = sorted_indices[:k]
        X_train_filtered = X_train_sae[:, top_by_average_diff]
        X_test_filtered = X_test_sae[:, top_by_average_diff]

        if binarize:
            X_train_filtered = X_train_filtered > 1
            X_test_filtered = X_test_filtered > 1

        if reg_type == "l1":
            metrics = find_best_reg(
                X_train=X_train_filtered,
                y_train=y_train,
                X_test=X_test_filtered,
                y_test=y_test,
                plot=False,
                n_jobs=-1,
                parallel=False,
                penalty="l1",
            )
        else:
            metrics = find_best_reg(
                X_train=X_train_filtered,
                y_train=y_train,
                X_test=X_test_filtered,
                y_test=y_test,
                plot=False,
                n_jobs=-1,
                parallel=False,
                penalty="l2",
            )
        metrics["k"] = k
        metrics["dataset"] = dataset
        metrics["layer"] = layer
        metrics["sae_id"] = sae_id
        metrics["reg_type"] = reg_type
        metrics["binarize"] = binarize
        all_metrics.append(metrics)

    print(f"Saving results to {paths['save_path']}")
    os.makedirs(os.path.dirname(paths["save_path"]), exist_ok=True)
    with open(paths["save_path"], "wb") as f:
        pkl.dump(all_metrics, f)

    return True


def run_normal_baselines(reg_type, model_name, binarize=False, target_sae_id=None):
    layers = get_sae_layers(model_name)
    while True:
        found_missing = False
        random_order_datasets = random.sample(datasets, len(datasets))
        for dataset in random_order_datasets:
            random_order_layers = random.sample(layers, len(layers))
            for layer in random_order_layers:
                if target_sae_id is not None:
                    sae_ids = [target_sae_id]
                else:
                    sae_ids = layer_to_sae_ids(layer, model_name)
                random_order_sae_ids = random.sample(sae_ids, len(sae_ids))
                for sae_id in random_order_sae_ids:
                    paths = get_normal_sae_paths(
                        dataset, layer, sae_id, reg_type, binarize, model_name
                    )
                    if not os.path.exists(paths["save_path"]) and os.path.exists(
                        paths["train_path"]
                    ):
                        found_missing = True
                        print(
                            f"Running probe for dataset {dataset}, layer {layer}, SAE {sae_id}, reg_type {reg_type}"
                        )
                        success = run_normal_baseline(
                            dataset, layer, sae_id, reg_type, binarize, model_name
                        )
                        if not success:
                            continue

        if not found_missing:
            print("All normal probes run. Exiting.")
            break


# Scarcity setting functions
def get_scarcity_sae_paths(
    dataset, layer, sae_id, reg_type, num_train, model_name="gemma-2-9b"
):
    if model_name == "gemma-2-9b":
        width = sae_id.split("/")[1]
        l0 = sae_id.split("/")[2]
        description_string = f"{dataset}_{layer}_{width}_{l0}"
    elif model_name == "llama-3.1-8b":
        description_string = f"{dataset}_{sae_id}"
    elif model_name == "gemma-2-2b":
        name = "_".join(sae_id[2].split("/")[0].split("_")[1:])
        l0 = sae_id[3]
        rounded_l0 = round(float(l0))
        description_string = f"{dataset}_{name}_{rounded_l0}"
    else:
        raise ValueError(f"Invalid model name: {model_name}")

    save_path = f"data/sae_probes_{model_name}/scarcity_setting/{description_string}_{reg_type}_{num_train}.pkl"
    train_path = f"data/sae_activations_{model_name}/scarcity_setting/{description_string}_{num_train}_X_train_sae.pt"
    test_path = f"data/sae_activations_{model_name}/scarcity_setting/{description_string}_{num_train}_X_test_sae.pt"
    y_train_path = f"data/sae_activations_{model_name}/scarcity_setting/{description_string}_{num_train}_y_train.pt"
    y_test_path = f"data/sae_activations_{model_name}/scarcity_setting/{description_string}_{num_train}_y_test.pt"
    return {
        "save_path": save_path,
        "train_path": train_path,
        "test_path": test_path,
        "y_train_path": y_train_path,
        "y_test_path": y_test_path,
    }


def run_scarcity_baseline(
    dataset, layer, sae_id, reg_type, num_train, model_name="gemma-2-9b"
):
    paths = get_scarcity_sae_paths(
        dataset, layer, sae_id, reg_type, num_train, model_name
    )
    train_path, test_path, y_train_path, y_test_path = (
        paths["train_path"],
        paths["test_path"],
        paths["y_train_path"],
        paths["y_test_path"],
    )

    # Check if all required files exist
    if not all(
        os.path.exists(p) for p in [train_path, test_path, y_train_path, y_test_path]
    ):
        print(
            f"Missing activation files for dataset {dataset}, layer {layer}, SAE {sae_id}, num_train {num_train}"
        )
        return False

    X_train_sae = load_activations(train_path)
    X_test_sae = load_activations(test_path)
    y_train = load_activations(y_train_path)
    y_test = load_activations(y_test_path)

    ks = [16, 128]  # For scarcity setting
    all_metrics = []
    X_train_diff = X_train_sae[y_train == 1].mean(dim=0) - X_train_sae[
        y_train == 0
    ].mean(dim=0)
    sorted_indices = torch.argsort(torch.abs(X_train_diff), descending=True)

    for k in tqdm(ks):
        top_by_average_diff = sorted_indices[:k]
        X_train_filtered = X_train_sae[:, top_by_average_diff]
        X_test_filtered = X_test_sae[:, top_by_average_diff]

        if reg_type == "l1":
            metrics = find_best_reg(
                X_train=X_train_filtered,
                y_train=y_train,
                X_test=X_test_filtered,
                y_test=y_test,
                plot=False,
                n_jobs=-1,
                parallel=False,
                penalty="l1",
            )
        else:
            metrics = find_best_reg(
                X_train=X_train_filtered,
                y_train=y_train,
                X_test=X_test_filtered,
                y_test=y_test,
                plot=False,
                n_jobs=-1,
                parallel=False,
                penalty="l2",
            )
        metrics["k"] = k
        metrics["dataset"] = dataset
        metrics["layer"] = layer
        metrics["sae_id"] = sae_id
        metrics["reg_type"] = reg_type
        metrics["num_train"] = num_train
        all_metrics.append(metrics)

    print(f"Saving results to {paths['save_path']}")
    os.makedirs(os.path.dirname(paths["save_path"]), exist_ok=True)
    with open(paths["save_path"], "wb") as f:
        pkl.dump(all_metrics, f)

    return True


def run_scarcity_baselines(reg_type, model_name, target_sae_id=None):
    layers = get_sae_layers(model_name)
    while True:
        found_missing = False
        random_order_datasets = random.sample(datasets, len(datasets))
        for dataset in random_order_datasets:
            random_order_layers = random.sample(layers, len(layers))
            for layer in random_order_layers:
                if target_sae_id is not None:
                    sae_ids = [target_sae_id]
                else:
                    sae_ids = layer_to_sae_ids(layer, model_name)
                random_order_sae_ids = random.sample(sae_ids, len(sae_ids))
                for sae_id in random_order_sae_ids:
                    random_order_train_sizes = random.sample(
                        train_sizes, len(train_sizes)
                    )
                    for num_train in random_order_train_sizes:
                        paths = get_scarcity_sae_paths(
                            dataset, layer, sae_id, reg_type, num_train, model_name
                        )
                        if not os.path.exists(paths["save_path"]) and os.path.exists(
                            paths["train_path"]
                        ):
                            found_missing = True
                            print(
                                f"Running probe for dataset {dataset}, layer {layer}, SAE {sae_id}, reg_type {reg_type}, num_train {num_train}"
                            )
                            success = run_scarcity_baseline(
                                dataset,
                                layer,
                                sae_id,
                                reg_type,
                                num_train,
                                model_name,
                            )
                            if not success:
                                continue

        if not found_missing:
            print("All scarcity probes run. Exiting.")
            break


# Noise setting functions
def get_noise_sae_paths(
    dataset, layer, sae_id, reg_type, corrupt_frac, model_name="gemma-2-9b"
):
    if model_name == "gemma-2-9b":
        width = sae_id.split("/")[1]
        l0 = sae_id.split("/")[2]
        description_string = f"{dataset}_{layer}_{width}_{l0}"
    elif model_name == "llama-3.1-8b":
        description_string = f"{dataset}_{sae_id}"
    elif model_name == "gemma-2-2b":
        name = "_".join(sae_id[2].split("/")[0].split("_")[1:])
        l0 = sae_id[3]
        rounded_l0 = round(float(l0))
        description_string = f"{dataset}_{name}_{rounded_l0}"
    else:
        raise ValueError(f"Invalid model name: {model_name}")

    save_path = f"data/sae_probes_{model_name}/noise_setting/{description_string}_{reg_type}_{corrupt_frac}.pkl"
    train_path = f"data/sae_activations_{model_name}/noise_setting/{description_string}_{corrupt_frac}_X_train_sae.pt"
    test_path = f"data/sae_activations_{model_name}/noise_setting/{description_string}_{corrupt_frac}_X_test_sae.pt"
    y_train_path = f"data/sae_activations_{model_name}/noise_setting/{description_string}_{corrupt_frac}_y_train.pt"
    y_test_path = f"data/sae_activations_{model_name}/noise_setting/{description_string}_{corrupt_frac}_y_test.pt"
    return {
        "save_path": save_path,
        "train_path": train_path,
        "test_path": test_path,
        "y_train_path": y_train_path,
        "y_test_path": y_test_path,
    }


def run_noise_baseline(
    dataset, layer, sae_id, reg_type, corrupt_frac, model_name="gemma-2-9b"
):
    paths = get_noise_sae_paths(
        dataset, layer, sae_id, reg_type, corrupt_frac, model_name
    )
    train_path, test_path, y_train_path, y_test_path = (
        paths["train_path"],
        paths["test_path"],
        paths["y_train_path"],
        paths["y_test_path"],
    )

    # Check if all required files exist
    if not all(
        os.path.exists(p) for p in [train_path, test_path, y_train_path, y_test_path]
    ):
        print(
            f"Missing activation files for dataset {dataset}, layer {layer}, SAE {sae_id}, corrupt_frac {corrupt_frac}"
        )
        return False

    X_train_sae = load_activations(train_path)
    X_test_sae = load_activations(test_path)
    y_train = load_activations(y_train_path)
    y_test = load_activations(y_test_path)

    ks = [16, 128]  # For noise setting
    all_metrics = []
    X_train_diff = X_train_sae[y_train == 1].mean(dim=0) - X_train_sae[
        y_train == 0
    ].mean(dim=0)
    sorted_indices = torch.argsort(torch.abs(X_train_diff), descending=True)

    for k in tqdm(ks):
        top_by_average_diff = sorted_indices[:k]
        X_train_filtered = X_train_sae[:, top_by_average_diff]
        X_test_filtered = X_test_sae[:, top_by_average_diff]

        if reg_type == "l1":
            metrics = find_best_reg(
                X_train=X_train_filtered,
                y_train=y_train,
                X_test=X_test_filtered,
                y_test=y_test,
                plot=False,
                n_jobs=-1,
                parallel=False,
                penalty="l1",
            )
        else:
            metrics = find_best_reg(
                X_train=X_train_filtered,
                y_train=y_train,
                X_test=X_test_filtered,
                y_test=y_test,
                plot=False,
                n_jobs=-1,
                parallel=False,
                penalty="l2",
            )
        metrics["k"] = k
        metrics["dataset"] = dataset
        metrics["layer"] = layer
        metrics["sae_id"] = sae_id
        metrics["reg_type"] = reg_type
        metrics["corrupt_frac"] = corrupt_frac
        all_metrics.append(metrics)

    print(f"Saving results to {paths['save_path']}")
    os.makedirs(os.path.dirname(paths["save_path"]), exist_ok=True)
    with open(paths["save_path"], "wb") as f:
        pkl.dump(all_metrics, f)

    return True


def run_noise_baselines(reg_type, model_name, target_sae_id=None):
    layers = get_sae_layers(model_name)
    while True:
        found_missing = False
        random_order_datasets = random.sample(datasets, len(datasets))
        for dataset in random_order_datasets:
            random_order_layers = random.sample(layers, len(layers))
            for layer in random_order_layers:
                if target_sae_id is not None:
                    sae_ids = [target_sae_id]
                else:
                    sae_ids = layer_to_sae_ids(layer, model_name)
                random_order_sae_ids = random.sample(sae_ids, len(sae_ids))
                for sae_id in random_order_sae_ids:
                    random_order_corrupt_fracs = random.sample(
                        corrupt_fracs, len(corrupt_fracs)
                    )
                    for corrupt_frac in random_order_corrupt_fracs:
                        paths = get_noise_sae_paths(
                            dataset,
                            layer,
                            sae_id,
                            reg_type,
                            corrupt_frac,
                            model_name,
                        )
                        if not os.path.exists(paths["save_path"]) and os.path.exists(
                            paths["train_path"]
                        ):
                            found_missing = True
                            print(
                                f"Running probe for dataset {dataset}, layer {layer}, SAE {sae_id}, reg_type {reg_type}, corrupt_frac {corrupt_frac}"
                            )
                            success = run_noise_baseline(
                                dataset,
                                layer,
                                sae_id,
                                reg_type,
                                corrupt_frac,
                                model_name,
                            )
                            if not success:
                                continue

        if not found_missing:
            print("All noise probes run. Exiting.")
            break


# Imbalance setting functions
def get_imbalance_sae_paths(
    dataset, layer, sae_id, reg_type, frac, model_name="gemma-2-9b"
):
    if model_name == "gemma-2-9b":
        width = sae_id.split("/")[1]
        l0 = sae_id.split("/")[2]
        description_string = f"{dataset}_{layer}_{width}_{l0}"
    elif model_name == "llama-3.1-8b":
        description_string = f"{dataset}_{sae_id}"
    elif model_name == "gemma-2-2b":
        name = "_".join(sae_id[2].split("/")[0].split("_")[1:])
        l0 = sae_id[3]
        rounded_l0 = round(float(l0))
        description_string = f"{dataset}_{name}_{rounded_l0}"
    else:
        raise ValueError(f"Invalid model name: {model_name}")

    save_path = f"data/sae_probes_{model_name}/imbalance_setting/{description_string}_{reg_type}_{frac}.pkl"
    train_path = f"data/sae_activations_{model_name}/imbalance_setting/{description_string}_{frac}_X_train_sae.pt"
    test_path = f"data/sae_activations_{model_name}/imbalance_setting/{description_string}_{frac}_X_test_sae.pt"
    y_train_path = f"data/sae_activations_{model_name}/imbalance_setting/{description_string}_{frac}_y_train.pt"
    y_test_path = f"data/sae_activations_{model_name}/imbalance_setting/{description_string}_{frac}_y_test.pt"
    return {
        "save_path": save_path,
        "train_path": train_path,
        "test_path": test_path,
        "y_train_path": y_train_path,
        "y_test_path": y_test_path,
    }


def run_imbalance_baseline(
    dataset, layer, sae_id, reg_type, frac, model_name="gemma-2-9b"
):
    paths = get_imbalance_sae_paths(dataset, layer, sae_id, reg_type, frac, model_name)
    train_path, test_path, y_train_path, y_test_path = (
        paths["train_path"],
        paths["test_path"],
        paths["y_train_path"],
        paths["y_test_path"],
    )

    # Check if all required files exist
    if not all(
        os.path.exists(p) for p in [train_path, test_path, y_train_path, y_test_path]
    ):
        print(
            f"Missing activation files for dataset {dataset}, layer {layer}, SAE {sae_id}, frac {frac}"
        )
        return False

    X_train_sae = load_activations(train_path)
    X_test_sae = load_activations(test_path)
    y_train = load_activations(y_train_path)
    y_test = load_activations(y_test_path)

    ks = [16, 128]  # For imbalance setting
    all_metrics = []
    X_train_diff = X_train_sae[y_train == 1].mean(dim=0) - X_train_sae[
        y_train == 0
    ].mean(dim=0)
    sorted_indices = torch.argsort(torch.abs(X_train_diff), descending=True)

    for k in tqdm(ks):
        top_by_average_diff = sorted_indices[:k]
        X_train_filtered = X_train_sae[:, top_by_average_diff]
        X_test_filtered = X_test_sae[:, top_by_average_diff]

        if reg_type == "l1":
            metrics = find_best_reg(
                X_train=X_train_filtered,
                y_train=y_train,
                X_test=X_test_filtered,
                y_test=y_test,
                plot=False,
                n_jobs=-1,
                parallel=False,
                penalty="l1",
            )
        else:
            metrics = find_best_reg(
                X_train=X_train_filtered,
                y_train=y_train,
                X_test=X_test_filtered,
                y_test=y_test,
                plot=False,
                n_jobs=-1,
                parallel=False,
                penalty="l2",
            )
        metrics["k"] = k
        metrics["dataset"] = dataset
        metrics["layer"] = layer
        metrics["sae_id"] = sae_id
        metrics["reg_type"] = reg_type
        metrics["frac"] = frac
        all_metrics.append(metrics)

    print(f"Saving results to {paths['save_path']}")
    os.makedirs(os.path.dirname(paths["save_path"]), exist_ok=True)
    with open(paths["save_path"], "wb") as f:
        pkl.dump(all_metrics, f)

    return True


def run_imbalance_baselines(reg_type, model_name, target_sae_id=None):
    layers = get_sae_layers(model_name)
    while True:
        found_missing = False
        random_order_datasets = random.sample(datasets, len(datasets))
        for dataset in random_order_datasets:
            random_order_layers = random.sample(layers, len(layers))
            for layer in random_order_layers:
                if target_sae_id is not None:
                    sae_ids = [target_sae_id]
                else:
                    sae_ids = layer_to_sae_ids(layer, model_name)
                random_order_sae_ids = random.sample(sae_ids, len(sae_ids))
                for sae_id in random_order_sae_ids:
                    random_order_fracs = random.sample(fracs, len(fracs))
                    for frac in random_order_fracs:
                        paths = get_imbalance_sae_paths(
                            dataset, layer, sae_id, reg_type, frac, model_name
                        )
                        if not os.path.exists(paths["save_path"]) and os.path.exists(
                            paths["train_path"]
                        ):
                            found_missing = True
                            print(
                                f"Running probe for dataset {dataset}, layer {layer}, SAE {sae_id}, reg_type {reg_type}, frac {frac}"
                            )
                            success = run_imbalance_baseline(
                                dataset, layer, sae_id, reg_type, frac, model_name
                            )
                            if not success:
                                continue

        if not found_missing:
            print("All imbalance probes run. Exiting.")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run SAE probe baselines for specified settings."
    )
    parser.add_argument(
        "--reg_type",
        type=str,
        required=True,
        choices=["l1", "l2"],
        help="Regularization type (l1 or l2).",
    )
    parser.add_argument(
        "--setting",
        type=str,
        required=True,
        choices=["normal", "scarcity", "noise", "imbalance"],
        help="Experiment setting to run.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gemma-2-9b",
        help="Model name (e.g., gemma-2-9b, llama-3.1-8b).",
    )
    parser.add_argument(
        "--binarize",
        action="store_true",
        help="Binarize SAE features for normal setting.",
    )
    parser.add_argument(
        "--target_sae_id",
        type=str,
        default=None,
        help="Specific SAE ID to run (optional). This will override the default behavior of running all SAEs for the specified layer and model.",
    )

    args = parser.parse_args()

    if args.setting == "normal":
        run_normal_baselines(
            args.reg_type, args.model_name, args.binarize, args.target_sae_id
        )
    elif args.setting == "scarcity":
        run_scarcity_baselines(args.reg_type, args.model_name, args.target_sae_id)
    elif args.setting == "noise":
        run_noise_baselines(args.reg_type, args.model_name, args.target_sae_id)
    elif args.setting == "imbalance":
        run_imbalance_baselines(args.reg_type, args.model_name, args.target_sae_id)
