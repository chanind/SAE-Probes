import os
import warnings
from pathlib import Path
from typing import Literal

import torch
from sae_lens import SAE
from sklearn.exceptions import ConvergenceWarning

from sae_probes.constants import DEFAULT_MODEL_CACHE_PATH, DEFAULT_SAE_CACHE_PATH
from sae_probes.utils_data import (
    get_class_imbalance,
    get_classimabalance_num_train,
    get_dataset_sizes,
    get_numbered_binary_tags,
    get_training_sizes,
    get_xy_traintest,
    get_xy_traintest_specify,
)

warnings.simplefilter("ignore", category=ConvergenceWarning)


# Common variables
dataset_sizes = get_dataset_sizes()
datasets = get_numbered_binary_tags()


# Helper functions for all settings
def save_activations(path, activation):
    """Save activations in sparse format to save space"""
    sparse_tensor = activation.to_sparse()
    torch.save(sparse_tensor, path)


def load_activations(path):
    """Load activations from sparse format"""
    return torch.load(path, weights_only=True).to_dense().float()


# Normal setting functions
def get_sae_paths_normal(
    model_name: str,
    dataset: str,
    layer: int,
    reg_type: str,
    binarize: bool = False,
    sae_cache_path: str | Path = DEFAULT_SAE_CACHE_PATH,
):
    """Get paths for normal setting"""
    os.makedirs(
        Path(sae_cache_path) / f"sae_probes_{model_name}/normal_setting",
        exist_ok=True,
    )
    os.makedirs(
        Path(sae_cache_path) / f"sae_activations_{model_name}/normal_setting",
        exist_ok=True,
    )

    description_string = f"{dataset}_{layer}"

    if binarize:
        reg_type += "_binarized"

    save_path = (
        Path(sae_cache_path)
        / f"sae_probes_{model_name}/normal_setting/{description_string}_{reg_type}.pkl"
    )
    train_path = (
        Path(sae_cache_path)
        / f"sae_activations_{model_name}/normal_setting/{description_string}_X_train_sae.pt"
    )
    test_path = (
        Path(sae_cache_path)
        / f"sae_activations_{model_name}/normal_setting/{description_string}_X_test_sae.pt"
    )
    y_train_path = (
        Path(sae_cache_path)
        / f"sae_activations_{model_name}/normal_setting/{description_string}_y_train.pt"
    )
    y_test_path = (
        Path(sae_cache_path)
        / f"sae_activations_{model_name}/normal_setting/{description_string}_y_test.pt"
    )
    return {
        "save_path": save_path,
        "train_path": train_path,
        "test_path": test_path,
        "y_train_path": y_train_path,
        "y_test_path": y_test_path,
    }


@torch.inference_mode()
def save_with_sae_normal(
    sae: SAE,
    layer: int,
    model_name: str,
    device: str,
    reg_type: str,
    binarize: bool,
    sae_cache_path: str | Path = DEFAULT_SAE_CACHE_PATH,
    model_cache_path: str | Path = DEFAULT_MODEL_CACHE_PATH,
):
    """Generate and save SAE activations for normal setting"""
    for dataset in datasets:
        paths = get_sae_paths_normal(
            dataset=dataset,
            layer=layer,
            reg_type=reg_type,
            binarize=binarize,
            model_name=model_name,
            sae_cache_path=sae_cache_path,
        )
        train_path, test_path, y_train_path, y_test_path = (
            paths["train_path"],
            paths["test_path"],
            paths["y_train_path"],
            paths["y_test_path"],
        )

        all_paths_exist = all(
            [
                os.path.exists(train_path),
                os.path.exists(test_path),
                os.path.exists(y_train_path),
                os.path.exists(y_test_path),
            ]
        )
        if all_paths_exist:
            continue

        size = dataset_sizes[dataset]
        num_train = min(size - 100, 1024)
        X_train, y_train, X_test, y_test = get_xy_traintest(
            num_train,
            dataset,
            layer,
            model_name=model_name,
            model_cache_path=model_cache_path,
        )

        batch_size = 128
        X_train_sae = []
        for i in range(0, len(X_train), batch_size):
            batch = X_train[i : i + batch_size].to(device)
            X_train_sae.append(sae.encode(batch).cpu())
        X_train_sae = torch.cat(X_train_sae)

        X_test_sae = []
        for i in range(0, len(X_test), batch_size):
            batch = X_test[i : i + batch_size].to(device)
            X_test_sae.append(sae.encode(batch).cpu())
        X_test_sae = torch.cat(X_test_sae)

        save_activations(train_path, X_train_sae)
        save_activations(test_path, X_test_sae)
        save_activations(y_train_path, torch.tensor(y_train))
        save_activations(y_test_path, torch.tensor(y_test))


# Data scarcity setting functions
def get_sae_paths_scarcity(
    dataset: str,
    layer: int,
    reg_type: str,
    num_train: int,
    model_name: str,
    sae_cache_path: str | Path = DEFAULT_SAE_CACHE_PATH,
):
    """Get paths for data scarcity setting"""
    os.makedirs(
        Path(sae_cache_path) / f"sae_probes_{model_name}/scarcity_setting",
        exist_ok=True,
    )
    os.makedirs(
        Path(sae_cache_path) / f"sae_activations_{model_name}/scarcity_setting",
        exist_ok=True,
    )

    description_string = f"{dataset}_{layer}"

    save_path = (
        Path(sae_cache_path)
        / f"sae_probes_{model_name}/scarcity_setting/{description_string}_{reg_type}_{num_train}.pkl"
    )
    train_path = (
        Path(sae_cache_path)
        / f"sae_activations_{model_name}/scarcity_setting/{description_string}_{num_train}_X_train_sae.pt"
    )
    test_path = (
        Path(sae_cache_path)
        / f"sae_activations_{model_name}/scarcity_setting/{description_string}_{num_train}_X_test_sae.pt"
    )
    y_train_path = (
        Path(sae_cache_path)
        / f"sae_activations_{model_name}/scarcity_setting/{description_string}_{num_train}_y_train.pt"
    )
    y_test_path = (
        Path(sae_cache_path)
        / f"sae_activations_{model_name}/scarcity_setting/{description_string}_{num_train}_y_test.pt"
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    os.makedirs(os.path.dirname(train_path), exist_ok=True)

    return {
        "save_path": save_path,
        "train_path": train_path,
        "test_path": test_path,
        "y_train_path": y_train_path,
        "y_test_path": y_test_path,
    }


@torch.inference_mode()
def save_with_sae_scarcity(
    sae: SAE,
    layer: int,
    model_name: str,
    device: str,
    reg_type: str,
    sae_cache_path: str | Path = DEFAULT_SAE_CACHE_PATH,
    model_cache_path: str | Path = DEFAULT_MODEL_CACHE_PATH,
):
    """Generate and save SAE activations for data scarcity setting"""
    train_sizes = get_training_sizes()

    for dataset in datasets:
        for num_train in train_sizes:
            if num_train > dataset_sizes[dataset] - 100:
                continue

            paths = get_sae_paths_scarcity(
                dataset=dataset,
                layer=layer,
                reg_type=reg_type,
                num_train=num_train,
                model_name=model_name,
                sae_cache_path=sae_cache_path,
            )
            train_path, test_path, y_train_path, y_test_path = (
                paths["train_path"],
                paths["test_path"],
                paths["y_train_path"],
                paths["y_test_path"],
            )

            if all(
                os.path.exists(p)
                for p in [train_path, test_path, y_train_path, y_test_path]
            ):
                continue

            X_train, y_train, X_test, y_test = get_xy_traintest(
                num_train,
                dataset,
                layer,
                model_name=model_name,
                model_cache_path=model_cache_path,
            )

            batch_size = 128
            X_train_sae = []
            for i in range(0, len(X_train), batch_size):
                batch = X_train[i : i + batch_size].to(device)
                X_train_sae.append(sae.encode(batch).cpu())
            X_train_sae = torch.cat(X_train_sae)

            X_test_sae = []
            for i in range(0, len(X_test), batch_size):
                batch = X_test[i : i + batch_size].to(device)
                X_test_sae.append(sae.encode(batch).cpu())
            X_test_sae = torch.cat(X_test_sae)

            save_activations(train_path, X_train_sae)
            save_activations(test_path, X_test_sae)
            save_activations(y_train_path, torch.tensor(y_train))
            save_activations(y_test_path, torch.tensor(y_test))


# Class imbalance setting functions
def get_sae_paths_imbalance(
    dataset: str,
    layer: int,
    reg_type: str,
    frac: float,
    model_name: str,
    sae_cache_path: str | Path = DEFAULT_SAE_CACHE_PATH,
):
    """Get paths for class imbalance setting"""
    os.makedirs(
        Path(sae_cache_path) / f"sae_probes_{model_name}/class_imbalance",
        exist_ok=True,
    )
    os.makedirs(
        Path(sae_cache_path) / f"sae_activations_{model_name}/class_imbalance",
        exist_ok=True,
    )

    description_string = f"{dataset}_{layer}"

    save_path = (
        Path(sae_cache_path)
        / f"sae_probes_{model_name}/class_imbalance/{description_string}_{reg_type}_frac{frac}.pkl"
    )
    train_path = (
        Path(sae_cache_path)
        / f"sae_activations_{model_name}/class_imbalance/{description_string}_frac{frac}_X_train_sae.pt"
    )
    test_path = (
        Path(sae_cache_path)
        / f"sae_activations_{model_name}/class_imbalance/{description_string}_frac{frac}_X_test_sae.pt"
    )
    y_train_path = (
        Path(sae_cache_path)
        / f"sae_activations_{model_name}/class_imbalance/{description_string}_frac{frac}_y_train.pt"
    )
    y_test_path = (
        Path(sae_cache_path)
        / f"sae_activations_{model_name}/class_imbalance/{description_string}_frac{frac}_y_test.pt"
    )
    return {
        "save_path": save_path,
        "train_path": train_path,
        "test_path": test_path,
        "y_train_path": y_train_path,
        "y_test_path": y_test_path,
    }


@torch.inference_mode()
def save_with_sae_imbalance(
    sae: SAE,
    layer: int,
    model_name: str,
    device: str,
    reg_type: str,
    sae_cache_path: str | Path = DEFAULT_SAE_CACHE_PATH,
    model_cache_path: str | Path = DEFAULT_MODEL_CACHE_PATH,
):
    """Generate and save SAE activations for class imbalance setting"""
    fracs = get_class_imbalance()

    for dataset in datasets:
        for frac in fracs:
            paths = get_sae_paths_imbalance(
                dataset=dataset,
                layer=layer,
                reg_type=reg_type,
                frac=frac,
                model_name=model_name,
                sae_cache_path=sae_cache_path,
            )
            train_path, test_path, y_train_path, y_test_path = (
                paths["train_path"],
                paths["test_path"],
                paths["y_train_path"],
                paths["y_test_path"],
            )

            if os.path.exists(train_path):
                continue

            num_train, num_test = get_classimabalance_num_train(dataset)
            X_train, y_train, X_test, y_test = get_xy_traintest_specify(
                num_train,
                dataset,
                layer,
                pos_ratio=frac,
                model_name=model_name,
                num_test=num_test,
                model_cache_path=model_cache_path,
            )

            batch_size = 128
            X_train_sae = []
            for i in range(0, len(X_train), batch_size):
                batch = X_train[i : i + batch_size].to(device)
                X_train_sae.append(sae.encode(batch).cpu())
            X_train_sae = torch.cat(X_train_sae)

            X_test_sae = []
            for i in range(0, len(X_test), batch_size):
                batch = X_test[i : i + batch_size].to(device)
                X_test_sae.append(sae.encode(batch).cpu())
            X_test_sae = torch.cat(X_test_sae)

            save_activations(train_path, X_train_sae)
            save_activations(test_path, X_test_sae)
            save_activations(y_train_path, torch.tensor(y_train))
            save_activations(y_test_path, torch.tensor(y_test))


# Process SAEs for a specific model and setting
@torch.inference_mode()
def process_model_setting(
    sae: SAE,
    model_name: str,
    layer: int,
    setting: Literal["normal", "scarcity", "imbalance"],
    device: str,
    reg_type: str,
    binarize: bool,
    sae_cache_path: str | Path = DEFAULT_SAE_CACHE_PATH,
    model_cache_path: str | Path = DEFAULT_MODEL_CACHE_PATH,
):
    print(f"Running SAE activation generation for {model_name} in {setting} setting")

    # Check if we need to generate activations for this SAE
    missing_data = False

    if setting == "normal":
        # Check normal setting
        for dataset in datasets:
            paths = get_sae_paths_normal(
                dataset=dataset,
                layer=layer,
                reg_type=reg_type,
                binarize=binarize,
                model_name=model_name,
                sae_cache_path=sae_cache_path,
            )
            if not all(
                os.path.exists(p)
                for p in [
                    paths["train_path"],
                    paths["test_path"],
                    paths["y_train_path"],
                    paths["y_test_path"],
                ]
            ):
                print(f"Missing data for dataset {dataset}")
                missing_data = True
                break

        if missing_data:
            print(f"Generating SAE data for layer {layer}")
            save_with_sae_normal(
                sae=sae,
                layer=layer,
                model_name=model_name,
                device=device,
                reg_type=reg_type,
                binarize=binarize,
                sae_cache_path=sae_cache_path,
                model_cache_path=model_cache_path,
            )

    elif setting == "scarcity":
        # Check data scarcity setting
        train_sizes = get_training_sizes()
        for dataset in datasets:
            for num_train in train_sizes:
                if num_train > dataset_sizes[dataset] - 100:
                    continue
                paths = get_sae_paths_scarcity(
                    dataset=dataset,
                    layer=layer,
                    reg_type=reg_type,
                    num_train=num_train,
                    model_name=model_name,
                    sae_cache_path=sae_cache_path,
                )
                if not all(
                    os.path.exists(p)
                    for p in [
                        paths["train_path"],
                        paths["test_path"],
                        paths["y_train_path"],
                        paths["y_test_path"],
                    ]
                ):
                    print(f"Missing data for dataset {dataset}, num_train {num_train}")
                    missing_data = True
                    break
            if missing_data:
                break

        if missing_data:
            print(f"Generating SAE data for layer {layer}")
            save_with_sae_scarcity(
                sae=sae,
                layer=layer,
                model_name=model_name,
                device=device,
                reg_type=reg_type,
                sae_cache_path=sae_cache_path,
                model_cache_path=model_cache_path,
            )

    elif setting == "imbalance":
        # Check class imbalance setting
        fracs = get_class_imbalance()
        for dataset in datasets:
            for frac in fracs:
                paths = get_sae_paths_imbalance(
                    dataset=dataset,
                    layer=layer,
                    reg_type=reg_type,
                    frac=frac,
                    model_name=model_name,
                    sae_cache_path=sae_cache_path,
                )
                if not all(
                    os.path.exists(p)
                    for p in [
                        paths["train_path"],
                        paths["test_path"],
                        paths["y_train_path"],
                        paths["y_test_path"],
                    ]
                ):
                    print(f"Missing data for dataset {dataset}, frac {frac}")
                    missing_data = True
                    break
            if missing_data:
                break

        if missing_data:
            print(f"Generating SAE data for layer {layer}")
            save_with_sae_imbalance(
                sae=sae,
                layer=layer,
                model_name=model_name,
                device=device,
                reg_type=reg_type,
                sae_cache_path=sae_cache_path,
                model_cache_path=model_cache_path,
            )
    else:
        raise ValueError(f"Invalid setting: {setting}")
