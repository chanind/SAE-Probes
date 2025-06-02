import json
import warnings
from dataclasses import asdict
from pathlib import Path

import torch
from sae_lens import SAE
from sklearn.exceptions import ConvergenceWarning
from tqdm import tqdm

from sae_probes.constants import (
    DEFAULT_MODEL_CACHE_PATH,
    DEFAULT_SAE_CACHE_PATH,
    RegType,
    Setting,
)
from sae_probes.generate_sae_activations import generate_sae_activations
from sae_probes.utils_data import (
    get_class_imbalance,
    get_corrupt_frac,
    get_dataset_sizes,
    get_numbered_binary_tags,
    get_training_sizes,
)
from sae_probes.utils_training import find_best_reg

warnings.simplefilter("ignore", category=ConvergenceWarning)


# Constants and datasets
DATASET_SIZES = get_dataset_sizes()
DATASETS = get_numbered_binary_tags()
TRAIN_SIZES = get_training_sizes()
CORRUPT_FRACS = get_corrupt_frac()
FRACS = get_class_imbalance()


def load_activations(path):
    return torch.load(path, weights_only=True).to_dense().float()


# Normal setting functions
def get_save_metrics_path(
    dataset: str,
    layer: int,
    reg_type: RegType,
    model_name: str,
    binarize: bool = False,
    setting: Setting = "normal",
    num_train: int | None = None,
    corrupt_frac: float | None = None,
    frac: float | None = None,
    sae_cache_path: str | Path = DEFAULT_SAE_CACHE_PATH,
):
    description_string = f"{dataset}_{layer}"

    if setting == "normal":
        extra_string = "_"
        assert num_train is None
        assert corrupt_frac is None
        assert frac is None
    elif setting == "scarcity":
        extra_string = f"{num_train}"
        assert num_train is not None
        assert corrupt_frac is None
        assert frac is None
    elif setting == "imbalance":
        extra_string = f"frac{frac}"
        assert frac is not None
        assert num_train is None
        assert corrupt_frac is None

    reg_type_str: str = reg_type
    if binarize:
        reg_type_str = f"{reg_type_str}_binarized"

    if extra_string != "_":
        if not extra_string.endswith("_"):
            extra_string = extra_string + "_"
        if not extra_string.startswith("_"):
            extra_string = "_" + extra_string

    extra_save_string = extra_string
    save_setting = setting

    save_path = (
        Path(sae_cache_path)
        / f"sae_probes_{model_name}/{save_setting}_setting/{description_string}{extra_save_string}{reg_type_str}.json"
    )
    return save_path


def get_sorted_indices(X_train_sae, y_train):
    X_train_diff = X_train_sae[y_train == 1].mean(dim=0) - X_train_sae[
        y_train == 0
    ].mean(dim=0)
    sorted_indices = torch.argsort(torch.abs(X_train_diff), descending=True)
    return sorted_indices


def get_sorted_indices_new(X_train_sae, y_train):
    col_sums = X_train_sae.sum(dim=0)
    col_nonzero_counts = (X_train_sae != 0).sum(dim=0)
    col_means = col_sums / (col_nonzero_counts + 1e-6)

    # Divide each col by the average of the col
    X_train_sae_normalized = X_train_sae / (col_means + 1e-6)
    X_train_diff = X_train_sae_normalized[y_train == 1].mean(
        dim=0
    ) - X_train_sae_normalized[y_train == 0].mean(dim=0)
    sorted_indices = torch.argsort(torch.abs(X_train_diff), descending=True)
    return sorted_indices


def run_sae_eval(
    sae: SAE,
    dataset: str,
    layer: int,
    reg_type: RegType,
    setting: Setting,
    model_name: str,
    binarize: bool = False,
    num_train: int | None = None,
    corrupt_frac: float | None = None,
    frac: float | None = None,
    device: str = "cuda",
    batch_size: int = 128,
    ks: list[int] | None = None,
    sae_cache_path: str | Path = DEFAULT_SAE_CACHE_PATH,
    model_cache_path: str | Path = DEFAULT_MODEL_CACHE_PATH,
):
    activations = generate_sae_activations(
        sae=sae,
        setting=setting,
        dataset=dataset,
        layer=layer,
        model_name=model_name,
        device=device,
        num_train=num_train,
        frac=frac,
        batch_size=batch_size,
        model_cache_path=model_cache_path,
    )

    X_train_sae = activations.X_train
    X_test_sae = activations.X_test
    y_train = activations.y_train
    y_test = activations.y_test

    # Set k values based on setting
    if ks is None:
        if setting == "normal":
            ks = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        else:
            ks = [16, 128]

    all_metrics = []
    sorted_indices = get_sorted_indices_new(X_train_sae, y_train)

    for k in tqdm(ks):
        top_by_average_diff = sorted_indices[:k]
        X_train_filtered = X_train_sae[:, top_by_average_diff]
        X_test_filtered = X_test_sae[:, top_by_average_diff]

        if binarize and setting == "normal":
            X_train_filtered = X_train_filtered > 1
            X_test_filtered = X_test_filtered > 1

        results = find_best_reg(
            X_train=X_train_filtered,
            y_train=y_train,
            X_test=X_test_filtered,
            y_test=y_test,
            plot=False,
            n_jobs=-1,
            parallel=False,
            penalty=reg_type,
        )
        metrics = asdict(results.metrics)

        # Add metadata to metrics
        metrics.update(
            {
                "k": k,
                "dataset": dataset,
                "layer": layer,
                "reg_type": reg_type,
                "binarize": binarize,
            }
        )

        # Add setting-specific metadata
        if setting == "scarcity":
            metrics["num_train"] = num_train
        elif setting == "imbalance":
            metrics["frac"] = frac

        all_metrics.append(metrics)

    save_path = get_save_metrics_path(
        dataset=dataset,
        layer=layer,
        reg_type=reg_type,
        binarize=binarize,
        model_name=model_name,
        setting=setting,
        num_train=num_train,
        corrupt_frac=corrupt_frac,
        frac=frac,
        sae_cache_path=sae_cache_path,
    )

    print(f"Saving results to {save_path}")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(all_metrics, f, indent=4, ensure_ascii=False)

    return True


def run_sae_evals(
    sae: SAE,
    model_name: str,
    layer: int,
    reg_type: RegType,
    setting: Setting,
    ks: list[int] | None = None,
    binarize: bool = False,
    sae_cache_path: str | Path = DEFAULT_SAE_CACHE_PATH,
    model_cache_path: str | Path = DEFAULT_MODEL_CACHE_PATH,
):
    for dataset in DATASETS:
        # Handle different settings
        if setting == "normal":
            save_path = get_save_metrics_path(
                dataset=dataset,
                layer=layer,
                reg_type=reg_type,
                binarize=binarize,
                model_name=model_name,
                setting=setting,
                sae_cache_path=sae_cache_path,
            )
            if not save_path.exists():
                print(
                    f"Running probe for dataset {dataset}, layer {layer}, reg_type {reg_type}, setting {setting}"
                )
                success = run_sae_eval(
                    sae,
                    dataset,
                    layer,
                    reg_type,
                    setting,
                    model_name,
                    binarize,
                    ks=ks,
                    sae_cache_path=sae_cache_path,
                    model_cache_path=model_cache_path,
                )
                assert success
        elif setting == "scarcity":
            for num_train in TRAIN_SIZES:
                if num_train > DATASET_SIZES[dataset] - 100:
                    continue
                save_path = get_save_metrics_path(
                    dataset=dataset,
                    layer=layer,
                    reg_type=reg_type,
                    binarize=binarize,
                    model_name=model_name,
                    setting=setting,
                    num_train=num_train,
                    sae_cache_path=sae_cache_path,
                )
                if not save_path.exists():
                    print(
                        f"Running probe for dataset {dataset}, layer {layer}, "
                        f"reg_type {reg_type}, num_train {num_train}, setting {setting}"
                    )
                    success = run_sae_eval(
                        sae,
                        dataset,
                        layer,
                        reg_type,
                        setting,
                        model_name,
                        num_train=num_train,
                        ks=ks,
                        sae_cache_path=sae_cache_path,
                        model_cache_path=model_cache_path,
                    )
                    assert success
        elif setting == "imbalance":
            for frac in FRACS:
                save_path = get_save_metrics_path(
                    dataset=dataset,
                    layer=layer,
                    reg_type=reg_type,
                    binarize=binarize,
                    model_name=model_name,
                    setting=setting,
                    frac=frac,
                    sae_cache_path=sae_cache_path,
                )
                if not save_path.exists():
                    print(
                        f"Running probe for dataset {dataset}, layer {layer}, "
                        f"reg_type {reg_type}, frac {frac}, setting {setting}"
                    )
                    success = run_sae_eval(
                        sae,
                        dataset,
                        layer,
                        reg_type=reg_type,
                        setting=setting,
                        model_name=model_name,
                        frac=frac,
                        ks=ks,
                        sae_cache_path=sae_cache_path,
                        model_cache_path=model_cache_path,
                    )
                    assert success
        else:
            raise ValueError(f"Invalid setting: {setting}")
