import os
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from sae_probes.constants import DEFAULT_RESULTS_PATH

from .generate_model_activations import ensure_dataset_activations
from .utils_data import (
    corrupt_ytrain,
    get_class_imbalance,
    get_classimabalance_num_train,
    get_corrupt_frac,
    get_dataset_sizes,
    get_datasets,
    get_numbered_binary_tags,
    get_training_sizes,
    get_xy_traintest,
    get_xy_traintest_specify,
)
from .utils_tmp import resolve_model_cache_path
from .utils_training import (
    BestClassifierResults,
    find_best_knn,
    find_best_mlp,
    find_best_pcareg,
    find_best_reg,
    find_best_xgboost,
)

DATASET_SIZES = get_dataset_sizes()
DATASETS = get_numbered_binary_tags()
Method = Literal["logreg", "pca", "knn", "xgboost", "mlp"]
METHODS: dict[Method, Callable[[Any, Any, Any, Any], BestClassifierResults]] = {
    "logreg": find_best_reg,
    "pca": find_best_pcareg,
    "knn": find_best_knn,
    "xgboost": find_best_xgboost,
    "mlp": find_best_mlp,
}
DEFAULT_METHODS: tuple[Method, ...] = ("logreg",)


"""
FUNCTIONS FOR STANDARD CONDITIONS 
"""


def run_baseline_dataset_layer(
    hook_name: str,
    numbered_dataset: str,
    method_name: Method,
    model_name: str,
    model_cache_path: str | Path,
    results_path: str | Path = DEFAULT_RESULTS_PATH,
):
    safe_hook = hook_name.replace("/", "-")
    base_path = f"baseline_results_{model_name}/normal/allruns/{safe_hook}_{numbered_dataset}_{method_name}"
    classifier_savepath = Path(results_path) / f"{base_path}_classifier.pt"
    metrics_savepath = Path(results_path) / f"{base_path}.csv"
    os.makedirs(os.path.dirname(metrics_savepath), exist_ok=True)
    if os.path.exists(metrics_savepath):
        return None
    size = DATASET_SIZES[numbered_dataset]
    num_train = min(size - 100, 1024)
    X_train, y_train, X_test, y_test = get_xy_traintest(
        num_train,
        numbered_dataset,
        hook_name,
        model_name=model_name,
        model_cache_path=model_cache_path,
    )

    # Run method and get metrics
    method = METHODS[method_name]
    results = method(X_train, y_train, X_test, y_test)

    # Create row with dataset and method metrics and save to csv
    row = {"dataset": numbered_dataset, "method": method_name}
    for metric_name, metric_value in asdict(results.metrics).items():
        row[f"{metric_name}"] = metric_value
    pd.DataFrame([row]).to_csv(metrics_savepath, index=False)
    torch.save(
        {"classifier": results.classifier, "scaler": results.scaler},
        classifier_savepath,
    )
    return True


def run_all_baseline_normal(
    model_name: str,
    hook_name: str,
    results_path: str | Path = DEFAULT_RESULTS_PATH,
    model_cache_path: str | Path | None = None,
    methods: Sequence[Method] = DEFAULT_METHODS,
):
    with resolve_model_cache_path(model_cache_path) as resolved_cache_path:
        # Ensure all activations exist
        ensure_dataset_activations(
            model_name=model_name,
            dataset_short_names=DATASETS,
            hook_names=[hook_name],
            model_cache_path=resolved_cache_path,
            device="cpu",
        )

        shuffled_datasets = get_datasets(
            model_name, hook_name=hook_name, model_cache_path=resolved_cache_path
        ).copy()
        np.random.shuffle(shuffled_datasets)
        for method_name in tqdm(methods, desc="Methods", position=0):
            for dataset in tqdm(
                shuffled_datasets,
                desc=f"{method_name} Datasets",
                position=1,
                leave=False,
            ):
                run_baseline_dataset_layer(
                    hook_name,
                    dataset,
                    method_name,
                    model_name=model_name,
                    results_path=results_path,
                    model_cache_path=resolved_cache_path,
                )


"""
FUNCTIONS FOR DATA SCARCITY CONDITION
"""


def run_baseline_scarcity(
    num_train: int,
    numbered_dataset: str,
    method_name: Method,
    model_name: str,
    hook_name: str,
    model_cache_path: str | Path,
    results_path: str | Path = DEFAULT_RESULTS_PATH,
):
    safe_hook = hook_name.replace("/", "-")
    base_path = f"baseline_results_{model_name}/scarcity/allruns/{safe_hook}_{numbered_dataset}_{method_name}_numtrain{num_train}"
    metrics_savepath = Path(results_path) / f"{base_path}.csv"
    classifier_savepath = Path(results_path) / f"{base_path}_classifier.pt"
    os.makedirs(os.path.dirname(metrics_savepath), exist_ok=True)
    if os.path.exists(metrics_savepath):
        return None
    size = DATASET_SIZES[numbered_dataset]
    if num_train > size - 100:
        # we dont have enough test examples
        return
    X_train, y_train, X_test, y_test = get_xy_traintest(
        num_train,
        numbered_dataset,
        hook_name,
        model_name=model_name,
        model_cache_path=model_cache_path,
    )
    # Run method and get metrics
    method = METHODS[method_name]
    results = method(X_train, y_train, X_test, y_test)
    # Create row with dataset and method metrics and save to csv
    row = {"dataset": numbered_dataset, "method": method_name, "num_train": num_train}
    for metric_name, metric_value in asdict(results.metrics).items():
        row[f"{metric_name}"] = metric_value
    pd.DataFrame([row]).to_csv(metrics_savepath, index=False)
    torch.save(
        {"classifier": results.classifier, "scaler": results.scaler},
        classifier_savepath,
    )
    return True


def run_all_baseline_scarcity(
    model_name: str,
    hook_name: str,
    results_path: str | Path = DEFAULT_RESULTS_PATH,
    model_cache_path: str | Path | None = None,
    methods: Sequence[Method] = DEFAULT_METHODS,
):
    with resolve_model_cache_path(model_cache_path) as resolved_cache_path:
        ensure_dataset_activations(
            model_name=model_name,
            dataset_short_names=DATASETS,
            hook_names=[hook_name],
            model_cache_path=resolved_cache_path,
            device="cpu",
        )

        shuffled_datasets = get_datasets(
            model_name, hook_name=hook_name, model_cache_path=resolved_cache_path
        ).copy()
        np.random.shuffle(shuffled_datasets)
        train_sizes = get_training_sizes()
        for method_name in tqdm(methods, desc="Methods", position=0):
            for train in tqdm(
                train_sizes, desc=f"{method_name} Train Sizes", position=1, leave=False
            ):
                for dataset in tqdm(
                    shuffled_datasets,
                    desc=f"{method_name} ({train}) Datasets",
                    position=2,
                    leave=False,
                ):
                    run_baseline_scarcity(
                        train,
                        dataset,
                        method_name,
                        model_name=model_name,
                        hook_name=hook_name,
                        results_path=results_path,
                        model_cache_path=resolved_cache_path,
                    )


"""
FUNCTIONS FOR CLASS IMBALANCE CONDITION
"""


def run_baseline_class_imbalance(
    dataset_frac: float,
    numbered_dataset: str,
    method_name: Method,
    model_name: str,
    hook_name: str,
    model_cache_path: str | Path,
    results_path: str | Path = DEFAULT_RESULTS_PATH,
):
    assert 0 < dataset_frac < 1
    dataset_frac = round(dataset_frac * 20) / 20
    safe_hook = hook_name.replace("/", "-")
    base_path = f"baseline_results_{model_name}/imbalance/allruns/{safe_hook}_{numbered_dataset}_{method_name}_frac{dataset_frac}"
    classifier_savepath = Path(results_path) / f"{base_path}_classifier.pt"
    metrics_savepath = Path(results_path) / f"{base_path}.csv"
    os.makedirs(os.path.dirname(metrics_savepath), exist_ok=True)
    if os.path.exists(metrics_savepath):
        return None
    num_train, num_test = get_classimabalance_num_train(numbered_dataset)
    X_train, y_train, X_test, y_test = get_xy_traintest_specify(
        num_train,
        numbered_dataset,
        hook_name,
        pos_ratio=dataset_frac,
        model_name=model_name,
        num_test=num_test,
        model_cache_path=model_cache_path,
    )
    # Run method and get metrics
    method = METHODS[method_name]
    results = method(X_train, y_train, X_test, y_test)
    torch.save(
        {"classifier": results.classifier, "scaler": results.scaler},
        classifier_savepath,
    )
    # Create row with dataset and method metrics and save to csv
    row = {
        "dataset": numbered_dataset,
        "method": method_name,
        "ratio": dataset_frac,
        "num_train": num_train,
    }
    for metric_name, metric_value in asdict(results.metrics).items():
        row[f"{metric_name}"] = metric_value
    pd.DataFrame([row]).to_csv(metrics_savepath, index=False)
    return True


def run_all_baseline_class_imbalance(
    model_name: str,
    hook_name: str,
    results_path: str | Path = DEFAULT_RESULTS_PATH,
    model_cache_path: str | Path | None = None,
    methods: Sequence[Method] = DEFAULT_METHODS,
):
    with resolve_model_cache_path(model_cache_path) as resolved_cache_path:
        ensure_dataset_activations(
            model_name=model_name,
            dataset_short_names=DATASETS,
            hook_names=[hook_name],
            model_cache_path=resolved_cache_path,
            device="cpu",
        )

        shuffled_datasets = get_datasets(
            model_name, hook_name=hook_name, model_cache_path=resolved_cache_path
        ).copy()
        np.random.shuffle(shuffled_datasets)
        fracs = get_class_imbalance()
        for method_name in tqdm(methods, desc="Methods", position=0):
            for frac in tqdm(
                fracs, desc=f"{method_name} Fractions", position=1, leave=False
            ):
                for dataset in tqdm(
                    shuffled_datasets,
                    desc=f"{method_name} (frac {frac:.2f}) Datasets",
                    position=2,
                    leave=False,
                ):
                    run_baseline_class_imbalance(
                        frac,
                        dataset,
                        method_name,
                        model_name=model_name,
                        hook_name=hook_name,
                        results_path=results_path,
                        model_cache_path=resolved_cache_path,
                    )


"""
FUNCTIONS FOR CORRUPT CONDITIONS
"""


def run_baseline_corrupt(
    corrupt_frac: float,
    numbered_dataset: str,
    method_name: Method,
    model_name: str,
    hook_name: str,
    model_cache_path: str | Path,
    results_path: str | Path = DEFAULT_RESULTS_PATH,
):
    assert 0 <= corrupt_frac <= 0.5
    corrupt_frac = round(corrupt_frac * 20) / 20
    safe_hook = hook_name.replace("/", "-")
    base_path = f"baseline_results_{model_name}/corrupt/allruns/{safe_hook}_{numbered_dataset}_{method_name}_corrupt{corrupt_frac}"
    classifier_savepath = Path(results_path) / f"{base_path}_classifier.pt"
    metrics_savepath = Path(results_path) / f"{base_path}.csv"
    os.makedirs(os.path.dirname(metrics_savepath), exist_ok=True)
    if os.path.exists(metrics_savepath):
        return None
    size = DATASET_SIZES[numbered_dataset]
    num_train = min(size - 100, 1024)
    X_train, y_train, X_test, y_test = get_xy_traintest(
        num_train,
        numbered_dataset,
        hook_name,
        model_name=model_name,
        model_cache_path=model_cache_path,
    )
    y_train = corrupt_ytrain(y_train, corrupt_frac)
    # Run method and get metrics
    method = METHODS[method_name]
    results = method(X_train, y_train, X_test, y_test)
    # Create row with dataset and method metrics and save to csv
    row = {
        "dataset": numbered_dataset,
        "method": method_name,
        "ratio": corrupt_frac,
        "num_train": num_train,
    }
    for metric_name, metric_value in asdict(results.metrics).items():
        row[f"{metric_name}"] = metric_value
    pd.DataFrame([row]).to_csv(metrics_savepath, index=False)
    torch.save(
        {"classifier": results.classifier, "scaler": results.scaler},
        classifier_savepath,
    )
    return True


def run_all_baseline_corrupt(
    model_name: str,
    hook_name: str,
    results_path: str | Path = DEFAULT_RESULTS_PATH,
    model_cache_path: str | Path | None = None,
):
    with resolve_model_cache_path(model_cache_path) as resolved_cache_path:
        ensure_dataset_activations(
            model_name=model_name,
            dataset_short_names=DATASETS,
            hook_names=[hook_name],
            model_cache_path=resolved_cache_path,
            device="cpu",
        )

        shuffled_datasets = get_datasets(
            model_name, hook_name=hook_name, model_cache_path=resolved_cache_path
        ).copy()
        np.random.shuffle(shuffled_datasets)
        fracs = get_corrupt_frac()
        for frac in tqdm(fracs, desc="Corrupt Fracs (logreg)", position=0):
            for dataset in tqdm(
                shuffled_datasets,
                desc=f"Datasets (logreg, frac {frac:.2f})",
                position=1,
                leave=False,
            ):
                run_baseline_corrupt(
                    frac,
                    dataset,
                    method_name="logreg",
                    model_name=model_name,
                    hook_name=hook_name,
                    results_path=results_path,
                    model_cache_path=resolved_cache_path,
                )
