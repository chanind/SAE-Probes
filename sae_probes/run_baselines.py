import os
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from sae_probes.constants import (
    DEFAULT_MODEL_CACHE_PATH,
    DEFAULT_RESULTS_PATH,
)

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
DEFAULT_METHODS: tuple[Method, ...] = ("logreg", "pca")


"""
FUNCTIONS FOR STANDARD CONDITIONS 
"""


def run_baseline_dataset_layer(
    layer: int,
    numbered_dataset: str,
    method_name: str,
    model_name: str,
    results_path: str | Path = DEFAULT_RESULTS_PATH,
    model_cache_path: str | Path = DEFAULT_MODEL_CACHE_PATH,
):
    base_path = f"baseline_results_{model_name}/normal/allruns/layer{layer}_{numbered_dataset}_{method_name}"
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
        layer,
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
    layer: int,
    results_path: str | Path = DEFAULT_RESULTS_PATH,
    model_cache_path: str | Path = DEFAULT_MODEL_CACHE_PATH,
    methods: Sequence[Method] = DEFAULT_METHODS,
):
    shuffled_datasets = get_datasets(
        model_name, model_cache_path=model_cache_path
    ).copy()
    np.random.shuffle(shuffled_datasets)
    for method_name in tqdm(methods, desc="Methods", position=0):
        for dataset in tqdm(
            shuffled_datasets, desc=f"{method_name} Datasets", position=1, leave=False
        ):
            run_baseline_dataset_layer(
                layer,
                dataset,
                method_name,
                model_name=model_name,
                results_path=results_path,
                model_cache_path=model_cache_path,
            )


def coalesce_all_baseline_normal(
    model_name: str,
    layer: int,
    results_path: str | Path = DEFAULT_RESULTS_PATH,
    methods: Sequence[Method] = DEFAULT_METHODS,
):
    # takes individual csvs and makes it into one big csv
    all_results = []
    for dataset in DATASETS:
        for method_name in methods:
            savepath = (
                Path(results_path)
                / f"baseline_results_{model_name}/normal/allruns/layer{layer}_{dataset}_{method_name}.csv"
            )
            if os.path.exists(savepath):
                df = pd.read_csv(savepath)
                all_results.append(df)
            else:
                print(f"Missing file {layer}, {method_name}, {dataset}")
                # raise ValueError(f'Missing file {layer}, {method_name}, {dataset}')

    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        layer_savepath = (
            Path(results_path)
            / f"baseline_probes_{model_name}/normal_settings/layer{layer}_results.csv"
        )
        os.makedirs(os.path.dirname(layer_savepath), exist_ok=True)
        combined_df.to_csv(layer_savepath, index=False)


"""
FUNCTIONS FOR DATA SCARCITY CONDITION
"""


def run_baseline_scarcity(
    num_train: int,
    numbered_dataset: str,
    method_name: str,
    model_name: str,
    layer: int,
    results_path: str | Path = DEFAULT_RESULTS_PATH,
    model_cache_path: str | Path = DEFAULT_MODEL_CACHE_PATH,
):
    base_path = f"baseline_results_{model_name}/scarcity/allruns/layer{layer}_{numbered_dataset}_{method_name}_numtrain{num_train}"
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
        layer,
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
    layer: int,
    results_path: str | Path = DEFAULT_RESULTS_PATH,
    model_cache_path: str | Path = DEFAULT_MODEL_CACHE_PATH,
    methods: Sequence[Method] = DEFAULT_METHODS,
):
    shuffled_datasets = get_datasets(
        model_name, model_cache_path=model_cache_path
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
                    layer=layer,
                    results_path=results_path,
                    model_cache_path=model_cache_path,
                )


def coalesce_all_scarcity(
    model_name: str,
    layer: int,
    results_path: str | Path = DEFAULT_RESULTS_PATH,
    methods: Sequence[Method] = DEFAULT_METHODS,
):
    # takes individual csvs and makes it into one big csv
    all_results = []
    train_sizes = get_training_sizes()

    # Create directories if they don't exist
    dataset_path = (
        Path(results_path) / f"baseline_results_{model_name}/scarcity/by_dataset"
    )
    allpath = Path(results_path) / f"baseline_probes_{model_name}/scarcity/"
    os.makedirs(dataset_path, exist_ok=True)
    os.makedirs(allpath, exist_ok=True)

    for dataset in DATASETS:
        dataset_results = []
        for num_train in train_sizes:
            for method_name in methods:
                savepath = (
                    Path(results_path)
                    / f"baseline_results_{model_name}/scarcity/allruns/layer{layer}_{dataset}_{method_name}_numtrain{num_train}.csv"
                )
                if os.path.exists(savepath):
                    df = pd.read_csv(savepath)
                    dataset_results.append(df)
                    all_results.append(df)
                else:
                    if num_train + 100 <= DATASET_SIZES[dataset]:
                        raise ValueError(
                            f"Missing file {method_name}, {dataset} ({num_train}/{DATASET_SIZES[dataset]})"
                        )

        # Save dataset-specific results
        if dataset_results:
            dataset_df = pd.concat(dataset_results, ignore_index=True)
            dataset_savepath = dataset_path / f"{dataset}.csv"
            dataset_df.to_csv(dataset_savepath, index=False)

    # Save combined results
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        summary_savepath = allpath / "all_results.csv"
        combined_df.to_csv(summary_savepath, index=False)


"""
FUNCTIONS FOR CLASS IMBALANCE CONDITION
"""


def run_baseline_class_imbalance(
    dataset_frac: float,
    numbered_dataset: str,
    method_name: str,
    model_name: str,
    layer: int,
    results_path: str | Path = DEFAULT_RESULTS_PATH,
    model_cache_path: str | Path = DEFAULT_MODEL_CACHE_PATH,
):
    assert 0 < dataset_frac < 1
    dataset_frac = round(dataset_frac * 20) / 20
    base_path = f"baseline_results_{model_name}/imbalance/allruns/layer{layer}_{numbered_dataset}_{method_name}_frac{dataset_frac}"
    classifier_savepath = Path(results_path) / f"{base_path}_classifier.pt"
    metrics_savepath = Path(results_path) / f"{base_path}.csv"
    os.makedirs(os.path.dirname(metrics_savepath), exist_ok=True)
    if os.path.exists(metrics_savepath):
        return None
    num_train, num_test = get_classimabalance_num_train(numbered_dataset)
    X_train, y_train, X_test, y_test = get_xy_traintest_specify(
        num_train,
        numbered_dataset,
        layer,
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
    layer: int,
    results_path: str | Path = DEFAULT_RESULTS_PATH,
    model_cache_path: str | Path = DEFAULT_MODEL_CACHE_PATH,
    methods: Sequence[Method] = DEFAULT_METHODS,
):
    shuffled_datasets = get_datasets(
        model_name, model_cache_path=model_cache_path
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
                    layer=layer,
                    results_path=results_path,
                    model_cache_path=model_cache_path,
                )


def coalesce_all_imbalance(
    model_name: str,
    layer: int,
    results_path: str | Path = DEFAULT_RESULTS_PATH,
    methods: Sequence[Method] = DEFAULT_METHODS,
):
    # takes individual csvs and makes it into one big csv
    all_results = []
    # Create directories if they don't exist
    dataset_path = (
        Path(results_path) / f"baseline_results_{model_name}/imbalance/by_dataset"
    )
    allpath = Path(results_path) / f"baseline_probes_{model_name}/imbalance"
    os.makedirs(dataset_path, exist_ok=True)
    os.makedirs(allpath, exist_ok=True)
    fracs = get_class_imbalance()
    i = 0
    for dataset in DATASETS:
        dataset_results = []
        for frac in fracs:
            for method_name in methods:
                frac = round(frac * 20) / 20
                savepath = (
                    Path(results_path)
                    / f"baseline_results_{model_name}/imbalance/allruns/layer{layer}_{dataset}_{method_name}_frac{frac}.csv"
                )
                if os.path.exists(savepath):
                    df = pd.read_csv(savepath)
                    dataset_results.append(df)
                    all_results.append(df)
                else:
                    i += 1
                    # raise ValueError(f'Missing file {savepath}, {dataset} ({frac}/{dataset_sizes[dataset]})')
                    # print(f'Missing file {method_name}, {dataset} ({num_train}/{dataset_sizes[dataset]})')

        # Save dataset-specific results
        if dataset_results:
            dataset_df = pd.concat(dataset_results, ignore_index=True)
            dataset_savepath = dataset_path / f"{dataset}.csv"
            dataset_df.to_csv(dataset_savepath, index=False)
    # Save combined results
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        summary_savepath = allpath / "all_results.csv"
        combined_df.to_csv(summary_savepath, index=False)


"""
FUNCTIONS FOR CORRUPT CONDITIONS
"""


def run_baseline_corrupt(
    corrupt_frac: float,
    numbered_dataset: str,
    method_name: str,
    model_name: str,
    layer: int,
    results_path: str | Path = DEFAULT_RESULTS_PATH,
    model_cache_path: str | Path = DEFAULT_MODEL_CACHE_PATH,
):
    assert 0 <= corrupt_frac <= 0.5
    corrupt_frac = round(corrupt_frac * 20) / 20
    base_path = f"baseline_results_{model_name}/corrupt/allruns/layer{layer}_{numbered_dataset}_{method_name}_corrupt{corrupt_frac}"
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
        layer,
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
    layer: int,
    results_path: str | Path = DEFAULT_RESULTS_PATH,
    model_cache_path: str | Path = DEFAULT_MODEL_CACHE_PATH,
):
    shuffled_datasets = get_datasets(
        model_name, model_cache_path=model_cache_path
    ).copy()
    np.random.shuffle(shuffled_datasets)
    fracs = get_corrupt_frac()
    for method_name in ["logreg"]:  # This loop is not tqdm wrapped by default
        # You could add a print here if you want to see which method is being processed, e.g.:
        # print(f"Processing corrupt baselines for method: {method_name}")
        for frac in tqdm(fracs, desc=f"Corrupt Fracs ({method_name})", position=0):
            for dataset in tqdm(
                shuffled_datasets,
                desc=f"Datasets ({method_name}, frac {frac:.2f})",
                position=1,
                leave=False,
            ):
                run_baseline_corrupt(
                    frac,
                    dataset,
                    method_name,
                    model_name=model_name,
                    layer=layer,
                    results_path=results_path,
                    model_cache_path=model_cache_path,
                )


def coalesce_all_corrupt(
    model_name: str,
    layer: int,
    results_path: str | Path = DEFAULT_RESULTS_PATH,
):
    # takes individual csvs and makes it into one big csv
    all_results = []
    # Create directories if they don't exist
    dataset_path = (
        Path(results_path) / f"baseline_results_{model_name}/corrupt/by_dataset"
    )
    allpath = Path(results_path) / f"baseline_probes_{model_name}/corrupt"
    os.makedirs(dataset_path, exist_ok=True)
    os.makedirs(allpath, exist_ok=True)
    fracs = get_corrupt_frac()
    for dataset in DATASETS:
        dataset_results = []
        for frac in fracs:
            for method_name in ["logreg"]:
                frac = round(frac * 20) / 20
                savepath = (
                    Path(results_path)
                    / f"baseline_results_{model_name}/corrupt/allruns/layer{layer}_{dataset}_{method_name}_corrupt{frac}.csv"
                )
                if os.path.exists(savepath):
                    df = pd.read_csv(savepath)
                    dataset_results.append(df)
                    all_results.append(df)
                else:
                    raise ValueError(
                        f"Missing file {method_name}, {dataset} ({frac}/{DATASET_SIZES[dataset]})"
                    )
                    # print(f'Missing file {method_name}, {dataset} ({num_train}/{dataset_sizes[dataset]})')

        # Save dataset-specific results
        if dataset_results:
            dataset_df = pd.concat(dataset_results, ignore_index=True)
            dataset_savepath = dataset_path / f"{dataset}.csv"
            dataset_df.to_csv(dataset_savepath, index=False)

    # Save combined results
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        summary_savepath = allpath / "all_results.csv"
        combined_df.to_csv(summary_savepath, index=False)
