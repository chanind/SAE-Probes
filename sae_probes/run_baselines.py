import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from sae_probes.constants import (
    DEFAULT_MODEL_CACHE_PATH,
    DEFAULT_RESULTS_PATH,
    DEFAULT_SAE_CACHE_PATH,
)

from .utils_data import (
    corrupt_ytrain,
    get_class_imbalance,
    get_classimabalance_num_train,
    get_corrupt_frac,
    get_dataset_sizes,
    get_datasets,
    get_numbered_binary_tags,
    get_OOD_datasets,
    get_OOD_traintest,
    get_training_sizes,
    get_xy_traintest,
    get_xy_traintest_specify,
)
from .utils_sae import get_xy_OOD_sae
from .utils_training import (
    find_best_knn,
    find_best_mlp,
    find_best_pcareg,
    find_best_reg,
    find_best_xgboost,
)

dataset_sizes = get_dataset_sizes()
datasets = get_numbered_binary_tags()
methods = {
    "logreg": find_best_reg,
    "pca": find_best_pcareg,
    "knn": find_best_knn,
    "xgboost": find_best_xgboost,
    "mlp": find_best_mlp,
}


"""
FUNCTIONS FOR STANDARD CONDITIONS 
"""


def run_baseline_dataset_layer(
    layer: int,
    numbered_dataset: str,
    method_name: str,
    model_name: str,
    results_path: str | Path = DEFAULT_RESULTS_PATH,
):
    savepath = (
        Path(results_path)
        / f"baseline_results_{model_name}/normal/allruns/layer{layer}_{numbered_dataset}_{method_name}.csv"
    )
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    if os.path.exists(savepath):
        return None
    size = dataset_sizes[numbered_dataset]
    num_train = min(size - 100, 1024)
    X_train, y_train, X_test, y_test = get_xy_traintest(
        num_train, numbered_dataset, layer, model_name=model_name
    )

    # Run method and get metrics
    method = methods[method_name]
    metrics = method(X_train, y_train, X_test, y_test)

    # Create row with dataset and method metrics and save to csv
    row = {"dataset": numbered_dataset, "method": method_name}
    for metric_name, metric_value in metrics.items():
        row[f"{metric_name}"] = metric_value
    pd.DataFrame([row]).to_csv(savepath, index=False)
    return True


def run_all_baseline_normal(model_name: str, layers: list[int]):
    shuffled_datasets = get_datasets(model_name).copy()
    np.random.shuffle(shuffled_datasets)
    for method_name in methods.keys():
        for layer in layers:
            for dataset in shuffled_datasets:
                run_baseline_dataset_layer(
                    layer, dataset, method_name, model_name=model_name
                )


def coalesce_all_baseline_normal(
    model_name: str, layers: list[int], results_path: str | Path = DEFAULT_RESULTS_PATH
):
    # takes individual csvs and makes it into one big csv
    for layer in layers:
        all_results = []
        for dataset in datasets:
            for method_name in methods.keys():
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
    savepath = (
        Path(results_path)
        / f"baseline_results_{model_name}/scarcity/allruns/layer{layer}_{numbered_dataset}_{method_name}_numtrain{num_train}.csv"
    )
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    if os.path.exists(savepath):
        return None
    size = dataset_sizes[numbered_dataset]
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
    method = methods[method_name]
    metrics = method(X_train, y_train, X_test, y_test)
    # Create row with dataset and method metrics and save to csv
    row = {"dataset": numbered_dataset, "method": method_name, "num_train": num_train}
    for metric_name, metric_value in metrics.items():
        row[f"{metric_name}"] = metric_value
    pd.DataFrame([row]).to_csv(savepath, index=False)
    return True


def run_all_baseline_scarcity(
    model_name: str,
    layer: int,
    results_path: str | Path = DEFAULT_RESULTS_PATH,
    model_cache_path: str | Path = DEFAULT_MODEL_CACHE_PATH,
):
    shuffled_datasets = get_datasets(
        model_name, model_cache_path=model_cache_path
    ).copy()
    np.random.shuffle(shuffled_datasets)
    train_sizes = get_training_sizes()
    for method_name in methods.keys():
        for train in train_sizes:
            for dataset in shuffled_datasets:
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

    for dataset in datasets:
        dataset_results = []
        for num_train in train_sizes:
            for method_name in methods.keys():
                savepath = (
                    Path(results_path)
                    / f"baseline_results_{model_name}/scarcity/allruns/layer{layer}_{dataset}_{method_name}_numtrain{num_train}.csv"
                )
                if os.path.exists(savepath):
                    df = pd.read_csv(savepath)
                    dataset_results.append(df)
                    all_results.append(df)
                else:
                    if num_train + 100 <= dataset_sizes[dataset]:
                        raise ValueError(
                            f"Missing file {method_name}, {dataset} ({num_train}/{dataset_sizes[dataset]})"
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
    savepath = (
        Path(results_path)
        / f"baseline_results_{model_name}/imbalance/allruns/layer{layer}_{numbered_dataset}_{method_name}_frac{dataset_frac}.csv"
    )
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    if os.path.exists(savepath):
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
    method = methods[method_name]
    metrics = method(X_train, y_train, X_test, y_test)
    # Create row with dataset and method metrics and save to csv
    row = {
        "dataset": numbered_dataset,
        "method": method_name,
        "ratio": dataset_frac,
        "num_train": num_train,
    }
    for metric_name, metric_value in metrics.items():
        row[f"{metric_name}"] = metric_value
    pd.DataFrame([row]).to_csv(savepath, index=False)
    return True


def run_all_baseline_class_imbalance(
    model_name: str,
    layer: int,
    results_path: str | Path = DEFAULT_RESULTS_PATH,
    model_cache_path: str | Path = DEFAULT_MODEL_CACHE_PATH,
):
    shuffled_datasets = get_datasets(
        model_name, model_cache_path=model_cache_path
    ).copy()
    np.random.shuffle(shuffled_datasets)
    fracs = get_class_imbalance()
    i = 0
    for method_name in methods.keys():
        for frac in fracs:
            for dataset in shuffled_datasets:
                val = run_baseline_class_imbalance(
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
    for dataset in datasets:
        dataset_results = []
        for frac in fracs:
            for method_name in methods.keys():
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
    savepath = (
        Path(results_path)
        / f"baseline_results_{model_name}/corrupt/allruns/layer{layer}_{numbered_dataset}_{method_name}_corrupt{corrupt_frac}.csv"
    )
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    if os.path.exists(savepath):
        return None
    size = dataset_sizes[numbered_dataset]
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
    method = methods[method_name]
    metrics = method(X_train, y_train, X_test, y_test)
    # Create row with dataset and method metrics and save to csv
    row = {
        "dataset": numbered_dataset,
        "method": method_name,
        "ratio": corrupt_frac,
        "num_train": num_train,
    }
    for metric_name, metric_value in metrics.items():
        row[f"{metric_name}"] = metric_value
    pd.DataFrame([row]).to_csv(savepath, index=False)
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
    for method_name in ["logreg"]:
        for frac in fracs:
            for dataset in shuffled_datasets:
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
    for dataset in datasets:
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
                        f"Missing file {method_name}, {dataset} ({frac}/{dataset_sizes[dataset]})"
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


"""
FUNCTIONS FOR OOD EXPERIMENTS. This is the only regime where the SAE and baseline runs are done together
"""


def run_datasets_OOD(
    model_name: str,
    runsae: bool,
    layer: int,
    translation: bool,
    results_path: str | Path = DEFAULT_RESULTS_PATH,
    sae_cache_path: str | Path = DEFAULT_SAE_CACHE_PATH,
    model_cache_path: str | Path = DEFAULT_MODEL_CACHE_PATH,
):
    # runs the baseline and sae probes for OOD generalization
    # trains on normal data but tests on the OOD activations
    # run_sae should be true to run the sae generalization experiments
    # translation = True runs the probe on 66_living_room translated into different languages.
    # You can likely set this to False
    datasets = get_OOD_datasets(translation=translation)
    results = []

    for dataset in tqdm(datasets):
        X_train, y_train, X_test, y_test = get_OOD_traintest(
            dataset=dataset,
            model_name=model_name,
            layer=layer,
            model_cache_path=model_cache_path,
        )
        metrics = find_best_reg(
            X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, penalty="l2"
        )
        tosave = {"dataset": dataset, "test_auc_baseline": metrics["test_auc"]}  # type: ignore
        if runsae:
            X_train_sae, y_train_sae, X_test_sae, y_test_sae = get_xy_OOD_sae(  # type: ignore
                dataset,
                model_name=model_name,
                layer=layer,
                sae_cache_path=sae_cache_path,
            )
            metrics_sae = find_best_reg(
                X_train=X_train_sae,
                y_train=y_train_sae,
                X_test=X_test_sae,
                y_test=y_test_sae,
                penalty="l1",
            )
            tosave["test_auc_sae"] = metrics_sae["test_auc"]  # type: ignore
        results.append(tosave)

    # Create and save results dataframe
    os.makedirs(
        Path(results_path) / f"baseline_probes_{model_name}/ood/", exist_ok=True
    )
    results_df = pd.DataFrame(results)
    results_df.to_csv(
        Path(results_path) / f"baseline_probes_{model_name}/ood/all_results.csv",
        index=False,
    )


def ood_pruning(
    dataset: str,
    model_name: str,
    layer: int,
    results_path: str | Path = DEFAULT_RESULTS_PATH,
    sae_cache_path: str | Path = DEFAULT_SAE_CACHE_PATH,
):
    # does OOD Pruning
    # We use o1 to rank the latents by usefulness to the task via auto-interp explanations,
    # and prune the least helpful latents to see if that helps performance
    # section
    fname = (
        Path(results_path)
        / f"sae_probes_{model_name}/OOD/OOD_latents/{dataset}/{dataset}_latent_aucs.csv"
    )
    df = pd.read_csv(fname)
    df = df.sort_values("Relevance")
    X_train, y_train, X_test, y_test, top_by_average_diff = get_xy_OOD_sae(  # type: ignore
        dataset,
        k=8,
        model_name=model_name,
        layer=layer,
        return_indices=True,
        num_train=1500,
        sae_cache_path=sae_cache_path,
    )

    results = []
    bar = tqdm(range(1, 9))
    for k in bar:
        # Get top k latents by relevance
        top_k_latents = df.head(k)["latent"].values

        # Find indices of these latents in top_by_average_diff
        indices = [
            i for i, x in enumerate(top_by_average_diff) if x.item() in top_k_latents
        ]

        # Index X_train and X_test with these indices
        X_train_filtered = X_train[:, indices]
        X_test_filtered = X_test[:, indices]

        # Run find_best_reg
        metrics = find_best_reg(
            X_train=X_train_filtered,
            y_train=y_train,
            X_test=X_test_filtered,
            y_test=y_test,
            penalty="l1",
        )
        results.append(
            {"k": k, "ood_auc": metrics["test_auc"], "val_auc": metrics["val_auc"]}  # type: ignore
        )
        bar.set_postfix(results[-1])
    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(
        Path(results_path)
        / f"sae_probes_{model_name}/OOD/OOD_latents/{dataset}/{dataset}_pruned.csv",
        index=False,
    )
