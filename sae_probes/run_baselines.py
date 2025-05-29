import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sae_lens import SAE
from tqdm import tqdm
from transformer_lens import HookedTransformer

from sae_probes.utils_data import (
    corrupt_ytrain,
    get_class_imbalance,
    get_classimabalance_num_train,
    get_corrupt_frac,
    get_dataset_sizes,
    get_datasets,
    get_disagree_glue_indices,
    get_glue_labels,
    get_glue_traintest,
    get_layers,
    get_model_activations_for_dataset,
    get_numbered_binary_tags,
    get_training_sizes,
    get_xy_traintest,
    get_xy_traintest_specify,
)
from sae_probes.utils_sae import get_sae_features
from sae_probes.utils_training import (
    find_best_knn,
    find_best_mlp,
    find_best_pcareg,
    find_best_reg,
    find_best_xgboost,
)

dataset_sizes_global = get_dataset_sizes()
datasets_global_tags = get_numbered_binary_tags()
methods_global = {
    "logreg": find_best_reg,
    "pca": find_best_pcareg,
    "knn": find_best_knn,
    "xgboost": find_best_xgboost,
    "mlp": find_best_mlp,
}


"""
FUNCTIONS FOR STANDARD CONDITIONS 
"""


def run_baseline_dataset_layer_generic(
    model: HookedTransformer,
    layer_idx: int,
    dataset_name: str,
    method_name: str,
    device: str | torch.device,
    max_seq_len: int = 1024,
    results_base_dir: str = "results/baseline_probes",
    cache_dir_base: str = "data/generated_model_activations",
):
    model_name_str = model.cfg.model_name.replace("/", "_")
    results_base_dir_path = Path(results_base_dir)
    save_dir = results_base_dir_path / model_name_str / "normal" / "allruns"
    save_dir.mkdir(parents=True, exist_ok=True)
    savepath = save_dir / f"layer{layer_idx}_{dataset_name}_{method_name}.csv"

    if savepath.exists():
        return None

    current_dataset_size = dataset_sizes_global.get(dataset_name, 0)
    if current_dataset_size == 0:
        print(
            f"Warning: Dataset {dataset_name} not found in dataset_sizes_global or size is 0. Skipping."
        )
        return False

    print(
        f"Running baseline: Model {model_name_str}, Layer {layer_idx}, Dataset {dataset_name}, Method {method_name}"
    )
    try:
        X_train, X_test, y_train, y_test = get_model_activations_for_dataset(
            model=model,
            dataset_name=dataset_name,
            layer_idx=layer_idx,
            device=device,
            setting_type="normal",
            max_seq_len=max_seq_len,
            cache_dir_base=Path(cache_dir_base),
            expected_activation_dim=model.cfg.d_model,
        )
    except Exception as e:
        print(
            f"Error getting model activations for {dataset_name}, layer {layer_idx}: {e}. Skipping baseline run."
        )
        error_df = pd.DataFrame(
            [
                {
                    "dataset": dataset_name,
                    "layer": layer_idx,
                    "method": method_name,
                    "error": str(e),
                }
            ]
        )
        error_df.to_csv(savepath, index=False)
        return False

    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        print(
            f"No train/test data for {dataset_name}, layer {layer_idx} after processing. Skipping method {method_name}."
        )
        error_df = pd.DataFrame(
            [
                {
                    "dataset": dataset_name,
                    "layer": layer_idx,
                    "method": method_name,
                    "error": "No train/test data",
                }
            ]
        )
        error_df.to_csv(savepath, index=False)
        return False

    method_func = methods_global[method_name]
    try:
        X_train_np = (
            X_train.cpu().numpy() if isinstance(X_train, torch.Tensor) else X_train
        )
        X_test_np = X_test.cpu().numpy() if isinstance(X_test, torch.Tensor) else X_test
        y_train_np = (
            y_train.cpu().numpy() if isinstance(y_train, torch.Tensor) else y_train
        )
        y_test_np = y_test.cpu().numpy() if isinstance(y_test, torch.Tensor) else y_test
        metrics = method_func(X_train_np, y_train_np, X_test_np, y_test_np)
    except Exception as e:
        print(
            f"Error running method {method_name} for {dataset_name}, layer {layer_idx}: {e}. Skipping."
        )
        error_row = {
            "dataset": dataset_name,
            "layer": layer_idx,
            "method": method_name,
            "error": str(e),
        }
        pd.DataFrame([error_row]).to_csv(savepath, index=False)
        return False

    row = {"dataset": dataset_name, "layer": layer_idx, "method": method_name}
    for metric_name_key, metric_value in metrics.items():
        row[f"{metric_name_key}"] = metric_value
    pd.DataFrame([row]).to_csv(savepath, index=False)
    return True


def run_all_baseline_normal_generic(
    model: HookedTransformer,
    layers_to_run: list[int],
    target_dataset_names: list[str] | None = None,
    target_method_names: list[str] | None = None,
    device: str | torch.device | None = None,
    max_seq_len_override: int | None = None,
    results_base_dir: str = "results/baseline_probes",
    cache_dir_base: str = "data/generated_model_activations",
):
    if device is None:
        device = model.cfg.device

    model.to(device)
    max_seq_len = (
        max_seq_len_override if max_seq_len_override is not None else model.cfg.n_ctx
    )

    datasets_to_process = (
        target_dataset_names if target_dataset_names else datasets_global_tags
    )
    methods_to_process = (
        target_method_names if target_method_names else list(methods_global.keys())
    )

    for method_name in tqdm(methods_to_process, desc="Methods"):
        for layer_idx in tqdm(
            layers_to_run, desc=f"Layers for {method_name}", leave=False
        ):
            for dataset_name in tqdm(
                datasets_to_process,
                desc=f"Datasets for L{layer_idx}, M{method_name}",
                leave=False,
            ):
                run_baseline_dataset_layer_generic(
                    model=model,
                    layer_idx=layer_idx,
                    dataset_name=dataset_name,
                    method_name=method_name,
                    device=device,
                    max_seq_len=max_seq_len,
                    results_base_dir=results_base_dir,
                    cache_dir_base=cache_dir_base,
                )
    print(f"Finished all normal baseline runs for model {model.cfg.model_name}.")


def coalesce_all_baseline_normal_generic(
    model_name_str: str,
    layers_run: list[int],
    target_dataset_names: list[str] | None = None,
    target_method_names: list[str] | None = None,
    results_base_dir: str = "results/baseline_probes",
    output_base_dir: str = "results/baseline_probes",
):
    datasets_to_coalesce = (
        target_dataset_names if target_dataset_names else datasets_global_tags
    )
    methods_to_coalesce = (
        target_method_names if target_method_names else list(methods_global.keys())
    )

    model_name_path_segment = model_name_str
    results_base_dir_path = Path(results_base_dir)
    output_base_dir_path = Path(output_base_dir)

    for layer_idx in layers_run:
        all_results_for_layer: list[pd.DataFrame] = []
        missing_files_count = 0
        for dataset_name in datasets_to_coalesce:
            for method_name in methods_to_coalesce:
                csv_path = (
                    results_base_dir_path
                    / model_name_path_segment
                    / "normal"
                    / "allruns"
                    / f"layer{layer_idx}_{dataset_name}_{method_name}.csv"
                )
                if csv_path.exists():
                    try:
                        df = pd.read_csv(csv_path)
                        all_results_for_layer.append(df)
                    except pd.errors.EmptyDataError:
                        print(f"Warning: Empty CSV file (or error marker): {csv_path}")
                    except Exception as e:
                        print(f"Error reading CSV {csv_path}: {e}")
                else:
                    missing_files_count += 1

        if missing_files_count > 0:
            print(
                f"Note: For layer {layer_idx}, {missing_files_count} expected result files were missing during coalescence."
            )

        if all_results_for_layer:
            combined_df = pd.concat(all_results_for_layer, ignore_index=True)
            layer_save_dir = (
                output_base_dir_path
                / model_name_path_segment
                / "normal_settings_summary"
            )
            layer_save_dir.mkdir(parents=True, exist_ok=True)
            layer_savepath = layer_save_dir / f"layer{layer_idx}_results.csv"
            combined_df.to_csv(layer_savepath, index=False)
            print(
                f"Saved combined normal results for layer {layer_idx} to {layer_savepath}"
            )
        else:
            print(
                f"No results found to coalesce for layer {layer_idx} for model {model_name_str} (normal setting)."
            )


"""
FUNCTIONS FOR DATA SCARCITY CONDITION
"""


def run_baseline_scarcity_generic(
    model: HookedTransformer,
    layer_idx: int,
    num_train_target: int,
    dataset_name: str,
    method_name: str,
    device: str | torch.device,
    max_seq_len: int = 1024,
    results_base_dir: str = "results/baseline_probes",
    cache_dir_base: str = "data/generated_model_activations",
):
    model_name_str = model.cfg.model_name.replace("/", "_")
    results_base_dir_path = Path(results_base_dir)
    save_dir = results_base_dir_path / model_name_str / "scarcity" / "allruns"
    save_dir.mkdir(parents=True, exist_ok=True)
    savepath = (
        save_dir
        / f"layer{layer_idx}_{dataset_name}_{method_name}_numtrain{num_train_target}.csv"
    )

    if savepath.exists():
        return None

    current_dataset_size = dataset_sizes_global.get(dataset_name, 0)
    if current_dataset_size > 0 and num_train_target >= current_dataset_size:
        print(
            f"Warning: num_train_target ({num_train_target}) >= total dataset size ({current_dataset_size}) for {dataset_name}. Skipping scarcity run."
        )
        return False
    if current_dataset_size == 0:
        print(f"Dataset {dataset_name} has size 0. Skipping scarcity.")
        return False

    print(
        f"Running scarcity baseline: Model {model_name_str}, L{layer_idx}, DS {dataset_name}, M{method_name}, NTrain {num_train_target}"
    )
    try:
        X_train, X_test, y_train, y_test = get_model_activations_for_dataset(
            model=model,
            dataset_name=dataset_name,
            layer_idx=layer_idx,
            device=device,
            setting_type="scarcity",
            num_train_samples_target=num_train_target,
            max_seq_len=max_seq_len,
            cache_dir_base=Path(cache_dir_base),
            expected_activation_dim=model.cfg.d_model,
        )
    except Exception as e:
        print(
            f"Error getting model activations for scarcity ({dataset_name}, L{layer_idx}, NTrain {num_train_target}): {e}. Skipping."
        )
        error_df = pd.DataFrame(
            [
                {
                    "dataset": dataset_name,
                    "layer": layer_idx,
                    "method": method_name,
                    "num_train_target": num_train_target,
                    "error": str(e),
                }
            ]
        )
        error_df.to_csv(savepath, index=False)
        return False

    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        print(
            f"No train/test data for scarcity ({dataset_name}, L{layer_idx}, NTrain {num_train_target}). Actual NTrain: {X_train.shape[0]}. Skipping method {method_name}."
        )
        error_val = (
            "No training samples generated"
            if X_train.shape[0] == 0 and num_train_target > 0
            else "No test samples generated"
        )
        error_row = {
            "dataset": dataset_name,
            "layer": layer_idx,
            "method": method_name,
            "num_train_target": num_train_target,
            "num_train_actual": X_train.shape[0],
            "error": error_val,
        }
        pd.DataFrame([error_row]).to_csv(savepath, index=False)
        return False

    actual_num_train = X_train.shape[0]

    method_func = methods_global[method_name]
    try:
        X_train_np = (
            X_train.cpu().numpy() if isinstance(X_train, torch.Tensor) else X_train
        )
        X_test_np = X_test.cpu().numpy() if isinstance(X_test, torch.Tensor) else X_test
        y_train_np = (
            y_train.cpu().numpy() if isinstance(y_train, torch.Tensor) else y_train
        )
        y_test_np = y_test.cpu().numpy() if isinstance(y_test, torch.Tensor) else y_test
        metrics = method_func(X_train_np, y_train_np, X_test_np, y_test_np)
    except Exception as e:
        print(
            f"Error running method {method_name} for scarcity ({dataset_name}, L{layer_idx}, NTrain {actual_num_train}): {e}"
        )
        error_row = {
            "dataset": dataset_name,
            "layer": layer_idx,
            "method": method_name,
            "num_train_target": num_train_target,
            "num_train_actual": actual_num_train,
            "error": str(e),
        }
        pd.DataFrame([error_row]).to_csv(savepath, index=False)
        return False

    row = {
        "dataset": dataset_name,
        "layer": layer_idx,
        "method": method_name,
        "num_train_target": num_train_target,
        "num_train_actual": actual_num_train,
    }
    for metric_name_key, metric_value in metrics.items():
        row[f"{metric_name_key}"] = metric_value
    pd.DataFrame([row]).to_csv(savepath, index=False)
    return True


def run_all_baseline_scarcity_generic(
    model: HookedTransformer,
    layer_to_run: int,
    target_dataset_names: list[str] | None = None,
    target_method_names: list[str] | None = None,
    train_sizes_to_run: list[int] | None = None,
    device: str | torch.device | None = None,
    max_seq_len_override: int | None = None,
    results_base_dir: str = "results/baseline_probes",
    cache_dir_base: str = "data/generated_model_activations",
):
    if device is None:
        device = model.cfg.device
    model.to(device)
    max_seq_len = (
        max_seq_len_override if max_seq_len_override is not None else model.cfg.n_ctx
    )

    datasets_to_process = (
        target_dataset_names if target_dataset_names else datasets_global_tags
    )
    methods_to_process = (
        target_method_names if target_method_names else list(methods_global.keys())
    )
    scarcity_train_sizes = (
        train_sizes_to_run if train_sizes_to_run else get_training_sizes()
    )

    for method_name in tqdm(methods_to_process, desc="Methods for Scarcity"):
        for num_train_target in tqdm(
            scarcity_train_sizes,
            desc=f"N_Train for M{method_name}, L{layer_to_run}",
            leave=False,
        ):
            for dataset_name in tqdm(
                datasets_to_process,
                desc=f"Datasets for NTrain {num_train_target}",
                leave=False,
            ):
                run_baseline_scarcity_generic(
                    model=model,
                    layer_idx=layer_to_run,
                    num_train_target=num_train_target,
                    dataset_name=dataset_name,
                    method_name=method_name,
                    device=device,
                    max_seq_len=max_seq_len,
                    results_base_dir=results_base_dir,
                    cache_dir_base=cache_dir_base,
                )
    print(
        f"Finished all scarcity baseline runs for model {model.cfg.model_name}, layer {layer_to_run}."
    )


def coalesce_all_scarcity_generic(
    model_name_str: str,
    layer_run: int,
    target_dataset_names: list[str] | None = None,
    target_method_names: list[str] | None = None,
    train_sizes_run: list[int] | None = None,
    results_base_dir: str = "results/baseline_probes",
    output_base_dir: str = "results/baseline_probes",
):
    datasets_to_coalesce = (
        target_dataset_names if target_dataset_names else datasets_global_tags
    )
    methods_to_coalesce = (
        target_method_names if target_method_names else list(methods_global.keys())
    )
    scarcity_train_sizes_used = (
        train_sizes_run if train_sizes_run else get_training_sizes()
    )
    model_name_path_segment = model_name_str
    results_base_dir_path = Path(results_base_dir)
    output_base_dir_path = Path(output_base_dir)

    all_results_for_layer: list[pd.DataFrame] = []
    missing_files_count = 0

    for dataset_name in datasets_to_coalesce:
        for num_train in scarcity_train_sizes_used:
            for method_name in methods_to_coalesce:
                csv_path = (
                    results_base_dir_path
                    / model_name_path_segment
                    / "scarcity"
                    / "allruns"
                    / f"layer{layer_run}_{dataset_name}_{method_name}_numtrain{num_train}.csv"
                )
                if csv_path.exists():
                    try:
                        df = pd.read_csv(csv_path)
                        all_results_for_layer.append(df)
                    except pd.errors.EmptyDataError:
                        print(f"Warning: Empty CSV file: {csv_path}")
                    except Exception as e:
                        print(f"Error reading CSV {csv_path}: {e}")
                else:
                    missing_files_count += 1

    if missing_files_count > 0:
        print(
            f"Note: For layer {layer_run} (scarcity), {missing_files_count} potential result files were missing during coalescence."
        )

    if all_results_for_layer:
        combined_df = pd.concat(all_results_for_layer, ignore_index=True)
        summary_save_dir = (
            output_base_dir_path / model_name_path_segment / "scarcity_settings_summary"
        )
        summary_save_dir.mkdir(parents=True, exist_ok=True)
        summary_savepath = summary_save_dir / f"layer{layer_run}_scarcity_results.csv"
        combined_df.to_csv(summary_savepath, index=False)
        print(
            f"Saved combined scarcity results for layer {layer_run} to {summary_savepath}"
        )
    else:
        print(
            f"No scarcity results found to coalesce for layer {layer_run} for model {model_name_str}."
        )


"""
FUNCTIONS FOR CLASS IMBALANCE CONDITION
"""


def run_baseline_class_imbalance(
    dataset_frac, numbered_dataset, method_name, model_name="gemma-2-9b", layer=20
):
    assert 0 < dataset_frac < 1
    dataset_frac = round(dataset_frac * 20) / 20
    savepath = f"data/baseline_results_{model_name}/imbalance/allruns/layer{layer}_{numbered_dataset}_{method_name}_frac{dataset_frac}.csv"
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
    )
    # Run method and get metrics
    method = methods_global[method_name]
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


def run_all_baseline_class_imbalance(model_name="gemma-2-9b", layer=20):
    assert layer in get_layers(model_name)
    shuffled_datasets = get_datasets().copy()
    np.random.shuffle(shuffled_datasets)
    fracs = get_class_imbalance()
    i = 0
    for method_name in methods_global.keys():
        for frac in fracs:
            for dataset in shuffled_datasets:
                val = run_baseline_class_imbalance(
                    frac, dataset, method_name, model_name=model_name, layer=layer
                )


def coalesce_all_imbalance(model_name="gemma-2-9b", layer=20):
    # takes individual csvs and makes it into one big csv
    all_results = []
    # Create directories if they don't exist
    dataset_path = f"data/baseline_results_{model_name}/imbalance/by_dataset"
    allpath = f"results/baseline_probes_{model_name}/imbalance"
    os.makedirs(dataset_path, exist_ok=True)
    os.makedirs(allpath, exist_ok=True)
    fracs = get_class_imbalance()
    i = 0
    for dataset in datasets_global_tags:
        dataset_results = []
        for frac in fracs:
            for method_name in methods_global.keys():
                frac = round(frac * 20) / 20
                savepath = f"data/baseline_results_{model_name}/imbalance/allruns/layer{layer}_{dataset}_{method_name}_frac{frac}.csv"
                if os.path.exists(savepath):
                    df = pd.read_csv(savepath)
                    dataset_results.append(df)
                    all_results.append(df)
                else:
                    i += 1
                    # raise ValueError(f'Missing file {savepath}, {dataset} ({frac}/{dataset_sizes_global[dataset]})')
                    # print(f'Missing file {method_name}, {dataset} ({num_train}/{dataset_sizes[dataset]})')

        # Save dataset-specific results
        if dataset_results:
            dataset_df = pd.concat(dataset_results, ignore_index=True)
            dataset_savepath = f"{dataset_path}/{dataset}.csv"
            dataset_df.to_csv(dataset_savepath, index=False)
    # Save combined results
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        summary_savepath = f"{allpath}/all_results.csv"
        combined_df.to_csv(summary_savepath, index=False)


"""
FUNCTIONS FOR CORRUPT CONDITIONS
"""


def run_baseline_corrupt(
    corrupt_frac_val, numbered_dataset, method_name, model_name="gemma-2-9b", layer=20
):
    assert 0 <= corrupt_frac_val <= 0.5
    corrupt_frac_str = str(round(corrupt_frac_val * 20) / 20).replace(".", "p")
    savepath = f"data/baseline_results_{model_name}/corrupt/allruns/layer{layer}_{numbered_dataset}_{method_name}_corrupt{corrupt_frac_str}.csv"
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    if os.path.exists(savepath):
        return None
    size = dataset_sizes_global[numbered_dataset]
    num_train = min(size - 100, 1024)
    X_train, y_train, X_test, y_test = get_xy_traintest(
        num_train, numbered_dataset, layer, model_name=model_name
    )
    y_train_corrupted = corrupt_ytrain(y_train, corrupt_frac_val)
    # Run method and get metrics
    method = methods_global[method_name]
    metrics = method(X_train, y_train_corrupted, X_test, y_test)
    # Create row with dataset and method metrics and save to csv
    row = {
        "dataset": numbered_dataset,
        "method": method_name,
        "corrupt_frac": corrupt_frac_val,
        "num_train": num_train,
    }
    for metric_name, metric_value in metrics.items():
        row[f"{metric_name}"] = metric_value
    pd.DataFrame([row]).to_csv(savepath, index=False)
    return True


def run_all_baseline_corrupt(model_name="gemma-2-9b", layer=20):
    assert layer in get_layers(model_name)
    shuffled_datasets = get_datasets().copy()
    np.random.shuffle(shuffled_datasets)
    fracs = get_corrupt_frac()
    for method_name in ["logreg"]:
        for frac in fracs:
            for dataset in shuffled_datasets:
                val = run_baseline_corrupt(
                    frac, dataset, method_name, model_name=model_name, layer=layer
                )
                # print(val, method_name, frac, dataset)


def coalesce_all_corrupt(model_name="gemma-2-9b", layer=20):
    # takes individual csvs and makes it into one big csv
    all_results = []
    # Create directories if they don't exist
    dataset_path = f"data/baseline_results_{model_name}/corrupt/by_dataset"
    allpath = f"results/baseline_probes_{model_name}/corrupt"
    os.makedirs(dataset_path, exist_ok=True)
    os.makedirs(allpath, exist_ok=True)
    fracs = get_corrupt_frac()
    for dataset in datasets_global_tags:
        dataset_results = []
        for frac_val in fracs:
            for method_name in ["logreg"]:
                corrupt_frac_str = str(round(frac_val * 20) / 20).replace(".", "p")
                savepath = f"data/baseline_results_{model_name}/corrupt/allruns/layer{layer}_{dataset}_{method_name}_corrupt{corrupt_frac_str}.csv"
                if os.path.exists(savepath):
                    df = pd.read_csv(savepath)
                    dataset_results.append(df)
                    all_results.append(df)
                else:
                    # This error might be too strict if some corrupt_fracs don't generate files by design
                    # raise ValueError(
                    #     f"Missing file {method_name}, {dataset} ({frac_val}/{dataset_sizes_global[dataset]})"
                    # )
                    print(f"Warning: Missing file {savepath}")

        # Save dataset-specific results
        if dataset_results:
            dataset_df = pd.concat(dataset_results, ignore_index=True)
            dataset_savepath = f"{dataset_path}/{dataset}.csv"
            dataset_df.to_csv(dataset_savepath, index=False)

    # Save combined results
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        summary_savepath = f"{allpath}/all_results.csv"
        combined_df.to_csv(summary_savepath, index=False)


"""
FUNCTIONS FOR OOD EXPERIMENTS. This is the only regime where the SAE and baseline runs are done together
"""


def run_datasets_OOD_generic(
    model: HookedTransformer,
    sae: SAE | None,
    train_dataset_name: str,
    ood_dataset_names: list[str],
    layer_idx: int,
    hook_point_name: str,
    method_name: str,
    results_dir_base: str | Path,
    device: str | torch.device,
    sae_k: int | None = 128,
    max_seq_len_override: int | None = None,
    cache_dir_base: str | Path = "data/generated_model_activations",
    seed: int = 42,
):
    results_list = []
    model_name_str = model.cfg.model_name.replace("/", "_")
    results_dir = Path(results_dir_base) / model_name_str / "ood_generic"
    results_dir.mkdir(parents=True, exist_ok=True)

    actual_max_seq_len = (
        max_seq_len_override if max_seq_len_override is not None else model.cfg.n_ctx
    )

    print(
        f"OOD: Training on {train_dataset_name}, Layer {layer_idx}, Hook {hook_point_name}"
    )
    # 1. Get training activations and labels
    try:
        X_train_model_acts, _, y_train, _ = get_model_activations_for_dataset(
            model=model,
            dataset_name=train_dataset_name,
            layer_idx=layer_idx,
            hook_point_name=hook_point_name,
            pooling_strategy="last",
            device=device,
            setting_type="normal",
            max_seq_len=actual_max_seq_len,
            cache_dir_base=Path(cache_dir_base),
            expected_activation_dim=model.cfg.d_model,
            seed=seed,
            test_set_ratio=0.01,
        )
    except Exception as e:
        print(
            f"Error getting training model activations for OOD ({train_dataset_name}, L{layer_idx}): {e}. Skipping OOD run."
        )
        return []

    if X_train_model_acts.shape[0] == 0:
        print(f"No training data for OOD base: {train_dataset_name}. Skipping.")
        return []

    # Prepare X_train based on whether SAE is used or not
    current_X_train = X_train_model_acts
    probe_type_str = "baseline"
    sae_name_for_path = "no_sae"

    if sae:
        probe_type_str = "sae"
        sae_name_for_path = getattr(sae.cfg, "sae_name", f"sae_L{layer_idx}").replace(
            "/", "_"
        )
        try:
            X_train_sae_features, _ = get_sae_features(
                sae, X_train_model_acts, device=device
            )
            if X_train_sae_features.shape[1] == 0:
                print(
                    f"SAE features for OOD train ({train_dataset_name}) have 0 dimension. Skipping SAE OOD."
                )
                return []

            if (
                sae_k is not None
                and sae_k > 0
                and X_train_sae_features.shape[1] > sae_k
            ):
                if len(torch.unique(y_train)) < 2:
                    print(
                        "Warning: y_train for OOD base has <2 unique classes. Using first k SAE features."
                    )
                    top_indices = torch.arange(
                        min(sae_k, X_train_sae_features.shape[1])
                    )
                else:
                    y_train_torch = (
                        y_train
                        if isinstance(y_train, torch.Tensor)
                        else torch.from_numpy(y_train)
                    )
                    diff = X_train_sae_features[y_train_torch == 1].mean(
                        0
                    ) - X_train_sae_features[y_train_torch == 0].mean(0)
                    top_indices = torch.argsort(torch.abs(diff), descending=True)[
                        :sae_k
                    ]
                current_X_train = X_train_sae_features[:, top_indices]
            else:
                current_X_train = X_train_sae_features
        except Exception as e:
            print(
                f"Error getting SAE features for OOD training ({train_dataset_name}): {e}. Skipping SAE OOD run."
            )
            return []

    # Ensure current_X_train and y_train are numpy for sklearn methods
    current_X_train_np = (
        current_X_train.cpu().numpy()
        if isinstance(current_X_train, torch.Tensor)
        else current_X_train
    )
    y_train_np = y_train.cpu().numpy() if isinstance(y_train, torch.Tensor) else y_train

    for ood_test_dataset_name in tqdm(
        ood_dataset_names, desc=f"OOD Test Datasets for {train_dataset_name}"
    ):
        save_path_ood = (
            results_dir
            / f"ood_{probe_type_str}_{model_name_str}_L{layer_idx}_hook_{hook_point_name.replace('.', '_')}_train_{train_dataset_name}_test_{ood_test_dataset_name}_{sae_name_for_path}.csv"
        )
        if save_path_ood.exists():
            print(f"Skipping existing OOD result: {save_path_ood}")
            continue

        print(f"  Testing on OOD dataset: {ood_test_dataset_name}")
        try:
            _, X_test_model_acts, _, y_test = get_model_activations_for_dataset(
                model=model,
                dataset_name=ood_test_dataset_name,
                layer_idx=layer_idx,
                hook_point_name=hook_point_name,
                pooling_strategy="last",
                device=device,
                setting_type="normal",
                max_seq_len=actual_max_seq_len,
                cache_dir_base=Path(cache_dir_base),
                expected_activation_dim=model.cfg.d_model,
                seed=seed,
            )
        except Exception as e:
            print(
                f"Error getting test model activations for OOD ({ood_test_dataset_name}, L{layer_idx}): {e}. Skipping."
            )
            results_list.append(
                {
                    "train_dataset": train_dataset_name,
                    "test_dataset": ood_test_dataset_name,
                    "layer": layer_idx,
                    "probe_type": probe_type_str,
                    "method": method_name,
                    "sae_name": sae_name_for_path if sae else "N/A",
                    "error": f"Failed to get test activations: {e}",
                }
            )
            continue

        if X_test_model_acts.shape[0] == 0:
            print(
                f"No test data for OOD dataset: {ood_test_dataset_name}. Skipping probe."
            )
            results_list.append(
                {
                    "train_dataset": train_dataset_name,
                    "test_dataset": ood_test_dataset_name,
                    "layer": layer_idx,
                    "probe_type": probe_type_str,
                    "method": method_name,
                    "sae_name": sae_name_for_path if sae else "N/A",
                    "error": "No test data",
                }
            )
            continue

        current_X_test = X_test_model_acts
        if sae:
            try:
                X_test_sae_features, _ = get_sae_features(
                    sae, X_test_model_acts, device=device
                )
                if X_test_sae_features.shape[1] == 0:
                    print(
                        f"SAE features for OOD test ({ood_test_dataset_name}) have 0 dimension. Skipping."
                    )
                    results_list.append(
                        {
                            "train_dataset": train_dataset_name,
                            "test_dataset": ood_test_dataset_name,
                            "layer": layer_idx,
                            "probe_type": "sae",
                            "method": method_name,
                            "sae_name": sae_name_for_path,
                            "error": "0-dim SAE features on test",
                        }
                    )
                    continue
                if (
                    sae_k is not None
                    and sae_k > 0
                    and X_train_sae_features.shape[1] > sae_k
                ):
                    current_X_test = X_test_sae_features[:, top_indices]
                else:
                    current_X_test = X_test_sae_features
            except Exception as e:
                print(
                    f"Error getting SAE features for OOD testing ({ood_test_dataset_name}): {e}. Skipping."
                )
                results_list.append(
                    {
                        "train_dataset": train_dataset_name,
                        "test_dataset": ood_test_dataset_name,
                        "layer": layer_idx,
                        "probe_type": "sae",
                        "method": method_name,
                        "sae_name": sae_name_for_path,
                        "error": f"Failed to get SAE test features: {e}",
                    }
                )
                continue

        current_X_test_np = (
            current_X_test.cpu().numpy()
            if isinstance(current_X_test, torch.Tensor)
            else current_X_test
        )
        y_test_np = y_test.cpu().numpy() if isinstance(y_test, torch.Tensor) else y_test

        # Run probe
        method_func = methods_global[method_name]
        try:
            metrics = method_func(
                current_X_train_np, y_train_np, current_X_test_np, y_test_np
            )
        except Exception as e:
            print(
                f"Error running OOD probe ({method_name} on {train_dataset_name} -> {ood_test_dataset_name}): {e}"
            )
            metrics = {"error": str(e)}

        row = {
            "train_dataset": train_dataset_name,
            "test_dataset": ood_test_dataset_name,
            "layer": layer_idx,
            "probe_type": probe_type_str,
            "method": method_name,
            "hook_point": hook_point_name,
            "sae_name": sae_name_for_path if sae else "N/A",
            "sae_k_features": sae_k
            if sae and sae_k is not None and X_train_sae_features.shape[1] > sae_k
            else "all_sae_features",
        }
        row.update(metrics)
        results_list.append(row)
        pd.DataFrame([row]).to_csv(save_path_ood, index=False)
        print(f"Saved OOD result to {save_path_ood}")

    return results_list


def latent_performance_generic(
    model: HookedTransformer,
    sae: SAE,
    train_dataset_name: str,  # Dataset to train the main probe and select features
    ood_test_dataset_name: str,  # OOD Dataset to test on
    layer_idx: int,
    hook_point_name: str,
    method_name: str,  # e.g. "logreg"
    results_dir_base: str | Path,
    device: str | torch.device,
    num_top_sae_features_to_analyze: int = 8,
    max_seq_len_override: int | None = None,
    cache_dir_base: str | Path = "data/generated_model_activations",
    seed: int = 42,
) -> list[dict[str, Any]]:
    model_name_str = model.cfg.model_name.replace("/", "_")
    sae_name_str = getattr(sae.cfg, "sae_name", f"sae_L{layer_idx}").replace("/", "_")
    # results_dir for this specific experiment run
    specific_results_path = (
        Path(results_dir_base)
        / model_name_str
        / "ood_generic"
        / "latent_performance"
        / sae_name_str
        / f"train_{train_dataset_name}"
        / f"test_{ood_test_dataset_name}"
    )
    specific_results_path.mkdir(parents=True, exist_ok=True)

    actual_max_seq_len = (
        max_seq_len_override if max_seq_len_override is not None else model.cfg.n_ctx
    )
    results_list = []

    # File for this specific run's results
    main_save_file = (
        specific_results_path
        / f"latent_perf_{method_name}_L{layer_idx}_hook_{hook_point_name.replace('.', '_')}.csv"
    )
    if main_save_file.exists():
        print(
            f"Latent performance results already exist at {main_save_file}, skipping."
        )
        # Optionally, load and return existing results
        # try:
        #     existing_df = pd.read_csv(main_save_file)
        #     return existing_df.to_dict('records')
        # except Exception as e:
        #     print(f"Could not load existing results: {e}")
        return []  # Or some indicator that it was skipped

    print(
        f"LatentPerf: Train {train_dataset_name}, Test {ood_test_dataset_name}, L{layer_idx}, SAE {sae_name_str}"
    )

    # 1. Get training data (for feature selection and training initial multi-feature probe)
    try:
        X_train_model_acts, _, y_train_labels, _ = get_model_activations_for_dataset(
            model=model,
            dataset_name=train_dataset_name,
            layer_idx=layer_idx,
            hook_point_name=hook_point_name,
            pooling_strategy="last",
            device=device,
            setting_type="normal",
            max_seq_len=actual_max_seq_len,
            cache_dir_base=Path(cache_dir_base),
            expected_activation_dim=model.cfg.d_model,
            seed=seed,
            test_set_ratio=0.2,  # Use a portion for training, rest for potential validation if method supports
        )
    except Exception as e:
        print(
            f"Error getting train activations for LatentPerf ({train_dataset_name}): {e}"
        )
        return [{"error": str(e), "step": "get_train_acts"}]

    if X_train_model_acts.shape[0] == 0 or len(torch.unique(y_train_labels)) < 2:
        print(
            f"No/insufficient training data or <2 classes for LatentPerf train ({train_dataset_name}). Skipping."
        )
        return [
            {"error": "Insufficient train data/classes", "step": "check_train_data"}
        ]

    # 2. Get OOD test data
    try:
        _, X_test_model_acts, _, y_test_labels = get_model_activations_for_dataset(
            model=model,
            dataset_name=ood_test_dataset_name,
            layer_idx=layer_idx,
            hook_point_name=hook_point_name,
            pooling_strategy="last",
            device=device,
            setting_type="normal",
            max_seq_len=actual_max_seq_len,
            cache_dir_base=Path(cache_dir_base),
            expected_activation_dim=model.cfg.d_model,
            seed=seed,
        )
    except Exception as e:
        print(
            f"Error getting OOD test activations for LatentPerf ({ood_test_dataset_name}): {e}"
        )
        return [{"error": str(e), "step": "get_ood_test_acts"}]

    if X_test_model_acts.shape[0] == 0 or len(torch.unique(y_test_labels)) < 2:
        print(
            f"No/insufficient OOD test data or <2 classes for LatentPerf test ({ood_test_dataset_name}). Skipping."
        )
        return [
            {
                "error": "Insufficient OOD test data/classes",
                "step": "check_ood_test_data",
            }
        ]

    # 3. Get SAE features for train and test sets
    try:
        X_train_sae_features_all, _ = get_sae_features(
            sae, X_train_model_acts, device=device
        )
        X_test_sae_features_all, _ = get_sae_features(
            sae, X_test_model_acts, device=device
        )
    except Exception as e:
        print(f"Error getting SAE features for LatentPerf: {e}")
        return [{"error": str(e), "step": "get_sae_features"}]

    if (
        X_train_sae_features_all.shape[1] == 0
        or X_test_sae_features_all.shape[1] == 0
        or X_train_sae_features_all.shape[1] != X_test_sae_features_all.shape[1]
    ):
        error_msg = (
            "0-dim SAE features"
            if X_train_sae_features_all.shape[1] == 0
            else "SAE feature dim mismatch"
        )
        print(f"{error_msg} for LatentPerf. Skipping.")
        return [{"error": error_msg, "step": "check_sae_dims"}]

    # 4. Select top N SAE features based on mean difference on the training set
    y_train_torch = (
        y_train_labels
        if isinstance(y_train_labels, torch.Tensor)
        else torch.from_numpy(y_train_labels)
    ).to(X_train_sae_features_all.device)
    mean_class1 = X_train_sae_features_all[y_train_torch == 1].mean(dim=0)
    mean_class0 = X_train_sae_features_all[y_train_torch == 0].mean(dim=0)
    diff_means = mean_class1 - mean_class0

    actual_num_to_analyze = min(
        num_top_sae_features_to_analyze, X_train_sae_features_all.shape[1]
    )
    top_k_indices_overall_sae = torch.argsort(torch.abs(diff_means), descending=True)[
        :actual_num_to_analyze
    ]

    X_train_sae_top_k = X_train_sae_features_all[:, top_k_indices_overall_sae]
    X_test_sae_top_k = X_test_sae_features_all[:, top_k_indices_overall_sae]

    y_train_np = (
        y_train_labels.cpu().numpy()
        if isinstance(y_train_labels, torch.Tensor)
        else y_train_labels
    )
    y_test_np = (
        y_test_labels.cpu().numpy()
        if isinstance(y_test_labels, torch.Tensor)
        else y_test_labels
    )
    method_func = methods_global[method_name]

    # 5. Train probe on all top_k selected SAE features
    coeffs_all_k = [np.nan] * actual_num_to_analyze  # Initialize placeholder
    try:
        X_train_all_top_k_np = X_train_sae_top_k.cpu().numpy()
        X_test_all_top_k_np = X_test_sae_top_k.cpu().numpy()
        metrics_all_k, classifier_all_k = method_func(
            X_train_all_top_k_np,
            y_train_np,
            X_test_all_top_k_np,
            y_test_np,
            return_classifier=True,
        )
        row_all_k = {
            "latent_feature_original_index": "all_top_k",
            "latent_feature_relative_rank": -1,
            "method": method_name,
            "train_dataset": train_dataset_name,
            "ood_test_dataset": ood_test_dataset_name,
            "sae_name": sae_name_str,
            "num_features_in_probe": actual_num_to_analyze,
        }
        row_all_k.update(metrics_all_k)
        results_list.append(row_all_k)
        if (
            hasattr(classifier_all_k, "coef_")
            and classifier_all_k.coef_ is not None
            and len(classifier_all_k.coef_) > 0
        ):
            coeffs_all_k = classifier_all_k.coef_[0]
    except Exception as e:
        print(f"Error training probe on all top_k features for LatentPerf: {e}")
        results_list.append(
            {
                "error": str(e),
                "step": "train_probe_all_top_k",
                "latent_feature_original_index": "all_top_k",
            }
        )

    # 6. Train probe on each of the top_k selected SAE features individually
    for i in tqdm(
        range(actual_num_to_analyze), desc="Individual Latent Feature Probes"
    ):
        original_feature_index = top_k_indices_overall_sae[i].item()
        X_train_single_feature = X_train_sae_top_k[:, i : i + 1].cpu().numpy()
        X_test_single_feature = X_test_sae_top_k[:, i : i + 1].cpu().numpy()

        current_coeff = np.nan
        if i < len(coeffs_all_k):
            current_coeff = coeffs_all_k[i]
        try:
            metrics_single = method_func(
                X_train_single_feature, y_train_np, X_test_single_feature, y_test_np
            )
            row_single = {
                "latent_feature_original_index": original_feature_index,
                "latent_feature_relative_rank": i,
                "method": method_name,
                "train_dataset": train_dataset_name,
                "ood_test_dataset": ood_test_dataset_name,
                "sae_name": sae_name_str,
                "num_features_in_probe": 1,
                "coeff_in_multi_feature_probe": current_coeff,
            }
            row_single.update(metrics_single)
            results_list.append(row_single)
        except Exception as e:
            print(
                f"Error training probe on single feature {original_feature_index} (rank {i}) for LatentPerf: {e}"
            )
            results_list.append(
                {
                    "error": str(e),
                    "step": f"train_probe_single_feature_rank_{i}",
                    "latent_feature_original_index": original_feature_index,
                    "latent_feature_relative_rank": i,
                    "coeff_in_multi_feature_probe": current_coeff,
                }
            )

    pd.DataFrame(results_list).to_csv(main_save_file, index=False)
    print(f"Saved Latent Performance results to {main_save_file}")
    return results_list


def ood_pruning_generic(
    model: HookedTransformer,
    sae: SAE,
    train_dataset_name: str,  # Dataset for fetching activations to be pruned
    ood_test_dataset_name: str,  # OOD Dataset to test the pruned probe on
    layer_idx: int,
    hook_point_name: str,
    method_name: str,  # e.g. "logreg"
    relevance_csv_path: str | Path,  # Path to CSV with feature 'latent' and 'Relevance'
    num_features_to_keep_iteratively: list[int],  # e.g. [1, 2, 3, 4, 5, 6, 7, 8]
    results_dir_base: str | Path,
    device: str | torch.device,
    max_seq_len_override: int | None = None,
    cache_dir_base: str | Path = "data/generated_model_activations",
    seed: int = 42,
) -> list[dict[str, Any]]:
    model_name_str = model.cfg.model_name.replace("/", "_")
    sae_name_str = getattr(sae.cfg, "sae_name", f"sae_L{layer_idx}").replace("/", "_")
    specific_results_path = (
        Path(results_dir_base)
        / model_name_str
        / "ood_generic"
        / "ood_pruning"
        / sae_name_str
        / f"train_{train_dataset_name}"
        / f"test_{ood_test_dataset_name}"
    )
    specific_results_path.mkdir(parents=True, exist_ok=True)

    actual_max_seq_len = (
        max_seq_len_override if max_seq_len_override is not None else model.cfg.n_ctx
    )
    results_list = []

    main_save_file = (
        specific_results_path
        / f"ood_pruning_{method_name}_L{layer_idx}_hook_{hook_point_name.replace('.', '_')}_relevance_{Path(relevance_csv_path).stem}.csv"
    )
    if main_save_file.exists():
        print(f"OOD Pruning results already exist at {main_save_file}, skipping.")
        return []

    print(
        f"OOD Pruning: Train {train_dataset_name}, Test {ood_test_dataset_name}, L{layer_idx}, SAE {sae_name_str}, Relevance CSV {relevance_csv_path}"
    )

    # 1. Load feature relevance data
    try:
        relevance_df = pd.read_csv(relevance_csv_path)
        if not all(col in relevance_df.columns for col in ["latent", "Relevance"]):
            raise ValueError(
                "Relevance CSV must contain 'latent' and 'Relevance' columns."
            )
        relevance_df = relevance_df.sort_values(
            "Relevance", ascending=True
        )  # Assuming lower relevance is pruned first (i.e. keep most relevant)
    except Exception as e:
        print(f"Error loading or validating relevance CSV {relevance_csv_path}: {e}")
        return [{"error": str(e), "step": "load_relevance_csv"}]

    # 2. Get training data
    try:
        X_train_model_acts, _, y_train_labels, _ = get_model_activations_for_dataset(
            model=model,
            dataset_name=train_dataset_name,
            layer_idx=layer_idx,
            hook_point_name=hook_point_name,
            pooling_strategy="last",
            device=device,
            setting_type="normal",
            max_seq_len=actual_max_seq_len,
            cache_dir_base=Path(cache_dir_base),
            expected_activation_dim=model.cfg.d_model,
            seed=seed,
            test_set_ratio=0.2,
        )
    except Exception as e:
        print(
            f"Error getting train activations for OOD Pruning ({train_dataset_name}): {e}"
        )
        return [{"error": str(e), "step": "get_train_acts"}]

    if X_train_model_acts.shape[0] == 0 or len(torch.unique(y_train_labels)) < 2:
        print(
            f"No/insufficient training data or <2 classes for OOD Pruning train ({train_dataset_name}). Skipping."
        )
        return [
            {"error": "Insufficient train data/classes", "step": "check_train_data"}
        ]

    # 3. Get OOD test data
    try:
        _, X_test_model_acts, _, y_test_labels = get_model_activations_for_dataset(
            model=model,
            dataset_name=ood_test_dataset_name,
            layer_idx=layer_idx,
            hook_point_name=hook_point_name,
            pooling_strategy="last",
            device=device,
            setting_type="normal",
            max_seq_len=actual_max_seq_len,
            cache_dir_base=Path(cache_dir_base),
            expected_activation_dim=model.cfg.d_model,
            seed=seed,
        )
    except Exception as e:
        print(
            f"Error getting OOD test activations for OOD Pruning ({ood_test_dataset_name}): {e}"
        )
        return [{"error": str(e), "step": "get_ood_test_acts"}]

    if X_test_model_acts.shape[0] == 0 or len(torch.unique(y_test_labels)) < 2:
        print(
            f"No/insufficient OOD test data or <2 classes for OOD Pruning test ({ood_test_dataset_name}). Skipping."
        )
        return [
            {
                "error": "Insufficient OOD test data/classes",
                "step": "check_ood_test_data",
            }
        ]

    # 4. Get SAE features for train and test sets
    try:
        X_train_sae_features_all, _ = get_sae_features(
            sae, X_train_model_acts, device=device
        )
        X_test_sae_features_all, _ = get_sae_features(
            sae, X_test_model_acts, device=device
        )
    except Exception as e:
        print(f"Error getting SAE features for OOD Pruning: {e}")
        return [{"error": str(e), "step": "get_sae_features"}]

    if (
        X_train_sae_features_all.shape[1] == 0
        or X_test_sae_features_all.shape[1] == 0
        or X_train_sae_features_all.shape[1] != X_test_sae_features_all.shape[1]
    ):
        error_msg = (
            "0-dim SAE features"
            if X_train_sae_features_all.shape[1] == 0
            else "SAE feature dim mismatch"
        )
        print(f"{error_msg} for OOD Pruning. Skipping.")
        return [{"error": error_msg, "step": "check_sae_dims"}]

    if X_train_sae_features_all.shape[1] < relevance_df["latent"].max() + 1:
        print(
            f"Warning: Max latent index in relevance CSV ({relevance_df['latent'].max()}) exceeds SAE dimension ({X_train_sae_features_all.shape[1]}). Check CSV and SAE compatibility."
        )
        # Filter relevance_df to only include latents within the SAE dimension
        relevance_df = relevance_df[
            relevance_df["latent"] < X_train_sae_features_all.shape[1]
        ]
        if relevance_df.empty:
            print(
                "No relevant features left after filtering by SAE dimension. Skipping OOD Pruning."
            )
            return [
                {
                    "error": "No relevant features after dim check",
                    "step": "filter_relevance_by_dim",
                }
            ]

    y_train_np = (
        y_train_labels.cpu().numpy()
        if isinstance(y_train_labels, torch.Tensor)
        else y_train_labels
    )
    y_test_np = (
        y_test_labels.cpu().numpy()
        if isinstance(y_test_labels, torch.Tensor)
        else y_test_labels
    )
    method_func = methods_global[method_name]

    # 5. Iterate, prune, and train probes
    for k_to_keep in tqdm(num_features_to_keep_iteratively, desc="Pruning Iterations"):
        if k_to_keep <= 0 or k_to_keep > relevance_df.shape[0]:
            print(
                f"Skipping k_to_keep = {k_to_keep} as it's out of bounds for available relevant features ({relevance_df.shape[0]})."
            )
            continue

        # Get top k most relevant latents (higher relevance scores are better, so sort descending after initial ascending for pruning)
        # The original code sorted by Relevance and took head(k). If Relevance means importance, sort descending.
        # The old code description says "prune the least helpful latents". So sort ascending by relevance and take tail(k) or sort descending and take head(k).
        # The provided relevance_df is sorted ascending by relevance (least helpful first). So we take tail(k) for most relevant.
        # OR, if relevance is higher = better, then relevance_df.sort_values("Relevance", ascending=False).head(k)
        # Let's assume higher relevance score = better, so sort descending.
        top_k_relevant_latents_df = relevance_df.sort_values(
            "Relevance", ascending=False
        ).head(k)
        indices_to_keep = top_k_relevant_latents_df["latent"].values.astype(int)

        # Ensure indices are within bounds of the actual SAE dimension
        indices_to_keep = indices_to_keep[
            indices_to_keep < X_train_sae_features_all.shape[1]
        ]
        if indices_to_keep.size == 0:
            print(
                f"No valid features to keep for k={k_to_keep} after ensuring indices are within SAE dimension bounds. Skipping this k."
            )
            continue

        X_train_pruned = X_train_sae_features_all[:, indices_to_keep].cpu().numpy()
        X_test_pruned = X_test_sae_features_all[:, indices_to_keep].cpu().numpy()

        try:
            metrics = method_func(X_train_pruned, y_train_np, X_test_pruned, y_test_np)
            row = {
                "k_features_kept_by_relevance": k_to_keep,
                "actual_features_in_probe": indices_to_keep.size,
                "method": method_name,
                "train_dataset": train_dataset_name,
                "ood_test_dataset": ood_test_dataset_name,
                "sae_name": sae_name_str,
                "relevance_csv": Path(relevance_csv_path).name,
            }
            row.update(metrics)
            results_list.append(row)
        except Exception as e:
            print(f"Error training probe for OOD Pruning (k={k_to_keep}): {e}")
            results_list.append(
                {
                    "error": str(e),
                    "step": f"train_probe_k_{k_to_keep}",
                    "k_features_kept_by_relevance": k_to_keep,
                }
            )

    pd.DataFrame(results_list).to_csv(main_save_file, index=False)
    print(f"Saved OOD Pruning results to {main_save_file}")
    return results_list


def examine_glue_classifier():
    # finds the prompts where baseline classifier most disagrees with
    # the given label. This allows us to see if baselines are able
    # to find incorrect labels as well. Table 7
    X_train, y_train, X_test, y_test_og = get_glue_traintest(toget="original_target")
    _, _, _, y_test_ens = get_glue_traintest(toget="ensemble")
    _, classifier = find_best_reg(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test_og,
        penalty="l2",
        return_classifier=True,
    )
    # Get predictions and probabilities
    probs = classifier.predict_proba(X_test)
    prob_class_0 = probs[:, 0]  # Probability of class 0

    # Find indices where actual is 1 but model is most confident it should be 0
    mask = y_test_og == 1
    confident_wrong_indices = np.argsort(prob_class_0[mask])[-5:][
        ::-1
    ]  # Top 5 most confident mistakes
    actual_indices = np.where(mask)[0][confident_wrong_indices]

    # Load prompts
    df = pd.read_csv("results/investigate/87_glue_cola_investigate.csv")
    prompts = df["prompt"].tolist()

    # Print table header
    print("\nMost confident mistakes (where true label is 1 but model predicts 0)")
    print("-" * 100)
    print(f"{'Prompt':<60} | {'P(y=0)':<10} | {'Original':<8} | {'Ensemble':<8}")
    print("-" * 100)

    # Print each row
    for idx in actual_indices:
        print(
            f"{prompts[idx]:<60} | {prob_class_0[idx]:.3f}     | {y_test_og[idx]:<8} | {y_test_ens[idx]:<8}"
        )


# --- GLUE Generic Function ---
def run_glue_generic(
    model: HookedTransformer,
    sae: SAE | None,  # If SAE is provided, run SAE probe, else baseline
    glue_dataset_name: str,  # e.g., "87_glue_cola"
    layer_idx: int,
    hook_point_name: str,  # e.g., from sae.cfg.hook_point or a standard residual stream hook
    method_name: str,  # e.g. "logreg"
    test_label_type: str,  # "original_target", "ensemble", or "disagree"
    results_dir_base: str | Path,
    device: str | torch.device,
    sae_k: int | None = 128,  # k for SAE feature selection if SAE is used
    max_seq_len_override: int | None = None,
    cache_dir_base: str | Path = "data/generated_model_activations",
    seed: int = 42,
    train_test_split_ratio_for_glue_train: float = 0.8,  # How much of GLUE dataset for training probe
) -> dict[str, Any] | None:
    model_name_str = model.cfg.model_name.replace("/", "_")
    results_dir = Path(results_dir_base) / model_name_str / "glue_generic"
    results_dir.mkdir(parents=True, exist_ok=True)

    actual_max_seq_len = (
        max_seq_len_override if max_seq_len_override is not None else model.cfg.n_ctx
    )

    probe_type_str = "sae" if sae else "baseline"
    sae_name_for_path = (
        getattr(sae.cfg, "sae_name", f"sae_L{layer_idx}").replace("/", "_")
        if sae
        else "no_sae"
    )

    save_filename = f"glue_{probe_type_str}_{model_name_str}_L{layer_idx}_hook_{hook_point_name.replace('.', '_')}_{glue_dataset_name}_testlabel_{test_label_type}_{sae_name_for_path}.csv"
    save_path_glue = results_dir / save_filename

    if save_path_glue.exists():
        print(f"Skipping existing GLUE result: {save_path_glue}")
        # Optionally load and return if needed, for now just skip
        # df_existing = pd.read_csv(save_path_glue)
        # return df_existing.to_dict('records')[0]
        return None

    print(
        f"GLUE Run: {glue_dataset_name}, Layer {layer_idx}, Hook {hook_point_name}, Probe: {probe_type_str}, TestLabels: {test_label_type}"
    )

    # 1. Get training activations and labels from the GLUE dataset's standard training split
    try:
        # We need a defined training set from the GLUE data.
        # get_model_activations_for_dataset will perform a train/test split internally.
        # We use `train_test_split_ratio_for_glue_train` to define the proportion for this internal split.
        # The X_train, y_train from this will be our probe training data.
        # The X_test, y_test from this initial split are not directly used for the final GLUE test types,
        # but the split needs to happen to get a distinct training set.
        (
            X_train_model_acts,
            X_test_pool_model_acts,
            y_train_labels,
            y_test_pool_labels,
        ) = get_model_activations_for_dataset(
            model=model,
            dataset_name=glue_dataset_name,
            layer_idx=layer_idx,
            hook_point_name=hook_point_name,
            pooling_strategy="last",  # Assuming pooled for these probes
            device=device,
            setting_type="normal",
            max_seq_len=actual_max_seq_len,
            cache_dir_base=Path(cache_dir_base),
            expected_activation_dim=model.cfg.d_model,
            seed=seed,
            test_set_ratio=1.0
            - train_test_split_ratio_for_glue_train,  # Proportion for the test pool
        )
    except Exception as e:
        print(
            f"Error getting model activations for GLUE train ({glue_dataset_name}, L{layer_idx}): {e}. Skipping GLUE run."
        )
        # Log error to file
        pd.DataFrame([{"error": str(e), "step": "get_train_acts"}]).to_csv(
            save_path_glue, index=False
        )
        return None

    if X_train_model_acts.shape[0] == 0:
        print(f"No training data for GLUE base: {glue_dataset_name}. Skipping.")
        pd.DataFrame(
            [{"error": "No training data generated", "step": "check_train_data"}]
        ).to_csv(save_path_glue, index=False)
        return None

    # Prepare X_train for the probe
    current_X_train_probe = X_train_model_acts
    X_train_sae_features_full_dim = (
        None  # For storing full dim SAE features if k-selection applied
    )

    if sae:
        try:
            X_train_sae_features_full_dim, _ = get_sae_features(
                sae, X_train_model_acts, device=device
            )
            if X_train_sae_features_full_dim.shape[1] == 0:
                print(
                    f"SAE features for GLUE train ({glue_dataset_name}) have 0 dimension. Skipping."
                )
                pd.DataFrame(
                    [
                        {
                            "error": "0-dim SAE train features",
                            "step": "get_sae_train_features",
                        }
                    ]
                ).to_csv(save_path_glue, index=False)
                return None

            if (
                sae_k is not None
                and sae_k > 0
                and X_train_sae_features_full_dim.shape[1] > sae_k
            ):
                if len(torch.unique(y_train_labels)) < 2:
                    print(
                        "Warning: y_train for GLUE base has <2 unique classes. Using first k SAE features."
                    )
                    top_indices_train = torch.arange(
                        min(sae_k, X_train_sae_features_full_dim.shape[1])
                    )
                else:
                    y_train_torch = (
                        y_train_labels
                        if isinstance(y_train_labels, torch.Tensor)
                        else torch.from_numpy(y_train_labels)
                    )
                    diff = X_train_sae_features_full_dim[y_train_torch == 1].mean(
                        0
                    ) - X_train_sae_features_full_dim[y_train_torch == 0].mean(0)
                    top_indices_train = torch.argsort(torch.abs(diff), descending=True)[
                        :sae_k
                    ]
                current_X_train_probe = X_train_sae_features_full_dim[
                    :, top_indices_train
                ]
            else:
                current_X_train_probe = X_train_sae_features_full_dim
        except Exception as e:
            print(
                f"Error getting SAE features for GLUE training ({glue_dataset_name}): {e}. Skipping."
            )
            pd.DataFrame(
                [{"error": str(e), "step": "process_sae_train_features"}]
            ).to_csv(save_path_glue, index=False)
            return None

    # 2. Prepare Test Data (X_test_probe, y_test_final_labels)
    # The X_test_pool_model_acts and y_test_pool_labels are from the initial split.
    # We use X_test_pool_model_acts as the source for X data.
    # The y_labels will be determined by test_label_type using helpers.

    if X_test_pool_model_acts.shape[0] == 0:
        print(
            f"No data in test pool for GLUE dataset: {glue_dataset_name} after initial split. Skipping probe."
        )
        pd.DataFrame(
            [{"error": "No test pool data", "step": "check_test_pool"}]
        ).to_csv(save_path_glue, index=False)
        return None

    y_test_final_labels: np.ndarray
    final_test_indices: np.ndarray | None = None

    if test_label_type == "disagree":
        disagree_indices_relative_to_full_investigation_df = get_disagree_glue_indices(
            glue_dataset_name
        )
        # We need to map these disagree_indices (which are for the *entire* GLUE investigation CSV)
        # to the y_test_pool_labels we have from get_model_activations_for_dataset's test split.
        # This is tricky unless get_model_activations_for_dataset returns original indices or the investigation CSV aligns perfectly with its output ordering.
        # For now, a simplification: assume y_test_pool_labels corresponds to the *full* GLUE test set or a known part of it that we can get labels for.
        # And that disagree_indices can be directly used if y_test_pool_labels *is* the full set of ensemble/original labels from investigation_df.
        # This part is fragile and depends heavily on how get_model_activations_for_dataset structures its test split for GLUE.
        # A more robust approach: load all prompts for GLUE test, run model, then filter.
        # For now, let's assume that the y_test_pool_labels *are* the "original_target" labels for the test pool.
        # And we need the "ensemble" labels for these same examples to find disagreement within this pool.
        # This requires `get_glue_labels` to be applicable to this subset if the investigation df is global.
        # This logic is getting complicated due to ambiguity of splits.
        # A simpler path for now: Assume get_disagree_glue_indices gives indices for the *full dataset*.
        # We need to use these indices on activations that correspond to the *full dataset*.
        # Let's re-fetch full activations for the test set, and then select.
        print(
            "Re-fetching full test set activations for GLUE disagree case to ensure index alignment."
        )
        try:
            _, X_test_full_model_acts, _, y_test_full_original_labels = (
                get_model_activations_for_dataset(
                    model=model,
                    dataset_name=glue_dataset_name,
                    layer_idx=layer_idx,
                    hook_point_name=hook_point_name,
                    pooling_strategy="last",
                    device=device,
                    setting_type="normal",
                    max_seq_len=actual_max_seq_len,
                    cache_dir_base=Path(cache_dir_base),
                    expected_activation_dim=model.cfg.d_model,
                    seed=seed,
                    test_set_ratio=0.99999,  # Try to get as many as possible, assuming it is the test part.
                )
            )
            if X_test_full_model_acts.shape[0] == 0:
                raise ValueError("X_test_full_model_acts is empty for disagree case")
        except Exception as e:
            print(f"Error re-fetching full test acts for GLUE disagree: {e}")
            pd.DataFrame(
                [{"error": str(e), "step": "refetch_full_test_acts_disagree"}]
            ).to_csv(save_path_glue, index=False)
            return None

        final_test_indices = get_disagree_glue_indices(glue_dataset_name)
        if final_test_indices.size == 0:
            print(
                f"No disagreeing examples found for GLUE {glue_dataset_name}. Skipping."
            )
            pd.DataFrame(
                [{"error": "No disagreeing examples", "step": "get_disagree_indices"}]
            ).to_csv(save_path_glue, index=False)
            return None

        # Ensure indices are within bounds of X_test_full_model_acts
        if np.max(final_test_indices) >= X_test_full_model_acts.shape[0]:
            print(
                f"Warning: Disagree indices out of bounds for X_test_full_model_acts. Max index: {np.max(final_test_indices)}, X_test shape: {X_test_full_model_acts.shape[0]}"
            )
            # This implies a mismatch between investigate.csv and the dataset used for activations.
            # Truncate indices to be safe, or error out.
            final_test_indices = final_test_indices[
                final_test_indices < X_test_full_model_acts.shape[0]
            ]
            if final_test_indices.size == 0:
                print("No valid disagreeing examples after bounds check. Skipping.")
                pd.DataFrame(
                    [
                        {
                            "error": "No valid disagreeing examples after bounds check",
                            "step": "check_disagree_indices_bounds",
                        }
                    ]
                ).to_csv(save_path_glue, index=False)
                return None

        X_test_model_acts_subset = X_test_full_model_acts[final_test_indices]
        # For disagree, the labels are typically the 'ensemble' (corrected) ones.
        y_test_final_labels = get_glue_labels(glue_dataset_name, "ensemble")[
            final_test_indices
        ]
        # Verify lengths
        if X_test_model_acts_subset.shape[0] != y_test_final_labels.shape[0]:
            print(
                "Mismatch length X_test_model_acts_subset and y_test_final_labels for disagree. Skipping."
            )
            pd.DataFrame(
                [
                    {
                        "error": "Mismatch length X/y for disagree",
                        "step": "verify_disagree_lengths",
                    }
                ]
            ).to_csv(save_path_glue, index=False)
            return None

    elif test_label_type in ["original_target", "ensemble"]:
        # Use the X_test_pool_model_acts from the initial split. The corresponding y_labels need to be fetched.
        # This assumes the investigation CSV aligns with the full dataset from which X_test_pool_model_acts was drawn.
        # This is still potentially fragile. A cleaner way: get ALL prompts, ALL labels from investigate.csv,
        # then split train/test indices, then generate activations for only those splits.
        # For now, stick to using X_test_pool_model_acts.
        # We need to ensure y_labels match the X_test_pool_model_acts. This requires careful index handling if test_pool is a subset.
        # The current get_glue_labels fetches for the *entire* investigation CSV. We need to select based on test_pool_indices.
        # This is where `y_test_pool_labels` from `get_model_activations_for_dataset` becomes important. It should be the "original_target" labels for the test pool.

        # If test_label_type is "original_target", we use y_test_pool_labels directly.
        if test_label_type == "original_target":
            y_test_final_labels = (
                y_test_pool_labels.cpu().numpy()
                if isinstance(y_test_pool_labels, torch.Tensor)
                else y_test_pool_labels
            )
        else:  # "ensemble"
            # This is the problematic part: get_glue_labels returns for the whole investigation CSV.
            # We need to align it with y_test_pool_labels / X_test_pool_model_acts.
            # This requires knowing which original indices X_test_pool_model_acts corresponds to.
            # For now, we make a simplifying assumption: run get_model_activations_for_dataset to get the *ENTIRE* dataset as test set
            # Then select labels. This is inefficient but safer for indexing.
            print(
                f"GLUE {test_label_type}: Re-fetching full dataset activations to align labels."
            )
            try:
                _, X_test_full_model_acts, _, _ = get_model_activations_for_dataset(
                    model=model,
                    dataset_name=glue_dataset_name,
                    layer_idx=layer_idx,
                    hook_point_name=hook_point_name,
                    pooling_strategy="last",
                    device=device,
                    setting_type="normal",
                    max_seq_len=actual_max_seq_len,
                    cache_dir_base=Path(cache_dir_base),
                    expected_activation_dim=model.cfg.d_model,
                    seed=seed,
                    test_set_ratio=0.99999,  # Get all/most as test set
                )
                if X_test_full_model_acts.shape[0] == 0:
                    raise ValueError(
                        "X_test_full_model_acts is empty for ensemble/original case"
                    )
            except Exception as e:
                print(f"Error re-fetching full acts for GLUE {test_label_type}: {e}")
                pd.DataFrame(
                    [
                        {
                            "error": str(e),
                            "step": f"refetch_full_test_acts_{test_label_type}",
                        }
                    ]
                ).to_csv(save_path_glue, index=False)
                return None

            X_test_model_acts_subset = X_test_full_model_acts
            y_test_final_labels = get_glue_labels(glue_dataset_name, test_label_type)
            # Ensure y_test_final_labels aligns with X_test_model_acts_subset (which should be all examples from investigate.csv)
            if X_test_model_acts_subset.shape[0] != y_test_final_labels.shape[0]:
                print(
                    f"Warning: Length mismatch between full X_test activations ({X_test_model_acts_subset.shape[0]}) and {test_label_type} labels ({y_test_final_labels.shape[0]}). Truncating to shortest."
                )
                min_len = min(
                    X_test_model_acts_subset.shape[0], y_test_final_labels.shape[0]
                )
                X_test_model_acts_subset = X_test_model_acts_subset[:min_len]
                y_test_final_labels = y_test_final_labels[:min_len]
                if min_len == 0:
                    print("Zero length after truncation. Skipping GLUE run.")
                    pd.DataFrame(
                        [
                            {
                                "error": "Zero length X/y after truncation",
                                "step": "truncate_Xy_glue",
                            }
                        ]
                    ).to_csv(save_path_glue, index=False)
                    return None
    else:
        raise ValueError(f"Unknown test_label_type for GLUE: {test_label_type}")

    current_X_test_probe = X_test_model_acts_subset
    if sae:
        try:
            X_test_sae_features, _ = get_sae_features(
                sae, X_test_model_acts_subset, device=device
            )
            if X_test_sae_features.shape[1] == 0:
                print(
                    f"SAE features for GLUE test ({glue_dataset_name}, {test_label_type}) have 0 dimension. Skipping."
                )
                pd.DataFrame(
                    [
                        {
                            "error": "0-dim SAE test features",
                            "step": "get_sae_test_features",
                        }
                    ]
                ).to_csv(save_path_glue, index=False)
                return None

            if (
                sae_k is not None
                and sae_k > 0
                and X_train_sae_features_full_dim is not None
                and X_train_sae_features_full_dim.shape[1] > sae_k
            ):
                # Use top_indices_train derived from the training set SAE features
                current_X_test_probe = X_test_sae_features[:, top_indices_train]
            else:
                current_X_test_probe = X_test_sae_features
        except Exception as e:
            print(
                f"Error getting SAE features for GLUE testing ({glue_dataset_name}, {test_label_type}): {e}. Skipping."
            )
            pd.DataFrame(
                [{"error": str(e), "step": "process_sae_test_features"}]
            ).to_csv(save_path_glue, index=False)
            return None

    # Ensure data is numpy for sklearn methods
    current_X_train_np = (
        current_X_train_probe.cpu().numpy()
        if isinstance(current_X_train_probe, torch.Tensor)
        else current_X_train_probe
    )
    y_train_np = (
        y_train_labels.cpu().numpy()
        if isinstance(y_train_labels, torch.Tensor)
        else y_train_labels
    )
    current_X_test_np = (
        current_X_test_probe.cpu().numpy()
        if isinstance(current_X_test_probe, torch.Tensor)
        else current_X_test_probe
    )
    # y_test_final_labels is already numpy

    if current_X_train_np.shape[0] == 0 or current_X_test_np.shape[0] == 0:
        print("Train or Test data is empty before running probe. Skipping GLUE run.")
        pd.DataFrame(
            [
                {
                    "error": "Empty train/test for probe",
                    "train_N": current_X_train_np.shape[0],
                    "test_N": current_X_test_np.shape[0],
                }
            ]
        ).to_csv(save_path_glue, index=False)
        return None

    # Run Probe
    method_func = methods_global[method_name]
    try:
        metrics = method_func(
            current_X_train_np, y_train_np, current_X_test_np, y_test_final_labels
        )
    except Exception as e:
        print(
            f"Error running GLUE probe ({method_name} on {glue_dataset_name}, test type {test_label_type}): {e}"
        )
        metrics = {"error": str(e)}

    row = {
        "glue_dataset": glue_dataset_name,
        "layer": layer_idx,
        "probe_type": probe_type_str,
        "method": method_name,
        "hook_point": hook_point_name,
        "test_label_type": test_label_type,
        "sae_name": sae_name_for_path if sae else "N/A",
        "sae_k_features": sae_k
        if sae
        and sae_k is not None
        and X_train_sae_features_full_dim is not None
        and X_train_sae_features_full_dim.shape[1] > sae_k
        else ("all_sae_features" if sae else "N/A"),
        "num_train_samples": current_X_train_np.shape[0],
        "num_test_samples": current_X_test_np.shape[0],
    }
    row.update(metrics)
    pd.DataFrame([row]).to_csv(save_path_glue, index=False)
    print(f"Saved GLUE result to {save_path_glue}")
    return row
