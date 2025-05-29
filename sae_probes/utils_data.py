import glob
import os
import pickle as pkl

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformer_lens import HookedTransformer

try:
    from IPython import get_ipython  # type: ignore

    ipython = get_ipython()
    assert ipython is not None
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")

    is_notebook = True
except:
    is_notebook = False


# DATA UTILS
def get_binary_df() -> pd.DataFrame:
    # returns a list of the data tags for all binary classification datasets
    try:
        df = pd.read_csv(
            os.path.join(
                os.path.dirname(__file__), "..", "data", "probing_datasets_MASTER.csv"
            )
        )
    except FileNotFoundError:
        # Fallback for when running tests or if structure is different
        # This path might need adjustment depending on execution context (e.g. running from root vs package internal)
        # A more robust solution might involve package resources or a configurable data root.
        alt_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "data", "probing_datasets_MASTER.csv"
        )  # Assuming data is one level above package root for tests
        try:
            df = pd.read_csv(alt_path)
        except FileNotFoundError:
            raise FileNotFoundError(
                "probing_datasets_MASTER.csv not found. Searched in expected locations."
            )

    # Filter for Binary Classification datasets
    binary_datasets = df[df["Data type"] == "Binary Classification"]
    return binary_datasets


def get_numbered_binary_tags() -> list[str]:
    df = get_binary_df()
    # Ensure "Dataset save name" column exists
    if "Dataset save name" not in df.columns:
        raise ValueError(
            "'Dataset save name' column not found in probing_datasets_MASTER.csv"
        )
    return [str(name).split("/")[-1].split(".")[0] for name in df["Dataset save name"]]


def read_dataset_df(dataset_tag: str) -> pd.DataFrame:
    df = get_binary_df()
    # Find the dataset save name for the given dataset tag
    # Ensure "Dataset Tag" and "Dataset save name" columns exist
    if "Dataset Tag" not in df.columns or "Dataset save name" not in df.columns:
        raise ValueError(
            "Required columns ('Dataset Tag' or 'Dataset save name') not in probing_datasets_MASTER.csv"
        )

    save_name_series = df[df["Dataset Tag"] == dataset_tag]["Dataset save name"]
    if save_name_series.empty:
        raise ValueError(
            f"Dataset tag '{dataset_tag}' not found in probing_datasets_MASTER.csv"
        )
    dataset_save_name = save_name_series.iloc[0]

    # Path to actual dataset CSVs needs to be robust
    # Assume they are in a 'data/' directory relative to where probing_datasets_MASTER.csv was found
    # (or a globally known data root)
    master_csv_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "probing_datasets_MASTER.csv"
    )  # Default assumption
    if not os.path.exists(master_csv_path):
        master_csv_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "data", "probing_datasets_MASTER.csv"
        )

    if not os.path.exists(master_csv_path):  # if still not found
        data_dir = "data/"  # A default guess
        print(
            f"Warning: Could not reliably determine data directory from probing_datasets_MASTER.csv location. Assuming data files are in '{data_dir}'"
        )
    else:
        data_dir = os.path.dirname(
            master_csv_path
        )  # Directory containing the master CSV
        # dataset_save_name might be like "binary_classification_datasets/1_ai_news.csv"
        # so we join data_dir with it.

    full_dataset_path = os.path.join(data_dir, dataset_save_name)

    try:
        return pd.read_csv(full_dataset_path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Dataset file {dataset_save_name} (path: {full_dataset_path}) not found."
        )


def read_numbered_dataset_df(numbered_dataset_tag: str) -> pd.DataFrame:
    if not isinstance(numbered_dataset_tag, str) or "_" not in numbered_dataset_tag:
        raise ValueError(
            f"Invalid numbered_dataset_tag: '{numbered_dataset_tag}'. Expected format like '1_dataset_name'."
        )
    dataset_tag = "_".join(numbered_dataset_tag.split("_")[1:])
    return read_dataset_df(dataset_tag)


def get_yvals(numbered_dataset_tag: str) -> np.ndarray:
    """Reads a dataset and returns only the (label encoded) target values."""
    df = read_numbered_dataset_df(numbered_dataset_tag)
    if "target" not in df.columns:
        raise ValueError(f"'target' column not found in dataset {numbered_dataset_tag}")
    le = LabelEncoder()
    yvals = le.fit_transform(df["target"].values)
    return yvals.astype(np.int64)


def get_train_test_indices(
    y: np.ndarray,
    num_train: int,
    num_test: int,
    pos_ratio_train: float = 0.5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    np.random.seed(seed)

    pos_indices_all = np.where(y == 1)[0]
    neg_indices_all = np.where(y == 0)[0]

    if len(pos_indices_all) == 0 or len(neg_indices_all) == 0:
        raise ValueError(
            "Dataset must contain samples from both classes for splitting."
        )

    # Calculate number of positive/negative samples for training
    # Ensure we don't request more samples than available for a class.
    num_pos_train = min(int(np.ceil(pos_ratio_train * num_train)), len(pos_indices_all))
    num_neg_train = min(num_train - num_pos_train, len(neg_indices_all))

    # Adjust num_pos_train if num_neg_train had to be capped and vice-versa, to try to meet num_train
    if num_neg_train < (num_train - num_pos_train):  # if neg_train was capped
        num_pos_train = min(num_train - num_neg_train, len(pos_indices_all))

    if num_pos_train + num_neg_train != num_train:
        print(
            f"Warning: Could not achieve exactly {num_train} training samples with pos_ratio {pos_ratio_train} due to class size limits. Actual train size: {num_pos_train + num_neg_train}"
        )

    train_pos_indices = np.random.choice(
        pos_indices_all, size=num_pos_train, replace=False
    )
    train_neg_indices = np.random.choice(
        neg_indices_all, size=num_neg_train, replace=False
    )

    train_indices = np.concatenate([train_pos_indices, train_neg_indices])
    np.random.shuffle(train_indices)

    # For test set, use remaining samples
    remaining_pos = np.setdiff1d(pos_indices_all, train_pos_indices)
    remaining_neg = np.setdiff1d(neg_indices_all, train_neg_indices)

    if len(remaining_pos) == 0 or len(remaining_neg) == 0:
        # This can happen if training set took all samples of one class.
        # In this case, the test set might not be representative or usable for some metrics.
        print(
            "Warning: Not enough remaining samples for one class to form a diverse test set for dataset involving y."
        )
        # Fallback: if test set cannot be balanced, take what's available up to num_test
        num_pos_test = min(
            len(remaining_pos), num_test // 2 if len(remaining_neg) > 0 else num_test
        )
        num_neg_test = min(len(remaining_neg), num_test - num_pos_test)

    else:
        # Try to keep test set balanced, or use pos_ratio_train as a guide if num_test is small
        num_pos_test = min(int(np.ceil(pos_ratio_train * num_test)), len(remaining_pos))
        num_neg_test = min(num_test - num_pos_test, len(remaining_neg))
        # Adjust if one class capped
        if num_neg_test < (num_test - num_pos_test):
            num_pos_test = min(num_test - num_neg_test, len(remaining_pos))

    test_pos_indices = np.random.choice(remaining_pos, size=num_pos_test, replace=False)
    test_neg_indices = np.random.choice(remaining_neg, size=num_neg_test, replace=False)

    test_indices = np.concatenate([test_pos_indices, test_neg_indices])
    np.random.shuffle(test_indices)

    return train_indices, test_indices


def get_dataset_sizes() -> dict[str, int]:
    dataset_tags = get_numbered_binary_tags()
    dataset_s: dict[str, int] = {}
    for dataset_tag in dataset_tags:
        try:
            df = read_numbered_dataset_df(dataset_tag)
            dataset_s[dataset_tag] = len(df)
        except Exception as e:
            print(f"Warning: Could not read or get size for dataset {dataset_tag}: {e}")
    return dataset_s


def get_training_sizes() -> list[int]:
    min_size, max_size, num_points = 1, 10, 20
    points = np.unique(
        np.round(np.logspace(min_size, max_size, num=num_points, base=2)).astype(int)
    )
    return points.tolist()


def get_class_imbalance() -> list[float]:
    min_frac, max_frac, num_points = 0.05, 0.95, 19
    points = np.linspace(min_frac, max_frac, num=num_points)
    return points.tolist()


def corrupt_ytrain(
    ytrain: np.ndarray, frac_to_corrupt: float, seed: int = 42
) -> np.ndarray:
    if not (0 <= frac_to_corrupt <= 0.5):
        raise ValueError(
            f"Corruption fraction must be between 0 and 0.5, got {frac_to_corrupt}"
        )
    np.random.seed(seed)

    num_to_flip = int(len(ytrain) * frac_to_corrupt)
    if num_to_flip == 0 and frac_to_corrupt > 0:
        if len(ytrain) > 0:
            num_to_flip = 1 if len(ytrain) * frac_to_corrupt > 0 else 0

    flip_indices = np.random.choice(len(ytrain), size=num_to_flip, replace=False)

    ytrain_corrupted = ytrain.copy()
    ytrain_corrupted[flip_indices] = 1 - ytrain_corrupted[flip_indices]
    return ytrain_corrupted


def get_corrupt_frac() -> list[float]:
    min_frac, max_frac, num_points = 0.0, 0.5, 11
    points = np.linspace(min_frac, max_frac, num=num_points)
    return points.tolist()


def get_OOD_datasets(translation: bool = True) -> list[str]:
    # This should return list of dataset_names (e.g. "66_living-room")
    # The paths like "data/OOD data/*.csv" need to be robust.
    base_data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "OOD data")
    if not os.path.isdir(base_data_dir):
        base_data_dir = os.path.join(
            os.path.dirname(__file__), "..", "..", "data", "OOD data"
        )  # Try one level higher for tests
    if not os.path.isdir(base_data_dir):
        print(
            f"Warning: OOD data directory not found at {base_data_dir}. Returning empty list."
        )
        return []

    dataset_files = glob.glob(os.path.join(base_data_dir, "*.csv"))

    datasets_found: list[str] = []
    for path in dataset_files:
        name = os.path.basename(path).replace("_OOD.csv", "")
        if not translation and "translation" in name:
            continue
        datasets_found.append(name)
    return datasets_found


def get_glue_investigation_df(dataset_name: str = "87_glue_cola") -> pd.DataFrame:
    """Loads the GLUE investigation dataframe, typically for CoLA.
    Path is relative to project root/results/investigate/.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    investigate_file_name = f"{dataset_name}_investigate.csv"
    # Primary expected path: <project_root>/results/investigate/<file_name>
    path1 = os.path.join(project_root, "results", "investigate", investigate_file_name)

    # Alternative path: if `sae_probes` is a subdir and `results` is a peer to `sae_probes` parent.
    # e.g. /some_dir/SAE-Probes/results/investigate/ vs /some_dir/SAE-Probes/sae_probes/utils_data.py
    # project_root would be SAE-Probes/ then parent is /some_dir/
    path2 = os.path.join(
        os.path.dirname(project_root), "results", "investigate", investigate_file_name
    )

    # Fallback: try assuming current working directory is project root
    path3 = os.path.join("results", "investigate", investigate_file_name)

    potential_paths = [path1, path2, path3]

    for p in potential_paths:
        if os.path.exists(p):
            try:
                df = pd.read_csv(p)
                return df
            except Exception as e:
                print(f"Error reading GLUE investigate CSV {p}: {e}. Trying next path.")
                continue

    raise FileNotFoundError(
        f"GLUE investigation CSV '{investigate_file_name}' not found. Searched in: {potential_paths}. Ensure the file exists."
    )


def get_glue_labels(
    dataset_name: str = "87_glue_cola", label_type: str = "ensemble"
) -> np.ndarray:
    """Fetches specific labels (e.g., 'original_target', 'ensemble') from a GLUE investigation CSV."""
    df = get_glue_investigation_df(dataset_name)

    if df.empty:
        print(
            f"Warning: GLUE investigation data for '{dataset_name}' is empty. Cannot fetch labels of type '{label_type}'."
        )
        return np.array([], dtype=np.int64)

    if label_type not in df.columns:
        raise ValueError(
            f"Label type '{label_type}' not found as a column in GLUE investigation data for {dataset_name}. Available columns: {df.columns.tolist()}"
        )

    return df[label_type].values.astype(np.int64)


def get_disagree_glue_indices(dataset_name: str = "87_glue_cola") -> np.ndarray:
    """Returns indices where 'original_target' and 'ensemble' labels disagree in GLUE investigation data."""
    df = get_glue_investigation_df(dataset_name)

    if df.empty:
        print(
            f"Warning: GLUE investigation data for '{dataset_name}' is empty. Cannot find disagreeing indices."
        )
        return np.array([], dtype=np.int64)

    if "original_target" not in df.columns or "ensemble" not in df.columns:
        print(
            f"Warning: 'original_target' or 'ensemble' columns not found in GLUE investigate data for {dataset_name}. Cannot find disagreeing indices. Available columns: {df.columns.tolist()}"
        )
        return np.array([], dtype=np.int64)

    return np.where(df["original_target"] != df["ensemble"])[0].astype(np.int64)


# --- New Core Function: get_model_activations_for_dataset ---
@torch.no_grad()
def _generate_activations_from_prompts(
    model: HookedTransformer,
    prompts: list[str],
    tokenizer,
    layer_idx: int,
    device: str | torch.device,
    max_seq_len: int,
    batch_size: int = 32,
    hook_point_name: str | None = None,
    pooling_strategy: str = "last",
) -> torch.Tensor:
    model.eval()
    model.to(device)
    all_model_acts_list: list[torch.Tensor] = []

    if hook_point_name is None:
        hook_point = f"blocks.{layer_idx}.hook_resid_pre"
        if hook_point not in [hp.name for hp in model.hook_points()]:
            alt_hook_point = f"blocks.{layer_idx}.hook_resid_post"
            if alt_hook_point in [hp.name for hp in model.hook_points()]:
                print(
                    f"Warning: Default hook point {hook_point} not found. Using {alt_hook_point} instead."
                )
                hook_point = alt_hook_point
            else:
                raise ValueError(
                    f"Default hook point '{hook_point}' (and fallback '{alt_hook_point}') not found in model. Please provide a valid 'hook_point_name'. Available: {[hp.name for hp in model.hook_points()]}"
                )
    else:
        hook_point = hook_point_name
        if hook_point not in [hp.name for hp in model.hook_points()]:
            raise ValueError(
                f"Invalid hook_point: {hook_point}. Valid hook points are: {[hp.name for hp in model.hook_points()]}"
            )

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]
        tokens = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=max_seq_len,
        )["input_ids"].to(device)

        if tokens.shape[0] == 0:
            continue

        _, cache = model.run_with_cache(tokens, names_filter=hook_point)
        model_acts_batch_3d = (
            cache[hook_point].detach().to(device="cpu", dtype=torch.float32)
        )

        if pooling_strategy == "last":
            model_acts_batch_2d = model_acts_batch_3d[:, -1, :]
        elif pooling_strategy == "mean":
            model_acts_batch_2d = model_acts_batch_3d.mean(dim=1)
        else:
            raise ValueError(f"Unknown pooling_strategy: {pooling_strategy}")

        all_model_acts_list.append(model_acts_batch_2d)

    if not all_model_acts_list:
        d_model = model.cfg.d_model
        return torch.empty((0, d_model), dtype=torch.float32)

    return torch.cat(all_model_acts_list, dim=0)


def get_model_activations_for_dataset(
    model: HookedTransformer,
    dataset_name: str,
    layer_idx: int,
    device: str | torch.device,
    setting_type: str = "normal",
    num_train_samples_target: int | None = None,
    corrupt_frac_val: float | None = None,
    class_imbalance_frac_target: float | None = None,
    max_seq_len: int = 1024,
    test_set_ratio: float = 0.2,
    cache_dir_base: str = "data/generated_model_activations",
    overwrite_cache: bool = False,
    seed: int = 42,
    pos_ratio_for_train_sampling: float = 0.5,
    expected_activation_dim: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    np.random.seed(seed)
    torch.manual_seed(seed)

    model_name_path = model.cfg.model_name.replace("/", "_")
    setting_params_str = f"setting_{setting_type}"
    if setting_type == "scarcity" and num_train_samples_target is not None:
        setting_params_str += f"_ntrain{num_train_samples_target}"
    if setting_type == "noise" and corrupt_frac_val is not None:
        setting_params_str += f"_noise{str(corrupt_frac_val).replace('.', 'p')}"
    if setting_type == "imbalance" and class_imbalance_frac_target is not None:
        setting_params_str += (
            f"_imbalance{str(class_imbalance_frac_target).replace('.', 'p')}"
        )

    cache_filename = f"{dataset_name}_model_{model_name_path}_layer{layer_idx}_{setting_params_str}.pkl"

    cache_path = os.path.join(cache_dir_base, model_name_path, f"layer_{layer_idx}")
    os.makedirs(cache_path, exist_ok=True)
    full_cache_file_path = os.path.join(cache_path, cache_filename)

    if not overwrite_cache and os.path.exists(full_cache_file_path):
        print(f"Loading cached processed data from: {full_cache_file_path}")
        try:
            with open(full_cache_file_path, "rb") as f:
                cached_data = pkl.load(f)
            if (
                isinstance(cached_data, tuple)
                and len(cached_data) == 4
                and all(isinstance(t, torch.Tensor) for t in cached_data)
            ):
                if (
                    expected_activation_dim is not None
                    and cached_data[0].shape[-1] != expected_activation_dim
                ):
                    print(
                        f"Cache dimension mismatch for {dataset_name}. Expected {expected_activation_dim}, got {cached_data[0].shape[-1]}. Regenerating."
                    )
                else:
                    return cached_data
            else:
                print(f"Invalid cache format in {full_cache_file_path}. Regenerating.")
        except Exception as e:
            print(
                f"Error loading from cache {full_cache_file_path}: {e}. Regenerating."
            )

    df = read_numbered_dataset_df(dataset_name)
    if "text" not in df.columns or "target" not in df.columns:
        raise ValueError(
            f"'text' or 'target' column not found in dataset {dataset_name}"
        )

    all_prompts: list[str] = df["text"].tolist()
    all_labels_raw: np.ndarray = (
        LabelEncoder().fit_transform(df["target"].values).astype(np.int64)
    )

    if len(all_prompts) == 0:
        raise ValueError(f"No prompts found in dataset {dataset_name}.")
    if len(np.unique(all_labels_raw)) < 2 and len(all_labels_raw) > 0:
        print(
            f"Warning: Dataset {dataset_name} contains only one class. Probing might not be meaningful."
        )

    num_total_samples = len(all_labels_raw)
    if num_total_samples == 0:
        d_model = model.cfg.d_model
        empty_X = torch.empty((0, d_model), dtype=torch.float32)
        empty_y = torch.empty((0,), dtype=torch.int64)
        return empty_X, empty_X, empty_y, empty_y

    if num_total_samples * test_set_ratio < 1 and num_total_samples > 1:
        num_test_samples = 1
    else:
        num_test_samples = int(num_total_samples * test_set_ratio)

    if num_test_samples == 0 and num_total_samples > 0:
        num_test_samples = 1 if num_total_samples > 1 else 0
        if num_total_samples == 1:
            num_test_samples = 0

    num_train_pool_samples = num_total_samples - num_test_samples

    if num_train_pool_samples <= 0 and num_total_samples > 0:
        print(
            f"Warning: No samples available for training pool after test set split for {dataset_name}. num_total={num_total_samples}, num_test={num_test_samples}"
        )
        train_indices_pool = np.array([], dtype=np.int64)
        test_indices = (
            np.arange(num_total_samples)
            if num_test_samples > 0
            else np.array([], dtype=np.int64)
        )
    elif num_test_samples == 0:
        train_indices_pool = np.arange(num_total_samples)
        test_indices = np.array([], dtype=np.int64)
    else:
        try:
            train_indices_pool, test_indices = train_test_split(
                np.arange(num_total_samples),
                test_size=num_test_samples,
                stratify=all_labels_raw if len(np.unique(all_labels_raw)) > 1 else None,
                random_state=seed,
            )
        except ValueError as e:
            print(
                f"Stratified split failed for {dataset_name} (possibly due to small class sizes): {e}. Using non-stratified split."
            )
            train_indices_pool, test_indices = train_test_split(
                np.arange(num_total_samples),
                test_size=num_test_samples,
                random_state=seed,
            )

    test_prompts = [all_prompts[i] for i in test_indices]
    y_test = torch.from_numpy(all_labels_raw[test_indices]).long()

    current_train_prompts_pool = [all_prompts[i] for i in train_indices_pool]
    current_train_labels_pool = all_labels_raw[train_indices_pool]

    final_train_prompts = current_train_prompts_pool
    y_train_intermediate = current_train_labels_pool.copy()

    if setting_type == "scarcity":
        if num_train_samples_target is not None and len(current_train_labels_pool) > 0:
            if num_train_samples_target > len(current_train_labels_pool):
                print(
                    f"Warning: num_train_samples_target ({num_train_samples_target}) > available pool ({len(current_train_labels_pool)}) for {dataset_name}. Using all available."
                )
                actual_num_train_target = len(current_train_labels_pool)
            else:
                actual_num_train_target = num_train_samples_target

            if actual_num_train_target > 0:
                temp_train_idx, _ = get_train_test_indices(
                    y=current_train_labels_pool,
                    num_train=actual_num_train_target,
                    num_test=0,
                    pos_ratio_train=pos_ratio_for_train_sampling,
                    seed=seed,
                )
                final_train_prompts = [
                    current_train_prompts_pool[i] for i in temp_train_idx
                ]
                y_train_intermediate = current_train_labels_pool[temp_train_idx]
            else:
                final_train_prompts = []
                y_train_intermediate = np.array([], dtype=np.int64)
        # If num_train_samples_target is None or pool is empty, y_train_intermediate remains as is (full pool or empty)

    elif setting_type == "imbalance":
        if (
            class_imbalance_frac_target is not None
            and len(current_train_labels_pool) > 0
        ):
            num_total_train = len(y_train_intermediate)
            num_pos_target = int(round(num_total_train * class_imbalance_frac_target))

            pos_indices_in_pool = np.where(y_train_intermediate == 1)[0]
            neg_indices_in_pool = np.where(y_train_intermediate == 0)[0]

            if len(pos_indices_in_pool) == 0 and num_pos_target > 0:
                print(
                    f"Warning: Cannot achieve target positive class {class_imbalance_frac_target} for {dataset_name} as no positive samples in training pool. Using original pool distribution."
                )
            elif len(neg_indices_in_pool) == 0 and num_pos_target < num_total_train:
                print(
                    f"Warning: Cannot achieve target positive class {class_imbalance_frac_target} for {dataset_name} as no negative samples in training pool. Using original pool distribution."
                )
            else:
                selected_pos_indices = np.random.choice(
                    pos_indices_in_pool,
                    size=num_pos_target,
                    replace=True
                    if num_pos_target > len(pos_indices_in_pool)
                    else False,
                )
                num_neg_target = num_total_train - num_pos_target
                selected_neg_indices = np.random.choice(
                    neg_indices_in_pool,
                    size=num_neg_target,
                    replace=True
                    if num_neg_target > len(neg_indices_in_pool)
                    else False,
                )

                imbalanced_indices = np.concatenate(
                    [selected_pos_indices, selected_neg_indices]
                )
                np.random.shuffle(imbalanced_indices)

                final_train_prompts = [
                    final_train_prompts[i] for i in imbalanced_indices
                ]
                y_train_intermediate = y_train_intermediate[imbalanced_indices]

    if setting_type == "noise" and corrupt_frac_val is not None:
        if len(y_train_intermediate) > 0:
            y_train_processed_np = corrupt_ytrain(
                y_train_intermediate, corrupt_frac_val, seed=seed
            )
        else:
            y_train_processed_np = y_train_intermediate
    else:
        y_train_processed_np = y_train_intermediate

    y_train = torch.from_numpy(y_train_processed_np).long()

    if not hasattr(model, "tokenizer") or model.tokenizer is None:
        raise ValueError("model.tokenizer is not available. Load model with tokenizer.")

    tokenizer = model.tokenizer

    if expected_activation_dim is not None and expected_activation_dim <= 0:
        print(
            f"Warning: Invalid expected_activation_dim ({expected_activation_dim}). Ignoring."
        )
        expected_activation_dim = None

    if final_train_prompts:
        X_train_model_acts = _generate_activations_from_prompts(
            model, final_train_prompts, tokenizer, layer_idx, device, max_seq_len
        )
        if (
            expected_activation_dim is not None
            and X_train_model_acts.shape[-1] != expected_activation_dim
        ):
            raise ValueError(
                f"Train activation dim mismatch for {dataset_name}. Model produced {X_train_model_acts.shape[-1]}, expected {expected_activation_dim} (likely SAE d_in)."
            )
    else:
        d_model_dim = (
            expected_activation_dim
            if expected_activation_dim is not None
            else model.cfg.d_model
        )
        X_train_model_acts = torch.empty((0, d_model_dim), dtype=torch.float32)

    if test_prompts:
        X_test_model_acts = _generate_activations_from_prompts(
            model, test_prompts, tokenizer, layer_idx, device, max_seq_len
        )
        if (
            expected_activation_dim is not None
            and X_test_model_acts.shape[-1] != expected_activation_dim
        ):
            raise ValueError(
                f"Test activation dim mismatch for {dataset_name}. Model produced {X_test_model_acts.shape[-1]}, expected {expected_activation_dim} (likely SAE d_in)."
            )

    else:
        d_model_dim = (
            expected_activation_dim
            if expected_activation_dim is not None
            else model.cfg.d_model
        )
        X_test_model_acts = torch.empty((0, d_model_dim), dtype=torch.float32)

    print(f"Saving generated data to cache: {full_cache_file_path}")
    with open(full_cache_file_path, "wb") as f:
        pkl.dump((X_train_model_acts, X_test_model_acts, y_train, y_test), f)

    return X_train_model_acts, X_test_model_acts, y_train, y_test
