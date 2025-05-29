# from sae_probes import utils_data as new_utils_data
# from tests._comparison import utils_data as old_utils_data

# We'll need to inspect both files to find comparable functions.

from pathlib import Path  # Keep Path for tmp_path in model activation tests

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.preprocessing import LabelEncoder
from transformer_lens import HookedTransformer

from sae_probes import utils_data as new_utils_data
from tests._comparison import utils_data as old_utils_data

# Store original pd.read_csv for use in mocks to avoid recursion - REMOVED
# original_pd_read_csv = pd.read_csv

# Helper to create dummy master CSV for tests - REMOVED
# @pytest.fixture
# def dummy_master_csv(tmp_path: Path) -> Path:
# ...

# Helper to create dummy dataset files - REMOVED
# @pytest.fixture
# def dummy_dataset_files(dummy_master_csv: Path):
# ...

# @pytest.fixture
# def dummy_ood_files(dummy_master_csv: Path) -> dict[str, Path]: # REMOVED
# ...

# @pytest.fixture
# def dummy_glue_exploration_file(dummy_master_csv: Path) -> Path: # REMOVED
# ...

# Test Explanation:
# The following tests for get_binary_df, get_numbered_binary_tags, read_dataset_df, get_yvals,
# get_OOD_datasets, get_glue_investigation_df, get_glue_labels, get_disagree_glue_indices,
# and get_model_activations_for_dataset_normal_setting have been refactored.
# They no longer use dummy CSV files or extensive mocking of file system operations (os.path, pd.read_csv).
# Instead, they rely on the actual data files expected to be present in the
# `sae_probes/data/` and `sae_probes/data/OOD data/` directories, and for GLUE data,
# in the `results/investigate/` directory relative to the project root.
#
# Tests that primarily verify algorithmic logic (e.g., get_train_test_indices, corrupt_ytrain)
# remain largely unchanged as they were not dependent on these file I/O mocks.
#
# The path logic within the main code (sae_probes.utils_data) has been updated to robustly find these
# files relative to its own location (`__file__`) or with fallbacks for common execution contexts.
#
# If specific datasets (e.g., "ai_news", "87_glue_cola") are not present,
# tests that depend on them will be skipped with a message.


def test_get_binary_df():
    # Relies on the actual probing_datasets_MASTER.csv
    try:
        actual_master_df_path = (
            Path(new_utils_data.__file__).resolve().parent
            / "data"
            / "probing_datasets_MASTER.csv"
        )
        if (
            not actual_master_df_path.exists()
        ):  # Fallback for different CWD scenarios during test
            actual_master_df_path = (
                Path("sae_probes") / "data" / "probing_datasets_MASTER.csv"
            )
        actual_master_df = pd.read_csv(actual_master_df_path)
    except FileNotFoundError:
        pytest.skip(
            "probing_datasets_MASTER.csv not found. Skipping test_get_binary_df."
        )

    expected_df_filtered = actual_master_df[
        actual_master_df["Data type"] == "Binary Classification"
    ].reset_index(drop=True)
    new_df = new_utils_data.get_binary_df().reset_index(drop=True)
    pd.testing.assert_frame_equal(new_df, expected_df_filtered)


def test_get_numbered_binary_tags():
    # Relies on the actual master CSV.
    try:
        actual_master_df_path = (
            Path(new_utils_data.__file__).resolve().parent
            / "data"
            / "probing_datasets_MASTER.csv"
        )
        if not actual_master_df_path.exists():
            actual_master_df_path = (
                Path("sae_probes") / "data" / "probing_datasets_MASTER.csv"
            )
        actual_master_df = pd.read_csv(actual_master_df_path)
    except FileNotFoundError:
        pytest.skip(
            "probing_datasets_MASTER.csv not found. Skipping test_get_numbered_binary_tags."
        )

    binary_df = actual_master_df[
        actual_master_df["Data type"] == "Binary Classification"
    ]
    # Handle cases where 'Dataset save name' might be missing or have NaN values before processing
    binary_df = binary_df.dropna(subset=["Dataset save name"])
    expected_tags = [
        str(name).split("/")[-1].split(".")[0]
        for name in binary_df["Dataset save name"]
    ]

    new_tags = new_utils_data.get_numbered_binary_tags()
    assert sorted(new_tags) == sorted(expected_tags)


def test_read_dataset_df():
    dataset_tag_to_test = "ai_news"  # Example, ensure this tag exists

    master_df_for_path_info = new_utils_data.get_binary_df()
    dataset_info = master_df_for_path_info[
        master_df_for_path_info["Dataset Tag"] == dataset_tag_to_test
    ]

    if dataset_info.empty:
        pytest.skip(
            f"Dataset tag '{dataset_tag_to_test}' not found in master CSV for binary datasets. Skipping test_read_dataset_df."
        )

    dataset_save_name = dataset_info["Dataset save name"].iloc[0]

    try:
        base_data_dir = Path(new_utils_data.__file__).resolve().parent / "data"
        if not base_data_dir.exists():  # Fallback
            base_data_dir = Path("sae_probes") / "data"
    except NameError:  # __file__ not defined (e.g. interactive)
        base_data_dir = Path("sae_probes") / "data"

    expected_file_path = base_data_dir / dataset_save_name
    if not expected_file_path.exists():
        pytest.skip(
            f"Expected dataset file {expected_file_path} does not exist. Skipping test_read_dataset_df."
        )

    expected_content_df = pd.read_csv(expected_file_path)
    new_df = new_utils_data.read_dataset_df(dataset_tag_to_test)
    pd.testing.assert_frame_equal(new_df, expected_content_df)


def test_get_yvals():
    numbered_dataset_tag_to_test = "1_ai_news"  # Example

    try:
        df_for_yvals = new_utils_data.read_numbered_dataset_df(
            numbered_dataset_tag_to_test
        )
    except (FileNotFoundError, ValueError) as e:
        pytest.skip(
            f"Cannot read dataset for yvals test ('{numbered_dataset_tag_to_test}'): {e}. Skipping."
        )

    if "target" not in df_for_yvals.columns:
        pytest.skip(
            f"'target' column not found in {numbered_dataset_tag_to_test}. Skipping yvals test."
        )

    le = LabelEncoder()
    expected_yvals_arr = le.fit_transform(df_for_yvals["target"].values).astype(
        np.int64
    )

    new_yvals = new_utils_data.get_yvals(numbered_dataset_tag_to_test)
    np.testing.assert_array_equal(new_yvals, expected_yvals_arr)


@pytest.mark.parametrize(
    "y_array, num_train, num_test, pos_ratio, seed, expected_train_len, expected_test_len, desc",
    [
        (np.array([0, 0, 0, 0, 1, 1, 1, 1]), 4, 4, 0.5, 42, 4, 4, "Balanced simple"),
        (
            np.array([0, 0, 1, 1, 1, 1, 1, 1, 1, 1]),
            6,
            2,
            0.5,
            42,
            6,
            2,
            "Unbalanced, train takes more of majority",
        ),
        (
            np.array([0, 0, 0, 0, 0, 0, 1, 1]),
            4,
            2,
            0.5,
            42,
            4,
            2,
            "Train takes most of minority, test gets rest",
        ),
        (np.array([0, 0, 0, 0, 1, 1]), 4, 0, 0.5, 42, 4, 0, "No test samples"),
        (np.array([0, 0, 1, 1]), 2, 2, 0.25, 42, 2, 2, "Low pos_ratio_train"),
        (np.array([0, 0, 1, 1]), 2, 2, 0.75, 42, 2, 2, "High pos_ratio_train"),
        (
            np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
            6,
            6,
            0.9,
            42,
            6,
            6,
            "High pos ratio, limited negatives",
        ),
        (
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]),
            6,
            6,
            0.1,
            42,
            6,
            6,
            "Low pos ratio, limited positives",
        ),
        (
            np.array([0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
            10,
            2,
            0.2,
            42,
            10,
            2,
            "Train nearly exhausts minority for test",
        ),
    ],
)
def test_get_train_test_indices_new_logic(
    y_array: np.ndarray,
    num_train: int,
    num_test: int,
    pos_ratio: float,
    seed: int,
    expected_train_len: int,
    expected_test_len: int,
    desc: str,
):
    new_train_idx, new_test_idx = new_utils_data.get_train_test_indices(
        y_array, num_train, num_test, pos_ratio_train=pos_ratio, seed=seed
    )
    assert len(new_train_idx) <= num_train
    assert len(new_test_idx) <= num_test
    if len(new_train_idx) > 0:
        assert len(np.unique(new_train_idx)) == len(new_train_idx)
    if len(new_test_idx) > 0:
        assert len(np.unique(new_test_idx)) == len(new_test_idx)
    if len(new_train_idx) > 0 and len(new_test_idx) > 0:
        assert np.intersect1d(new_train_idx, new_test_idx).size == 0

    if (
        desc == "Balanced simple"
    ):  # Only compare to old logic where behavior was expected to be similar
        old_train_idx, old_test_idx = old_utils_data.get_train_test_indices(
            y_array, num_train, num_test, pos_ratio=pos_ratio, seed=seed
        )
        np.testing.assert_array_equal(np.sort(new_train_idx), np.sort(old_train_idx))
        assert len(old_train_idx) == expected_train_len
        assert len(old_test_idx) == expected_test_len


def test_get_dataset_sizes():
    new_sizes = new_utils_data.get_dataset_sizes()

    expected_s: dict[str, int] = {}
    dataset_tags = (
        new_utils_data.get_numbered_binary_tags()
    )  # these are already filtered for binary
    for tag in dataset_tags:
        try:
            df = new_utils_data.read_numbered_dataset_df(tag)
            expected_s[tag] = len(df)
        except Exception:
            # If read_numbered_dataset_df fails, get_dataset_sizes also skips it.
            pass

    assert new_sizes == expected_s
    if new_sizes:  # only check if not empty
        assert all(isinstance(v, int) for v in new_sizes.values())


def test_get_training_sizes():
    new_sizes = new_utils_data.get_training_sizes()
    old_sizes = old_utils_data.get_training_sizes()
    np.testing.assert_array_equal(new_sizes, old_sizes)
    assert isinstance(new_sizes, list)
    assert all(isinstance(x, int) for x in new_sizes)
    assert len(new_sizes) <= 20
    if len(new_sizes) > 0:
        assert new_sizes[0] >= 2**1 and new_sizes[-1] <= 2**10


def test_get_class_imbalance():
    new_fracs = new_utils_data.get_class_imbalance()
    old_fracs = old_utils_data.get_class_imbalance()
    np.testing.assert_allclose(new_fracs, old_fracs, rtol=1e-7)
    assert isinstance(new_fracs, list)
    assert len(new_fracs) == 19
    if len(new_fracs) > 0:
        assert np.isclose(new_fracs[0], 0.05) and np.isclose(new_fracs[-1], 0.95)


@pytest.mark.parametrize(
    "ytrain_array, frac_to_corrupt, seed, expected_corruption_count_min_max",
    [
        (np.array([0, 0, 0, 0, 1, 1, 1, 1]), 0.25, 42, (2, 2)),
        (np.array([0] * 10 + [1] * 10), 0.1, 42, (2, 2)),
        (np.array([0] * 10), 0.2, 42, (2, 2)),
        (np.array([1] * 7), 0.0, 42, (0, 0)),
        (np.array([0, 1, 0, 1, 0, 1]), 0.5, 42, (3, 3)),
        (np.array([0, 1]), 0.49, 42, (0, 1)),
    ],
)
def test_corrupt_ytrain(
    ytrain_array: np.ndarray,
    frac_to_corrupt: float,
    seed: int,
    expected_corruption_count_min_max: tuple[int, int],
):
    original_ytrain = ytrain_array.copy()
    new_corrupted = new_utils_data.corrupt_ytrain(
        ytrain_array.copy(), frac_to_corrupt, seed=seed
    )
    old_corrupted = old_utils_data.corrupt_ytrain(
        ytrain_array.copy(),
        frac_to_corrupt,  # Old version hardcodes seed
    )

    assert new_corrupted.shape == original_ytrain.shape
    new_diff_count = np.sum(new_corrupted != original_ytrain)

    min_expected, max_expected = expected_corruption_count_min_max
    if (
        frac_to_corrupt > 0
        and len(ytrain_array) > 0
        and int(len(ytrain_array) * frac_to_corrupt) == 0
    ):  # Special handling in new code for frac > 0 but int(len*frac) == 0
        assert new_diff_count >= 0 and new_diff_count <= 1
        max_expected = 1  # Adjust for this edge case in new logic
    else:
        assert new_diff_count >= min_expected and new_diff_count <= max_expected

    if (
        not (
            frac_to_corrupt > 0
            and int(len(ytrain_array) * frac_to_corrupt) == 0
            and len(ytrain_array) > 0
        )
        and seed == 42
    ):
        np.testing.assert_array_equal(new_corrupted, old_corrupted)


def test_get_corrupt_frac():
    new_fracs = new_utils_data.get_corrupt_frac()
    old_fracs = old_utils_data.get_corrupt_frac()
    np.testing.assert_allclose(new_fracs, old_fracs, rtol=1e-7)
    assert isinstance(new_fracs, list)
    assert len(new_fracs) == 11
    if len(new_fracs) > 0:
        assert np.isclose(new_fracs[0], 0.0) and np.isclose(new_fracs[-1], 0.5)


def test_get_OOD_datasets():
    try:
        current_file_dir = Path(new_utils_data.__file__).resolve().parent
        ood_data_path = current_file_dir / "data" / "OOD data"
        if not ood_data_path.is_dir():  # Fallback
            ood_data_path = Path("sae_probes") / "data" / "OOD data"
    except NameError:  # __file__ not defined
        ood_data_path = Path("sae_probes") / "data" / "OOD data"

    if not ood_data_path.is_dir():
        pytest.skip(
            f"OOD data directory not found at {ood_data_path}. Skipping test_get_OOD_datasets."
        )

    actual_files = list(ood_data_path.glob("*.csv"))

    expected_trans_list = []
    expected_no_trans_list = []
    for f_path in actual_files:
        name = f_path.name.replace("_OOD.csv", "")
        expected_trans_list.append(name)
        if "translation" not in name:
            expected_no_trans_list.append(name)

    new_ood_datasets_trans = new_utils_data.get_OOD_datasets(translation=True)
    assert sorted(new_ood_datasets_trans) == sorted(expected_trans_list)

    new_ood_datasets_no_trans = new_utils_data.get_OOD_datasets(translation=False)
    assert sorted(new_ood_datasets_no_trans) == sorted(expected_no_trans_list)


def test_get_glue_investigation_df():
    dataset_name_to_test = (
        "87_glue_cola"  # Default, or adapt if another GLUE file is primary
    )

    try:
        df_new = new_utils_data.get_glue_investigation_df(
            dataset_name=dataset_name_to_test
        )
        assert not df_new.empty
        # Basic check for expected columns, assuming a standard GLUE investigation format
        assert "prompt" in df_new.columns
        assert "original_target" in df_new.columns
    except FileNotFoundError:
        pytest.skip(
            f"GLUE investigation file for {dataset_name_to_test} not found by the function. Skipping test."
        )
    except Exception as e:
        pytest.fail(f"get_glue_investigation_df failed for {dataset_name_to_test}: {e}")


def test_get_glue_labels():
    dataset_name_to_test = "87_glue_cola"

    try:
        df_investigate = new_utils_data.get_glue_investigation_df(
            dataset_name=dataset_name_to_test
        )
    except FileNotFoundError:
        pytest.skip(
            f"GLUE investigation file for {dataset_name_to_test} not found. Skipping test_get_glue_labels."
        )
    except Exception as e:
        pytest.fail(
            f"Failed to load GLUE investigation df for {dataset_name_to_test} in labels test: {e}"
        )

    if df_investigate.empty:
        pytest.skip(
            f"GLUE investigation df for {dataset_name_to_test} is empty. Skipping test_get_glue_labels."
        )

    if "original_target" in df_investigate.columns:
        labels_orig_new = new_utils_data.get_glue_labels(
            dataset_name=dataset_name_to_test, label_type="original_target"
        )
        expected_orig = df_investigate["original_target"].values.astype(np.int64)
        np.testing.assert_array_equal(labels_orig_new, expected_orig)
    else:
        print(
            f"Warning: 'original_target' column not in {dataset_name_to_test}_investigate.csv for test_get_glue_labels."
        )

    if (
        "ensemble" in df_investigate.columns
    ):  # Assuming 'ensemble' is a standard column for this
        labels_ens_new = new_utils_data.get_glue_labels(
            dataset_name=dataset_name_to_test, label_type="ensemble"
        )
        expected_ens = df_investigate["ensemble"].values.astype(np.int64)
        np.testing.assert_array_equal(labels_ens_new, expected_ens)
    else:
        print(
            f"Warning: 'ensemble' column not in {dataset_name_to_test}_investigate.csv for test_get_glue_labels."
        )


def test_get_disagree_glue_indices():
    dataset_name_to_test = "87_glue_cola"

    try:
        df_investigate = new_utils_data.get_glue_investigation_df(
            dataset_name=dataset_name_to_test
        )
    except FileNotFoundError:
        pytest.skip(
            f"GLUE investigation file for {dataset_name_to_test} not found. Skipping test_get_disagree_glue_indices."
        )
    except Exception as e:
        pytest.fail(
            f"Failed to load GLUE df for {dataset_name_to_test} in disagree_indices test: {e}"
        )

    if df_investigate.empty:
        pytest.skip(
            f"GLUE investigation df for {dataset_name_to_test} is empty. Skipping disagree_indices test."
        )

    if (
        "original_target" not in df_investigate.columns
        or "ensemble" not in df_investigate.columns
    ):
        pytest.skip(
            f"Required columns for disagreement not in {dataset_name_to_test}_investigate.csv. Skipping."
        )

    disagree_indices_new = new_utils_data.get_disagree_glue_indices(
        dataset_name=dataset_name_to_test
    )
    expected_indices = np.where(
        df_investigate["original_target"] != df_investigate["ensemble"]
    )[0].astype(np.int64)
    np.testing.assert_array_equal(disagree_indices_new, expected_indices)


def test_get_model_activations_for_dataset_normal_setting(
    gpt2_model: HookedTransformer,
    tmp_path: Path,
):
    dataset_name_to_test = "1_ai_news"

    try:
        df_check = new_utils_data.read_numbered_dataset_df(dataset_name_to_test)
        if df_check.empty:
            pytest.skip(
                f"Dataset {dataset_name_to_test} is empty. Skipping model activations test."
            )
        if len(df_check) < 2:
            pytest.skip(
                f"Dataset {dataset_name_to_test} has < 2 samples. Skipping test with 0.5 split."
            )
    except (FileNotFoundError, ValueError) as e:
        pytest.skip(
            f"Cannot read '{dataset_name_to_test}' for model activations test: {e}. Skipping."
        )

    layer_idx = 2
    device = "cpu"
    cache_dir = tmp_path / "model_acts_cache"

    num_total_samples = len(df_check)
    # Simplified expected sample calculation for test_set_ratio=0.5
    # The function has more detailed logic for edge cases (e.g. very small total_samples)
    if num_total_samples == 1:
        expected_test_samples = 0
    elif num_total_samples * 0.5 < 1:  # test_set_ratio * num_total_samples
        expected_test_samples = 1  # if total > 1
    else:
        expected_test_samples = int(num_total_samples * 0.5)

    expected_train_samples = num_total_samples - expected_test_samples

    if (
        expected_train_samples == 0
        and expected_test_samples == 0
        and num_total_samples > 0
    ):
        pytest.skip(
            f"Dataset {dataset_name_to_test} ({num_total_samples} samples) leads to 0 train/test. Skipping."
        )

    X_train, X_test, y_train, y_test = new_utils_data.get_model_activations_for_dataset(
        model=gpt2_model,
        dataset_name=dataset_name_to_test,
        layer_idx=layer_idx,
        device=device,
        setting_type="normal",
        max_seq_len=gpt2_model.cfg.n_ctx,
        test_set_ratio=0.5,
        cache_dir_base=str(cache_dir),
        overwrite_cache=True,
        expected_activation_dim=gpt2_model.cfg.d_model,
    )

    assert isinstance(X_train, torch.Tensor)
    assert isinstance(y_train, torch.Tensor)
    assert isinstance(X_test, torch.Tensor)
    assert isinstance(y_test, torch.Tensor)

    assert X_train.shape[0] == expected_train_samples
    assert y_train.shape[0] == expected_train_samples
    assert X_test.shape[0] == expected_test_samples
    assert y_test.shape[0] == expected_test_samples

    if expected_train_samples > 0:
        assert X_train.shape[1] == gpt2_model.cfg.d_model
    if expected_test_samples > 0:
        assert X_test.shape[1] == gpt2_model.cfg.d_model

    model_name_sanitized = gpt2_model.cfg.model_name.replace("/", "_")
    setting_params_str = "setting_normal"
    cache_filename = f"{dataset_name_to_test}_model_{model_name_sanitized}_layer{layer_idx}_{setting_params_str}.pkl"
    expected_cache_file_dir = cache_dir / model_name_sanitized / f"layer_{layer_idx}"
    expected_cache_file = expected_cache_file_dir / cache_filename
    assert expected_cache_file.exists()

    X_train_cached, X_test_cached, y_train_cached, y_test_cached = (
        new_utils_data.get_model_activations_for_dataset(
            model=gpt2_model,
            dataset_name=dataset_name_to_test,
            layer_idx=layer_idx,
            device=device,
            setting_type="normal",
            max_seq_len=gpt2_model.cfg.n_ctx,
            test_set_ratio=0.5,
            cache_dir_base=str(cache_dir),
            overwrite_cache=False,
            expected_activation_dim=gpt2_model.cfg.d_model,
        )
    )
    torch.testing.assert_close(X_train, X_train_cached)
    torch.testing.assert_close(X_test, X_test_cached)
    torch.testing.assert_close(y_train, y_train_cached)
    torch.testing.assert_close(y_test, y_test_cached)

    new_utils_data.get_model_activations_for_dataset(
        model=gpt2_model,
        dataset_name=dataset_name_to_test,
        layer_idx=layer_idx,
        device=device,
        setting_type="normal",
        max_seq_len=gpt2_model.cfg.n_ctx,
        test_set_ratio=0.5,
        cache_dir_base=str(cache_dir),
        overwrite_cache=True,
        expected_activation_dim=gpt2_model.cfg.d_model,
    )
