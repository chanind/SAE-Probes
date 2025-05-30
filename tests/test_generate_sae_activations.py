from pathlib import Path

from sae_probes.generate_sae_activations import (
    get_sae_paths_imbalance,
    get_sae_paths_normal,
    get_sae_paths_scarcity,
)


def test_get_sae_paths_normal():
    res = get_sae_paths_normal(
        model_name="gpt2",
        dataset="testdata",
        layer=1,
        reg_type="l1",
        sae_cache_path="test_cache",
    )
    assert res == {
        "save_path": Path(
            "test_cache/sae_probes_gpt2/normal_setting/testdata_1_l1.pkl"
        ),
        "test_path": Path(
            "test_cache/sae_activations_gpt2/normal_setting/testdata_1_X_test_sae.pt"
        ),
        "train_path": Path(
            "test_cache/sae_activations_gpt2/normal_setting/testdata_1_X_train_sae.pt"
        ),
        "y_test_path": Path(
            "test_cache/sae_activations_gpt2/normal_setting/testdata_1_y_test.pt"
        ),
        "y_train_path": Path(
            "test_cache/sae_activations_gpt2/normal_setting/testdata_1_y_train.pt"
        ),
    }


def test_get_sae_paths_imbalance():
    res = get_sae_paths_imbalance(
        dataset="testdata",
        layer=1,
        reg_type="l1",
        frac=0.5,
        model_name="gpt2",
        sae_cache_path="test_cache",
    )
    assert res == {
        "save_path": Path(
            "test_cache/sae_probes_gpt2/class_imbalance/testdata_1_l1_frac0.5.pkl"
        ),
        "test_path": Path(
            "test_cache/sae_activations_gpt2/class_imbalance/testdata_1_frac0.5_X_test_sae.pt"
        ),
        "train_path": Path(
            "test_cache/sae_activations_gpt2/class_imbalance/testdata_1_frac0.5_X_train_sae.pt"
        ),
        "y_test_path": Path(
            "test_cache/sae_activations_gpt2/class_imbalance/testdata_1_frac0.5_y_test.pt"
        ),
        "y_train_path": Path(
            "test_cache/sae_activations_gpt2/class_imbalance/testdata_1_frac0.5_y_train.pt"
        ),
    }


def test_get_sae_paths_scarcity():
    res = get_sae_paths_scarcity(
        dataset="testdata",
        layer=1,
        reg_type="l1",
        num_train=100,
        model_name="gpt2",
        sae_cache_path="test_cache",
    )
    assert res == {
        "save_path": Path(
            "test_cache/sae_probes_gpt2/scarcity_setting/testdata_1_l1_100.pkl"
        ),
        "test_path": Path(
            "test_cache/sae_activations_gpt2/scarcity_setting/testdata_1_100_X_test_sae.pt"
        ),
        "train_path": Path(
            "test_cache/sae_activations_gpt2/scarcity_setting/testdata_1_100_X_train_sae.pt"
        ),
        "y_test_path": Path(
            "test_cache/sae_activations_gpt2/scarcity_setting/testdata_1_100_y_test.pt"
        ),
        "y_train_path": Path(
            "test_cache/sae_activations_gpt2/scarcity_setting/testdata_1_100_y_train.pt"
        ),
    }
