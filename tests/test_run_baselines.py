from pathlib import Path

import pandas as pd
from transformer_lens import HookedTransformer

from sae_probes.run_baselines import (
    run_baseline_class_imbalance,
    run_baseline_dataset_layer,
    run_baseline_scarcity,
)
from tests.helpers import TEST_DATASET_NAME, generate_model_activations


def test_run_baseline_dataset_layer(gpt2_model: HookedTransformer, tmp_path: Path):
    model_cache_path = tmp_path / "model_cache"
    results_path = tmp_path / "results"
    generate_model_activations(gpt2_model, model_cache_path, layers=[4])
    run_baseline_dataset_layer(
        model_name="gpt2",
        layer=4,
        numbered_dataset=TEST_DATASET_NAME,
        method_name="logreg",
        results_path=results_path,
        model_cache_path=model_cache_path,
    )
    results_files = list(results_path.glob("**/*.csv"))
    assert len(results_files) == 1
    relative_path = results_files[0].relative_to(results_path)
    assert (
        str(relative_path)
        == "baseline_results_gpt2/normal/allruns/layer4_119_us_state_TX_logreg.csv"
    )
    df = pd.read_csv(results_files[0])
    assert set(df.columns.tolist()) == {
        "dataset",
        "method",
        "test_f1",
        "test_acc",
        "test_auc",
        "val_auc",
    }
    assert df.iloc[0]["dataset"] == "119_us_state_TX"
    assert df.iloc[0]["method"] == "logreg"
    assert df.iloc[0]["test_f1"] > 0.6
    assert df.iloc[0]["test_acc"] > 0.6
    assert df.iloc[0]["test_auc"] > 0.6
    assert df.iloc[0]["val_auc"] > 0.6


def test_run_baseline_scarcity(gpt2_model: HookedTransformer, tmp_path: Path):
    model_cache_path = tmp_path / "model_cache"
    results_path = tmp_path / "results"
    generate_model_activations(gpt2_model, model_cache_path, layers=[4])
    run_baseline_scarcity(
        model_name="gpt2",
        layer=4,
        numbered_dataset=TEST_DATASET_NAME,
        method_name="logreg",
        num_train=25,
        results_path=results_path,
        model_cache_path=model_cache_path,
    )
    results_files = list(results_path.glob("**/*.csv"))
    assert len(results_files) == 1
    relative_path = results_files[0].relative_to(results_path)
    assert (
        str(relative_path)
        == "baseline_results_gpt2/scarcity/allruns/layer4_119_us_state_TX_logreg_numtrain25.csv"
    )
    df = pd.read_csv(results_files[0])
    assert set(df.columns.tolist()) == {
        "dataset",
        "method",
        "num_train",
        "test_f1",
        "test_acc",
        "test_auc",
        "val_auc",
    }
    assert df.iloc[0]["dataset"] == "119_us_state_TX"
    assert df.iloc[0]["method"] == "logreg"
    assert df.iloc[0]["num_train"] == 25
    assert df.iloc[0]["test_f1"] > 0.5
    assert df.iloc[0]["test_acc"] > 0.5
    assert df.iloc[0]["test_auc"] > 0.5
    assert df.iloc[0]["val_auc"] > 0.5


def test_run_baseline_class_imbalance(gpt2_model: HookedTransformer, tmp_path: Path):
    model_cache_path = tmp_path / "model_cache"
    results_path = tmp_path / "results"
    generate_model_activations(gpt2_model, model_cache_path, layers=[4])
    run_baseline_class_imbalance(
        model_name="gpt2",
        layer=4,
        numbered_dataset=TEST_DATASET_NAME,
        method_name="logreg",
        dataset_frac=0.1,
        results_path=results_path,
        model_cache_path=model_cache_path,
    )
    results_files = list(results_path.glob("**/*.csv"))
    assert len(results_files) == 1
    relative_path = results_files[0].relative_to(results_path)
    assert (
        str(relative_path)
        == "baseline_results_gpt2/imbalance/allruns/layer4_119_us_state_TX_logreg_frac0.1.csv"
    )
    df = pd.read_csv(results_files[0])
    assert set(df.columns.tolist()) == {
        "dataset",
        "method",
        "num_train",
        "ratio",
        "test_f1",
        "test_acc",
        "test_auc",
        "val_auc",
    }

    assert df.iloc[0]["dataset"] == "119_us_state_TX"
    assert df.iloc[0]["method"] == "logreg"
    assert df.iloc[0]["num_train"] == 426
    assert df.iloc[0]["ratio"] == 0.1
    assert df.iloc[0]["test_f1"] > 0.5
    assert df.iloc[0]["test_acc"] > 0.5
    assert df.iloc[0]["test_auc"] > 0.5
    assert df.iloc[0]["val_auc"] > 0.5
