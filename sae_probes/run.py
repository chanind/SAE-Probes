"""Main API for running SAE probing tasks."""

import json
import traceback
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from sae_lens import SAE
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from sae_probes.activations import (
    ActivationConfig,
    generate_model_activations,
    generate_sae_activations,
    load_model,
)
from sae_probes.datasets import DEFAULT_DATASET_PATH, get_binary_datasets, load_dataset
from sae_probes.evaluation import (
    EvaluationConfig,
    summarize_results,
)
from sae_probes.probing import (
    ProbeConfig,
    save_probe_results,
    train_baseline_probe,
    train_k_sparse_probe,
)
from sae_probes.utils import ensure_path, get_device, set_seed


@dataclass
class RunSaeProbeConfig:
    """Configuration for running SAE probing."""

    model_name: str
    hook_name: str
    k_values: list[int] | None = None
    reg_type: Literal["l1", "l2", "elasticnet"] = "l1"
    binarize: bool = False
    max_seq_len: int = 1024
    batch_size: int = 128
    device: str | None = None
    seed: int = 42
    sae_batch_size: int = 512
    llm_batch_size: int = 48
    autocast: bool = False

    def __post_init__(self):
        if self.k_values is None:
            self.k_values = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

    @property
    def torch_device(self) -> torch.device:
        return get_device(self.device)


@dataclass
class RunBaselineProbeConfig:
    """Configuration for running baseline probes."""

    model_name: str
    layer: int | str
    reg_type: Literal["l1", "l2"] = "l2"
    num_train: int = 1024
    max_seq_len: int = 1024
    batch_size: int = 48
    device: str | None = None
    seed: int = 42
    autocast: bool = False

    @property
    def hook_name(self) -> str:
        """Get the hook name based on the layer."""
        if isinstance(self.layer, int):
            return f"blocks.{self.layer}.hook_resid_post"
        elif self.layer == "embed":
            return "hook_embed"
        else:
            return self.layer

    @property
    def torch_device(self) -> torch.device:
        return get_device(self.device)


def run_baseline_probes(
    config: RunBaselineProbeConfig,
    results_dir: str | Path,
    cache_dir: str | Path,
    dataset_dir: str | Path = DEFAULT_DATASET_PATH,
    force_regenerate: bool = False,
) -> None:
    """
    Run baseline probes (standard neural probes) directly on model activations.

    Args:
        config: Configuration for baseline probes
        dataset_dir: Path to directory containing datasets
        results_dir: Path for storing results
        cache_dir: Path for storing/reusing model activations
        force_regenerate: Whether to force regeneration of activations
    """
    # Set random seed
    set_seed(config.seed)

    # Create activation config
    activation_config = ActivationConfig(
        model_name=config.model_name,
        hook_name=config.hook_name,
        max_seq_len=config.max_seq_len,
        device=str(config.torch_device),
    )

    # Create results directory
    results_dir = (
        Path(results_dir) / f"baseline_probes_{config.model_name}/normal_settings"
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    # Get list of datasets
    datasets = get_binary_datasets(Path(dataset_dir))
    print(f"Found {len(datasets)} binary classification datasets")

    # Load model
    model = load_model(config.model_name, device=str(config.torch_device))

    # Process each dataset
    for dataset_info in tqdm(datasets, desc="Processing datasets"):
        dataset_tag = dataset_info.tag
        try:
            # Load dataset
            df, train_indices, test_indices = load_dataset(
                dataset_tag=dataset_tag,
                dataset_path=Path(dataset_dir),
                num_train=config.num_train,
                test_size=None,
                pos_ratio=0.5,
                seed=config.seed,
            )
        except Exception as e:
            print(f"Error loading dataset {dataset_tag}: {e}")
            print(f"Stack trace:\n{traceback.format_exc()}")
            continue

        # Generate model activations
        model_activations = generate_model_activations(
            model=model,
            dataset_tag=dataset_tag,
            dataset_df=df,
            config=activation_config,
            cache_path=Path(cache_dir),
            force_regenerate=force_regenerate,
            autocast=config.autocast,
            batch_size=config.batch_size,
        )

        # Prepare data for training
        X_train = model_activations[train_indices]
        X_test = model_activations[test_indices]
        y_train = df["target"].values[train_indices]
        y_test = df["target"].values[test_indices]

        # Encode labels if needed
        if not set(y_train).issubset({0, 1}):
            le = LabelEncoder()
            y_train = le.fit_transform(y_train)
            y_test = le.transform(y_test)

        # Train baseline probe
        results = train_baseline_probe(
            X_train=X_train,
            y_train=y_train,  # type: ignore
            X_test=X_test,
            y_test=y_test,  # type: ignore
            reg_type=config.reg_type,
            seed=config.seed,
        )

        # Save results
        results_dict = results.to_dict()
        results_dict["dataset"] = dataset_tag
        results_dict["hook_name"] = config.hook_name
        results_dict["model"] = config.model_name
        results_dict["num_train"] = config.num_train
        results_dict["reg_type"] = config.reg_type

        # Extract layer identifier from hook name for file naming
        hook_parts = config.hook_name.split(".")
        hook_identifier = hook_parts[1] if len(hook_parts) > 1 else "custom"
        output_file = (
            Path(results_dir) / f"{hook_identifier}_{dataset_tag}_baseline_results.json"
        )
        with open(output_file, "w") as f:
            json.dump(results_dict, f, indent=4)

        print(f"Saved baseline probe results for {dataset_tag} to {output_file}")


def run_sae_probe(
    sae: SAE,
    config: RunSaeProbeConfig,
    results_dir: str | Path,
    cache_dir: str | Path,
    dataset_dir: str | Path = DEFAULT_DATASET_PATH,
    force_regenerate: bool = False,
) -> dict[str, dict]:
    """
    Run SAE probing evaluation on a given SAE.

    Args:
        sae: SAELens SAE object
        model_name: Name of model architecture
        layer: Layer number in model to probe
        dataset_dir: Path to directory containing datasets
        cache_dir: Path for storing/reusing model activations
        results_dir: Path for storing results
        k_values: List of feature selection values to evaluate
        reg_type: Regularization type (l1 or l2)
        binarize: Whether to binarize SAE features
        device: Device to use for computation
        force_regenerate: Whether to force regeneration of activations

    Returns:
        Dictionary containing probing results for each dataset and setting
    """
    # Set random seed
    set_seed(config.seed)

    # Create activation config
    activation_config = ActivationConfig(
        model_name=config.model_name,
        hook_name=config.hook_name,
        max_seq_len=config.max_seq_len,
        device=str(config.torch_device),
    )

    # Create probe config
    probe_config = ProbeConfig(
        reg_type=config.reg_type,
        k_values=config.k_values,
        binarize=config.binarize,
        seed=config.seed,
    )

    # Create save paths
    ensure_path(Path(results_dir))

    # Get SAE ID or use custom identifier
    sae_id = getattr(sae, "sae_id", "custom_sae")

    # Get list of datasets
    datasets = get_binary_datasets(Path(dataset_dir))
    print(f"Found {len(datasets)} binary classification datasets")

    # Move SAE to device
    sae = sae.to(config.torch_device)
    model = load_model(config.model_name, device=str(config.torch_device))

    # Process each dataset
    all_results = defaultdict(list)

    for dataset_info in tqdm(datasets, desc="Processing datasets"):
        dataset_tag = dataset_info.tag
        try:
            # Load dataset
            df, train_indices, test_indices = load_dataset(
                dataset_tag=dataset_tag,
                dataset_path=Path(dataset_dir),
                num_train=1024,  # Default for normal setting
                test_size=None,
                pos_ratio=0.5,
                seed=config.seed,
            )
        except Exception as e:
            print(f"Error loading dataset {dataset_tag}: {e}")
            print(f"Stack trace:\n{traceback.format_exc()}")
            continue

        # Generate model activations
        model_activations = generate_model_activations(
            model=model,
            dataset_tag=dataset_tag,
            dataset_df=df,
            config=activation_config,
            cache_path=Path(cache_dir),
            force_regenerate=force_regenerate,
            autocast=config.autocast,
            batch_size=config.llm_batch_size,
        )

        # Generate SAE activations
        sae_activations = generate_sae_activations(
            dataset_tag=dataset_tag,
            sae=sae,
            model_activations=model_activations,
            config=activation_config,
            autocast=config.autocast,
            batch_size=config.sae_batch_size,
        )

        # Prepare data for normal setting
        X_train = sae_activations[train_indices]
        X_test = sae_activations[test_indices]
        y_train = df["target"].values[train_indices]
        y_test = df["target"].values[test_indices]

        # Encode labels if needed
        if not set(y_train).issubset({0, 1}):
            le = LabelEncoder()
            y_train = le.fit_transform(y_train)
            y_test = le.transform(y_test)

        # Train and evaluate probes
        results = train_k_sparse_probe(
            X_train=X_train,
            y_train=y_train,  # type: ignore
            X_test=X_test,
            y_test=y_test,  # type: ignore
            config=probe_config,
        )

        save_probe_results(
            results=results,
            dataset=dataset_tag,
            config=probe_config,
            sae_id=sae_id,
            save_path=Path(results_dir)
            / f"{dataset_tag}_{sae_id}_{probe_config.reg_type}.json",
        )

        # Add to all results
        all_results["normal"].extend([r.to_dict() for r in results])

    # Create evaluation config
    eval_config = EvaluationConfig(model_name=config.model_name)

    with open(Path(results_dir) / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)

    # Summarize results
    summaries = summarize_results(
        results=all_results,
        config=eval_config,
    )

    return {
        "results": all_results,
        "summaries": summaries,
    }
