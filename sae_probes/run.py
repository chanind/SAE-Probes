"""Main API for running SAE probing tasks."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from sae_lens import SAE
from tqdm import tqdm

from sae_probes.activations import (
    ActivationConfig,
    generate_model_activations,
    generate_sae_activations,
)
from sae_probes.datasets import get_binary_datasets, load_dataset
from sae_probes.evaluation import (
    EvaluationConfig,
    summarize_results,
)
from sae_probes.probing import (
    ProbeConfig,
    save_probe_results,
    train_probe,
)
from sae_probes.utils import ensure_path, get_device, set_seed


@dataclass
class RunConfig:
    """Configuration for running SAE probing."""

    model_name: str
    layer: int
    settings: list[str] | None = None
    k_values: list[int] | None = None
    reg_type: Literal["l1", "l2", "elasticnet"] = "l1"
    binarize: bool = False
    max_seq_len: int = 1024
    batch_size: int = 128
    device: str | None = None
    seed: int = 42

    def __post_init__(self):
        if self.settings is None:
            self.settings = ["normal"]

        if self.k_values is None:
            self.k_values = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

    @property
    def torch_device(self) -> torch.device:
        return get_device(self.device)


def run_sae_probe(
    sae: SAE,
    model_name: str,
    layer: int,
    dataset_path: Path,
    cache_path: Path,
    settings: list[str] | None = None,
    k_values: list[int] | None = None,
    reg_type: Literal["l1", "l2", "elasticnet"] = "l1",
    binarize: bool = False,
    device: str | None = None,
    force_regenerate: bool = False,
) -> dict[str, dict]:
    """
    Run SAE probing evaluation on a given SAE.

    Args:
        sae: SAELens SAE object
        model_name: Name of model architecture
        layer: Layer number in model to probe
        dataset_path: Path to directory containing datasets
        cache_path: Path for storing/reusing model activations
        settings: List of evaluation settings (normal, scarcity, imbalance, noise)
        k_values: List of feature selection values to evaluate
        reg_type: Regularization type (l1 or l2)
        binarize: Whether to binarize SAE features
        device: Device to use for computation
        force_regenerate: Whether to force regeneration of activations

    Returns:
        Dictionary containing probing results for each dataset and setting
    """
    # Create configuration
    config = RunConfig(
        model_name=model_name,
        layer=layer,
        settings=settings,
        k_values=k_values,
        reg_type=reg_type,
        binarize=binarize,
        device=device,
        seed=42,
    )

    # Set random seed
    set_seed(config.seed)

    # Create activation config
    activation_config = ActivationConfig(
        model_name=config.model_name,
        layer=config.layer,
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
    results_path = cache_path / "results"
    ensure_path(results_path)

    # Get SAE ID or use custom identifier
    sae_id = getattr(sae, "sae_id", "custom_sae")

    # Get list of datasets
    datasets = get_binary_datasets(dataset_path)
    print(f"Found {len(datasets)} binary classification datasets")

    # Move SAE to device
    sae = sae.to(device)

    # Process each dataset
    all_results = {setting: [] for setting in config.settings or []}

    for dataset_info in tqdm(datasets, desc="Processing datasets"):
        dataset_tag = dataset_info.tag

        # Generate model activations
        model_activations = generate_model_activations(
            dataset_tag=dataset_tag,
            config=activation_config,
            dataset_path=dataset_path,
            cache_path=cache_path,
            force_regenerate=force_regenerate,
        )

        # Generate SAE activations
        sae_activations = generate_sae_activations(
            dataset_tag=dataset_tag,
            sae=sae,
            model_activations=model_activations,
            config=activation_config,
            cache_path=cache_path,
            force_regenerate=force_regenerate,
        )

        # Load dataset
        df, train_indices, test_indices = load_dataset(
            dataset_tag=dataset_tag,
            dataset_path=dataset_path,
            num_train=1024,  # Default for normal setting
            test_size=None,
            pos_ratio=0.5,
            seed=config.seed,
        )

        # Prepare data for normal setting
        X_train = sae_activations[train_indices]
        X_test = sae_activations[test_indices]
        y_train = df["target"].values[train_indices]
        y_test = df["target"].values[test_indices]

        # Encode labels if needed
        if not set(y_train).issubset({0, 1}):
            from sklearn.preprocessing import LabelEncoder

            le = LabelEncoder()
            y_train = le.fit_transform(y_train)
            y_test = le.transform(y_test)

        # Train and evaluate probes
        results = train_probe(
            X_train=X_train,
            y_train=y_train,  # type: ignore
            X_test=X_test,
            y_test=y_test,  # type: ignore
            config=probe_config,
        )

        # Save results
        save_path = (
            results_path
            / f"sae_probes_{config.model_name}"
            / "normal_setting"
            / f"{dataset_tag}_{config.layer}_{sae_id}_{config.reg_type}.pkl"
        )
        save_probe_results(
            results=results,
            dataset=dataset_tag,
            config=probe_config,
            sae_id=sae_id,
            save_path=save_path,
        )

        # Add to all results
        all_results["normal"].extend([r.as_dict() for r in results])

    # Create evaluation config
    eval_config = EvaluationConfig(
        model_name=config.model_name,
        settings=config.settings or [],
    )

    # Summarize results
    summaries = summarize_results(
        results=all_results,
        config=eval_config,
    )

    return {
        "results": all_results,
        "summaries": summaries,
    }
