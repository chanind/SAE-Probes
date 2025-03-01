"""Tests for the activations module."""

from pathlib import Path

import pandas as pd
import pytest
import torch

from sae_probes.activations import (
    ActivationConfig,
    generate_model_activations,
    generate_sae_activations,
)


@pytest.fixture
def sample_dataset() -> pd.DataFrame:
    """Create a sample dataset for testing."""
    return pd.DataFrame(
        {
            "prompt": [
                "This is a test sentence.",
                "Another test sentence for good measure.",
                "A third test sentence to make the batch interesting.",
            ]
        }
    )


@pytest.fixture
def activation_config() -> ActivationConfig:
    """Create a sample activation config for testing."""
    return ActivationConfig(
        model_name="gpt2",
        hook_name="blocks.0.hook_resid_post",
        max_seq_len=128,
        device="cpu",
    )


def test_generate_model_activations_caching(
    gpt2_model, sample_dataset, activation_config, tmp_path: Path
):
    """Test that model activations are cached correctly."""
    # Generate activations for the first time
    activations1 = generate_model_activations(
        model=gpt2_model,
        dataset_tag="test_dataset",
        dataset_df=sample_dataset,
        config=activation_config,
        cache_path=tmp_path,
        batch_size=1,
        force_regenerate=False,
        autocast=False,
    )

    # Check that the cache file exists
    cache_dir = tmp_path / f"model_activations_{activation_config.model_name}"
    cache_file = cache_dir / f"test_dataset_{activation_config.hook_name}.pt"
    assert cache_file.exists()

    # Generate activations again without force_regenerate
    activations2 = generate_model_activations(
        model=gpt2_model,
        dataset_tag="test_dataset",
        dataset_df=sample_dataset,
        config=activation_config,
        cache_path=tmp_path,
        batch_size=1,
        force_regenerate=False,
        autocast=False,
    )

    # Check that the activations are the same object (loaded from cache)
    assert torch.equal(activations1, activations2)

    # Generate activations with force_regenerate
    activations3 = generate_model_activations(
        model=gpt2_model,
        dataset_tag="test_dataset",
        dataset_df=sample_dataset,
        config=activation_config,
        cache_path=tmp_path,
        batch_size=1,
        force_regenerate=True,
        autocast=False,
    )

    # Check that the activations are still equal in value
    assert torch.equal(activations1, activations3)


def test_generate_model_activations_batch_size_consistency(
    gpt2_model, sample_dataset, activation_config, tmp_path: Path
):
    """Test that different batch sizes produce the same activations."""
    # Generate activations with batch_size=1
    activations_batch1 = generate_model_activations(
        model=gpt2_model,
        dataset_tag="test_batch1",
        dataset_df=sample_dataset,
        config=activation_config,
        cache_path=tmp_path,
        batch_size=1,
        force_regenerate=True,
        autocast=False,
    )

    # Generate activations with batch_size=3 (full dataset)
    activations_batch3 = generate_model_activations(
        model=gpt2_model,
        dataset_tag="test_batch3",
        dataset_df=sample_dataset,
        config=activation_config,
        cache_path=tmp_path,
        batch_size=3,
        force_regenerate=True,
        autocast=False,
    )

    # Check that the activations are the same
    assert torch.allclose(activations_batch1, activations_batch3, atol=1e-5)


def test_generate_sae_activations_batch_size_consistency(
    gpt2_l4_sae, activation_config
):
    """Test that different batch sizes produce the same SAE activations."""
    # Create some dummy model activations
    model_activations = torch.randn(10, 768)

    # Generate SAE activations with batch_size=1
    sae_activations_batch1 = generate_sae_activations(
        dataset_tag="test_sae",
        sae=gpt2_l4_sae,
        model_activations=model_activations,
        config=activation_config,
        batch_size=1,
        autocast=False,
    )

    # Generate SAE activations with batch_size=10 (full dataset)
    sae_activations_batch10 = generate_sae_activations(
        dataset_tag="test_sae",
        sae=gpt2_l4_sae,
        model_activations=model_activations,
        config=activation_config,
        batch_size=10,
        autocast=False,
    )

    # Check that the activations are the same
    assert torch.allclose(sae_activations_batch1, sae_activations_batch10, atol=1e-5)
