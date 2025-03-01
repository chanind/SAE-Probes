"""Utilities for generating model and SAE activations."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from sae_lens import SAE
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer


@dataclass
class ActivationConfig:
    """Configuration for activation generation."""

    model_name: str
    hook_name: str
    max_seq_len: int = 1024
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"


@torch.inference_mode()
def generate_model_activations(
    model: HookedTransformer,
    dataset_tag: str,
    dataset_df: pd.DataFrame,
    config: ActivationConfig,
    cache_path: Path,
    batch_size: int,
    autocast: bool,
    force_regenerate: bool,
) -> torch.Tensor:
    """
    Generate model activations for a dataset.

    Args:
        dataset_tag: Tag identifying the dataset
        config: Configuration for activation generation
        dataset_path: Path to the directory containing dataset files
        cache_path: Path to cache directory for storing activations
        force_regenerate: Whether to force regeneration of activations

    Returns:
        Tensor of model activations
    """
    # Create cache directory if it doesn't exist
    activation_dir = cache_path / f"model_activations_{config.model_name}"
    activation_dir.mkdir(parents=True, exist_ok=True)

    # Define hook names and activation file path
    hook_name = config.hook_name
    activation_file = activation_dir / f"{dataset_tag}_{hook_name}.pt"

    # Check if activations already exist and we're not forcing regeneration
    if activation_file.exists() and not force_regenerate:
        print(f"Loading cached activations from {activation_file}")
        return torch.load(activation_file)

    # Important to ensure correct token is at the correct position
    tokenizer = model.tokenizer
    assert tokenizer is not None
    tokenizer.truncation_side = "left"
    tokenizer.padding_side = "right"

    # Get text samples
    text = dataset_df["prompt"].tolist()

    # Get token lengths
    text_lengths = []
    for t in text:
        text_lengths.append(len(tokenizer(t)["input_ids"]))  # type: ignore

    # Generate activations
    print(f"Generating activations for {dataset_tag}")
    all_activations = []

    with torch.autocast(
        device_type=config.device, dtype=torch.bfloat16, enabled=autocast
    ):
        for i in tqdm(range(0, len(text), batch_size)):
            batch_text = text[i : i + batch_size]
            batch_lengths = text_lengths[i : i + batch_size]

            batch = tokenizer(
                batch_text,
                padding=True,
                truncation=True,
                max_length=config.max_seq_len,
                return_tensors="pt",
            )
            batch = batch.to(config.device)

            logits, cache = model.run_with_cache(
                batch["input_ids"], names_filter=[hook_name]
            )

            for j, length in enumerate(batch_lengths):
                activation_pos = min(length - 1, config.max_seq_len - 1)
                all_activations.append(cache[hook_name][j, activation_pos].cpu())

    # Concatenate activations
    activations = torch.stack(all_activations)

    # Save activations
    torch.save(activations, activation_file)

    return activations


@torch.inference_mode()
def generate_sae_activations(
    dataset_tag: str,
    sae: SAE,
    model_activations: torch.Tensor,
    config: ActivationConfig,
    batch_size: int,
    autocast: bool,
) -> torch.Tensor:
    """
    Generate SAE activations for a dataset.

    Args:
        dataset_tag: Tag identifying the dataset
        sae: SAE object
        model_activations: Model activations to encode
        config: Configuration for activation generation
        cache_path: Path to cache directory for storing activations
        force_regenerate: Whether to force regeneration of activations

    Returns:
        Tensor of SAE activations
    """
    # Create cache directory if it doesn't exist

    # Get SAE identifier
    sae_id = getattr(sae, "sae_id", "custom_sae")

    # Generate SAE activations
    print(f"Generating SAE activations for {dataset_tag} with SAE {sae_id}")
    sae_activations = []

    with torch.autocast(
        device_type=config.device, dtype=torch.bfloat16, enabled=autocast
    ):
        for i in tqdm(range(0, len(model_activations), batch_size)):
            batch = model_activations[i : i + batch_size].to(config.device)
            sae_activations.append(sae.encode(batch).cpu())

    # Concatenate activations
    activations = torch.cat(sae_activations)
    return activations


def load_model(
    model_name: str,
    from_pretrained_kwargs: dict[str, Any] | None = None,
    device: str = "cuda",
) -> HookedTransformer:
    """
    Load a model using HookedTransformer.

    Args:
        model_name: Name of the model
        from_pretrained_kwargs: Keyword arguments for from_pretrained
        device: Device to load model on
    Returns:
        HookedTransformer model
    """
    return HookedTransformer.from_pretrained_no_processing(
        model_name, device=device, **(from_pretrained_kwargs or {})
    )
