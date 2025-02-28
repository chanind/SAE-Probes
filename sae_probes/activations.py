"""Utilities for generating model and SAE activations."""

from dataclasses import dataclass
from pathlib import Path

import torch
from sae_lens import SAE
from tqdm import tqdm
from transformer_lens import HookedTransformer

from sae_probes.datasets import load_dataset


@dataclass
class ActivationConfig:
    """Configuration for activation generation."""

    model_name: str
    layer: int
    max_seq_len: int = 1024
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"


def get_hook_names(model_name: str, layer: int | None = None) -> list[str]:
    """
    Get the hook names for a specific model and optionally a specific layer.

    Args:
        model_name: Name of the model
        layer: Layer number (if None, returns hooks for all layers)

    Returns:
        List of hook names
    """
    if model_name == "gemma-2-9b":
        if layer is not None:
            return [f"blocks.{layer}.hook_resid_post"]
        else:
            return ["hook_embed"] + [
                f"blocks.{layer_num}.hook_resid_post" for layer_num in [9, 20, 31, 41]
            ]
    elif model_name == "llama-3.1-8b":
        if layer is not None:
            return [f"blocks.{layer}.hook_resid_post"]
        else:
            return ["hook_embed"] + [
                f"blocks.{layer_num}.hook_resid_post" for layer_num in [8, 16, 24, 31]
            ]
    elif model_name == "gemma-2-2b":
        if layer is not None:
            return [f"blocks.{layer}.hook_resid_post"]
        else:
            return ["hook_embed"] + [
                f"blocks.{layer}.hook_resid_post" for layer in [12]
            ]
    else:
        raise ValueError(f"Model {model_name} not supported")


def generate_model_activations(
    dataset_tag: str,
    config: ActivationConfig,
    dataset_path: Path,
    cache_path: Path,
    force_regenerate: bool = False,
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
    hook_name = f"blocks.{config.layer}.hook_resid_post"
    activation_file = activation_dir / f"{dataset_tag}_{hook_name}.pt"

    # Check if activations already exist and we're not forcing regeneration
    if activation_file.exists() and not force_regenerate:
        print(f"Loading cached activations from {activation_file}")
        return torch.load(activation_file)

    # Load dataset
    df, _, _ = load_dataset(dataset_tag, dataset_path)

    # Load model
    model = load_model(config.model_name, config.device)

    # Important to ensure correct token is at the correct position
    tokenizer = model.tokenizer
    assert tokenizer is not None
    tokenizer.truncation_side = "left"
    tokenizer.padding_side = "right"

    # Get text samples
    text = df["prompt"].tolist()

    # Get token lengths
    text_lengths = []
    for t in text:
        text_lengths.append(len(tokenizer(t)["input_ids"]))  # type: ignore

    # Generate activations
    print(f"Generating activations for {dataset_tag}")
    batch_size = 1
    all_activations = []

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


def generate_sae_activations(
    dataset_tag: str,
    sae: SAE,
    model_activations: torch.Tensor,
    config: ActivationConfig,
    cache_path: Path,
    force_regenerate: bool = False,
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
    sae_dir = cache_path / f"sae_activations_{config.model_name}"
    sae_dir.mkdir(parents=True, exist_ok=True)

    # Get SAE identifier
    sae_id = getattr(sae, "sae_id", "custom_sae")

    # Define activation file path
    activation_file = sae_dir / f"{dataset_tag}_{config.layer}_{sae_id}.pt"

    # Check if activations already exist and we're not forcing regeneration
    if activation_file.exists() and not force_regenerate:
        print(f"Loading cached SAE activations from {activation_file}")
        return torch.load(activation_file)

    # Generate SAE activations
    print(f"Generating SAE activations for {dataset_tag} with SAE {sae_id}")
    batch_size = 128
    sae_activations = []

    for i in tqdm(range(0, len(model_activations), batch_size)):
        batch = model_activations[i : i + batch_size].to(config.device)
        sae_activations.append(sae.encode(batch).cpu())

    # Concatenate activations
    activations = torch.cat(sae_activations)

    # Save activations
    torch.save(activations, activation_file)

    return activations


def load_model(model_name: str, device: str = "cuda:0") -> HookedTransformer:
    """
    Load a model using HookedTransformer.

    Args:
        model_name: Name of the model
        device: Device to load model on

    Returns:
        HookedTransformer model
    """
    if model_name == "gemma-2-9b":
        return HookedTransformer.from_pretrained("google/gemma-2-9b", device=device)
    elif model_name == "llama-3.1-8b":
        return HookedTransformer.from_pretrained(
            "meta-llama/Llama-3.1-8B", device=device
        )
    elif model_name == "gemma-2-2b":
        return HookedTransformer.from_pretrained("google/gemma-2-2b", device=device)
    else:
        raise ValueError(f"Model {model_name} not supported")
