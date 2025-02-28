"""SAE probing model training and evaluation."""

import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm


@dataclass
class ProbeConfig:
    """Configuration for probe training."""

    reg_type: Literal["l1", "l2", "elasticnet"] = "l1"  # l1 or l2 regularization
    k_values: list[int] | None = None
    binarize: bool = False  # Whether to binarize SAE features
    seed: int = 42  # Random seed

    def __post_init__(self):
        if self.k_values is None:
            self.k_values = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]


@dataclass
class ProbeResults:
    """Results from probe training."""

    auc: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    model: Any
    feature_indices: list[int]
    k: int


def select_features(X_train: torch.Tensor, y_train: np.ndarray, k: int) -> list[int]:
    """
    Select top k features based on class difference.

    Args:
        X_train: Training features
        y_train: Training labels
        k: Number of features to select

    Returns:
        Indices of top k features
    """
    # Calculate class difference
    X_train_diff = X_train[y_train == 1].mean(dim=0) - X_train[y_train == 0].mean(dim=0)
    # Sort by absolute difference
    sorted_indices = torch.argsort(torch.abs(X_train_diff), descending=True)
    # Select top k
    return sorted_indices[:k].tolist()


def train_probe(
    X_train: torch.Tensor | np.ndarray,
    y_train: np.ndarray,
    X_test: torch.Tensor | np.ndarray,
    y_test: np.ndarray,
    config: ProbeConfig,
) -> list[ProbeResults]:
    """
    Train linear probes on SAE features.

    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        config: Probe configuration

    Returns:
        List of probe results for different k values
    """
    results = []

    # Convert tensors to numpy if needed
    if isinstance(X_train, torch.Tensor):
        X_train = X_train.cpu().numpy()
    if isinstance(X_test, torch.Tensor):
        X_test = X_test.cpu().numpy()

    # Set random seed
    np.random.seed(config.seed)

    # Get class balance of training set
    pos_ratio = np.mean(y_train)
    neg_weight = pos_ratio / (1 - pos_ratio) if pos_ratio < 0.5 else 1.0
    class_weight = {0: neg_weight, 1: 1.0}

    # Get feature indices sorted by class difference
    feature_indices = select_features(
        torch.tensor(X_train), y_train, max(config.k_values or [])
    )

    # Train models for different k values
    for k in tqdm(config.k_values):
        # Select top k features
        k_indices = feature_indices[:k]
        X_train_k = X_train[:, k_indices]
        X_test_k = X_test[:, k_indices]

        # Binarize if requested
        if config.binarize:
            X_train_k = (X_train_k > 0).astype(float)
            X_test_k = (X_test_k > 0).astype(float)

        # Train model
        C_values = np.logspace(-4, 4, 20)
        best_auc = -1
        best_model = None

        for C in C_values:
            # Create model
            model = LogisticRegression(
                penalty=config.reg_type,
                C=C,
                solver="liblinear",
                max_iter=1000,
                class_weight=class_weight,
                random_state=config.seed,
            )

            # Train model
            model.fit(X_train_k, y_train)

            # Evaluate on validation set
            y_pred_proba = model.predict_proba(X_test_k)[:, 1]
            auc = roc_auc_score(y_test, y_pred_proba)

            if auc > best_auc:
                best_auc = auc
                best_model = model

        if best_model is None:
            raise ValueError("No model was trained")

        # Evaluate best model
        y_pred_proba = best_model.predict_proba(X_test_k)[:, 1]
        y_pred = best_model.predict(X_test_k)

        auc = roc_auc_score(y_test, y_pred_proba)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)  # type: ignore
        recall = recall_score(y_test, y_pred, zero_division=0)  # type: ignore
        f1 = f1_score(y_test, y_pred, zero_division=0)  # type: ignore

        # Store results
        results.append(
            ProbeResults(
                auc=float(auc),
                accuracy=float(accuracy),
                precision=float(precision),
                recall=float(recall),
                f1=float(f1),
                model=best_model,
                feature_indices=k_indices,
                k=k,
            )
        )

    return results


def save_probe_results(
    results: list[ProbeResults],
    dataset: str,
    config: ProbeConfig,
    sae_id: str,
    save_path: Path,
) -> None:
    """
    Save probe results to file.

    Args:
        results: List of probe results
        dataset: Dataset tag
        config: Probe configuration
        sae_id: SAE identifier
        save_path: Path to save results
    """
    # Create parent directory if it doesn't exist
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert results to dictionaries
    results_dicts = [asdict(r) for r in results]

    # Add metadata
    for r in results_dicts:
        r["dataset"] = dataset
        r["reg_type"] = config.reg_type
        r["binarize"] = config.binarize
        r["sae_id"] = sae_id

    # Save results
    with open(save_path, "w") as f:
        json.dump(results_dicts, f, indent=4, ensure_ascii=False)


def load_probe_results(load_path: Path) -> list[dict]:
    """
    Load probe results from file.

    Args:
        load_path: Path to load results from

    Returns:
        List of result dictionaries
    """
    with open(load_path, "rb") as f:
        return pickle.load(f)
