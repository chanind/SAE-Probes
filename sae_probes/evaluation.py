"""Evaluation metrics and results processing."""

import glob
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""

    model_name: str
    settings: list[str]  # normal, scarcity, imbalance, noise
    metrics: list[str] = None  # Metrics to include in summary

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = ["auc", "accuracy", "precision", "recall", "f1"]


def collect_results(
    results_path: Path,
    config: EvaluationConfig,
) -> dict[str, dict]:
    """
    Collect and organize probe results.

    Args:
        results_path: Path to results directory
        config: Evaluation configuration

    Returns:
        Dictionary mapping from setting to results data
    """
    all_results = {}

    for setting in config.settings:
        setting_path = (
            results_path / f"sae_probes_{config.model_name}" / f"{setting}_setting"
        )

        if not setting_path.exists():
            print(f"WARNING: Path {setting_path} does not exist, skipping")
            continue

        # Use glob to find all result files
        result_files = glob.glob(str(setting_path / "*.pkl"))

        if not result_files:
            print(f"WARNING: No result files found in {setting_path}")
            continue

        # Load and combine results
        combined_results = []
        for file_path in result_files:
            with open(file_path, "rb") as f:
                results = pickle.load(f)
                combined_results.extend(results)

        all_results[setting] = combined_results

    return all_results


def summarize_results(
    results: dict[str, list],
    config: EvaluationConfig,
    k_focus: list[int] = None,
) -> dict[str, pd.DataFrame]:
    """
    Summarize results by setting.

    Args:
        results: Dictionary of results by setting
        config: Evaluation configuration
        k_focus: List of k values to focus on (if None, includes all)

    Returns:
        Dictionary of summary DataFrames by setting
    """
    summaries = {}

    for setting, setting_results in results.items():
        # Convert to DataFrame
        df = pd.DataFrame(setting_results)

        # Filter by k values if specified
        if k_focus:
            df = df[df["k"].isin(k_focus)]

        # Group by dataset, k and compute mean, std
        grouped = df.groupby(["dataset", "k"])

        # Create summary with mean values
        summary = grouped[config.metrics].mean().reset_index()

        # Rename columns for clarity
        for metric in config.metrics:
            summary = summary.rename(columns={metric: f"{metric}_mean"})

        # Add std values
        for metric in config.metrics:
            stds = grouped[metric].std().reset_index()[metric]
            summary[f"{metric}_std"] = stds.values

        summaries[setting] = summary

    return summaries


def compare_sae_vs_baseline(
    sae_results: dict[str, pd.DataFrame],
    baseline_results: dict[str, pd.DataFrame],
    metric: str = "auc",
    k_value: int = 128,
) -> pd.DataFrame:
    """
    Compare SAE probe results with baseline results.

    Args:
        sae_results: Dictionary of SAE probe results by setting
        baseline_results: Dictionary of baseline probe results by setting
        metric: Metric to compare
        k_value: k value to compare

    Returns:
        DataFrame with comparison statistics
    """
    all_comparisons = []

    for setting in sae_results.keys():
        if setting not in baseline_results:
            print(f"WARNING: Setting {setting} not found in baseline results, skipping")
            continue

        sae_df = sae_results[setting]
        baseline_df = baseline_results[setting]

        # Filter by k value
        sae_df = sae_df[sae_df["k"] == k_value]
        baseline_df = baseline_df[baseline_df["k"] == k_value]

        # Join on dataset
        merged = sae_df.merge(baseline_df, on="dataset", suffixes=("_sae", "_baseline"))

        # Calculate improvement
        merged[f"{metric}_diff"] = (
            merged[f"{metric}_mean_sae"] - merged[f"{metric}_mean_baseline"]
        )
        merged[f"{metric}_rel_imp"] = (
            merged[f"{metric}_diff"] / merged[f"{metric}_mean_baseline"] * 100
        )

        # Count wins
        merged["sae_wins"] = merged[f"{metric}_diff"] > 0

        # Add setting column
        merged["setting"] = setting

        all_comparisons.append(merged)

    return pd.concat(all_comparisons, ignore_index=True)


def calculate_win_rate(
    sae_results: dict[str, list],
    baseline_results: dict[str, list],
    metric: str = "auc",
) -> dict[str, pd.DataFrame]:
    """
    Calculate win rate of SAE probes vs baseline.

    Args:
        sae_results: Dictionary of SAE probe results by setting
        baseline_results: Dictionary of baseline probe results by setting
        metric: Metric to compare

    Returns:
        Dictionary of win rate DataFrames by setting
    """
    win_rates = {}

    for setting in sae_results.keys():
        if setting not in baseline_results:
            print(f"WARNING: Setting {setting} not found in baseline results, skipping")
            continue

        # Convert to DataFrames
        sae_df = pd.DataFrame(sae_results[setting])
        baseline_df = pd.DataFrame(baseline_results[setting])

        # Unique datasets and k values
        datasets = sorted(sae_df["dataset"].unique())
        k_values = sorted(sae_df["k"].unique())

        # Initialize win rate matrix
        win_matrix = np.zeros((len(datasets), len(k_values)))

        # Calculate win rate for each dataset and k value
        for i, dataset in enumerate(datasets):
            for j, k in enumerate(k_values):
                # Filter by dataset and k
                sae_subset = sae_df[(sae_df["dataset"] == dataset) & (sae_df["k"] == k)]
                baseline_subset = baseline_df[
                    (baseline_df["dataset"] == dataset) & (baseline_df["k"] == k)
                ]

                if sae_subset.empty or baseline_subset.empty:
                    win_matrix[i, j] = np.nan
                    continue

                # Compare metric
                sae_score = sae_subset[metric].mean()
                baseline_score = baseline_subset[metric].mean()

                win_matrix[i, j] = 1 if sae_score > baseline_score else 0

        # Create DataFrame
        win_df = pd.DataFrame(win_matrix, index=datasets, columns=k_values)

        # Add overall win rate
        win_df["overall"] = win_df.mean(axis=1)

        # Add dataset-level win rate
        win_df.loc["overall"] = win_df.mean(axis=0)

        win_rates[setting] = win_df

    return win_rates
