"""Evaluation metrics and results processing."""

import pandas as pd


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
