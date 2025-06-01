# %%
import pickle as pkl
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from sae_probes.utils_data import (
    get_dataset_sizes,
    get_numbered_binary_tags,
)

warnings.simplefilter("ignore", category=ConvergenceWarning)

model_name = "gemma-2-9b"
max_seq_len = 256
layer = 20
k = 128

# SAE ID we want to analyze
l0 = 408
sae_id = f"layer_20/width_16k/average_l0_{l0}"

binarize = True

baseline_csv = pd.read_csv(
    f"results/baseline_probes_{model_name}/normal_settings/layer{layer}_results.csv"
)
if binarize:
    sae_csv = pd.read_csv(
        f"results/sae_probes_{model_name}/normal_setting/all_metrics_binarized.csv"
    )
else:
    sae_csv = pd.read_csv(
        f"results/sae_probes_{model_name}/normal_setting/all_metrics.csv"
    )

datasets = get_numbered_binary_tags()
dataset_sizes = get_dataset_sizes()


# %%
# Get all available files in consolidated_probing dir
import glob

# Create data for DataFrame
data = []
for dataset in datasets:
    row = {"Dataset": dataset}

    # Get baseline test AUC from CSV
    baseline_row = baseline_csv[
        (baseline_csv["method"] == "logreg") & (baseline_csv["dataset"] == dataset)
    ]
    row["Baseline (last)"] = (
        baseline_row["test_auc"].iloc[0] if not baseline_row.empty else None
    )
    row["Baseline (last) val"] = (
        baseline_row["val_auc"].iloc[0] if not baseline_row.empty else None
    )

    # Get SAE original AUC from CSV for both l0 values
    for l0 in [68, 408]:
        sae_id = f"layer_20/width_16k/average_l0_{l0}"
        sae_row = sae_csv[
            (sae_csv["dataset"] == dataset)
            & (sae_csv["k"] == k)
            & (sae_csv["sae_id"] == sae_id)
        ]
        row[f"SAE (last) l0={l0}"] = (
            sae_row["test_auc"].iloc[0] if not sae_row.empty else None
        )
        row[f"SAE (last) l0={l0} val"] = (
            sae_row["val_auc"].iloc[0] if not sae_row.empty else None
        )
    # Load consolidated probing results from pickle files if they exist
    consolidated_files = glob.glob(
        f"data/consolidated_probing_{model_name}/{dataset}_{layer}_*.pkl"
    )
    for file in consolidated_files:
        with open(file, "rb") as f:
            metrics = pkl.load(f)

        if "width16k_l0" in file:
            # Extract l0 value from filename
            l0_val = int(file.split("l0")[-1].split("_")[0])
            if (file.endswith("_mean_binarized.pkl") and binarize) or (
                file.endswith("_mean.pkl") and not binarize
            ):
                row[f"SAE (mean) l0={l0_val}"] = metrics["test_auc"]
                row[f"SAE (mean) l0={l0_val} val"] = metrics["val_auc"]
            elif (file.endswith("_max_binarized.pkl") and binarize) or (
                file.endswith("_max.pkl") and not binarize
            ):
                row[f"SAE (max) l0={l0_val}"] = metrics["test_auc"]
                row[f"SAE (max) l0={l0_val} val"] = metrics["val_auc"]
        elif "baseline_255_20" in file:
            row["Baseline (concat)"] = metrics["test_auc"]
            row["Baseline (concat) val"] = metrics["val_auc"]
        elif "attn_probing" in file:
            row["Attention-Like Probe"] = metrics["test_auc"]
            row["Attention-Like Probe val"] = metrics["val_auc"]
    data.append(row)

# Create DataFrame
df = pd.DataFrame(data)

# Drop rows where we only have baseline and SAE (last) results
df = df.dropna(
    subset=[
        "SAE (mean) l0=68",
        "SAE (max) l0=68",
        "SAE (mean) l0=408",
        "SAE (max) l0=408",
        "Baseline (concat)",
        "Attention-Like Probe",
    ],
    how="all",
)

# %%
# Create LaTeX table
print("\\begin{tabular}{l" + "c" * 10 + "}")
print("\\toprule")
print(
    "Dataset & Baseline (last) & Baseline (concat) & SAE (last) l0=68 & SAE (mean) l0=68 & SAE (max) l0=68 & SAE (last) l0=408 & SAE (mean) l0=408 & SAE (max) l0=408 & Attention\\-Like Probe \\\\"
)
print("\\midrule")

for _, row in df.iterrows():
    values = [
        row["Baseline (last)"],
        row["Baseline (concat)"],
        row["SAE (last) l0=68"],
        row["SAE (mean) l0=68"],
        row["SAE (max) l0=68"],
        row["SAE (last) l0=408"],
        row["SAE (mean) l0=408"],
        row["SAE (max) l0=408"],
        row["Attention-Like Probe"],
    ]
    max_val = max(values)

    # Format each value, making max bold with dollar signs
    formatted = []
    for v in values:
        if v == max_val:
            formatted.append(f"$\\textbf{{{v:.3f}}}$")
        else:
            formatted.append(f"{v:.3f}")

    # Shorten dataset name if longer than 10 chars
    dataset_name = row["Dataset"]
    if len(dataset_name) > 10:
        dataset_name = dataset_name[:10]

    table_row = (
        f"{dataset_name.replace('_', '\\_')} & " + " & ".join(formatted) + " \\\\"
    )
    print(table_row)

print("\\bottomrule")
print("\\end{tabular}")

# %%
# Create HTML table
html_table = "<table style='border-collapse: collapse; text-align: center'>\n"

# Add header row
cols = [
    "Dataset",
    "Baseline (last)",
    "Baseline (concat)",
    "SAE (last) l0=68",
    "SAE (mean) l0=68",
    "SAE (max) l0=68",
    "SAE (last) l0=408",
    "SAE (mean) l0=408",
    "SAE (max) l0=408",
    "Attention-Like Probe",
]
html_table += "<tr>"
for col in cols:
    html_table += f"<th style='padding: 8px; border: 1px solid'>{col}</th>"
html_table += "</tr>\n"

# Add data rows
for _, row in df.iterrows():
    values = [row[col] for col in cols[1:]]  # Skip 'Dataset' column for values
    max_val = max(values)

    html_table += "<tr>"
    html_table += f"<td style='padding: 8px; border: 1px solid'>{row['Dataset']}</td>"

    for v in values:
        style = "padding: 8px; border: 1px solid"
        if abs(v - max_val) <= 0.005:
            style += "; font-weight: bold; color: #0066cc"
        html_table += f"<td style='{style}'>{v:.3f}</td>"

    html_table += "</tr>\n"

html_table += "</table>"

# Save HTML table to file
with open("probing_results_table.html", "w") as f:
    f.write("""
    <html>
    <head>
        <style>
            body { font-family: Arial, sans-serif; }
        </style>
    </head>
    <body>
    """)
    f.write(html_table)
    f.write("\n</body></html>")

from IPython.display import HTML

display(HTML(html_table))

# %%

margin_of_error = 0.005

# Calculate percentages
baseline_last = df["Baseline (last)"]
baseline_attn = df["Attention-Like Probe"]
sae_408_last = df["SAE (last) l0=408"]
sae_408_max = df["SAE (max) l0=408"]

# % where baseline last > SAE 408 last (with margin)
pct_baseline_beats_sae_last = (
    (baseline_last - sae_408_last) > margin_of_error
).mean() * 100

pct_sae_last_beats_baseline = (
    (sae_408_last - baseline_last) > margin_of_error
).mean() * 100

# % where baseline last > both SAE 408 last and max (with margin)
pct_baseline_beats_both = (
    ((baseline_last - sae_408_last) > margin_of_error)
    & ((baseline_last - sae_408_max) > margin_of_error)
).mean() * 100

pct_sae_max_or_sae_last_beats_baseline = (
    ((sae_408_max - baseline_last) > margin_of_error)
    | ((sae_408_last - baseline_last) > margin_of_error)
).mean() * 100

pct_baseline_last_or_baseline_attn_beats_sae_last_and_max = (
    ((baseline_last - sae_408_last) > margin_of_error)
    & ((baseline_last - sae_408_max) > margin_of_error)
    | ((baseline_attn - sae_408_last) > margin_of_error)
    & ((baseline_attn - sae_408_max) > margin_of_error)
).mean() * 100

pct_sae_last_or_sae_max_beats_baseline_last_and_baseline_attn = (
    ((sae_408_last - baseline_last) > margin_of_error)
    & ((sae_408_last - baseline_attn) > margin_of_error)
    | ((sae_408_max - baseline_last) > margin_of_error)
    & ((sae_408_max - baseline_attn) > margin_of_error)
).mean() * 100

percent_baseline_attn_beats_sae_max = (
    (baseline_attn - sae_408_max) > margin_of_error
).mean() * 100
percent_sae_max_beats_baseline_attn = (
    (sae_408_max - baseline_attn) > margin_of_error
).mean() * 100

percent_sae_max_beats_baseline_last = (
    (sae_408_max - baseline_last) > margin_of_error
).mean() * 100
percent_baseline_last_beats_sae_max = (
    (baseline_last - sae_408_max) > margin_of_error
).mean() * 100

print(
    f"Percentage where Baseline (last) > SAE l0=408 (last) by {margin_of_error}: {pct_baseline_beats_sae_last:.1f}%"
)
print(
    f"Percentage where SAE l0=408 (last) > Baseline (last) by {margin_of_error}: {pct_sae_last_beats_baseline:.1f}%"
)
print(
    f"Percentage where Baseline (last) > both SAE l0=408 (last) and (max) by {margin_of_error}: {pct_baseline_beats_both:.1f}%"
)
print(
    f"Percentage where SAE l0=408 (max) or SAE l0=408 (last) > Baseline (last) by {margin_of_error}: {pct_sae_max_or_sae_last_beats_baseline:.1f}%"
)
print(
    f"Percentage where Baseline (last) or Baseline (attn) > SAE l0=408 (last) and (max) by {margin_of_error}: {pct_baseline_last_or_baseline_attn_beats_sae_last_and_max:.1f}%"
)
print(
    f"Percentage where SAE l0=408 (last) or SAE l0=408 (max) > Baseline (last) and (attn) by {margin_of_error}: {pct_sae_last_or_sae_max_beats_baseline_last_and_baseline_attn:.1f}%"
)
print(
    f"Percentage where Baseline (attn) > SAE l0=408 (max) by {margin_of_error}: {percent_baseline_attn_beats_sae_max:.1f}%"
)
print(
    f"Percentage where SAE l0=408 (max) > Baseline (attn) by {margin_of_error}: {percent_sae_max_beats_baseline_attn:.1f}%"
)
print(
    f"Percentage where Baseline (last) > SAE l0=408 (max) by {margin_of_error}: {percent_baseline_last_beats_sae_max:.1f}%"
)
print(
    f"Percentage where SAE l0=408 (max) > Baseline (last) by {margin_of_error}: {percent_sae_max_beats_baseline_last:.1f}%"
)

# %%


def get_test_aucs_using_quiver(df, methods):
    """Get test AUCs by selecting best method per dataset based on validation AUC.

    Args:
        df: DataFrame containing validation and test AUCs for different methods
        methods: List of method names to compare

    Returns:
        Dictionary mapping dataset names to their best test AUCs and mean/std
    """
    best_test_aucs = {}

    # Get unique datasets
    datasets = df["Dataset"].unique()

    for dataset in datasets:
        dataset_df = df[df["Dataset"] == dataset]

        # Find method with best validation AUC for this dataset
        best_val_auc = -float("inf")
        best_test_auc = None

        for method in methods:
            val_col = method + " val"
            val_auc = dataset_df[val_col].iloc[0]
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_test_auc = dataset_df[method].iloc[0]

        if best_test_auc is not None:
            best_test_aucs[dataset] = best_test_auc

    values = list(best_test_aucs.values())
    n = len(values)
    mean = np.mean(values)
    std = np.std(values)
    sem = std / np.sqrt(n)  # Standard error of the mean
    return best_test_aucs, mean, sem


baseline_methods = ["Baseline (last)", "Attention-Like Probe"]
best_baseline_aucs = get_test_aucs_using_quiver(df, baseline_methods)[0]
df["best_baseline_test_auc"] = df.apply(
    lambda row: best_baseline_aucs[row["Dataset"]], axis=1
)

best_sae_test_aucs = get_test_aucs_using_quiver(
    df, ["SAE (last) l0=408", "SAE (max) l0=408"]
)[0]
df["best_sae_test_auc"] = df.apply(
    lambda row: best_sae_test_aucs[row["Dataset"]], axis=1
)
# %%

pct_best_baseline_beats_best_sae = (
    (df["best_baseline_test_auc"] - df["best_sae_test_auc"]) > margin_of_error
).mean() * 100
pct_best_sae_beats_best_baseline = (
    (df["best_sae_test_auc"] - df["best_baseline_test_auc"]) > margin_of_error
).mean() * 100

# Create 3 bar plots showing the percentages


def plot_comparison_bars(
    fontsize=6,
    bar_font_size=6,
    titles=[
        "FAIR \nBoth = Last",
        "ILLUSION \nBaseline = Last\nSAE = Pool",
        "FAIR \nBoth =\nQuiver(Pool, Last)",
    ],
    sae_color="blue",
    baseline_color="green",
    bar_alpha=0.7,
):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(3.25, 1.75), sharey=True)

    # Plot 1: Baseline vs SAE Last
    bars1 = ax1.bar(
        [0, 1],
        [pct_baseline_beats_sae_last, pct_sae_last_beats_baseline],
        color=[baseline_color, sae_color],
        alpha=bar_alpha,
    )
    ax1.set_title(titles[0], fontsize=fontsize)
    # ax1.set_ylabel(f'Win Rate Percentage', fontsize=fontsize)
    ax1.set_ylabel(
        f"Win Rate Percentage \n> {margin_of_error} $\\Delta$ AUC", fontsize=fontsize
    )
    ax1.set_ylim(0, 70)  # Set y-axis limit to 70%
    ax1.tick_params(axis="both", labelsize=bar_font_size)
    ax1.set_xticks([])  # Remove x-axis ticks
    ax1.grid(True, axis="y", linestyle="-", alpha=0.3)
    for bar in bars1:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.1f}%",
            ha="center",
            va="bottom",
            fontsize=bar_font_size,
        )

    # Plot 2: Baseline vs Both SAE
    ax2.set_facecolor("#ffdbdb")  # Light red background
    bars2 = ax2.bar(
        [0, 1],
        [percent_baseline_last_beats_sae_max, percent_sae_max_beats_baseline_last],
        color=[baseline_color, sae_color],
        alpha=bar_alpha,
    )
    ax2.set_title(titles[1], fontsize=fontsize)
    ax2.set_ylim(0, 70)  # Set y-axis limit to 70%
    ax2.tick_params(axis="both", labelsize=bar_font_size)
    ax2.set_xticks([])  # Remove x-axis ticks
    ax2.grid(True, axis="y", linestyle="-", alpha=0.3)
    for bar in bars2:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.1f}%",
            ha="center",
            va="bottom",
            fontsize=bar_font_size,
        )

    # Plot 3: Either Baseline vs Either SAE
    bars3 = ax3.bar(
        [0, 1],
        [pct_best_baseline_beats_best_sae, pct_best_sae_beats_best_baseline],
        color=[baseline_color, sae_color],
        alpha=bar_alpha,
    )
    ax3.set_title(titles[2], fontsize=fontsize)
    ax3.set_ylim(0, 70)  # Set y-axis limit to 70%
    ax3.tick_params(axis="both", labelsize=bar_font_size)
    ax3.set_xticks([])  # Remove x-axis ticks
    ax3.grid(True, axis="y", linestyle="-", alpha=0.3)
    for bar in bars3:
        height = bar.get_height()
        ax3.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.1f}%",
            ha="center",
            va="bottom",
            fontsize=bar_font_size,
        )

    # Add a single legend for all subplots
    legend_elements = [
        plt.Rectangle(
            (0, 0), 1, 1, color=baseline_color, alpha=bar_alpha, label="Baseline > SAE"
        ),
        plt.Rectangle(
            (0, 0), 1, 1, color=sae_color, alpha=bar_alpha, label="SAE > Baseline"
        ),
    ]
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.08),
        ncol=2,
        fontsize=bar_font_size,
    )

    fig.suptitle(
        f"Consolidated Probing Comparison{' (Binarized)' if binarize else ''}",
        fontsize=fontsize,
        y=0.9,
    )

    plt.tight_layout()
    plt.savefig(
        f"plots/consolidated_probing_comparison_bars{'_binarized' if binarize else '_unbinarized'}.pdf",
        bbox_inches="tight",
    )


# Call the function with default font size
plot_comparison_bars()
# %%
