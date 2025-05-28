# %%

import matplotlib.pyplot as plt
import numpy as np
import natsort
import os

os.makedirs("plots", exist_ok=True)
import pandas as pd

unique_widths = ["16k", "131k", "1m"]
df = pd.read_csv("results/sae_probes_gemma-2-9b/normal_setting/all_metrics.csv")
df_l1 = df[df["reg_type"] == "l1"]
df_l2 = df[df["reg_type"] == "l2"]

chosen_sae_id = "layer_20/width_1m/average_l0_193"

# Filter for chosen SAE ID
df_chosen = df[df["sae_id"] == chosen_sae_id]

# Get number of datasets and calculate subplot layout
datasets = natsort.natsorted(df_chosen["dataset"].unique())
n_datasets = len(datasets)
n_cols = 10
n_rows = (n_datasets + n_cols - 1) // n_cols

# Create figure with subplots
fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.75, 9), sharex=True, sharey=True)
axes = axes.flatten()
# Plot line for each dataset in its own subplot
for idx, dataset in enumerate(datasets):
    df_dataset = df_chosen[df_chosen["dataset"] == dataset]

    # Plot L1 regularization in blue
    df_l1_dataset = df_dataset[df_dataset["reg_type"] == "l1"]
    by_k_l1 = df_l1_dataset.groupby("k")["test_auc"].mean()
    axes[idx].semilogx(by_k_l1.index, by_k_l1.values, "-o", color="blue", markersize=2)
    axes[idx].set_xticklabels([])  # Hide x-axis tick labels

    # Truncate to first 30 chars
    dataset = dataset[:30]

    # Word wrap title to ~12 chars per line
    title_words = dataset.replace("-", "_").split("_")
    wrapped_lines = []
    current_line = []
    current_length = 0

    for word in title_words:
        if current_length + len(word) > 12:
            if current_line:  # Only join if there are words to join
                wrapped_lines.append(" ".join(current_line))
            current_line = [word]
            current_length = len(word)
        else:
            current_line.append(word)
            current_length += len(word) + 1  # +1 for space

    if current_line:  # Add any remaining words
        wrapped_lines.append(" ".join(current_line))

    wrapped_title = "\n".join(wrapped_lines)
    axes[idx].set_title(wrapped_title, fontsize=6)
    axes[idx].set_ylim(0.49, 1.02)

    # Hide ticks and axes
    axes[idx].set_xticks([])
    axes[idx].set_yticks([])


# Hide empty subplots
for idx in range(n_datasets, len(axes)):
    axes[idx].set_visible(False)


fig.supxlabel(
    "Number of Features in $\\mathcal{I}$ (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)",
    fontsize=8,
)
fig.supylabel("Test AUC (0.5 - 1.0)", fontsize=8)

# plt.suptitle(f'Test AUC vs k by Dataset\nSAE: {chosen_sae_id}')
plt.tight_layout(h_pad=0.5, w_pad=0.5)

plt.show()
# %%
