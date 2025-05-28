# Are Sparse Autoencoders Useful? A Case Study in Sparse Probing

<img width="1213" alt="Screenshot 2025-02-24 at 9 58 54 PM" src="https://github.com/user-attachments/assets/09a20f0b-9f45-4382-b6c2-e70bba6c17db" />

This repository contains code to replicate experiments from our paper [_Are Sparse Autoencoders Useful? A Case Study in Sparse Probing_](https://arxiv.org/pdf/2502.16681). The workflow of our code involves three primary stages. Each part should be mostly executable independently from artifacts we make available:

1. **Generating Model and SAE Activations:**

   - Model activations for probing datasets are generated in `generate_model_activations.py`
   - SAE activations are generated in `generate_sae_activations.py`
   - OOD regime activations are specifically generated in `plot_ood.ipynb`.
   - Mutli-token activations are specifically generated in `generate_model_and_sae_multi_token_acts.py`. Caution: this will take up a lot of memory (~1TB).

2. **Training Probes:**

   - Baseline probes are trained using `run_baselines.py`. This script also includes additional functions for OOD experiments related to probe pruning and latent interpretability (see Sections 4.1 and 4.2 of the paper).
   - SAE probes are trained using `train_sae_probes.py`. Sklearn regression is most efficient when run in a single thread, and then many of those threads can be run in parallel. We include an example of how to do this in `train_sae_probes.sh`.
   - Multi token SAE probes and baseline probes are trained using `run_multi_token_acts.py`.
   - Combining all results into csvs after they are done is done with `combine_results.py`.

3. **Visualizing Results:**
   - Standard condition plots: `plot_normal.ipynb`
   - Data scarcity, class imbalance, and corrupted data regimes: `plot_combined.ipynb`
   - OOD plots: `plot_ood.ipynb`
   - Llama-3.1-8B results replication: `plot_llama.ipynb`
   - GLUE CoLA and AIMade investigations (Sections 4.3.1 and 4.3.2): `dataset_investigations/`
   - AI vs. human final token plots: `ai_vs_humanmade_plot.py`
   - SAE architectural improvements (Section 6): `sae_improvement.ipynb`
   - Multi token: `plot_multi_token.py`
   - K vs. AUC plot broken down by dataset (in appendix): `k_vs_auc_plot.py`

Note that these should all be runnable as is from the results data in the repo.

### Datasets

- **Raw Text Datasets:** Accessible via [Dropbox link](https://www.dropbox.com/scl/fo/lvajx9100jsy3h9cvis7q/AIocXXICIwHsz-HsXSekC3Y?rlkey=tq7td61h1fufm01cbdu2oqsb5&st=aorlnph5&dl=0).
- **Model Activations:** Also stored on Dropbox (Note: Files are large).

## Requirements

We recommend you create a new python venv named probing and install required packages with pip:

```
python -m venv probing
source probing/bin/activate
pip install transformer_lens sae_lens transformers datasets torch xgboost sae_bench scikit-learn natsort
```

Let us know if anything does not work with this environment!

For any questions or clarifications, please open an issue or reach out to us!

# SAE Probes

A Python package for running and analyzing SAE (Sparse Autoencoder) probe experiments and baseline interpretability methods, designed to reproduce the results from the paper [_Are Sparse Autoencoders Useful? A Case Study in Sparse Probing_](https://arxiv.org/pdf/2502.16681).

## Installation

Ensure you have [Poetry](https://python-poetry.org/docs/#installation) installed. Then, from the root of this repository, run:

```bash
poetry install
```

This will install all necessary dependencies, including PyTorch with CUDA support if available.

## Reproducing Paper Results

The core logic for running experiments is now part of the `sae_probes` package. The original scripts have been moved into this package.

### Data Setup

1.  **Raw Text Datasets:** Download and place them into a `data/` directory at the root of this project. Accessible via [Dropbox link](https://www.dropbox.com/scl/fo/lvajx9100jsy3h9cvis7q/AIocXXICIwHsz-HsXSekC3Y?rlkey=tq7td61h1fufm01cbdu2oqsb5&st=aorlnph5&dl=0).
2.  **Model and SAE Activations:** These are also required and should be placed appropriately within the `data/` directory.
    - Model activations should be in `data/model_activations_{model_name}_{max_seq_len}/`
    - SAE activations should be in `data/sae_activations_{model_name}/` for single token experiments and `data/sae_activations_{model_name}_{max_seq_len}/` for multi-token experiments.
    - (Note: Files are large and also available on Dropbox - see original links in the paper or above section if still present).

### Running Experiments

The primary scripts for training probes and generating results are:

- `sae_probes.run_baselines`: For training baseline probes.
- `sae_probes.train_sae_probes`: For training SAE probes.
- `sae_probes.run_multi_token_acts`: For multi-token experiments.
- `sae_probes.combine_results`: For combining results from various experiments into CSV files.

You can run these modules directly using `python -m`:

**1. Training Baseline Probes:**

```bash
python -m sae_probes.run_baselines --reg_type <l1_or_l2> --setting <normal|scarcity|noise|imbalance> --model_name <model_name>
```

Example:

```bash
python -m sae_probes.run_baselines --reg_type l1 --setting normal --model_name gemma-2-9b
```

**2. Training SAE Probes:**

The `train_sae_probes.py` script is designed to run multiple experiments, iterating through datasets, layers, and SAEs. It can be launched similarly:

```bash
python -m sae_probes.train_sae_probes --reg_type <l1_or_l2> --setting <normal|scarcity|noise|imbalance> --model_name <model_name>
```

Example:

```bash
python -m sae_probes.train_sae_probes --reg_type l1 --setting normal --model_name gemma-2-9b
```

For parallel execution of SAE probe training (as recommended in the paper), you might adapt the example `train_sae_probes.sh` script or use your preferred method for running multiple Python processes. The script itself will loop through configurations until all probes are trained.

**3. Running Multi-Token Experiments:**

```bash
python -m sae_probes.run_multi_token_acts --l0 <l0_value> --to_run_list <baseline_attn|sae_aggregated|attn_probing> [<another_choice> ...]
```

Example:

```bash
python -m sae_probes.run_multi_token_acts --l0 68 --to_run_list baseline_attn sae_aggregated attn_probing
```

### Expected Output

Results (pickle files from individual probe trainings and combined CSVs) will be saved in a `results/` directory at the root of this project, structured by model name and experiment setting (e.g., `results/sae_probes_gemma-2-9b/normal_setting/`).

### Running Experiments Programmatically (Python API)

Instead of running the scripts from the command line, you can import and use the functions directly in Python. This offers more flexibility for custom workflows or integration into other projects.

**1. Training Baseline Probes (`sae_probes.run_baselines`):**

The primary functions for running different experiment settings are:

```python
from sae_probes import run_baselines

# For normal setting (iterates through all layers and datasets)
run_baselines.run_all_baseline_normal(model_name="gemma-2-9b")
run_baselines.coalesce_all_baseline_normal(model_name="gemma-2-9b") # To combine results

# For data scarcity (default layer is 20)
run_baselines.run_all_baseline_scarcity(model_name="gemma-2-9b", layer=20)
run_baselines.coalesce_all_scarcity(model_name="gemma-2-9b", layer=20)

# For class imbalance (default layer is 20)
run_baselines.run_all_baseline_class_imbalance(model_name="gemma-2-9b", layer=20)
run_baselines.coalesce_all_imbalance(model_name="gemma-2-9b", layer=20)

# For label noise/corruption (default layer is 20)
run_baselines.run_all_baseline_corrupt(model_name="gemma-2-9b", layer=20)
run_baselines.coalesce_all_corrupt(model_name="gemma-2-9b", layer=20)
```

Other specialized experiment functions like `run_datasets_OOD`, `run_glue`, `latent_performance`, `ood_pruning`, and `examine_glue_classifier` are also available in `sae_probes.run_baselines`.

**2. Training SAE Probes (`sae_probes.train_sae_probes`):**

Similar to baselines, there are functions for different settings:

```python
from sae_probes import train_sae_probes

# For normal setting
train_sae_probes.run_normal_baselines(reg_type="l1", model_name="gemma-2-9b", binarize=False, target_sae_id=None)

# For data scarcity
train_sae_probes.run_scarcity_baselines(reg_type="l1", model_name="gemma-2-9b", target_sae_id=None)

# For label noise
train_sae_probes.run_noise_baselines(reg_type="l1", model_name="gemma-2-9b", target_sae_id=None)

# For class imbalance
train_sae_probes.run_imbalance_baselines(reg_type="l1", model_name="gemma-2-9b", target_sae_id=None)
```

These functions will iterate through relevant datasets, layers, and SAEs as defined in the utility functions they call. Results for each individual run are saved as pickle files. You would typically call `sae_probes.combine_results.process_setting` (see below) afterwards to aggregate these.

**3. Running Multi-Token Experiments (`sae_probes.run_multi_token_acts`):**

This module requires careful setup of parameters. The main functions to call, likely within a loop for each `dataset` you want to process, are:

```python
from sae_probes import run_multi_token_acts
from sae_probes.utils_data import get_numbered_binary_tags # To get dataset names

# Example: Set parameters (these might need to be adjusted or exposed as function args for full flexibility)
model_name = "gemma-2-9b" # As used internally in the script
layer = 20 # As used internally
l0_val = 68 # Example L0 value
sae_identifier = f"layer_{layer}/width_16k/average_l0_{l0_val}"
k_features = 128 # As used internally for SAE aggregated probes

datasets_to_run = get_numbered_binary_tags() # Or a subset

all_baseline_concat_results = []
all_sae_aggregated_results = []
all_attn_probing_results = []

for dataset_name in datasets_to_run:
    print(f"Running multi-token experiments for {dataset_name}")

    # Baseline concatenation
    baseline_res = run_multi_token_acts.run_baseline_concat_probing(dataset_name, layer, sae_identifier)
    all_baseline_concat_results.append(baseline_res)

    # SAE aggregated
    sae_agg_res = run_multi_token_acts.run_sae_aggregated_probing(dataset_name, layer, sae_identifier, k=k_features, binarize=False)
    all_sae_aggregated_results.append(sae_agg_res)

    # Attention probing
    attn_res = run_multi_token_acts.train_attn_probing_on_model_acts(dataset_name, layer)
    all_attn_probing_results.append(attn_res)

# Then save all_baseline_concat_results, all_sae_aggregated_results, all_attn_probing_results to .pkl files as done in the original script.
# e.g., using pickle.dump. Paths would be like: f"results/multi_token_probes_{model_name}/baseline_concat_results_l0_{l0_val}.pkl"
```

Note: The `run_multi_token_acts.py` script has several hardcoded variables (e.g., `data_dir`, `model_name`, `max_seq_len`, `layer`, `k`, `device`). For robust programmatic use, it's recommended to refactor these functions to accept such parameters as arguments.

**4. Combining Results (`sae_probes.combine_results`):**

After individual experiment runs (especially for SAE probes which save many small files), you can aggregate them:

```python
from sae_probes import combine_results

# Example for 'normal' setting and 'gemma-2-9b' model
combine_results.process_setting(setting="normal", model_name="gemma-2-9b")

# Example for 'scarcity' setting (assuming results are from SAE probes)
combine_results.process_setting(setting="scarcity", model_name="gemma-2-9b")
```

The `process_setting` function looks for `.pkl` files in structured directories (e.g., `data/sae_probes_{model_name}/{setting}_setting/`) and creates a combined CSV in `results/sae_probes_{model_name}/{setting}_setting/`.

## Original Code for Comparison
