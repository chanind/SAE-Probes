# Are Sparse Autoencoders Useful? A Case Study in Sparse Probing

<img width="1213" alt="Screenshot 2025-02-24 at 9 58 54 PM" src="https://github.com/user-attachments/assets/09a20f0b-9f45-4382-b6c2-e70bba6c17db" />

This repository contains code to replicate experiments from our paper [_Are Sparse Autoencoders Useful? A Case Study in Sparse Probing_](https://arxiv.org/pdf/2502.16681). The workflow of our code involves three primary stages. Each part should be mostly executable independently from artifacts we make available:

1. **Generating Model and SAE Activations:**

   - Model activations for probing datasets are generated dynamically and cached by the experiment functions (e.g., `get_model_activations_for_dataset` in `sae_probes.utils_data`).
   - SAE features are computed on-the-fly from a loaded SAE and model activations (e.g., using `get_sae_features` in `sae_probes.utils_sae`).
   - The system is designed to be flexible, allowing you to use any `transformer_lens` compatible model and `sae-lens` compatible SAE.

2. **Training Probes:**

   - Probes (both baseline and SAE-based) are trained programmatically using functions within the `sae_probes` package. This allows for greater flexibility in specifying models, SAEs, layers, and datasets.
   - Key modules for training include `sae_probes.run_baselines`, `sae_probes.train_sae_probes`, and `sae_probes.run_multi_token_acts`.
   - Results are typically saved as pickle files for individual runs and can then be aggregated.

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

### Data Setup

1.  **Raw Text Datasets:** Download and place them into a `data/` directory at the root of this project. Accessible via [Dropbox link](https://www.dropbox.com/scl/fo/lvajx9100jsy3h9cvis7q/AIocXXICIwHsz-HsXSekC3Y?rlkey=tq7td61h1fufm01cbdu2oqsb5&st=aorlnph5&dl=0). These are used by the data loading utilities to fetch prompts and labels.
2.  **Model Activations (Cache):** The refactored code, particularly `sae_probes.utils_data.get_model_activations_for_dataset`, will automatically generate model activations when first needed and cache them. By default, these caches are stored in `data/generated_model_activations/`, organized by model name and layer. If you have previously computed activations matching the old path structure (e.g., `data/model_activations_{model_name}_{max_seq_len}/`), they might not be directly used by the new functions unless you adapt the caching paths or manually load them.
3.  **SAE Features:** SAE features are generally not pre-computed and stored. Instead, you load your SAE model, and the experiment functions will use `sae_probes.utils_sae.get_sae_features` to compute SAE features from model activations on the fly.
4.  **SAE Models:** You will need to provide your own trained SAE models (compatible with `sae-lens` or your custom SAE class structure). Store them in a location accessible by your scripts.

### Running Experiments

With the recent refactoring, the **primary and recommended way to run experiments is by using the Python API directly**. This offers the most flexibility for using custom models, SAEs, and configurations.

The original scripts (e.g., `run_baselines.py`, `train_sae_probes.py`) have been refactored into modules containing functions that accept `HookedTransformer` model objects and `SAE` objects. While some modules might retain a runnable `if __name__ == "__main__":` block for specific legacy workflows or examples, direct CLI execution like `python -m sae_probes.run_baselines ...` with extensive arguments for model/SAE selection is generally superseded by writing a Python script that loads your specific model/SAE and then calls the appropriate functions from the `sae_probes` package.

Please refer to the "Running Experiments Programmatically (Python API)" section below for detailed examples.

The primary modules for training probes and generating results are:

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

Results (pickle files from individual probe trainings and combined CSVs) will be saved in a `results/` directory at the root of this project, structured by model name and experiment setting (e.g., `results/sae_probes_gemma-2-9b/normal_setting/`). The exact paths will depend on the `results_base_dir` arguments used in the programmatic API calls.

### Running Experiments Programmatically (Python API)

Instead of running the scripts from the command line, you can import and use the functions directly in Python. This offers more flexibility for custom workflows or integration into other projects.

**Core Setup: Loading Model and SAE**

All programmatic experiment runs start by loading your `transformer_lens` model and your Sparse Autoencoder (SAE).

````python
import torch
from transformer_lens import HookedTransformer
from sae_lens import SAE # Or your specific SAE class if different

# --- 1. Load your Transformer Model ---
model_name = "gpt2-small" # Or "meta-llama/Llama-2-7b-hf", "gemma-2-9b", etc.
device = "cuda" if torch.cuda.is_available() else "cpu"
model = HookedTransformer.from_pretrained(model_name, device=device)

# Ensure model.cfg.model_name is set, as it's used for constructing cache paths
if not hasattr(model.cfg, 'model_name') or not model.cfg.model_name:
    model.cfg.model_name = model_name.replace("/", "_") # Basic sanitization for path

# --- 2. Load your SAE ---
# This is an example; adapt to how your SAEs are saved and structured.
# Make sure your SAE is compatible with the sae-lens SAE API (e.g., has .encode(), .W_enc, .b_enc, .device, cfg.d_in, cfg.d_sae attributes)

# Example: Loading an SAE trained with sae-lens, from a local .pt file that contains the state_dict
# sae_path = "path/to/your/sae_model/layer_10/sae_weights.pt" # Adjust path
# sae_config_path = "path/to/your/sae_model/layer_10/cfg.json" # If you have a config file
# For this example, let's assume you have the necessary config parameters:
# sae_cfg_dict = {
#     "d_in": model.cfg.d_model,       # Input dimension, should match model's residual stream
#     "d_sae": model.cfg.d_model * 4,  # SAE dictionary size (e.g., 4x d_model)
#     "hook_point": f"blocks.10.hook_resid_pre", # Hook point SAE was trained on
#     "device": device,
#     # Add other necessary SAEConfig parameters here...
#     "sae_lens_version": "0.2.0" # Example
# }
# from sae_lens.training.config import LanguageModelSAERunnerConfig
# from sae_lens.training.sae_group import SAEGroup
# runner_cfg = LanguageModelSAERunnerConfig(**sae_cfg_dict) #This is a placeholder, adapt to your SAE loading
# sae = SAE(runner_cfg.get_sae_config()) # Fictional loading
# sae.load_state_dict(torch.load(sae_path, map_location=device))
# sae.eval()
# sae.to(device)

# --- Placeholder SAE for example purposes ---
# Replace this with your actual SAE loading code
class MockSAE:
    def __init__(self, d_in, d_sae, hook_point, device_val, sae_name="mock_sae"):
        self.W_enc = torch.randn(d_in, d_sae, device=device_val)
        self.b_enc = torch.randn(d_sae, device=device_val)
        self.device = device_val
        self.cfg = MockSAEConfig(d_in, d_sae, hook_point, device_val, sae_name)

    def encode(self, x):
        # Simplified encode for mock
        x_cent = x - x.mean(dim=-1, keepdim=True) # Simplified centering
        return torch.relu(x_cent @ self.W_enc + self.b_enc)

    @staticmethod
    def load_from_pretrained(path, device): # Basic mock, replace with real loading
        print(f"Mock loading SAE from {path}")
        # Assuming d_in, d_sae can be inferred or are standard for the mock
        # In a real scenario, d_in would come from the model or SAE config
        # d_sae would come from the SAE config
        # hook_point would come from the SAE config
        # For this mock, let's use some defaults based on the loaded model.
        d_in_val = 768 # For gpt2-small like models, replace with model.cfg.d_model
        d_sae_val = d_in_val * 4
        hook_point_val = "blocks.10.hook_resid_pre" # Example
        return MockSAE(d_in_val, d_sae_val, hook_point_val, device)

    def to(self, device_val):
        self.W_enc = self.W_enc.to(device_val)
        self.b_enc = self.b_enc.to(device_val)
        self.device = device_val
        self.cfg.device = device_val
        return self

class MockSAEConfig:
    def __init__(self, d_in, d_sae, hook_point, device_val, sae_name="mock_sae"):
        self.d_in = d_in
        self.d_sae = d_sae
        self.hook_point = hook_point
        self.device = device_val
        self.sae_name = sae_name # Used for path construction in some cases


# --- Use actual SAE loading for your use case ---
# For example, if your SAE is from SAELens and saved with SAE.save_model()
# sae = SAE.load_from_pretrained("path/to/your/sae_directory_or_hf_name", device=device)
# Or if you have a state_dict and a config object:
# from sae_lens.training.config import SAEConfig
# sae_cfg_params = {"d_in": model.cfg.d_model, "d_sae": model.cfg.d_model * 4, "hook_point": f"blocks.10.hook_resid_pre", "device": device}
# sae_cfg = SAEConfig(**sae_cfg_params)
# sae = SAE(sae_cfg)
# sae.load_state_dict(torch.load("path/to/sae_state_dict.pt", map_location=device))
# sae.eval()
# sae.to(device)

# For the examples below, we'll assume 'sae' is a loaded SAE object.
# We'll use a mock SAE if actual loading is not set up.
try:
    assert 'sae' in locals() and sae is not None
except (NameError, AssertionError):
    print("Using Mock SAE for README examples. Replace with your actual SAE loading.")
    # Infer d_in from the loaded model for the mock SAE
    # This requires 'model' to be loaded first, which it is.
    example_d_in = model.cfg.d_model
    example_d_sae = example_d_in * 4 # A common expansion factor
    example_sae_layer = 10 # Make sure this layer exists in your model
    example_hook_point = f"blocks.{example_sae_layer}.hook_resid_pre"
    sae = MockSAE(example_d_in, example_d_sae, example_hook_point, device, sae_name=f"mock_sae_L{example_sae_layer}")


# --- Common Experiment Parameters ---
# Most experiment functions will require the model's layer index and the specific hook point name.
# If using an SAE, its hook point should match.
layer_idx = 10 # Example: layer 10. For SAEs, this should match the layer the SAE was trained on.
# Hook point name is often derived from layer_idx, e.g., for residual stream before MLP:
hook_point_name = f"blocks.{layer_idx}.hook_resid_pre"
# Or for attention output:
# hook_point_name = f"blocks.{layer_idx}.attn.hook_z"
# Ensure this hook_point_name is valid for your model and consistent with your SAE if used.
# The SAE's own config usually stores its hook_point: sae.cfg.hook_point

# --- Cache Directory for Generated Activations ---
# Model activations generated by utils_data.get_model_activations_for_dataset will be cached here.
# You can override this path in most experiment functions.
cache_dir_base = "data/generated_model_activations"
results_base_dir = "results" # Base directory for saving experiment outputs

# Ensure the model's device is correctly picked up
if not hasattr(model, 'device'): # Should be set by HookedTransformer
    model.device = device

# Ensure the SAE's device is correctly picked up
# Typically, sae.device or sae.cfg.device
if not hasattr(sae, 'device'): # Should be set by your SAE class
    sae.device = device
if not hasattr(sae.cfg, 'device'):
    sae.cfg.device = device


# Now you can call the experiment functions:

**1. Training Baseline Probes (`sae_probes.run_baselines`):**

The primary functions for running different baseline experiment settings have been refactored to accept the `model` object.

```python
from sae_probes import run_baselines
from sae_probes.utils_data import get_layers # Helper to get valid layer indices

# Determine valid layers for your loaded model
all_layers = get_layers(model.cfg.model_name, hf_name=True if "/" in model.cfg.model_name else False) # Adjust hf_name based on model_name format
# Example: run for a subset of layers
layers_to_run_baselines = all_layers[::max(1, len(all_layers)//3)] # e.g., a few layers spread out

# For normal setting (iterates through all layers and datasets by default if not specified)
run_baselines.run_all_baseline_normal_generic(
    model=model,
    layers_to_run=layers_to_run_baselines, # Specify layers
    device=model.device,
    results_base_dir=f"{results_base_dir}/baseline_probes",
    cache_dir_base=cache_dir_base
)
run_baselines.coalesce_all_baseline_normal_generic(
    model_name_str=model.cfg.model_name,
    layers_run=layers_to_run_baselines,
    output_base_dir=f"{results_base_dir}/baseline_probes"
)

# For data scarcity (default layer is often middle or later one, e.g., layer 20 for a 32L model)
# Choose a relevant layer for scarcity experiments
scarcity_layer = all_layers[len(all_layers)//2] if all_layers else layer_idx
run_baselines.run_all_baseline_scarcity_generic(
    model=model,
    layer_to_run=scarcity_layer,
    device=model.device,
    results_base_dir=f"{results_base_dir}/baseline_probes",
    cache_dir_base=cache_dir_base
)
run_baselines.coalesce_all_scarcity_generic(
    model_name_str=model.cfg.model_name,
    layer_run=scarcity_layer,
    output_base_dir=f"{results_base_dir}/baseline_probes"
)

# Note: Class imbalance and label noise/corruption functions in run_baselines
# may not have been updated to the generic pattern accepting a 'model' object yet.
# They might still rely on 'model_name' strings and older data loading.
# Please check their signatures in sae_probes/run_baselines.py.
# If they still use model_name, you'd call them like:
# run_baselines.run_all_baseline_class_imbalance(model_name=model.cfg.model_name, layer=scarcity_layer)
# run_baselines.coalesce_all_imbalance(model_name=model.cfg.model_name, layer=scarcity_layer)
# run_baselines.run_all_baseline_corrupt(model_name=model.cfg.model_name, layer=scarcity_layer)
# run_baslines.coalesce_all_corrupt(model_name=model.cfg.model_name, layer=scarcity_layer)

# Specialized OOD, GLUE, and other experiments now also take model/SAE objects:
# These often involve both a HookedTransformer model and an SAE.

# Example for OOD (Out-of-Distribution)
# You need a train_dataset_name and a list of ood_test_dataset_names
from sae_probes.utils_data import get_numbered_binary_tags, get_OOD_datasets
train_datasets = get_numbered_binary_tags()
ood_test_sets = get_OOD_datasets(translation=False) # Example
if train_datasets and ood_test_sets:
    run_baselines.run_datasets_OOD_generic(
        model=model,
        sae=None, # For baseline OOD. Pass your 'sae' object for SAE-based OOD.
        train_dataset_name=train_datasets[0], # Example train dataset
        ood_dataset_names=ood_test_sets[:2],    # Example OOD test datasets
        layer_idx=layer_idx, # Defined in common setup
        hook_point_name=hook_point_name, # Defined in common setup
        method_name="logreg", # Example method
        results_dir_base=f"{results_base_dir}/ood_experiments",
        device=model.device,
        cache_dir_base=cache_dir_base
    )
    # If running with an SAE:
    # run_baselines.run_datasets_OOD_generic(
    #     model=model,
    #     sae=sae, # Pass the loaded SAE object
    #     train_dataset_name=train_datasets[0],
    #     ood_dataset_names=ood_test_sets[:2],
    #     layer_idx=int(sae.cfg.hook_point.split('.')[1]), # Layer from SAE config
    #     hook_point_name=sae.cfg.hook_point, # Hook point from SAE config
    #     method_name="logreg",
    #     results_dir_base=f"{results_base_dir}/ood_experiments",
    #     device=model.device,
    #     sae_k=128, # Number of top SAE features to use
    #     cache_dir_base=cache_dir_base
    # )


# Example for GLUE (e.g., CoLA dataset)
# glue_dataset_name should be like "87_glue_cola" from get_numbered_binary_tags()
glue_cola_dataset = next((name for name in train_datasets if "glue_cola" in name), None)
if glue_cola_dataset:
    run_baselines.run_glue_generic(
        model=model,
        sae=None, # For baseline GLUE. Pass 'sae' for SAE-based GLUE.
        glue_dataset_name=glue_cola_dataset,
        layer_idx=layer_idx,
        hook_point_name=hook_point_name,
        method_name="logreg",
        test_label_type="original_target", # or "ensemble", "disagree"
        results_dir_base=f"{results_base_dir}/glue_experiments",
        device=model.device,
        cache_dir_base=cache_dir_base
    )
    # If running with an SAE:
    # run_baselines.run_glue_generic(
    #     model=model,
    #     sae=sae,
    #     glue_dataset_name=glue_cola_dataset,
    #     layer_idx=int(sae.cfg.hook_point.split('.')[1]),
    #     hook_point_name=sae.cfg.hook_point,
    #     method_name="logreg",
    #     test_label_type="original_target",
    #     results_dir_base=f"{results_base_dir}/glue_experiments",
    #     device=model.device,
    #     sae_k=128,
    #     cache_dir_base=cache_dir_base
    # )

# Latent Performance (requires an SAE)
if train_datasets and ood_test_sets and sae:
    run_baselines.latent_performance_generic(
        model=model,
        sae=sae,
        train_dataset_name=train_datasets[0],
        ood_test_dataset_name=ood_test_sets[0],
        layer_idx=int(sae.cfg.hook_point.split('.')[1]), # Layer from SAE
        hook_point_name=sae.cfg.hook_point, # Hook point from SAE
        method_name="logreg",
        results_dir_base=f"{results_base_dir}/latent_performance",
        device=model.device,
        num_top_sae_features_to_analyze=8,
        cache_dir_base=cache_dir_base
    )

# OOD Pruning (requires an SAE and a relevance CSV)
# relevance_csv_path = "path/to/your/feature_relevance.csv" # CSV with 'latent' and 'Relevance' columns
# if train_datasets and ood_test_sets and sae and Path(relevance_csv_path).exists():
#     run_baselines.ood_pruning_generic(
#         model=model,
#         sae=sae,
#         train_dataset_name=train_datasets[0],
#         ood_test_dataset_name=ood_test_sets[0],
#         layer_idx=int(sae.cfg.hook_point.split('.')[1]),
#         hook_point_name=sae.cfg.hook_point,
#         method_name="logreg",
#         relevance_csv_path=relevance_csv_path,
#         num_features_to_keep_iteratively=[1, 2, 4, 8, 16, 32, 64, 128], # Example
#         results_dir_base=f"{results_base_dir}/ood_pruning",
#         device=model.device,
#         cache_dir_base=cache_dir_base
#     )
````

**2. Training SAE Probes (`sae_probes.train_sae_probes`):**

These functions now require both the `model` and the specific `sae` object you want to probe.

```python
from sae_probes import train_sae_probes

# Ensure 'model' and 'sae' are loaded as per the Core Setup section.
# The layer_idx and hook_point_name should correspond to the loaded SAE.
sae_layer_idx = int(sae.cfg.hook_point.split('.')[1])
sae_hook_point = sae.cfg.hook_point # Or ensure hook_point_name from common setup matches sae.cfg.hook_point

# Example for normal setting
# This function will iterate through relevant datasets.
train_sae_probes.run_normal_experiments_generic(
    model=model,
    sae=sae,
    layer_idx=sae_layer_idx, # Layer SAE was trained on
    hook_point_name=sae_hook_point, # Hook point SAE was trained on
    reg_type="l1",
    binarize=False,
    device=model.device, # Device for model activations, SAE features handled internally
    results_base_dir=f"{results_base_dir}/sae_probes",
    cache_dir_base=cache_dir_base,
    # target_sae_id can be used if you have multiple SAEs for the same layer/hook_point
    # and need to distinguish their results. Often derived from sae.cfg.sae_name or similar.
    # target_sae_id=sae.cfg.sae_name.replace("/", "_") # Example
)

# Example for data scarcity
train_sae_probes.run_scarcity_experiments_generic(
    model=model,
    sae=sae,
    layer_idx=sae_layer_idx,
    hook_point_name=sae_hook_point,
    reg_type="l1",
    device=model.device,
    results_base_dir=f"{results_base_dir}/sae_probes",
    cache_dir_base=cache_dir_base,
    # target_sae_id=sae.cfg.sae_name.replace("/", "_")
)

# Similar generic functions likely exist for label noise and class imbalance:
# train_sae_probes.run_noise_experiments_generic(...)
# train_sae_probes.run_imbalance_experiments_generic(...)
# Check train_sae_probes.py for exact function names and parameters.
```

These functions will iterate through relevant datasets, and SAEs (if you pass a list or have a mechanism for discovering them). Results for each individual run are saved as pickle files. You would typically call `sae_probes.combine_results.process_setting` afterwards to aggregate these.

**3. Running Multi-Token Experiments (`sae_probes.run_multi_token_acts`):**

This module also requires the `model` and `sae` objects for its generic functions. The internal logic for handling multi-token aggregation (e.g. "mean" or "concat") and specific hook points (e.g. attention heads) will be crucial.

```python
from sae_probes import run_multi_token_acts
from sae_probes.utils_data import get_numbered_binary_tags # To get dataset names

# Ensure 'model' and 'sae' are loaded.
# Multi-token experiments often focus on specific layers and might use different hook points (e.g., attention heads).
# Adjust layer_idx and hook_point_name accordingly.
multi_token_layer_idx = int(sae.cfg.hook_point.split('.')[1]) # Or specific layer for multi-token
# Example: hook point for attention output (this needs to match what the SAE was trained on if it's an attention SAE)
# or a relevant hook for baseline attention probing.
multi_token_hook_point = f"blocks.{multi_token_layer_idx}.attn.hook_z" # Example for attention output
# Or if using a residual stream SAE:
# multi_token_hook_point = sae.cfg.hook_point

# Parameters for multi-token experiments
# sae_identifier might be used for path naming, derived from sae.cfg if applicable
sae_identifier_multi = sae.cfg.sae_name.replace("/", "_") if hasattr(sae.cfg, 'sae_name') else "default_sae_id"
k_features = 128 # Example for SAE aggregated probes

datasets_to_run = get_numbered_binary_tags() # Or a subset

all_baseline_concat_results = []
all_sae_aggregated_results = []
all_attn_probing_results = []

# Max sequence length for multi-token can be important
max_seq_len_multi = model.cfg.n_ctx # Or a smaller value if desired

for dataset_name in datasets_to_run[:1]: # Example: run on first dataset
    print(f"Running multi-token experiments for {dataset_name}")

    # Baseline concatenation probing (assumes a _generic version exists)
    # This would typically use model activations directly.
    # The hook_point_name here is crucial for defining what gets concatenated.
    # If it's blocks.L.attn.hook_z, it would be head outputs.
    # if 'run_baseline_concat_probing_generic' exists and takes the model:
    try:
        baseline_res = run_multi_token_acts.run_baseline_concat_probing_generic(
            model=model,
            dataset_name=dataset_name,
            layer_idx=multi_token_layer_idx,
            hook_point_name=multi_token_hook_point, # e.g. "blocks.{L}.attn.hook_z" to get per-head activations
            # pooling_strategy_per_token might be relevant if activations are [batch, seq, heads, d_head]
            # or [batch, seq, d_model] - check function for details on aggregation.
            device=model.device,
            results_base_dir=f"{results_base_dir}/multi_token_probes/{model.cfg.model_name}",
            cache_dir_base=cache_dir_base,
            max_seq_len=max_seq_len_multi
        )
        if baseline_res: all_baseline_concat_results.append(baseline_res)
    except AttributeError:
        print("run_baseline_concat_probing_generic not found, skipping.")
    except Exception as e:
        print(f"Error in run_baseline_concat_probing_generic: {e}")


    # SAE aggregated probing (uses the provided SAE)
    try:
        sae_agg_res = run_multi_token_acts.run_sae_aggregated_probing_generic(
            model=model,
            sae=sae, # SAE used to get features from multi-token activations
            dataset_name=dataset_name,
            layer_idx=multi_token_layer_idx, # Layer for model activations (should match SAE's layer)
            model_hook_point_name=sae.cfg.hook_point, # Model hook point to get activations for SAE
            # sae_hook_point_name is sae.cfg.hook_point implicitly by passing the sae object
            aggregation_method="mean", # "mean" or "max" of SAE features over token dimension
            k_sae_features=k_features,
            binarize=False,
            device=model.device,
            results_base_dir=f"{results_base_dir}/multi_token_probes/{model.cfg.model_name}",
            cache_dir_base=cache_dir_base,
            max_seq_len=max_seq_len_multi
        )
        if sae_agg_res: all_sae_aggregated_results.append(sae_agg_res)
    except Exception as e:
        print(f"Error in run_sae_aggregated_probing_generic: {e}")

    # Attention probing on model activations (assumes a _generic version)
    # This probes attention head outputs directly from the model.
    try:
        attn_res = run_multi_token_acts.train_attn_probing_on_model_acts_generic(
            model=model,
            dataset_name=dataset_name,
            layer_idx=multi_token_layer_idx,
            # hook_point_for_attn_acts would be something like "blocks.{L}.attn.hook_z"
            # This function needs to specify which attention heads or how to aggregate them.
            # It might iterate through heads or take specific head indices.
            attn_hook_point_name=f"blocks.{multi_token_layer_idx}.attn.hook_z", # Example
            device=model.device,
            results_base_dir=f"{results_base_dir}/multi_token_probes/{model.cfg.model_name}",
            cache_dir_base=cache_dir_base,
            max_seq_len=max_seq_len_multi
        )
        if attn_res: all_attn_probing_results.append(attn_res)
    except AttributeError:
        print("train_attn_probing_on_model_acts_generic not found, skipping.")
    except Exception as e:
        print(f"Error in train_attn_probing_on_model_acts_generic: {e}")


# After collecting results, you might save them as pickle files, similar to original scripts.
# import pickle
# Path(f"results/multi_token_probes_{model.cfg.model_name}").mkdir(parents=True, exist_ok=True)
# if all_baseline_concat_results:
#    with open(f"results/multi_token_probes_{model.cfg.model_name}/baseline_concat_results_generic.pkl", "wb") as f:
#        pickle.dump(all_baseline_concat_results, f)
# if all_sae_aggregated_results:
#    with open(f"results/multi_token_probes_{model.cfg.model_name}/sae_aggregated_results_generic_k{k_features}.pkl", "wb") as f:
#        pickle.dump(all_sae_aggregated_results, f)
# if all_attn_probing_results:
#    with open(f"results/multi_token_probes_{model.cfg.model_name}/attn_probing_results_generic.pkl", "wb") as f:
#        pickle.dump(all_attn_probing_results, f)

```

Note: The `run_multi_token_acts.py` functions require careful setup of parameters related to which activations to use (e.g., specific attention heads, residual stream positions for SAEs) and how they are aggregated across tokens. The `_generic` versions should clearly document these or accept them as arguments. The examples above make some assumptions; you'll need to adapt them to the exact signatures and capabilities of the refactored functions.

**4. Combining Results (`sae_probes.combine_results`):**

After individual experiment runs (especially for SAE probes which save many small files), you can aggregate them. The `process_setting` function is used for this. The `model_name` parameter should match `model.cfg.model_name` used when running the experiments, as this often forms part of the directory structure where results are saved.

```python
from sae_probes import combine_results

# Example for 'normal' setting and your model
# Assumes results were saved in directories like:
# results/sae_probes/<model_name_str>/normal_setting/ (for SAE probes)
# or results/baseline_probes/<model_name_str>/normal/ (for baselines)
# The 'process_setting' function needs to know where to look.
# It might internally construct paths based on model_name and setting.
# You may need to adjust 'input_base_dir' or ensure 'process_setting' correctly finds your results.

# For SAE Probe results:
combine_results.process_setting(
    setting="normal", # The setting subdir, e.g. "normal_experiments" or "normal_setting"
    model_name=model.cfg.model_name, # Model name used in result paths
    # input_dir_leaf might be needed if paths are like 'results/sae_probes/your_model_name/normal_experiments_generic/'
    # Check combine_results.py for how paths are constructed.
    # Example if results are in "results/sae_probes/your_model_name/normal_experiments_generic/"
    # You might need to pass parts of this path or ensure default matches.
    # A common structure is f"{results_base_dir}/sae_probes/{model.cfg.model_name}/{setting_type}_experiments_generic/"
    # process_setting would need to be compatible with this.
)

# Example for 'scarcity' setting for SAE Probes
# combine_results.process_setting(setting="scarcity", model_name=model.cfg.model_name)

# For Baseline Probe results, they are often coalesced by the baseline run scripts themselves
# (e.g., coalesce_all_baseline_normal_generic). If combine_results is also used for them,
# ensure the paths and settings match.
```

The `process_setting` function looks for `.pkl` files in structured directories and creates a combined CSV. Ensure the `results_base_dir` used in your experiment calls and the paths expected by `combine_results.py` are consistent.

## Original Code for Comparison
