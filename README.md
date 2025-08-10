# Are Sparse Autoencoders Useful? A Case Study in Sparse Probing

This repository conains the code for the paper [_Are Sparse Autoencoders Useful? A Case Study in Sparse Probing_](https://arxiv.org/pdf/2502.16681), but has been reformatted into a Python package that will work with any residual stream SAE that can be loaded in [SAELens](https://github.com/jbloomAus/SAELens). This makes it easy to use the sparse probing tasks from the paper as a standalone SAE benchmark.

# Installation

```
pip install git+https://github.com/chanind/SAE-Probes.git@package2
```

## Running evaluations

You can run benchmarks directly; any missing model activations are generated on demand. If you don't pass a `model_cache_path`, a temporary directory is used and cleaned up when the function completes. To persist activations across runs (recommended for repeated experiments), provide a `model_cache_path`.

### Optional: Pre-generating model activations

Pre-generating can speed up repeated runs and lets you inspect the saved tensors. It's optional because benchmarks will auto-generate missing activations.

```python
from sae_probes import generate_dataset_activations

generate_dataset_activations(
  model_name="gemma-2-2b", # the TransformerLens name of the model
  hook_names=["blocks.12.hook_resid_post"], # Any TLens hook names
  batch_size=64,
  device="cuda",
  model_cache_path="/path/to/save/activations",
)
```

If you skip pre-generation, the benchmarks will create any missing activations automatically. Passing a `model_cache_path` persists them; if omitted, activations will be written to a temporary directory that is deleted after the run.

## Training Probes

Probes can be trained directly on the model activations (baselines) or on SAE activations. In both cases, the following test data-balance settings are available: `"normal"`, `"scarcity"`, and `"imbalance"`. For more details about these settings, see the original paper. For the most standard sparse-probing benchmark, use the `normal` setting.

### SAE Probes

The most standard use of this library is as a sparse probing benchmark for SAEs using the `normal` setting. This is demonstrated below:

```python
from sae_probes import run_sae_evals
from sae_lens import SAE

# run the benchmark on a Gemma Scope SAE
release = "gemma-scope-2b-pt-res-canonical"
sae_id = "layer_12/width_16k/canonical"
sae = SAE.from_pretrained(release, sae_id)[0]

run_sae_evals(
  sae=sae,
  model_name="gemma-2-2b",
  hook_name="blocks.12.hook_resid_post",
  reg_type="l1",
  setting="normal",
  sae_cache_path="/results/output/path",
  # model_cache_path is optional; if omitted, a temp dir is used and cleaned
  model_cache_path="/path/to/saved/activations",
  ks=[1, 16],
)
```

The sparse probing results for each dataset will be saved to `sae_cache_path` as a JSON file per dataset.

### Baseline Probes

The baseline probes can be run using the functions `run_all_baseline_normal`, `run_all_baseline_scarcity`, `run_all_baseline_corrupt`, and `run_all_baseline_class_imbalance`. These functions will run the baseline probes for all datasets and methods, and save the results to the `results_path` directory. Using the `run_all_baseline_normal` function is demonstrated below:

```python
from sae_probes import run_all_baseline_normal

run_all_baseline_normal(
  model_name="gemma-2-2b",
  hook_name="blocks.12.hook_resid_post",
  results_path="/results/output/path",
  # model_cache_path is optional; if omitted, a temp dir is used and cleaned
  model_cache_path="/path/to/saved/activations",
)
```

The baseline probes will be saved to `results_path` as a CSV file per dataset.
