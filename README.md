# Are Sparse Autoencoders Useful? A Case Study in Sparse Probing

This repository conains the code for the paper [_Are Sparse Autoencoders Useful? A Case Study in Sparse Probing_](https://arxiv.org/pdf/2502.16681), but has been reformatted into a Python package that will work with any residual stream SAE that can be loaded in [SAELens](https://github.com/jbloomAus/SAELens). This makes it easy to use the sparse probing tasks from the paper as a standalone SAE benchmark.

# Installation

```
pip install git+https://github.com/chanind/SAE-Probes.git@package2
```

# Running evaluations

The process of running evaluations is split into two parts: generating and saving LLM model activations, and then running sparse probing on those activations.

## Generating Model Activations

The main method for generating model activations is `generate_dataset_activations`, demonstrated below:

```python
from sae_probes import generate_dataset_activations

generate_dataset_activations(
   model_name="gemma-2-2b", # the TransformerLens name of the model
   layers=[12], # Layers to extract activations from (will use hook_resid_post)
   batch_size=64,
   device="cuda",
   model_cache_path="/path/to/save/activations",
)
```

This must be run before any probing evals can be run, as these activations are used both for SAE evals and baseline evals. Importantly, the `model_cache_path` must be the same when train probes.

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
   layer=12,
   reg_type="l1",
   setting="normal",
   sae_cache_path="/results/output/path,
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
   layer=12,
   results_path="/results/output/path",
   model_cache_path="/path/to/saved/activations",
)
```

The baseline probes will be saved to `results_path` as a CSV file per dataset.
