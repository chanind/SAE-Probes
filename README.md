# SAE Probes

A Python package for evaluating sparse autoencoders and baseline neural probes on probing tasks, from [_Are Sparse Autoencoders Useful? A Case Study in Sparse Probing_](https://arxiv.org/pdf/2502.16681).

## Installation

```bash
pip install git+https://github.com/chanind/SAE-Probes.git@packify
```

## Usage

### SAE Probing

```python
from pathlib import Path
from sae_lens import SAE
from sae_probes import RunSaeProbeConfig, run_sae_probe

# Load your SAE
sae, _, _ = SAE.from_pretrained(
    release="gemma-scope-9b-pt-res",
    sae_id="layer_20/width_16k/l0_408",
    device="cuda"
)

# Create config
config = RunSaeProbeConfig(
    model_name="gemma-2-9b",
    hook_name="blocks.20.hook_resid_post",
    k_values=[16, 32, 64, 128],
    device="cuda"
)

# Run probing on all datasets
results = run_sae_probe(
    sae=sae,
    config=config,
    dataset_path=Path("data/cleaned_data"),
    cache_path=Path("cache/activations"),
)

# Access results
for dataset, metrics in results["summaries"]["normal"].iterrows():
    dataset_name, k = dataset
    print(f"Dataset: {dataset_name}, k={k}")
    print(f"  AUC: {metrics['auc_mean']:.4f} ± {metrics['auc_std']:.4f}")
```

### Baseline Neural Probing

You can also run standard neural probes (without SAEs) for comparison:

```python
from pathlib import Path
from sae_probes import RunBaselineProbeConfig, run_baseline_probes

# Create config for baseline probes
config = RunBaselineProbeConfig(
    model_name="gemma-2-9b",
    layer=20,  # This will be converted to the appropriate hook name
    reg_type="l2",  # L2 regularization is typically better for baseline probes
    num_train=1024,
    device="cuda"
)

# Run baseline probes
run_baseline_probes(
    config=config,
    dataset_path=Path("data/cleaned_data"),
    cache_path=Path("cache/activations"),
    results_path=Path("results"),
)
```

## Development

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/sae-probes.git
cd sae-probes

# Install development dependencies
poetry install --with dev
```

### Testing

```bash
# Run tests
poetry run pytest

# Run tests with coverage
poetry run pytest --cov=sae_probes
```

### Linting

```bash
# Run ruff
poetry run ruff check .

# Run pyright
poetry run pyright sae_probes
```
