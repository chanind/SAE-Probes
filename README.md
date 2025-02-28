# SAE Probes

A Python package for evaluating sparse autoencoders through probing tasks.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/sae-probes.git
cd sae-probes

# Install with Poetry
poetry install
```

## Usage

### Basic Usage

```python
from pathlib import Path
from sae_lens import SAE
from sae_probes import run_sae_probe

# Load your SAE
sae, _, _ = SAE.from_pretrained(
    release="gemma-scope-9b-pt-res",
    sae_id="layer_20/width_16k/l0_408",
    device="cuda:0"
)

# Run probing on all datasets
results = run_sae_probe(
    sae=sae,
    model_name="gemma-2-9b",
    layer=20,
    dataset_path=Path("data/cleaned_data"),
    cache_path=Path("cache/activations"),
    settings=["normal"],
    k_values=[16, 32, 64, 128],
    reg_type="l1",
)

# Access results
for dataset, metrics in results["summaries"]["normal"].iterrows():
    dataset_name, k = dataset
    print(f"Dataset: {dataset_name}, k={k}")
    print(f"  AUC: {metrics['auc_mean']:.4f} ± {metrics['auc_std']:.4f}")
```

### Advanced Usage

```python
import torch
from pathlib import Path
from sae_lens import SAE
from sae_probes.activations import ActivationConfig, generate_model_activations, generate_sae_activations
from sae_probes.probing import ProbeConfig, train_probe
from sae_probes.datasets import load_dataset

# Configuration
model_name = "gemma-2-9b"
layer = 20
dataset_tag = "100_news_fake"
dataset_path = Path("data/cleaned_data")
cache_path = Path("cache")

# Create activation config
activation_config = ActivationConfig(
    model_name=model_name,
    layer=layer,
    device="cuda:0",
)

# Load SAE
sae, _, _ = SAE.from_pretrained(
    release="gemma-scope-9b-pt-res",
    sae_id="layer_20/width_16k/l0_408",
    device="cuda:0",
)

# Generate or load model activations
model_activations = generate_model_activations(
    dataset_tag=dataset_tag,
    config=activation_config,
    dataset_path=dataset_path,
    cache_path=cache_path,
)

# Generate or load SAE activations
sae_activations = generate_sae_activations(
    dataset_tag=dataset_tag,
    sae=sae,
    model_activations=model_activations,
    config=activation_config,
    cache_path=cache_path,
)

# Load dataset and create splits
df, train_indices, test_indices = load_dataset(
    dataset_tag=dataset_tag,
    dataset_path=dataset_path,
)

# Prepare data
X_train = sae_activations[train_indices]
X_test = sae_activations[test_indices]
y_train = df["target"].values[train_indices]
y_test = df["target"].values[test_indices]

# Create probe config
probe_config = ProbeConfig(
    reg_type="l1",
    k_values=[16, 128, 512],
    binarize=False,
)

# Train and evaluate probes
results = train_probe(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    config=probe_config,
)

# Print results
for result in results:
    print(f"k={result.k}, AUC={result.auc:.4f}")
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
