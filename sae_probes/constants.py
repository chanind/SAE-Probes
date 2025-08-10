from pathlib import Path
from typing import Literal

DATA_PATH = Path(__file__).parent / "data"
DEFAULT_SAE_CACHE_PATH = Path("sae_cache")
DEFAULT_RESULTS_PATH = Path("results")

RegType = Literal["l1", "l2"]
Setting = Literal["normal", "scarcity", "imbalance"]
