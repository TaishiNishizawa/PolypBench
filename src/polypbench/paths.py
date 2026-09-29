import os
from pathlib import Path
import yaml

# src/polypbench/paths.py -> parents[0] = src/polypbench, [1] = src, [2] = repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"
PATHS_CONFIG = CONFIG_DIR / "paths.yaml"
PATHS_EXAMPLE = CONFIG_DIR / "paths.example.yaml"


def _load_paths_config():
    if not PATHS_CONFIG.exists():
        raise FileNotFoundError(
            f"{PATHS_CONFIG} not found - copy {PATHS_EXAMPLE.name} to "
            f"{PATHS_CONFIG.name} and edit it for this machine."
        )
    with open(PATHS_CONFIG) as f:
        return yaml.safe_load(f) or {}


def _resolve_root(config, key):
    """Env var POLYPBENCH_<KEY> wins over the yaml value; relative -> repo root."""
    value = os.environ.get(f"POLYPBENCH_{key.upper()}") or config.get(key)
    if value is None:
        raise KeyError(f"'{key}' missing from {PATHS_CONFIG} (see {PATHS_EXAMPLE.name})")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


_config = _load_paths_config()
DATA_ROOT = _resolve_root(_config, "data_root")
ARTIFACTS_ROOT = _resolve_root(_config, "artifacts_root")
