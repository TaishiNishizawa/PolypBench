"""Model loading. Each model is described by configs/models/<model_id>.yaml.

Seed runs share one config: '<base>-s<N>' (e.g. emcad-b2@sunseg-ft-s1) is
<base>.yaml trained with seed N, with its own checkpoint folder
checkpoints/<base>-s<N>/, so predictions and metrics keep one model_id per run.
"""

import re

import yaml

from polypbench.paths import ARTIFACTS_ROOT, CONFIG_DIR

MODELS_CONFIG_DIR = CONFIG_DIR / "models"
_SEEDED = re.compile(r"^(.+)-s(\d+)$")


def load_model_config(model_id):
    path = MODELS_CONFIG_DIR / f"{model_id}.yaml"
    base, seed = model_id, None
    m = _SEEDED.match(model_id)
    if not path.exists() and m and (MODELS_CONFIG_DIR / f"{m.group(1)}.yaml").exists():
        base, seed = m.group(1), int(m.group(2))
        path = MODELS_CONFIG_DIR / f"{base}.yaml"
    if not path.exists():
        available = sorted(p.stem for p in MODELS_CONFIG_DIR.glob("*.yaml"))
        raise FileNotFoundError(f"No model config for '{model_id}'. Available: {available}")
    with open(path) as f:
        config = yaml.safe_load(f)
    if seed is not None:
        config["model_id"] = model_id
        config["seed"] = seed
        config["weights"] = config["weights"].replace(f"checkpoints/{base}/", f"checkpoints/{model_id}/")
    return config


def load_groups():
    with open(CONFIG_DIR / "groups.yaml") as f:
        return yaml.safe_load(f)


def resolve_models(names):
    """Expand a --models argument: 'all', group names (configs/groups.yaml) and model ids, in order."""
    if names == ["all"]:
        return sorted(p.stem for p in MODELS_CONFIG_DIR.glob("*.yaml"))
    groups = load_groups()["groups"]
    model_ids = []
    for name in names:
        for model_id in groups.get(name, [name]):
            if model_id not in model_ids:
                model_ids.append(model_id)
    return model_ids


BACKENDS = ("pth", "onnx")


def weights_path(config, backend):
    """The weights file a backend loads: "onnx" = the config's `weights` (the exported model.onnx);
    "pth" = checkpoints/<model_id>/best.pth written by scripts/train.py. None if the model has no
    such weights (models we didn't train have no best.pth or no `train:` section)."""
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got '{backend}'")
    if backend == "onnx":
        return ARTIFACTS_ROOT / config["weights"]
    if "train" not in config:
        return None
    return ARTIFACTS_ROOT / "checkpoints" / config["model_id"] / "best.pth"


def load_model(model_id, threads=None, backend="onnx"):
    """Build a predictor from its config: .predict(rgb_frame) -> probability map.
    backend "pth" runs the PyTorch checkpoint (needs the `train` env), "onnx" the exported model."""
    config = load_model_config(model_id)
    if config["backend"] == "onnx" and backend == "pth":
        path = weights_path(config, "pth")
        if path is None:
            raise ValueError(f"{model_id}: no `train:` section, so no PyTorch checkpoint; use backend onnx")
        try:
            from polypbench.models.pth import TorchPredictor
        except ImportError as e:
            raise ImportError(f"backend pth needs PyTorch and the model code: run in the `train` env ({e})") from e
        return TorchPredictor(config, path, threads=threads)
    if config["backend"] == "onnx":
        from polypbench.models.onnx import OnnxPredictor
        return OnnxPredictor(
            model_id=config["model_id"],
            weights_path=weights_path(config, "onnx"),
            input_size=config["input_size"],
            input_channels=config["input_channels"],
            mean=config["mean"],
            std=config["std"],
            output_index=config["output_index"],
            threads=threads,
        )
    raise ValueError(f"{model_id}: unknown backend '{config['backend']}'")
