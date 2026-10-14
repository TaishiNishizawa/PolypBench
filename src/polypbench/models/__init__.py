"""Model loading. Each model is described by configs/models/<model_id>.yaml."""

import yaml

from polypbench.paths import ARTIFACTS_ROOT, CONFIG_DIR

MODELS_CONFIG_DIR = CONFIG_DIR / "models"


def load_model_config(model_id):
    path = MODELS_CONFIG_DIR / f"{model_id}.yaml"
    if not path.exists():
        available = sorted(p.stem for p in MODELS_CONFIG_DIR.glob("*.yaml"))
        raise FileNotFoundError(f"No model config for '{model_id}'. Available: {available}")
    with open(path) as f:
        return yaml.safe_load(f)


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


def load_model(model_id, threads=None):
    """Build a predictor from its config: .predict(rgb_frame) -> probability map."""
    config = load_model_config(model_id)
    if config["backend"] == "onnx":
        from polypbench.models.onnx import OnnxPredictor
        return OnnxPredictor(
            model_id=config["model_id"],
            weights_path=ARTIFACTS_ROOT / config["weights"],
            input_size=config["input_size"],
            input_channels=config["input_channels"],
            mean=config["mean"],
            std=config["std"],
            output_index=config["output_index"],
            threads=threads,
        )
    raise ValueError(f"{model_id}: unknown backend '{config['backend']}'")
