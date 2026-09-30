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


def load_model(model_id):
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
        )
    raise ValueError(f"{model_id}: unknown backend '{config['backend']}'")
