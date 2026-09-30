"""Post-processing applied to per-frame probability maps, configured in configs/postproc/."""

import yaml

from polypbench.paths import CONFIG_DIR

POSTPROC_CONFIG_DIR = CONFIG_DIR / "postproc"


def load_postproc_config(name):
    path = POSTPROC_CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        available = sorted(p.stem for p in POSTPROC_CONFIG_DIR.glob("*.yaml"))
        raise FileNotFoundError(f"No postproc config for '{name}'. Available: {available}")
    with open(path) as f:
        return yaml.safe_load(f)


def load_postproc(name):
    """Return a stateful callable (prob, rgb) -> prob with .reset(), or None for 'none'."""
    params = dict(load_postproc_config(name))
    params.pop("name")
    if name == "none":
        return None
    if name == "temporal_smooth":
        from polypbench.postproc.temporal import TemporalSmoother
        return TemporalSmoother(**params)
    raise ValueError(f"Unknown postproc '{name}'")
