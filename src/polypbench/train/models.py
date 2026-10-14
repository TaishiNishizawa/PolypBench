"""Build trainable PyTorch models from the `train:` section of configs/models/<model_id>.yaml."""

import importlib
import sys

import torch
from torch import nn

from polypbench.paths import ARTIFACTS_ROOT, REPO_ROOT


def build_model(model_config):
    """Instantiate the architecture from its third_party code, with the pretrained backbone loaded."""
    spec = model_config["train"]
    code_dir = REPO_ROOT / spec["code"]
    if not code_dir.is_dir():
        raise FileNotFoundError(f"{code_dir} not found (git submodule update --init?)")
    sys.path.insert(0, str(code_dir))
    module = importlib.import_module(spec["module"])
    cls = getattr(module, spec["class"])

    kwargs = dict(spec["kwargs"])
    if "encoder" in kwargs:          # EMCAD / SACNet: we load the backbone ourselves, with a check
        kwargs["pretrain"] = False
    model = cls(**kwargs)

    if spec.get("pretrained_backbone"):
        load_backbone(model.backbone, ARTIFACTS_ROOT / spec["pretrained_backbone"])
    return model


def load_backbone(backbone, path):
    """Load ImageNet weights into model.backbone. The original code silently keeps
    only matching keys; here a backbone left (partly) uninitialized is an error."""
    state = torch.load(path, map_location="cpu")
    own = backbone.state_dict()
    matched = {k: v for k, v in state.items() if k in own and v.shape == own[k].shape}
    missing = [k for k in own if k not in matched]
    if missing:
        raise RuntimeError(f"{path.name}: {len(missing)} of {len(own)} backbone tensors not found, e.g. {missing[:3]}")
    backbone.load_state_dict(matched)
    print(f"Loaded pretrained backbone {path.name}: {len(matched)}/{len(own)} tensors "
          f"({len(state) - len(matched)} checkpoint tensors unused, e.g. the classification head)")


def as_list(outputs):
    return outputs if isinstance(outputs, (list, tuple)) else [outputs]


class FinalOutput(nn.Module):
    """Wraps a model so it returns only its final prediction (logits) - used for ONNX export."""

    def __init__(self, model, final_output):
        super().__init__()
        self.model = model
        self.final_output = final_output

    def forward(self, x):
        return as_list(self.model(x))[self.final_output]
