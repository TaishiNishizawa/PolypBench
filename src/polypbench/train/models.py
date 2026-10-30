"""Build trainable PyTorch models from the `train:` section of configs/models/<model_id>.yaml."""

import hashlib
import importlib
import sys

import torch
from torch import nn

from polypbench.paths import ARTIFACTS_ROOT, REPO_ROOT


def import_repo_module(code_dir, module):
    """Import `module` from a third_party repo folder. Several repos name their package `lib`
    (EMCAD, CASCADE, SALI, CMSA-Net, ...): a cached one from another repo is dropped first, or a
    second model in the same process would import the first one's code."""
    if not code_dir.is_dir():
        raise FileNotFoundError(f"{code_dir} not found (git submodule update --init?)")
    top = module.split(".")[0]
    for name in [n for n in sys.modules if n == top or n.startswith(top + ".")]:
        file = getattr(sys.modules[name], "__file__", None) or ""
        if not file.startswith(str(code_dir)):
            del sys.modules[name]
    if str(code_dir) in sys.path:
        sys.path.remove(str(code_dir))
    sys.path.insert(0, str(code_dir))
    return importlib.import_module(module)


def build_model(model_config, pretrained=True):
    """Instantiate the architecture from its third_party code, with the pretrained backbone loaded
    (pretrained=False skips it, e.g. when trained weights are loaded right after)."""
    spec = model_config["train"]
    module = import_repo_module(REPO_ROOT / spec["code"], spec["module"])
    cls = getattr(module, spec["class"])

    kwargs = dict(spec["kwargs"])
    if "encoder" in kwargs:          # EMCAD / SACNet: we load the backbone ourselves, with a check
        kwargs["pretrain"] = False
    model = cls(**kwargs)

    if pretrained and spec.get("pretrained_backbone"):
        path = ARTIFACTS_ROOT / spec["pretrained_backbone"]
        if "pretrained_sha256" in spec and sha256(path) != spec["pretrained_sha256"]:
            raise RuntimeError(f"{path}: sha256 does not match pretrained_sha256 in the model config")
        backbone = getattr(model, spec.get("backbone_attr", "backbone"))   # PraNet: model.resnet
        load_backbone(backbone, path)
    return model


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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


def final_prediction(outputs, final_output):
    """The model's prediction (logits): one output by index, or "sum" of all outputs (CASCADE's additive aggregation)."""
    outputs = as_list(outputs)
    return sum(outputs) if final_output == "sum" else outputs[final_output]


class FinalOutput(nn.Module):
    """Wraps a model so it returns only its final prediction (logits) - used for ONNX export."""

    def __init__(self, model, final_output):
        super().__init__()
        self.model = model
        self.final_output = final_output

    def forward(self, x):
        return final_prediction(self.model(x), self.final_output)
