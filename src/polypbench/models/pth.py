"""Run a model trained by scripts/train.py straight from its PyTorch checkpoint
(checkpoints/<model_id>/best.pth), without exporting to ONNX first.

Needs the `train` env (envs/train.yml): PyTorch plus the model's third_party
code, built by the same build_model() as training. Preprocessing and the final
output are exactly those of the ONNX predictor (polypbench.models.preprocess,
FinalOutput), so the two backends give the same probability maps up to float
rounding.
"""

import numpy as np
import torch

from polypbench.models.preprocess import preprocess, to_probability
from polypbench.train.models import FinalOutput, build_model


class TorchPredictor:
    """One PyTorch model. predict(rgb_frame) -> probability map at input_size."""

    def __init__(self, config, weights_path, threads=None):
        if not weights_path.exists():
            raise FileNotFoundError(f"{config['model_id']}: weights not found at {weights_path}")
        self.model_id = config["model_id"]
        self.input_size = config["input_size"]
        self.input_channels = config["input_channels"]
        self.mean = np.array(config["mean"], dtype=np.float32)
        self.std = np.array(config["std"], dtype=np.float32)
        if threads:
            torch.set_num_threads(threads)

        # Full fp32. PyTorch's default TF32 convolutions on Ampere+ GPUs move probabilities by up
        # to ~1e-2 vs fp32 (measured on EMCAD/CASCADE); with TF32 off the GPU matches CPU to ~1e-4.
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cpu":
            print(f"[WARN] {self.model_id} is running on CPU (torch sees no GPU)")
        self.providers = [f"torch-{self.device.type}"]   # recorded in run.json, like onnxruntime's providers

        model = build_model(config, pretrained=False)
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        self.model = FinalOutput(model, config["train"]["final_output"]).to(self.device).eval()

    @torch.no_grad()
    def predict(self, rgb):
        """Return the foreground probability map (float32 SxS, values in [0, 1])."""
        x = torch.from_numpy(preprocess(rgb, self.input_size, self.input_channels, self.mean, self.std))
        logits = self.model(x.to(self.device)).float().cpu().numpy()
        return to_probability(self.model_id, logits)
