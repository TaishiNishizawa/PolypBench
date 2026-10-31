"""Frame preprocessing and output activation shared by the ONNX and PyTorch predictors,
so both backends see exactly the same network input.

Preprocessing must match how each model was trained: resize to input_size,
scale to [0, 1], normalize with the config's mean/std.
"""

import cv2
import numpy as np

from polypbench.io.predictions import sigmoid


def preprocess(rgb, input_size, input_channels, mean, std):
    """uint8 HxWx3 RGB frame -> float32 1xCxSxS network input."""
    size = (input_size, input_size)
    if input_channels == 1:
        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)[..., None]
    else:
        img = cv2.resize(rgb, size, interpolation=cv2.INTER_LINEAR)
    img = (img.astype(np.float32) / 255.0 - mean) / std
    return np.ascontiguousarray(img.transpose(2, 0, 1)[None], dtype=np.float32)


def to_logits(model_id, output):
    """Network output (1x1xSxS or similar) -> float32 SxS logit map: what the sigmoid is applied to."""
    logits = np.squeeze(output).astype(np.float32)
    if logits.ndim != 2:
        raise ValueError(f"{model_id}: expected a single-channel output, got shape {output.shape}")
    return logits


def to_probability(model_id, output):
    """Network output -> float32 SxS probability map."""
    return sigmoid(to_logits(model_id, output))
