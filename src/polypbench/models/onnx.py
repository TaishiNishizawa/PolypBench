"""Run an exported .onnx segmentation model on single RGB frames.

Preprocessing must match how each model was trained: resize to
input_size, scale to [0, 1], normalize with the config's mean/std.
"""

import cv2
import numpy as np
import onnxruntime as ort


class OnnxPredictor:
    """One ONNX model. predict(rgb_frame) -> probability map at input_size."""

    def __init__(self, model_id, weights_path, input_size, input_channels, mean, std, output_index, threads=None):
        if not weights_path.exists():
            raise FileNotFoundError(f"{model_id}: weights not found at {weights_path}")
        self.model_id = model_id
        self.input_size = input_size
        self.input_channels = input_channels
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.output_index = output_index

        options = ort.SessionOptions()
        if threads:   # default is one thread per CPU core, per process
            options.intra_op_num_threads = threads
            options.inter_op_num_threads = threads
        self.session = ort.InferenceSession(
            str(weights_path), sess_options=options, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        # get_available_providers() only reports what onnxruntime was compiled
        # with; check what this session actually loaded, since CUDA can fail
        # to load at runtime and silently fall back to CPU.
        self.providers = self.session.get_providers()
        if "CUDAExecutionProvider" not in self.providers:
            print(f"[WARN] {model_id} is running on CPU (active providers: {self.providers})")
        self.input_name = self.session.get_inputs()[0].name

    def preprocess(self, rgb):
        """uint8 HxWx3 RGB frame -> float32 1xCxSxS network input."""
        size = (self.input_size, self.input_size)
        if self.input_channels == 1:
            img = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)[..., None]
        else:
            img = cv2.resize(rgb, size, interpolation=cv2.INTER_LINEAR)
        img = (img.astype(np.float32) / 255.0 - self.mean) / self.std
        return np.ascontiguousarray(img.transpose(2, 0, 1)[None], dtype=np.float32)

    def predict(self, rgb):
        """Return the foreground probability map (float32 SxS, values in [0, 1])."""
        outputs = self.session.run(None, {self.input_name: self.preprocess(rgb)})
        logits = np.squeeze(outputs[self.output_index]).astype(np.float32)
        if logits.ndim != 2:
            raise ValueError(f"{self.model_id}: expected a single-channel output, got shape "
                             f"{outputs[self.output_index].shape}")
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))   # clip: avoid exp overflow
