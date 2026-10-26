"""Run an exported .onnx segmentation model on single RGB frames.

Preprocessing (polypbench.models.preprocess) is shared with the PyTorch
predictor (polypbench.models.pth).
"""

import numpy as np
import onnxruntime as ort

from polypbench.models.preprocess import preprocess, to_probability


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

    def predict(self, rgb):
        """Return the foreground probability map (float32 SxS, values in [0, 1])."""
        x = preprocess(rgb, self.input_size, self.input_channels, self.mean, self.std)
        outputs = self.session.run(None, {self.input_name: x})
        return to_probability(self.model_id, outputs[self.output_index])
