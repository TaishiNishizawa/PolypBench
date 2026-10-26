"""Export a trained model (best.pth) to ONNX, keeping only its final prediction.

The exported graph has ONE output: the final head's logits (configs/models/<id>.yaml,
train.final_output). Exporting every deep-supervision head lets inference
code pick the wrong one by accident.

The export is checked against PyTorch with onnxruntime on real validation
frames before it is written. Runs in the `train` env.

Writes <artifacts>/checkpoints/<model_id>/model.onnx (+ export.json), which
is what configs/models/<id>.yaml points predict.py at.

Usage:
    python scripts/export_onnx.py --model_id emcad-b2@sunseg
"""

import argparse
import json
from datetime import datetime

import numpy as np
import onnxruntime as ort
import torch
import yaml

from polypbench.models import load_model_config
from polypbench.paths import ARTIFACTS_ROOT, CONFIG_DIR
from polypbench.train.data import SegmentationFrames, build_transform, train_val_pairs
from polypbench.train.models import FinalOutput, build_model

OPSET = 17
MAX_ABS_DIFF = 1e-3   # on logits


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--recipe", help="for the input size/normalization and val_split of the check frames "
                                         "(default: the model config's train.recipe, else sunseg)")
    parser.add_argument("--checkpoint", default="best.pth")
    parser.add_argument("--smoke", action="store_true", help="export the train.py --smoke checkpoint instead")
    args = parser.parse_args()

    model_config = load_model_config(args.model_id)
    with open(CONFIG_DIR / "train" / f"{args.recipe or model_config['train'].get('recipe', 'sunseg')}.yaml") as f:
        recipe = yaml.safe_load(f)
    out_dir = ARTIFACTS_ROOT / "checkpoints" / ("_smoke" if args.smoke else "") / args.model_id
    ckpt = out_dir / args.checkpoint

    model = build_model(model_config)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    wrapped = FinalOutput(model, model_config["train"]["final_output"]).eval()

    size = model_config["input_size"]
    dummy = torch.randn(1, model_config["input_channels"], size, size)
    onnx_path = out_dir / "model.onnx"
    tmp_path = out_dir / "model.onnx.tmp"
    torch.onnx.export(wrapped, dummy, str(tmp_path), opset_version=OPSET, dynamo=False,
                      input_names=["input"], output_names=["output"],
                      dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}})

    # Check against PyTorch on real validation frames.
    _, val_pairs = train_val_pairs(recipe["val_split"])
    frames = SegmentationFrames(val_pairs[::200], build_transform(recipe, train=False))
    session = ort.InferenceSession(str(tmp_path), providers=["CPUExecutionProvider"])
    worst = 0.0
    with torch.no_grad():
        for i in range(len(frames)):
            x = frames[i][0].unsqueeze(0)
            ref = wrapped(x).numpy()
            got = session.run(None, {"input": x.numpy()})[0]
            if got.shape != ref.shape:
                raise RuntimeError(f"ONNX output shape {got.shape} != PyTorch {ref.shape}")
            worst = max(worst, float(np.abs(got - ref).max()))
    if worst > MAX_ABS_DIFF:
        tmp_path.unlink()
        raise RuntimeError(f"ONNX differs from PyTorch by up to {worst:.2e} (> {MAX_ABS_DIFF}); not written")
    tmp_path.replace(onnx_path)

    with open(out_dir / "export.json", "w") as f:
        json.dump({"checkpoint": args.checkpoint, "opset": OPSET, "check_frames": len(frames),
                   "max_abs_logit_diff_vs_torch": worst, "torch": torch.__version__,
                   "exported": datetime.now().isoformat(timespec="seconds")}, f, indent=2)
    print(f"Wrote {onnx_path} (1 output; max |ONNX - PyTorch| on {len(frames)} val frames: {worst:.2e})")


if __name__ == "__main__":
    main()
