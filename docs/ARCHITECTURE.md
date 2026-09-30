# PolypBench architecture

PolypBench benchmarks and trains image and video polyp segmentation models:
our lab's models (EMCAD, MK-UNet, PVT-CASCADE, SACNet) and state-of-the-art
video polyp segmentation models (SALI, PNS+, ...), on an in-house colonoscopy
video set and on SUN-SEG.

This document describes where things go and which decisions the rest of the
codebase depends on. Read it before adding a dataset, model, metric, or
training setup. If a change would break one of the
[invariants](#invariants), discuss it first. Everything else can be changed
freely.

## Repository layout

The project folder holds two siblings: the git repo and a folder of generated
outputs that is kept out of git.

```
PolypBench/
├── polypbench/          # this git repo: code, configs, scripts
└── artifacts/           # not in git: checkpoints, predictions, metrics, profiles, reports
```

Raw data lives outside both of them, under `data_root` (see
[Paths](#paths)).

```
polypbench/
├── environment.yml          # main conda-forge env (the single source of dependencies)
├── pyproject.toml           # package metadata only; dependencies = []
├── envs/                    # extra envs for third-party models (sali.yml, ...) and *.lock.yml exports
├── configs/
│   ├── paths.example.yaml   # template; copy to paths.yaml (gitignored) per machine
│   ├── datasets/            # what the data is
│   ├── models/              # what the model is
│   ├── postproc/            # what happens to raw model output
│   ├── benchmarks/          # what we measure, on what (one per results table)
│   └── train/               # how a model is trained
├── src/polypbench/
│   ├── paths.py             # DATA_ROOT, ARTIFACTS_ROOT, CONFIG_DIR
│   ├── config.py            # load and validate yaml configs
│   ├── data/                # Sequence / Dataset interface + one loader per dataset
│   ├── models/              # Predictor interface, backends (onnx, torch), adapters/
│   ├── postproc/            # temporal smoothing etc.
│   ├── metrics/             # metric registry + implementations
│   ├── profiling/           # latency / FPS / memory / FLOPs
│   ├── io/                  # the only code that reads/writes predictions and results
│   ├── viz/                 # overlays, plots, side-by-side videos
│   └── train/               # training loop, losses, augmentation
├── scripts/                 # thin command-line entry points (logic lives in src/)
├── third_party/             # pinned clones/submodules of external model repos
├── tests/
└── docs/
```

Create a folder when you first need it, not ahead of time. This layout says
where new things go.

## Configs: one question per folder

Each config folder answers exactly one question. An experiment combines them;
none of them refers to another's contents.

| Folder | Question | Contains | Never contains |
|---|---|---|---|
| `datasets/` | What is the data? | root path, sequences, annotated frames, splits | metrics, models, thresholds |
| `models/` | What is the model? | `model_id`, architecture, weights path, input size/channels/normalization, backend, env | dataset choices |
| `postproc/` | What happens after the forward pass? | `none`, `temporal_smooth` + parameters | model-specific hacks |
| `benchmarks/` | What do we measure, on what? | datasets × models × postprocs × metrics, profiling on/off | data facts |
| `train/` | How is a model trained? | train/val data, model, loss, schedule, augmentation | evaluation choices |

A benchmark config is the definition of one results table:

```yaml
name: inhouse
datasets: [inhouse]
models: [emcad@colondb, mkunet@colondb, pvt-cascade@colondb, sacnet-b0@colondb, sacnet-b2@colondb, sali@sunseg]
postprocs: [none, temporal_smooth]
metrics: [dice, iou, interframe_iou]
profile: true
```

## Pipeline

```
configs/benchmarks/X.yaml
   ├─► scripts/predict.py   dataset × model × postproc  ─► artifacts/predictions/…/*.png
   ├─► scripts/evaluate.py  predictions + ground truth  ─► artifacts/metrics/…/*.parquet
   ├─► scripts/profile.py   model × device              ─► artifacts/profiles/…/*.json
   └─► scripts/report.py    joins the above             ─► artifacts/reports/X/

configs/train/Y.yaml
   └─► scripts/train.py ─► artifacts/checkpoints/<model_id>/
                            └─► add configs/models/<model_id>.yaml → usable by any benchmark
```

Stages communicate only through files in `artifacts/`. That is what lets a
third-party model run in its own conda env: all it has to do is write
predictions in the standard format.

## Measurements

| Kind | Examples | Inputs | Computed by |
|---|---|---|---|
| Accuracy | Dice, IoU; SUN-SEG S-α, E-φ, weighted F | predictions + ground truth at annotated frames | `evaluate.py` |
| Temporal stability | interframe IoU, flicker | predictions at every frame (no ground truth needed) | `evaluate.py` |
| Efficiency | latency, FPS, GPU memory, params, FLOPs | model + hardware only | `profile.py` |

Two consequences:

- `predict.py` writes a prediction for **every** frame, not just the annotated
  ones, so any temporal metric can be computed later without re-running
  inference.
- Latency is **not** measured during `predict.py`, where timing would include
  video decoding and PNG writing. `profile.py` times the forward pass alone
  and records results per device, so server (onnxruntime/torch) and Jetson
  (TensorRT) numbers stay separate and comparable.

## Invariants

These are the decisions the rest of the codebase is built on. Changing one
means touching every stage.

1. **Predictions on disk are the contract between stages.** Each one is a
   single-channel `uint8` PNG probability map (0–255) at the original frame
   resolution, one per frame, at
   `artifacts/predictions/<dataset>/<model_id>/<postproc>/<sequence>/<frame:06d>.png`,
   with a `run.json` alongside. Only `io/predictions.py` reads or writes them.
   Probabilities are saved, not binary masks, so threshold-sweep metrics stay
   possible.
2. **`model_id` = `arch-size@trainset[-tag]`**, e.g. `sacnet-b2@colondb`,
   `sacnet-b2@sunseg-v2`, `sali@sunseg`. It is the key that joins model
   configs, checkpoints, predictions, metrics, and profiles. The training set
   is always part of the name.
3. **Data is reached only through `Sequence`.**
   - Frames are read **sequentially**; never seek (`cv2`'s
     `CAP_PROP_POS_FRAMES` can land a few frames off without any error).
   - Frame indices are **0-based**.
   - Images are **RGB** inside the package; convert from OpenCV's BGR as soon
     as a frame is loaded.
   - Annotations are `{frame_index: mask_path}`. Sparse (in-house) and dense
     (SUN-SEG) datasets look the same to callers.
4. **Metrics are functions registered by name** in `metrics/`. A frame metric
   takes `(pred, gt)` at one annotated frame. A sequence metric takes all of a
   sequence's predictions (and ground truth if it needs it). Adding a metric
   means one function plus one name in a benchmark config.
5. **Results are stored in long format.** Columns are
   `dataset, sequence, frame, model_id, postproc, metric, value`, with `frame`
   empty for sequence-level metrics. A new metric adds rows, never columns.

Every generated output directory gets a `run.json` with the resolved config,
git commit, date, and device, so any number in `artifacts/` can be traced back
to what produced it.

## Paths

`configs/paths.yaml` (gitignored, per machine; template in
`paths.example.yaml`) sets `data_root` and `artifacts_root`. Relative paths
resolve against the repo root. The environment variables
`POLYPBENCH_DATA_ROOT` and `POLYPBENCH_ARTIFACTS_ROOT` override them for a
single run. Code imports the resolved values from `polypbench.paths` and never
hardcodes a path. Dataset configs give their `root` relative to `data_root`.

## Environments

- Main env: `conda env create -f environment.yml`, then
  `pip install -e . --no-deps`. All dependencies are listed in
  `environment.yml` (conda-forge first, pip only when conda-forge has no good
  build). `pyproject.toml` lists none, so pip never installs duplicate
  packages such as a second OpenCV.
- Third-party models whose dependencies conflict (old torch, custom CUDA ops)
  get their own `envs/<name>.yml`. They run in that env and write predictions
  in the standard format.
- For results that go into a paper, export the exact environment
  (`conda env export > envs/<name>.lock.yml`) and commit it.

## Datasets

### In-house colonoscopy video (`configs/datasets/inhouse.yaml`)

- Four 15 s clips cut from one 8-minute procedure video: `first_polyp_seen`,
  `first_polyp_removed`, `second_polyp_seen`, `second_polyp_removed`. Each is
  456 frames at 480×360 and 29.97 fps.
- 123 image/mask pairs (`frame1.png` … `frame123.png`), binary masks (0/255).
  Each annotation is an exact video frame, at 0-based frames 0, 15, 30, … of
  its clip.
- Known exceptions:
  - `frame70` belongs to `second_polyp_removed` at frame 105, not to
    `second_polyp_seen`. As a result, `second_polyp_seen` has 30 annotations
    (no annotation at frame 105) and the other three clips have 31 each.
  - `frame53` is `first_polyp_removed` frame 315. The old flow-based matcher
    in LambdaDiceEval never matched it.
- `scripts/build_inhouse_manifest.py` generates the annotation mapping by
  pixel-matching each annotation against the video and fails if any match is
  poor. Regenerate the config with that script; don't edit the annotation
  mapping by hand.
- The primary accuracy metric is computed on the 123 annotated frames against
  their real masks. The old LambdaDiceEval scores used flow-warped
  pseudo-labels on every frame and are not directly comparable. Only its rows
  at exactly the annotated frames are.

### SUN-SEG (`configs/datasets/sunseg.yaml`)

- Training data: `TrainDataset`. Test splits: `TestEasyDataset` and
  `TestHardDataset`, each with `Seen` and `Unseen` parts.
- Frames must be reorganized into the official layout (`Frame/` next to
  `GT/`) with the official SUN-SEG script, so that splits match published
  results exactly.
- Before using a third-party model on in-house data, reproduce its published
  SUN-SEG numbers with its official checkpoint.

## How to add…

- **A dataset:** a `configs/datasets/<name>.yaml` plus a loader in `data/`
  that returns `Sequence`s. Reuse `VideoFileSequence` (mp4) or
  `FrameFolderSequence` (image folder) where possible. Add a test that checks
  annotation–frame alignment.
- **A model:** a `configs/models/<model_id>.yaml`. If an existing backend
  can't run it as is, add an adapter in `models/adapters/`. A third-party repo
  goes in `third_party/`, pinned to a fixed commit, and gets an `envs/` file if
  its dependencies conflict.
- **A metric:** a function in `metrics/`, registered by name, then list it in
  a benchmark config. Rerun `evaluate.py`; no inference needed.
- **A post-processing variant:** code in `postproc/` and a
  `configs/postproc/<name>.yaml`.
- **A trained model:** a `configs/train/<name>.yaml`; `train.py` writes to
  `artifacts/checkpoints/<model_id>/`. Then add the matching model config.

## Related folders (outside this repo)

- `../../LambdaDiceEval/` is the frozen reference implementation of the
  original in-house Dice and stability evaluation. Don't modify it; use it to
  check that ported code reproduces its results.
- `../../Benchmarking/` holds real-time Jetson/TensorRT deployment scripts.
  Its latency results can later be saved to `artifacts/profiles/` in the
  standard format.
- `../../repo_clones/` holds the original model repositories.
