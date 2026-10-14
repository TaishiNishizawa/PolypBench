# PolypBench

Benchmarks image and video polyp segmentation models on colonoscopy video.
It compares our lab's models (EMCAD, MK-UNet, PVT-CASCADE, SACNet) with
state-of-the-art video polyp segmentation models on an in-house colonoscopy
video set, and later on SUN-SEG.

For the design (why things are organized this way, and the rules the code
relies on), see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). This README
covers what exists and how to use it.

## Status

| Stage | Status |
|---|---|
| In-house data loading + tests | done |
| Inference for our lab's 4 models (ONNX) | done, checked against LambdaDiceEval |
| Per-frame / per-clip evaluation | done |
| Overlay videos + comparison sheets | done |
| SALI (SOTA video model) | done: in-house + SUN-SEG paper numbers reproduced |
| More SOTA models | next |
| Temporal stability metrics + plots | done (ported from LambdaDiceEval) |
| SUN-SEG dataset | planned |
| Training | planned |

## Setup

```bash
conda env create -f environment.yml
conda activate polypbench
pip install -e . --no-deps
cp configs/paths.example.yaml configs/paths.yaml   # then edit for this machine
pytest                                             # should show 7 passed
```

`configs/paths.yaml` sets two machine-specific folders:

- `data_root`: raw datasets, e.g. `/home/tn23794/data`
- `artifacts_root`: all generated output, default `../artifacts` (resolved against the repo root)

For a single run, you can override either one with the environment variables
`POLYPBENCH_DATA_ROOT` or `POLYPBENCH_ARTIFACTS_ROOT`.

After updating `environment.yml`, apply the change with
`conda env update -f environment.yml`.

## Everyday commands

Run these from anywhere with the `polypbench` env active. Each step reads the
previous step's saved output, so you only rerun what changed.

`--models` takes model ids, `all`, or a **group** from `configs/groups.yaml`:
`colondb` (our ColonDB-trained models + SALI) or `sunseg` (everything trained on
SUN-SEG: our retrained models + SALI). Tables always include every model.
Side-by-side visuals (comparison sheets, stability comparison graphs) are made
per group, in a folder named after it, so they stay readable as models are
added.

```bash
# 1. Run models over the dataset, saving a probability map for every frame  (~2 min per model)
python scripts/predict.py --dataset inhouse --models all
#    SALI runs in its own env (external model; predict.py skips it):
conda activate sali && python scripts/predict_sali.py --dataset inhouse && conda activate polypbench

# 2. Score the predictions: per-frame CSVs + per-clip summary tables  (seconds)
python scripts/evaluate.py --dataset inhouse --models all

# 3. Draw comparison sheets (PNG, one folder per group) and overlay videos  (~2 min per group)
python scripts/visualize.py --dataset inhouse --models colondb
python scripts/visualize.py --dataset inhouse --models sunseg

# 4. Inter-frame stability from the saved predictions: CSVs + per-clip tables  (~5-10 min)
python scripts/stability.py --dataset inhouse --models all

# 5. Stability graphs + side-by-side videos  (~15 min; --no-videos for graphs only)
python scripts/visualize_stability.py --dataset inhouse --models colondb
python scripts/visualize_stability.py --dataset inhouse --models sunseg

# 6. Same architecture, ColonDB- vs SUN-SEG-trained (pairs in configs/groups.yaml)  (seconds)
python scripts/compare_trainsets.py --dataset inhouse

# 7. Presentation tables (Markdown + CSV) from everything above + SUN-SEG toolbox results  (seconds)
python scripts/make_tables.py                     # -> artifacts/reports/tables/

# Tests
pytest -v
```

### `predict.py`

| Option | Meaning |
|---|---|
| `--models all` or `--models sacnet-b2@colondb emcad@colondb` | which models (ids = file names in `configs/models/`) |
| `--postprocs none temporal_smooth` | post-processing variants to save (default: both) |
| `--overwrite` | redo runs that are already complete (by default they are skipped) |
| `--threads 4` | CPU threads for OpenCV/onnxruntime per run (their default is every core: several runs at once then overload the shared server) |

Each model runs once per clip, and each post-processing variant is applied to
that same output. A run counts as complete once its `run.json` exists.

### `evaluate.py`

| Option | Meaning |
|---|---|
| `--models`, `--postprocs` | as above; runs without complete predictions are skipped |
| `--threshold 0.5` | probability threshold for binarizing predictions (default 0.5) |

It prints the Dice and false-alarm tables and writes all reports (see
[Outputs](#outputs-artifacts)).

### `visualize.py`

| Option | Meaning |
|---|---|
| `--models`, `--postprocs`, `--threshold` | as above |
| `--sequences first_polyp_seen ...` | only these clips (default: all four) |
| `--no-videos` / `--no-sheets` | skip one of the two outputs |
| `--name` | sheet folder for a custom list of models (a group name is used automatically) |

The CPU-heavy scripts (`evaluate`, `visualize`, `stability`,
`visualize_stability`) also take `--threads` (default 8).

### One-off

```bash
# Regenerate configs/datasets/inhouse.yaml (only if the annotations change)
python scripts/build_inhouse_manifest.py

# Build the official SUN-SEG layout from the raw downloads (hard links; safe to rerun)
python scripts/build_sunseg.py
```

### Training our models on SUN-SEG

One shared script trains EMCAD, MK-UNet and SACNet. Their original
`train_polyp.py` scripts use the same recipe (data loader, augmentation,
structure loss, AdamW, cosine schedule, gradient clipping, multi-scale
training), now in `configs/train/sunseg.yaml`. Per-model choices (which outputs
are supervised, pretrained backbone, architecture arguments) are in the
`train:` section of each `configs/models/<model_id>.yaml`. Model code comes
from the pinned submodules in `third_party/`. Runs in the `train` env:

```bash
conda activate train
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --model_id emcad-b2@sunseg      # ~6 h on an A6000
CUDA_VISIBLE_DEVICES=1 python scripts/train.py --model_id mkunet@sunseg        # ~3.5 h
python scripts/train.py --model_id emcad-b2@sunseg --resume                   # continue after a crash
python scripts/export_onnx.py --model_id emcad-b2@sunseg                      # best.pth -> model.onnx
python scripts/train.py --model_id mkunet@sunseg --smoke                      # 20-step end-to-end check
```

- **Validation:** SUN-SEG has no validation split, so 5 whole videos (11
  clips, 10.6% of frames) are held out of TrainDataset
  (`configs/datasets/sunseg.yaml`). The best epoch is chosen on these, never on
  a test split. (SALI's own script picks its checkpoint on
  TestHardDataset/Seen, which flatters its Hard/Seen numbers.)
- **Outputs:** `artifacts/checkpoints/<model_id>/`: `best.pth` (state_dict),
  `last_state.pth` (for `--resume`), `train_log.csv` (loss and validation Dice
  per epoch), `run.json` (recipe, config, git commits incl. the model code,
  device), `train.log`, and after export `model.onnx` + `export.json`.
- **Export** writes a single output, the final head. A multi-output export
  invites using the wrong head (this happened with the old SACNet-b2 ONNX). It
  refuses to write a file whose output differs from PyTorch by more than 1e-3
  on validation frames.
- Once `model.onnx` exists, the model is used like any other:
  `predict.py --models emcad-b2@sunseg`, then `evaluate.py`, etc.

### Training a causal SALI

SALI's released code segments frame i with frame i+1 (its loader comes from
SLT-Net, an offline model), although the paper describes an online model that
conditions on frame i-1. `scripts/train_sali.py` retrains SALI with the
previous frame instead (`sali@sunseg-causal`). It uses SALI's own recipe, not
the shared one (`configs/train/sali.yaml`: the paper's 30 epochs, batch 14,
Adam 1e-4, weight decay 1e-4, plus the code's gradient clamp, augmentation,
loss and `max_skip` schedule). The model code is imported unchanged from
`third_party/SALI`, the encoder starts from ImageNet PVTv2-b5 like our models,
and the best epoch is chosen on the same held-out videos as `train.py`. Runs in
the `sali` env; batch 14 needs ~33 GiB of GPU memory:

```bash
conda activate sali
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 python -u scripts/train_sali.py --model_id sali@sunseg-causal
python scripts/train_sali.py --model_id sali@sunseg-causal --resume          # continue after a crash
python scripts/predict_sali.py --model_id sali@sunseg-causal --dataset inhouse
```

### SUN-SEG benchmark numbers (standard toolbox)

SUN-SEG papers report S-α, mean E-φ, weighted F, mean F, mean sensitivity and
max Dice from the PNS+ evaluation toolbox (copy in `third_party/SALI/eval`).
`eval_sunseg_toolbox.py` runs that toolbox unchanged, but scores clips in
parallel (minutes instead of hours). Run it in the `sali` env:

```bash
# Our models: predict on the four test splits (datasets sunseg_easy_seen, sunseg_hard_seen,
# sunseg_easy_unseen, sunseg_hard_unseen), then score.
for ds in sunseg_easy_seen sunseg_hard_seen sunseg_easy_unseen sunseg_hard_unseen; do
  python scripts/predict.py --dataset $ds --models emcad-b2@sunseg --postprocs none     # ~10 min/model
done
conda activate sali
python scripts/eval_sunseg_toolbox.py --model_id emcad-b2@sunseg
# SALI: its own test_video.py output is already in the toolbox format
python scripts/eval_sunseg_toolbox.py --model_id sali@sunseg \
    --pred_dir third_party/SALI/res/Net_epoch_8_best/SUN-SEG
# -> artifacts/reports/sunseg/toolbox/<model_id>/<split>_eval.txt
```

The toolbox expects per-frame min-max normalized PNGs named after the original
frames. Without `--pred_dir`, the script converts this model's `predict.py`
output into that format (in `artifacts/predictions/sunseg_toolbox/<model_id>/`).
SUN-SEG frames are ordered by (segment, image number), as the toolbox does,
because some clips are spliced from several segments of the original video.

SALI reproduction (our run vs. the paper's Table 1, in %), all within 0.1:

| Split | S-α | mean E-φ | weighted F | mean F | Sen | Dice |
|---|---|---|---|---|---|---|
| Easy/Seen | 94.4 (94.4) | 97.3 (97.2) | 91.1 (91.0) | 92.5 (92.5) | 92.4 (92.4) | 92.7 (92.7) |
| Hard/Seen | 91.6 (91.6) | 95.2 (95.2) | 86.5 (86.5) | 88.9 (88.9) | 89.8 (89.8) | 89.1 (89.1) |
| Easy/Unseen | 87.0 (87.0) | 92.0 (92.0) | 79.5 (79.4) | 83.1 (83.1) | 81.2 (81.1) | 82.6 (82.5) |
| Hard/Unseen | 87.4 (87.4) | 92.0 (92.0) | 79.0 (79.0) | 82.2 (82.2) | 83.1 (83.0) | 82.3 (82.2) |

## Repository layout

```
polypbench/
├── README.md                    this file
├── docs/ARCHITECTURE.md         design, invariants, how to add things
├── environment.yml              conda-forge env (the only list of dependencies)
├── pyproject.toml               package metadata so `import polypbench` works
├── configs/
│   ├── paths.example.yaml       template for paths.yaml (paths.yaml itself is gitignored)
│   ├── datasets/inhouse.yaml    in-house clips + annotated frame -> mask file (generated)
│   ├── models/<model_id>.yaml   one per model: weights, input size, normalization, provenance (+ `train:`)
│   ├── groups.yaml              model groups for --models, and ColonDB/SUN-SEG training pairs
│   ├── datasets/sunseg.yaml     SUN-SEG layout, test splits, held-out validation videos
│   ├── train/sunseg.yaml        shared training recipe
│   └── postproc/                none.yaml, temporal_smooth.yaml (+ its parameters)
├── scripts/
│   ├── build_inhouse_manifest.py  locates each annotation in its clip -> datasets/inhouse.yaml
│   ├── predict.py                 models -> per-frame probability PNGs
│   ├── evaluate.py                PNGs + ground truth -> metric CSVs + per-clip tables
│   ├── visualize.py               PNGs + ground truth -> sheets + overlay videos
│   ├── stability.py, visualize_stability.py   inter-frame stability tables, graphs, videos
│   ├── compare_trainsets.py       ColonDB- vs SUN-SEG-trained, per architecture
│   ├── predict_sali.py            SALI (own env) -> same prediction layout
│   ├── train_sali.py              retraining SALI causally (own env)
│   ├── train.py, export_onnx.py   training our models; best.pth -> model.onnx
│   ├── build_sunseg.py            official SUN-SEG layout (hard links)
│   └── eval_sunseg_toolbox.py     SUN-SEG benchmark metrics (PNS+ toolbox, parallel)
├── src/polypbench/
│   ├── paths.py                 DATA_ROOT, ARTIFACTS_ROOT, CONFIG_DIR, REPO_ROOT
│   ├── data/
│   │   ├── base.py              Sequence (frames + annotations), VideoFileSequence (mp4), Dataset
│   │   ├── inhouse.py           load_inhouse(): builds the Dataset from datasets/inhouse.yaml
│   │   ├── sunseg.py            load_sunseg(): the four test splits as datasets (FrameFolderSequence)
│   │   └── __init__.py          load_dataset(name)
│   ├── models/
│   │   ├── onnx.py              OnnxPredictor: preprocessing + onnxruntime + sigmoid
│   │   └── __init__.py          load_model(model_id) from configs/models/
│   ├── postproc/
│   │   ├── temporal.py          TemporalSmoother (flow-warped smoothing + area fallback)
│   │   └── __init__.py          load_postproc(name) from configs/postproc/
│   ├── io/predictions.py        the only code that reads/writes saved predictions
│   ├── metrics/overlap.py       Dice, IoU, precision, recall, false alarm, FP area
│   └── viz/overlays.py          drawing helpers + RGB video writer
└── tests/test_inhouse.py        data-loader tests, including frame/annotation alignment
```

## Outputs (`artifacts/`)

`artifacts/` sits next to the repo (`../artifacts`) and is not in git.

```
artifacts/
├── checkpoints/<model_id>/                     model.onnx (+ best.pth, logs for models we trained)
├── predictions/<dataset>/<model_id>/<postproc>/
│   ├── run.json                               config, git commit, device, timestamps
│   └── <clip>/000000.png ...                  one probability map per frame (0-255 = 0-1)
├── metrics/<dataset>/<model_id>/<postproc>.csv   one row per annotated frame x metric
├── stability/<dataset>/<model_id>/<postproc>.csv one row per frame transition (k-1 -> k)
└── reports/<dataset>/
    ├── dice_by_clip.csv                       rows: model/postproc; cols: clips + ALL
    ├── iou_ / precision_ / recall_by_clip.csv
    ├── false_alarm_ / fp_area_frac_by_clip.csv   frames without a polyp
    ├── summary.csv                            mean, std, n for every model/postproc/clip/metric
    ├── sheets/<group>/<postproc>/<clip>.png   rows: annotated frames; cols: source, GT, each model of the group
    ├── trainsets/summary.csv, dice_by_clip.csv   ColonDB- vs SUN-SEG-trained, per architecture
    ├── videos/<postproc>/<model_id>/<clip>.mp4   prediction on every frame; GT + Dice on annotated frames
    └── stability/
        ├── summary.csv                        per model/postproc/clip (+ ALL): IoU mean/median/p5/min, appear/vanish
        ├── iou_warped_mean_by_clip.csv        main stability table (also: iou_warped_p5_, iou_raw_mean_, appear_vanish_)
        ├── compare/<group>/<postproc>/<clip>.png   motion-compensated IoU of each model in the group
        └── <postproc>/<model_id>/<clip>.png, <clip>_side_by_side.mp4
reports/sunseg/toolbox/<model_id>/<split>_eval.txt  SUN-SEG benchmark metrics
```

Colors: **green** = prediction, **red** = ground truth.

## Data

### In-house video (`/home/tn23794/data/in-house-polyp-video/`)

- Four 15-second clips from one colonoscopy (EMR) procedure: `first_polyp_seen`,
  `first_polyp_removed`, `second_polyp_seen`, `second_polyp_removed`. Each is
  456 frames at 480×360 and 29.97 fps.
- 123 annotated frames (`annotations/images` + `annotations/masks`), one every
  15 frames (0-based frames 0, 15, ..., 450). Each is an exact video frame.
- `frame70.png` belongs to `second_polyp_removed` (frame 105), despite its
  number. So `second_polyp_seen` has 30 annotations and the other clips have 31.
- **14 masks are empty** (no polyp in view after removal): `first_polyp_removed`
  frames 330 and 345, and `second_polyp_removed` frames 255 and 300–450. These
  are scored separately (see below).

### SUN-SEG (`/home/tn23794/data/SUN-SEG/`)

- Raw downloads: `SUN-Positive/` (49,136 polyp frames), `SUN-Negative/`
  (13 polyp-free videos, 109,554 frames), and
  `SUN-SEG-Annotation/SUN-SEG-Annotation/` (masks + splits, the 2022 version).
- `scripts/build_sunseg.py` builds the official layout in `SUN-SEG/SUN-SEG/`,
  using the same mapping as PNS+'s `reorganize.py`. Files are hard-linked, not
  moved, so it uses no extra space and leaves the raw folders untouched:

  | Split | Clips | Frames |
  |---|---|---|
  | TrainDataset | 112 | 19,544 |
  | TestEasyDataset/Seen | 33 | 4,719 |
  | TestEasyDataset/Unseen | 86 | 12,351 |
  | TestHardDataset/Seen | 17 | 3,882 |
  | TestHardDataset/Unseen | 37 | 8,640 |

- No SUN-SEG mask is empty: every annotated frame contains a polyp.

Conventions in code: frame indices are **0-based**, frames are **RGB**, and
video is read **sequentially** (never by seeking).

## Models

| model_id | Input | Source |
|---|---|---|
| `emcad@colondb` | RGB 352×352 | ONNX export, trained on ColonDB for 200 epochs (previous master's student) |
| `mkunet@colondb` | RGB 352×352 | same |
| `pvt-cascade@colondb` | RGB 352×352 | same |
| `sacnet-b2@colondb` | RGB 352×352 | same (4-output export; the final head is output 3) |
| `emcad-b2@sunseg`, `mkunet@sunseg`, `sacnet-b2@sunseg` | RGB 352×352 | trained by us on SUN-SEG (see Training) |
| `sali@sunseg` | RGB 352×352, current + next frame + memory | official SUN-SEG checkpoint (MICCAI 2024); runs in `envs/sali.yml` via `scripts/predict_sali.py` |
| `sali@sunseg-causal` | RGB 352×352, current + previous frame + memory | SALI retrained by us on SUN-SEG with causal frame pairs (see Training a causal SALI) |

Model ids follow `arch-size@trainset[-tag]`; the training set is always part of
the name. Training code for each model is in `../../repo_clones/`
(`train_polyp.py` in EMCAD, MK-UNet, CASCADE and SACNet).

SACNet-b0 was dropped: its only available ONNX export is broken (Dice ≈ 0.02).

`sali@sunseg` is not causal: it uses the next frame to segment the current
one (`partner: next` in its config; `sali@sunseg-causal` uses `prev`). Its
saved outputs are the raw sigmoid, not its own per-frame min-max rescaling.
`scripts/predict_sali.py` reproduces SALI's `test_video.py` outputs exactly.

**To add an ONNX model:** put the weights at
`artifacts/checkpoints/<model_id>/model.onnx`, copy one of
`configs/models/*.yaml` and adjust it, then rerun the three commands.
Models that need PyTorch or their own environment (e.g. SALI) only need to
write predictions in the same PNG layout, using `polypbench.io.predictions`.
After that, evaluation and visualization work unchanged.

## Post-processing

- `none`: raw model probabilities.
- `temporal_smooth`: ported from LambdaDiceEval. Each frame's probability map
  is blended with the previous frame's, warped by optical flow. The blend
  weight adapts to how much the scene moved, a large change resets it, and an
  area fallback keeps the previous map when the mask suddenly collapses. The
  old script's hysteresis step isn't ported, because its result never reached
  the final mask.

## Metrics

- Predictions are binarized at **0.5** by default, with no morphological
  cleanup. This is the standard in polyp papers and is fair to SOTA models.
  LambdaDiceEval used 0.75 plus morphological cleanup; `--threshold` is saved
  with every result.
- **Frames with a polyp** (109) get Dice, IoU, precision and recall. Precision
  is left blank (NaN) when nothing was predicted.
- **Frames without a polyp** (14) get `false_alarm` (1 if anything was
  predicted) and `fp_area_frac` (the fraction of the frame wrongly marked as
  polyp). Dice is undefined there.
- Per-clip values are means over that clip's frames. **ALL pools every frame**
  across clips; it is not the average of the four clip means.

### Inter-frame stability

Ported from LambdaDiceEval (`log_interframe_iou.py`, `plot_interframe_iou.py`),
but computed from the saved predictions of **every frame**, so it covers every
model and post-processing variant with no extra inference. For each pair of
consecutive frames, with masks binarized at the same threshold as above:

- `iou_warped` (**main measure**): IoU between the previous mask, warped into
  the current frame by optical flow, and the current mask. Camera and polyp
  motion is compensated, so a drop means the model changed its mind.
- `iou_raw`: the same without motion compensation.
- `frame_diff`, `flow_mag`, `warp_residual`: how much the frame itself changed,
  how far it moved, and how much of the change motion doesn't explain.
- Appear/vanish events: transitions where a mask is present in exactly one of
  the two frames.

IoU is undefined (NaN) when both masks are empty, and those transitions are
excluded from the averages. `p5` is the 5th percentile, i.e. how bad the worst
transitions get.

## Relation to LambdaDiceEval

`../../LambdaDiceEval/` is the frozen original evaluation. The port reproduces
its per-frame Dice at the annotated frames to within 0.007, using its settings
(threshold 0.75, cleanup, temporal smoothing). There are two deliberate
differences:

- It scored every frame against flow-warped copies of the masks. This repo
  scores only the 123 real annotated frames.
- Its matcher mis-paired `frame53`, so its scores for frames near 315 of
  `first_polyp_removed` are unreliable.

## Known issues and notes

- Inference runs at ~20 fps regardless of the model. The limit is CPU work
  (optical flow, PNG writing), not the GPU. That's fine for accuracy runs;
  latency will be measured separately.
- On the frames without a polyp, all models mark the resection site (10–20% of
  the frame) as polyp.
- onnxruntime comes from conda-forge's CUDA build. If a model prints
  `running on CPU`, the GPU libraries didn't load.

## Next steps

1. Retrain our lab's models on SUN-SEG (`.pth` checkpoints, optional ONNX
   export) for a like-for-like comparison with SALI.
2. `--name` option for sheets, so subset comparisons don't overwrite the full
   sheet.
