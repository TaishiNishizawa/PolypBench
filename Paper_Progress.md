# Paper progress: "Automated Real-time Polyp Detection: Hope, Hype, and Reality"

_Last updated: 2026-10-21. Owner: tn23794 (UT Austin, Marculescu lab). Target venue: MIDL 2027 (check the call for dates and back-plan the timeline below)._

This file is the single source of truth for the project's goals, decisions, verified facts, and
open work. Read it fully before making changes. When a decision changes or a task finishes,
update this file in the same commit.

---

## 1. Paper vision

**Story:** the *hope* (CADe can cut polyp miss rates), the *hype* (leading models report
excellent numbers on curated benchmarks such as SUN-SEG), and the *reality* (how those same
models behave on unedited, full-length colonoscopies), ending with concrete next steps for the field.

**Make "hype" concrete with three numbers per model:**

| Column | Meaning |
|---|---|
| **Reported** | What the original paper claims on SUN-SEG |
| **Reproduced** | What we get on SUN-SEG with the released code, under a clean protocol (checkpoint chosen on validation, no per-frame normalization, causal frame access) |
| **Real** | What we get on REAL-Colon full procedures |

The Reported → Reproduced gap (code/paper mismatches, test-set threshold and checkpoint
selection, normalization tricks) is part of the hype story, not only the REAL-Colon gap.

**Title note:** the title says "Detection" but most benchmarked models are segmentation models.
Either add "and Segmentation", or state explicitly that segmentation models are judged on the
per-frame detection decision their masks imply (empty mask = "no polyp").

**Positioning vs TRUE-Colon** (Doerrich et al., MICCAI 2026 EndoLINA workshop, arXiv 2608.13711):
they benchmarked 4 generic single-frame detectors (Faster R-CNN, YOLOv8, YOLOv11, RT-DETR)
on 21 test lesions. We extend to the polyp *segmentation* field's own models, including video
(temporal) models, with causal evaluation, more lesions, and a reproducibility audit. Their
limitations section names temporal models and more lesions as open; we cover both.

---

## 2. Current status (2026-10-21)

**Done**
- SUN-SEG downloaded (full).
- Trained on SUN-SEG (clean train/val/test split, 20-30 epochs) and evaluated on SUN-SEG:
  EMCAD, MK-UNet, CASCADE (2D, our lab) and SALI (video; our fixed causal version).
- REAL-Colon: metadata + all 60 annotation archives; frames for all 20 val/test videos,
  verified complete (`verify` command: 20/20).
- Tools written (see section 9): `realcolon_subset.py`, `view_annotations.py`, `make_clip.py`.
- REAL-Colon in PolypBench (2026-10-21): dataset loader (`realcolon_val/test`, tests pass) and
  npz prediction I/O (section 8). `predict.py` / `predict_sali.py` write npz for REAL-Colon;
  in-house PNG output verified byte-identical to before (EMCAD none + temporal_smooth, causal SALI);
  both models smoke-tested on 300 REAL-Colon frames.
- Throughput seen so far (A6000, in-house, incl. PNG writing): EMCAD ~29 fps, SALI ~7.5 fps.
  At 7.5 fps one SALI pass over val + test (963k frames) is ~36 h: measure properly in the pilot.

**In progress**
- REAL-Colon training split (40 videos, ~595 GB) downloading to the NAS (`--jobs 4`).

**Not started:** remaining models, REAL-Colon inference, evaluation code, analyses, writing.

---

## 3. Environment and storage

| Item | Value |
|---|---|
| Main server | lambdax (`ece-a55005`), 4x NVIDIA RTX A6000 (48 GB each) |
| Code | `~/projects/segmentation/` (this repo: `PolypBench/polypbench`; REAL-Colon tools: `realcolon/`) |
| Conda env | `sali` (used for SALI); others per model as needed |
| REAL-Colon data | `/sld-shared/home/tn23794/realcolon/` (lab NAS, NFS) |
| NAS bandwidth | ~1 Gb/s link: measured 94 MB/s write, 108 MB/s read, shared with the lab |

**Rules**
- **Never write data to the system disk** (`/`, home): it is ~96% full and shared by everyone.
- **The NAS maps every user to the same UID (1024)**, so permissions do not protect files.
  REAL-Colon is re-downloadable, but **keep model predictions and results backed up elsewhere**
  (git for code; a second copy for results).
- Treat the REAL-Colon folder as read-only. Write derived data (caches, predictions) to separate folders.
- **`/mnt/dm-3/taishi`** (local 16 TB HDD, 13 TB free): our folder, granted 2026-10-21. Owned by
  us, mode 700. The rest of `/mnt/dm-3`, `/mnt/dm-5` (16 TB HDD) and `/mnt/dm-4` (3.5 TB SATA SSD,
  **100% full**) are not writable by us (group 1011).

**Storage plan for REAL-Colon (decided 2026-10-21)**
| What | Where | Size |
|---|---|---|
| Original frames | NAS, read-only (all 60 videos) | ~296 KB/frame; ~285 GB val + test |
| Local copy of originals (what models read) | `/mnt/dm-3/taishi/realcolon` (val + test frames, all metadata and annotations); `~/data/REAL-Colon` points here | ~285 GB |
| Frame cache (section 7) | `/mnt/dm-3/taishi/realcolon-cache/L640` (local) | ~140 KB/frame; 134 GB val + test (built) |
| Predictions (section 8) | `/mnt/dm-3/taishi/polypbench-artifacts/predictions` (local), symlinked as `artifacts/predictions/realcolon_<split>` | ≤7-9 GB per run; ≤350-450 GB for ~50 runs |
| Backup of predictions | NAS `/sld-shared/home/tn23794/polypbench-artifacts` (rsync after each run) | same |

Predictions moved to dm-3 as the main copy on 2026-10-22 (before any were written): same
inference speed, but every rescoring reads them all back (~2x faster from local disk).
| Metrics, results, checkpoints | `artifacts/` (local) | small |

Local disk is for speed and a second copy, not capacity: the NAS has ~51 TB free.

**Links (created 2026-10-21).** Code only sees `data_root` and `artifacts_root`; these symlinks
decide where things physically live, so moving storage never needs a code change.
| Link | Target |
|---|---|
| `~/data/REAL-Colon` | `/mnt/dm-3/taishi/realcolon` (local copy, 2026-10-22: val + test frames, all metadata and annotations; train frames only on the NAS at `/sld-shared/home/tn23794/realcolon`) |
| `~/data/REAL-Colon-cache` | `/mnt/dm-3/taishi/realcolon-cache` |
| `artifacts/predictions/realcolon_{val,test}` | `/mnt/dm-3/taishi/polypbench-artifacts/predictions/realcolon_{val,test}` |

These links live outside git (in `data_root` and the `artifacts/` sibling), so this table is
their only record; recreate them with `ln -sn <target> <link>` on a new machine.

Backup after each REAL-Colon run:
```
rsync -a /mnt/dm-3/taishi/polypbench-artifacts/ /sld-shared/home/tn23794/polypbench-artifacts/
```
- Run long jobs in `tmux`. Do not run two `realcolon_subset.py frames` processes on the same
  `--out` with an old script version (fixed in the current version, but avoid anyway).

**REAL-Colon folder layout**
```
/sld-shared/home/tn23794/realcolon/
  video_info.csv, lesion_info.csv, dataset_description.md    # organizers' metadata
  annotations/SSS-VVV_annotations.tar.gz                     # per-frame VOC XML, kept compressed
  frames/SSS-VVV_frames/SSS-VVV_<frame>.jpg                  # original-resolution frames
  summary/videos.csv, summary/lesions.csv                    # our summaries (realcolon_subset.py)
  summary/frame_manifest.json                                # stored vs original frame size per video
```

---

## 4. REAL-Colon: verified facts

Source: Biffi et al., *Sci Data* 2024 (arXiv 2403.02163); Figshare article 22202866; our own summaries.

**Content**
- 60 complete, unedited procedures; 4 cohorts × 15 videos:
  001 (US, GI Genius trial), 002 (Italy), 003 (Austria), 004 (Japan).
- 2,757,723 frames; 351,264 boxes; 132 removed lesions, of which 19 were histologically
  *not* polyps ("NO POLYP": lymphoid aggregates, lipomas, ulcers, normal tissue).
- 14 of the 60 videos contain no polyps at all.
- Videos start outside the patient (videos starting inside the colon were excluded at curation)
  and contain insertion, withdrawal, the removals, and footage after each removal.
- Durations mostly 10-30 min (40 videos), 30-50 min (13), longer (7).
- Frame rate varies: 29.97 fps (cohorts 001 and 004, and Fujifilm videos in 002),
  25 fps (cohort 003 and Olympus videos in 002), **60 fps for 001-012**.
  **Always use per-video fps from `video_info.csv`; never assume 30.**
- Frame size varies by cohort (e.g. 001: 1352×1080; the official README example shows 1240×1080).
  **Always read size per video** (XML `<size>` or `frame_manifest.json`).
- Endoscopes: 53 Olympus, 7 Fujifilm (all Fujifilm in cohort 002).
- Dataset release tag in XML: `v1.0_20230228` (cite in the paper).

**Annotations**
- One Pascal-VOC XML per frame (`SSS-VVV_<frame>.xml`). Frames without polyps have no `<object>`.
- Each box has `<unique_id>` (e.g. `001-014_1`), constant for that lesion across the video,
  including through temporary disappearances. `<box_id>` only numbers boxes within one frame.
- `unique_id` matches `unique_object_id` in `lesion_info.csv` (size in mm, site, histology).
- Annotation protocol: each lesion was boxed **backwards from its complete removal to its first
  appearance**. So onset frames are annotated, and **frames after removal are labeled polyp-free**
  (the resection site is a negative).
- Known quirk: **003-012 has 3 annotated lesions but `video_info.csv` lists 2** (only mismatch
  in the dataset). `lesion_info.csv` has all three (incl. size and histology), so only
  `video_info.csv`'s count is wrong. Note it in the data section.
- **1,725 boxes extend past the frame edge** (mostly cohort 004; e.g. ymax 1087 on a 1080-tall
  frame): clipped in the ground-truth tables, flagged `clipped`. Coordinates are 0-based.
- **8,062 frames carry 2+ boxes** (351,264 boxes on 342,109 boxed frames).
- `lesion_info.csv` site spellings vary (`caecum`/`cecum`, `trasnverse`): normalized in the GT tables.
- Known quirk: **001-012 names its frame and annotation files `001-012_<frame>.0.jpg/.xml`**
  (all other downloaded videos: `<video>_<frame>.jpg`). Recorded as `float_frame_names` in
  `configs/datasets/realcolon.yaml`; recheck once the train split is complete.
- Frame ids are 0-based and contiguous (checked on all val/test videos by `tests/test_realcolon.py`).
- In-patient ranges are inclusive `[first, last]`: the test ranges sum to exactly 572,085.
- **Box tags are written `xmin, xmax, ymin, ymax`** (not VOC's usual xmin, ymin, xmax, ymax;
  seen in 003 and 004 archives). Always parse coordinates by tag name, never by position: a
  positional parse silently scrambles every box (it did in a pilot script, 2026-10-22).
- A lesion's visible span has gaps (004-012: boxed 9,307-13,494 but only 3,592 boxed frames;
  e.g. frame 11,000 has no box). Use per-frame boxes, never first/last frame, as ground truth.
- Some frames show a solid blue rounded rectangle (likely an on-screen-information redaction);
  decided to treat as an occluder and not mask it.

**Official split** (REAL-Colon paper; same in TRUE-Colon): per cohort, videos 1-10 train,
11-12 validation, 13-15 test. Deterministic, no seed. One video per patient.

| Split | Videos | Frames (all) | Polyp frames | Lesions | Polyp-free videos |
|---|---|---|---|---|---|
| Train | 40 | 1,794,907 | 230,936 | 101 | 001-002, 001-008, 002-001, 004-002, 004-007, 004-010 |
| Val | 8 | 371,169 | 29,067 | 10 | 001-011, 001-012, 002-012, 003-011 |
| Test | 12 | 591,647 (572,085 in-patient) | 82,106 | 21 | 001-013, 001-015, 002-013, 002-015 |
| **Total** | 60 | 2,757,723 | 342,109 | 132 | 14 |

These reconcile with both papers: test 591,647 frames (REAL-Colon) and 572,085 after
TRUE-Colon's in-patient trimming; polyp frames 29,067 (val) and 82,106 (test) match TRUE-Colon's
Table 1 exactly, and train differs by one frame (230,936 vs their 230,935): the single 002-003
annotated frame outside their in-patient range. Train has exactly 1,794,907 frames, the
REAL-Colon paper's baseline training-set size.

**Notable lesions:** 004-013 has one lesion visible in ~43% of its frames (~23k frames); this is
TRUE-Colon's "lesion 14", a sessile serrated lesion that every detector struggled with.

---

## 5. What TRUE-Colon did differently from raw REAL-Colon

| Aspect | REAL-Colon as released | TRUE-Colon |
|---|---|---|
| Frames used | All, including outside the patient | Trimmed to per-video in-patient ranges (their Table B.1; transcribed in `realcolon_subset.py`, verified against annotations) |
| Image size | Native | Letterboxed to 640×640; **their prepared annotation files are in 640×640 coordinates**, do not reuse them |
| Scoring | COCO AP | AP plus frame-level hit (any overlap, IoU>0); thresholds fixed (0.2) and matched to FPR 4-5% **on test** |
| Early detection | AP over first 1 s / 3 s | Frames to first detection; lesions with an annotation gap >50 frames in their first 250 excluded from the latency mean; seconds assume 30 fps |
| Lesions | All 132 incl. 19 not polyps | All annotated lesions kept (test includes one "NP" lesion) |
| SUN | n/a | Original SUN database with their own case split (not SUN-SEG); training included SUN negatives |

**Notable:** REAL-Colon's authors deliberately kept out-of-patient footage ("AI must operate
beyond the colon"); TRUE-Colon removes it. We report both.

**Reference numbers from TRUE-Colon (REAL-Colon test, 21 lesions):**
- SUN-trained detectors on REAL-Colon: frame TPR 0.24-0.31 at FPR ~0.04 (their Table 3).
- REAL-Colon-trained, at FPR ~4-5%: TPR 0.46-0.65; best RT-DETR 0.651; mean first-detection
  latency 38-64 frames. Lesion-level differences were within seed noise (n=21).
- All their detectors run >30 fps on an H100.

---

## 6. Models to benchmark

**Inclusion criteria (state in the paper):** public training + inference code; trainable on
SUN-SEG; automatic (no human prompts). **Excluded:** WavePolyp, HFSTI-Net (ICLR 2026) and MAST
(no public code); SAM2 / MedSAM2 used directly (need prompts); PolyMamba (repo unreachable when checked).

### Priority: 2D (single-image), 6
| Model | Repo | Status | Why |
|---|---|---|---|
| PraNet (MICCAI 2020) | github.com/DengPingFan/PraNet | 3 Seed SUNSEG training ... | Root of most later polyp models |
| Polyp-PVT | github.com/DengPingFan/Polyp-PVT | 3 Seed SUNSEG training ... | Standard transformer baseline |
| EMCAD (CVPR 2024) | github.com/SLDGroup/EMCAD | Trained + SUN-SEG eval (3 seeds) | Lab model |
| SAM2-UNet | github.com/WZH0120/SAM2-UNet | TODO | Foundation-model encoder |
| SSFormer 2D | github.com/shiwt03/SSformer | TODO | Generic strong baseline |
| nnU-Net 2D | github.com/MIC-DKFZ/nnUNet | TODO | Generic strong baseline |

### Priority: video ("3D" = 2D + time), 4
| Model | Repo | How it uses other frames | Status |
|---|---|---|---|
| PNS+ (MIR 2022) | github.com/GewelsJI/VPS | First frame of clip as anchor + 5-frame window | TODO |
| CMSA-Net (MICCAI 2026) | github.com/wangtong627/CMSA-Net | Causal attention (past only) + first-frame anchor (verified in loader) | SUNSEG training (s0 done, eval in progress), s1 + s2 to follow  |
| SALI (MICCAI 2024) | github.com/Scatteredrain/SALI | Memory bank + **frame t+1 as released** (verified) | Fixed causal version trained (3 seeds) |
| STDDNet (ICCV 2025) | github.com/C-GLGLGL/STDDNet | Non-overlapping 5-frame windows + first-frame anchor (verified in loader) | TODO |

### Extra (only if time allows)
| Model | Type | Repo | Notes | Status |
|---|---|---|---|---|
| CASCADE | 2D | github.com/SLDGroup/CASCADE | Lab model | Trained + SUN-SEG eval (3 seeds) |
| MK-UNet | 2D | github.com/SLDGroup/MK-UNet | Lab model (lightweight) | Trained + SUN-SEG eval |
| MEGANet (WACV 2024) / HSNet | 2D | github.com/UARK-AICV/MEGANet, github.com/baiboat/HSNet | Extra recent architecture | Optional |
| FLA-Net (MICCAI 2023) | Video | github.com/jhl-Det/FLA-Net | Neighboring frames (**verify direction**) | TODO |
| LGRNet (MICCAI 2024) | Video | github.com/bio-mlhui/LGRNet | Clip-level global attention (**verify**) | TODO |
| Diff-VPS (MICCAI 2024) | Video | github.com/lydia-yllu/Diff-VPS | 5-frame clips; **no released weights** | TODO |
| YOLO-SAM 2 | Video | github.com/sajjad-sh33/YOLO_SAM2 | Detector prompts SAM2 video tracking (**verify**) | Optional |

### Reference detectors, 2
YOLOv11-M and RT-DETR (Ultralytics), trained on boxes derived from SUN-SEG masks, same split.
Purpose: answer "do segmentation models detect better or worse than detectors?" and link our
numbers to TRUE-Colon's.

### Verified evaluation pitfalls in released code (feed the "hype" section)
- **SALI** (commit `1e64786`): test loader pairs frame *i* with *i+1*
  (`dataloaders/video_list.py` lines 218-223); `val_longshort` aligns and fuses the *next*
  frame's features into frame *i*'s prediction; training also uses *i+1*. The paper claims it
  "sees no future frames". Memory is cleared and seeded at each clip's first frame (always
  polyp-positive in SUN-SEG). `test_video.py` min-max normalizes every output.
- **Per-frame min-max normalization** (`(res - min) / (max - min)`) in PraNet-lineage test
  scripts forces a confident prediction on every frame, making false alarms unmeasurable.
  **Check every repo's test script and remove it.**
- **First-frame anchors** (PNS+, STDDNet, CMSA-Net): in SUN-SEG the anchor always shows the
  polyp; in a live procedure it does not.
- **SUN-SEG protocol:** reports *maximum* Dice over thresholds (threshold chosen on test);
  evaluates polyp-positive clips only; "Seen" test clips come from training videos.
- **Checkpoint selection on test** is common in these repos. Check each training script.

### Training protocol (all models)
- Same data: SUN-SEG train; validation split carved from training *cases* (never test).
- Keep each method's own recipe (optimizer, LR, schedule, epochs, loss); hyperparameters are part of the method.
- Same input resolution policy (352 standard; PNS+ native 256×448), same pretrained-backbone policy.
- 3 seeds per model; checkpoint chosen on validation only.
- Record per model: repo commit, config, deviations from paper, weight source.
- Sanity check: released weights (where available) on SUN-SEG test should reproduce reported numbers in our evaluator.
- Fairness ablation: at least one configuration per family trained **with SUN negative clips**
  (TRUE-Colon's SUN training included negatives; standard SUN-SEG training does not).

---

## 7. REAL-Colon inference protocol

**Frames**
- **Main results:** 12 official test videos, TRUE-Colon in-patient ranges, every frame (no
  negative subsampling) = 572,085 frames.
- **Threshold selection:** 8 validation videos, same trimming.
- **Secondary:** (a) all frames including out-of-patient; (b) extended evaluation on train +
  test videos (~120 lesions) for statistical power, since SUN-SEG-trained models never see
  REAL-Colon. Report the official test set separately for comparability and future REAL-Colon-trained models.

**Running models**
- Preprocess exactly as in SUN-SEG training (resize + normalization). Frames are already
  cropped to the field of view; do not crop further.
- One continuous pass per video, frame by frame, **past frames only**, **starting at frame 0**
  (including out-of-patient footage, as deployed; decided 2026-10-21). In-patient trimming is
  applied only at evaluation, so one pass serves both the main and the all-frames results.
  Differs from TRUE-Colon (trimmed before inference); matters only for memory/anchor models.
- Memory models (SALI): reset only at video start.
- Window models (PNS+, STDDNet, Diff-VPS, LGRNet): predict frame *t* from the window ending at *t* (one forward pass per frame).
- Anchor models: anchor = first frame of the evaluated procedure (realistic). Rolling anchor as an ablation.
- Also run **as-released** (future-frame) variants of SALI and window models, reported
  separately with their built-in lookahead delay.
- **Save raw probability maps, no normalization** (format: section 8). All box extraction,
  thresholds and metrics are computed offline from the saved maps, so models never need re-running for scoring changes.
- I/O: one pass over the NAS frames (`scripts/build_realcolon_cache.py`) writes a smaller copy
  of every frame and all models run from it. Decided 2026-10-21: **640 px long side** (INTER_AREA,
  aspect kept, no crop or pad, every frame, ids and fps unchanged), JPEG q95 with **4:4:4 chroma**
  (4:2:0 would store colour at 320 px, below the 352 model input). The cache does not set the
  model input: each model's own preprocessing (stretch to 352×352 for ours, as in SUN-SEG
  training) starts from it. 640 is ≥ every planned model input and equals TRUE-Colon's 640×640
  letterbox, so the reference detectors can be run exactly as TRUE-Colon did (pad only, no
  resampling). We do not letterbox our segmentation models: SUN-SEG training never padded.
  Measured on 001-013: original 1352×1080 JPEG 296 KB, 6.6 ms decode; cached 640×511 121 KB,
  2.3 ms decode; ~117 GB for val + test. The NAS is latency-bound (~65 ms per file read), so the
  builder reads ahead in threads.
  The cache adds one extra resize (original → 640 → model input); on the pilot, check that
  predictions from the cache match predictions from the originals.
- **Superseded 2026-10-22: models read the original frames, not the cache.** Pilot on 004-012
  (frames 8,500-11,500, around the lesion's onset), each model run twice, cache vs original:
  | | EMCAD (`emcad-b2@sunseg`) | causal SALI |
  |---|---|---|
  | Dice@0.5 between the two runs' masks, mean (p1) | 0.87 (0.02) | 0.97 (0.79) |
  | frames whose any-pixel ≥ 0.5 decision flips | 8 / 3,000 | 0 / 3,000 |
  | mean probability difference | 0.003 | 0.001 |

  Decomposed for EMCAD (1,500 frames, vs the training-matching input, original → 352 bilinear):
  JPEG alone Dice 0.97; the extra area-averaging resize alone 0.88; original → 352 with
  INTER_AREA 0.88. So the **resize kernel**, not JPEG, causes it: EMCAD (and the other lab
  models) trained with `A.Resize` = cv2 INTER_LINEAR from the full frame, which keeps grain and
  aliasing; area averaging smooths it. SALI trained with PIL's antialiased resize, close to the
  cache path, so it barely changes. Differences sit at blob edges and in faint blobs near 0.5
  (EMCAD's median predicted area here: 0.7% of the frame), which feed mask → box extraction.
  **Rule: each model applies its exact training resize to the original frame.** Side finding for
  the paper: a "harmless" change of resize kernel moves EMCAD's masks by ~12% (Dice 0.88), a
  robustness point. The cache is kept for visualization and possibly the 640-letterbox detectors.
  Originals copied to `/mnt/dm-3/taishi/realcolon` (val + test frames + all metadata) so runs
  don't depend on the shared NAS.

**Mask → boxes** (offline)
1. Threshold probability map at τ.
2. Connected components (8-connectivity); drop components below a fixed minimum area.
3. One box per component, scaled to original pixel coordinates.
4. Box score = maximum probability inside the component.
Sweeping τ gives the full operating curve.

---

## 8. Prediction file format (all model runners must write this)

Follows the repo's prediction contract (`docs/ARCHITECTURE.md`, invariant 1); only
`src/polypbench/io/predictions.py` reads or writes these files.

```
artifacts/predictions/<dataset>/<model_id>/<postproc>/   # dataset = realcolon_val | realcolon_test (proposed names)
  run.json                      # written last; its presence marks a complete run
  <video>/chunk_<k:04d>.npz     # k = 0,1,2,...; up to 2,000 frames per chunk
```
- `model_id` = `arch-size@trainset[-tag]` (repo invariant 2). The tag carries the variant
  (`sali@sunseg` = as released, `sali@sunseg-causal`) and, if adopted, the seed (open decision, section 12).
- Each chunk is written to a temp file and renamed, and each video is marked complete on its own,
  so a crash costs one video, not the run.

Each `chunk_*.npz` contains:
- `frames`: int32 array (N,), REAL-Colon frame ids, ascending (0-based, = frame index).
- `probs`: uint8 array (N, H, W) = round(255 × sigmoid probability), **no min-max normalization**.
  H, W = the **model's own output resolution**: whatever its forward pass returns as the final
  prediction, after the model's own upsampling (learned or fixed) and **before any resizing by us**
  (352×352 for EMCAD and SALI; PNS+ will differ). Not resized to the frame, not downsampled.

Why not smaller (decided 2026-10-21): EMCAD and SALI compute logits on an 88×88 grid and upsample
×4 bilinearly inside `forward`, but storing that grid would need model-specific hooks and
re-creating each model's upsampling in the evaluator. Downsampling to 176 px (the earlier plan)
was measured on SUN-SEG Hard-Unseen and in-house predictions: Dice vs original 0.997 (worst
frame 0.939), no frame's any-detection decision changed, but **~8% of connected components were
lost** (small fragments), which affects mask→box counts and the minimum-area rule. Saving the raw
output loses nothing. Measured size: 7-9 KB/frame at 352 vs 2-3 KB at 176 (upper bound; measured on
polyp-heavy frames, and polyp-free REAL-Colon frames compress better).

`run.json` contains the resolved model config plus: `model`, `repo`, `commit`, `checkpoint`,
`seed`, `train_data` (e.g. `SUN-SEG train`), `input_hw`, `map_hw`, `resize_mode` (`stretch` =
whole frame resized, no crop or pad; only mode supported by the evaluator for now),
`normalization` (mean/std), `frame_access` (`causal` | `as_released`), `anchor_rule`, `window`,
`memory_reset`, `notes`. Per-video frame size for scaling maps back to original pixels comes from
`summary/frame_manifest.json` (`original_wh`); sizes vary per video (1160-1352 × 959-1080).

Predictions live on `/mnt/dm-3/taishi`, symlinked into `artifacts/`, and are backed up to the
NAS (storage plan, section 3).

---

## 9. Evaluation protocol

**Frame level** (TRUE-Colon definitions)
- Positive frame = ≥1 ground-truth box. Hit = any predicted box overlapping any ground-truth box (IoU > 0).
- False alarm = any predicted box on a polyp-free frame.
- Report TPR and FPR at three thresholds:
  1. **Primary:** τ chosen on the **validation** videos for FPR ≈ 4-5%.
  2. **Fixed:** τ = 0.5.
  3. **Comparison only:** τ chosen on test for FPR ≈ 4-5% (as TRUE-Colon did).
- Full curve over τ (AFROC: lesion-box sensitivity vs fraction of polyp-free frames with an alert).

**Lesion level**
- Detected (≥1 hit frame); persistent (hits on ≥50% of visible frames).
- Time to first detection in **seconds using per-video fps**; detected within 1 / 3 / 5 s.
- TRUE-Colon's latency exclusion rule (annotation gap >50 frames within first 250).

**Secondary:** box AP50 (caveat: mask-derived boxes are shaped differently from annotators' boxes).

**Breakdowns**
- Lesion size (mm) and histology, with and without "NO POLYP" lesions.
- Cohort and endoscope brand.
- False alarms by phase: outside patient; inspection; **after removal** (window after each
  lesion's last box; define length, e.g. 60 s); clean negatives.

**Statistics:** bootstrap over lesions for confidence intervals; seed variation.

**Same yardstick on SUN-SEG:** compute the frame-level metrics above on SUN-SEG test
*including its polyp-free clips*, alongside the standard SUN-SEG metrics. Never compare
SUN-SEG Dice against REAL-Colon box metrics directly.

**Cross-check:** on shared metrics, our evaluator must agree with TRUE-Colon's
(`pip install true-colon`, dataset preset `realcolon`).

**Ground truth files** (to build): per frame of val/test, boxes in original pixels with
`unique_id`, image size, in-patient flag, phase label; per lesion: first/last frame, fps,
size, histology, latency-exclusion flag.

---

## 10. Tools

| Script | Location | Purpose |
|---|---|---|
| `realcolon_subset.py` | `realcolon/` | `meta` (metadata + annotations + summaries), `summary`, `frames` (download subset, `--jobs N`, resume, MD5, extract), `verify` (check completeness, rebuild manifest records) |
| `view_annotations.py` | `realcolon/` | Draw GT boxes on selected frames (before / onset / spread / after removal) |
| `make_clip.py` | `realcolon/` | Render a video clip with GT boxes (`--anchor onset|end|full`, `--start/--end`, `--speed`) |
| `build_realcolon_gt.py` | `PolypBench/polypbench/scripts/` | Ground-truth tables (boxes, lesions, videos) for all 60 videos → `artifacts/ground_truth/realcolon/`; read with `load_ground_truth()` |
| `extract_boxes.py` | `PolypBench/polypbench/scripts/` | REAL-Colon npz maps → boxes (one per connected component, peak + mean score, area), scaled to the original frame → `artifacts/boxes/<dataset>/<model_id>/<postproc>/<boxes>/<video>.csv.gz`; config `configs/boxes/cc-0.5.yaml`. ~250 frames/s per video (npz decode bound), one video per worker |
| `evaluate_detection.py` | TODO | Boxes → frame / lesion metrics, curves, phase breakdowns (`metrics/detection.py`) |
| Model runners | TODO | One adapter per model writing section 8 format |
| `view_predictions.py` | `PolypBench/polypbench/scripts/` | Local frame browser for one REAL-Colon video: per-model prediction overlays + GT boxes, clickable timeline of predicted area, threshold control. Serves `127.0.0.1:8765`; open via VS Code port forwarding |

---

## 11. TODO (tiered)

### Tier 0: this week (pipeline end to end on a pilot)
Build order agreed 2026-10-21 (steps 0-6). Each step is tested before the next.

Done:
- [x] **Step 0, storage + links:** `/mnt/dm-3/taishi` granted; symlinks for data, cache and
  predictions (section 3).
- [x] **Step 1, dataset:** `configs/datasets/realcolon.yaml` + `data/realcolon.py`
  (`realcolon_{train,val,test}`, every frame from 0, per-video fps, `inside_patient()`);
  `tests/test_realcolon.py`. Found: 001-012 `.0` file names; in-patient ranges inclusive (sum 572,085).
- [x] **Step 2, prediction I/O:** npz chunks at model output resolution for REAL-Colon, PNG
  unchanged for in-house / SUN-SEG (byte-identical regression check); `tests/test_predictions.py`.
  Committed as `2b198df` (submodules PraNet, Polyp-PVT, VPS: `1d5b763`).
- [x] **Step 3, frame cache:** `scripts/build_realcolon_cache.py` (640 px, q95 4:4:4, NAS
  read-ahead). All 20 val + test videos cached 2026-10-21 (962,816 frames, 134 GB, 79 min);
  PSNR vs resized originals 46-49 dB on clean videos, ~38 dB on grainy 004-012 (test: ≥ 35 dB).
  **Not used for inference** (step 4 finding, section 7: `frames_from: original`); kept for
  viewing and fast experiments.
- [x] **Step 4, pilot inference** (2026-10-22) on **004-012** (val, not test: the pilot informs
  evaluation choices, so it must not look at test). `--sequences` in `predict.py` /
  `predict_sali.py` (run.json only once every video is done; tests). Found the cache's resize
  kernel changes EMCAD's masks (Dice 0.88 vs the training resize; JPEG alone 0.97) → each model
  applies its own training resize to the original frame; val + test originals copied to dm-3.
  Cache-based outputs moved to `/mnt/dm-3/taishi/pilot_from_cache_2026-10-22`.
  From originals: `emcad-b2@sunseg-prime` 22.4 fps, `sali@sunseg-causal` 6.4 fps (from the
  cache: 50 / 7.7 fps; SALI model 135 ms/frame, memory bank capped at 35 entries).
  Storage: EMCAD 11.5 KB/frame, SALI 6.6 KB/frame. Viewer: `scripts/view_predictions.py`.
  **First look (004-012 only, 1 lesion; "fires" = any pixel ≥ 0.5, no boxes or min area yet):**
  | | EMCAD-b2 (PRIME) | SALI (causal) |
  |---|---|---|
  | fires on outside-patient frames (1,893) | 100% | 99.9% |
  | fires on in-patient polyp-free frames (18,042) | 100% (peak ≥ 0.9: 99.9%) | 99.9% (98.5%) |
  | ≥ 0.5 region touches the GT box, boxed frames (3,592) | 90.6% | 87.9% |
  | median ≥ 0.5 area on polyp-free in-patient frames | 3.4% of frame | 1.7% |

  
  Both find the lesion when it is in view (outline matches it closely, incl. NBI views), but
  predict something on essentially every frame: debris and fluid, bright out-of-focus regions,
  specular highlights; no fixed screen element (firing is spread over the tissue). EMCAD's false
  regions are larger and more numerous than SALI's. Consistent with positive-only training
  (every SUN-SEG train frame has a polyp, so nothing teaches "predict nothing"): the "hype"
  story, and the reason for the with-negatives ablation (section 6). Frame-level any-pixel
  scoring is useless here; step 6 must judge boxes, minimum area and per-box scores.
  Pilot predictions stay for step 6 development; the models will be retrained on the full
  SUN-SEG train set, then everything is rerun under the new model ids.
- [x] **Step 5, ground truth** (2026-10-26): `scripts/build_realcolon_gt.py` +
  `data/realcolon_gt.py` → `artifacts/ground_truth/realcolon/{boxes,lesions,videos}.csv.gz`
  for all 60 videos (51 s); read with `load_ground_truth()`. Facts only: per-frame facts (in
  patient or not) follow from each video's 0..n-1 frames and in-patient range; **phases ("after
  removal" etc.) are derived at evaluation** from lesion last frames + fps + the chosen window,
  so the open window length never forces a rebuild. Boxes clipped to the frame (1,725, mostly
  cohort 004; `clipped` flag); 8,062 frames have 2+ boxes. Matches every published count
  (`tests/test_realcolon_gt.py`, 8 tests; 38 in the repo pass): 351,264 boxes; frames
  1,794,907 / 371,169 / 591,647; boxed frames 230,936 / 29,067 / 82,106 (TRUE-Colon Table 1 for
  val/test); lesions 101 / 10 / 21, all boxed and in `lesion_info.csv`; 19 not polyps;
  polyp-free videos as listed in section 4; only lesion-count mismatch is 003-012 in
  `video_info.csv`. TRUE-Colon latency exclusion (our reading: a gap of > 50 missing frames
  starting within a lesion's first 250 frames) flags 7 / 21 test lesions, 0 / 10 val,
  37 / 101 train: **cross-check against `true-colon` in step 6.**

Next:
- [ ] **Step 6, evaluation** (plan agreed 2026-10-26: 6a boxes, 6b metrics, 6c evaluator, 6d
  `true-colon` cross-check).
  - [x] **6a, maps → boxes:** `postproc/boxes.py` + `scripts/extract_boxes.py` +
    `configs/boxes/cc-0.5.yaml` (binarize at 0.5, 8-connected components, boxes scaled from the
    352 map to the frame, axis by axis: the model saw the whole frame resized, no letterbox).
    Each box keeps area and both candidate scores, so score rule and minimum area are chosen
    at evaluation without re-extracting. `tests/test_boxes.py`.
  - [x] **6b, metrics:** `metrics/detection.py`: per-frame alarm / hit scores (TRUE-Colon:
    hit = a box sharing a pixel with any GT box; false alarm = any box on a polyp-free frame),
    TPR / FPR / box sensitivity over tau, tau for a target FPR, per-lesion detected /
    persistent (≥ 50% of boxed frames) / latency (s, per-video fps) / within 1-3-5 s, per-frame
    phases (positive, outside, inspection = unboxed frames inside a lesion's span,
    after_removal = `after_removal_s` after a lesion's last box, clean). `tests/test_detection.py`.
  - **Pilot finding (004-012, val, old pilot models, 1 lesion; FPR target 5%):** the **peak**
    score is unusable: > 5% of polyp-free frames have a box with peak probability exactly 1.0
    (uint8 255), so FPR ≤ 5% needs tau > 1 and TPR is 0. The **mean** probability over the
    component works: tau ≈ 0.96 (EMCAD) / 0.98 (SALI), frame TPR 0.50 / 0.48, the lesion first
    hit 0.6 s / 1.8 s after its first box, hit on ~50% of its frames. Minimum area barely
    matters (EMCAD TPR 0.50 → 0.55 at 1% of the map, worse at 2%; SALI flat). In-patient-only
    vs all frames: same here. One lesion only: recheck on all of val with the `-ft-s0` models.
  - [ ] **6c, `evaluate_detection.py`:** thresholds (val FPR 5%, fixed 0.5, test-matched),
    long-format results to `artifacts/results/`, both frame scopes.
  - [ ] **6d, cross-check vs `true-colon`** (pilot boxes + latency-exclusion list; needs
    `pip install true-colon`, ask first).
- [ ] Before full runs: decode frames in a background thread in `predict.py` / `predict_sali.py`
  (EMCAD val + test ~12 h at 22 fps from originals; SALI ~35 h on one GPU, ~9 h over 4 by video).
- [x] Retrain on the full SUN-SEG train set (2026-10-26, user): `emcad-b2@sunseg-ft-s0`,
  `pvt-cascade@sunseg-ft-s0`, `sali@sunseg-causal-ft-s0` (and `cmsa-net@sunseg-ft-s0`);
  checkpoints in `artifacts/checkpoints/`, seed suffix resolved by `load_model_config`.
- [ ] Run the `-ft-s0` models on REAL-Colon (val first, for the step 6 choices).
- [ ] Training data loader: cap worker threads (`worker_init_fn` with `cv2.setNumThreads(0)`,
  `torch.set_num_threads(1)`); the last training run hit load ~650.
- [ ] Pilot on 001-013, 001-014, 002-014, 003-013, 004-014 (4 cohorts, 12 lesions, 1 polyp-free
  video). **These are test videos:** use them only to check the pipeline runs, never to choose
  scores, areas or thresholds (those come from val).
- [ ] Visual sanity check: overlay predicted vs GT boxes on onset and post-removal frames
  (`view_predictions.py` once step 6 produces boxes).
- [ ] As-released (next-frame) SALI on REAL-Colon, alongside causal.
- [ ] Check the train-split download finishes; run `verify --splits train`; recheck
  `float_frame_names` on the new videos.
- [ ] Audit our lab models' SUN-SEG evaluation: checkpoint chosen on val? min-max normalization removed?

Open decisions to settle before multi-seed runs: seeds in `model_id` tag (section 12).

### Tier 1: next 2-3 weeks (models)
- [ ] Clone and train the remaining priority models on SUN-SEG under the section 6 protocol (3 seeds): PraNet, Polyp-PVT, SAM2-UNet, SSFormer, nnU-Net, PNS+, STDDNet, CMSA-Net (s1, s2); detectors YOLOv11-M, RT-DETR. Extra models (section 6) only if time allows.
- [ ] Per model: record commit and deviations; remove per-frame normalization. (Frame access still unverified for the extra models FLA-Net, LGRNet, Diff-VPS, YOLO-SAM 2.)
- [ ] Reproduce reported SUN-SEG numbers with released weights (Reported vs Reproduced).
- [ ] SUN-SEG evaluation under: standard protocol; corrected protocol (fixed val threshold, no normalization, causal); frame metrics with polyp-free clips.
- [ ] One with-negatives training configuration per model family.

### Tier 2: weeks 3-6 (REAL-Colon results)
- [ ] Run all models on val + test from originals (causal; plus as-released variants).
- [ ] Thresholds from val; full metrics on test; comparison-only test-matched thresholds.
- [ ] Breakdowns (size, histology, cohort, phases incl. after-removal false alarms).
- [ ] Extended evaluation on train + test videos (~120 lesions).
- [ ] Rank agreement: SUN-SEG ranking (standard vs corrected) vs REAL-Colon ranking.
- [ ] Bootstrap CIs; seed variance.

### Tier 3: weeks 6-8 (analyses, figures, writing)
- [ ] Latency: effective latency (compute + lookahead wait) on A6000; optional Jetson Orin Nano constrained-edge table.
- [ ] SUN-SEG vs REAL-Colon characterization (onset size, blur, positive prevalence, clip structure).
- [ ] Figures: Reported → Reproduced → Real per model; accuracy vs effective latency; AFROC curves; example failure frames.
- [ ] Write: hope → hype → reality → next steps.

### Later / follow-up papers
- [ ] Box-supervised (or SAM-pseudo-mask) training of segmentation models on REAL-Colon train split.
- [ ] Resection-site assessment (residual polyp) with clinical collaborator data; ask whether margin biopsies are available.

---

## 12. Open decisions

- Anchor rule for anchor models (main: first frame of procedure; ablation: rolling).
- Length of the "after removal" window for phase-wise false alarms.
- Minimum component area for mask → box conversion (pilot: barely matters; applied at evaluation).
- Box score: mean probability over the component (pilot: peak saturates at 1.0 on false boxes).
- Whether to use in-house clinical videos (limited, curated) and in what role (external curated test set).
- Final title wording (detection vs detection and segmentation).
- Seeds in the `model_id` tag (e.g. `emcad-b2@sunseg-s1`)? Needed before multi-seed REAL-Colon runs.

---

## 13. Key references

- REAL-Colon: Biffi et al., *Sci Data* 11:539 (2024); arXiv 2403.02163; github.com/cosmoimd/real-colon-dataset
- TRUE-Colon: Doerrich et al., MICCAI 2026 EndoLINA; arXiv 2608.13711; github.com/sdoerrich97/true-colon
- SUN-SEG / PNS+: Ji et al., *Machine Intelligence Research* 2022; github.com/GewelsJI/VPS
- SUN database: Misawa et al., *Gastrointest Endosc* 2021