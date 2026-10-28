# SUN-SEG cheat sheet

Run everything from `polypbench/`. Seed runs are `<model_id>-s<N>`, e.g. `emcad-b2@sunseg-ft-s1`.
Logs go to `../artifacts/logs/`.

| Model | Env | Train | Predict |
|---|---|---|---|
| EMCAD, CASCADE (2D) | `train` | `scripts/train.py` | `scripts/predict.py` (`.pth` by default) |
| SALI (causal) | `sali` | `scripts/train_sali.py` | `scripts/predict_sali.py` |
| CMSA-Net | `sali` | `scripts/train_cmsa.py` | `scripts/predict_cmsa.py` |

Toolbox scoring needs the `sali` env; tables need the `polypbench` env.

Splits: `sunseg_easy_seen sunseg_hard_seen sunseg_easy_unseen sunseg_hard_unseen`

## One step at a time

```bash
# 1. Train (one GPU each; add --resume to continue a crashed run)
conda activate train
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 python scripts/train.py --model_id emcad-b2@sunseg-ft-s1 --test_every 1
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 python scripts/train.py --model_id pvt-cascade@sunseg-ft-s1 --test_every 1
conda activate sali
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 python scripts/train_sali.py --model_id sali@sunseg-causal-ft-s1
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 python scripts/train_cmsa.py --model_id cmsa-net@sunseg-ft-s1

# 2. Predict (once per split)
conda activate train
CUDA_VISIBLE_DEVICES=0 python scripts/predict.py --dataset sunseg_hard_unseen --models emcad-b2@sunseg-ft-s1 --postprocs none
conda activate sali
CUDA_VISIBLE_DEVICES=0 python scripts/predict_sali.py --dataset sunseg_hard_unseen --model_id sali@sunseg-causal-ft-s1
CUDA_VISIBLE_DEVICES=0 python scripts/predict_cmsa.py --dataset sunseg_hard_unseen --model_id cmsa-net@sunseg-ft-s1

# 3. Score with the SUN-SEG toolbox (all four splits)
conda activate sali
python scripts/eval_sunseg_toolbox.py --model_id emcad-b2@sunseg-ft-s1 --workers 32

# 4. Tables -> ../artifacts/reports/tables/sunseg_*.md
conda activate polypbench
python scripts/make_tables.py
```

ONNX (Jetsons only): `python scripts/export_onnx.py --model_id <id>` (train env), then `predict.py --backend onnx`.

## All at once: train, predict, score, tables

Paste one block per GPU, inside tmux. It stops at the first error.
Change `CUDA_VISIBLE_DEVICES` and the model list as needed.

**EMCAD and CASCADE**
```bash
(
  set -eo pipefail
  TRAIN=~/miniconda3/envs/train/bin/python
  SALI=~/miniconda3/envs/sali/bin/python
  PB=~/miniconda3/envs/polypbench/bin/python
  export CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=8
  for m in emcad-b2@sunseg-ft-s1 pvt-cascade@sunseg-ft-s1 emcad-b2@sunseg-ft-s2 pvt-cascade@sunseg-ft-s2; do
    $TRAIN scripts/train.py --model_id $m --test_every 1 2>&1 | tee ../artifacts/logs/$m.out
    for s in sunseg_easy_seen sunseg_hard_seen sunseg_easy_unseen sunseg_hard_unseen; do
      $TRAIN scripts/predict.py --dataset $s --models $m --postprocs none
    done 2>&1 | tee ../artifacts/logs/predict_$m.out
    $SALI scripts/eval_sunseg_toolbox.py --model_id $m --workers 32
    $PB scripts/make_tables.py
  done
)
```

**SALI** (for CMSA-Net, swap in `train_cmsa.py` / `predict_cmsa.py` and `cmsa-net@sunseg-ft-s<N>`)
```bash
(
  set -eo pipefail
  SALI=~/miniconda3/envs/sali/bin/python
  PB=~/miniconda3/envs/polypbench/bin/python
  export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8
  for m in sali@sunseg-causal-ft-s1 sali@sunseg-causal-ft-s2; do
    $SALI scripts/train_sali.py --model_id $m 2>&1 | tee ../artifacts/logs/$m.out
    for s in sunseg_easy_seen sunseg_hard_seen sunseg_easy_unseen sunseg_hard_unseen; do
      $SALI scripts/predict_sali.py --dataset $s --model_id $m
    done 2>&1 | tee ../artifacts/logs/predict_$m.out
    $SALI scripts/eval_sunseg_toolbox.py --model_id $m --workers 32
    $PB scripts/make_tables.py
  done
)
```

**Predict + score + tables only** (model already trained)
```bash
(
  set -eo pipefail
  SALI=~/miniconda3/envs/sali/bin/python
  PB=~/miniconda3/envs/polypbench/bin/python
  for s in sunseg_easy_seen sunseg_hard_seen sunseg_easy_unseen sunseg_hard_unseen; do
    CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8 $SALI scripts/predict_cmsa.py --dataset $s --model_id cmsa-net@sunseg-ft-s0
  done 2>&1 | tee ../artifacts/logs/predict_cmsa-net@sunseg-ft-s0.out
  $SALI scripts/eval_sunseg_toolbox.py --model_id cmsa-net@sunseg-ft-s0 --workers 32
  $PB scripts/make_tables.py
)
```

## If something stops

- The trainers refuse to overwrite an existing `best.pth`. To continue a crashed run, use `--resume`. To start over, use `--overwrite`.
- Prediction skips splits that are already done. Rerunning the toolbox or `make_tables.py` is safe.
- `--test_every` only adds logging; it never changes which epoch is chosen. It's cheap for 2D models but slow for SALI and CMSA-Net (`--test_every 4` roughly doubled SALI's epochs).
- Keep `--workers` and `OMP_NUM_THREADS` capped; this is a shared server.
