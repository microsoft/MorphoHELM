# MorphoHELM benchmark — pipeline state, run & resume guide

This document lets anyone (or a new Copilot chat) understand the current state and
run or resume the **full benchmark pipeline** — download → inference → normalize →
benchmark → figures — including reproducing it on another server or adding new splits.
See **Status — 2026-07-27** and **Full pipeline (end-to-end)** for the current state and
exact commands; the sections above them document the download/channel/inference details.

- **Container root:** `/raid/cache/gpznx/data/microsoft_benchmark` (referred to as `$ROOT`)
- **Repo:** `/home/gpznx/projects/MorphoHELM`

## Goal

Evaluate a user-supplied model on the MorphoHELM benchmark using the repo's
evaluation logic and metrics. The user's model produces well-level embeddings
elsewhere, so we are downloading the raw/illumination-corrected images to run
that model over them. HuggingFace conversion (README step 3) is **deferred**.

## Scope

| Dataset | Split(s) | Plates |
|---|---|---|
| BBBC036 | MoA | 406 |
| CPG0016 | cpg-moa | 667 |
| CPG0016 | cpg-crispr | 148 |
| CPG0016 | combined unique | 815 |

- Corrected **uint8 PNGs only** (no HuggingFace Arrow duplication yet).
- Estimated footprint ~6–9 TB. `/raid` had ~10 TB free. Disk guard stops each
  driver when free space drops below 1 TB.

## Layout under `$ROOT`

```
metadata_cache/               JUMP metadata clone + per-plate manifests (= "done" markers)
corrected/cpg0016/<plate>/    corrected PNGs
corrected/bbbc036/<plate>/    corrected PNGs
huggingface/                  (empty, HF deferred)
_download/cpg_plates.txt      815 CPG plate IDs
_download/bbbc036_plates.txt  406 BBBC036 plate IDs
_download/run_batch_download.sh   resumable, disk-guarded driver
_download/logs/{cpg0016,bbbc036}_driver.log
```

## How to resume (after internet / SSH / reboot / disk-guard stop)

Downloads are fully resumable: the driver skips any plate whose manifest exists,
and the downloader skips already-written PNGs within a plate. `nohup` survives
SSH disconnects; a machine reboot kills the processes, so re-run after a reboot.

```bash
ROOT=/raid/cache/gpznx/data/microsoft_benchmark

# 1. Confirm they are NOT already running:
pgrep -af run_batch_download.sh

# 2. If not running, relaunch (idempotent — skips finished plates):
WORKERS=128 nohup bash $ROOT/_download/run_batch_download.sh cpg0016  $ROOT/_download/cpg_plates.txt    1000 >> $ROOT/_download/logs/cpg0016_nohup.log 2>&1 &
WORKERS=128 nohup bash $ROOT/_download/run_batch_download.sh bbbc036 $ROOT/_download/bbbc036_plates.txt 1000 >> $ROOT/_download/logs/bbbc036_nohup.log 2>&1 &
```

A plate that failed mid-download leaves no manifest and is retried on the next run.

## Monitor

```bash
ROOT=/raid/cache/gpznx/data/microsoft_benchmark
tail -f $ROOT/_download/logs/cpg0016_driver.log
ls $ROOT/metadata_cache/cpg0016/manifests | wc -l   # completed of 815
ls $ROOT/metadata_cache/bbbc036/manifests | wc -l   # completed of 406
df -h /raid | tail -1
```

## Split → plate logic

From `feature-extraction/src/run_inference.py` (`_select_plates_from_csv`):

- **cpg-crispr:** `plate.csv.gz` `Metadata_PlateType == "CRISPR"`
- **cpg-moa:** plates listed in `feature-extraction/configs/labeled_moa_samples.csv`
- **bbbc036:** all plate subdirectories (inference excludes plate `25503`)

## Channel mapping & model input (DINOv2_CellPainting, 5-channel)

Target model: **DINOv2 ViT-S/16**, `in_chans=5`, input **224x224**.
Repo: `/home/gpznx/projects/DINOv2_CellPainting`.

Corrected PNGs are **single-channel** grayscale files (mode `L`, uint8 `[0,255]`),
one file per channel. You assemble the 5-channel stack in the **model's TRAINING
channel order**, which differs from MorphoHELM's canonical order.

**Model training order (index 0..4) = `[Mito, AGP, RNA, ER, DNA]`**
(DNA is index 4 = the Otsu / cell-detection channel).

MorphoHELM canonical is `[AGP, DNA, ER, Mito, RNA]`. Remap each dataset to the
model order:

| Model idx | Stain | CPG0016 token | BBBC036 token (`w#` / stain) |
|---|---|---|---|
| 0 | **Mito** | `_ch3` | `w5` · `Mito`        |
| 1 | **AGP**  | `_ch0` | `w4` · `Ph_golgi`    |
| 2 | **RNA**  | `_ch4` | `w3` · `ERSytoBleed` |
| 3 | **ER**   | `_ch2` | `w2` · `ERSyto`      |
| 4 | **DNA**  | `_ch1` | `w1` · `Hoechst`     |

- CPG0016 stack order = `[ch3, ch0, ch4, ch2, ch1]`.
- BBBC036 stack order = `[w5, w4, w3, w2, w1]` (i.e. reverse of `w1..w5`).

**Normalization (per channel, in model order `[Mito,AGP,RNA,ER,DNA]`):**
1. Clip each channel at its **0.01st and 99.9th percentiles**, scale to `[0,1]`.
2. `transforms.Normalize(mean, std)` with:
   - `CP_MEANS = [0.13849893, 0.18710597, 0.1586524, 0.15757588, 0.08674719]`
   - `CP_STDS  = [0.13005716, 0.15461144, 0.15929441, 0.16021383, 0.16686504]`

> Fairness/faithfulness note: MorphoHELM corrected PNGs are uint8. CPG uses a
> per-image percentile rescale (0.05/99.95); BBBC036 uses min-max
> (data_installation/illumination.py). Re-applying the model's
> `scale_intensities(99.9)` reproduces the model's per-channel percentile
> normalization (important for BBBC036's min-max output) and is the SAME uint8
> input every benchmark model receives — using 16-bit would be unfair. The main
> fairness lever is downstream: normalize all models with the identical profile
> (CSAll_Plate__PCA64__MADCtrl_Plate__NoSph) + same 5% cell-count QC.

**Native sizes:** CPG0016 `996x996`; BBBC036 `696x520`. The model does **not**
resize the FOV — at inference each FOV is **tiled into 224x224 crops** (stride 224).

**Inference procedure (matches DINOv2_CellPainting):**
1. Build the 5-channel FOV tensor in model order; percentile-clip+scale; Normalize.
2. Compute the **Otsu threshold on the DNA channel** (index 4). Tile the FOV into
   `224x224` crops; **drop crops** whose DNA-foreground fraction is below the
   training threshold (empty-crop exclusion).
3. Backbone -> per-crop features; **mean over crops -> FOV feature**;
   **mean over FOVs -> WELL feature**.
4. Emit well-level embeddings -> `normalize_splits.py` -> benchmarks.

Minimal per-FOV loader (produces model-order stack):

```python
import glob, re
import numpy as np
from PIL import Image

# CPG0016: index into ch0..ch4 to get model order [Mito,AGP,RNA,ER,DNA]
CPG_MODEL_IDX = [3, 0, 4, 2, 1]
# BBBC036: stain -> model slot
BBBC_STAIN_TO_MODEL = {"Mito": 0, "Ph_golgi": 1, "ERSytoBleed": 2, "ERSyto": 3, "Hoechst": 4}

def load_cpg_fov(plate_dir, well, fov):          # fov like "i1"
    ch = [np.asarray(Image.open(f"{plate_dir}/{well}_{fov}_ch{i}.png")) for i in range(5)]
    return np.stack([ch[i] for i in CPG_MODEL_IDX], axis=0)   # (5,H,W) model order

def load_bbbc036_fov(plate_dir, well, fov):      # lowercase, e.g. "a01","s1"
    slots = [None] * 5
    for p in glob.glob(f"{plate_dir}/*_{well}_{fov}_w*_ch_*.png"):
        stain = re.search(r"_ch_(.+)\.png$", p).group(1)
        slots[BBBC_STAIN_TO_MODEL[stain]] = np.asarray(Image.open(p))
    return np.stack(slots, axis=0)               # (5,H,W) model order
# then: percentile-clip per channel -> [0,1], Normalize(CP_MEANS,CP_STDS),
#       Otsu on channel 4 (DNA), tile 224x224, drop empty crops, backbone, mean.
```

## Environment

- Python: `/home/gpznx/miniforge3/bin/python` (base env); `requirements.txt` installed.
- `torch` not yet verified (only needed for later inference / HF conversion).

## Next steps (after downloads finish)

> The full run is already complete (see **Status — 2026-07-27** and **Full pipeline
> (end-to-end)** below). This section documents the per-stage logic; use the pipeline
> section for the exact commands.

1. Read corrected PNGs directly and build 5-channel FOV tensors in the model's
   training order `[Mito, AGP, RNA, ER, DNA]` (see channel mapping above).
2. Run DINOv2_CellPainting inference via the wrapper (sharded multi-GPU — stage 2).
3. Normalize via `normalize_splits.py`, then run benchmarks + figures (stages 3–6,
   `model_integration/reproduce_benchmarks.sh`).

## DINOv2_CellPainting inference wrapper

Script: `model_integration/dinov2_cellpainting_inference.py`. It is a faithful
port of `DINOv2_CellPainting/inference.py` that reads MorphoHELM corrected PNGs
instead of pre-merged TIFFs. It reuses the repo's own
`source.image_ops.scale_intensities`, `pt_threshold_otsu`,
`inference_utils.forward_inference`, and `aggregate_embeddings_plate`, plus the
backbone/checkpoint loaders from `inference.py`.

Faithful reproduction:
- percentile scaling `scale_intensities(img, 99.9)` (lower bound 0.1 pct);
- `Normalize(CP_MEANS, CP_STDS)` on the crops;
- Otsu on DNA channel (idx 4), tile into 224 crops (stride 224), drop crops below
  `min_area_ratio=0.01` foreground;
- backbone CLS per crop, empty crops -> NaN, pool all crops across all FOVs of a
  well via nanmean (`aggregate_embeddings_plate`).
- Crop generation is made rectangle-safe (the repo's `generate_cellcrops`
  assumes square FOVs and breaks on BBBC036 696x520); byte-identical for square.

Validated crop behavior on real plates: CPG0016 996x996 -> 16 crops (4x4);
BBBC036 520x696 -> 6 crops (2x3); crop shape (5, 224, 224).

Output: `<outdir>/<split>/<model_name>_aggregated.parquet` with columns
`Metadata_Plate, Metadata_Well, feat_0..feat_{D-1}` — a drop-in for
`normalize_splits.py` (which joins control metadata itself on plate+well).

Example (validated with the E07 checkpoint on the b300 env):

```bash
ROOT=/raid/cache/gpznx/data/microsoft_benchmark
PY=/home/gpznx/miniforge3/envs/dinov2_cellpainting_b300/bin/python
SCRIPT=model_integration/dinov2_cellpainting_inference.py
CKPT=/raid/cache/gpznx/dinov2_exp/E07_dinov2_ebtp_mad_teacher_wo_ibot_koleo/model_0249899.pth
CFG=/raid/cache/gpznx/dinov2_exp/E07_dinov2_ebtp_mad_teacher_wo_ibot_koleo/config.yaml

# --- cpg-moa (667 MoA plates only) ---
$PY $SCRIPT --dataset cpg0016 --split cpg-moa \
  --corrected-root $ROOT/corrected/cpg0016 \
  --plates-file $ROOT/_download/cpg_moa_plates.txt \
  --outdir $ROOT/aggregated --ckpt $CKPT --config-file $CFG \
  --size 224 --stride 224 --batch-size 8 --gpus 0

# --- cpg-crispr (148 CRISPR plates only) ---
$PY $SCRIPT --dataset cpg0016 --split cpg-crispr \
  --corrected-root $ROOT/corrected/cpg0016 \
  --plates-file $ROOT/_download/cpg_crispr_plates.txt \
  --outdir $ROOT/aggregated --ckpt $CKPT --config-file $CFG \
  --size 224 --stride 224 --batch-size 8 --gpus 0

# --- bbbc036 (all plates, excludes 25503 like the benchmark) ---
$PY $SCRIPT --dataset bbbc036 --split bbbc036 \
  --corrected-root $ROOT/corrected/bbbc036 --exclude-plates 25503 \
  --outdir $ROOT/aggregated --ckpt $CKPT --config-file $CFG \
  --size 224 --stride 224 --batch-size 8 --gpus 0

# smoke any of the above: add  --max-plates 1 --max-fovs 8
```

Per-split plate lists: `cpg_moa_plates.txt` (667), `cpg_crispr_plates.txt` (148),
`bbbc036_plates.txt` (406). The combined `cpg_plates.txt` (815) is only for the
downloader, not per-split inference. Outputs land at
`$ROOT/aggregated/<split>/dino_v2_cellpainting_aggregated.parquet`.

## Run benchmarks

- `benchmarks/enrichment/run_enrichment_sweep.py`
- `benchmarks/replicate_analysis/run_replicate_analysis_sweep.py`
- config: `configs/benchmarks.yaml`

## Status — 2026-07-27  (PIPELINE COMPLETE — RobuDINO #1 on every benchmark)

All three splits are downloaded, inferred, normalized, benchmarked, and plotted.
Our model `dino_v2_cellpainting` (displayed as **RobuDINO**) ranks **#1** on every
split and every paradigm.

| Split | Download | Wells (aggregated) | Benchmark | RobuDINO geom-OR | Rank |
|---|---|---|---|---|---|
| BBBC036 MoA | 406/406 | 153,007 | MoA enrichment | 11.42 | #1 |
| cpg-crispr | 148/148 | 56,832 | pathway (5 protein DBs), NR / NSB | 13.29 / 12.25 | #1 |
| cpg-moa | 667/667 | 307,579 | MoA, 4 paradigms | Global 12.59 · Within-Source 10.77 · Not-Same-Batch 10.57 · Not-Same-Source 9.90 | #1 |

Artifacts (all under `$ROOT`):
- **aggregated** (well-level, 384-d): `aggregated/<split>/dino_v2_cellpainting_aggregated.parquet`
- **normalized** (64 PC): `normalized/<PROFILE>/<split>/dino_v2_cellpainting_normalized.parquet`
- **results**: `results/{bbbc036_moa,crispr_no_restriction,crispr_not_same_batch,cpg_moa_enrichment}.pkl`,
  `results/moa_profiles/moa_cross_source_profiles.pkl`
- **figures**: `results/graphs/{bbbc036_moa_summary,cpg_crispr_summary,cpg_crispr_per_database,cpg_moa_summary,cpg_moa_enrichment}.png`
- **manifest**: `results/REPRODUCIBILITY.txt` — SHA256 of every input/output + env versions.

`PROFILE = CSAll_Plate__PCA64__MADCtrl_Plate__NoSph`.

Historical: 2026-07-17 downloads started; inference wrapper validated end-to-end on GPU
(E07 ckpt loaded "All keys matched successfully", 384 feats/well, no NaN; BBBC036 non-square path OK).

---

## Full pipeline (end-to-end)

```
raw images ──(1 download)──▶ corrected PNGs + manifests
           ──(2 inference)─▶ per-plate partials ─▶ aggregated/<split>/<model>_aggregated.parquet
           ──(3 normalize)─▶ normalized/<PROFILE>/<split>/<model>_normalized.parquet
           ──(4 align)─────▶ well set matched to baselines, placed next to the 8 baselines
           ──(5 benchmark)─▶ results/*.pkl   (seeded, deterministic)
           ──(6 figures)───▶ results/graphs/*.png + REPRODUCIBILITY.txt
```

Two Python environments are used:
- **base / CPU** — `/home/gpznx/miniforge3/bin/python`: download, normalize, benchmarks, plots.
  (needs `datasets, boto3, scikit-learn, numpy, pandas, scipy, pyarrow, matplotlib`.)
- **GPU inference** — `/home/gpznx/miniforge3/envs/dinov2_cellpainting_b300/bin/python`:
  torch 2.11+cu128, 8 GPUs. Only used for stage 2.

Model checkpoint (E07 DINOv2 ViT-S/16, in_chans=5):
- `CKPT=/raid/cache/gpznx/dinov2_exp/E07_dinov2_ebtp_mad_teacher_wo_ibot_koleo/model_0249899.pth`
- `CFG=/raid/cache/gpznx/dinov2_exp/E07_dinov2_ebtp_mad_teacher_wo_ibot_koleo/config.yaml`

### Stage 1 — download (resumable, disk-guarded)
See "How to resume" above. Corrected PNGs land in `corrected/<dataset>/<plate>/`; a written
per-plate manifest under `metadata_cache/<dataset>/manifests/` is the "done" marker.

### Stage 2 — inference (sharded multi-GPU, resumable)
The wrapper `model_integration/dinov2_cellpainting_inference.py` loops plates in a **single
process** on one GPU by default. For real runs, **shard the plate list across GPUs** — the
per-plate `_partial/<model>/<plate>.parquet` checkpoints make this safe and resumable.

```bash
ROOT=/raid/cache/gpznx/data/microsoft_benchmark
REPO=/home/gpznx/projects/MorphoHELM
PY=/home/gpznx/miniforge3/envs/dinov2_cellpainting_b300/bin/python
CKPT=/raid/cache/gpznx/dinov2_exp/E07_dinov2_ebtp_mad_teacher_wo_ibot_koleo/model_0249899.pth
CFG=/raid/cache/gpznx/dinov2_exp/E07_dinov2_ebtp_mad_teacher_wo_ibot_koleo/config.yaml
SPLIT=cpg-moa                       # cpg-moa | cpg-crispr | bbbc036
DATASET=cpg0016                     # cpg0016 | bbbc036
PLATES=$ROOT/_download/cpg_moa_plates.txt
GPUS="0 1 2"                        # pick the free GPUs

# round-robin shard the plate list, one worker per GPU
mkdir -p "$ROOT/_download/${SPLIT}_shards" "$ROOT/logs/inference"
awk 'NF' "$PLATES" | awk -v n=$(echo $GPUS|wc -w) '{print > ("'"$ROOT"'/_download/'"$SPLIT"'_shards/shard" (NR%n) ".txt")}'
i=0
for g in $GPUS; do
  nohup "$PY" "$REPO/model_integration/dinov2_cellpainting_inference.py" \
    --dataset $DATASET --split $SPLIT \
    --corrected-root "$ROOT/corrected/$DATASET" --outdir "$ROOT/aggregated" \
    --plates-file "$ROOT/_download/${SPLIT}_shards/shard${i}.txt" \
    --ckpt "$CKPT" --config-file "$CFG" \
    --gpus "$g" --num-workers 32 --batch-size 16 \
    > "$ROOT/logs/inference/${SPLIT}.gpu${g}.log" 2>&1 &
  i=$((i+1))
done
# bbbc036 only: add  --exclude-plates 25503   (matches the benchmark)
```

Pipeline is **CPU-bound** on crop generation (Otsu + PIL tiling), so GPU util ~20–25% is
normal; throughput scales with dataloader workers, not GPUs. Monitor:
`ls $ROOT/aggregated/$SPLIT/_partial/dino_v2_cellpainting | wc -l`.

**Finalize-combine (required after sharding).** Each shard writes an *incomplete* final
parquet (only its own plates), so after all shards finish, combine every per-plate partial:

```bash
"$PY" - "$ROOT" "$SPLIT" <<'PYEOF'
import sys, os, glob, pandas as pd
root, split = sys.argv[1], sys.argv[2]
pdir = f"{root}/aggregated/{split}/_partial/dino_v2_cellpainting"
out  = f"{root}/aggregated/{split}/dino_v2_cellpainting_aggregated.parquet"
parts = sorted(glob.glob(f"{pdir}/*.parquet"))
df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
df["Metadata_Plate"] = df["Metadata_Plate"].astype(str)
df["Metadata_Well"]  = df["Metadata_Well"].astype(str)
df = df.drop_duplicates(subset=["Metadata_Plate","Metadata_Well"], keep="first")
df.to_parquet(out + ".tmp", index=False); os.replace(out + ".tmp", out)
print(f"{split}: {len(parts)} plates -> {len(df)} wells")
PYEOF
```
(base-env `PY` is fine for the combine; it is pure pandas.)

### Stages 3–6 — normalize, align, benchmark, figures (one command)
Once the aggregated parquet exists for a split, everything else is reproducible via the
driver script (deterministic: PCA pinned `svd_solver=full, random_state=0`; enrichment
`RANDOM_STATE=61`, `n_resamples=100`):

```bash
cd /home/gpznx/projects/MorphoHELM
DATA_ROOT=/raid/cache/gpznx/data/microsoft_benchmark \
PY=/home/gpznx/miniforge3/bin/python \
bash model_integration/reproduce_benchmarks.sh
```

What it does (see `model_integration/reproduce_benchmarks.sh`):
1. `data-preprocessing/src/normalize_splits.py` — normalize RobuDINO for all splits
   (`--config $DATA_ROOT/configs/normalize_run.yaml --profile $PROFILE`).
2. `model_integration/prepare_for_benchmark.py` — align our well set to the baseline
   (intersect on Plate+Well; for bbbc036 also drop `Metadata_broad_sample/moa/target`),
   then copy our parquet next to the 8 baselines in
   `data/main_paper_inputs/normalized/$PROFILE/<split>/`.
3. Enrichment:
   - `benchmarks/enrichment/crispr/run_crispr_enrichment.py` (modes `no_restriction`, `not_same_batch`),
   - `benchmarks/enrichment/bbbc036_moa/run_bbbc036_moa.py`,
   - `benchmarks/enrichment/cpg_moa/build_moa_profiles.py` then `run_cpg_moa.py`
     (`--bbbc-results …/bbbc036_moa.pkl` so the combined heatmap's BBBC036 column is populated).
4. `model_integration/plot_benchmark_results.py` — publication figures (RobuDINO highlighted,
   YlOrRd, colorbars). The cpg-moa figure is a combined **BBBC036-MoA + 4 cpg-moa** heatmap.

QC is identical for every model: `--cell-count-qc --qc-cell-count-cutoff 0.05
--qc-sample-wells 5000 --qc-seed 42` against `data/qc/cell_counts/`.

---

## Running on a fresh server (or a new checkout)

Everything is path-driven, so only a few things must exist on the new machine:

1. **Repo + git-lfs assets.** Clone MorphoHELM, then pull the LFS payloads (baseline
   normalized parquets for all 8 models × splits, `data/qc/cell_counts/`,
   `data/enrichment/{moa_databases,protein_databases}`, `data/main_paper_inputs/metadata/`,
   MoA labels):
   ```bash
   git lfs install && git lfs pull
   ```
2. **Data container** `$ROOT` with enough space (~6–9 TB for corrected PNGs). Set it once:
   ```bash
   export ROOT=/path/on/new/server/microsoft_benchmark
   ```
3. **Two Python envs** (base CPU + GPU inference) — recreate from `requirements.txt` and the
   DINOv2_CellPainting inference env (torch matching the GPUs).
4. **Model checkpoint + config** (`CKPT`, `CFG`) copied to the new server; update the paths in
   the stage-2 commands (and in `model_integration/reproduce_benchmarks.sh` defaults if desired).
5. **`$ROOT/configs/normalize_run.yaml`** — the only run config. Point every root at the new
   `$ROOT` (`aggregated_root`, `normalized_root`, `metadata_root`) and set each split's
   `control_column/value` + `combined_metadata_path` (CPG) / `per_plate_metadata_glob` (BBBC036).
6. **CPG combined metadata** `metadata_cache/cpg0016/metadata/metadata.parquet` — built from the
   JUMP `well.csv.gz` + `plate.csv.gz` (+ `crispr.csv.gz` for `Metadata_Symbol`). Columns:
   `Metadata_Source/Plate/Well/JCP2022/Batch/PlateType/Symbol`. Controls: moa `JCP2022_033924`,
   crispr `JCP2022_800001`.

Then run stages 1→2 (download + sharded inference, GPU) and 3→6 (`reproduce_benchmarks.sh`, CPU).

Environment note: benchmark values are a **reproduction** of the paper (baselines differ slightly
because odds ratios use a Haldane–Anscombe zero-cell correction and embeddings were regenerated).
The pipeline is deterministic, so a rerun on the same inputs reproduces the manifest hashes.

---

## Adding another split / dataset

1. **Build the plate list** for the split (`$ROOT/_download/<split>_plates.txt`, one plate ID per
   line) using the split→plate logic (see "Split → plate logic" above) — e.g. filter
   `plate.csv.gz` by `Metadata_PlateType`, or use a labelled-sample CSV.
2. **Download** it: append the plates to the driver's plate list and rerun the resumable driver
   (it skips already-downloaded plates):
   ```bash
   WORKERS=128 nohup bash $ROOT/_download/run_batch_download.sh <dataset> <plate-list> 1000 \
     >> $ROOT/_download/logs/<dataset>_nohup.log 2>&1 &
   ```
3. **Inference** — run the stage-2 sharded recipe with `--split <split> --dataset <dataset>` and
   the split's plate list, then finalize-combine.
4. **Register the split** in `$ROOT/configs/normalize_run.yaml` under `splits:` with its
   `control_column`, `control_value`, `plate_column`, and metadata path.
5. **Benchmark**: pick the matching enrichment runner
   (`crispr/`, `bbbc036_moa/`, or `cpg_moa/`) — or add the split to
   `model_integration/reproduce_benchmarks.sh` if it fits an existing paradigm. For a brand-new
   metric, add a runner under `benchmarks/enrichment/<name>/` and a `plot_*` in
   `model_integration/plot_benchmark_results.py`.
6. **Baselines**: to rank against the 8 baselines, the split needs baseline normalized parquets in
   `data/main_paper_inputs/normalized/$PROFILE/<split>/`; `prepare_for_benchmark.py` then aligns
   our well set to them. Without baselines you can still score RobuDINO alone.

---



## BBBC036 corrupt-archive recovery — 2026-07-21

- Plates `25695` and `26060` repeatedly failed with `zlib.error: invalid literal/length code`.
- **Verified** the GigaDB illumination `.tar.gz` for both is genuinely corrupt: full download matches
  the server `Content-Length` exactly, yet `gzip -t` fails (`invalid compressed data--format violated`).
  Corruption is partway through, **after** the illumination-correction `.mat` files.
- Recovery: partial `tar --ignore-zeros` extract recovers all 5 illum `.mat` (valid `(520,696)`)
  + `profiles/mean_well_profiles.csv` (both before the corruption point). Staged at
  `$ROOT/_download/bbbc036_recovered_illum/gigascience_upload/Plate_<plate>/...`.
- Re-ran downloader with `--bbbc036-illum-root $ROOT/_download/bbbc036_recovered_illum` to bypass the
  corrupt archive (raw images come from UCSD, a separate source). Both plates completed:
  manifest 11521 rows, 11520 corrected PNGs, 5 illum npy each. **BBBC036 now 406/406 complete.**
- Recovery log: `$ROOT/_download/logs/bbbc036_recovery.log`.
- To recover again if cache is cleared:
  ```bash
  ROOT=/raid/cache/gpznx/data/microsoft_benchmark
  python data_installation/download_illum_corrected.py --dataset bbbc036 \
    --metadata-root $ROOT/metadata_cache --bbbc036-output-root $ROOT/corrected/bbbc036 \
    --bbbc036-illum-root $ROOT/_download/bbbc036_recovered_illum \
    --plates 25695 --plates 26060 --workers 128 --validate
  ```
