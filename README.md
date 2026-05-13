# MorphoHELM Cell Painting benchmark pipeline

This repository contains the MorphoHELM Cell Painting benchmark pipeline:

```text
Data download -> Preprocess -> Turn into HF -> Inference -> Aggregate -> Normalize -> QC -> Run benchmarks -> Generate graphs
```

In this repository, **Preprocess** currently means illumination correction plus
uint8 PNG materialization.

The maintained entrypoints, configs, checkpoints, support metadata, and current
main-paper normalized embeddings are included here.

## 1. Data download and installation

The installation stage downloads public raw images, downloads/caches benchmark
metadata, applies illumination correction immediately, and saves full-size uint8
PNGs in plate folders. These corrected image folders are the input to
`raw_to_huggingface/transform_dataset.py`.

### 1.1 Outputs

Use separate roots for CPG0016 and BBBC036:

```text
<cpg-output-root>/
  <CPG plate>/
    A01_i1_ch0.png
    A01_i1_ch1.png
    ...

<bbbc036-output-root>/
  <BBBC036 plate>/
    cdp2bioactives_a01_s1_w..._ch_Hoechst.png
    cdp2bioactives_a01_s1_w..._ch_ERSyto.png
    ...
```

CPG0016 uses full-size corrected images only. This matches the benchmark path
that uses the `cellpaintingcold` layout; center-cropped `cellpaintinghot` images
are not used.

All saved images are uint8 PNGs. The source image files are not modified.

### 1.2 Main command

The single user-facing downloader is:

```bash
python data_installation/download_illum_corrected.py \
  --dataset cpg0016 \
  --metadata-root /path/to/metadata_cache \
  --cpg-output-root /path/to/cpg0016_corrected \
  --plates 1053597837 \
  --workers 16 \
  --validate
```

For BBBC036:

```bash
python data_installation/download_illum_corrected.py \
  --dataset bbbc036 \
  --metadata-root /path/to/metadata_cache \
  --bbbc036-output-root /path/to/bbbc036_corrected \
  --plates 24277 \
  --workers 16 \
  --validate
```

Run both datasets by using `--dataset all` and passing both output roots.

### 1.3 Common options

| Option | Meaning |
|---|---|
| `--dataset cpg0016\|bbbc036\|all` | Dataset to install. |
| `--metadata-root` | Cache for cloned metadata, illumination files, manifests, and temporary archives. |
| `--cpg-output-root` | Corrected CPG0016 plate-folder root. Required for CPG0016. |
| `--bbbc036-output-root` | Corrected BBBC036 plate-folder root. Required for BBBC036. |
| `--cpg-metadata-repo` | Existing `jump-cellpainting/datasets` checkout. If omitted, the installer clones it into the metadata cache. |
| `--bbbc036-ground-truth` | Existing `BBBC036_v1_DatasetGroundTruth.csv`. If omitted, the installer downloads it from BBBC. |
| `--plates` | Plate IDs to process. Can be repeated or comma-separated. If omitted, all known plates are selected. |
| `--plates-file` | File with one plate ID per line. |
| `--max-plates` | Smoke-test limit on selected plates. |
| `--workers` | Number of parallel workers. |
| `--overwrite` | Regenerate outputs that already exist. Without this, existing outputs are reused. |
| `--validate` | Decode a saved PNG and confirm it is uint8. |

Smoke-test options:

| Option | Dataset | Meaning |
|---|---|---|
| `--max-wells` | CPG0016 | Limit number of wells per plate. |
| `--max-images-per-channel` | BBBC036 | Limit number of images processed from each raw channel ZIP or local channel folder. |

### 1.4 CPG0016 details

The CPG0016 installer uses the public JUMP metadata repository:

```text
https://github.com/jump-cellpainting/datasets.git
```

If `--cpg-metadata-repo` is not supplied, the script clones this repository under
`<metadata-root>/cpg0016/datasets/`. The benchmark-relevant public metadata is
also copied into:

```text
<metadata-root>/cpg0016/metadata/
```

Cached files include:

```text
plate.csv.gz
well.csv.gz
compound.csv.gz
compound_source.csv.gz
orf.csv.gz
crispr.csv.gz
microscope_config.csv
microscope_filter.csv
cellprofiler_version.csv
```

For each selected plate, it loads the public Cell Painting Gallery
`load_data_with_illum` file, downloads the five raw TIFF channels and five
illumination arrays, applies correction, rescales to uint8, and writes:

```text
<cpg-output-root>/<plate>/<well>_i<site>_ch0.png  # AGP
<cpg-output-root>/<plate>/<well>_i<site>_ch1.png  # DNA
<cpg-output-root>/<plate>/<well>_i<site>_ch2.png  # ER
<cpg-output-root>/<plate>/<well>_i<site>_ch3.png  # Mito
<cpg-output-root>/<plate>/<well>_i<site>_ch4.png  # RNA
```

It also caches the full per-plate `load_data_with_illum` table under:

```text
<metadata-root>/cpg0016/load_data/<plate>_load_data_with_illum.csv
```

Example smoke test using an existing metadata checkout:

```bash
python data_installation/download_illum_corrected.py \
  --dataset cpg0016 \
  --metadata-root /tmp/cellpainting_install_meta \
  --cpg-output-root /tmp/cpg0016_corrected \
  --cpg-metadata-repo /path/to/jump-cellpainting-datasets \
  --plates 1053597837 \
  --max-wells 1 \
  --workers 2 \
  --validate
```

### 1.5 BBBC036 details

The BBBC036 installer uses:

- raw channel ZIPs from:
  `http://cildata.crbs.ucsd.edu/broad_data/plate_<plate>/<plate>-<channel>.zip`
- illumination metadata tarballs from:
  `https://s3.ap-northeast-1.wasabisys.com/gigadb-datasets/live/pub/10.5524/100001_101000/100351/Plate_<plate>.tar.gz`
- MoA ground truth from:
  `https://data.broadinstitute.org/bbbc/BBBC036/BBBC036_v1_DatasetGroundTruth.csv`

Channels:

```text
Hoechst, ERSyto, ERSytoBleed, Ph_golgi, Mito
```

Illumination mapping:

```text
Hoechst -> IllumDNA
ERSyto -> IllumER
ERSytoBleed -> IllumRNA
Ph_golgi -> IllumAGP
Mito -> IllumMito
```

By default, temporary BBBC036 raw ZIPs and illumination tarballs are deleted
after successful correction. Use `--keep-archives` to retain them.

The installer caches benchmark metadata under:

```text
<metadata-root>/bbbc036/metadata/BBBC036_v1_DatasetGroundTruth.csv
<metadata-root>/bbbc036/metadata/per_plate/<plate>_metadata.parquet
<metadata-root>/bbbc036/metadata/per_plate/<plate>_qc.csv
```

`<plate>_metadata.parquet` is extracted from the plate `mean_well_profiles.csv`
metadata columns and merged with the BBBC036 ground truth so future benchmark
steps have `Metadata_broad_sample`, `Metadata_moa`, and `Metadata_target`.

Example public-download command for one plate. Even with
`--max-images-per-channel`, this still downloads the plate-level illumination
archive and selected raw channel ZIPs unless you provide local cache roots.

```bash
python data_installation/download_illum_corrected.py \
  --dataset bbbc036 \
  --metadata-root /tmp/cellpainting_install_meta \
  --bbbc036-output-root /tmp/bbbc036_corrected \
  --plates 24277 \
  --max-images-per-channel 1 \
  --workers 2 \
  --validate
```

If raw TIFFs and GigaDB illumination files have already been extracted locally,
reuse them without downloading the large archives:

```bash
python data_installation/download_illum_corrected.py \
  --dataset bbbc036 \
  --metadata-root /tmp/cellpainting_install_meta \
  --bbbc036-output-root /tmp/bbbc036_corrected \
  --bbbc036-illum-root /path/to/bbbc036_metadata_unzipped/gigascience_upload \
  --bbbc036-raw-unzipped-root /path/to/bbbc036_unzipped \
  --bbbc036-ground-truth /path/to/BBBC036_v1_DatasetGroundTruth.csv \
  --plates 24277 \
  --max-images-per-channel 1 \
  --workers 2 \
  --validate
```

### 1.6 Manifests and resumability

Each processed plate writes a manifest under:

```text
<metadata-root>/cpg0016/manifests/<plate>.csv
<metadata-root>/bbbc036/manifests/<plate>.csv
```

If corrected PNGs already exist, rerunning the command skips them unless
`--overwrite` is passed. This makes interrupted runs resumable.

### 1.7 Convert corrected images to HuggingFace

After installation, convert plate folders to HuggingFace datasets:

```bash
python raw_to_huggingface/transform_dataset.py \
  --dataset cpg0016 \
  --raw-root /path/to/cpg0016_corrected \
  --output-root /path/to/cpg0016_huggingface \
  --plate 1053597837 \
  --validate
```

```bash
python raw_to_huggingface/transform_dataset.py \
  --dataset bbbc036 \
  --raw-root /path/to/bbbc036_corrected \
  --output-root /path/to/bbbc036_huggingface \
  --plate 24277 \
  --validate
```

## 2. Preprocess

Preprocess currently refers to the illumination-correction work performed during
data installation:

```text
raw public image -> illumination correction -> intensity rescale -> uint8 full-size PNG
```

The implementation lives in `data_installation/illumination.py` and is called by
`data_installation/download_illum_corrected.py`.

## 3. Turn into HF

The current raw/corrected image to HuggingFace converter is:

```text
raw_to_huggingface/transform_dataset.py
```

By default, the converter looks for bounding boxes under:

```text
data/bounding_boxes/bounding_boxes_cpg
data/bounding_boxes/bounding_boxes_bbbc036
```

Those full bounding-box trees are large and are not required for the normalized
embedding graph workflow. If they live elsewhere, pass `--bbox-root` explicitly.

## 4. Inference

The inference stage consumes the per-plate HuggingFace datasets created by
`raw_to_huggingface/transform_dataset.py` and writes per-model feature pickles.
The model wrappers, preprocessing transforms, dataloader behavior, GPU spawning,
and output schema remain in `feature-extraction/src/run_inference.py`.

### 4.1 Streamlined HuggingFace inference

Use the multi-split wrapper when running one or more HuggingFace-backed splits
consecutively:

```bash
python feature-extraction/src/run_inference_splits.py \
  --config feature-extraction/configs/inference_splits.yaml \
  --splits cpg-tgt2,cpg-crispr,bbbc036
```

Supported split names:

```text
cpg-crispr
cpg-tgt2
cpg-compound
cpg-moa
bbbc036
```

Run only one model across all requested splits:

```bash
python feature-extraction/src/run_inference_splits.py \
  --config feature-extraction/configs/inference_splits.yaml \
  --splits cpg-tgt2 cpg-crispr \
  --model dino_v2
```

Dry-run the split expansion without requiring GPUs:

```bash
python feature-extraction/src/run_inference_splits.py \
  --config feature-extraction/configs/inference_splits.yaml \
  --splits cpg-tgt2,bbbc036 \
  --max-plates 1 \
  --dry-run
```

By default, each split writes to its own subdirectory under the configured
output root:

```text
<output-root>/cpg-tgt2/<model_name>/
<output-root>/cpg-crispr/<model_name>/
<output-root>/cpg-compound/<model_name>/
<output-root>/cpg-moa/<model_name>/
<output-root>/bbbc036/<model_name>/
```

The wrapper stops on the first split failure by default. Use
`--continue-on-error` only when you explicitly want later splits to continue
after a failed split.

### 4.2 Single-split HuggingFace inference

The original single-config entrypoint is still available:

```bash
python feature-extraction/src/run_inference.py \
  --config feature-extraction/configs/inference_config.yaml
```

It now also supports `--dry-run` and `--max-plates` for quick validation:

```bash
python feature-extraction/src/run_inference.py \
  --config feature-extraction/configs/inference_config.yaml \
  --max-plates 1 \
  --dry-run
```

### 4.3 Direct-mode cpg-MoA inference

cpg-MoA has a dedicated direct-image inference path for efficiency. Keep using
it when you want the existing MOA-specific filtering and direct image loading:

```bash
python feature-extraction/src/run_inference_direct.py \
  --config feature-extraction/configs/direct_moa_config.yaml
```

This direct path is intentionally documented separately because it does not use
the generic per-plate HuggingFace split wrapper.

### 4.4 Model details, preprocessing, and checkpoints

The streamlined inference wrapper keeps the existing model-specific logic in
`feature-extraction/src/config.py` and `feature-extraction/src/models/`. The
new pipeline installs and converts images as uint8 PNGs, so the streamlined
`inference_splits.yaml` uses `bit_depth: 8` for both CPG0016 and BBBC036.
Legacy configs that point at older 16-bit BBBC036 data can still use
`bit_depth: 16`, which activates the existing 16-bit to 8-bit thresholding path.

| Model key | Source / weights | Input preprocessing in this repo | Output |
|---|---|---|---|
| `dino_v2` | `torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")` | Single channel -> `Resize(256)` -> `CenterCrop(224)` -> divide by 255 -> repeat to RGB -> ImageNet normalization. | CLS token features and mean patch-token features. |
| `dino_v2_448` | Same DINOv2 ViT-B/14 weights as `dino_v2`; separate key/output folder for the 448 path. | Single channel -> `Resize(512)` -> `CenterCrop(448)` -> divide by 255 -> repeat to RGB -> ImageNet normalization. | CLS token features and mean patch-token features. |
| `resnet` | `torchvision.models.resnet101(weights=ResNet101_Weights.DEFAULT)` | Single channel -> `Resize(256)` -> `CenterCrop(224)` -> divide by 255 -> repeat to RGB -> ImageNet normalization. Final FC is replaced with identity. | 2048-d penultimate features. |
| `resnet_untrained` | `torchvision.models.resnet101(weights=None)` | Same preprocessing and architecture as `resnet`, but random weights. | 2048-d baseline features. |
| `vgg19` | `torchvision.models.vgg19(weights=VGG19_Weights.DEFAULT)` | Same classic ImageNet preprocessing as `resnet`. Final classifier layer is replaced with identity. | 4096-d classifier features. |
| `open_phenom` | `MAEModel.from_pretrained("recursionpharma/OpenPhenom")`; vendored model code under `feature-extraction/src/models/openphenom/`. | Channels are grouped into 5-channel FOV tensors, resized to `256x256`, and kept in 0-255 range for the model's internal normalization. | 384-d OpenPhenom features. |
| `cloome` | Vendored CLOOME code/config under `feature-extraction/src/models/cloome/`; checkpoint supplied by config. | Channels are grouped into 5-channel FOV tensors, reordered from `[AGP,DNA,ER,Mito,RNA]` to `[Mito,ER,RNA,AGP,DNA]`, center-cropped, and normalized with CLOOME channel statistics. HF mode uses `CenterCrop(996)` for CPG0016 and `CenterCrop(520)` for BBBC036; direct mode uses `CenterCrop(1024)` for cpg-MoA direct images. | 512-d L2-normalized embeddings. |
| `subcell` | Vendored SubCell-compatible wrapper; encoder checkpoint supplied by config. | Channels are grouped into 5-channel FOV tensors and resized to `448x448`. The wrapper runs the ER-DNA-Protein model three times, using ER and DNA as reference channels and AGP, Mito, RNA as the profiling channel. Each 3-channel pass is min-max normalized. | 4608-d concatenated embeddings. |

Checkpoint files currently expected under `feature-extraction/checkpoints/`:

| File | Used by | Notes | SHA256 |
|---|---|---|---|
| `cloome-retrieval-zero-shot.pt` | `cloome` | Default checkpoint in the streamlined config; retrieval/MoA-oriented. | `39e0c98d47b18ce913f4bcb1a1bc89d26ca9938ee74646c32a46ad236cddbc38` |
| `cloome-bioactivity.pt` | `cloome` | Alternative CLOOME checkpoint retained for bioactivity-style runs. | `359de652c2cd76bb189c37e006d565317b93080d89bd2b711deea102cd8e103a` |
| `ER-DNA-Protein_MAE-CellS-ProtS-Pool.pth` | `subcell` | SubCell ER-DNA-Protein encoder and attention-pooler weights. | `ed8c22b42e1b0665b2363e7a1ae56b42a8a93ac93203bb89a13b6be7ad878e51` |

Default batch sizes in `inference_splits.yaml`:

| Model key | Batch size | Notes |
|---|---:|---|
| `dino_v2` | 1024 | Per single-channel image. |
| `dino_v2_448` | 512 | Larger crop, lower default batch size. |
| `resnet` | 1024 | Per single-channel image. |
| `resnet_untrained` | 1024 | Mixed precision disabled by default for this baseline. |
| `vgg19` | 512 | Disabled by default in the streamlined config. |
| `open_phenom` | 1280 | Must be compatible with 5-channel grouping. |
| `cloome` | 15 | Must be a multiple of 5 in HF mode. |
| `subcell` | 60 | Must be a multiple of 5 in HF mode. |

The local validation environment used for this inference refactor reported:

```text
torch==2.9.0+cu128
torchvision==0.24.0+cu128
timm==1.0.26
transformers==4.57.6
datasets==4.4.1
```

Update this section if the image-preprocessing logic, model source, or checkpoint
file changes.

### 4.5 Inference validation status

The streamlined inference code has been validated at smoke-test scale:

- Python compilation passed for `run_inference.py` and
  `run_inference_splits.py`.
- `run_inference_splits.py --dry-run` resolved consecutive CPG and BBBC036
  split execution and split-specific output paths.
- `run_inference.py --dry-run` resolved the original single-split path.
- A tiny GPU smoke test ran `resnet_untrained` on a synthetic one-plate
  BBBC036-style HuggingFace dataset and wrote a feature pickle.

A full all-model/all-plate benchmark inference run has not been executed in
this documentation pass.

## 5. Aggregate

Aggregation converts per-GPU inference pickles into one well-level parquet per
model or feature variant. This stage intentionally hides whether inference came
from the HuggingFace path or the direct cpg-MoA path: downstream normalization
always reads the same schema.

Input layout:

```text
<inference-output-root>/<split>/<model_name>/gpu_*_results/results_part_*.pkl
```

Output layout:

```text
<aggregated-root>/<split>/<model_or_variant>_aggregated.parquet
<aggregated-root>/<split>/aggregation_manifest.json
```

Each parquet has one row per `(Metadata_Plate, Metadata_Well)`, metadata columns
first, and numeric feature columns after metadata.

Run aggregation for one or more splits:

```bash
python feature-extraction/src/aggregate_splits.py \
  --config feature-extraction/configs/inference_splits.yaml \
  --splits cpg-tgt2,bbbc036 \
  --model all \
  --num-workers 4
```

The paper model set currently produces:

```text
dino_v2_cls_token_aggregated.parquet
dino_v2_patch_token_aggregated.parquet
resnet_aggregated.parquet
resnet_untrained_aggregated.parquet
open_phenom_aggregated.parquet
cloome_aggregated.parquet
subcell_aggregated.parquet
```

`dino_v2_aggregated.parquet` is also written as a canonical alias of the CLS
token output for compatibility.

The aggregator validates required metadata columns, duplicate well rows, numeric
feature columns, and all-NaN feature columns. It fails fast by default; use
`--continue-on-error` only when intentionally collecting partial results.

## 6. Normalize

Normalization consumes aggregated well-level parquets and writes benchmark-ready
normalized features under a named profile.

The normalization stage is configured through:

```text
data-preprocessing/configs/normalize_splits.yaml
```

Main production profile:

```text
CSAll_Plate__PCA64__MADCtrl_Plate__NoSph
```

This means:

```text
CenterScale(all wells, per plate) -> PCA(64) -> MAD robustize(controls, per plate) -> no spherization
```

Additional paper/appendix profiles are also defined in the config:

| Profile | Meaning |
|---|---|
| `CSAll_Plate__PCA64__MADCtrl_Plate__NoSph` | Main profile. |
| `CSAll_Plate__PCA64__MADCtrl_Plate__SphCtrl_Batch` | Main profile plus control-fitted batch sphering. |
| `NoCS__PCA64__MADCtrl_Plate__NoSph` | PCA64 + plate MAD without CenterScale. |
| `CSAll_Plate__NoPCA__MADCtrl_Plate__NoSph` | Plate CenterScale + plate MAD without PCA. |
| `CSAll_Plate__PCA8__MADCtrl_Plate__NoSph` | Small smoke profile. |

Run normalization:

```bash
python data-preprocessing/src/normalize_splits.py \
  --config data-preprocessing/configs/normalize_splits.yaml \
  --profile CSAll_Plate__PCA64__MADCtrl_Plate__NoSph \
  --splits cpg-tgt2,bbbc036 \
  --model all
```

Output layout:

```text
<normalized-root>/CSAll_Plate__PCA64__MADCtrl_Plate__NoSph/<split>/<model_or_variant>_normalized.parquet
<normalized-root>/CSAll_Plate__PCA64__MADCtrl_Plate__NoSph/normalization_manifest.json
```

Control definitions:

| Split | Control column | Control value |
|---|---|---|
| `bbbc036` | `Metadata_ASSAY_WELL_ROLE` | `mock` |
| `cpg-crispr` | `Metadata_JCP2022` | `JCP2022_800001` |
| `cpg-tgt2` | `Metadata_JCP2022` | `JCP2022_033924` |
| `cpg-compound` | `Metadata_JCP2022` | `JCP2022_033924` |
| `cpg-moa` | `Metadata_JCP2022` | `JCP2022_033924` |

For small validation runs, use:

```text
CSAll_Plate__PCA8__MADCtrl_Plate__NoSph
```

The normalizer validates profile names, scopes, metadata joins, control
presence, duplicate wells, and finite numeric outputs. It writes a manifest with
the resolved profile config, input/output files, row counts, control counts,
plate counts, and output feature dimensions. No-PCA profiles use `F_*` feature
columns; PCA profiles use `PC_*`.

### 6.1 Raw-link-to-normalized smoke

The planned end-to-end smoke uses BBBC036 plate `24277` and the paper DL model
set (`dino_v2`, `resnet`, `resnet_untrained`, `open_phenom`, `cloome`,
`subcell`):

```bash
python data_installation/download_illum_corrected.py \
  --dataset bbbc036 \
  --metadata-root /tmp/cellpainting_smoke/metadata \
  --bbbc036-output-root /tmp/cellpainting_smoke/bbbc036_corrected \
  --plates 24277 \
  --workers 8 \
  --validate

python raw_to_huggingface/transform_dataset.py \
  --dataset bbbc036 \
  --raw-root /tmp/cellpainting_smoke/bbbc036_corrected \
  --output-root /tmp/cellpainting_smoke/bbbc036_huggingface \
  --plate 24277 \
  --validate

python feature-extraction/src/run_inference_splits.py \
  --config /tmp/cellpainting_smoke/inference_splits_smoke.yaml \
  --splits bbbc036

python feature-extraction/src/aggregate_splits.py \
  --config /tmp/cellpainting_smoke/inference_splits_smoke.yaml \
  --splits bbbc036 \
  --model all \
  --num-workers 4

python data-preprocessing/src/normalize_splits.py \
  --config /tmp/cellpainting_smoke/normalize_splits_smoke.yaml \
  --profile CSAll_Plate__PCA8__MADCtrl_Plate__NoSph \
  --splits bbbc036 \
  --model all
```

Validation note: the public BBBC036 CIL raw ZIPs are about 1 GB per channel and
timed out during the smoke run from this environment before writing bytes. The
completed validation used the existing extracted raw TIFF cache with the same
illumination-correction, HF conversion, paper-model inference, aggregation, and
normalization code paths. It processed 600 corrected PNGs from plate `24277`
(`--max-images-per-channel 120`), produced 20 well-level aggregate rows per
model/variant, and wrote normalized PCA8 outputs for all paper model variants.

## 7. QC

The benchmark figures currently use 5% well-level cell-count QC. The shared QC
helpers live in:

```text
benchmarks/qc/cell_count_qc.py
```

Most benchmark scripts accept the same options through `add_cell_count_qc_args`:

```bash
--cell-count-qc \
--qc-cell-counts-dir data/qc/cell_counts \
--qc-cell-count-cutoff 0.05 \
--qc-sample-wells 5000 \
--qc-seed 42
```

The graph-generation wrapper below passes this QC configuration automatically.

## 8. Run benchmarks

The paper benchmarks consume normalized embeddings from:

```text
data/main_paper_inputs/normalized/CSAll_Plate__PCA64__MADCtrl_Plate__NoSph/
```

Expected normalized input splits:

```text
bbbc036/
cpg-moa/
cpg-crispr/
cpg-tgt2/
cpg-compound/
```

Each split should contain the eight benchmark model parquets:

```text
cellprofiler_normalized.parquet
cloome_normalized.parquet
dino_v2_cls_token_normalized.parquet
dino_v2_patch_token_normalized.parquet
open_phenom_normalized.parquet
resnet_normalized.parquet
resnet_untrained_normalized.parquet
subcell_normalized.parquet
```

Benchmarks and their current scripts:

| Paper output | Dataset(s) | Metrics / restrictions | Script(s) |
|---|---|---|---|
| MoA enrichment heatmap | `bbbc036`, `cpg-moa` | fraction significant and geometric mean OR; BBBC036 NR; cpg-MoA NR, NSB, NSS | `benchmarks/run_bbbc036_enrichment_sweep.py`, `benchmarks/enrichment/moa/build_moa_profiles.py`, `benchmarks/run_moa_sweep.py` |
| CRISPR pathway enrichment heatmap | `cpg-crispr` | fraction significant and geometric mean OR; NR and NSB; databases CORUM, HuMAP, REACTOME, SIGNOR, StringDB | `benchmarks/enrichment/crispr/run_crispr_enrichment.py` |
| Per-database CRISPR bar plot | `cpg-crispr` | same CRISPR metrics, shown separately per database | `paper/Cellprofiling_Benchmark/scripts/plot_crispr_database_barplot.py` |
| kNN replicate retrieval heatmap | `cpg-tgt2`, `cpg-compound` | Recall@1 and mAP; NR, NSB, NSS, NSL | `benchmarks/run_knn_sweep.py` |
| Negative-control mAP appendix heatmap | `cpg-tgt2`, `cpg-compound` | DMSO distractors plus same-compound positives; NR, NSB, NSS, NSL | `benchmarks/run_negative_control_map.py` |
| Normalization/post-processing appendix heatmaps | same benchmark datasets | MoA, CRISPR, and kNN heatmaps for three additional normalization profiles | `paper/Cellprofiling_Benchmark/scripts/generate_paper_graphs.py`, `paper/Cellprofiling_Benchmark/scripts/export_qc_heatmap_dat.py` |
| Discovery-set Jaccard appendix heatmaps | `bbbc036`, `cpg-moa`, `cpg-crispr` | pairwise Jaccard over least-restrictive significant compounds/genes | `paper/Cellprofiling_Benchmark/scripts/export_jaccard_dat.py` |
| Haldane-Anscombe OR diagnostic | BBBC036 MoA result pickles | old imputation vs corrected OR distribution | `benchmarks/enrichment/moa/plot_moa_pooled_or_skewness.py` |

The discrimination-score figure block is commented out in
`paper/Cellprofiling_Benchmark/main.tex`; it is not part of the active paper
figure set.

## 9. Generate graphs

Paper graph regeneration is configured through:

```text
paper/Cellprofiling_Benchmark/configs/paper_graphs.yaml
```

Use the main wrapper to regenerate benchmark result pickles and graph inputs from
the normalized embeddings for the main profile:

```bash
bash paper/Cellprofiling_Benchmark/scripts/generate_main_paper_graphs.sh
```

The shell wrapper delegates to the config-driven Python entrypoint:

```bash
python paper/Cellprofiling_Benchmark/scripts/generate_paper_graphs.py \
  --config paper/Cellprofiling_Benchmark/configs/paper_graphs.yaml \
  --profiles main
```

Regenerate all configured main + appendix normalization profiles:

```bash
python paper/Cellprofiling_Benchmark/scripts/generate_paper_graphs.py --profiles all
```

If normalized embeddings do not already exist for the selected profile(s), run
normalization first from aggregated parquets:

```bash
python paper/Cellprofiling_Benchmark/scripts/generate_paper_graphs.py \
  --profiles all \
  --run-normalization
```

The wrapper runs:

1. cpg-MoA profile construction;
2. BBBC036 MoA enrichment with permutation-style p-values;
3. cpg-MoA enrichment;
4. cpg-CRISPR enrichment for NR and NSB;
5. kNN/mAP retrieval for cpg-target2 and cpg-compound;
6. cpg-compound all-eligible kNN selection;
7. negative-control mAP;
8. heatmap `.dat` export;
9. discovery-set Jaccard `.dat` export;
10. per-database CRISPR bar-plot rendering.
11. the Haldane-Anscombe BBBC036 OR diagnostic when its comparison pickles are
    present under `results/moa_enrichment/`.

To only regenerate graph inputs from existing benchmark result pickles:

```bash
python paper/Cellprofiling_Benchmark/scripts/generate_paper_graphs.py \
  --profiles all \
  --only-export-graphs
```

Preview commands without executing them:

```bash
python paper/Cellprofiling_Benchmark/scripts/generate_paper_graphs.py \
  --profiles all \
  --dry-run
```

Smoke-test the graph wrapper without full benchmark computation:

```bash
bash paper/Cellprofiling_Benchmark/scripts/generate_main_paper_graphs.sh --mock
```

Expected graph-input outputs include:

```text
paper/Cellprofiling_Benchmark/data/qc/*-moa-*.dat
paper/Cellprofiling_Benchmark/data/qc/*-crispr-*.dat
paper/Cellprofiling_Benchmark/data/qc/*-knn-*.dat
paper/Cellprofiling_Benchmark/data/qc/*-negative-control-map-*.dat
paper/Cellprofiling_Benchmark/data/qc/qc_heatmap_labels.tex
paper/Cellprofiling_Benchmark/data/jaccard/*.dat
paper/Cellprofiling_Benchmark/graphs/Results/CRISPR_Enrichment/crispr_database_barplot.pdf
paper/Cellprofiling_Benchmark/all_graphs/MoA/BBBC036/bbbc036_logged_counts_moa_pooled_or_skewness.png
```
