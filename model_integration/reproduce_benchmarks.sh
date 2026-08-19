#!/usr/bin/env bash
# Deterministic reproduction of the RobuDINO + baselines enrichment benchmarks.
#
# Chain (per ready split):
#   aggregated parquet -> normalize (pinned PCA) -> align to baseline well set
#   -> place next to the 8 baselines -> run enrichment sweeps (seeded).
#
# Baselines are the repo's shipped normalized parquets (content-hashed via LFS);
# only RobuDINO is (re)normalized here. All seeds/params are the repo defaults.
#
# Configure via environment variables (defaults shown):
#   DATA_ROOT   data container with aggregated/, normalized/, configs/, results/
#   PY          python interpreter
#   PROFILE     normalization profile
#   MODEL       our model name (parquet stem)
#
# Usage:
#   bash model_integration/reproduce_benchmarks.sh
#   DATA_ROOT=/path/to/container PY=python bash model_integration/reproduce_benchmarks.sh
set -euo pipefail

# Repo root = two levels up from this script (model_integration/ -> repo/).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"

DATA_ROOT="${DATA_ROOT:-/raid/cache/gpznx/data/microsoft_benchmark}"
PY="${PY:-python}"
PROFILE="${PROFILE:-CSAll_Plate__PCA64__MADCtrl_Plate__NoSph}"
MODEL="${MODEL:-dino_v2_cellpainting}"

BASELINE_ROOT="$REPO/data/main_paper_inputs/normalized/$PROFILE"
QC="--cell-count-qc --qc-cell-counts-dir data/qc/cell_counts --qc-cell-count-cutoff 0.05 --qc-sample-wells 5000 --qc-seed 42"

cd "$REPO"

# 1) Normalize RobuDINO for all splits (PCA is pinned in normalize_utils.py).
$PY data-preprocessing/src/normalize_splits.py \
  --config "$DATA_ROOT/configs/normalize_run.yaml" --profile "$PROFILE" \
  --splits cpg-crispr bbbc036 cpg-moa --model all

# 2) Align well set to the baselines + drop MoA-conflicting cols; place next to baselines.
for split in cpg-crispr bbbc036 cpg-moa; do
  ours="$DATA_ROOT/normalized/$PROFILE/$split/${MODEL}_normalized.parquet"
  $PY model_integration/prepare_for_benchmark.py \
    --split "$split" --our "$ours" --baseline "$BASELINE_ROOT/$split/resnet_normalized.parquet"
  cp "$ours" "$BASELINE_ROOT/$split/"
done

# 3) Enrichment sweeps (all 9 models; seeded permutation tests -> deterministic).
mkdir -p "$DATA_ROOT/results"
$PY benchmarks/enrichment/crispr/run_crispr_enrichment.py \
  --features-dir "$BASELINE_ROOT/cpg-crispr" --mode no_restriction  --n_resamples 100 $QC \
  --output "$DATA_ROOT/results/crispr_no_restriction.pkl"
$PY benchmarks/enrichment/crispr/run_crispr_enrichment.py \
  --features-dir "$BASELINE_ROOT/cpg-crispr" --mode not_same_batch --n_resamples 100 $QC \
  --output "$DATA_ROOT/results/crispr_not_same_batch.pkl"
$PY benchmarks/enrichment/bbbc036_moa/run_bbbc036_moa.py \
  --features-dir "$BASELINE_ROOT/bbbc036" \
  --metadata data/main_paper_inputs/metadata/bbbc036_metadata.parquet --n-resamples 100 $QC \
  --output "$DATA_ROOT/results/bbbc036_moa.pkl"

# 3b) cpg-MoA enrichment: build cross-source compound profiles (all 9 models),
#     then run the 4-paradigm seeded enrichment (Global/Within-Source/Not-Same-Batch/Not-Same-Source).
$PY benchmarks/enrichment/cpg_moa/build_moa_profiles.py \
  --normalized-dir "$BASELINE_ROOT/cpg-moa" \
  --moa-labels feature-extraction/configs/labeled_moa_samples.csv \
  --plate-metadata feature-extraction/metadata/metadata.parquet \
  --output "$DATA_ROOT/results/moa_profiles/moa_cross_source_profiles.pkl" $QC
$PY benchmarks/enrichment/cpg_moa/run_cpg_moa.py \
  --profiles "$DATA_ROOT/results/moa_profiles/moa_cross_source_profiles.pkl" \
  --output "$DATA_ROOT/results/cpg_moa_enrichment.pkl" \
  --bbbc-results "$DATA_ROOT/results/bbbc036_moa.pkl"
mkdir -p "$DATA_ROOT/results/graphs"
cp "$DATA_ROOT/results/cpg_moa_enrichment.png" "$DATA_ROOT/results/graphs/cpg_moa_enrichment.png"

# 4) Figures.
$PY model_integration/plot_benchmark_results.py \
  --results-dir "$DATA_ROOT/results" --out-dir "$DATA_ROOT/results/graphs"

echo "Reproduction complete. Results -> $DATA_ROOT/results ; figures -> $DATA_ROOT/results/graphs"
