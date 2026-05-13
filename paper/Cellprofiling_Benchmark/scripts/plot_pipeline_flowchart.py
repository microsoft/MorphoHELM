#!/usr/bin/env python3
"""Render a standalone, linear data-to-benchmark pipeline flowchart."""

from __future__ import annotations

from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "graphs" / "Introduction"


COLORS = {
    "data": "#d9ecff",
    "curation": "#e6f4d7",
    "inference": "#fff2cc",
    "post": "#eadcf8",
    "benchmark": "#f8d7da",
    "output": "#d9ead3",
    "edge": "#444444",
    "number": "#2f5597",
}


STAGES = [
    (
        "Download BBBC036 Cell Painting images",
        "Single-source compound-perturbation imaging dataset used for MoA enrichment.",
        "data",
    ),
    (
        "Download CPG0016/JUMP Cell Painting images",
        "Multi-source compound and CRISPR imaging data used for cross-batch, cross-source, and replicate retrieval tasks.",
        "data",
    ),
    (
        "Curate plate and well metadata",
        "Standardize plate, well, source, batch, plate type, and layout fields across datasets.",
        "data",
    ),
    (
        "Curate perturbation annotations",
        "Attach compound IDs, CRISPR target genes, controls, and perturbation-level identifiers.",
        "data",
    ),
    (
        "Curate compound MoA labels",
        "Load labeled compound mechanism-of-action groups for BBBC036 and cpg-MoA enrichment.",
        "data",
    ),
    (
        "Curate CRISPR gene labels",
        "Map CRISPR samples to target genes for pathway and protein-complex enrichment.",
        "data",
    ),
    (
        "Curate protein/pathway databases",
        "Load CORUM, HuMAP, REACTOME, SIGNOR, and StringDB relationships for CRISPR enrichment.",
        "data",
    ),
    (
        "Build cell-count QC tables",
        "Create well-level cell-count inputs used to mask low-quality samples before scoring.",
        "curation",
    ),
    (
        "Create BBBC036 benchmark split",
        "Select BBBC036 wells and labels for single-source compound MoA enrichment.",
        "curation",
    ),
    (
        "Create cpg-MoA benchmark split",
        "Select labeled CPG0016 compounds for multi-source MoA enrichment.",
        "curation",
    ),
    (
        "Create cpg-CRISPR benchmark split",
        "Select CRISPR perturbations and gene labels for pathway enrichment.",
        "curation",
    ),
    (
        "Create cpg-target2 benchmark split",
        "Select strong-phenotype target2 compounds for replicate retrieval.",
        "curation",
    ),
    (
        "Create cpg-compound benchmark split",
        "Select large-scale compound wells for broad replicate retrieval.",
        "curation",
    ),
    (
        "Preprocess images for each model",
        "Load channels and fields of view; apply model-specific resizing, cropping, transforms, and patching.",
        "curation",
    ),
    (
        "Run CellProfiler feature extraction",
        "Generate classical hand-crafted morphology profiles for each well.",
        "inference",
    ),
    (
        "Run learned-model inference",
        "Generate representations from CLOOME, DINOv2 CLS, DINOv2 Patch, OpenPhenom, ResNet, ResNet-UT, and SubCell.",
        "inference",
    ),
    (
        "Aggregate to well-level profiles",
        "Average or concatenate field-of-view and channel embeddings into one feature vector per well.",
        "inference",
    ),
    (
        "Join metadata and benchmark labels",
        "Attach source, batch, plate, well, perturbation IDs, compound MoA labels, and CRISPR gene labels.",
        "post",
    ),
    (
        "Apply sample quality control",
        "Use cell-count QC masks to remove low-quality wells before benchmark scoring.",
        "post",
    ),
    (
        "Center-scale features",
        "Apply the selected center-scaling scope so each representation enters the same postprocessing pipeline.",
        "post",
    ),
    (
        "Reduce dimensionality with PCA",
        "Project profiles to the selected dimensionality, with the main pipeline using PCA64.",
        "post",
    ),
    (
        "Apply control-based MAD scaling",
        "Use control wells to robustly scale features within plate-level groups.",
        "post",
    ),
    (
        "Optionally apply sphering",
        "Run control-based sphering variants for normalization sweeps; main NoSph results skip this step.",
        "post",
    ),
    (
        "Write normalized feature tables",
        "Save one normalized parquet per model and dataset for benchmark execution.",
        "post",
    ),
    (
        "Define No Restriction candidate pools",
        "Allow any non-self candidate for baseline retrieval and enrichment.",
        "benchmark",
    ),
    (
        "Define Not Same Batch candidate pools",
        "Exclude candidates from the query's experimental batch.",
        "benchmark",
    ),
    (
        "Define Not Same Source candidate pools",
        "Exclude candidates from the query's institution/source where supported.",
        "benchmark",
    ),
    (
        "Define layout-restricted candidate pools",
        "Exclude same-well-position candidates for NSL and same-source/same-layout candidates for NSSL where supported.",
        "benchmark",
    ),
    (
        "Run BBBC036 MoA enrichment",
        "Score compound MoA retrieval on the single-source BBBC036 dataset.",
        "benchmark",
    ),
    (
        "Run cpg-MoA enrichment",
        "Score compound MoA retrieval under NR, NSB, and NSS restrictions.",
        "benchmark",
    ),
    (
        "Run cpg-CRISPR pathway enrichment",
        "Score gene-pathway and protein-complex enrichment under NR and NSB restrictions.",
        "benchmark",
    ),
    (
        "Run cpg-target2 replicate retrieval",
        "Compute KNN Recall@1 and mAP for strong-phenotype compound replicates.",
        "benchmark",
    ),
    (
        "Run cpg-compound replicate retrieval",
        "Compute KNN Recall@1 and mAP for the large compound replicate benchmark.",
        "benchmark",
    ),
    (
        "Run negative-control mAP",
        "Use DMSO distractors and retained same-compound positives as a control retrieval analysis.",
        "benchmark",
    ),
    (
        "Run normalization pipeline sweeps",
        "Repeat benchmark exports across postprocessing variants to quantify preprocessing sensitivity.",
        "benchmark",
    ),
    (
        "Compute discovery-set overlap",
        "Measure Jaccard overlap among significant MoA compounds and CRISPR genes discovered by each representation.",
        "benchmark",
    ),
    (
        "Export benchmark result files",
        "Save result pickles and heatmap data tables for each task and pipeline variant.",
        "output",
    ),
    (
        "Render figures and reproduction inputs",
        "Generate figure assets and package the code/data needed to reproduce the main benchmark outputs.",
        "output",
    ),
]


def draw_stage(ax, number: int, y: float, title: str, detail: str, color_key: str) -> None:
    x = 0.8
    w = 11.2
    h = 0.68
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.03,rounding_size=0.06",
        linewidth=1.0,
        edgecolor="#333333",
        facecolor=COLORS[color_key],
    )
    ax.add_patch(patch)

    circle = Circle((x + 0.42, y + h / 2), 0.21, facecolor=COLORS["number"], edgecolor="white", linewidth=1.0)
    ax.add_patch(circle)
    ax.text(x + 0.42, y + h / 2, str(number), ha="center", va="center", fontsize=8.5, color="white", fontweight="bold")

    ax.text(x + 0.82, y + 0.46, title, ha="left", va="center", fontsize=9.1, fontweight="bold", color="#111111")
    ax.text(x + 0.82, y + 0.21, fill(detail, width=118), ha="left", va="center", fontsize=7.4, color="#222222")


def draw_arrow(ax, y_top: float, y_bottom: float) -> None:
    x = 6.4
    ax.add_patch(
        FancyArrowPatch(
            (x, y_top),
            (x, y_bottom),
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.1,
            color=COLORS["edge"],
            shrinkA=1,
            shrinkB=1,
        )
    )


def render() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13.2, 32.0))
    ax.set_xlim(0, 12.8)
    ax.set_ylim(0, 31.8)
    ax.axis("off")

    ax.text(
        6.4,
        31.25,
        "MorphoHELM data-to-benchmark pipeline",
        ha="center",
        va="center",
        fontsize=20,
        fontweight="bold",
    )
    ax.text(
        6.4,
        30.82,
        "Granular single-flow view from individual data/resource inputs through QC, inference, postprocessing, each benchmark task, and final outputs",
        ha="center",
        va="center",
        fontsize=10.2,
        color="#333333",
    )

    start_y = 29.9
    step = 0.88
    box_h = 0.68
    y_positions = [start_y - i * step for i in range(len(STAGES))]

    for i, (title, detail, color_key) in enumerate(STAGES, start=1):
        y = y_positions[i - 1]
        draw_stage(ax, i, y, title, detail, color_key)
        if i < len(STAGES):
            draw_arrow(ax, y - 0.03, y_positions[i] + box_h + 0.03)

    ax.text(
        6.4,
        0.42,
        "The same quality-controlled, normalized feature tables feed each task so model comparisons are paired across datasets, benchmark tasks, and stringency levels.",
        ha="center",
        va="center",
        fontsize=8.6,
        color="#333333",
    )

    for suffix in ("pdf", "png"):
        fig.savefig(
            OUT_DIR / f"data_to_benchmark_pipeline_flowchart.{suffix}",
            bbox_inches="tight",
            dpi=300,
        )
    plt.close(fig)


if __name__ == "__main__":
    render()
