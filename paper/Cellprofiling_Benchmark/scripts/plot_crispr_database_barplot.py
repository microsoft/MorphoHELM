#!/usr/bin/env python3
"""Plot per-database cpg-CRISPR enrichment performance."""

from __future__ import annotations

import pickle
from pathlib import Path
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


PAPER_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = PAPER_DIR.parents[1]
RESULT_DIR = (
    PROJECT_DIR
    / "results"
    / "normalization_sweep"
    / "QC_CSAll_Plate__PCA64__MADCtrl_Plate__NoSph"
)
OUT_DIR = PAPER_DIR / "graphs" / "Results" / "CRISPR_Enrichment"
OUT_PDF = OUT_DIR / "crispr_database_barplot.pdf"
OUT_PNG = OUT_DIR / "crispr_database_barplot.png"

MODELS = [
    "cellprofiler",
    "cloome",
    "dino_v2_cls_token",
    "dino_v2_patch_token",
    "open_phenom",
    "resnet",
    "resnet_untrained",
    "subcell",
]
MODEL_LABELS = {
    "cellprofiler": "CellProfiler",
    "cloome": "CLOOME",
    "dino_v2_cls_token": "DINOv2\nCLS",
    "dino_v2_patch_token": "DINOv2\nPatch",
    "open_phenom": "OpenPhenom",
    "resnet": "ResNet",
    "resnet_untrained": "ResNet\nUntrained",
    "subcell": "SubCell",
}
DATABASES = ["CORUM", "HuMAP", "REACTOME", "SIGNOR", "StringDB"]


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot per-database cpg-CRISPR enrichment performance.")
    parser.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    nr = load_pickle(args.result_dir / "crispr_enrichment_no_restriction.pkl")
    nsb = load_pickle(args.result_dir / "crispr_enrichment_not_same_batch.pkl")

    # Match the main CRISPR heatmap convention: order by aggregate No Restriction
    # fraction-significant performance.
    model_order = sorted(
        MODELS,
        key=lambda model: -sum(
            float(nr[model][db]["fraction_significant"]) for db in DATABASES
        )
        / len(DATABASES),
    )

    model_colors = dict(
        zip(model_order, plt.get_cmap("tab10").colors[: len(model_order)], strict=True)
    )

    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "legend.title_fontsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(7.15, 6.6),
        sharex=True,
        gridspec_kw={"hspace": 0.24},
    )

    x = np.arange(len(DATABASES))
    bar_width = 0.088
    offsets = (np.arange(len(model_order)) - (len(model_order) - 1) / 2) * bar_width

    for ax, metric, title, ylabel, ylim in [
        (
            axes[0],
            "fraction_significant",
            "A. Fraction of genes with significant enrichment",
            "Fraction Significant (p < 0.05)",
            (0, 0.115),
        ),
        (
            axes[1],
            "geometric_mean_odds",
            "B. Geometric mean odds ratio for top 1% similar genes",
            "Geometric Mean Odds Ratio",
            (0, 15.5),
        ),
    ]:
        for offset, model in zip(offsets, model_order, strict=True):
            nr_values = [float(nr[model][db][metric]) for db in DATABASES]
            nsb_values = [float(nsb[model][db][metric]) for db in DATABASES]
            ax.bar(
                x + offset,
                nr_values,
                width=bar_width,
                color=model_colors[model],
                label=MODEL_LABELS[model].replace("\n", " "),
                alpha=0.70,
                edgecolor="white",
                linewidth=0.25,
            )
            ax.bar(
                x + offset,
                nsb_values,
                width=bar_width * 0.55,
                color=model_colors[model],
                alpha=1.0,
                edgecolor="black",
                linewidth=0.30,
                hatch="////",
            )

        ax.set_title(title, loc="left", fontweight="bold", pad=6)
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)
        ax.grid(axis="y", color="0.88", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.margins(x=0.045)

    for ax in axes:
        ax.set_xticks(x, DATABASES)
        ax.tick_params(axis="x", labelbottom=True, labelrotation=0)
        for label in ax.get_xticklabels():
            label.set_ha("center")
    axes[1].set_xlabel("Protein database")

    model_handles = [
        Patch(facecolor=model_colors[model], edgecolor="none", label=MODEL_LABELS[model].replace("\n", " "))
        for model in model_order
    ]
    model_legend = fig.legend(
        handles=model_handles,
        title="Model",
        loc="upper center",
        bbox_to_anchor=(0.50, 0.990),
        frameon=False,
        ncol=4,
        columnspacing=1.0,
        handlelength=1.5,
    )
    fig.add_artist(model_legend)

    restriction_handles = [
        Patch(facecolor="0.70", edgecolor="none", alpha=0.70, label="No Restriction"),
        Patch(facecolor="0.70", edgecolor="black", hatch="////", linewidth=0.30, label="Not Same Batch"),
    ]
    axes[0].legend(
        handles=restriction_handles,
        loc="upper right",
        frameon=False,
        handlelength=1.7,
        borderaxespad=0.2,
    )
    fig.subplots_adjust(left=0.12, right=0.985, top=0.80, bottom=0.11)

    out_pdf = args.out_dir / "crispr_database_barplot.pdf"
    out_png = args.out_dir / "crispr_database_barplot.png"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_pdf}")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
