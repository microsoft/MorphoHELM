#!/usr/bin/env python
"""
Generate a CRISPR enrichment heatmap aggregated across 5 protein databases,
replicating the visual style of the MOA enrichment heatmap.
"""

import os
import pickle
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.abspath(os.path.join(ROOT, "..", "..", ".."))

NO_RESTRICTION_PATH = os.path.join(PROJECT, "results", "crispr_enrichment_no_restriction.pkl")
NOT_SAME_BATCH_PATH = os.path.join(PROJECT, "results", "crispr_enrichment_not_same_batch.pkl")
OUTPUT_PNG = os.path.join(
    PROJECT, "graphics", "enrichment", "crispr_enrichment", "crispr_enrichment_heatmap.png"
)

ALL_MODELS = [
    "cellprofiler", "cloome", "dino_v2_cls_token", "dino_v2_patch_token",
    "open_phenom", "resnet", "resnet_untrained", "subcell",
]
MODEL_LABELS = {
    "cellprofiler": "CellProfiler",
    "cloome": "CLOOME",
    "dino_v2_cls_token": "DINOv2 CLS",
    "dino_v2_patch_token": "DINOv2 Patch",
    "open_phenom": "OpenPhenom",
    "resnet": "ResNet",
    "resnet_untrained": "ResNet (Untrained)",
    "subcell": "SubCell",
}

DATABASES = ["CORUM", "HuMAP", "REACTOME", "SIGNOR", "StringDB"]


def load_and_aggregate():
    """Load both pickle files and aggregate metrics across the 5 protein databases."""
    with open(NO_RESTRICTION_PATH, "rb") as f:
        no_restr = pickle.load(f)
    with open(NOT_SAME_BATCH_PATH, "rb") as f:
        not_batch = pickle.load(f)

    n_models = len(ALL_MODELS)
    # columns: [No Restriction, Not Same Batch]
    frac_data = np.full((n_models, 2), np.nan)
    geom_data = np.full((n_models, 2), np.nan)

    for i, model in enumerate(ALL_MODELS):
        for j, raw in enumerate([no_restr, not_batch]):
            if model not in raw:
                continue
            model_dbs = raw[model]
            frac_vals = []
            geom_vals = []
            for db in DATABASES:
                if db in model_dbs:
                    frac_vals.append(model_dbs[db]["fraction_significant"])
                    geom_vals.append(model_dbs[db]["geometric_mean_odds"])
            if frac_vals:
                frac_data[i, j] = np.mean(frac_vals)
            if geom_vals:
                # geometric mean of the per-DB geometric_mean_odds
                geom_data[i, j] = np.exp(np.mean(np.log(geom_vals)))

    return frac_data, geom_data


def plot_heatmap(frac_data, geom_data):
    """Plot the combined heatmap matching the MOA enrichment style."""
    col_labels = ["No Restriction", "Not Same Batch"]
    row_labels = [MODEL_LABELS[m] for m in ALL_MODELS]
    n_models = len(ALL_MODELS)

    frac_vmin, frac_vmax = np.nanmin(frac_data), np.nanmax(frac_data)
    geom_vmin, geom_vmax = np.nanmin(geom_data), np.nanmax(geom_data)

    cmap = plt.cm.YlOrRd.copy()
    cmap.set_bad(color="0.88")

    # Layout: [frac_panel, cbar_frac, gap, geom_panel, cbar_geom]
    fig = plt.figure(figsize=(14, 6))
    gs = gridspec.GridSpec(
        1, 5,
        width_ratios=[3, 0.2, 1.2, 3, 0.2],
        wspace=0.05,
    )

    def draw_panel(ax, data, xlabels, title, fmt, vmin, vmax, show_ylabels):
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        masked = np.ma.array(data, mask=np.isnan(data))
        im = ax.imshow(masked, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)

        ax.set_xticks(range(len(xlabels)))
        ax.set_xticklabels(xlabels, rotation=35, ha="right", fontsize=10)
        ax.set_yticks(range(n_models))
        if show_ylabels:
            ax.set_yticklabels(row_labels, fontsize=10)
        else:
            ax.set_yticklabels([])

        ax.set_title(title, fontsize=11, fontweight="bold", pad=6)

        for r in range(data.shape[0]):
            for c in range(data.shape[1]):
                val = data[r, c]
                if np.isnan(val):
                    continue
                rgba = cmap(norm(val))
                lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                color = "white" if lum < 0.5 else "black"
                ax.text(c, r, f"{val:{fmt}}", ha="center", va="center",
                        color=color, fontsize=10)
        return im

    # Fraction Significant panel
    ax_frac = fig.add_subplot(gs[0, 0])
    ax_fcb = fig.add_subplot(gs[0, 1])
    im_f = draw_panel(ax_frac, frac_data, col_labels, "cpg-CRISPR", ".3f",
                       frac_vmin, frac_vmax, True)
    fig.colorbar(im_f, cax=ax_fcb)

    # Gap
    ax_gap = fig.add_subplot(gs[0, 2])
    ax_gap.axis("off")

    # Geometric Mean OR panel
    ax_geom = fig.add_subplot(gs[0, 3])
    ax_gcb = fig.add_subplot(gs[0, 4])
    im_g = draw_panel(ax_geom, geom_data, col_labels, "cpg-CRISPR", ".2f",
                       geom_vmin, geom_vmax, True)
    fig.colorbar(im_g, cax=ax_gcb)

    # Metric supertitles
    fig.canvas.draw()
    frac_mid = (ax_frac.get_position().x0 + ax_fcb.get_position().x1) / 2
    fig.text(frac_mid, 0.98, "Fraction Significant",
             ha="center", va="bottom", fontsize=13, fontweight="bold")
    geom_mid = (ax_geom.get_position().x0 + ax_gcb.get_position().x1) / 2
    fig.text(geom_mid, 0.98, "Geometric Mean OR",
             ha="center", va="bottom", fontsize=13, fontweight="bold")

    os.makedirs(os.path.dirname(OUTPUT_PNG), exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Heatmap saved to {OUTPUT_PNG}")


def main():
    global NO_RESTRICTION_PATH, NOT_SAME_BATCH_PATH, OUTPUT_PNG

    parser = argparse.ArgumentParser(description="Generate CRISPR enrichment heatmap")
    parser.add_argument("--no-restriction", default=NO_RESTRICTION_PATH,
                        help="Path to crispr_enrichment_no_restriction.pkl")
    parser.add_argument("--not-same-batch", default=NOT_SAME_BATCH_PATH,
                        help="Path to crispr_enrichment_not_same_batch.pkl")
    parser.add_argument("--output", default=OUTPUT_PNG,
                        help="Output PNG path")
    args = parser.parse_args()

    NO_RESTRICTION_PATH = args.no_restriction
    NOT_SAME_BATCH_PATH = args.not_same_batch
    OUTPUT_PNG = args.output

    frac_data, geom_data = load_and_aggregate()
    print("Fraction Significant data:\n", frac_data)
    print("Geometric Mean OR data:\n", geom_data)
    plot_heatmap(frac_data, geom_data)


if __name__ == "__main__":
    main()
