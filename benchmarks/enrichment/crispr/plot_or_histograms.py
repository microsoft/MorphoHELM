#!/usr/bin/env python3
"""
Generate OR distribution histograms comparing three approaches:
  1. Old: background-count imputation + arithmetic mean OR
  2. Haldane-Anscombe correction + arithmetic mean OR
  3. Haldane-Anscombe correction + geometric mean OR (per model)

Produces a 5-row (DBs) × 3-col (approaches) grid per enrichment mode.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

RESULTS_DIR = "results/crispr_enrichment"
OUTPUT_DIR = "graphics/enrichment/crispr_enrichment"
DBS = ["CORUM", "HuMAP", "REACTOME", "SIGNOR", "StringDB"]

MODEL_LABELS = {
    "cellprofiler": "CellProf",
    "cloome": "CLOOME",
    "dino_v2_cls_token": "DINOv2-C",
    "dino_v2_patch_token": "DINOv2-P",
    "open_phenom": "OpenPhen",
    "resnet": "ResNet",
    "resnet_untrained": "ResNet-UT",
    "subcell": "SubCell",
}

OLD_PKL = os.path.join(RESULTS_DIR, "crispr_enrichment_with_pca.pkl")
NEW_PKLS = {
    "no_restriction": os.path.join(RESULTS_DIR, "crispr_enrichment_no_restriction.pkl"),
    "not_same_batch": os.path.join(RESULTS_DIR, "crispr_enrichment_not_same_batch.pkl"),
}


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def collect_all_odds(data, db):
    """Pool per-gene ORs across all models for a given DB."""
    all_or = []
    for model in data:
        if db in data[model]:
            all_or.extend(data[model][db]["all_odds"])
    return np.array(all_or, dtype=float)


def get_per_model_geom_mean(data, db):
    """Compute geometric mean OR per model for a given DB."""
    models, geom_means = [], []
    for model in sorted(data.keys()):
        if db not in data[model]:
            continue
        odds = np.array(data[model][db]["all_odds"], dtype=float)
        pos = odds[odds > 0]
        if len(pos) == 0:
            continue
        gm = np.exp(np.mean(np.log(pos)))
        models.append(MODEL_LABELS.get(model, model))
        geom_means.append(gm)
    return models, geom_means


def get_per_model_arith_mean(data, db):
    """Compute arithmetic mean OR per model for a given DB (old approach)."""
    models, means = [], []
    for model in sorted(data.keys()):
        if db not in data[model]:
            continue
        odds = np.array(data[model][db]["all_odds"], dtype=float)
        models.append(MODEL_LABELS.get(model, model))
        means.append(odds.mean())
    return models, means


def plot_grid(old_data, new_data, mode, out_path):
    has_old = old_data is not None
    n_cols = 3 if has_old else 2

    fig, axes = plt.subplots(len(DBS), n_cols, figsize=(5 * n_cols, 3.5 * len(DBS)))

    if has_old:
        col_titles = [
            "Old: Background Imputation\n(Arithmetic Mean OR per Model)",
            "Haldane-Anscombe\n(Arithmetic Mean OR per Model)",
            "Haldane-Anscombe\n(Geometric Mean OR per Model)",
        ]
    else:
        col_titles = [
            "Haldane-Anscombe\n(Arithmetic Mean OR per Model)",
            "Haldane-Anscombe\n(Geometric Mean OR per Model)",
        ]

    bar_colors = plt.cm.tab10(np.linspace(0, 1, 8))

    for row_i, db in enumerate(DBS):
        col = 0

        # Column: Old approach — arithmetic mean per model
        if has_old:
            ax = axes[row_i, col]
            models, means = get_per_model_arith_mean(old_data, db)
            x = np.arange(len(models))
            ax.bar(x, means, color="#d62728", alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=45, ha="right", fontsize=7)
            ax.set_ylabel(db, fontsize=12, fontweight="bold")
            ax.set_xlabel("Arithmetic Mean OR")
            avg = np.mean(means)
            ax.axhline(avg, color="black", linestyle="--", linewidth=1, alpha=0.6)
            ax.text(0.95, 0.95, f"avg={avg:.1f}", transform=ax.transAxes,
                    fontsize=9, va="top", ha="right",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
            if row_i == 0:
                ax.set_title(col_titles[col], fontsize=11, fontweight="bold")
            col += 1

        # Column: Haldane-Anscombe — arithmetic mean per model
        ax = axes[row_i, col]
        models, means = get_per_model_arith_mean(new_data, db)
        x = np.arange(len(models))
        ax.bar(x, means, color="#1f77b4", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha="right", fontsize=7)
        if not has_old:
            ax.set_ylabel(db, fontsize=12, fontweight="bold")
        ax.set_xlabel("Arithmetic Mean OR")
        avg = np.mean(means)
        ax.axhline(avg, color="black", linestyle="--", linewidth=1, alpha=0.6)
        ax.text(0.95, 0.95, f"avg={avg:.1f}", transform=ax.transAxes,
                fontsize=9, va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        if row_i == 0:
            ax.set_title(col_titles[col], fontsize=11, fontweight="bold")
        col += 1

        # Column: Haldane-Anscombe — geometric mean per model
        ax = axes[row_i, col]
        models, geom_means = get_per_model_geom_mean(new_data, db)
        x = np.arange(len(models))
        ax.bar(x, geom_means, color="#2ca02c", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha="right", fontsize=7)
        ax.set_xlabel("Geometric Mean OR")
        avg = np.mean(geom_means)
        ax.axhline(avg, color="black", linestyle="--", linewidth=1, alpha=0.6)
        ax.text(0.95, 0.95, f"avg={avg:.1f}", transform=ax.transAxes,
                fontsize=9, va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        if row_i == 0:
            ax.set_title(col_titles[col], fontsize=11, fontweight="bold")

    mode_label = mode.replace("_", " ").title()
    fig.suptitle(
        f"OR Comparison — {mode_label}\n(Per-model summary across 3 approaches)",
        fontsize=14, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    old_data = load_pkl(OLD_PKL) if os.path.exists(OLD_PKL) else None

    for mode, pkl_path in NEW_PKLS.items():
        new_data = load_pkl(pkl_path)
        use_old = old_data if mode == "no_restriction" else None
        out = os.path.join(OUTPUT_DIR, f"or_distribution_{mode}.png")
        plot_grid(use_old, new_data, mode, out)


if __name__ == "__main__":
    main()
