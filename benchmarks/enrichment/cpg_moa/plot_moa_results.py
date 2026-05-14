#!/usr/bin/env python3
"""
Generate MoA enrichment bar charts comparing old imputation vs Haldane-Anscombe.

Produces a 4-panel bar chart:
  1. Fraction significant (old imputation method)
  2. Fraction significant (Haldane-Anscombe correction)
  3. Arithmetic mean OR (Haldane)
  4. Geometric mean OR (Haldane)

Usage:
  python plot_moa_results.py \
    --input results/moa_enrichment/bbbc036_moa_haldane_1pct.pkl \
    --old-input results/moa_enrichment/bbbc036_moa_with_pca.pkl \
    --labels with_pca
"""

import argparse
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt

MODEL_LABELS = {
    "cellprofiler": "CellProf",
    "cloome": "CLOOME",
    "dino_v2_cls": "DINOv2-C",
    "dino_v2_patch": "DINOv2-P",
    "open_phenom": "OpenPhen",
    "resnet": "ResNet",
    "resnet_untrained": "ResNet-UT",
    "subcell": "SubCell",
}

OUTPUT_DIR = "graphics/enrichment/moa_enrichment"


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def plot_moa_enrichment(data, title_suffix, out_path, old_data=None, baseline_data=None):
    """Four-panel figure: old frac_sig, new frac_sig, arithmetic mean OR, geometric mean OR."""
    models = sorted(data.keys())
    labels = [MODEL_LABELS.get(m, m) for m in models]
    n = len(models)

    frac_sig = [data[m]["fraction_significant"] for m in models]
    arith_or = [data[m]["mean_odds"] for m in models]
    geom_or = [data[m]["geometric_mean_odds"] for m in models]

    # Compute baseline averages across models
    bl = {}
    if baseline_data is not None:
        bl_models = [m for m in models if m in baseline_data]
        bl["frac_sig"] = np.mean([baseline_data[m]["fraction_significant"] for m in bl_models])
        bl["arith_or"] = np.mean([baseline_data[m]["mean_odds"] for m in bl_models])
        bl["geom_or"] = np.mean([baseline_data[m]["geometric_mean_odds"] for m in bl_models])

    has_old = old_data is not None
    n_panels = 4 if has_old else 3

    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    bar_colors = plt.cm.tab10(np.linspace(0, 1, n))
    x = np.arange(n)
    panel_idx = 0

    if has_old:
        old_frac_sig = [old_data[m]["fraction_significant"] for m in models if m in old_data]
        ax = axes[panel_idx]
        ax.bar(x, old_frac_sig, color=bar_colors, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("Fraction Significant (p < 0.05)")
        ax.set_title("Frac Sig (Old Imputation)", fontweight="bold")
        if bl:
            ax.axhline(bl["frac_sig"], color="gray", linestyle="--", linewidth=1.2,
                        label=f"Baseline ({bl['frac_sig']:.3f})")
            ax.legend(fontsize=8, loc="upper right")
        panel_idx += 1

    # Fraction significant (Haldane)
    ax = axes[panel_idx]
    ax.bar(x, frac_sig, color=bar_colors, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Fraction Significant (p < 0.05)")
    title = "Frac Sig (Haldane-Anscombe)" if has_old else "Fraction Significant"
    ax.set_title(title, fontweight="bold")
    if bl:
        ax.axhline(bl["frac_sig"], color="gray", linestyle="--", linewidth=1.2,
                    label=f"Baseline ({bl['frac_sig']:.3f})")
        ax.legend(fontsize=8, loc="upper right")
    panel_idx += 1

    # Arithmetic mean OR
    ax = axes[panel_idx]
    ax.bar(x, arith_or, color="#1f77b4", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Arithmetic Mean OR")
    ax.set_title("Arithmetic Mean Odds Ratio", fontweight="bold")
    if bl:
        ax.axhline(bl["arith_or"], color="gray", linestyle="--", linewidth=1.2,
                    label=f"Baseline ({bl['arith_or']:.1f})")
        ax.legend(fontsize=8, loc="upper right")
    panel_idx += 1

    # Geometric mean OR
    ax = axes[panel_idx]
    ax.bar(x, geom_or, color="#2ca02c", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Geometric Mean OR")
    ax.set_title("Geometric Mean Odds Ratio", fontweight="bold")
    if bl:
        ax.axhline(bl["geom_or"], color="gray", linestyle="--", linewidth=1.2,
                    label=f"Baseline ({bl['geom_or']:.1f})")
        ax.legend(fontsize=8, loc="upper right")


    fig.suptitle(f"BBBC036 MoA Enrichment — {title_suffix}\n"
                 "(Haldane-Anscombe correction, top 1% cutoff)",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot BBBC036 MoA Enrichment Results")
    parser.add_argument("--input", required=True, nargs="+",
                        help="One or more result pickle files (Haldane-Anscombe)")
    parser.add_argument("--old-input", default=None,
                        help="Old imputation method pickle for comparison panel")
    parser.add_argument("--baseline-input", default=None,
                        help="Baseline pickle (shuffled labels) for dashed reference lines")
    parser.add_argument("--labels", nargs="*", default=None,
                        help="Labels for each input file (default: filename stem)")
    parser.add_argument("--output-dir", default=OUTPUT_DIR,
                        help="Output directory for plots")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    old_data = None
    if args.old_input:
        old_data = load_pkl(args.old_input)
        print(f"Loaded old results: {args.old_input}")

    baseline_data = None
    if args.baseline_input:
        baseline_data = load_pkl(args.baseline_input)
        print(f"Loaded baseline results: {args.baseline_input}")

    for i, pkl_path in enumerate(args.input):
        data = load_pkl(pkl_path)
        if args.labels and i < len(args.labels):
            label = args.labels[i]
        else:
            label = os.path.basename(pkl_path).replace(".pkl", "")

        out_path = os.path.join(args.output_dir, f"moa_enrichment_{label}.png")
        plot_moa_enrichment(data, label, out_path, old_data=old_data,
                            baseline_data=baseline_data)

        # Print summary table
        print(f"\n{'Model':<20} {'Frac Sig':>10} {'Arith OR':>10} {'Geom OR':>10} {'N Proc':>8}")
        print("=" * 62)
        for m in sorted(data.keys()):
            r = data[m]
            print(f"{m:<20} {r['fraction_significant']:>10.4f} {r['mean_odds']:>10.2f} "
                  f"{r['geometric_mean_odds']:>10.2f} {r['n_processed']:>8}")


if __name__ == "__main__":
    main()
