#!/usr/bin/env python3
"""Readable pooled BBBC036 MoA odds-ratio distribution comparison."""

from __future__ import annotations

import pickle
from pathlib import Path
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT = Path(__file__).resolve().parents[3]
OLD_PATH = PROJECT / "results" / "moa_enrichment" / "bbbc036_moa_imputation_1pct.pkl"
HALDANE_PATH = PROJECT / "results" / "moa_enrichment" / "bbbc036_moa_haldane_1pct.pkl"
OUT_DIR = PROJECT / "graphics" / "enrichment" / "moa_enrichment"
OUT_PNG = OUT_DIR / "moa_pooled_or_skewness.png"
OUT_PDF = OUT_DIR / "moa_pooled_or_skewness.pdf"
PAPER_OUT_DIR = (
    PROJECT
    / "paper"
    / "Cellprofiling_Benchmark"
    / "all_graphs"
    / "MoA"
    / "BBBC036"
)
PAPER_OUT_PNG = PAPER_OUT_DIR / "bbbc036_logged_counts_moa_pooled_or_skewness.png"


def pooled_odds(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        data = pickle.load(handle)
    values = []
    for result in data.values():
        values.extend(float(value) for value in result["all_odds"])
    return np.asarray(values, dtype=np.float64)


def add_summary_box(ax, text: str, y: float = 0.95) -> None:
    ax.text(
        0.975,
        y,
        text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=14,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "0.25",
            "alpha": 1.0,
        },
    )


def add_reference_lines(ax, refs: list[tuple[float, str, str]], ymax_fraction: float = 0.96) -> None:
    for value, color, linestyle in refs:
        ax.axvline(value, color=color, linestyle=linestyle, linewidth=2.0, ymax=ymax_fraction)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot BBBC036 MoA odds-ratio distribution diagnostics.")
    parser.add_argument("--old-path", type=Path, default=OLD_PATH)
    parser.add_argument("--haldane-path", type=Path, default=HALDANE_PATH)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--paper-out-dir", type=Path, default=PAPER_OUT_DIR)
    args = parser.parse_args()

    old = pooled_odds(args.old_path)
    haldane = pooled_odds(args.haldane_path)
    log_haldane = np.log(haldane[haldane > 0])

    old_zero_count = int(np.sum(old == 0))
    old_nonzero = old[old > 0]

    old_median = float(np.median(old))
    old_max = float(np.max(old))

    haldane_median = float(np.median(haldane))
    haldane_max = float(np.max(haldane))

    log_median = float(np.log(haldane_median))

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.titlesize": 14,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(7.2, 10.0),
        gridspec_kw={"hspace": 0.42},
    )
    fig.suptitle(
        "BBBC036 MoA odds-ratio distributions: old imputation vs corrected reporting",
        fontweight="bold",
        y=0.992,
    )

    # Panel A: old imputation. Keep the full x-range visible so the extreme tail
    # remains apparent, while drawing the zero mass explicitly.
    ax = axes[0]
    old_bins = np.linspace(0, old_max, 85)
    ax.hist(old_nonzero, bins=old_bins, color="#d62728", alpha=0.78, edgecolor="white", linewidth=0.35)
    ax.bar(
        0,
        old_zero_count,
        width=old_max / 85,
        align="edge",
        color="#7f0000",
        alpha=0.78,
    )
    ax.set_yscale("log")
    ax.set_xlim(0, old_max)
    ax.set_ylim(1, 2.0e4)
    ax.set_title("A. Background imputation (old method)", loc="left", fontweight="bold")
    ax.set_xlabel("Odds ratio (full range)")
    ax.set_ylabel("Count (log scale)")
    add_reference_lines(ax, [(old_median, "orange", ":")])
    add_summary_box(
        ax,
        f"Median OR = {old_median:.1f}",
    )
    ax.set_xticks([0, 300, 600, 900, 1200, 1500])

    # Panel B: corrected Haldane-Anscombe OR on the full odds-ratio range.
    ax = axes[1]
    haldane_bins = np.linspace(0, haldane_max, 95)
    ax.hist(haldane, bins=haldane_bins, color="#1f77b4", alpha=0.82, edgecolor="white", linewidth=0.35)
    ax.set_yscale("log")
    ax.set_xlim(0, haldane_max)
    ax.set_ylim(1, 2.0e4)
    ax.set_title("B. Haldane-Anscombe correction", loc="left", fontweight="bold")
    ax.set_xlabel("Odds ratio (full range)")
    ax.set_ylabel("Count (log scale)")
    add_reference_lines(ax, [(haldane_median, "orange", ":")])
    add_summary_box(
        ax,
        f"Median OR = {haldane_median:.1f}",
    )
    ax.set_xticks([0, 400, 800, 1200, 1600, 2000])

    # Panel C: log-transformed corrected OR on a linear count scale.
    ax = axes[2]
    log_bins = np.linspace(0, 8, 70)
    ax.hist(log_haldane, bins=log_bins, color="#2ca02c", alpha=0.82, edgecolor="white", linewidth=0.35)
    ax.set_xlim(0, 8)
    ax.set_title("C. Haldane-Anscombe correction after log transform", loc="left", fontweight="bold")
    ax.set_xlabel("log(Odds ratio)")
    ax.set_ylabel("Count")
    add_reference_lines(ax, [(log_median, "orange", ":")])
    add_summary_box(
        ax,
        f"Median log(OR) = {log_median:.2f}",
    )

    for ax in axes:
        ax.grid(axis="y", color="0.88", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    fig.subplots_adjust(left=0.11, right=0.985, top=0.94, bottom=0.065)

    out_png = args.out_dir / "moa_pooled_or_skewness.png"
    out_pdf = args.out_dir / "moa_pooled_or_skewness.pdf"
    paper_out_png = args.paper_out_dir / "bbbc036_logged_counts_moa_pooled_or_skewness.png"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.paper_out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(paper_out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}")
    print(f"Wrote {out_pdf}")
    print(f"Wrote {paper_out_png}")


if __name__ == "__main__":
    main()
