#!/usr/bin/env python3
"""Publication-style plots of MorphoHELM enrichment results.

Your model (``dino_v2_cellpainting``) is displayed as **RobuDINO** and
highlighted. Bars/cells are colored on a red-yellow scale by value, with a
colorbar. No gridlines; layout is managed to avoid overlapping text. Saves:
  - bbbc036_moa_summary.png
  - cpg_crispr_summary.png
  - cpg_crispr_per_database.png

Usage:
  python model_integration/plot_benchmark_results.py \
    --results-dir /raid/cache/gpznx/data/microsoft_benchmark/results \
    --out-dir     /raid/cache/gpznx/data/microsoft_benchmark/results/graphs
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np

OURS_KEY = "dino_v2_cellpainting"
CMAP = colormaps["YlOrRd"]  # red-yellow: low = pale yellow, high = deep red

DISPLAY = {
    "dino_v2_cellpainting": "RobuDINO",
    "dino_v2_cls_token": "DINOv2 (CLS)",
    "dino_v2_patch_token": "DINOv2 (patch)",
    "resnet": "ResNet-101",
    "resnet_untrained": "ResNet-101 (untrained)",
    "open_phenom": "OpenPhenom",
    "cloome": "CLOOME",
    "subcell": "SubCell",
    "cellprofiler": "CellProfiler",
}

# cpg-moa profiles use canonical names (dino_v2 = CLS, openphenom, ...).
CPGMOA_DISPLAY = {
    "dino_v2_cellpainting": "RobuDINO",
    "dino_v2": "DINOv2 (CLS)",
    "dino_v2_patch": "DINOv2 (patch)",
    "resnet": "ResNet-101",
    "resnet_untrained": "ResNet-101 (untrained)",
    "openphenom": "OpenPhenom",
    "cloome": "CLOOME",
    "subcell": "SubCell",
    "cellprofiler": "CellProfiler",
}

# Map cpg-moa canonical names -> keys used in the BBBC036 MoA result pickle.
BBBC_FROM_CANONICAL = {
    "dino_v2_cellpainting": "dino_v2_cellpainting",
    "dino_v2": "dino_v2_cls_token",
    "dino_v2_patch": "dino_v2_patch_token",
    "openphenom": "open_phenom",
    "resnet": "resnet",
    "resnet_untrained": "resnet_untrained",
    "cloome": "cloome",
    "subcell": "subcell",
    "cellprofiler": "cellprofiler",
}


def _style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.9,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "font.family": "DejaVu Sans",
        "savefig.bbox": "tight",
    })


def _ylabel(m: str) -> str:
    return f"\u2605 {DISPLAY.get(m, m)}" if m == OURS_KEY else DISPLAY.get(m, m)


def _load(path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _colored_barh(ax, models, values, *, fmt="{:.2f}"):
    """Horizontal bars colored by value; returns a ScalarMappable for a colorbar."""
    vmin, vmax = min(values), max(values)
    norm = Normalize(vmin=vmin - (vmax - vmin) * 0.18, vmax=vmax)
    for i, (m, v) in enumerate(zip(models, values)):
        is_ours = m == OURS_KEY
        ax.barh(i, v, color=CMAP(norm(v)),
                edgecolor="black" if is_ours else "#888888",
                linewidth=2.4 if is_ours else 0.7, zorder=3)
        ax.text(v, i, f"  {fmt.format(v)}", va="center", ha="left",
                fontsize=9, fontweight="bold" if is_ours else "normal", zorder=4)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([_ylabel(m) for m in models])
    for tick, m in zip(ax.get_yticklabels(), models):
        if m == OURS_KEY:
            tick.set_fontweight("bold")
            tick.set_color(CMAP(0.99))
    ax.tick_params(length=0)
    ax.margins(x=0.22)  # room for value labels
    return ScalarMappable(norm=norm, cmap=CMAP)


def plot_bbbc036(results_dir: str, out_dir: str) -> str:
    b = _load(os.path.join(results_dir, "bbbc036_moa.pkl"))
    models = sorted(b, key=lambda m: b[m]["geometric_mean_or"])
    gor = [b[m]["geometric_mean_or"] for m in models]
    fsig = [b[m]["fraction_significant"] for m in models]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), layout="constrained")
    sm1 = _colored_barh(axes[0], models, gor, fmt="{:.2f}")
    axes[0].set_title("Geometric-mean odds ratio")
    fig.colorbar(sm1, ax=axes[0], fraction=0.046, pad=0.02, label="odds ratio")

    sm2 = _colored_barh(axes[1], models, fsig, fmt="{:.3f}")
    axes[1].set_title("Fraction significant")
    axes[1].set_yticklabels([])
    fig.colorbar(sm2, ax=axes[1], fraction=0.046, pad=0.02, label="fraction")

    fig.suptitle("BBBC036 MoA enrichment  \u2014  higher is better  (\u2605 RobuDINO = your model)",
                 fontsize=14, fontweight="bold")
    path = os.path.join(out_dir, "bbbc036_moa_summary.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def _crispr(results_dir):
    out = {}
    for tag, fname in [("no_restriction", "crispr_no_restriction.pkl"),
                       ("not_same_batch", "crispr_not_same_batch.pkl")]:
        r = _load(os.path.join(results_dir, fname))
        dbs = sorted({db for m in r for db in r[m]})
        means = {m: float(np.mean([r[m][db]["geometric_mean_odds"] for db in dbs])) for m in r}
        out[tag] = (r, dbs, means)
    return out


def plot_crispr_summary(results_dir: str, out_dir: str) -> str:
    data = _crispr(results_dir)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), sharey=True, layout="constrained")
    for ax, tag in zip(axes, ["no_restriction", "not_same_batch"]):
        _, _, means = data[tag]
        models = sorted(means, key=means.get)
        sm = _colored_barh(ax, models, [means[m] for m in models], fmt="{:.2f}")
        ax.set_title(f"CRISPR \u2014 {tag}")
        ax.set_xlabel("Geometric-mean odds ratio (mean over 5 databases)")
        fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02, label="odds ratio")
    fig.suptitle("cpg-crispr pathway enrichment  \u2014  higher is better  (\u2605 RobuDINO = your model)",
                 fontsize=14, fontweight="bold")
    path = os.path.join(out_dir, "cpg_crispr_summary.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_crispr_per_database(results_dir: str, out_dir: str) -> str:
    data = _crispr(results_dir)
    allv = [r[m][db]["geometric_mean_odds"]
            for tag in data for (r, dbs, _) in [data[tag]] for m in r for db in dbs]
    norm = Normalize(vmin=min(allv), vmax=max(allv))

    # Stack the two panels vertically so y-axis labels never collide.
    fig, axes = plt.subplots(2, 1, figsize=(9, 12), layout="constrained")
    ims = None
    for ax, tag in zip(axes, ["no_restriction", "not_same_batch"]):
        r, dbs, means = data[tag]
        models = sorted(r, key=lambda m: means[m], reverse=True)
        mat = np.array([[r[m][db]["geometric_mean_odds"] for db in dbs] for m in models])
        ims = ax.imshow(mat, aspect="auto", cmap=CMAP, norm=norm)
        ax.set_xticks(range(len(dbs)))
        ax.set_xticklabels(dbs, rotation=20, ha="right")
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels([_ylabel(m) for m in models])
        for tick, m in zip(ax.get_yticklabels(), models):
            if m == OURS_KEY:
                tick.set_fontweight("bold")
        ax.tick_params(length=0)
        ax.set_title(f"CRISPR \u2014 {tag}")
        if OURS_KEY in models:
            ri = models.index(OURS_KEY)
            ax.add_patch(plt.Rectangle((-0.5, ri - 0.5), len(dbs), 1, fill=False,
                                       edgecolor="black", lw=2.5, zorder=5))
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat[i, j]:.1f}", ha="center", va="center", fontsize=8,
                        color="black" if norm(mat[i, j]) > 0.55 else "#222222")
    fig.colorbar(ims, ax=axes, fraction=0.04, pad=0.02,
                 label="geometric-mean odds ratio")
    fig.suptitle("cpg-crispr per-database enrichment  \u2014  higher is better  (\u2605 RobuDINO = your model)",
                 fontsize=14, fontweight="bold")
    path = os.path.join(out_dir, "cpg_crispr_per_database.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def _crispr_nr_nsb(results_dir):
    """Per-model NR/NSB values (mean over the 5 databases) for both metrics."""
    nr = _load(os.path.join(results_dir, "crispr_no_restriction.pkl"))
    nsb = _load(os.path.join(results_dir, "crispr_not_same_batch.pkl"))
    models = sorted(set(nr) & set(nsb))

    def agg(r, m, key):
        return float(np.mean([r[m][db][key] for db in r[m]]))

    data = {m: {
        "nr_frac": agg(nr, m, "fraction_significant"),
        "nr_geom": agg(nr, m, "geometric_mean_odds"),
        "nsb_frac": agg(nsb, m, "fraction_significant"),
        "nsb_geom": agg(nsb, m, "geometric_mean_odds"),
    } for m in models}
    return models, data


def plot_crispr_nr_nsb(results_dir: str, out_dir: str) -> str:
    """Paper Figure-3 style: two panels (Fraction Significant, Geometric Mean OR),
    each a horizontal grouped bar chart with No-Restriction (NR) and
    Not-Same-Batch (NSB) bars per method (mean over the 5 databases)."""
    models, data = _crispr_nr_nsb(results_dir)
    c_nr, c_nsb = CMAP(0.82), CMAP(0.38)  # NR = deep, NSB = pale

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2), layout="constrained")
    panels = [("Fraction Significant", "frac", "{:.3f}"),
              ("Geometric Mean OR", "geom", "{:.2f}")]
    for ax, (title, suf, fmt) in zip(axes, panels):
        # Sort each panel by its own NR value (ascending -> best at top), like the paper.
        order = sorted(models, key=lambda m: data[m][f"nr_{suf}"])
        h = 0.38
        for i, m in enumerate(order):
            ours = m == OURS_KEY
            ec = "black" if ours else "#888888"
            lw = 2.0 if ours else 0.5
            nrv, nsbv = data[m][f"nr_{suf}"], data[m][f"nsb_{suf}"]
            ax.barh(i + h / 2, nrv, height=h, color=c_nr, edgecolor=ec, linewidth=lw,
                    zorder=3, label="No Restriction" if i == 0 else None)
            ax.barh(i - h / 2, nsbv, height=h, color=c_nsb, edgecolor=ec, linewidth=lw,
                    zorder=3, label="Not Same Batch" if i == 0 else None)
            ax.text(nrv, i + h / 2, " " + fmt.format(nrv), va="center", ha="left",
                    fontsize=8, fontweight="bold" if ours else "normal")
            ax.text(nsbv, i - h / 2, " " + fmt.format(nsbv), va="center", ha="left",
                    fontsize=8, fontweight="bold" if ours else "normal")
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([_ylabel(m) for m in order])
        for tick, m in zip(ax.get_yticklabels(), order):
            if m == OURS_KEY:
                tick.set_fontweight("bold")
                tick.set_color(CMAP(0.99))
        ax.tick_params(length=0)
        ax.margins(x=0.18)
        ax.set_title(title)
        if suf == "frac":  # legend only on the left panel (its lower-right is empty)
            ax.legend(loc="lower right", frameon=False, fontsize=9)
    fig.suptitle("cpg-CRISPR pathway enrichment (mean over 5 databases)  \u2014  higher is better  "
                 "(\u2605 RobuDINO = your model)", fontsize=14, fontweight="bold")
    path = os.path.join(out_dir, "cpg_crispr_nr_nsb.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def _cpg_moa_matrix(results_dir):
    """Return (models, paradigms, geom_or matrix) from the cpg-moa enrichment pkl.

    Uses the enrichment module's own aggregation so values match the benchmark.
    The first column is BBBC036 MoA geom-OR (fetched from bbbc036_moa.pkl); the
    remaining four are the cpg-moa paradigms. Missing cells are NaN.
    """
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for p in (repo, os.path.join(repo, "benchmarks", "enrichment", "cpg_moa")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import run_cpg_moa_enrichment as M  # noqa: E402

    res = _load(os.path.join(results_dir, "cpg_moa_enrichment.pkl"))
    models = list(res["global"].keys())
    columns = ["BBBC036 MoA", "cpg Global", "cpg Within-Source",
               "cpg Not-Same-Batch", "cpg Not-Same-Source"]

    # BBBC036 MoA geom-OR per model (from the BBBC split results), keyed by canonical name.
    bbbc_path = os.path.join(results_dir, "bbbc036_moa.pkl")
    bbbc = _load(bbbc_path) if os.path.exists(bbbc_path) else {}

    def bbbc_or(m):
        key = BBBC_FROM_CANONICAL.get(m)
        if key and key in bbbc:
            r = bbbc[key]
            return r.get("geometric_mean_or", r.get("geometric_mean_odds", np.nan))
        return np.nan

    def geom(m):
        g = res["global"][m]["geometric_mean_or"]
        ws = M.compute_aggregate(res["within_source"][m])["geometric_mean_or"]
        nsb = M.compute_aggregate_majority_vote(res["not_same_batch"][m])["geometric_mean_or"]
        nss = M.compute_aggregate_majority_vote(res["not_same_source"][m])["geometric_mean_or"]
        return [bbbc_or(m), g, ws, nsb, nss]

    mat = {m: geom(m) for m in models}
    # Sort by mean of the cpg paradigms (cols 1..) so ranking is unaffected by BBBC NaNs.
    order = sorted(models, key=lambda m: np.nanmean(mat[m][1:]), reverse=True)
    matrix = np.array([mat[m] for m in order], dtype=float)
    return order, columns, matrix


def plot_cpg_moa(results_dir: str, out_dir: str) -> str:
    models, columns, mat = _cpg_moa_matrix(results_dir)
    masked = np.ma.masked_invalid(mat)
    norm = Normalize(vmin=masked.min(), vmax=masked.max())
    cmap = CMAP.copy()
    cmap.set_bad("#dddddd")  # gray for missing cells

    fig, ax = plt.subplots(figsize=(10.5, 6.5), layout="constrained")
    im = ax.imshow(masked, aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels(columns, rotation=20, ha="right")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([
        f"\u2605 {CPGMOA_DISPLAY.get(m, m)}" if m == OURS_KEY else CPGMOA_DISPLAY.get(m, m)
        for m in models
    ])
    for tick, m in zip(ax.get_yticklabels(), models):
        if m == OURS_KEY:
            tick.set_fontweight("bold")
            tick.set_color(CMAP(0.99))
    ax.tick_params(length=0)
    # Separator between the BBBC036 column and the cpg-moa paradigms.
    ax.axvline(0.5, color="#333333", lw=1.5)
    if OURS_KEY in models:
        ri = models.index(OURS_KEY)
        ax.add_patch(plt.Rectangle((-0.5, ri - 0.5), len(columns), 1, fill=False,
                                   edgecolor="black", lw=2.5, zorder=5))
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isnan(v):
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=8, color="#888888")
            else:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                        color="black" if norm(v) > 0.55 else "#222222")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="geometric-mean odds ratio")
    fig.suptitle("MoA enrichment: BBBC036 + cpg-MoA  \u2014  higher is better  (\u2605 RobuDINO = your model)",
                 fontsize=14, fontweight="bold")
    path = os.path.join(out_dir, "cpg_moa_summary.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    _style()
    plots = [plot_bbbc036, plot_crispr_summary, plot_crispr_per_database, plot_crispr_nr_nsb]
    if os.path.exists(os.path.join(args.results_dir, "cpg_moa_enrichment.pkl")):
        plots.append(plot_cpg_moa)
    print("Wrote:")
    for fn in plots:
        print("  " + fn(args.results_dir, args.out_dir))


if __name__ == "__main__":
    main()
