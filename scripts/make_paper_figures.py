"""Generate IEEE paper figures (Fig. 1-4) for the AMLN paper.

Fig. 1  System pipeline block diagram            -> paper_fig1_pipeline.png
Fig. 2  Memory landscape schematic               -> paper_fig2_landscape.png
Fig. 3  Detection quality bar chart (real data)  -> paper_fig3_quality.png
Fig. 4  Accuracy vs memory scatter (real data)   -> paper_fig4_tradeoff.png

Figs. 3-4 read benchmark_results.csv (produced by run_benchmark.py).
Run:  python scripts/make_paper_figures.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "benchmark_results.csv"

# Grayscale-friendly, print-safe palette; proposed method highlighted in red.
C_PROP = "#c1272d"
C_BASE = "#2a5783"
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "figure.dpi": 300,
})


# --------------------------------------------------------------------------
def _load_rows():
    with CSV.open(newline="") as fh:
        return list(csv.DictReader(fh))


# --------------------------------------------------------------------------
def fig1_pipeline(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.0, 7.0))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 15)
    ax.axis("off")

    blocks = [
        (13.0, "Raw CIC-IDS2017 flows\n(network capture, 78 features)", "#eaeef4"),
        (10.9, "Stream construction\n(stratified, chronological,\ncausal robust scaling)", "#eaeef4"),
        (8.5, "Transformer behavioral\nautoencoder", "#dfe9f3"),
        (5.6, "Adaptive Memory Landscape\n(surprise gate, energy budget,\ndrift-aware split/merge)", "#f6e2e4"),
    ]
    boxw, boxh, cx = 6.6, 1.35, 5.2
    for y, label, color in blocks:
        box = FancyBboxPatch((cx - boxw / 2, y - boxh / 2), boxw, boxh,
                             boxstyle="round,pad=0.08,rounding_size=0.15",
                             linewidth=1.2, edgecolor="#333", facecolor=color)
        ax.add_patch(box)
        ax.text(cx, y, label, ha="center", va="center", fontsize=9)

    # vertical arrows between the first four blocks
    ys = [b[0] for b in blocks]
    for y_top, y_bot in zip(ys[:-1], ys[1:]):
        ax.add_patch(FancyArrowPatch((cx, y_top - boxh / 2),
                                     (cx, y_bot + boxh / 2),
                                     arrowstyle="-|>", mutation_scale=14,
                                     linewidth=1.3, color="#333"))

    # autoencoder dual output annotation (placed on the arrow, clear of boxes)
    ax.text(cx + boxw / 2 + 0.25, 7.05,
            "64-d embedding +\nreconstruction error\n(surprise signal)",
            fontsize=7.0, style="italic", ha="left", va="center", color="#444")

    # two parallel outputs from the landscape
    out_y = 2.6
    outs = [("Live dashboard\n(real-time\nvisualization)", 2.2),
            ("Benchmark harness\n(prequential\nevaluation)", 8.2)]
    obw = 3.9
    for label, ox in outs:
        box = FancyBboxPatch((ox - obw / 2, out_y - boxh / 2 - 0.25), obw,
                             boxh + 0.5,
                             boxstyle="round,pad=0.08,rounding_size=0.15",
                             linewidth=1.1, edgecolor="#333", facecolor="#e8f0e8")
        ax.add_patch(box)
        ax.text(ox, out_y, label, ha="center", va="center", fontsize=8)
        ax.add_patch(FancyArrowPatch((cx, 5.6 - boxh / 2),
                                     (ox, out_y + boxh / 2 + 0.25),
                                     arrowstyle="-|>", mutation_scale=13,
                                     linewidth=1.2, color="#333"))

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
def fig2_landscape(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    x = np.linspace(0, 12, 1000)

    def bump(c, w, h):
        return h * np.exp(-((x - c) ** 2) / (2 * w ** 2))

    hill1 = bump(2.0, 0.55, 3.0)
    hill2 = bump(5.0, 0.6, 2.6)
    slope = bump(8.0, 0.5, 1.2)
    valley = bump(10.6, 0.45, 0.35)
    y = hill1 + hill2 + slope + valley

    ax.plot(x, y, color="#2a5783", linewidth=2.0)
    ax.fill_between(x, y, color="#cdd9e8", alpha=0.6)

    hill_th, valley_th = 2.0, 0.6
    ax.axhline(hill_th, ls="--", color=C_PROP, linewidth=1.1)
    ax.axhline(valley_th, ls=":", color="#777", linewidth=1.1)
    ax.text(12.05, hill_th, " hill threshold", color=C_PROP, va="center", fontsize=8)
    ax.text(12.05, valley_th, " valley threshold", color="#555", va="center", fontsize=8)

    ax.annotate("Hill\n(stable long-term)", xy=(2.0, 3.0), xytext=(2.0, 3.65),
                ha="center", fontsize=8.5,
                arrowprops=dict(arrowstyle="->", color="#333"))
    ax.annotate("Hill", xy=(5.0, 2.6), xytext=(5.0, 3.2), ha="center", fontsize=8.5,
                arrowprops=dict(arrowstyle="->", color="#333"))
    ax.annotate("Slope\n(forming)", xy=(8.0, 1.2), xytext=(8.0, 2.1), ha="center",
                fontsize=8.5, arrowprops=dict(arrowstyle="->", color="#333"))
    ax.annotate("Valley\n(forgotten)", xy=(10.6, 0.35), xytext=(10.6, 1.2),
                ha="center", fontsize=8.5,
                arrowprops=dict(arrowstyle="->", color="#333"))

    # dynamics arrows
    ax.annotate("", xy=(5.0, 2.75), xytext=(8.0, 1.35),
                arrowprops=dict(arrowstyle="->", color=C_PROP, lw=1.3,
                                connectionstyle="arc3,rad=-0.3"))
    ax.text(6.7, 2.55, "reinforce /\nconsolidate", color=C_PROP, fontsize=7.5,
            ha="center")
    ax.annotate("", xy=(9.3, 0.5), xytext=(8.3, 1.1),
                arrowprops=dict(arrowstyle="->", color="#555", lw=1.2,
                                connectionstyle="arc3,rad=0.3"))
    ax.text(9.1, 0.95, "decay", color="#555", fontsize=7.5)

    ax.set_xlabel("Behavioral embedding position (projected)")
    ax.set_ylabel("Memory height (stability)")
    ax.set_ylim(0, 4.3)
    ax.set_xlim(0, 13.4)
    ax.set_xticks([])
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
def fig3_quality(rows, out: Path) -> None:
    meths = [r["method"] for r in rows]
    f1 = [float(r["macro_f1"]) for r in rows]
    # Both AMLN operating points highlighted; tuned (*) slightly darker red.
    colors = [(C_PROP if "AMLN" in m else C_BASE) for m in meths]

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    bars = ax.bar(range(len(meths)), f1, color=colors, edgecolor="#222",
                  linewidth=0.6, width=0.62)
    for b, v in zip(bars, f1):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(meths)))
    ax.set_xticklabels(meths, rotation=22, ha="right", fontsize=8.5)
    ax.set_ylabel("Macro F1 score (dimensionless)")
    ax.set_xlabel("Method")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
def fig4_tradeoff(rows, out: Path) -> None:
    meths = [r["method"] for r in rows]
    f1 = [float(r["macro_f1"]) for r in rows]
    kb = [float(r["footprint_kb"]) for r in rows]
    colors = [C_PROP if "AMLN" in m else C_BASE for m in meths]

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.scatter(kb, f1, c=colors, s=130, edgecolor="#222", linewidth=0.7, zorder=3)
    offsets = {
        "AMLN-V2": (6, -16), "AMLN-V2*": (8, 6), "StoreAll": (-10, -16),
        "SlidingWindow(5000)": (8, 6), "Reservoir(5000)": (8, -14),
        "OnlineKMeans(32)": (-6, 8),
    }
    for m, x, y in zip(meths, kb, f1):
        dx, dy = offsets.get(m, (6, 6))
        ax.annotate(m, (x, y), fontsize=8, xytext=(dx, dy),
                    textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("Memory footprint (KB)")
    ax.set_ylabel("Macro F1 score (dimensionless)")
    ax.set_ylim(0, 1.0)
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    ax.annotate("better", xy=(kb[0], 0.92), fontsize=8, style="italic",
                color="#c1272d", ha="left")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
def main() -> None:
    rows = _load_rows()
    fig1_pipeline(ROOT / "paper_fig1_pipeline.png")
    print("Saved paper_fig1_pipeline.png")
    fig2_landscape(ROOT / "paper_fig2_landscape.png")
    print("Saved paper_fig2_landscape.png")
    fig3_quality(rows, ROOT / "paper_fig3_quality.png")
    print("Saved paper_fig3_quality.png")
    fig4_tradeoff(rows, ROOT / "paper_fig4_tradeoff.png")
    print("Saved paper_fig4_tradeoff.png")


if __name__ == "__main__":
    main()
