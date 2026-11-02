"""Experiment 2: mechanism ablation.

AMLN's contribution is three co-operating mechanisms. A reviewer rightly asks:
does each one actually matter, or is the result carried by a single part? This
experiment answers that by disabling each mechanism in turn (while holding the
tuned operating point fixed) and measuring the change in detection quality and
memory behaviour:

  * -surprise : plasticity and zero-day flagging no longer scale with surprise
                (fixed learning rate, no novelty signal).
  * -energy   : the energy budget no longer bounds total stored height
                (memory can grow -- this is the bound we claim to provide).
  * -drift    : drifting clusters are never split into sibling memories.

The full model is included as the reference. A mechanism "matters" if removing
it degrades a metric we care about (macro-F1, zero-day recall) or breaks the
bounded-memory guarantee (peak nodes).

Outputs:
  * experiment_ablation.csv
  * paper_fig6_ablation.png
"""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

from amln.config import (TUNED_DECAY_RATE, TUNED_ENERGY_BUDGET,
                         TUNED_MATCH_SIMILARITY)
from amln.memory.adaptive_v2 import (AdaptiveMemoryLandscapeV2,
                                     LandscapeV2Config)
from experiments.common import ROOT, build_context

# Tuned operating point found by the capacity sweep (match-sim 0.96, gentle
# decay). Ablations are run at this point so the comparison is apples-to-apples.
TUNED_MATCH_SIM = TUNED_MATCH_SIMILARITY
TUNED_BUDGET = TUNED_ENERGY_BUDGET
TUNED_DECAY = TUNED_DECAY_RATE

ALL = set(AdaptiveMemoryLandscapeV2.ALL_MECHANISMS)
VARIANTS = [
    ("Full AMLN", ALL),
    ("-surprise", ALL - {"surprise"}),
    ("-energy", ALL - {"energy"}),
    ("-drift", ALL - {"drift"}),
]


def run() -> list[dict]:
    ctx = build_context()
    land_cfg = replace(ctx.cfg.landscape, match_similarity=TUNED_MATCH_SIM)
    cfg2 = replace(LandscapeV2Config(), energy_budget=TUNED_BUDGET,
                   decay_rate=TUNED_DECAY)

    rows: list[dict] = []
    print("\nAblating mechanisms at the tuned operating point ...")
    for name, mechs in VARIANTS:
        res = ctx.harness.run_amln(cfg2, name=name, land_cfg=land_cfg,
                                   mechanisms=mechs)
        row = {
            "variant": name,
            "accuracy": round(res.accuracy, 4),
            "macro_f1": round(res.macro_f1, 4),
            "attack_recall": round(res.attack_recall, 4),
            "zero_day_recall": round(res.zero_day_recall, 4),
            "footprint_kb": round(res.footprint_kb, 2),
            "peak_items": res.peak_items,
            "final_items": res.final_items,
            "splits": res.extra.get("splits", 0),
            "final_energy": res.extra.get("final_energy", 0.0),
        }
        rows.append(row)
        print(f"  {name:<11} F1={row['macro_f1']:.3f} acc={row['accuracy']:.3f} "
              f"zd={row['zero_day_recall']:.3f} peak={row['peak_items']:>4} "
              f"KB={row['footprint_kb']:>7} splits={row['splits']}")

    out_csv = ROOT / "experiment_ablation.csv"
    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {out_csv}")

    _plot(rows)
    _report(rows)
    return rows


def _plot(rows: list[dict]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("(matplotlib missing -> skipped ablation plot)")
        return

    plt.rcParams.update({"font.family": "serif", "font.size": 10,
                         "figure.dpi": 300})
    variants = [r["variant"] for r in rows]
    f1 = [r["macro_f1"] for r in rows]
    zd = [r["zero_day_recall"] for r in rows]
    x = np.arange(len(variants))
    w = 0.38

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    b1 = ax.bar(x - w / 2, f1, w, label="Macro F1", color="#2a5783",
                edgecolor="#222", linewidth=0.6)
    b2 = ax.bar(x + w / 2, zd, w, label="Zero-day recall", color="#c1272d",
                edgecolor="#222", linewidth=0.6)
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.012,
                    f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(variants, fontsize=9)
    ax.set_ylabel("Score (dimensionless)")
    ax.set_xlabel("Ablated configuration")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = ROOT / "paper_fig6_ablation.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def _report(rows: list[dict]) -> None:
    full = next(r for r in rows if r["variant"] == "Full AMLN")
    print("\n" + "=" * 72)
    print("ABLATION SUMMARY (delta vs. Full AMLN)")
    print("=" * 72)
    for r in rows:
        if r["variant"] == "Full AMLN":
            continue
        d_f1 = r["macro_f1"] - full["macro_f1"]
        d_zd = r["zero_day_recall"] - full["zero_day_recall"]
        d_kb = r["footprint_kb"] - full["footprint_kb"]
        print(f"  {r['variant']:<11} dF1={d_f1:+.3f}  dZeroDay={d_zd:+.3f}  "
              f"dFootprint={d_kb:+.2f} KB  (peak {r['peak_items']} nodes)")


if __name__ == "__main__":
    run()
