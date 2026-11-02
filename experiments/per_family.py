"""Experiment 4: per-attack-family breakdown under severe class imbalance.

Macro-F1 is a single number; a security operator needs to know *which* families
are detected and which are missed. CIC-IDS2017 is extremely imbalanced (benign
dominates; the rarest families have tens of events), so an honest paper must
report per-family precision/recall/F1 and support, not just an aggregate.

This experiment runs AMLN at the tuned operating point and tabulates, for every
family, how well it is recovered -- separating the easy high-volume families
(DoS, PortScan, DDoS) from the genuinely hard rare ones. The two held-out
zero-day families are additionally scored by the surprise gate (novelty flag),
which is the correct target for a family the system was never allowed to label.

Outputs:
  * experiment_per_family.csv
  * paper_fig8_per_family.png
"""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import numpy as np

from amln.config import (TUNED_DECAY_RATE, TUNED_ENERGY_BUDGET,
                         TUNED_MATCH_SIMILARITY)
from amln.memory.adaptive_v2 import LandscapeV2Config
from experiments.common import ROOT, build_context

TUNED_MATCH_SIM = TUNED_MATCH_SIMILARITY
TUNED_BUDGET = TUNED_ENERGY_BUDGET
TUNED_DECAY = TUNED_DECAY_RATE


def _prf(y_true, y_pred, cls):
    tp = int(np.sum((y_pred == cls) & (y_true == cls)))
    fp = int(np.sum((y_pred == cls) & (y_true != cls)))
    fn = int(np.sum((y_pred != cls) & (y_true == cls)))
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


def run() -> list[dict]:
    ctx = build_context()
    land_cfg = replace(ctx.cfg.landscape, match_similarity=TUNED_MATCH_SIM)
    cfg2 = replace(LandscapeV2Config(), energy_budget=TUNED_BUDGET,
                   decay_rate=TUNED_DECAY)

    print("\nComputing per-family detection breakdown (tuned AMLN) ...")
    y_true, y_pred, anomaly = ctx.harness.run_amln_labeled(
        cfg2, land_cfg=land_cfg)

    names = ctx.names
    rows: list[dict] = []
    for cls in sorted(set(int(c) for c in np.unique(y_true))):
        support = int(np.sum(y_true == cls))
        prec, rec, f1 = _prf(y_true, y_pred, cls)
        name = names.get(cls, str(cls)).encode("ascii", "replace").decode("ascii")
        is_zero_day = cls in ctx.zero_day
        # For a held-out zero-day family, the meaningful score is the fraction
        # the surprise gate flagged as novel (it cannot recover the exact label).
        novelty_recall = (float(np.mean(anomaly[y_true == cls]))
                          if (is_zero_day and support) else None)
        rows.append({
            "family": name,
            "support": support,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "zero_day": is_zero_day,
            "novelty_recall": (round(novelty_recall, 4)
                               if novelty_recall is not None else ""),
        })

    for r in rows:
        tag = "  [zero-day]" if r["zero_day"] else ""
        extra = (f" novelty_recall={r['novelty_recall']}"
                 if r["novelty_recall"] != "" else "")
        print(f"  {r['family']:<28} n={r['support']:>6}  P={r['precision']:.3f} "
              f"R={r['recall']:.3f} F1={r['f1']:.3f}{tag}{extra}")

    out_csv = ROOT / "experiment_per_family.csv"
    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {out_csv}")

    _plot(rows)
    return rows


def _plot(rows: list[dict]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib missing -> skipped per-family plot)")
        return

    plt.rcParams.update({"font.family": "serif", "font.size": 9,
                         "figure.dpi": 300})
    # Sort by support (descending) so the imbalance is visible left-to-right.
    rows = sorted(rows, key=lambda r: r["support"], reverse=True)
    fams = [r["family"] for r in rows]
    f1 = [r["f1"] for r in rows]
    colors = ["#c1272d" if r["zero_day"] else "#2a5783" for r in rows]

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    y = range(len(fams))
    ax.barh(list(y), f1, color=colors, edgecolor="#222", linewidth=0.5)
    for i, r in enumerate(rows):
        ax.text(min(0.98, r["f1"] + 0.01), i,
                f" {r['f1']:.2f} (n={r['support']})", va="center", fontsize=7)
    ax.set_yticks(list(y))
    ax.set_yticklabels(fams, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Per-family F1 (dimensionless); red = held-out zero-day")
    ax.set_title("Per-family detection under class imbalance (tuned AMLN)")
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    out = ROOT / "paper_fig8_per_family.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    run()
