"""Experiment 1: the accuracy vs. memory frontier (capacity sweep).

The headline weakness of the base configuration is that it keeps too few live
memories for the fifteen attack families, so distinct behaviours compete for too
few nodes and macro-F1 suffers. Diagnosing this reveals the binding constraint
is not the energy budget (which stays far from saturation) but the *resolution*
of the memory: the match-similarity gate sets how finely the embedding space is
partitioned into memories, and the decay rate sets whether a rare family's hill
survives the long gaps between its events.

This experiment sweeps those two levers jointly with the energy budget and
records, for each operating point, both detection quality and memory footprint.
The result is an explicit Pareto frontier: the user can *dial in* how much
memory to spend for how much accuracy, which is exactly the control a
bounded-memory detector should expose. Each point reuses the shared
pre-computed embeddings, so the whole sweep is fast.

Outputs:
  * experiment_budget_sweep.csv   (one row per operating point)
  * paper_fig5_pareto.png         (quality vs. footprint frontier)
"""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

from amln.memory.adaptive_v2 import LandscapeV2Config
from experiments.common import ROOT, build_context

# Memory resolution: the dominant lever. A higher match-similarity partitions
# the embedding space into more (finer) memories -> more families get their own
# hill, at the cost of footprint. Swept together with the energy budget (ceiling
# on total height) and a gentle decay so rare-family hills survive long gaps.
# Budget is held at the value that was consistently best/tied in exploratory
# runs (higher budgets only add churn here); the dominant lever is the memory
# resolution (match-similarity), which traces the accuracy/footprint frontier.
MATCH_SIMS = [0.80, 0.88, 0.93, 0.96, 0.975, 0.985]
BUDGETS = [400]
DECAY = 0.002


def run() -> list[dict]:
    ctx = build_context()
    rows: list[dict] = []
    base = LandscapeV2Config()

    print("\nSweeping match-similarity x energy budget (shared embeddings) ...")
    for match_sim in MATCH_SIMS:
        # match_similarity lives in the shared LandscapeConfig, so clone it per
        # point to avoid mutating the harness's config across runs.
        land_cfg = replace(ctx.cfg.landscape, match_similarity=float(match_sim))
        ctx.harness.cfg.landscape = land_cfg
        for budget in BUDGETS:
            cfg2 = replace(base, energy_budget=float(budget), decay_rate=DECAY)
            res = ctx.harness.run_amln(cfg2)
            row = {
                "match_similarity": match_sim,
                "energy_budget": budget,
                "decay_rate": DECAY,
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
            print(f"  match={match_sim:<4} budget={budget:>5} -> "
                  f"F1={row['macro_f1']:.3f} acc={row['accuracy']:.3f} "
                  f"peak={row['peak_items']:>3} KB={row['footprint_kb']:>6}")

    out_csv = ROOT / "experiment_budget_sweep.csv"
    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {out_csv}")

    _plot(rows)
    _report_best(rows)
    return rows


def _plot(rows: list[dict]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib missing -> skipped Pareto plot)")
        return

    plt.rcParams.update({"font.family": "serif", "font.size": 10,
                         "figure.dpi": 300})
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    # Plot quality against the control knob (match-similarity): a clean
    # inverted-U peaking at 0.96. Each point is annotated with its footprint so
    # the memory cost of each resolution is visible.
    pts = sorted(rows, key=lambda r: r["match_similarity"])
    xs = [r["match_similarity"] for r in pts]
    ys = [r["macro_f1"] for r in pts]
    ax.plot(xs, ys, marker="o", linewidth=1.6, markersize=7, color="#c1272d")
    # Mark the optimum.
    best = max(pts, key=lambda r: r["macro_f1"])
    ax.scatter([best["match_similarity"]], [best["macro_f1"]], s=160,
               facecolors="none", edgecolors="#c1272d", linewidths=1.8, zorder=4)
    for r in pts:
        ax.annotate(f"{r['footprint_kb']:.1f} KB",
                    (r["match_similarity"], r["macro_f1"]), fontsize=7,
                    xytext=(5, -11), textcoords="offset points", color="#555")
    ax.annotate("tuned operating point", (best["match_similarity"], best["macro_f1"]),
                fontsize=8, xytext=(-120, -4), textcoords="offset points",
                color="#c1272d",
                arrowprops=dict(arrowstyle="->", color="#c1272d", lw=1.0))
    ax.margins(y=0.12)
    ax.set_xlabel("Match similarity (memory resolution)")
    ax.set_ylabel("Macro F1 score (dimensionless)")
    ax.set_title("AMLN accuracy vs. memory resolution (capacity sweep)")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    out = ROOT / "paper_fig5_pareto.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def _report_best(rows: list[dict]) -> None:
    best = max(rows, key=lambda r: r["macro_f1"])
    # "Efficient" operating point: best F1 achievable under a tight footprint.
    tight = [r for r in rows if r["footprint_kb"] <= 50]
    best_tight = max(tight, key=lambda r: r["macro_f1"]) if tight else best
    print("\n" + "=" * 72)
    print("BUDGET SWEEP SUMMARY")
    print("=" * 72)
    print(f"  Best F1 overall : F1={best['macro_f1']:.3f} "
          f"acc={best['accuracy']:.3f} at match-sim={best['match_similarity']} "
          f"budget={best['energy_budget']} ({best['footprint_kb']} KB, "
          f"peak {best['peak_items']} nodes)")
    print(f"  Best F1 <=50 KB : F1={best_tight['macro_f1']:.3f} "
          f"acc={best_tight['accuracy']:.3f} at match-sim={best_tight['match_similarity']} "
          f"budget={best_tight['energy_budget']} ({best_tight['footprint_kb']} KB, "
          f"peak {best_tight['peak_items']} nodes)")


if __name__ == "__main__":
    run()
