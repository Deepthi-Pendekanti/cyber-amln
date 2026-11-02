"""Experiment 3: empirical proof that AMLN's memory stays bounded.

The paper's headline property is a footprint that is *bounded by construction*
and roughly constant regardless of stream length, which a store-everything or
growing buffer cannot provide. Section III proves the bound K <= E_max / h_min
analytically; this experiment demonstrates it empirically by tracking the live
memory footprint of AMLN against StoreAll as the stream advances.

AMLN should trace a flat line (constant tens of KB) while StoreAll grows
linearly into tens of MB. The crossover makes the memory argument visual and
undeniable.

Outputs:
  * experiment_bounded_memory.csv
  * paper_fig7_bounded_memory.png
"""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

from amln.config import (TUNED_DECAY_RATE, TUNED_ENERGY_BUDGET,
                         TUNED_MATCH_SIMILARITY)
from amln.memory.adaptive_v2 import AdaptiveMemoryLandscapeV2, LandscapeV2Config
from experiments.common import ROOT, build_context

TUNED_MATCH_SIM = TUNED_MATCH_SIMILARITY
TUNED_BUDGET = TUNED_ENERGY_BUDGET
TUNED_DECAY = TUNED_DECAY_RATE
SNAPSHOT_EVERY = 2000


def run() -> list[dict]:
    ctx = build_context()
    h = ctx.harness
    land_cfg = replace(ctx.cfg.landscape, match_similarity=TUNED_MATCH_SIM)
    cfg2 = replace(LandscapeV2Config(), energy_budget=TUNED_BUDGET,
                   decay_rate=TUNED_DECAY)
    land = AdaptiveMemoryLandscapeV2(h.latent_dim, land_cfg, cfg2)

    bytes_per_item = h.bytes_per_item
    n = len(h._embeddings)
    rows: list[dict] = []
    E_max = cfg2.energy_budget
    h_min = ctx.cfg.landscape.valley_threshold
    bound_nodes = E_max / h_min   # analytic ceiling K <= E_max / h_min

    print("\nTracking live footprint vs. stream position ...")
    for i in range(n):
        land.observe(h._embeddings[i], float(h.replayer.severity[i]),
                     int(h.replayer.labels[i]), recon_error=float(h._recon[i]))
        if (i + 1) % SNAPSHOT_EVERY == 0 or (i + 1) == n:
            seen = i + 1
            amln_nodes = len(land)
            rows.append({
                "events_seen": seen,
                "amln_nodes": amln_nodes,
                "amln_kb": round(amln_nodes * bytes_per_item / 1024.0, 3),
                "amln_energy": round(land.total_energy(), 2),
                "storeall_items": seen,
                "storeall_kb": round(seen * bytes_per_item / 1024.0, 1),
                "analytic_bound_nodes": round(bound_nodes, 1),
            })

    last = rows[-1]
    print(f"  final: AMLN {last['amln_nodes']} nodes "
          f"({last['amln_kb']} KB) vs StoreAll {last['storeall_items']:,} items "
          f"({last['storeall_kb']} KB); analytic ceiling {bound_nodes:.0f} nodes")

    out_csv = ROOT / "experiment_bounded_memory.csv"
    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {out_csv}")

    _plot(rows, bound_nodes, bytes_per_item)
    return rows


def _plot(rows, bound_nodes, bytes_per_item) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib missing -> skipped bounded-memory plot)")
        return

    plt.rcParams.update({"font.family": "serif", "font.size": 10,
                         "figure.dpi": 300})
    seen = [r["events_seen"] for r in rows]
    amln_kb = [r["amln_kb"] for r in rows]
    store_kb = [r["storeall_kb"] for r in rows]
    bound_kb = bound_nodes * bytes_per_item / 1024.0

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(seen, store_kb, color="#2a5783", linewidth=1.8,
            label="StoreAll (unbounded)")
    ax.plot(seen, amln_kb, color="#c1272d", linewidth=1.8,
            label="AMLN (bounded)")
    ax.axhline(bound_kb, color="#555", linestyle="--", linewidth=1.1,
               label=f"analytic bound E_max/h_min ({bound_kb:.1f} KB)")
    ax.set_yscale("log")
    ax.set_xlabel("Events streamed")
    ax.set_ylabel("Memory footprint (KB, log scale)")
    ax.set_title("Memory footprint vs. stream length")
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, loc="center right")
    fig.tight_layout()
    out = ROOT / "paper_fig7_bounded_memory.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    run()
