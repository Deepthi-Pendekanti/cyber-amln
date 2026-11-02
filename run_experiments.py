"""Run the full conference-grade experiment suite for AMLN.

This reproduces every quantitative claim added to strengthen the paper:

  1. Budget / capacity sweep  -> accuracy-vs-memory Pareto frontier (Fig. 5)
  2. Mechanism ablation       -> contribution of each novel part   (Fig. 6)
  3. Bounded-memory proof     -> footprint vs. stream length       (Fig. 7)
  4. Per-family breakdown     -> detection under class imbalance   (Fig. 8)

The encoder is pre-trained once and cached (data/exp_cache/encoder.pt); every
experiment reuses it, so the whole suite runs quickly after the first call.
CSV + PNG artifacts are written to the project root.

Run:  python run_experiments.py
"""

from __future__ import annotations

import torch

from experiments import ablation, bounded_memory, budget_sweep, per_family
from experiments.common import build_context


def main() -> None:
    torch.set_num_threads(max(1, torch.get_num_threads() or 4))
    # Warm the encoder cache once up front so each experiment is fast.
    print("=" * 72)
    print("Preparing shared encoder (cached after first run) ...")
    print("=" * 72)
    build_context(verbose=True)

    print("\n" + "=" * 72 + "\n[1/4] BUDGET / CAPACITY SWEEP\n" + "=" * 72)
    budget_sweep.run()

    print("\n" + "=" * 72 + "\n[2/4] MECHANISM ABLATION\n" + "=" * 72)
    ablation.run()

    print("\n" + "=" * 72 + "\n[3/4] BOUNDED-MEMORY PROOF\n" + "=" * 72)
    bounded_memory.run()

    print("\n" + "=" * 72 + "\n[4/4] PER-FAMILY BREAKDOWN\n" + "=" * 72)
    per_family.run()

    print("\n" + "=" * 72)
    print("All experiments complete. Artifacts written to project root:")
    print("  experiment_budget_sweep.csv   paper_fig5_pareto.png")
    print("  experiment_ablation.csv       paper_fig6_ablation.png")
    print("  experiment_bounded_memory.csv paper_fig7_bounded_memory.png")
    print("  experiment_per_family.csv     paper_fig8_per_family.png")
    print("=" * 72)


if __name__ == "__main__":
    main()
