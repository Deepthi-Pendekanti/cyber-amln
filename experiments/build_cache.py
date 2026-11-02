"""One-time: pre-train the encoder and cache its weights for fast experiments.

After this runs once, every experiment loads the cached encoder instantly
instead of re-training for 12 epochs.
"""
from __future__ import annotations

import torch

from experiments.common import ENCODER_CKPT, build_context


def main() -> None:
    # Cap threads to avoid oversubscription slowing the CPU training loop.
    torch.set_num_threads(max(1, (torch.get_num_threads() or 4)))
    build_context(verbose=True, use_cache=True)
    print(f"\nEncoder cache ready at {ENCODER_CKPT}")


if __name__ == "__main__":
    main()
