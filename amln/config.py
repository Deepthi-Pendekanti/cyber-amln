"""Central configuration for the AMLN framework.

Grouping the hyper-parameters here keeps the encoder, the adaptive memory
landscape, and the training loop in sync and makes experiments reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EncoderConfig:
    """Transformer encoder / decoder settings."""

    num_features: int = 32          # raw structured features per cyber event
    d_model: int = 64               # transformer hidden width
    latent_dim: int = 64            # the 64-d behavioral embedding (the paper's core)
    num_heads: int = 4
    num_layers: int = 2
    ff_dim: int = 128
    dropout: float = 0.1


@dataclass
class LandscapeConfig:
    """Adaptive memory landscape dynamics.

    These constants define how memories are reinforced, how they decay, when
    they graduate from slope (short-term) to hill (long-term), and when they
    fall into a valley and are forgotten.
    """

    # A new observation reinforces an existing memory only if its cosine
    # similarity to the nearest centroid exceeds this value; otherwise a fresh
    # slope memory is created.
    match_similarity: float = 0.85

    # Centroids merge when their pairwise similarity exceeds this (related
    # attack families collapsing into one hill).
    merge_similarity: float = 0.95

    # Online centroid update rate (weighted move toward new observation).
    centroid_lr: float = 0.15

    # Reinforcement gain: how much a matched, severe observation raises height.
    reinforce_gain: float = 1.0

    # Exponential forgetting rate applied per unit of elapsed time.
    decay_rate: float = 0.05

    # Height thresholds separating the three landscape regions.
    hill_threshold: float = 2.0     # >= hill  -> stable long-term memory
    valley_threshold: float = 0.25  # <  valley -> forgotten / pruned

    # Weighting of the importance score components.
    w_frequency: float = 0.4
    w_severity: float = 0.4
    w_recency: float = 0.2


@dataclass
class TrainConfig:
    """Autoencoder pre-training + continual-learning settings."""

    epochs: int = 12
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-5
    seed: int = 42
    device: str = "cpu"             # set to "cuda" if available

    # Denoising augmentation: during pre-training, randomly mask up to this
    # fraction of input features (per sample) while reconstructing the clean
    # full event. This makes the encoder robust to partial observations, which
    # is what enables behavior-driven pattern completion from a fragment.
    denoise_mask: float = 0.6


@dataclass
class AMLNConfig:
    """Top-level bundle passed around the framework."""

    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    landscape: LandscapeConfig = field(default_factory=LandscapeConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


# --------------------------------------------------------------------------
# Named operating points.
#
# The capacity sweep (experiments/budget_sweep.py) identified a clear optimum
# on the accuracy-vs-memory frontier: a higher match-similarity (finer memory
# resolution, ~one hill per family) with a gentle decay. Exposing it as a named
# config makes the tuned result a reproducible artifact rather than scattered
# magic numbers.
# --------------------------------------------------------------------------

#: Match-similarity for the tuned ("AMLN-V2*") operating point.
TUNED_MATCH_SIMILARITY: float = 0.96
#: Energy budget for the tuned operating point.
TUNED_ENERGY_BUDGET: float = 400.0
#: V2 decay rate for the tuned operating point.
TUNED_DECAY_RATE: float = 0.002


def tuned_landscape_config(base: LandscapeConfig | None = None) -> LandscapeConfig:
    """Return a :class:`LandscapeConfig` at the tuned memory resolution."""
    from dataclasses import replace
    base = base or LandscapeConfig()
    return replace(base, match_similarity=TUNED_MATCH_SIMILARITY)
