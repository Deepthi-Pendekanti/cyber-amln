"""Shared setup for all AMLN experiments.

Loading the stream and pre-training the encoder is the expensive part. Doing it
once and sharing the resulting :class:`ExperimentContext` (which also builds a
:class:`BenchmarkHarness` holding the pre-computed embeddings + reconstruction
errors) keeps every downstream experiment fast and guarantees they all see the
*exact same* embeddings -- essential for a fair sweep/ablation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from amln.benchmark.harness import BenchmarkHarness
from amln.config import AMLNConfig
from amln.models.encoder import BehaviorAutoencoder
from amln.streaming import StreamReplayer
from amln.training import pretrain_autoencoder

ROOT = Path(__file__).resolve().parent.parent
STREAM_NPZ = ROOT / "data" / "stream.npz"
CICIDS_DIR = ROOT / "data" / "raw" / "cicids2017"
CACHE_DIR = ROOT / "data" / "exp_cache"
ENCODER_CKPT = CACHE_DIR / "encoder.pt"
EMBED_CACHE = CACHE_DIR / "embeddings.npz"


@dataclass
class ExperimentContext:
    """Everything an experiment needs, built once."""

    cfg: AMLNConfig
    harness: BenchmarkHarness
    replayer: StreamReplayer
    names: dict          # label id -> family name
    zero_day: set        # held-out family ids
    warmup: int


def _load_stream(cfg: AMLNConfig):
    """Return (features, labels, severity, timestamps, names, zero_day_ids).

    Mirrors run_benchmark._load_stream: cached stream first, then raw CSVs, then
    a synthetic fallback so experiments run even without the dataset.
    """
    if STREAM_NPZ.exists():
        z = np.load(STREAM_NPZ, allow_pickle=True)
        feats, labels = z["features"], z["labels"]
        severity, ts = z["severity"], z["timestamps"]
        names = {int(i): str(n) for i, n in zip(z["label_ids"], z["label_names"])}
        cfg.encoder.num_features = feats.shape[1]
        attack_ids = [i for i in names if i != 0]
        counts = {i: int(np.sum(labels == i)) for i in attack_ids}
        zero_day = set(sorted(counts, key=counts.get)[:2]) if len(counts) >= 2 else set()
        return feats, labels, severity, ts, names, zero_day

    if CICIDS_DIR.exists() and list(CICIDS_DIR.rglob("*.csv")):
        from amln.data.cicids import load_cicids2017, stratified_stream
        data = load_cicids2017(CICIDS_DIR)
        data = stratified_stream(data, max_benign=40000, max_per_attack=8000,
                                 seed=cfg.train.seed)
        cfg.encoder.num_features = data.features.shape[1]
        attack_ids = [i for i in data.label_names if i != 0]
        counts = {i: int(np.sum(data.labels == i)) for i in attack_ids}
        zero_day = set(sorted(counts, key=counts.get)[:2]) if len(counts) >= 2 else set()
        return (data.features, data.labels, data.severity, data.timestamps,
                data.label_names, zero_day)

    from amln.data import ATTACK_FAMILIES, CyberEventGenerator
    gen = CyberEventGenerator(num_features=cfg.encoder.num_features, seed=cfg.train.seed)
    zero_day = {ATTACK_FAMILIES.index("apt_persistence"),
                ATTACK_FAMILIES.index("insider_threat")}
    batch = gen.sample(15000)
    names = {i: n for i, n in enumerate(ATTACK_FAMILIES)}
    ts = np.arange(len(batch), dtype=np.float64)
    return batch.features, batch.labels, batch.severity, ts, names, zero_day


def build_context(verbose: bool = True, use_cache: bool = True
                  ) -> ExperimentContext:
    """Load the stream, pre-train (or reuse) the encoder, build a shared harness.

    The 12-epoch encoder pre-training is the expensive step and is identical
    across experiments (same warmup prefix, same seed). We therefore cache the
    trained weights to ``data/exp_cache/encoder.pt`` and reuse them, so only the
    very first experiment pays the training cost.
    """
    cfg = AMLNConfig()
    if torch.cuda.is_available():
        cfg.train.device = "cuda"
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)

    feats, labels, severity, ts, names, zero_day = _load_stream(cfg)
    if verbose:
        # Family names in the cached stream may contain non-ASCII punctuation;
        # sanitize so logging never crashes on a narrow console codepage.
        zd = [names[i].encode("ascii", "replace").decode("ascii") for i in zero_day]
        print(f"Stream: events={len(feats):,} families={len(names)} zero-day={zd}")

    warmup = min(5000, len(feats) // 3)
    replayer = StreamReplayer(feats, labels, severity, ts, warmup=warmup)

    model = BehaviorAutoencoder(cfg.encoder)
    if use_cache and ENCODER_CKPT.exists():
        if verbose:
            print(f"Reusing cached encoder weights from {ENCODER_CKPT} ...")
        state = torch.load(ENCODER_CKPT, map_location=cfg.train.device)
        model.load_state_dict(state)
        model.to(cfg.train.device)
    else:
        if verbose:
            print("Pre-training behavioral autoencoder on warmup prefix ...")
        train_x = replayer.standardized_block(0, warmup)
        pretrain_autoencoder(model, train_x, cfg)
        if use_cache:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ENCODER_CKPT)
            if verbose:
                print(f"Cached encoder weights to {ENCODER_CKPT}")

    harness = BenchmarkHarness(model, replayer, cfg, cfg.encoder.latent_dim,
                               benign_label=0, zero_day_labels=zero_day)
    return ExperimentContext(cfg=cfg, harness=harness, replayer=replayer,
                             names=names, zero_day=zero_day, warmup=warmup)
