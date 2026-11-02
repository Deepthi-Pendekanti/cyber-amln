"""Conference-grade experiments for AMLN.

This package contains the rigorous evaluations that turn AMLN from a promising
prototype into a defensible research contribution:

* ``common``          - shared stream loading + encoder pretraining (done once,
                        reused by every experiment so they are fast and fair).
* ``budget_sweep``    - Pareto curve of detection quality vs. memory footprint
                        as the energy budget is swept, showing the tunable
                        accuracy/memory frontier.
* ``ablation``        - turns each of the three novel mechanisms off in turn to
                        quantify its individual contribution.
* ``bounded_memory``  - tracks footprint vs. stream position to empirically
                        prove the memory stays bounded and roughly constant.
* ``per_family``      - per-attack-family precision/recall/F1 under severe class
                        imbalance (the honest, fine-grained view of quality).

All experiments read the cached CIC-IDS2017 stream (``data/stream.npz``) and use
the fixed seed from :class:`amln.config.AMLNConfig` for reproducibility.
"""
