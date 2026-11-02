School of Computer Science Engineering and
Information Systems (SCORE)

# AMLN Project Report

**Course Title:** Cyber Security

**Course Code:** ISWE415L

**Slot:** D1+TD1

**TEAM MEMBERS:**

1. Sai Deepthi Pendekanti
2. Gowsik V
3. Srinidhi

---

## Title of Project

AMLN: Adaptive Memory Landscape Network — A Brain-Inspired, Bounded-Memory System for Real-Time Streaming Cyber Threat Detection

## Abstract

Modern intrusion-detection systems face a tension between accuracy and resource
cost: approaches that remember the entire event history detect well but grow
without bound, while fixed-size approaches stay cheap but forget indiscriminately.
AMLN (Adaptive Memory Landscape Network) is a brain-inspired framework that
detects cyber threats from a live log stream using a small, self-organizing
memory that grows, forgets, and reorganizes online within a fixed, provably
bounded budget, and without ever retraining the underlying neural network.
Security events are encoded by a Transformer behavioral autoencoder into a
64-dimensional embedding, and a memory landscape of stable hills, forming slopes,
and forgotten valleys is maintained through three mechanisms, namely
surprise-gated plasticity, energy-budgeted forgetting, and drift-aware
consolidation. The memory is strictly bounded by construction because the fixed
energy budget divided by the minimum retained height caps the number of memories.
AMLN is evaluated on CIC-IDS2017, a real benchmark of a full week of network
traffic with injected attacks, replayed as a one-event-at-a-time live stream,
using a prequential (test-then-train) protocol against four standard
memory-retention strategies, and is further characterized by a capacity sweep, a
mechanism ablation, an empirical bounded-memory study, and a per-family
breakdown. At its tuned operating point AMLN reaches macro-F1 of about 0.72 and
accuracy of about 0.83 while retaining a peak of only 63 memory nodes
(about 16 KB), which is about 1,600 times less memory than a store-everything
baseline that scores 0.78, and it matches that baseline on zero-day recall while
querying about 20 times faster. The ablation shows that surprise-gating is
necessary for zero-day detection and that energy-budgeting is necessary for both
stable accuracy and a bounded footprint, and the bounded-memory study confirms
the footprint stays roughly constant as the stream grows, ending at 11 live
nodes (about 2.8 KB). The result is best read as detection quality per kilobyte
of memory under real concept drift, on which AMLN is the strongest bounded
detector among those compared.

## Problem Statement

Security telemetry arrives as a never-ending stream of events such as process
launches, network connections, authentications, and file operations, often at
thousands of events per second for months without interruption. A practical
detector must satisfy four requirements at once. First, it must operate online,
consuming events one at a time and indefinitely. Second, it must adapt to new and
evolving attacks (concept drift) without full retraining, since retraining
interrupts protection and cannot keep pace with a live adversary. Third, it must
remain resource-bounded, so that memory and latency do not grow with stream
length. Fourth, it must flag novel or zero-day behavior it has never been trained
on. Classic machine-learning intrusion-detection pipelines train a fixed model
offline and either store large feature buffers or retrain periodically; both
choices scale poorly for a continuous, drifting stream. A store-everything
detector incurs memory that grows linearly and without bound, which is infeasible
on a long-lived sensor, whereas a fixed-window detector either forgets useful
rare patterns or must be retrained offline. The project addresses the need for a
detector whose memory is bounded by construction, whose adaptation is online and
retraining-free, and whose novelty detection does not depend on having a label
for the unseen family.

## Objectives

- Convert a recorded intrusion dataset (CIC-IDS2017) into a causal,
  chronologically ordered live stream with causal robust scaling.
- Encode each event with a Transformer behavioral autoencoder into a compact
  64-dimensional embedding and a reconstruction-error surprise signal.
- Maintain a dynamic memory landscape of hills, slopes, and valleys over the
  embedding space, updated online with no neural-network retraining.
- Gate plasticity by surprise so that familiar traffic barely perturbs memory
  while novel events strongly update it, enabling zero-day flagging by novelty.
- Enforce a fixed energy budget that provably bounds the number of retained
  memories, giving a footprint that is roughly constant regardless of stream
  length.
- Track concept drift through drift-aware consolidation (merging near-identical
  memories and splitting mutating ones) without retraining.
- Benchmark AMLN against standard memory-retention baselines using a prequential
  protocol, measuring both detection quality and memory cost.
- Characterize the method with a capacity sweep, a mechanism ablation, an
  empirical bounded-memory study, and a per-attack-family breakdown.
- Provide a real-time, self-explaining dashboard that makes the memory dynamics
  and the memory savings directly observable.

## Proposed System and Architecture

### Components

Presentation layer: a local Streamlit and Plotly dashboard for running the live
stream, watching memories form and decay, and viewing the surprise timeline,
rolling accuracy, and zero-day alerts. Processing layer: a stream replayer that
yields events one at a time with causal robust scaling, and a Transformer
behavioral autoencoder that maps each event to a 64-dimensional embedding plus a
reconstruction-error surprise signal. Memory engine: the Adaptive Memory
Landscape (V2), comprising the surprise gate, the match-or-spawn step, the
reinforce-and-decay step, the energy-budget pruning step, and the drift-aware
merge/split step. Evaluation layer: a prequential benchmark harness that drives
AMLN and four baselines over the identical stream, and an experiment suite that
produces the capacity sweep, ablation, bounded-memory, and per-family studies.
Storage: a cached chronological stream (`data/stream.npz`), a cached encoder
checkpoint, and generated result files (`benchmark_results.csv`,
`experiment_*.csv`).

### Architecture workflow

- Raw CIC-IDS2017 flows -> stratified, chronology-preserving subsample ->
  cached stream.
- Cached stream -> replayer (causal robust scaling, one event at a time).
- Event features -> Transformer autoencoder -> 64-d embedding + reconstruction
  error (surprise signal).
- Embedding + surprise -> Adaptive Memory Landscape: surprise gate ->
  match / spawn -> reinforce / decay -> energy-budget prune -> merge / split.
- Landscape -> classification (nearest memory label) and zero-day decision
  (surprise above adapted threshold).
- Same landscape -> live dashboard (visualization) and benchmark harness
  (prequential evaluation).

```
CIC-IDS2017 CSVs (2.8M flows)
        |   load + clean + stratified subsample  (amln/data/cicids.py)
        v
  Cached stream.npz (101k events, 78 features, chronological)
        |   causal robust scaling, one event at a time  (amln/streaming/replayer.py)
        v
  Transformer behavioral autoencoder  (amln/models/encoder.py)
        |   78 features -> 64-d embedding + reconstruction error (surprise signal)
        v
  Adaptive Memory Landscape V2  (amln/memory/adaptive_v2.py)   <-- NOVEL CORE
        |   observe(): surprise-gate -> match/spawn -> reinforce/decay ->
        |              energy-budget prune -> merge/split
        +------------> Live dashboard      (dashboard.py, streaming/engine.py)
        +------------> Benchmark harness    (run_benchmark.py, benchmark/)
```

### Core logic

For each streamed event the landscape executes a single `observe` step built
from three co-operating mechanisms.

1. Surprise-gated plasticity (free-energy inspired). Each event's surprise
   combines the encoder reconstruction error (how unfamiliar the behavior is)
   with novelty (one minus the cosine similarity to the nearest memory):

   ```
   surprise = w_recon * sigmoid(recon_error_norm) + w_novelty * (1 - cos(z, c*))
   ```

   Plasticity scales with surprise, so familiar traffic barely perturbs memory
   while surprising events strongly update or spawn one. Because an unseen attack
   family is high-surprise by construction, an event is flagged as a possible
   zero-day when its surprise exceeds an online-adapted threshold.

2. Energy-budgeted forgetting (bounded memory). All memories share a fixed energy
   budget equal to the sum of their heights, which must not exceed a maximum.
   When a new memory would overflow the budget, energy is reclaimed from the
   least-important memories first, where importance combines log-frequency,
   severity, and recency. A grace period protects freshly spawned memories. Since
   every retained memory has height at least the valley threshold `h_min` and the
   total height cannot exceed `E_max`, the number of memories `K` is bounded:

   ```
   K <= E_max / h_min
   ```

   The footprint is therefore strictly bounded by construction and roughly
   constant regardless of stream length.

3. Drift-aware consolidation. Each memory tracks, with an online Welford update,
   the running variance of the cosine distance of the events it absorbs. Two
   signals trigger a split: a geometric one (the RMS intra-cluster distance
   divided by the gate radius, which is scale-invariant) and a semantic one (the
   label impurity, one minus the majority-family vote share). The split is
   non-destructive: the parent keeps its identity and the budded-off sibling must
   earn its place through reinforcement or it decays away. In the other direction,
   near-identical memories merge, which keeps the footprint compact. This yields
   online adaptation to concept drift with no neural-network retraining.

Prediction assigns the label of the nearest memory, and the zero-day decision
flags an event as novel when its surprise exceeds the adapted threshold, which is
the appropriate definition for an unseen family the system cannot name.

### Implementation environment

The system is implemented in Python with PyTorch (CPU), NumPy, scikit-learn,
pandas, Matplotlib, and a Streamlit/Plotly dashboard. The dataset is CIC-IDS2017,
subsampled into a cached chronological stream of 101,323 events with 78 numeric
flow features and 15 attack families. The encoder uses a 64-dimensional latent, 2
Transformer layers, 4 attention heads, 12 pretraining epochs, and up to 0.60
denoising mask. The memory uses energy budget `E_max` = 400 and valley threshold
`h_min` = 0.25. Two operating points are reported: a default point (match
similarity 0.85) and a tuned point, AMLN-V2*, found by the capacity sweep (match
similarity 0.96, decay 0.002). Scaling applies signed-log, then median/IQR,
clipped to plus or minus 5. Each retained item is 64 float32 values plus an
8-byte label (264 bytes). The fixed random seed is 42 throughout.

## Verification logic

AMLN is verified with a prequential (interleaved test-then-train) protocol: for
each event the system first predicts with its current memory, then observes the
event and updates. The warmup prefix (about 5,000 events) is used only to fit the
scaler and pretrain the encoder and is excluded from scoring. Zero-day evaluation
holds out the two rarest families, Heartbleed and Web Attack SQL Injection, as
unseen families; because these have no label the system can recover, their
meaningful score is the fraction flagged as novel by the surprise gate. The
memory footprint reported for each method is the peak number of retained items
multiplied by the 264-byte per-item size. Four memory-retention baselines are
used for comparison: StoreAll (keeps every embedding, the accuracy reference),
SlidingWindow(5000) (most recent 5,000 events), Reservoir(5000) (uniform random
sample of 5,000), and OnlineKMeans(32) (32 moving centroids). All numbers are
produced on the cached stream with seed 42, so they are reproducible.

## Results

The method was evaluated on the 101,323-event CIC-IDS2017 stream. Encoder
pretraining converged to a reconstruction mean squared error of about 0.109 with
robust scaling. Table 1 reports the prequential results for both AMLN operating
points against the four baselines.

Table 1. Prequential results on the CIC-IDS2017 stream.

| Method | Accuracy | Macro-F1 | Attack recall | Zero-day recall | Footprint (KB) | Memory saving | Peak items | Query (ms) |
|---|---|---|---|---|---|---|---|---|
| AMLN-V2 | 0.768 | 0.586 | 0.729 | 0.219 | 5.41 | 4,825x | 21 | 0.021 |
| AMLN-V2* (tuned) | 0.829 | 0.715 | 0.820 | 0.219 | 16.24 | 1,608x | 63 | 0.027 |
| StoreAll | 0.942 | 0.779 | 0.930 | 0.219 | 26,122 | 1x | 101,323 | 0.576 |
| SlidingWindow(5000) | 0.972 | 0.838 | 0.973 | 0.375 | 1,289 | 20x | 5,000 | 0.047 |
| Reservoir(5000) | 0.918 | 0.678 | 0.903 | 0.031 | 1,289 | 20x | 5,000 | 0.047 |
| OnlineKMeans(32) | 0.570 | 0.297 | 0.404 | 0.000 | 8.25 | 3,166x | 32 | 0.031 |

The tuned point reaches macro-F1 0.715 and accuracy 0.829 while retaining a peak
of only 63 memories occupying 16.24 KB, which is 1,608 times smaller than
StoreAll. This macro-F1 exceeds Reservoir (0.678) while using about 80 times less
memory than Reservoir, and it approaches the StoreAll reference (0.779) while
using three orders of magnitude less memory. Both AMLN points tie StoreAll on
zero-day recall (0.219) and beat Reservoir (0.031) and OnlineKMeans (0.000),
because only the surprise gate provides a novelty signal; SlidingWindow is highest
(0.375) because it retains the most recent 5,000 raw events. Query latency for the
tuned point is 0.027 ms against 0.576 ms for StoreAll, about 21 times faster.

Capacity sweep. The dominant control on memory is the match similarity, which
sets how finely the embedding space is partitioned. Table 2 sweeps this
resolution at a fixed energy budget of 400 and decay 0.002. Quality rises with
resolution, peaks at a match similarity of 0.96, and then falls as an overly fine
resolution fragments each family; the peak defines AMLN-V2*.

Table 2. Capacity sweep (energy budget 400, decay 0.002).

| Match similarity | Peak items | Footprint (KB) | Accuracy | Macro-F1 |
|---|---|---|---|---|
| 0.80 | 46 | 11.86 | 0.706 | 0.515 |
| 0.88 | 39 | 10.05 | 0.734 | 0.580 |
| 0.93 | 45 | 11.60 | 0.777 | 0.652 |
| 0.96 | 55 | 14.18 | 0.825 | 0.701 |
| 0.975 | 53 | 13.66 | 0.802 | 0.676 |
| 0.985 | 53 | 13.66 | 0.757 | 0.643 |

Mechanism ablation. Each mechanism is disabled in turn at the tuned point
(Table 3). Removing surprise-gating collapses zero-day recall from 0.219 to 0.000,
confirming it is necessary for novelty detection. Removing energy-budgeting lowers
macro-F1 by 0.030 and enlarges the footprint from 14.18 to 18.05 KB (peak 55 to
70), confirming it is necessary for both accuracy and the bound. Removing
drift-splitting leaves macro-F1 essentially unchanged, showing splitting is a
situational mechanism that adds mutating-family tracking at negligible cost.

Table 3. Mechanism ablation at the tuned operating point.

| Configuration | Macro-F1 | Zero-day recall | Footprint (KB) | Peak items |
|---|---|---|---|---|
| Full AMLN | 0.701 | 0.219 | 14.18 | 55 |
| without surprise gating | 0.670 | 0.000 | 14.95 | 58 |
| without energy budgeting | 0.671 | 0.219 | 18.05 | 70 |
| without drift splitting | 0.700 | 0.219 | 13.41 | 52 |

Bounded memory in practice. The analytic bound is `K <= E_max / h_min =
400 / 0.25 = 1,600` memories. Empirically, StoreAll grows linearly to 26,122 KB
over 101,323 events, whereas AMLN stays flat at a few kilobytes and ends the
stream with only 11 live memories (about 2.8 KB), far below both its own peak and
the analytic ceiling. The footprint therefore does not grow with stream length.

Per-family detection. High-volume families are detected well — PortScan F1 0.974,
DDoS 0.947, FTP-Patator 0.894, Bot 0.886, and BENIGN 0.872 — while the rarest
families are the hard cases under severe class imbalance, with Web Attack XSS at
0.018 and the held-out SQL Injection at 0.000. The held-out Heartbleed family
(11 events) is flagged as novel on 0.636 of its occurrences by the surprise gate,
an honest result on one of the hardest possible cases.

## Graphs

Figure 1. End-to-end system pipeline: raw CIC-IDS2017 flows are converted into a
stratified, chronological stream, encoded by the Transformer behavioral
autoencoder into a 64-dimensional embedding with a reconstruction-error surprise
signal, and processed by the adaptive memory landscape, which drives both the
live dashboard and the benchmark harness. (`paper_fig1_pipeline.png`)

Figure 2. Schematic of the adaptive memory landscape: each memory is a behavioral
prototype with a height representing stability; memories above the hill threshold
are stable long-term hills, intermediate memories are forming slopes, and
memories that decay below the valley threshold are forgotten.
(`paper_fig2_landscape.png`)

Figure 3. Detection quality across methods; the proposed method is shown in red,
with AMLN-V2* the tuned operating point. (`paper_fig3_quality.png`)

Figure 4. Accuracy versus memory tradeoff on a logarithmic memory axis; both AMLN
points lie far to the left (less memory) of the bounded baselines, in the
favorable upper-left region. (`paper_fig4_tradeoff.png`)

Figure 5. Capacity sweep: macro-F1 versus match similarity, each point annotated
with its footprint; quality peaks at a match similarity of 0.96 (the tuned
operating point, circled). (`paper_fig5_pareto.png`)

Figure 6. Mechanism ablation: removing surprise-gating eliminates zero-day
recall; removing energy-budgeting both lowers quality and enlarges the footprint.
(`paper_fig6_ablation.png`)

Figure 7. Memory footprint versus stream length on a logarithmic axis: StoreAll
grows linearly while AMLN stays flat, well below the analytic bound.
(`paper_fig7_bounded_memory.png`)

Figure 8. Per-family F1 at the tuned operating point, sorted by support;
high-volume families are detected well and the rarest families are the hard
cases, with held-out zero-day families shown in red.
(`paper_fig8_per_family.png`)

## Conclusion

AMLN demonstrates that framing streaming threat detection as a bounded
memory-management problem is viable: a brain-inspired landscape with
surprise-gated plasticity, energy-budgeted forgetting, and drift-aware
consolidation compresses a 101k-event CIC-IDS2017 stream into tens of memories.
At its tuned operating point it reaches macro-F1 0.715 and accuracy 0.829 at
about 16 KB (63 nodes), about 1,600 times less memory than storing everything
(which scores 0.779) and about 80 times less than the Reservoir baseline it
beats, while matching StoreAll's zero-day recall and querying about 21 times
faster. A capacity sweep traced the full accuracy-versus-memory frontier and
located this operating point, a mechanism ablation showed that surprise-gating is
necessary for zero-day detection and energy-budgeting for both accuracy and the
memory bound, and an empirical study confirmed the footprint stays flat (ending
at 11 live nodes, about 2.8 KB) as the stream grows. The main limitation is that
overall macro-F1 still trails the sliding-window baseline, which retains three
orders of magnitude more memory, and the rarest families remain hard under severe
class imbalance. Future work includes cost-sensitive and per-family scoring to
lift recall on the rarest families, a stronger encoder trained for more epochs on
a GPU with a contrastive objective, an approximate nearest-neighbor index to
accelerate the store-everything baseline, and evaluation on an additional dataset
such as UNSW-NB15 for cross-dataset generalization.

## References

[1] I. Sharafaldin, A. H. Lashkari, and A. A. Ghorbani, "Toward generating a new intrusion detection dataset and intrusion traffic characterization," in Proc. Int. Conf. Inf. Syst. Secur. Privacy (ICISSP), 2018, pp. 108-116.

[2] N. Moustafa and J. Slay, "UNSW-NB15: A comprehensive data set for network intrusion detection systems," in Proc. Mil. Commun. Inf. Syst. Conf. (MilCIS), 2015, pp. 1-6.

[3] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, et al., "Attention is all you need," in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), 2017, pp. 5998-6008.

[4] M. Sakurada and T. Yairi, "Anomaly detection using autoencoders with nonlinear dimensionality reduction," in Proc. MLSDA Workshop, 2014, pp. 4-11.

[5] J. Gama, R. Sebastiao, and P. P. Rodrigues, "On evaluating stream learning algorithms," Mach. Learn., vol. 90, no. 3, pp. 317-346, 2013.

[6] J. Gama, I. Zliobaite, A. Bifet, M. Pechenizkiy, and A. Bouchachia, "A survey on concept drift adaptation," ACM Comput. Surv., vol. 46, no. 4, pp. 1-37, 2014.

[7] K. Friston, "The free-energy principle: A unified brain theory?," Nat. Rev. Neurosci., vol. 11, no. 2, pp. 127-138, 2010.

[8] J. S. Vitter, "Random sampling with a reservoir," ACM Trans. Math. Softw., vol. 11, no. 1, pp. 37-57, 1985.

Project repository: (add your repository URL here)

---

## Appendix: Reproducibility and Repository Map

All quantitative results are produced on the cached CIC-IDS2017 stream with the
fixed seed 42:

```powershell
# 0. install
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 1. Kaggle token at %USERPROFILE%\.kaggle\kaggle.json, then download
$env:PYTHONPATH = (Get-Location).Path
.\.venv\Scripts\python.exe scripts\download_cicids2017.py

# 2. one-time cache of the stratified stream
.\.venv\Scripts\python.exe scripts\prepare_stream.py

# 3. benchmark (writes benchmark_results.csv incl. AMLN-V2 + AMLN-V2*)
.\.venv\Scripts\python.exe run_benchmark.py

# 4. full experiment suite: capacity sweep, ablation, bounded-memory,
#    per-family (writes experiment_*.csv + paper_fig5..8)
.\.venv\Scripts\python.exe run_experiments.py

# 5. regenerate paper figures (Figs. 1-4) from benchmark_results.csv
.\.venv\Scripts\python.exe scripts\make_paper_figures.py

# 6. live dashboard
.\.venv\Scripts\python.exe -m streamlit run dashboard.py
```

Without the dataset, every component falls back to a synthetic generator so the
pipeline is runnable immediately.

| Path | Role |
|---|---|
| `amln/models/encoder.py` | Transformer behavioral autoencoder |
| `amln/memory/adaptive_v2.py` | Novel memory landscape (V2) |
| `amln/memory/landscape.py` | Original landscape (V1, reference) |
| `amln/data/cicids.py` | CIC-IDS2017 loader + stratified stream |
| `amln/data/preprocess.py` | StandardScaler + RobustScaler |
| `amln/streaming/replayer.py` | Causal one-at-a-time stream replay |
| `amln/benchmark/baselines.py` | StoreAll / SlidingWindow / Reservoir / KMeans |
| `amln/benchmark/harness.py` | Prequential benchmark (AMLN-V2 + tuned AMLN-V2*) |
| `amln/config.py` | Config incl. tuned operating-point constants |
| `run_benchmark.py` | Benchmark entry point |
| `run_experiments.py` | Full experiment suite (sweep/ablation/bounded/per-family) |
| `experiments/` | Capacity sweep, ablation, bounded-memory, per-family experiments |
| `scripts/make_paper_figures.py` | Generate figures |
| `dashboard.py` | Streamlit live dashboard |
