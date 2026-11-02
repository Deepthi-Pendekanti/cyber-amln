# AMLN Paper — Figures and Tables

Quick-reference collection of every figure and table in `PAPER_IEEE.md`, for
assembling the IEEE Word template. All figures are 300 DPI PNG, serif font
(pairs with IEEE Times New Roman), print-safe colors with the proposed method
highlighted in red. All numbers come from the generated result files:
`benchmark_results.csv` (Table II) and `experiment_*.csv` (Tables III–V).

Regenerate everything:

```powershell
$env:PYTHONPATH = (Get-Location).Path
python run_benchmark.py            # benchmark_results.csv (Table II)
python run_experiments.py          # experiment_*.csv (Tables III-V, Figs 5-8)
python scripts\make_paper_figures.py   # Figs 1-4
python scripts\build_paper_docx.py     # assembled PAPER_IEEE.docx
```

---

## Figures

| File | Figure | Caption (place BELOW the figure) |
|---|---|---|
| `paper_fig1_pipeline.png` | Fig. 1 | Fig. 1. End to end system pipeline: raw CIC-IDS2017 flows are converted into a stratified, chronological stream, encoded by a Transformer behavioral autoencoder into a 64 dimensional embedding with a reconstruction error surprise signal, and processed by the adaptive memory landscape, which drives both the live dashboard and the benchmark harness. |
| `paper_fig2_landscape.png` | Fig. 2 | Fig. 2. Schematic of the adaptive memory landscape. Each memory is a behavioral prototype with a height representing stability; memories above the hill threshold are stable long term hills, intermediate memories are forming slopes, and memories that decay below the valley threshold are forgotten. |
| `paper_fig3_quality.png` | Fig. 3 | Fig. 3. Detection quality across methods on the CIC-IDS2017 stream. The horizontal axis is the method and the vertical axis is the Macro F1 score (dimensionless). The proposed method is shown in red; AMLN-V2* is the tuned operating point. |
| `paper_fig4_tradeoff.png` | Fig. 4 | Fig. 4. Accuracy versus memory tradeoff. The horizontal axis is the memory footprint (KB) on a logarithmic scale and the vertical axis is the Macro F1 score (dimensionless). The upper left region denotes the best tradeoff. Both AMLN points lie far to the left of the bounded baselines. |
| `paper_fig5_pareto.png` | Fig. 5 | Fig. 5. Capacity sweep. The horizontal axis is the match similarity (memory resolution) and the vertical axis is the Macro F1 score; each point is annotated with its peak memory footprint. Quality rises with resolution, peaks at a match similarity of 0.96 (the tuned operating point, circled), and then falls as too fine a partition fragments each family. |
| `paper_fig6_ablation.png` | Fig. 6 | Fig. 6. Mechanism ablation. For each configuration the Macro F1 score and the zero day recall are shown. Removing surprise gating eliminates zero day recall; removing energy budgeting both lowers quality and enlarges the footprint. |
| `paper_fig7_bounded_memory.png` | Fig. 7 | Fig. 7. Memory footprint versus stream length on a logarithmic vertical axis. StoreAll grows linearly while the proposed method stays flat, well below the analytic bound E max divided by h min. |
| `paper_fig8_per_family.png` | Fig. 8 | Fig. 8. Per family F1 at the tuned operating point, sorted by support. High volume families are detected well; the rarest families are the hard cases. Held out zero day families are shown in red. |

---

## TABLE I — Implementation environment, tools, dataset, and parameters

| Item | Value |
|---|---|
| Language | Python |
| Deep learning framework | PyTorch (CPU) a |
| Visualization | Streamlit, Plotly |
| Numerical and ML libraries | NumPy, scikit-learn, pandas, Matplotlib |
| Dataset | CIC-IDS2017 |
| Flow features | 78 numeric features per event |
| Stream size | 101,323 events (chronological) |
| Attack families in stream | 15 |
| Warmup prefix | about 5,000 events (scaler fit and encoder pretrain) |
| Latent embedding size | 64 dimensions |
| Transformer layers | 2 |
| Attention heads | 4 |
| Pretraining epochs | 12 |
| Denoising mask fraction | up to 0.60 |
| Energy budget (E max) | 400 |
| Valley threshold (h min) | 0.25 |
| Match similarity, default point | 0.85 |
| Match similarity, tuned point (AMLN-V2*) | 0.96 |
| V2 decay rate, tuned point | 0.002 |
| Random seed | 42 |
| Scaling | signed log, then median and interquartile range, clipped to plus or minus 5 |
| Footprint per item | 64 float32 values plus an 8 byte label (264 bytes) |

a. Training and evaluation were performed on CPU; CUDA was not available in the test environment.

---

## TABLE II — Prequential results on the CIC-IDS2017 stream

| Method | Accuracy | Macro-F1 | Attack recall | Zero-day recall b | Footprint (KB) | Memory saving c | Peak items | Final items | Query time (ms) |
|---|---|---|---|---|---|---|---|---|---|
| AMLN-V2 | 0.768 | 0.586 | 0.729 | 0.219 | 5.41 | 4,825x | 21 | 10 | 0.021 |
| AMLN-V2* (tuned) | 0.829 | 0.715 | 0.820 | 0.219 | 16.24 | 1,608x | 63 | 12 | 0.027 |
| StoreAll | 0.942 | 0.779 | 0.930 | 0.219 | 26,122 | 1x | 101,323 | 101,323 | 0.576 |
| SlidingWindow(5000) | 0.972 | 0.838 | 0.973 | 0.375 | 1,289 | 20x | 5,000 | 5,000 | 0.047 |
| Reservoir(5000) | 0.918 | 0.678 | 0.903 | 0.031 | 1,289 | 20x | 5,000 | 5,000 | 0.047 |
| OnlineKMeans(32) | 0.570 | 0.297 | 0.404 | 0.000 | 8.25 | 3,166x | 32 | 32 | 0.031 |

b. For the proposed method, zero day recall is the fraction of held out family events flagged as novel by the surprise gate; for baselines it is exact label recovery, because they have no novelty signal.

c. Memory saving is the StoreAll footprint divided by the method footprint, both computed on the peak number of retained items at 264 bytes per item.

---

## TABLE III — Capacity sweep (energy budget 400, decay 0.002)

| Match similarity | Peak items | Footprint (KB) | Accuracy | Macro-F1 |
|---|---|---|---|---|
| 0.80 | 46 | 11.86 | 0.706 | 0.515 |
| 0.88 | 39 | 10.05 | 0.734 | 0.580 |
| 0.93 | 45 | 11.60 | 0.777 | 0.652 |
| 0.96 | 55 | 14.18 | 0.825 | 0.701 |
| 0.975 | 53 | 13.66 | 0.802 | 0.676 |
| 0.985 | 53 | 13.66 | 0.757 | 0.643 |

---

## TABLE IV — Mechanism ablation at the tuned operating point

| Configuration | Macro-F1 | Zero-day recall | Footprint (KB) | Peak items |
|---|---|---|---|---|
| Full AMLN | 0.701 | 0.219 | 14.18 | 55 |
| without surprise gating | 0.670 | 0.000 | 14.95 | 58 |
| without energy budgeting | 0.671 | 0.219 | 18.05 | 70 |
| without drift splitting | 0.700 | 0.219 | 13.41 | 52 |

---

## TABLE V — Per family detection at the tuned operating point (selected)

| Family | Support | Precision | Recall | F1 |
|---|---|---|---|---|
| PortScan | 8,000 | 0.951 | 0.998 | 0.974 |
| DDoS | 8,000 | 0.959 | 0.935 | 0.947 |
| FTP-Patator | 7,938 | 0.974 | 0.827 | 0.894 |
| Bot | 1,966 | 0.919 | 0.856 | 0.886 |
| BENIGN | 35,000 | 0.912 | 0.835 | 0.872 |
| DoS Slowhttptest | 5,499 | 0.995 | 0.756 | 0.859 |
| DoS slowloris | 5,796 | 0.789 | 0.721 | 0.753 |
| SSH-Patator | 5,897 | 0.626 | 0.932 | 0.749 |
| DoS Hulk | 8,000 | 0.563 | 0.995 | 0.719 |
| Infiltration | 36 | 0.760 | 0.528 | 0.623 |
| DoS GoldenEye | 8,000 | 0.667 | 0.433 | 0.525 |
| Web Attack XSS | 652 | 0.857 | 0.009 | 0.018 |
| Heartbleed (zero-day) d | 11 | 1.000 | 0.909 | 0.952 |
| Web Attack SQL Injection (zero-day) d | 21 | 0.000 | 0.000 | 0.000 |

d. The zero day families are held out from labeling; their meaningful score is the surprise gate novelty recall, which is 0.636 for Heartbleed and 0.000 for SQL Injection.
