# TASK 5.5 Empirical Validation Report

**Data sources**: `Code/results/results.csv` (270 rows, no NaN), `summary_stats.csv`, `training_log.txt`, `data_generation_log.txt`, Visuals G1–G5. All numbers below are read or computed directly from these files.

## 1. Hypothesis outcomes

**H1 (NN ≤ Manhattan on nodes at 50×50)  REFUTED.** In-distribution, NN expands 155.8 nodes on average vs. Manhattan's 115.1 (+35%); paired per-problem, NN strictly wins only 12/30 (3 ties). NN does dominate Euclidean decisively: 155.8 vs. 344.9 nodes, winning 28/30 (2 ties, 0 losses).

**H2 (Manhattan wins wall-clock at 10×10/50×50; possible crossover at 100×100) first part CONFIRMED, crossover REFUTED.** Manhattan wins wall-clock on every problem at every size. The gap *widens* with scale: NN is 29.6× slower at 10×10 (4.54 ms vs. 0.153 ms), 37.5× at 50×50 (51.9 ms vs. 1.38 ms), 93.0× at 100×100 (392.9 ms vs. 4.23 ms). Inference accounts for 92–96% of NN search time (`wall_time_heuristic`/`wall_time_total`), at ~1.5–2.4×10⁻⁴ s per call vs. ~6–9×10⁻⁷ s for Manhattan (≈250–300×).

**H3 (violation rate 30–60%) CONFIRMED in-distribution, with a transfer surprise.** At 50×50 the mean violation rate is 61.0% (per-problem range 28.0–93.0%), matching the symmetric-MSE prediction. Off-distribution it becomes bimodal: 99.6% at 10×10 (mean overestimation 29.6 steps — ~4× the mean optimal path length of 7.5) and 0.6% at 100×100 (overestimation 0.66 steps).

**H4 (suboptimality mostly 1.00–1.10, occasional outliers) CONFIRMED, stronger than predicted.** All 90 NN runs returned exactly optimal paths (suboptimality = 1.000, std = 0; G3). No outliers in the main experiment; the Gate-3 pilot (different problem set) produced one at 1.154, so outliers exist but are rare.

**H5 (transfer degradation) CONFIRMED, asymmetrically.** At 100×100, NN expands 2,244 nodes vs. Manhattan's 452 (5.0×) and is even worse than Euclidean (1,427). At 10×10 degradation appears not in nodes (NN is best: 8.5 vs. 10.8, never worse in 30/30) but in calibration: near-universal overestimation makes it behave like an aggressive weighted-A* heuristic.

## 2. The three insights

**Insight 2 (Guarantee Laundering) mechanism confirmed, cost not observed.** Symmetric MSE produced violations on 61% of sampled in-distribution states, so the admissibility guarantee is demonstrably void. Yet path cost in this sample was zero (90/90 optimal). This *is* the laundering: optimality survives only as an unprovable empirical regularity — observed here, guaranteed nowhere.

**Insight 3 (Rigged Metric) confirmed at 2 of 3 sizes.** At 10×10 the two metrics name opposite winners: NN is best on nodes, worst on wall-clock (30× slower than Manhattan). At 50×50, NN beats Euclidean on nodes (28/30) while Euclidean is 13.7× faster. At 100×100 the metrics agree (NN loses both). G1 vs. G2 is the headline figure.

**Insight 1 (Pattern A)** is validated architecturally: the identical `astar()` ran all 270 searches.

## 3. Confident claims / limitations

**Confident**: the violation-rate numbers; the metric-divergence result; no wall-clock crossover up to 100×100; transfer degradation; training cost 176.18 s + 3.49 s data generation [R5] — time enough for Manhattan to solve ~127,000 50×50 problems.

**Limitations**: n = 30/size, one seed, one trained model (val MAE 2.34 ≈ 8% of mean h* = 29.38); unbatched Python inference inflates per-call cost (compiled/batched settings would change constants, not structure); violation rates from 500-node samples; unit-cost 4-connected grids have many co-optimal paths, which — together with the reopening A* variant — plausibly masks suboptimality; single domain.

## 4. Numbers for Section 5

Table: per size × heuristic means of nodes, wall-clock, violation rate, overestimation (from `summary_stats.csv`). Figures: G1+G2 (rigged metric, core), G4 (violations, Insight 2), G5 (transfer). Cite: 61.0%; 90/90 optimal; 29.6×/37.5×/93.0×; 5.0× nodes at 100×100; 92–96% inference share; 176.18 s training; 22,371 training pairs; seed 42.
