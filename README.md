# Classical A* vs. A* with a Learned Neural Heuristic

Source code for a controlled empirical comparison of classical A* heuristics
(Manhattan, Euclidean) against a learned neural-network heuristic, on grid
pathfinding at three scales (10x10, 50x50, 100x100).

The neural heuristic is trained **only** on 50x50 maps; the 10x10 and 100x100
runs are an out-of-distribution transfer test. The same `astar()` implementation
runs all three heuristics unmodified it never knows which one it is using.

## Key findings

- The learned heuristic violates admissibility ($h(n) > h^*(n)$) in **61.0%** of
  sampled states, yet still returned an optimal path in **all 90** search
  problems it was tested on in this run.
- Node count and wall-clock time disagree on a winner at 10x10: the neural
  heuristic expands fewer nodes than Manhattan there, but is ~30x slower in
  wall-clock time a "rigged metric" effect that a nodes-only comparison
  would miss entirely.
- Full methodology, related work, and discussion are in the companion report
  (see `docs/`).

## Repository structure

```
config.py            All shared constants (seed, grid sizes, hyperparameters)
grid.py               GridWorld: obstacles, 4-connected neighbors, feature extraction
astar.py              Heuristic-agnostic A* search (Manhattan/Euclidean/NN interchangeable)
heuristics.py         manhattan(), euclidean(), and the NNHeuristic wrapper
nn_model.py           PyTorch MLP heuristic model (7 -> 64 -> 32 -> 1)
generate_data.py      Builds the training set from 200 random 50x50 maps
train.py              Trains the MLP on (state, h*) pairs
experiments.py        Full sweep: 3 sizes x 30 problems x 3 heuristics -> results.csv
analysis.py           Figures (G1-G5) and summary statistics from results.csv
test_gate1.py..4.py   Verification checks run after each implementation stage
results/              Generated data, trained model, experiment results, logs
Visuals/              Figures produced by analysis.py
docs/                 Architecture notes and validation report
```

## Reproducing the results

```bash
pip install -r requirements.txt

python generate_data.py   # builds results/training_data_50x50.npz
python train.py           # trains results/model_50x50.pt
python experiments.py     # runs the full sweep -> results/results.csv
python analysis.py        # produces the figures in Visuals/ and summary_stats.csv
```

Each stage has a corresponding gate test that can be run independently to
verify correctness before moving to the next stage:

```bash
python test_gate1.py   # A* + Manhattan finds the known-optimal path on a hand-checkable grid
python test_gate2.py   # NN heuristic validation MAE and sanity checks
python test_gate3.py   # A* + NN finds valid, terminating (if suboptimal) paths
python test_gate4.py   # results.csv integrity checks (row counts, no NaNs, admissible heuristics are optimal)
```

## Requirements

Python 3.9+, numpy, torch (CPU), matplotlib, pandas see `requirements.txt`.

## Author

Bagmbaye Aggée — MSc Dalian University of Technology
