"""Full experiment sweep: Manhattan vs Euclidean vs NN across grid sizes
10/50/100, 30 solvable problems each, fixed seeds for reproducibility.

The NN heuristic is always model_50x50.pt, including at 10x10 and 100x100 --
it is never retrained here, so those two sizes test how well it transfers
to grids it wasn't trained on.

Writes results/results.csv: 3 sizes x 30 problems x 3 heuristics = 270 rows,
columns: size, problem_id, heuristic, nodes_expanded, wall_time_total,
wall_time_heuristic, path_length, optimal_len, suboptimality,
violation_rate, mean_overestimation, h_calls.
"""
import csv
import random
import sys
import time
from collections import Counter
from pathlib import Path

from config import SEED, GRID_SIZES, OBSTACLE_DENSITY, N_TEST_PROBLEMS, TRAIN_GRID
from grid import GridWorld
from astar import astar
from heuristics import manhattan, euclidean, NNHeuristic
from generate_data import pick_start_goal, backward_distances

RESULTS_DIR = Path(__file__).parent / "results"
MODEL_PATH = RESULTS_DIR / f"model_{TRAIN_GRID}x{TRAIN_GRID}.pt"
ADMISSIBILITY_SAMPLE = 500
EPSILON = 1e-6

FIELDNAMES = [
    "size", "problem_id", "heuristic", "nodes_expanded",
    "wall_time_total", "wall_time_heuristic",
    "path_length", "optimal_len", "suboptimality",
    "violation_rate", "mean_overestimation", "h_calls",
]


def problem_seeds(size, problem_id):
    """Deterministic (map, start/goal, admissibility-sample) seeds per problem.

    Derived from (SEED, size, problem_id) rather than a running RNG stream,
    so results are reproducible regardless of iteration order.
    """
    rng = random.Random((SEED, size, problem_id))
    return rng.randrange(1 << 30), rng.randrange(1 << 30), rng.randrange(1 << 30)


def print_progress(done, total, width=40):
    frac = done / total
    filled = int(width * frac)
    bar = "#" * filled + "-" * (width - filled)
    sys.stdout.write(f"\r[{bar}] {done}/{total} ({frac * 100:5.1f}%)")
    sys.stdout.flush()
    if done == total:
        sys.stdout.write("\n")


def measure_admissibility(nn_h, h_star_map, goal, sample_rng):
    """violation_rate + mean overestimation magnitude for h_nn, over a
    random sample of up to ADMISSIBILITY_SAMPLE reachable nodes."""
    candidates = list(h_star_map.keys())
    n_sample = min(ADMISSIBILITY_SAMPLE, len(candidates))
    sampled = sample_rng.sample(candidates, n_sample)

    violations = 0
    overestimations = []
    for node in sampled:
        h_star = h_star_map[node]
        h_pred = nn_h(node, goal)
        if h_pred > h_star + EPSILON:
            violations += 1
            overestimations.append(h_pred - h_star)

    violation_rate = violations / n_sample if n_sample else 0.0
    mean_overestimation = (
        sum(overestimations) / len(overestimations) if overestimations else 0.0
    )
    return violation_rate, mean_overestimation


def run_problem(size, problem_id, writer):
    map_seed, sg_seed, sample_seed = problem_seeds(size, problem_id)
    density = OBSTACLE_DENSITY[size]

    grid = GridWorld(size=size, obstacle_density=density, seed=map_seed)
    start, goal = pick_start_goal(grid, random.Random(sg_seed))
    grid.ensure_solvable(start, goal)

    nn_h = NNHeuristic(MODEL_PATH, grid)  # always trained on 50x50, regardless of this grid's size
    h_star_map = None  # computed lazily; only needed for the nn row's admissibility check

    optimal_len = None
    for name, h_func in (("manhattan", manhattan), ("euclidean", euclidean), ("nn", nn_h)):
        path, metrics = astar(grid, start, goal, h_func)
        assert path is not None, f"A*+{name} failed to find a path on a solvable problem"

        if optimal_len is None:  # manhattan runs first -> defines ground truth (admissible)
            optimal_len = metrics.path_length
        suboptimality = metrics.path_length / optimal_len

        if name == "nn":
            if h_star_map is None:
                h_star_map = backward_distances(grid, goal)
            violation_rate, mean_overestimation = measure_admissibility(
                nn_h, h_star_map, goal, random.Random(sample_seed)
            )
        else:
            # manhattan/euclidean are provably admissible & consistent on a
            # 4-connected unit-cost grid -- no violations possible.
            violation_rate, mean_overestimation = 0.0, 0.0

        writer.writerow({
            "size": size,
            "problem_id": problem_id,
            "heuristic": name,
            "nodes_expanded": metrics.nodes_expanded,
            "wall_time_total": metrics.wall_time_total,
            "wall_time_heuristic": metrics.wall_time_heuristic,
            "path_length": metrics.path_length,
            "optimal_len": optimal_len,
            "suboptimality": suboptimality,
            "violation_rate": violation_rate,
            "mean_overestimation": mean_overestimation,
            "h_calls": metrics.h_calls,
        })


def main():
    t_start = time.perf_counter()
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "results.csv"

    total = len(GRID_SIZES) * N_TEST_PROBLEMS
    done = 0
    print_progress(done, total)

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for size in GRID_SIZES:
            for problem_id in range(N_TEST_PROBLEMS):
                run_problem(size, problem_id, writer)
                done += 1
                print_progress(done, total)

    elapsed = time.perf_counter() - t_start
    n_rows = total * len(("manhattan", "euclidean", "nn"))
    print(f"wrote {out_path} ({n_rows} rows), elapsed = {elapsed:.1f}s")

    with open(out_path, newline="") as f:
        counts = Counter(row["size"] for row in csv.DictReader(f))
    print("rows per size:", dict(counts))


if __name__ == "__main__":
    main()
