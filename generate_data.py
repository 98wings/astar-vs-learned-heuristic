"""Generate NN training data from N_TRAIN_MAPS random 50x50 maps.

For each map: pick a random solvable (start, goal) pair, run A*+Manhattan
forward to get the set of nodes a classical search actually expands, then
run one BFS backward from the goal (unit edge costs, so BFS distance is
exactly the optimal cost-to-goal) to get h*(n) for every reachable node.
Each expanded node's features and h*(n) become one training pair.

Total wall-time is logged to results/data_generation_log.txt.
"""
import random
import time
from collections import deque
from pathlib import Path

import numpy as np

from config import SEED, N_TRAIN_MAPS, TRAIN_GRID, OBSTACLE_DENSITY
from grid import GridWorld
from astar import astar
from heuristics import manhattan

RESULTS_DIR = Path(__file__).parent / "results"


def backward_distances(grid, goal):
    """Exact h*(n) for every node reachable from goal (BFS, unit edge costs)."""
    dist = {goal: 0}
    queue = deque([goal])
    while queue:
        current = queue.popleft()
        for neighbor in grid.neighbors(current):
            if neighbor not in dist:
                dist[neighbor] = dist[current] + 1
                queue.append(neighbor)
    return dist


def pick_start_goal(grid, rng):
    open_cells = [
        (r, c)
        for r in range(grid.size)
        for c in range(grid.size)
        if (r, c) not in grid.obstacles
    ]
    start, goal = rng.sample(open_cells, 2)
    return start, goal


def generate_map_samples(map_seed, start_goal_rng):
    density = OBSTACLE_DENSITY[TRAIN_GRID]
    grid = GridWorld(size=TRAIN_GRID, obstacle_density=density, seed=map_seed)
    start, goal = pick_start_goal(grid, start_goal_rng)
    grid.ensure_solvable(start, goal)

    _, metrics = astar(grid, start, goal, manhattan, track_expanded=True)
    h_star = backward_distances(grid, goal)

    features, targets = [], []
    for node in metrics.expanded_nodes:
        if node in h_star:  # always true for a solvable problem, guard for safety
            features.append(grid.to_features(node, goal))
            targets.append(h_star[node])
    return features, targets


def main():
    t_start = time.perf_counter()
    master_rng = random.Random(SEED)

    all_features, all_targets = [], []
    for i in range(N_TRAIN_MAPS):
        map_seed = master_rng.randrange(1 << 30)
        start_goal_seed = master_rng.randrange(1 << 30)
        features, targets = generate_map_samples(map_seed, random.Random(start_goal_seed))
        all_features.extend(features)
        all_targets.extend(targets)
        if (i + 1) % 20 == 0 or i == N_TRAIN_MAPS - 1:
            print(f"  map {i + 1}/{N_TRAIN_MAPS}: {len(all_targets)} samples so far")

    X = np.stack(all_features).astype(np.float32)
    y = np.array(all_targets, dtype=np.float32)
    elapsed = time.perf_counter() - t_start

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"training_data_{TRAIN_GRID}x{TRAIN_GRID}.npz"
    np.savez(out_path, X=X, y=y)

    log_lines = [
        f"maps generated:        {N_TRAIN_MAPS}",
        f"grid size:              {TRAIN_GRID}x{TRAIN_GRID}",
        f"obstacle density:       {OBSTACLE_DENSITY[TRAIN_GRID]}",
        f"seed:                   {SEED}",
        f"total (state, h*) pairs: {len(y)}",
        f"h* mean / std:          {y.mean():.3f} / {y.std():.3f}",
        f"data generation wall-time: {elapsed:.2f} s",
        f"saved to:               {out_path}",
    ]
    log_text = "\n".join(log_lines)
    print("\n" + log_text)
    (RESULTS_DIR / "data_generation_log.txt").write_text(log_text + "\n")


if __name__ == "__main__":
    main()
