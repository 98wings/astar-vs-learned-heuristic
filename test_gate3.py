"""GATE 3 validation.

On 20 random solvable 50x50 problems:
  - A*+NN finds a VALID path (4-connected adjacency, no obstacles)
  - the search TERMINATES (guaranteed: g-values are non-negative integers,
    so any node can be reopened only finitely many times before its g
    stabilizes -> the loop cannot run forever, even with an inadmissible h)
  - paths may be suboptimal -- expected and fine, since an inadmissible
    heuristic gives up A*'s optimality guarantee

Prints, per problem: nodes expanded, path_length NN vs path_length
Manhattan (optimal), and the suboptimality ratio NN/optimal.
"""
import random
from pathlib import Path

from config import SEED, TRAIN_GRID, OBSTACLE_DENSITY
from grid import GridWorld
from astar import astar
from heuristics import manhattan, NNHeuristic
from generate_data import pick_start_goal

MODEL_PATH = Path(__file__).parent / "results" / f"model_{TRAIN_GRID}x{TRAIN_GRID}.pt"
N_PROBLEMS = 20


def validate_path(grid, path, start, goal):
    assert path is not None, "no path found (A*+NN failed to reach the goal)"
    assert path[0] == start, f"path does not start at {start}"
    assert path[-1] == goal, f"path does not end at {goal}"
    for pos in path:
        assert grid.is_valid(pos), f"path passes through invalid/obstacle cell {pos}"
    for a, b in zip(path, path[1:]):
        assert b in grid.neighbors(a), f"illegal (non-4-connected) move {a} -> {b}"


def main():
    master_rng = random.Random(SEED + 999)  # distinct stream from generate_data.py
    density = OBSTACLE_DENSITY[TRAIN_GRID]

    ratios = []
    print(f"{'#':>3} {'NN nodes':>9} {'NN len':>7} {'opt len':>8} {'suboptimality':>14}")

    for i in range(N_PROBLEMS):
        map_seed = master_rng.randrange(1 << 30)
        sg_seed = master_rng.randrange(1 << 30)

        grid = GridWorld(size=TRAIN_GRID, obstacle_density=density, seed=map_seed)
        start, goal = pick_start_goal(grid, random.Random(sg_seed))
        grid.ensure_solvable(start, goal)

        nn_h = NNHeuristic(MODEL_PATH, grid)

        path_nn, metrics_nn = astar(grid, start, goal, nn_h)
        # nn_h is passed to astar() exactly like manhattan/euclidean above --
        # astar() never branches on which heuristic it received.
        assert metrics_nn.nodes_expanded <= grid.size * grid.size * 4, (
            "nodes_expanded suspiciously large -- possible runaway reopening"
        )
        validate_path(grid, path_nn, start, goal)

        path_opt, metrics_opt = astar(grid, start, goal, manhattan)
        validate_path(grid, path_opt, start, goal)

        ratio = metrics_nn.path_length / metrics_opt.path_length
        ratios.append(ratio)
        print(
            f"{i + 1:>3} {metrics_nn.nodes_expanded:>9} {metrics_nn.path_length:>7} "
            f"{metrics_opt.path_length:>8} {ratio:>14.3f}"
        )

    print(
        f"\nmean suboptimality ratio = {sum(ratios) / len(ratios):.3f}, "
        f"max = {max(ratios):.3f}, min = {min(ratios):.3f}"
    )
    print(f"\nGATE 3: PASSED ({N_PROBLEMS}/{N_PROBLEMS} valid, terminating A*+NN searches)")


if __name__ == "__main__":
    main()
