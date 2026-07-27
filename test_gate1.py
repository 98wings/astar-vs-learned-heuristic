"""GATE 1 validation.

  1. On a hand-checkable 10x10 map, A*+Manhattan returns the known-optimal path.
  2. nodes_expanded(Manhattan) <= nodes_expanded(Dijkstra, h=0).
"""
from grid import GridWorld
from astar import astar
from heuristics import manhattan, euclidean


def zero_heuristic(pos, goal):
    """h(n) = 0 everywhere -> A* degenerates to Dijkstra. Baseline for GATE 1."""
    return 0.0


def build_hand_checkable_grid():
    """10x10 grid, two full-width walls with offset single-cell gaps.

    Wall at row 3: passable only at column 8.
    Wall at row 6: passable only at column 1.
    Rows 0,1,2,4,5,7,8,9 are completely open.

    Hand-computed optimal path (0,0) -> (9,9):
      (0,0) -> (3,8): must cross the row-3 gap -> |3-0| + |8-0| = 11
      (3,8) -> (6,1): must cross the row-6 gap -> |6-3| + |1-8| = 10
      (6,1) -> (9,9): open space                -> |9-6| + |9-1| = 11
      TOTAL = 11 + 10 + 11 = 32 steps

    Because rows 0-2, 4-5, 7-9 have zero obstacles, each leg is free to
    move in straight monotone steps and therefore achieves its own
    Manhattan lower bound exactly -> 32 is provably the true optimum,
    not just a lower bound.
    """
    grid = GridWorld(size=10, obstacle_density=0.0, seed=1)
    grid.obstacles = set()
    for c in range(10):
        if c != 8:
            grid.obstacles.add((3, c))
        if c != 1:
            grid.obstacles.add((6, c))
    return grid


def validate_path(grid, path, start, goal):
    assert path is not None, "no path found"
    assert path[0] == start, f"path does not start at {start}"
    assert path[-1] == goal, f"path does not end at {goal}"
    for pos in path:
        assert grid.is_valid(pos), f"path passes through invalid/obstacle cell {pos}"
    for a, b in zip(path, path[1:]):
        assert b in grid.neighbors(a), f"illegal move {a} -> {b}"


def main():
    grid = build_hand_checkable_grid()
    start, goal = (0, 0), (9, 9)
    EXPECTED_OPTIMAL = 32

    # --- Check 1: A* + Manhattan finds the known-optimal path ---
    path_m, metrics_m = astar(grid, start, goal, manhattan)
    validate_path(grid, path_m, start, goal)
    assert metrics_m.path_length == EXPECTED_OPTIMAL, (
        f"A*+Manhattan path_length={metrics_m.path_length} != hand-computed optimal {EXPECTED_OPTIMAL}"
    )
    print(
        f"[OK] A*+Manhattan path_length = {metrics_m.path_length} "
        f"(matches hand-computed optimal), nodes_expanded = {metrics_m.nodes_expanded}"
    )

    # --- Sanity: A* + Euclidean (also admissible) finds the same optimum ---
    path_e, metrics_e = astar(grid, start, goal, euclidean)
    validate_path(grid, path_e, start, goal)
    assert metrics_e.path_length == EXPECTED_OPTIMAL, (
        f"A*+Euclidean path_length={metrics_e.path_length} != hand-computed optimal {EXPECTED_OPTIMAL}"
    )
    print(
        f"[OK] A*+Euclidean  path_length = {metrics_e.path_length} "
        f"(matches hand-computed optimal), nodes_expanded = {metrics_e.nodes_expanded}"
    )

    # --- Check 2: nodes(Manhattan) <= nodes(Dijkstra, h=0) ---
    path_d, metrics_d = astar(grid, start, goal, zero_heuristic)
    validate_path(grid, path_d, start, goal)
    assert metrics_d.path_length == EXPECTED_OPTIMAL, (
        f"Dijkstra(h=0) path_length={metrics_d.path_length} != hand-computed optimal {EXPECTED_OPTIMAL}"
    )
    print(
        f"[OK] Dijkstra(h=0) path_length = {metrics_d.path_length}, "
        f"nodes_expanded = {metrics_d.nodes_expanded}"
    )

    assert metrics_m.nodes_expanded <= metrics_d.nodes_expanded, (
        f"nodes(Manhattan)={metrics_m.nodes_expanded} > nodes(Dijkstra)={metrics_d.nodes_expanded}"
    )
    print(
        f"[OK] nodes(Manhattan)={metrics_m.nodes_expanded} <= "
        f"nodes(Dijkstra h=0)={metrics_d.nodes_expanded}"
    )

    print("\nGATE 1: PASSED")


if __name__ == "__main__":
    main()
