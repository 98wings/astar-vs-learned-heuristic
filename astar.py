"""Heuristic-agnostic A* search: astar(grid, start, goal, h_func) -> (path, Metrics).

The search code never inspects h_func's identity — the SAME code path runs
classical (Manhattan/Euclidean) and, later, a learned NN heuristic.
"""
import heapq
import time
from dataclasses import dataclass, field


@dataclass
class Metrics:
    nodes_expanded: int = 0
    wall_time_total: float = 0.0
    wall_time_heuristic: float = 0.0
    path_length: int = 0
    h_calls: int = 0
    expanded_nodes: list = field(default_factory=list)


def _reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def astar(grid, start, goal, h_func, track=True, track_expanded=False):
    """Metrics.path_length is the number of steps (edges), i.e. len(path) - 1.

    closed[pos] stores the g-value at which pos was last expanded. A node
    can be re-expanded if a strictly better g is later discovered, since an
    inadmissible heuristic (like the NN one) can otherwise leave A* stuck
    with a suboptimal result instead of terminating cleanly.

    track_expanded collects every expanded position into
    Metrics.expanded_nodes; off by default since it's only needed when
    building NN training pairs, not during the full experiment sweep.
    """
    t_start = time.perf_counter()
    metrics = Metrics()

    def h(pos):
        if track:
            t0 = time.perf_counter()
            value = h_func(pos, goal)
            metrics.wall_time_heuristic += time.perf_counter() - t0
        else:
            value = h_func(pos, goal)
        metrics.h_calls += 1
        return value

    counter = 0
    g_score = {start: 0.0}
    came_from = {}
    closed = {}

    h_start = h(start)
    # heap entries: (f, h, insertion_counter, pos, g) — tie-break on h, then FIFO order
    open_heap = [(h_start, h_start, counter, start, 0.0)]

    while open_heap:
        f_val, h_val, _, current, g_current = heapq.heappop(open_heap)

        if g_current > g_score.get(current, float("inf")) + 1e-9:
            continue  # stale entry: a better path to `current` was found since this was pushed

        if current in closed and closed[current] <= g_current:
            continue  # already expanded at an equal-or-better g

        if current == goal:
            path = _reconstruct_path(came_from, current)
            metrics.path_length = len(path) - 1
            metrics.wall_time_total = time.perf_counter() - t_start
            return path, metrics

        closed[current] = g_current
        metrics.nodes_expanded += 1
        if track_expanded:
            metrics.expanded_nodes.append(current)

        for neighbor in grid.neighbors(current):
            tentative_g = g_current + 1
            if tentative_g < g_score.get(neighbor, float("inf")):
                g_score[neighbor] = tentative_g
                came_from[neighbor] = current
                h_neighbor = h(neighbor)
                counter += 1
                heapq.heappush(
                    open_heap,
                    (tentative_g + h_neighbor, h_neighbor, counter, neighbor, tentative_g),
                )

    metrics.wall_time_total = time.perf_counter() - t_start
    return None, metrics  # no path exists
