"""GridWorld environment: 2D grid with random obstacles, 4-connected."""
import random
from collections import deque

import numpy as np


class GridWorld:
    def __init__(self, size, obstacle_density, seed=None):
        self.size = size
        self.obstacle_density = obstacle_density
        self.rng = random.Random(seed)
        self.obstacles = self._generate_obstacles()

    def _generate_obstacles(self):
        cells = [(r, c) for r in range(self.size) for c in range(self.size)]
        n_obstacles = int(len(cells) * self.obstacle_density)
        return set(self.rng.sample(cells, n_obstacles))

    def is_valid(self, pos):
        r, c = pos
        return 0 <= r < self.size and 0 <= c < self.size and pos not in self.obstacles

    def neighbors(self, pos):
        r, c = pos
        candidates = ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
        return [p for p in candidates if self.is_valid(p)]

    def ensure_solvable(self, start, goal, max_attempts=1000):
        """Regenerate obstacles (via BFS check) until start->goal is reachable."""
        attempts = 0
        while True:
            self.obstacles.discard(start)
            self.obstacles.discard(goal)
            if self._bfs_reachable(start, goal):
                return self
            attempts += 1
            if attempts > max_attempts:
                raise RuntimeError(
                    f"Could not generate a solvable {self.size}x{self.size} grid "
                    f"after {max_attempts} attempts"
                )
            self.obstacles = self._generate_obstacles()

    def _bfs_reachable(self, start, goal):
        if not self.is_valid(start) or not self.is_valid(goal):
            return False
        if start == goal:
            return True
        visited = {start}
        queue = deque([start])
        while queue:
            current = queue.popleft()
            for nxt in self.neighbors(current):
                if nxt == goal:
                    return True
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)
        return False

    def to_features(self, pos, goal):
        """7 normalized features for the NN heuristic."""
        r, c = pos
        gr, gc = goal
        size = self.size
        manhattan_dist = abs(r - gr) + abs(c - gc)
        euclidean_dist = ((r - gr) ** 2 + (c - gc) ** 2) ** 0.5
        density = self._local_density(pos)
        return np.array(
            [
                r / size,
                c / size,
                gr / size,
                gc / size,
                manhattan_dist / (2 * size),
                euclidean_dist / ((2 ** 0.5) * size),
                density,
            ],
            dtype=np.float32,
        )

    def _local_density(self, pos, window=1):
        r, c = pos
        cells = 0
        blocked = 0
        for dr in range(-window, window + 1):
            for dc in range(-window, window + 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < self.size and 0 <= cc < self.size:
                    cells += 1
                    if (rr, cc) in self.obstacles:
                        blocked += 1
        return blocked / cells if cells else 0.0
