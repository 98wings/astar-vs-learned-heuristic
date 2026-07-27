"""Manhattan, Euclidean, and the learned NN heuristic all share the same
(pos, goal) -> float call interface, so astar() cannot tell them apart.
"""
import math

import torch

from nn_model import HeuristicMLP


def manhattan(pos, goal):
    return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])


def euclidean(pos, goal):
    return math.hypot(pos[0] - goal[0], pos[1] - goal[1])


class NNHeuristic:
    """Learned heuristic: wraps a trained HeuristicMLP behind the classical
    heuristic interface, __call__(pos, goal) -> float, so it is passable to
    astar() exactly like manhattan/euclidean.
    """

    def __init__(self, model_path, grid):
        self.grid = grid
        self.model = HeuristicMLP()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def __call__(self, pos, goal):
        features = torch.from_numpy(self.grid.to_features(pos, goal)).unsqueeze(0)
        with torch.no_grad():
            # A* queries one node at a time; no batching, so this per-call
            # overhead is the real cost of using the model as a heuristic.
            value = self.model(features).item()
        return value
