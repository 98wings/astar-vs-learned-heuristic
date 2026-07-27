"""PyTorch MLP heuristic: 7 normalized input features -> 1 predicted h(n).

Input features (all normalized to [0,1], see grid.py GridWorld.to_features):
    x/size, y/size, gx/size, gy/size,
    manhattan(pos,goal)/(2*size),
    euclidean(pos,goal)/(sqrt(2)*size),
    local obstacle density in 3x3 window around pos

Including manhattan/euclidean as INPUT features means the network only has
to learn the CORRECTION over the classical heuristic (residual learning) —
realistic and trains fast on CPU.
"""
import torch.nn as nn

from config import NN_HIDDEN

INPUT_DIM = 7


class HeuristicMLP(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden=None):
        super().__init__()
        hidden = hidden if hidden is not None else NN_HIDDEN
        layers = []
        prev = input_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)
