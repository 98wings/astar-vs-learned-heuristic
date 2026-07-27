"""GATE 2 validation.

  1. val MAE < 15% of mean h* on the validation split.
  2. Sanity: h_nn(node adjacent to goal) ~= 1 (loose tolerance: [0, 3]).
"""
import torch

from config import TRAIN_GRID
from grid import GridWorld
from nn_model import HeuristicMLP
from train import RESULTS_DIR, load_dataset, split_train_val, evaluate
import torch.nn as nn


def main():
    # --- Check 1: val MAE < 15% of mean h* ---
    X, y = load_dataset()
    _, _, X_val, y_val = split_train_val(X, y)
    X_val_t = torch.from_numpy(X_val)
    y_val_t = torch.from_numpy(y_val)

    model = HeuristicMLP()
    model.load_state_dict(torch.load(RESULTS_DIR / f"model_{TRAIN_GRID}x{TRAIN_GRID}.pt"))

    val_mse, val_mae = evaluate(model, X_val_t, y_val_t, nn.MSELoss())
    mean_h_star = float(y_val.mean())
    threshold = 0.15 * mean_h_star

    print(f"val MAE = {val_mae:.4f}, mean h* = {mean_h_star:.4f}, "
          f"threshold (15%) = {threshold:.4f}")
    assert val_mae < threshold, (
        f"val MAE {val_mae:.4f} >= 15% of mean h* ({threshold:.4f})"
    )
    print(f"[OK] val MAE {val_mae:.4f} < 15% of mean h* ({threshold:.4f})")

    # --- Check 2: sanity — node adjacent to goal should predict h ~ 1 ---
    grid = GridWorld(size=TRAIN_GRID, obstacle_density=0.0, seed=1)
    goal = (TRAIN_GRID // 2, TRAIN_GRID // 2)
    adjacent = (goal[0], goal[1] + 1)
    assert adjacent in grid.neighbors(goal), "test setup error: node is not actually adjacent"

    features = torch.from_numpy(grid.to_features(adjacent, goal)).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        pred = model(features).item()

    print(f"h_nn(adjacent-to-goal) = {pred:.4f}")
    assert 0.0 <= pred <= 3.0, (
        f"h_nn(adjacent-to-goal) = {pred:.4f}, expected in loose range [0, 3]"
    )
    print(f"[OK] h_nn(adjacent-to-goal) = {pred:.4f} within tolerance [0, 3]")

    print("\nGATE 2: PASSED")


if __name__ == "__main__":
    main()
