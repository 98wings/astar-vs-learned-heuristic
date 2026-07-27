"""Train the NN heuristic on the 50x50 dataset produced by generate_data.py.

Standard supervised regression: loss = MSE(pred, h*). This symmetric loss
is used deliberately rather than "fixed" to only penalize overestimation:
the resulting admissibility violations are exactly what gets measured
later in experiments.py, not something to engineer away here.

80/20 train/val split, early stopping on val loss. Total training wall-time
and epoch count are logged to results/training_log.txt. Saves
results/model_50x50.pt.
"""
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import SEED, NN_EPOCHS, NN_BATCH, NN_LR, TRAIN_GRID
from nn_model import HeuristicMLP

RESULTS_DIR = Path(__file__).parent / "results"
PATIENCE = 10  # early-stopping patience, in epochs without val-loss improvement


def load_dataset():
    data = np.load(RESULTS_DIR / f"training_data_{TRAIN_GRID}x{TRAIN_GRID}.npz")
    return data["X"], data["y"]


def split_train_val(X, y, val_fraction=0.2, seed=SEED):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(X))
    n_val = int(len(X) * val_fraction)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


def evaluate(model, X, y, loss_fn):
    model.eval()
    with torch.no_grad():
        pred = model(X)
        mse = loss_fn(pred, y).item()
        mae = torch.mean(torch.abs(pred - y)).item()
    return mse, mae


def main():
    t_start = time.perf_counter()
    torch.manual_seed(SEED)

    X, y = load_dataset()
    X_train, y_train, X_val, y_val = split_train_val(X, y)

    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train)
    X_val_t = torch.from_numpy(X_val)
    y_val_t = torch.from_numpy(y_val)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t), batch_size=NN_BATCH, shuffle=True
    )

    model = HeuristicMLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=NN_LR)
    loss_fn = nn.MSELoss()

    best_val_mse = float("inf")
    best_state = None
    epochs_without_improvement = 0
    epochs_run = 0

    for epoch in range(1, NN_EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()

        val_mse, val_mae = evaluate(model, X_val_t, y_val_t, loss_fn)
        epochs_run = epoch

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"  epoch {epoch:3d}: val_mse={val_mse:.3f} val_mae={val_mae:.3f}")

        if epochs_without_improvement >= PATIENCE:
            print(f"  early stopping at epoch {epoch} (no val improvement for {PATIENCE} epochs)")
            break

    model.load_state_dict(best_state)
    final_val_mse, final_val_mae = evaluate(model, X_val_t, y_val_t, loss_fn)
    elapsed = time.perf_counter() - t_start

    RESULTS_DIR.mkdir(exist_ok=True)
    model_path = RESULTS_DIR / f"model_{TRAIN_GRID}x{TRAIN_GRID}.pt"
    torch.save(model.state_dict(), model_path)

    log_lines = [
        f"dataset size:            {len(X)} (train={len(X_train)}, val={len(X_val)})",
        f"epochs run:               {epochs_run} / {NN_EPOCHS} (patience={PATIENCE})",
        f"best val MSE:             {best_val_mse:.4f}",
        f"final val MAE:            {final_val_mae:.4f}",
        f"val h* mean:              {y_val.mean():.4f}",
        f"training wall-time:       {elapsed:.2f} s",
        f"model saved to:           {model_path}",
    ]
    log_text = "\n".join(log_lines)
    print("\n" + log_text)
    (RESULTS_DIR / "training_log.txt").write_text(log_text + "\n")


if __name__ == "__main__":
    main()
