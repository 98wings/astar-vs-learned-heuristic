"""All constants in one place, for reproducibility."""

SEED = 42

GRID_SIZES = [10, 50, 100]
OBSTACLE_DENSITY = {10: 0.20, 50: 0.25, 100: 0.25}

N_TEST_PROBLEMS = 30
N_TRAIN_MAPS = 200
MOVES = 4

NN_HIDDEN = [64, 32]
NN_EPOCHS = 100
NN_BATCH = 32
NN_LR = 1e-3
TRAIN_GRID = 50
