"""GATE 4 validation.

  1. results.csv has exactly 270 rows (3 sizes x 30 problems x 3 heuristics).
  2. no missing/NaN values in any field.
  3. suboptimality >= 1.0 for manhattan/euclidean, and (sanity) exactly 1.0,
     since both are provably admissible on this 4-connected unit-cost grid.
"""
import csv
import math
from collections import Counter
from pathlib import Path

RESULTS_PATH = Path(__file__).parent / "results" / "results.csv"

EXPECTED_ROWS = 270
NUMERIC_FIELDS = [
    "size", "problem_id", "nodes_expanded", "wall_time_total",
    "wall_time_heuristic", "path_length", "optimal_len", "suboptimality",
    "violation_rate", "mean_overestimation", "h_calls",
]


def main():
    with open(RESULTS_PATH, newline="") as f:
        rows = list(csv.DictReader(f))

    # --- Check 1: exactly 270 rows ---
    assert len(rows) == EXPECTED_ROWS, f"expected {EXPECTED_ROWS} rows, got {len(rows)}"
    print(f"[OK] results.csv has exactly {len(rows)} rows")

    # --- Check 2: no missing / NaN values ---
    for i, row in enumerate(rows):
        for field in NUMERIC_FIELDS:
            raw = row[field]
            assert raw not in (None, ""), f"row {i}: missing value for {field}"
            value = float(raw)
            assert not math.isnan(value), f"row {i}: NaN in {field}"
        assert row["heuristic"] in ("manhattan", "euclidean", "nn"), (
            f"row {i}: unexpected heuristic {row['heuristic']!r}"
        )
    print(f"[OK] no missing/NaN values across {len(rows)} rows x {len(NUMERIC_FIELDS)} numeric fields")

    # --- Check 3: admissible heuristics are exactly optimal ---
    admissible = [r for r in rows if r["heuristic"] in ("manhattan", "euclidean")]
    for row in admissible:
        sub = float(row["suboptimality"])
        assert sub >= 1.0 - 1e-9, (
            f"{row['heuristic']} size={row['size']} problem={row['problem_id']}: "
            f"suboptimality {sub} < 1.0 (violates the admissibility guarantee!)"
        )
        assert abs(sub - 1.0) < 1e-9, (
            f"{row['heuristic']} size={row['size']} problem={row['problem_id']}: "
            f"suboptimality {sub} != 1.0 (admissible heuristic must find the optimal path)"
        )
    print(f"[OK] all {len(admissible)} manhattan/euclidean rows have suboptimality == 1.0")

    # --- Bonus cross-check: exactly 30 rows per (size, heuristic) combination ---
    counts = Counter((r["size"], r["heuristic"]) for r in rows)
    for size in ("10", "50", "100"):
        for h in ("manhattan", "euclidean", "nn"):
            assert counts[(size, h)] == 30, (
                f"expected 30 rows for size={size} heuristic={h}, got {counts[(size, h)]}"
            )
    print("[OK] exactly 30 rows for every (size, heuristic) combination")

    print("\nGATE 4: PASSED")


if __name__ == "__main__":
    main()
