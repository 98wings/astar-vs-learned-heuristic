"""Reads results/results.csv, produces figures G1-G5 (300 DPI, exported to
../Visuals/) and results/summary_stats.csv.

Categorical color assignment is fixed across every figure (manhattan/
euclidean/nn always map to the same hue), so color always identifies the
same heuristic regardless of how a chart ranks them.
"""
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).parent
RESULTS_DIR = ROOT_DIR / "results"
VISUALS_DIR = ROOT_DIR / "Visuals"

COLOR = {
    "manhattan": "#2a78d6",  # categorical slot 1 (blue)
    "euclidean": "#1baf7a",  # categorical slot 2 (aqua)
    "nn": "#eda100",         # categorical slot 3 (yellow)
}
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID_COLOR = "#e1e0d9"

HEURISTIC_ORDER = ["manhattan", "euclidean", "nn"]
HEURISTIC_LABEL = {"manhattan": "Manhattan", "euclidean": "Euclidean", "nn": "Neural Net"}
SIZES = [10, 50, 100]
TRAIN_SIZE = 50

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 10.5,
    "axes.edgecolor": GRID_COLOR,
    "axes.labelcolor": INK_SECONDARY,
    "text.color": INK_PRIMARY,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "axes.titlesize": 11.5,
    "axes.titleweight": "bold",
    "figure.dpi": 100,
    "savefig.dpi": 300,
})


def style_axis(ax, log_y=False):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    if log_y:
        ax.set_yscale("log")


def asymmetric_lower(means, stds):
    """Clip the lower error bar so mean - err stays > 0 (required for log axes)."""
    return np.where(means - stds > 0, stds, means * 0.999)


def grouped_bar(ax, summary, value_col, ylabel, title, log_y=False):
    x = np.arange(len(SIZES))
    width = 0.24
    for i, h in enumerate(HEURISTIC_ORDER):
        sub = summary[summary["heuristic"] == h].set_index("size").reindex(SIZES)
        means = sub[f"{value_col}_mean"].values
        stds = sub[f"{value_col}_std"].values
        offset = (i - 1) * width
        ax.bar(
            x + offset, means, width=width * 0.9,
            yerr=[asymmetric_lower(means, stds), stds], capsize=3,
            color=COLOR[h], edgecolor=INK_PRIMARY, linewidth=0.5,
            label=HEURISTIC_LABEL[h], zorder=3,
            error_kw={"elinewidth": 1, "ecolor": INK_SECONDARY},
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}x{s}" for s in SIZES])
    ax.set_xlabel("Grid size")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    style_axis(ax, log_y=log_y)


def plot_g1_g2(summary):
    """G1 (nodes, log y) + G2 (wall-clock) side by side -- the 'rigged metric' figure."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    grouped_bar(axes[0], summary, "nodes_expanded", "Nodes expanded (log scale)",
                "G1: Nodes Expanded by Heuristic x Grid Size", log_y=True)
    grouped_bar(axes[1], summary, "wall_time_total", "Wall-clock time (s)",
                "G2: Wall-Clock Time by Heuristic x Grid Size", log_y=False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.06))
    fig.suptitle(
        'The "rigged metric" figure -- nodes expanded and wall-clock time '
        "don't always agree on a winner",
        fontsize=10.5, fontweight="normal", color=INK_SECONDARY, y=1.14,
    )
    fig.tight_layout()
    fig.savefig(VISUALS_DIR / "G1_G2_rigged_metric.png", bbox_inches="tight")
    plt.close(fig)


def plot_g3(df):
    """Distribution of NN path suboptimality per size, with outliers annotated."""
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    data = [df[(df.heuristic == "nn") & (df["size"] == s)]["suboptimality"].values for s in SIZES]

    ax.boxplot(
        data, positions=range(len(SIZES)), widths=0.35, showfliers=False, patch_artist=True,
        boxprops=dict(facecolor=COLOR["nn"], alpha=0.35, edgecolor=INK_PRIMARY, linewidth=1),
        medianprops=dict(color=INK_PRIMARY, linewidth=1.5),
        whiskerprops=dict(color=INK_SECONDARY, linewidth=1),
        capprops=dict(color=INK_SECONDARY, linewidth=1),
    )

    rng = np.random.RandomState(0)
    total_outliers = 0
    for i, size in enumerate(SIZES):
        sub = df[(df.heuristic == "nn") & (df["size"] == size)]
        q1, q3 = np.percentile(sub["suboptimality"], [25, 75])
        upper_fence = q3 + 1.5 * (q3 - q1)
        jitter = rng.uniform(-0.12, 0.12, size=len(sub))
        ax.scatter(i + jitter, sub["suboptimality"], s=16, color=COLOR["nn"],
                   edgecolor=INK_PRIMARY, linewidth=0.3, alpha=0.75, zorder=3)
        outliers = sub[sub["suboptimality"] > upper_fence]
        total_outliers += len(outliers)
        for _, row in outliers.iterrows():
            ax.annotate(f"#{int(row.problem_id)}", (i, row.suboptimality),
                        xytext=(6, 3), textcoords="offset points",
                        fontsize=8, color=INK_SECONDARY)

    all_values = np.concatenate(data)
    lo, hi = float(all_values.min()), float(all_values.max())
    pad = max(0.02, (hi - lo) * 2)
    ax.set_ylim(min(1.0, lo) - pad, max(1.0, hi) + pad)

    ax.axhline(1.0, color=INK_MUTED, linewidth=1, linestyle="--", zorder=1)
    ax.set_xticks(range(len(SIZES)))
    ax.set_xticklabels([f"{s}x{s}" for s in SIZES])
    ax.set_xlabel("Grid size")
    ax.set_ylabel("Suboptimality ratio (NN path length / optimal)")
    ax.set_title("G3: NN Path Suboptimality Distribution")
    style_axis(ax)
    if total_outliers == 0:
        ax.text(0.5, 0.94,
                 f"All {len(all_values)} NN paths matched the optimal length in this "
                 "run (no suboptimality outliers)",
                 transform=ax.transAxes, ha="center", va="top", fontsize=9,
                 color=INK_SECONDARY, style="italic")
    fig.tight_layout()
    fig.savefig(VISUALS_DIR / "G3_nn_suboptimality_distribution.png")
    plt.close(fig)


def plot_g4(summary):
    """Admissibility violation rate + mean overestimation magnitude, by size."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    nn = summary[summary["heuristic"] == "nn"].set_index("size").reindex(SIZES)
    x = np.arange(len(SIZES))

    ax = axes[0]
    means = nn["violation_rate_mean"].values * 100
    stds = nn["violation_rate_std"].values * 100
    ax.bar(x, means, width=0.5, yerr=[asymmetric_lower(means, stds), stds], capsize=3,
           color=COLOR["nn"], edgecolor=INK_PRIMARY, linewidth=0.5, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels([f"{s}x{s}" for s in SIZES])
    ax.set_xlabel("Grid size")
    ax.set_ylabel("Admissibility violation rate (%)")
    ax.set_title("G4a: NN Admissibility Violations")
    style_axis(ax)

    ax = axes[1]
    means = nn["mean_overestimation_mean"].values
    stds = nn["mean_overestimation_std"].values
    ax.bar(x, means, width=0.5, yerr=[asymmetric_lower(means, stds), stds], capsize=3,
           color=COLOR["nn"], edgecolor=INK_PRIMARY, linewidth=0.5, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels([f"{s}x{s}" for s in SIZES])
    ax.set_xlabel("Grid size")
    ax.set_ylabel("Mean overestimation magnitude (steps)")
    ax.set_title("G4b: Overestimation Magnitude (violating nodes only)")
    style_axis(ax)

    fig.suptitle("G4: Admissibility Violations -- NN Heuristic", fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.savefig(VISUALS_DIR / "G4_admissibility_violations.png", bbox_inches="tight")
    plt.close(fig)


def plot_g5(summary, df):
    """NN trained on 50x50 only, evaluated at 10/50/100 to see how well it transfers."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    nn = summary[summary["heuristic"] == "nn"].set_index("size").reindex(SIZES)
    x = np.arange(len(SIZES))
    train_idx = SIZES.index(TRAIN_SIZE)

    def mark_training_size(ax, y_at_train):
        ax.annotate("trained here", (train_idx, y_at_train), xytext=(0, 14),
                    textcoords="offset points", ha="center", fontsize=8.5,
                    color=INK_SECONDARY,
                    arrowprops=dict(arrowstyle="-", color=INK_MUTED, lw=0.8))

    ax = axes[0]
    means = nn["suboptimality_mean"].values
    stds = nn["suboptimality_std"].values
    ax.bar(x, means, width=0.5, yerr=[asymmetric_lower(means, stds), stds], capsize=3,
           color=COLOR["nn"], edgecolor=INK_PRIMARY, linewidth=0.5, zorder=3)
    ax.axhline(1.0, color=INK_MUTED, linewidth=1, linestyle="--", zorder=1)
    ax.set_xticks(x); ax.set_xticklabels([f"{s}x{s}" for s in SIZES])
    ax.set_xlabel("Grid size")
    ax.set_ylabel("Mean suboptimality ratio")
    ax.set_title("G5a: NN Path Quality by Size")
    style_axis(ax)
    y_top = max(1.0, float((means + stds).max()))
    ax.set_ylim(0, y_top * 1.2 + 0.05)  # headroom so the annotation clears the title
    mark_training_size(ax, means[train_idx] + stds[train_idx])

    ax = axes[1]
    ratio_means, ratio_stds = [], []
    for size in SIZES:
        nn_nodes = df[(df.heuristic == "nn") & (df["size"] == size)].sort_values("problem_id")["nodes_expanded"].values
        man_nodes = df[(df.heuristic == "manhattan") & (df["size"] == size)].sort_values("problem_id")["nodes_expanded"].values
        ratio = nn_nodes / man_nodes
        ratio_means.append(ratio.mean())
        ratio_stds.append(ratio.std())
    ratio_means = np.array(ratio_means)
    ratio_stds = np.array(ratio_stds)
    ax.bar(x, ratio_means, width=0.5,
           yerr=[asymmetric_lower(ratio_means, ratio_stds), ratio_stds], capsize=3,
           color=COLOR["nn"], edgecolor=INK_PRIMARY, linewidth=0.5, zorder=3)
    ax.axhline(1.0, color=INK_MUTED, linewidth=1, linestyle="--", zorder=1)
    ax.set_xticks(x); ax.set_xticklabels([f"{s}x{s}" for s in SIZES])
    ax.set_xlabel("Grid size")
    ax.set_ylabel("Nodes expanded ratio (NN / Manhattan, log scale)")
    ax.set_title("G5b: Search Efficiency Relative to Manhattan")
    # log scale: the 100x100 error bar spans an order of magnitude more than
    # 10x10/50x50 -- a linear axis would flatten the smaller two bars to a sliver.
    style_axis(ax, log_y=True)
    mark_training_size(ax, ratio_means[train_idx] + ratio_stds[train_idx])

    fig.suptitle("G5: NN Transfer Across Grid Sizes (trained on 50x50 only)",
                 fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.savefig(VISUALS_DIR / "G5_transfer_across_sizes.png", bbox_inches="tight")
    plt.close(fig)


def write_summary_csv(summary):
    """Mean +/- std of every metric by (size x heuristic), plus the data-generation
    and training wall-times."""
    out_path = RESULTS_DIR / "summary_stats.csv"
    summary.to_csv(out_path, index=False)

    gen_log = (RESULTS_DIR / "data_generation_log.txt").read_text()
    train_log = (RESULTS_DIR / "training_log.txt").read_text()
    gen_time = float(re.search(r"data generation wall-time:\s*([\d.]+)\s*s", gen_log).group(1))
    train_time = float(re.search(r"training wall-time:\s*([\d.]+)\s*s", train_log).group(1))

    with open(out_path, "a", newline="") as f:
        f.write("\n")
        f.write("metric,value\n")
        f.write(f"data_generation_wall_time_s,{gen_time}\n")
        f.write(f"training_wall_time_s,{train_time}\n")

    return out_path


def print_terminal_summary(df, summary):
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    for size in SIZES:
        s = summary[summary["size"] == size].set_index("heuristic")
        nodes_winner = s["nodes_expanded_mean"].idxmin()
        time_winner = s["wall_time_total_mean"].idxmin()
        crossover = " <-- rigged metric: winners disagree" if nodes_winner != time_winner else ""

        print(f"\nsize {size}x{size}:")
        print(f"  winner on NODES:      {HEURISTIC_LABEL[nodes_winner]:<10} "
              f"({s.loc[nodes_winner, 'nodes_expanded_mean']:.1f} nodes)")
        print(f"  winner on WALL-CLOCK: {HEURISTIC_LABEL[time_winner]:<10} "
              f"({s.loc[time_winner, 'wall_time_total_mean'] * 1000:.2f} ms){crossover}")

        nn_rows = df[(df["size"] == size) & (df["heuristic"] == "nn")]
        violation_rate_mean = nn_rows["violation_rate"].mean() * 100
        pct_optimal = (nn_rows["suboptimality"] <= 1.0 + 1e-9).mean() * 100
        print(f"  NN mean violation_rate:       {violation_rate_mean:.1f}%")
        print(f"  NN problems solved optimally: {pct_optimal:.1f}% ({int(round(pct_optimal / 100 * len(nn_rows)))}/{len(nn_rows)})")
    print("\n" + "=" * 72)


def main():
    df = pd.read_csv(RESULTS_DIR / "results.csv")
    VISUALS_DIR.mkdir(exist_ok=True)

    metrics = ["nodes_expanded", "wall_time_total", "wall_time_heuristic", "path_length",
               "suboptimality", "violation_rate", "mean_overestimation", "h_calls"]
    grouped = df.groupby(["size", "heuristic"])[metrics].agg(["mean", "std"])
    grouped.columns = [f"{metric}_{stat}" for metric, stat in grouped.columns]
    summary = grouped.reset_index()

    summary_path = write_summary_csv(summary)

    plot_g1_g2(summary)
    plot_g3(df)
    plot_g4(summary)
    plot_g5(summary, df)

    print_terminal_summary(df, summary)
    print(f"\nfigures written to {VISUALS_DIR}")
    print(f"summary table written to {summary_path}")


if __name__ == "__main__":
    main()
