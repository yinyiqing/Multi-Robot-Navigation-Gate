#!/usr/bin/env python3
"""Plot zero-update robot-count generalization from frozen Q1 statistics."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATS = (
    ROOT
    / "experiments/03_保留专门化/02_论文主线"
    / "26_数量泛化与外部切换基线/local_data/q1/results/q1_statistics.json"
)
DEFAULT_OUTPUT = ROOT / "paper/generated/figS1_generalization/generalization.svg"

PINK = "#C47FA0"
BASELINE = "#7F8C96"
CONNECTOR = "#B9C7CD"
INK = "#4E5D66"
GRID = "#D9E0E5"


def style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8.0,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.5,
        "axes.linewidth": 0.65,
        "svg.fonttype": "none",
        "svg.hashsalt": "piroute-q1-generalization",
        "axes.unicode_minus": False,
    })


def metric_values(payload, robot_count, metric, scale):
    block = payload["vehicle_counts"][robot_count]
    base = scale * block["pooled_descriptive"]["5a"][metric]
    method = scale * block["pooled_descriptive"]["b2"][metric]
    effect = block["exploratory_b2_minus_5a"][metric]
    difference = scale * effect["mean_difference"]
    interval = [scale * value for value in effect["scene_cluster_bca_95_ci"]]
    return base, method, difference, interval


def draw_panel(
    ax, payload, panel, title, metric, scale, xlim, xlabel, value_unit, effect_unit
):
    for y, robot_count in [(1, "3"), (0, "7")]:
        base, method, difference, interval = metric_values(
            payload, robot_count, metric, scale
        )
        ax.plot([base, method], [y, y], color=CONNECTOR, linewidth=1.5, zorder=1)
        ax.scatter(
            base, y, s=42, color=BASELINE, edgecolor="white", linewidth=0.7, zorder=3
        )
        ax.scatter(
            method, y, s=52, color=PINK, edgecolor="white", linewidth=0.7, zorder=3
        )

        span = xlim[1] - xlim[0]
        if abs(method - base) < 0.08 * span:
            base_offset, base_align, base_y = 0, "center", 7
            method_offset, method_align, method_y = 0, "center", 18
        elif method >= base:
            base_offset, base_align, base_y = -5, "right", 7
            method_offset, method_align, method_y = 5, "left", 7
        else:
            base_offset, base_align, base_y = 5, "left", 7
            method_offset, method_align, method_y = -5, "right", 7
        ax.annotate(
            f"{base:.2f}{value_unit}", (base, y), xytext=(base_offset, base_y),
            textcoords="offset points", ha=base_align, va="bottom",
            fontsize=6.8, color=BASELINE,
        )
        ax.annotate(
            f"{method:.2f}{value_unit}", (method, y), xytext=(method_offset, method_y),
            textcoords="offset points", ha=method_align, va="bottom",
            fontsize=6.8, color=PINK, fontweight="bold",
        )
        ax.text(
            0.98, y - 0.25,
            f"{difference:+.2f} {effect_unit}  [{interval[0]:+.2f}, {interval[1]:+.2f}]",
            transform=ax.get_yaxis_transform(), ha="right", va="center",
            fontsize=6.15, color=INK,
        )
        ax.text(
            -0.055, y, f"{robot_count} robots", transform=ax.get_yaxis_transform(),
            ha="right", va="center", fontsize=7.4, color=INK,
        )

    ax.set_xlim(*xlim)
    ax.set_ylim(-0.48, 1.48)
    ax.set_yticks([])
    ax.set_xlabel(xlabel, color=INK)
    ax.set_title(f"({panel}) {title}", color=INK, fontweight="bold", pad=7)
    ax.grid(axis="x", color=GRID, linewidth=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.spines["bottom"].set_linewidth(0.65)


def make_figure(payload):
    style()
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 3.2))
    draw_panel(
        axes[0], payload, "a", "Robot success rate", "agent_success", 100,
        (0, 100), "rate (%)\n(higher is better)", "%", "pp",
    )
    draw_panel(
        axes[1], payload, "b", "Robot collision rate", "collision", 100,
        (0, 50), "rate (%)\n(lower is better)", "%", "pp",
    )
    draw_panel(
        axes[2], payload, "c", "Completion-step cost", "raw_steps", 1,
        (0, 45), "raw termination steps\n(lower is faster)", "", "steps",
    )

    legend = [
        Line2D(
            [0], [0], marker="o", color="none", markerfacecolor=BASELINE,
            markeredgecolor="white", markersize=5.8, label="5A",
        ),
        Line2D(
            [0], [0], marker="o", color="none", markerfacecolor=PINK,
            markeredgecolor="white", markersize=5.8, label="PIRoute",
        ),
    ]
    fig.legend(
        handles=legend, loc="lower center", ncol=2, frameon=False, fontsize=7.3,
        bbox_to_anchor=(0.5, 0.065), handletextpad=0.35, columnspacing=1.1,
    )
    fig.suptitle(
        "Zero-update deployment across robot counts",
        fontsize=10.5, fontweight="bold", color=INK, y=0.965,
    )
    fig.text(
        0.015, 0.018,
        "Within-count comparisons only; brackets are exploratory scene-cluster BCa 95% CIs. "
        "Full-episode success remains reported in the supplementary table.",
        fontsize=6.5, color="#66747C",
    )
    fig.subplots_adjust(
        left=0.105, right=0.99, bottom=0.27, top=0.82, wspace=0.44
    )
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    payload = json.loads(args.stats.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig = make_figure(payload)
    fig.savefig(args.output, format="svg", facecolor="white", metadata={"Date": None})
    plt.close(fig)
    args.output.write_text(
        "\n".join(
            line.rstrip()
            for line in args.output.read_text(encoding="utf-8").splitlines()
        )
        + "\n",
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
