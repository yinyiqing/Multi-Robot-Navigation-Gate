#!/usr/bin/env python3
"""Plot Router mode/probability timelines for selected qualitative rollouts."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


MODE_COLORS = {
    "standard": "#A7CED9",
    "dense": "#DDB2C5",
    "inactive": "#EEF1F2",
}
INK = "#4E5D66"
GRID = "#D9E0E5"
PANEL_TITLES = {
    "PIRoute succeeds, 5A fails": "PIRoute succeeds; 5A fails",
    "Both succeed": "Both succeed",
    "Both fail": "Neither succeeds",
}


def load_trajectory(path):
    episodes = defaultdict(list)
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                episodes[int(record["episode"])].append(record)
    return dict(episodes)


def load_selected(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return list(zip(payload["episode_indices"], payload["selected"]))


def add_lane(ax, records, robot, lane):
    steps = [int(record["step"]) for record in records]
    if not steps:
        return
    mode_values = []
    probabilities = []
    for record in records:
        mode = (record.get("actor_modes") or {}).get(robot)
        mode_values.append(mode if mode in ("standard", "dense") else "inactive")
        probabilities.append((record.get("gate_probabilities") or {}).get(robot))

    # A constant-height strip makes every robot's decision state visible even
    # when the probability curve is close to a threshold.
    for step, mode in zip(steps, mode_values):
        ax.add_patch(Rectangle(
            (step - 0.5, lane - 0.34), 1.0, 0.68,
            facecolor=MODE_COLORS[mode], edgecolor="white", linewidth=0.25,
            alpha=0.9,
        ))
    valid = [(step, value) for step, value in zip(steps, probabilities)
             if isinstance(value, (int, float))]
    if valid:
        xs = [item[0] for item in valid]
        ys = [lane - 0.30 + 0.60 * float(item[1]) for item in valid]
        ax.plot(xs, ys, color=INK, linewidth=0.8, drawstyle="steps-post")
        # The Router is evaluated every two environment steps. Mark only
        # those routing instants; the step line shows the held probability.
        evaluated = [(step, value) for step, value in valid if step % 2 == 1]
        ax.scatter(
            [item[0] for item in evaluated],
            [lane - 0.30 + 0.60 * float(item[1]) for item in evaluated],
            s=5.5, color=INK, edgecolor="white", linewidth=0.2, zorder=3,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8.0,
        "axes.unicode_minus": False,
        "svg.hashsalt": "piroute-supplement-timeline",
        "svg.fonttype": "none",
    })

    trajectories = load_trajectory(args.trajectory)
    selected = load_selected(args.selection)
    if not selected:
        raise SystemExit("selection is empty")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(selected), figsize=(2.39 * len(selected), 3.05),
                             squeeze=False, sharey=True)
    axes = axes[0]
    for col, (episode, metadata) in enumerate(selected):
        records = trajectories.get(int(episode), [])
        if not records:
            raise SystemExit(f"missing trajectory episode {episode}")
        ax = axes[col]
        robots = sorted(records[0]["positions"])
        for lane, robot in enumerate(robots):
            add_lane(ax, records, robot, lane)
        ax.set_yticks(range(len(robots)))
        ax.set_yticklabels(robots, fontsize=7, color=INK)
        ax.set_ylim(-0.7, len(robots) - 0.3)
        ax.invert_yaxis()
        ax.set_xlim(0.5, max(int(record["step"]) for record in records) + 0.5)
        ax.set_xlabel("environment step", fontsize=8, color=INK)
        title = PANEL_TITLES.get(metadata["label"], metadata["label"])
        ax.set_title(f"({chr(ord('a') + col)})  {title}", fontsize=8.0,
                     fontweight="bold", color=INK, pad=8)
        ax.grid(axis="x", color=GRID, linewidth=0.45)
        ax.tick_params(axis="x", labelsize=7, colors=INK)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#7F8C96")
        ax.spines["bottom"].set_color("#7F8C96")
        ax.spines["left"].set_linewidth(0.65)
        ax.spines["bottom"].set_linewidth(0.65)
    axes[0].set_ylabel("Robot", fontsize=8, color=INK)
    fig.suptitle("Per-robot routing dynamics in representative rollouts", fontsize=10.0,
                 fontweight="bold", color=INK)
    handles = [
        Rectangle((0, 0), 1, 1, facecolor=MODE_COLORS["standard"], label="standard Actor"),
        Rectangle((0, 0), 1, 1, facecolor=MODE_COLORS["dense"], label="interaction Actor"),
        Rectangle((0, 0), 1, 1, facecolor=MODE_COLORS["inactive"], label="inactive"),
        Line2D([0], [0], color=INK, linewidth=0.8, marker="o", markersize=2.4,
               label="Router probability (within-lane p=0-1)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=6.8, bbox_to_anchor=(0.5, 0.0), handletextpad=0.45,
               columnspacing=1.15)
    fig.tight_layout(rect=(0, 0.14, 1, 0.92))
    output = args.output_dir / "gate_timeline.svg"
    fig.savefig(output, dpi=600, bbox_inches="tight", facecolor="white",
                metadata={"Date": None})
    output.write_text(
        "\n".join(line.rstrip() for line in output.read_text(encoding="utf-8").splitlines()) + "\n",
        encoding="utf-8",
    )
    plt.close(fig)
    print("Wrote Gate timeline to", args.output_dir)


if __name__ == "__main__":
    main()
