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
    "standard": "#0072B2",
    "dense": "#D55E00",
    "inactive": "#D9D9D9",
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
        ax.plot(xs, ys, color="#222222", linewidth=0.8, marker=".", markersize=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    trajectories = load_trajectory(args.trajectory)
    selected = load_selected(args.selection)
    if not selected:
        raise SystemExit("selection is empty")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(selected), figsize=(2.45 * len(selected), 3.55),
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
        ax.set_yticklabels(robots, fontsize=7)
        ax.set_ylim(-0.7, len(robots) - 0.3)
        ax.invert_yaxis()
        ax.set_xlim(0.5, max(int(record["step"]) for record in records) + 0.5)
        ax.set_xlabel("environment step", fontsize=8)
        ax.set_title(f"{metadata['label']}\n{metadata['scenario_id'][-8:]}", fontsize=8)
        ax.grid(axis="x", linewidth=0.35, alpha=0.35)
        ax.tick_params(axis="x", labelsize=7)
        ax.text(0.02, 1.02, f"({chr(ord('a') + col)})", transform=ax.transAxes,
                fontsize=8, fontweight="bold", va="bottom")
    axes[0].set_ylabel("robot\n(line height: p=0 to 1)", fontsize=8)
    fig.suptitle("PIRoute Router decisions on matched qualitative rollouts", fontsize=10.5)
    handles = [
        Rectangle((0, 0), 1, 1, facecolor=MODE_COLORS["standard"], label="standard Actor"),
        Rectangle((0, 0), 1, 1, facecolor=MODE_COLORS["dense"], label="interaction Actor"),
        Rectangle((0, 0), 1, 1, facecolor=MODE_COLORS["inactive"], label="inactive"),
        Line2D([0], [0], color="#222222", linewidth=0.8, marker=".", label="Gate probability"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=7, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.12, 1, 0.93))
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(args.output_dir / f"gate_timeline.{suffix}", dpi=300, bbox_inches="tight")
    (args.output_dir / "gate_timeline_selection.json").write_text(
        json.dumps({"selection": str(args.selection), "trajectory": str(args.trajectory),
                    "episodes": [episode for episode, _ in selected]}, indent=2) + "\n",
        encoding="utf-8",
    )
    print("Wrote Gate timeline to", args.output_dir)


if __name__ == "__main__":
    main()
