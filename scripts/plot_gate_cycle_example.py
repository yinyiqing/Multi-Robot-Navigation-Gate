#!/usr/bin/env python3
"""Plot a publication-scale real Router cycle from qualitative rollout data."""

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


STANDARD = "#0072B2"
INTERACTION = "#D55E00"
NEUTRAL = "#777777"
LIGHT = "#C7C7C7"


def load_episodes(path):
    episodes = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                episodes[int(record["episode"])].append(record)
    for records in episodes.values():
        records.sort(key=lambda item: int(item["step"]))
    return dict(episodes)


def mode_runs(steps, modes):
    runs = []
    for step, mode in zip(steps, modes):
        if not runs or runs[-1]["mode"] != mode:
            runs.append({"mode": mode, "start": step, "end": step})
        else:
            runs[-1]["end"] = step
    return runs


def set_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.6,
            "lines.linewidth": 1.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )


def shade_modes(ax, runs):
    for run in runs:
        color = STANDARD if run["mode"] == "standard" else INTERACTION
        ax.axvspan(run["start"] - 0.5, run["end"] + 0.5, color=color, alpha=0.10, lw=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-index", type=int, default=0)
    args = parser.parse_args()

    analysis = json.loads(args.analysis.read_text(encoding="utf-8"))
    candidates = analysis["ranked_cycle_candidates"]
    if not 0 <= args.candidate_index < len(candidates):
        raise SystemExit("candidate index is outside ranked_cycle_candidates")
    candidate = candidates[args.candidate_index]
    episode = int(candidate["episode"])
    robot = str(candidate["robot"])
    records = load_episodes(args.trajectory).get(episode)
    if not records:
        raise SystemExit(f"trajectory episode {episode} is missing")

    with gzip.open(args.manifest, "rt", encoding="utf-8") as handle:
        manifest = json.load(handle)
    scenario = manifest["scenarios"][episode - 1]
    if str(scenario["scenario_id"]) != str(candidate["scenario_id"]):
        raise SystemExit("analysis candidate does not match manifest order")

    robots = sorted(records[0]["positions"])
    robot_index = robots.index(robot)
    active_records = [
        record
        for record in records
        if robot in (record.get("actor_modes") or {})
        and (record.get("active_before") or [False] * len(robots))[robot_index]
    ]
    steps = np.asarray([int(record["step"]) for record in active_records])
    modes = [(record.get("actor_modes") or {})[robot] for record in active_records]
    probabilities = np.asarray(
        [float((record.get("gate_probabilities") or {})[robot]) for record in active_records]
    )
    nearest = []
    for record in active_records:
        value = record["agents"][robot]["nearest_robot_distance"]
        nearest.append(float(value) if value is not None and np.isfinite(value) else np.nan)
    nearest = np.asarray(nearest)
    runs = mode_runs(steps, modes)

    set_style()
    fig = plt.figure(figsize=(7.16, 3.55))
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=(1.05, 1.35),
        height_ratios=(1, 1),
        left=0.075,
        right=0.985,
        bottom=0.19,
        top=0.96,
        wspace=0.28,
        hspace=0.18,
    )
    ax_path = fig.add_subplot(grid[:, 0])
    ax_probability = fig.add_subplot(grid[0, 1])
    ax_distance = fig.add_subplot(grid[1, 1], sharex=ax_probability)

    all_xy = []
    for other in robots:
        start = np.asarray(scenario["agents"][other]["start"], dtype=float)
        goal = np.asarray(scenario["agents"][other]["goal"], dtype=float)
        path = np.asarray([record["positions"][other] for record in records], dtype=float)
        all_xy.extend((start, goal, *path))
        if other != robot:
            ax_path.plot(path[:, 0], path[:, 1], color=LIGHT, lw=0.9, zorder=1)
            ax_path.plot(start[0], start[1], "o", ms=3.0, mfc="white", mec=NEUTRAL, mew=0.7, zorder=2)
            ax_path.plot(goal[0], goal[1], marker="*", ms=5.0, mfc=LIGHT, mec=NEUTRAL, mew=0.5, zorder=2)

    focal_start = np.asarray(scenario["agents"][robot]["start"], dtype=float)
    focal_goal = np.asarray(scenario["agents"][robot]["goal"], dtype=float)
    focal_path = np.asarray([record["positions"][robot] for record in active_records], dtype=float)
    segment_start = focal_start
    for position, mode in zip(focal_path, modes):
        color = STANDARD if mode == "standard" else INTERACTION
        linestyle = "-" if mode == "standard" else (0, (3.0, 1.4))
        ax_path.plot(
            [segment_start[0], position[0]],
            [segment_start[1], position[1]],
            color=color,
            ls=linestyle,
            lw=2.2,
            solid_capstyle="round",
            zorder=4,
        )
        segment_start = position
    ax_path.plot(focal_start[0], focal_start[1], "o", ms=5.0, mfc="white", mec="#111111", mew=0.9, zorder=5)
    ax_path.plot(focal_goal[0], focal_goal[1], marker="*", ms=8.0, mfc="#F0E442", mec="#111111", mew=0.7, zorder=5)
    ax_path.plot(
        [focal_start[0], focal_goal[0]],
        [focal_start[1], focal_goal[1]],
        color="#999999",
        ls=(0, (1.2, 2.0)),
        lw=0.7,
        zorder=0,
    )
    xy = np.asarray(all_xy)
    span = max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1]))
    margin = max(0.25, 0.08 * span)
    ax_path.set_xlim(xy[:, 0].min() - margin, xy[:, 0].max() + margin)
    ax_path.set_ylim(xy[:, 1].min() - margin, xy[:, 1].max() + margin)
    ax_path.set_aspect("equal", adjustable="box")
    ax_path.set_xlabel("x position (m)")
    ax_path.set_ylabel("y position (m)")
    ax_path.grid(color="#E6E6E6", lw=0.45)

    shade_modes(ax_probability, runs)
    ax_probability.step(steps, probabilities, where="post", color="#222222", lw=1.25)
    evaluated = np.r_[True, probabilities[1:] != probabilities[:-1]]
    ax_probability.scatter(
        steps[evaluated], probabilities[evaluated], s=13, facecolor="white", edgecolor="#222222", lw=0.6, zorder=3
    )
    ax_probability.axhline(0.43, color=INTERACTION, ls=(0, (4, 2)), lw=0.9)
    ax_probability.axhline(0.33, color=STANDARD, ls=(0, (1.5, 2)), lw=0.9)
    ax_probability.text(steps[-1] - 0.25, 0.43, "enter 0.43", color=INTERACTION, va="center", ha="right", fontsize=7)
    ax_probability.text(steps[-1] - 0.25, 0.33, "exit 0.33", color=STANDARD, va="center", ha="right", fontsize=7)
    ax_probability.set_ylim(0, 1.0)
    ax_probability.set_ylabel("Gate probability")
    ax_probability.tick_params(axis="x", labelbottom=False)
    ax_probability.grid(axis="y", color="#E6E6E6", lw=0.45)

    shade_modes(ax_distance, runs)
    ax_distance.plot(steps, nearest, color="#222222", lw=1.25, marker="o", ms=2.3, markevery=2)
    missing_neighbor = ~np.isfinite(nearest)
    if np.any(missing_neighbor):
        marker_height = 3.02
        missing_steps = steps[missing_neighbor]
        ax_distance.hlines(
            marker_height,
            float(missing_steps[0]),
            float(missing_steps[-1]),
            color=NEUTRAL,
            ls=(0, (1.5, 1.8)),
            lw=0.9,
            zorder=3,
        )
        ax_distance.text(
            float(np.mean(missing_steps)),
            marker_height,
            "no active neighbor",
            color=NEUTRAL,
            fontsize=7,
            ha="center",
            va="bottom",
        )
    ax_distance.set_xlabel("environment step")
    ax_distance.set_ylabel("Nearest robot distance (m)")
    ax_distance.grid(axis="y", color="#E6E6E6", lw=0.45)
    ax_distance.set_xlim(steps[0] - 0.5, steps[-1] + 2.8)
    ax_distance.set_ylim(0, 3.2)

    for label, ax in zip(("(a)", "(b)", "(c)"), (ax_path, ax_probability, ax_distance)):
        ax.text(0.0, 1.03, label, transform=ax.transAxes, fontweight="bold", fontsize=9, va="bottom")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    legend = [
        Line2D([0], [0], color=STANDARD, lw=2.0, ls="-", label="standard Actor"),
        Line2D([0], [0], color=INTERACTION, lw=2.0, ls=(0, (3, 1.4)), label="interaction Actor"),
        Line2D([0], [0], color=LIGHT, lw=1.0, label="other robots"),
        Line2D([0], [0], marker="o", color="none", mfc="white", mec="#111111", ms=4.5, label="start"),
        Line2D([0], [0], marker="*", color="none", mfc="#F0E442", mec="#111111", ms=7, label="goal"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.025), handlelength=2.2)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    basename = args.output_dir / "gate_cycle_example"
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(f"{basename}.{suffix}", dpi=600, facecolor="white")
    selection = {
        "candidate_index": args.candidate_index,
        "candidate": candidate,
        "manifest": str(args.manifest),
        "trajectory": str(args.trajectory),
        "analysis": str(args.analysis),
        "figure_size_inches": [7.16, 3.55],
        "nearest_robot_distance_is_analysis_only": True,
    }
    (args.output_dir / "gate_cycle_example_selection.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "gate_cycle_example_caption.txt").write_text(
        "Real PIRoute cycle in a full-success qualitative rollout. The focal robot executes the "
        "standard Actor, enters the interaction Actor after the Gate probability crosses the 0.43 "
        "entry threshold, and returns to the standard Actor after crossing the 0.33 exit threshold. "
        "Nearest-robot distance is reconstructed from simulator truth for post-hoc analysis only and "
        "is not available to PIRoute at deployment.\n",
        encoding="utf-8",
    )
    plt.close(fig)
    print(f"Wrote {basename}.{{png,pdf,svg}}")


if __name__ == "__main__":
    main()
