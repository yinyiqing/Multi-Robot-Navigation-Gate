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


STANDARD = "#56A1B8"
INTERACTION = "#C47FA0"
STANDARD_FILL = "#A3DCEC"
INTERACTION_FILL = "#ECAFC6"
GOAL = "#56A1B8"
INK = "#4E5D66"
NEUTRAL = "#7F8C96"
LIGHT = "#C6D5DB"
GRID = "#D9E0E5"


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


def normalize_mode(mode):
    """Map rollout log names to the paper's two policy roles."""
    return "standard" if str(mode).lower() == "standard" else "interaction"


def set_style():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.3,
            "axes.titleweight": "bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "piroute-gate-cycle",
            "savefig.facecolor": "white",
        }
    )


def shade_modes(ax, runs):
    for run in runs:
        color = STANDARD_FILL if run["mode"] == "standard" else INTERACTION_FILL
        ax.axvspan(run["start"] - 0.5, run["end"] + 0.5, color=color, alpha=0.42, lw=0, zorder=0)


def label_modes(ax, runs):
    """Place compact phase labels inside the shaded time windows."""
    for index, run in enumerate(runs):
        if index == 0:
            center, alignment = run["start"] + 0.18, "left"
        elif index == len(runs) - 1:
            center, alignment = run["end"] - 0.18, "right"
        else:
            center, alignment = 0.5 * (run["start"] + run["end"]), "center"
        color = STANDARD if run["mode"] == "standard" else INTERACTION
        ax.text(
            center,
            0.94,
            run["mode"],
            transform=ax.get_xaxis_transform(),
            ha=alignment,
            va="top",
            fontsize=6.6,
            color=color,
            fontweight="bold",
            clip_on=True,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-index", type=int, default=1)
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
    modes = [normalize_mode((record.get("actor_modes") or {})[robot]) for record in active_records]
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
    fig = plt.figure(figsize=(7.16, 3.35))
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=(1.02, 1.48),
        height_ratios=(1, 1),
        left=0.075,
        right=0.985,
        bottom=0.19,
        top=0.92,
        wspace=0.24,
        hspace=0.20,
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
            ax_path.plot(path[:, 0], path[:, 1], color=LIGHT, lw=0.8, zorder=1)
            ax_path.plot(start[0], start[1], "o", ms=2.8, mfc="white", mec=NEUTRAL, mew=0.65, zorder=2)
            ax_path.plot(goal[0], goal[1], marker="*", ms=4.8, mfc=LIGHT, mec=NEUTRAL, mew=0.45, zorder=2)

    focal_start = np.asarray(scenario["agents"][robot]["start"], dtype=float)
    focal_goal = np.asarray(scenario["agents"][robot]["goal"], dtype=float)
    focal_path = np.asarray([record["positions"][robot] for record in active_records], dtype=float)
    segment_start = focal_start
    for position, mode in zip(focal_path, modes):
        color = STANDARD if mode == "standard" else INTERACTION
        linestyle = "-" if mode == "standard" else (0, (4.0, 1.8))
        ax_path.plot(
            [segment_start[0], position[0]],
            [segment_start[1], position[1]],
            color=color,
            ls=linestyle,
            lw=1.65,
            solid_capstyle="round",
            zorder=4,
        )
        segment_start = position
    ax_path.plot(focal_start[0], focal_start[1], "o", ms=5.0, mfc="white", mec="#111111", mew=0.9, zorder=5)
    ax_path.plot(focal_goal[0], focal_goal[1], marker="*", ms=8.0, mfc=GOAL, mec=INK, mew=0.7, zorder=5)
    ax_path.annotate("start", focal_start, xytext=(5, 3), textcoords="offset points", fontsize=6.5, color=INK)
    ax_path.annotate("goal", focal_goal, xytext=(5, 3), textcoords="offset points", fontsize=6.5, color=INK)
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
    ax_path.grid(color=GRID, lw=0.45)

    shade_modes(ax_probability, runs)
    label_modes(ax_probability, runs)
    ax_probability.step(steps, probabilities, where="post", color=INK, lw=0.82, zorder=2)
    evaluated = np.r_[True, probabilities[1:] != probabilities[:-1]]
    ax_probability.scatter(
        steps[evaluated], probabilities[evaluated], s=11, facecolor="white", edgecolor=INK, lw=0.5, zorder=3
    )
    ax_probability.axhline(0.43, color=INTERACTION, ls=(0, (4, 2)), lw=0.9)
    ax_probability.axhline(0.33, color=STANDARD, ls=(0, (1.5, 2)), lw=0.9)
    threshold_x = steps[0] + 0.2
    ax_probability.text(threshold_x, 0.43, "enter", color=INTERACTION, va="bottom", ha="left", fontsize=6.7, fontweight="bold")
    ax_probability.text(threshold_x, 0.33, "exit", color=STANDARD, va="top", ha="left", fontsize=6.7, fontweight="bold")
    ax_probability.set_ylim(0, 1.0)
    ax_probability.set_ylabel("Router probability")
    ax_probability.tick_params(axis="x", labelbottom=False)
    ax_probability.grid(axis="y", color=GRID, lw=0.45)

    shade_modes(ax_distance, runs)
    ax_distance.plot(steps, nearest, color=INK, lw=0.82, marker="o", ms=1.8, markevery=2, zorder=2)
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
            lw=0.7,
            zorder=3,
        )
        ax_distance.text(
            float(np.mean(missing_steps)),
            marker_height,
            "no active neighbor",
            color=NEUTRAL,
            fontsize=6.5,
            ha="center",
            va="bottom",
        )
    ax_distance.set_xlabel("environment step")
    ax_distance.set_ylabel("Nearest robot distance (m)")
    ax_distance.axhline(2.0, color=NEUTRAL, ls=(0, (2.0, 2.0)), lw=0.65, zorder=1)
    ax_distance.text(steps[-1] - 0.25, 2.0, "2 m reference", color=NEUTRAL, va="bottom", ha="right", fontsize=6.4)
    ax_distance.grid(axis="y", color=GRID, lw=0.45)
    ax_distance.set_xlim(steps[0] - 0.5, steps[-1] + 2.8)
    ax_distance.set_ylim(0, 3.2)

    panel_titles = (
        "(a) Focal robot trajectory",
        "(b) Router probability: standard -> interaction -> standard",
        "(c) Nearest-robot distance (post-hoc)",
    )
    for title, ax in zip(panel_titles, (ax_path, ax_probability, ax_distance)):
        ax.text(0.5, 1.045, title, transform=ax.transAxes, fontweight="bold", fontsize=8.3, va="bottom", ha="center")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    legend = [
        Line2D([0], [0], color=STANDARD, lw=1.55, ls="-", label="standard Actor"),
        Line2D([0], [0], color=INTERACTION, lw=1.55, ls=(0, (4, 1.8)), label="interaction Actor"),
        Line2D([0], [0], color=LIGHT, lw=0.9, label="other robots"),
    ]
    fig.legend(
        handles=legend,
        loc="lower left",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.075, 0.018),
        handlelength=2.0,
        columnspacing=1.15,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    basename = args.output_dir / "gate_cycle_example"
    fig.savefig(f"{basename}.svg", facecolor="white", metadata={"Date": None})
    plt.close(fig)
    print(f"Wrote {basename}.svg")


if __name__ == "__main__":
    main()
