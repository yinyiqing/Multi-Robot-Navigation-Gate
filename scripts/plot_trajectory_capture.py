#!/usr/bin/env python3
"""Plot paired qualitative trajectories from the overnight capture queue."""

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


INK = "#4E5D66"
GRID = "#D9E0E5"
ROBOT_COLORS = ["#56A1B8", "#8FC3D0", "#7F8C96", "#AAB9BF", "#3E7183"]


def load_manifest(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {str(item["scenario_id"]): item for item in payload["scenarios"]}


def load_trajectory(path):
    episodes = defaultdict(list)
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                episodes[int(record["episode"])].append(record)
    return dict(episodes)


def load_results(path):
    rows = np.load(path, allow_pickle=True)
    if rows.ndim != 2 or rows.shape[1] != 17:
        raise ValueError(f"invalid result shape: {path}: {rows.shape}")
    return {str(row[12]): row for row in rows}


def full_success(row):
    return bool(int(row[8]))


def interaction_share(row):
    return float(row[14])


def choose_scenes(manifest, five_a, b2, limit):
    ids = list(manifest)
    rescue = [sid for sid in ids if full_success(b2[sid]) and not full_success(five_a[sid])]
    both_success = [sid for sid in ids if full_success(b2[sid]) and full_success(five_a[sid])]
    both_fail = [sid for sid in ids if not full_success(b2[sid]) and not full_success(five_a[sid])]
    five_a_only = [sid for sid in ids if full_success(five_a[sid]) and not full_success(b2[sid])]
    groups = [
        ("PIRoute succeeds, 5A fails", rescue),
        ("Both succeed", sorted(both_success, key=lambda sid: interaction_share(b2[sid]), reverse=True)),
        ("Both fail", both_fail),
        ("5A succeeds, PIRoute fails", five_a_only),
    ]
    chosen = []
    labels = []
    per_group = max(1, limit // len(groups))
    for label, candidates in groups:
        for sid in candidates[:per_group]:
            if sid not in chosen:
                chosen.append(sid)
                labels.append(label)
    for sid in ids:
        if len(chosen) >= limit:
            break
        if sid not in chosen:
            chosen.append(sid)
            labels.append("Additional scene")
    return list(zip(chosen[:limit], labels[:limit]))


def path_points(records, robot):
    if not records:
        return np.empty((0, 2))
    first = records[0]["actor_poses"][robot]
    points = [[float(first["x"]), float(first["y"])]]
    points.extend([record["positions"][robot] for record in records])
    return np.asarray(points, dtype=float)


def scene_limits(scenario, *record_sets):
    """Return shared x/y limits for both methods in one matched scene."""
    points = []
    for agent in scenario["agents"].values():
        points.extend([agent["start"], agent["goal"]])
    for records in record_sets:
        for record in records:
            points.extend(record.get("positions", {}).values())
    points = np.asarray(points, dtype=float)
    lower = points.min(axis=0) - 0.5
    upper = points.max(axis=0) + 0.5
    return (float(lower[0]), float(upper[0])), (float(lower[1]), float(upper[1]))


def plot_episode(ax, records, scenario, title, is_piroute, limits=None):
    robots = sorted(scenario["agents"])
    all_points = []
    for index, robot in enumerate(robots):
        points = path_points(records, robot)
        if len(points) == 0:
            continue
        all_points.append(points)
        color = ROBOT_COLORS[index % len(ROBOT_COLORS)]
        if is_piroute:
            modes = ["standard"] + [
                (record.get("actor_modes") or {}).get(robot, "standard")
                for record in records
            ]
            for start, end, mode in zip(points[:-1], points[1:], modes[:-1]):
                ax.plot(
                    [start[0], end[0]], [start[1], end[1]],
                    color=color, linewidth=1.5,
                    linestyle="--" if mode == "dense" else "-",
                )
        else:
            ax.plot(points[:, 0], points[:, 1], color=color, linewidth=1.5)
        agent = scenario["agents"][robot]
        ax.plot(agent["start"][0], agent["start"][1], marker="o", color=color, markersize=3)
        ax.plot(agent["goal"][0], agent["goal"][1], marker="x", color=color, markersize=5, mew=1.2)
        for record in records:
            state = record.get("agents", {}).get(robot, {})
            if state.get("collision"):
                pos = record["positions"][robot]
                ax.plot(pos[0], pos[1], marker="X", color="#B2182B", markersize=6, mew=0.8)
            if state.get("target"):
                pos = record["positions"][robot]
                ax.plot(pos[0], pos[1], marker="*", color=color, markersize=7, mew=0.5)
    for box in scenario.get("boxes", []):
        ax.plot(box[0], box[1], marker="s", color="#666666", markersize=4, alpha=0.7)
    if limits is None and all_points:
        points = np.concatenate(all_points, axis=0)
        limits = (
            (points[:, 0].min() - 0.5, points[:, 0].max() + 0.5),
            (points[:, 1].min() - 0.5, points[:, 1].max() + 0.5),
        )
    if limits is not None:
        ax.set_xlim(*limits[0])
        ax.set_ylim(*limits[1])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color=GRID, linewidth=0.45)
    ax.set_title(title, fontsize=7.5, color=INK)
    ax.tick_params(labelsize=7, colors=INK)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#7F8C96")
    ax.spines["bottom"].set_color("#7F8C96")
    ax.spines["left"].set_linewidth(0.65)
    ax.spines["bottom"].set_linewidth(0.65)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--five-a", type=Path, required=True)
    parser.add_argument("--b2", type=Path, required=True)
    parser.add_argument("--five-a-trajectory", type=Path, required=True)
    parser.add_argument("--b2-trajectory", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument(
        "--selection",
        type=Path,
        help="Optional frozen selection JSON; preserves registered scene ordering.",
    )
    args = parser.parse_args()

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8.0,
        "axes.unicode_minus": False,
        "svg.hashsalt": "piroute-supplement-trajectories",
    })

    manifest = load_manifest(args.manifest)
    five_a = load_results(args.five_a)
    b2 = load_results(args.b2)
    five_a_traj = load_trajectory(args.five_a_trajectory)
    b2_traj = load_trajectory(args.b2_trajectory)
    if args.selection:
        selection_payload = json.loads(args.selection.read_text(encoding="utf-8"))
        selected = [
            (item["scenario_id"], item["label"])
            for item in selection_payload["selected"]
        ]
        missing = [scenario_id for scenario_id, _ in selected if scenario_id not in manifest]
        if missing:
            raise SystemExit(f"selected scenes missing from manifest: {missing}")
    else:
        selected = choose_scenes(manifest, five_a, b2, args.limit)
    if not selected:
        raise SystemExit("no scenes available for plotting")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, len(selected), figsize=(2.6 * len(selected), 5.3), squeeze=False)
    for col, (scenario_id, label) in enumerate(selected):
        scenario = manifest[scenario_id]
        row5 = five_a[scenario_id]
        rowb = b2[scenario_id]
        episode_index = list(manifest).index(scenario_id) + 1
        records5 = five_a_traj.get(episode_index, [])
        recordsb = b2_traj.get(episode_index, [])
        limits = scene_limits(scenario, records5, recordsb)
        plot_episode(
            axes[0, col], records5, scenario,
            f"5A | {label}\n{scenario_id[-8:]}", False, limits,
        )
        plot_episode(
            axes[1, col], recordsb, scenario,
            f"PIRoute | {label}\n{scenario_id[-8:]}", True, limits,
        )
        axes[0, col].text(
            0.02, 0.98,
            f"full={int(row5[8])}, coll={int(row5[7])}",
            transform=axes[0, col].transAxes, va="top", fontsize=6.5,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 1.5},
        )
        axes[1, col].text(
            0.02, 0.98,
            f"full={int(rowb[8])}, I-share={interaction_share(rowb):.2f}",
            transform=axes[1, col].transAxes, va="top", fontsize=6.5,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 1.5},
        )
    axes[0, 0].set_ylabel("5A\n y (m)", fontsize=8, color=INK)
    axes[1, 0].set_ylabel("PIRoute\n y (m)", fontsize=8, color=INK)
    panel_labels = "abcdef"
    for index, ax in enumerate(axes.flat):
        ax.text(
            0.02, 1.02, f"({panel_labels[index]})", transform=ax.transAxes,
            va="bottom", ha="left", fontsize=8, fontweight="bold", color=INK,
        )
    for ax in axes[-1, :]:
        ax.set_xlabel("x (m)", fontsize=8)
    fig.suptitle("Qualitative multi-robot trajectories on matched scenes", fontsize=10.5,
                 fontweight="bold", color=INK)
    robot_handles = [
        Line2D([0], [0], color=color, marker="o", linewidth=1.5,
               markersize=4, label=f"r{i + 1}")
        for i, color in enumerate(ROBOT_COLORS)
    ]
    symbol_handles = [
        Line2D([0], [0], color=INK, marker="o", linestyle="None", markersize=4, label="start"),
        Line2D([0], [0], color=INK, marker="x", linestyle="None", markersize=5, label="goal"),
        Line2D([0], [0], color="#B2182B", marker="X", linestyle="None", markersize=5, label="collision"),
        Line2D([0], [0], color=INK, linestyle="--", linewidth=1.2, label="interaction Actor"),
    ]
    fig.legend(handles=robot_handles + symbol_handles, loc="lower center", ncol=9,
               frameon=False, fontsize=7, bbox_to_anchor=(0.5, 0.005))
    fig.tight_layout(rect=(0, 0.075, 1, 0.95))
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(args.output_dir / f"trajectory_overview.{suffix}", dpi=300, bbox_inches="tight")
    (args.output_dir / "trajectory_overview_selection.json").write_text(
        json.dumps(
            {
                "selected": [{"scenario_id": sid, "label": label} for sid, label in selected],
                "episode_indices": [list(manifest).index(sid) + 1 for sid, _ in selected],
                "source": "G25 sealed outcome strata; qualitative visualization only",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print("Wrote trajectory overview to", args.output_dir)


if __name__ == "__main__":
    main()
