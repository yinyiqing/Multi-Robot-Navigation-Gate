#!/usr/bin/env python3
"""Plot a data-grounded method figure from the frozen qualitative capture.

This is a qualitative visualization only. It reads the post-sealed capture and
does not recompute any G25/G26 statistic.
"""

import argparse
import gzip
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "paper/generated/captures/method_figure_capture"
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
INK = "#26323A"
MUTED = "#5B6872"
PURPLE = "#6E4F92"


def load_capture(root, episode):
    manifest_path = root / "trajectory_subset_64.json.gz"
    trajectory_path = root / "trajectories/trajectory_capture_b2_s20260910.jsonl"
    with gzip.open(manifest_path, "rt", encoding="utf-8") as handle:
        manifest = json.load(handle)
    scenario = manifest["scenarios"][episode - 1]
    rows = []
    with trajectory_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if int(row["episode"]) == episode:
                rows.append(row)
    if not rows:
        raise SystemExit("episode %d is absent from %s" % (episode, trajectory_path))
    return scenario, rows


def world_to_local(origin, yaw, point):
    delta = np.asarray(point, dtype=float) - np.asarray(origin, dtype=float)
    c, s = np.cos(yaw), np.sin(yaw)
    return np.asarray([c * delta[0] + s * delta[1], -s * delta[0] + c * delta[1]])


def contiguous_spans(values):
    spans = []
    start = 0
    for index in range(1, len(values) + 1):
        if index == len(values) or values[index] != values[start]:
            spans.append((start, index, values[start]))
            start = index
    return spans


def save(fig, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        path = output.with_suffix("." + ext)
        fig.savefig(path, dpi=600, bbox_inches="tight", facecolor="white")
        if ext == "svg":
            path.write_text(
                "\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()) + "\n",
                encoding="utf-8",
            )
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--episode", type=int, default=2)
    parser.add_argument("--robot", default="r1")
    parser.add_argument("--output", type=Path, default=DEFAULT_ROOT / "piroute_method_evidence")
    args = parser.parse_args()

    scenario, rows = load_capture(args.capture_root, args.episode)
    robot_names = list(rows[0]["positions"].keys())
    if args.robot not in robot_names:
        raise SystemExit("unknown robot %s" % args.robot)
    robot_index = robot_names.index(args.robot)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8.0,
        "axes.titlesize": 9.0,
        "axes.labelsize": 8.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "axes.linewidth": 0.75,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
    })

    fig = plt.figure(figsize=(7.16, 4.55), facecolor="white")
    grid = fig.add_gridspec(2, 2, height_ratios=[1.12, 0.88], width_ratios=[1.05, 0.95],
                            left=0.06, right=0.985, bottom=0.12, top=0.90,
                            wspace=0.28, hspace=0.34)
    ax_scene = fig.add_subplot(grid[:, 0])
    ax_local = fig.add_subplot(grid[0, 1])
    ax_gate = fig.add_subplot(grid[1, 1])

    # Panel (a): actual world-frame rollout from the JSONL capture.
    for index, name in enumerate(robot_names):
        path = np.asarray([row["positions"][name] for row in rows], dtype=float)
        color = COLORS[index % len(COLORS)]
        width = 2.5 if name == args.robot else 1.2
        alpha = 1.0 if name == args.robot else 0.72
        ax_scene.plot(path[:, 0], path[:, 1], color=color, linewidth=width, alpha=alpha,
                      solid_capstyle="round", zorder=2)
        start = scenario["agents"][name]["start"]
        goal = scenario["agents"][name]["goal"]
        ax_scene.scatter(*start, s=20, color="white", edgecolor=color, linewidth=1.0, zorder=4)
        ax_scene.scatter(*goal, marker="*", s=62, color=color, edgecolor="white", linewidth=.35, zorder=4)
    # The simulator uses axis-aligned cardboard boxes of 0.5 x 0.4 m
    # (verified in TD3.world); draw their actual footprint rather than points.
    for box in scenario.get("boxes", []):
        ax_scene.add_patch(Rectangle((box[0] - .25, box[1] - .20), .50, .40,
                                     facecolor="#DCE2E6", edgecolor="#8E9AA2",
                                     linewidth=.55, alpha=.8, zorder=1))
    modes = [((row.get("router_diagnostics") or {}).get(args.robot) or {}).get("mode", "standard") for row in rows]
    switch_indices = [i for i in range(1, len(modes)) if modes[i] != modes[i - 1]]
    selected_path = np.asarray([row["positions"][args.robot] for row in rows], dtype=float)
    if switch_indices:
        ax_scene.scatter(selected_path[switch_indices, 0], selected_path[switch_indices, 1],
                         marker="D", s=28, facecolor="white", edgecolor=PURPLE, linewidth=1.1, zorder=5)
    ax_scene.set_aspect("equal", adjustable="box")
    ax_scene.set_title("(a) World-frame rollout", loc="left", fontweight="bold", color=INK)
    ax_scene.set_xlabel("World x (m)")
    ax_scene.set_ylabel("World y (m)")
    ax_scene.grid(color="#D9E0E5", linewidth=.5, alpha=.65)
    ax_scene.spines["top"].set_visible(False); ax_scene.spines["right"].set_visible(False)
    handles = [Line2D([0], [0], color=COLORS[i], lw=2, label=name) for i, name in enumerate(robot_names)]
    handles.append(Line2D([0], [0], marker="D", color="white", markeredgecolor=PURPLE,
                          lw=0, label="Router switch", markersize=5))
    ax_scene.legend(handles=handles, loc="lower left", fontsize=6.5, frameon=False, ncol=2,
                    handlelength=1.5, columnspacing=0.8)

    # Select the frame with the largest measured candidate-action disagreement.
    def action_diff(row):
        diag = (row.get("router_diagnostics") or {}).get(args.robot)
        if not diag or diag.get("standard_action") is None or diag.get("dense_action") is None:
            return -1.0
        a = np.asarray(diag["standard_action"], dtype=float)
        b = np.asarray(diag["dense_action"], dtype=float)
        return float(np.linalg.norm(a - b))
    frame_index = int(np.argmax([action_diff(row) for row in rows]))
    frame = rows[frame_index]
    diag = frame["router_diagnostics"][args.robot]
    # Mark the exact world-frame state used for panel (b).
    ax_scene.scatter([selected_path[frame_index, 0]], [selected_path[frame_index, 1]],
                     marker="o", s=42, facecolor=PURPLE, edgecolor="white", linewidth=.8, zorder=6)
    lidar = np.asarray(frame["raw_lidar_points"][args.robot], dtype=float)
    if lidar.ndim == 2 and lidar.shape[1] >= 2:
        ax_local.scatter(lidar[:, 0], lidar[:, 1], s=1.2, color="#7D8A93", alpha=.28, linewidths=0)
    centers = np.asarray(diag.get("candidate_centers") or [], dtype=float)
    probs = np.asarray(diag.get("candidate_detector_probabilities") or [], dtype=float)
    if len(centers):
        sizes = 18 + 42 * probs
        ax_local.scatter(centers[:, 0], centers[:, 1], s=sizes, facecolor="none",
                         edgecolor=COLORS[1], linewidth=.9, alpha=.9)
    origin = np.zeros(2)
    ax_local.scatter([0], [0], s=44, marker="o", facecolor="white", edgecolor=COLORS[0], linewidth=1.5, zorder=4)
    ax_local.arrow(0, 0, .28, 0, width=.004, head_width=.045, head_length=.06,
                   color=COLORS[0], length_includes_head=True, zorder=4)
    ax_local.text(.30, .08, "robot i", fontsize=6.5, color=COLORS[0], va="bottom")
    for name in robot_names:
        if name == args.robot:
            continue
        pose = frame["actor_poses"][name]
        local = world_to_local(frame["positions"][args.robot], frame["actor_poses"][args.robot]["yaw"],
                               frame["positions"][name])
        ax_local.scatter(local[0], local[1], s=32, marker="o", facecolor="white",
                         edgecolor=COLORS[1], linewidth=1.0, zorder=4)
    ax_local.set_aspect("equal", adjustable="box")
    ax_local.set_xlim(-4.8, 4.8); ax_local.set_ylim(-4.8, 4.8)
    ax_local.set_title("(b) Local LiDAR and candidate actions", loc="left", fontweight="bold", color=INK)
    ax_local.set_xlabel("Local x (m)"); ax_local.set_ylabel("Local y (m)")
    ax_local.grid(color="#E1E5E8", linewidth=.45, alpha=.7)
    ax_local.spines["top"].set_visible(False); ax_local.spines["right"].set_visible(False)
    ax_local.text(.02, .95, "step %d | p=%.2f | %s" % (frame["step"], diag["gate_probability"], diag["mode"]),
                  transform=ax_local.transAxes, fontsize=6.8, color=MUTED, va="top")
    # Action-command inset: values are the actual policy outputs, not invented paths.
    inset = ax_local.inset_axes([.66, .07, .30, .28])
    a_n = np.asarray(diag["standard_action"], dtype=float)
    a_i = np.asarray(diag["dense_action"], dtype=float)
    inset.axhline(0, color="#AAB4BB", linewidth=.5); inset.axvline(0, color="#AAB4BB", linewidth=.5)
    inset.arrow(0, 0, a_n[0], a_n[1], color=COLORS[2], width=.012, head_width=.10, length_includes_head=True)
    inset.arrow(0, 0, a_i[0], a_i[1], color=COLORS[1], width=.012, head_width=.10, length_includes_head=True)
    inset.set_xlim(-1.05, 1.05); inset.set_ylim(-1.05, 1.05)
    inset.set_xticks([]); inset.set_yticks([]); inset.set_title("a_N / a_I", fontsize=6.5, color=MUTED, pad=1)
    inset.spines["top"].set_visible(False); inset.spines["right"].set_visible(False)
    inset.text(.02, .04, "normalized actor command", transform=inset.transAxes,
               fontsize=5.5, color=MUTED, va="bottom")

    # Panel (c): measured temporal Gate output and actual selected mode.
    steps = np.asarray([row["step"] for row in rows], dtype=float)
    for index, name in enumerate(robot_names):
        values = np.asarray([
            ((row.get("router_diagnostics") or {}).get(name) or {}).get("gate_probability", np.nan)
            for row in rows
        ], dtype=float)
        if name == args.robot:
            ax_gate.plot(steps, values, color=PURPLE, linewidth=1.9, marker="o", markersize=2.4, label=name)
        else:
            ax_gate.plot(steps, values, color="#AAB4BB", linewidth=.65, alpha=.55)
    for start, end, mode in contiguous_spans(modes):
        if mode == "dense":
            ax_gate.axvspan(steps[start], steps[min(end - 1, len(steps) - 1)], color="#F6E5D6", alpha=.65, linewidth=0)
    ax_gate.axhline(.43, color=COLORS[1], linewidth=.8, linestyle=(0, (3, 2)))
    ax_gate.axhline(.33, color=COLORS[2], linewidth=.8, linestyle=(0, (3, 2)))
    ax_gate.text(.99, .43, "on .43", transform=ax_gate.get_yaxis_transform(), ha="right", va="bottom", fontsize=6.2, color=COLORS[1])
    ax_gate.text(.99, .33, "off .33", transform=ax_gate.get_yaxis_transform(), ha="right", va="top", fontsize=6.2, color=COLORS[2])
    ax_gate.set_ylim(-.03, 1.03); ax_gate.set_xlim(steps.min(), steps.max())
    ax_gate.set_xlabel("Environment step"); ax_gate.set_ylabel("Gate probability")
    ax_gate.set_title("(c) Gate probability and selected mode", loc="left", fontweight="bold", color=INK)
    ax_gate.grid(axis="y", color="#D9E0E5", linewidth=.5)
    ax_gate.spines["top"].set_visible(False); ax_gate.spines["right"].set_visible(False)
    ax_gate.legend(frameon=False, fontsize=6.7, loc="upper right")

    fig.suptitle("Measured PIRoute behavior in one dense episode",
                 fontsize=10.8, fontweight="bold", color=INK, y=.965)
    fig.text(.06, .035,
             "One post-sealed qualitative episode; points and actions are recorded values, not aggregate test statistics. "
             "Shading marks the selected interaction Actor.", fontsize=6.8, color=MUTED)
    save(fig, args.output)
    print("Wrote", args.output.with_suffix(".png"))


if __name__ == "__main__":
    main()
