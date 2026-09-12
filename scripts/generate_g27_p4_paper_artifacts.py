#!/usr/bin/env python3
"""Generate publication figures and a LaTeX table from the completed G27-P4 test.

This command is intentionally fail-closed.  It never reads partial ``.npy`` files
and never writes paper artifacts until the independent-test statistics have been
validated by ``analyze_g27_p4_test.py``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import numpy as np
except ModuleNotFoundError as exc:  # pragma: no cover - environment guidance
    raise SystemExit(
        "This generator requires the project environment; run "
        "`source ./env.python.sh` before invoking it."
    ) from exc


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/03_保留专门化/02_论文主线"
G27 = BASE / "27_反事实Reward增强Gate监督"
STATS = G27 / "local_data/test/p4_statistics.json"
PAPER = ROOT / "paper/icra2027_template/piroute_overleaf_fonts_fixed_20260908"
OUT = PAPER / "generated/g27_p4"

EXPECTED_METHODS = ("5a", "b2", "b3")
LABELS = {"5a": "5A", "b2": "B2\n(proximity)", "b3": "B3\n(reward-aware)"}
COLORS = {"5a": "#7F8C96", "b2": "#56A1B8", "b3": "#C47FA0"}
INK = "#26323A"
MUTED = "#5B6872"
GRID = "#D9E0E5"


def load_and_validate():
    if not STATS.is_file():
        raise SystemExit(
            "G27-P4 is incomplete: p4_statistics.json is not available; "
            "run the P4 worker and analyzer first."
        )
    data = json.loads(STATS.read_text(encoding="utf-8"))
    protocol = data.get("protocol", {})
    if protocol.get("experiment_id") != "G27-P4-independent-test":
        raise SystemExit("unexpected G27-P4 experiment id")
    if tuple(protocol.get("methods", ())) != EXPECTED_METHODS:
        raise SystemExit("unexpected G27-P4 method order")
    if protocol.get("scene_clusters") != 256 or protocol.get("total_episodes") != 2304:
        raise SystemExit("G27-P4 statistics do not describe the frozen 2304-episode test")
    if protocol.get("sealed_test_read") is not True or protocol.get("actor_or_router_updated") is not False:
        raise SystemExit("G27-P4 completion guard failed")
    for method in EXPECTED_METHODS:
        if method not in data.get("pooled_descriptive", {}):
            raise SystemExit("missing pooled result for %s" % method)
    for name in ("b3_minus_5a", "b3_minus_b2"):
        if name not in data.get("comparisons", {}):
            raise SystemExit("missing comparison %s" % name)
    return data


def style():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "g27-p4-paper",
            "axes.unicode_minus": False,
        }
    )


def finish_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#71808B")
    ax.spines["bottom"].set_color("#71808B")
    ax.grid(axis="y", color=GRID, linewidth=0.55, alpha=0.8)
    ax.set_axisbelow(True)


def save(fig, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "pdf"):
        path = OUT / (stem + "." + ext)
        fig.savefig(path, dpi=600, bbox_inches="tight", facecolor="white")
        if ext == "svg":
            path.write_text(
                "\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines())
                + "\n",
                encoding="utf-8",
            )
    plt.close(fig)


def outcome_figure(data):
    methods = EXPECTED_METHODS
    metrics = (("full_success", "Full-team success (%)"), ("robot_collision", "Robot collision (%)"))
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.65), sharey=False)
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.23, top=0.82, wspace=0.30)
    x = np.arange(len(methods))
    for ax, (key, ylabel) in zip(axes, metrics):
        values = [100.0 * float(data["pooled_descriptive"][m][key]) for m in methods]
        bars = ax.bar(
            x,
            values,
            width=0.62,
            color=[COLORS[m] for m in methods],
            edgecolor=INK,
            linewidth=0.55,
        )
        ax.set_xticks(x, [LABELS[m] for m in methods])
        ax.set_ylim(0, 100)
        ax.set_yticks(np.arange(0, 101, 20))
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", length=0, pad=3)
        finish_axes(ax)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 2.4,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color=INK,
            )
    axes[0].set_title("(a) Team completion", loc="left", fontweight="bold", color=INK)
    axes[1].set_title("(b) Individual safety", loc="left", fontweight="bold", color=INK)
    fig.suptitle("G27-P4 independent test", fontsize=10.5, fontweight="bold", color=INK, y=0.98)
    save(fig, "g27_p4_outcomes")


def extract_effect(data, comparison, metric):
    item = data["comparisons"][comparison][metric]
    return float(item["mean_difference"]), tuple(float(v) for v in item["scene_cluster_bca_95_ci"])


def forest_figure(data):
    comparisons = (("B3 - 5A", "b3_minus_5a"), ("B3 - B2", "b3_minus_b2"))
    metrics = (
        ("Full success difference", "full_success", "Percentage points", 0.20),
        ("Collision difference", "robot_collision", "Percentage points", 0.20),
        ("Paired-success steps", "paired_success_steps", "Steps", 24.0),
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.78), sharey=False)
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.24, top=0.80, wspace=0.38)
    y = np.arange(len(comparisons))
    colors = (COLORS["b3"], COLORS["b2"])
    for ax, (title, metric, xlabel, half_range) in zip(axes, metrics):
        for row, ((label, comparison), color) in enumerate(zip(comparisons, colors)):
            mean, interval = extract_effect(data, comparison, metric)
            if metric != "paired_success_steps":
                mean *= 100.0
                interval = (interval[0] * 100.0, interval[1] * 100.0)
            ax.plot(interval, (row, row), color=color, linewidth=3.0, solid_capstyle="round")
            ax.scatter([mean], [row], s=30, color=color, edgecolor=INK, linewidth=0.45, zorder=3)
            ax.text(
                half_range * 0.98,
                row + 0.22,
                f"{mean:+.2f}",
                ha="right",
                va="bottom",
                fontsize=7.2,
                color=INK,
            )
        ax.axvline(0.0, color="#4D5965", linewidth=0.8, linestyle=(0, (3, 3)))
        ax.set_xlim(-half_range, half_range)
        ax.set_ylim(-0.55, 1.55)
        ax.set_yticks(y, [label for label, _ in comparisons])
        ax.set_xlabel(xlabel)
        ax.set_title(title, loc="left", fontweight="bold", color=INK)
        ax.grid(axis="x", color=GRID, linewidth=0.55, alpha=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#71808B")
        ax.spines["bottom"].set_color("#71808B")
        ax.tick_params(axis="y", length=0, pad=3)
    fig.suptitle("Reward-aware router effects in G27-P4", fontsize=10.5, fontweight="bold", color=INK, y=0.98)
    save(fig, "g27_p4_effects")


def table_tex(data):
    rows = []
    for method in EXPECTED_METHODS:
        item = data["pooled_descriptive"][method]
        label = {"5a": "5A", "b2": "Proximity B2", "b3": "Reward-aware B3"}[method]
        rows.append(
            "%s & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f \\\\"
            % (
                label,
                100.0 * float(item["full_success"]),
                100.0 * float(item["agent_success"]),
                100.0 * float(item["robot_collision"]),
                100.0 * float(item["robot_unresolved"]),
                100.0 * float(item["episode_timeout"]),
                float(item["raw_steps"]),
            )
        )
    content = [
        "% Generated from the audited G27-P4 independent test.",
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Method & Full & Agent & Coll. & Unres. & Timeout & Steps\\\\",
        "\\midrule",
        *rows,
        "\\bottomrule",
        "\\end{tabular}",
        "",
    ]
    return "\n".join(content)


def main():
    data = load_and_validate()
    style()
    outcome_figure(data)
    forest_figure(data)
    (OUT / "g27_p4_table.tex").write_text(table_tex(data), encoding="utf-8")
    print("Generated G27-P4 figures and table under %s" % OUT)


if __name__ == "__main__":
    main()
