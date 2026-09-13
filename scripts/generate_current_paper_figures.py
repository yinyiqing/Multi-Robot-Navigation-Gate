#!/usr/bin/env python3
"""Generate the quantitative figures for the current reward-aware PIRoute paper.

The script reads the frozen G25 matched controls and the G34 reward-aware
results. It does not read historical Router results or modify experiment data.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/03_保留专门化/02_论文主线"
SEALED = BASE / "25_最终消融与Sealed评测/local_data/sealed/sealed_statistics.json"
MATCHED = BASE / "36_G34_G25slice_matched/local_data/matched/statistics.json"
MATCHED_RESULTS = BASE / "36_G34_G25slice_matched/local_data/matched/results"
CONTROL_RESULTS = BASE / "25_最终消融与Sealed评测/local_data/sealed/results"
OUTPUT = ROOT / "paper/icra2027_template/figures"
SEEDS = (20260901, 20260902, 20260903)

INK = "#394A54"
MUTED = "#687983"
GRID = "#DDE6EA"
BLUE = "#56A1B8"
BLUE_MID = "#7DB5C4"
BLUE_LIGHT = "#A9CCD5"
BLUE_PALE = "#C7DCE2"
GRAY_BLUE = "#AEBCC3"
PINK = "#C47FA0"
PINK_DARK = "#985D79"

METHODS = (
    ("navigation", "Navigation Actor", BLUE),
    ("interaction", "Interaction Actor", BLUE_MID),
    ("min_lidar", "Min-LiDAR selector", BLUE_LIGHT),
    ("ttc_cpa", "TTC/CPA selector", BLUE_PALE),
    ("parameter_matched", "Parameter-matched Actor", GRAY_BLUE),
    ("piroute", "PIRoute", PINK),
)

# Audited scene-cluster BCa intervals reported in PROJECT_STATUS.md and main.tex.
def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "TeX Gyre Termes", "Liberation Serif"],
            "font.size": 8.0,
            "axes.labelsize": 8.0,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.4,
            "axes.linewidth": 0.65,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "piroute-current-paper-figures",
            "axes.unicode_minus": False,
            "savefig.facecolor": "white",
        }
    )


def load_metrics() -> dict[str, dict[str, float]]:
    sealed = load_json(SEALED)
    matched = load_json(MATCHED)
    if sealed["protocol"]["manifest_sha256"] != matched["manifest_sha256"]:
        raise ValueError("matched controls and PIRoute do not use the same scene manifest")

    old = sealed["pooled_descriptive"]
    current = matched["overall"]["g34"]
    metrics = {
        "navigation": {
            "full": 100 * old["5a"]["full_success"],
            "collision": 100 * old["5a"]["collision"],
            "timeout": 100 * old["5a"]["timeout"],
        },
        "interaction": {
            "full": 100 * old["epoch16"]["full_success"],
            "collision": 100 * old["epoch16"]["collision"],
            "timeout": 100 * old["epoch16"]["timeout"],
        },
        "min_lidar": {
            "full": 100 * old["min_lidar"]["full_success"],
            "collision": 100 * old["min_lidar"]["collision"],
            "timeout": 100 * old["min_lidar"]["timeout"],
        },
        "ttc_cpa": {
            "full": 100 * old["ttc_cpa"]["full_success"],
            "collision": 100 * old["ttc_cpa"]["collision"],
            "timeout": 100 * old["ttc_cpa"]["timeout"],
        },
        "parameter_matched": {
            "full": 100 * old["r2b"]["full_success"],
            "collision": 100 * old["r2b"]["collision"],
            "timeout": 100 * old["r2b"]["timeout"],
        },
        "piroute": {
            "full": 100 * current["full_success"],
            "collision": 100 * current["robot_collision"],
            "timeout": 100 * current["episode_timeout"],
        },
    }
    return metrics


def clean_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.spines["bottom"].set_linewidth(0.65)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", colors=MUTED, width=0.55, length=2.5)
    ax.grid(axis="x", color=GRID, linewidth=0.55)
    ax.set_axisbelow(True)


def paired_success_step_means() -> tuple[float, float, int]:
    navigation = np.stack(
        [
            np.load(CONTROL_RESULTS / f"g25_sealed_5a_s{seed}.npy", allow_pickle=True)
            for seed in SEEDS
        ],
        axis=1,
    )
    piroute = np.stack(
        [
            np.load(MATCHED_RESULTS / f"g34_g25slice_s{seed}.npy", allow_pickle=True)
            for seed in SEEDS
        ],
        axis=1,
    )
    joint_success = (navigation[:, :, 8].astype(float) == 1) & (
        piroute[:, :, 8].astype(float) == 1
    )
    return (
        float(navigation[:, :, 3].astype(float)[joint_success].mean()),
        float(piroute[:, :, 3].astype(float)[joint_success].mean()),
        int(joint_success.sum()),
    )


def save(fig: plt.Figure, stem: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("svg", "pdf"):
        path = OUTPUT / f"{stem}.{suffix}"
        fig.savefig(path, metadata={"Date": None})
        if suffix == "svg":
            path.write_text(
                "\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()) + "\n",
                encoding="utf-8",
            )
    fig.savefig(OUTPUT / f"_{stem}_preview.png", dpi=180)
    plt.close(fig)


def figure_method_comparison(metrics: dict[str, dict[str, float]]) -> None:
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.16, 2.75),
        gridspec_kw={"width_ratios": (1.42, 1.0, 1.0)},
    )
    fig.subplots_adjust(left=0.19, right=0.985, bottom=0.20, top=0.88, wspace=0.23)

    y = np.arange(len(METHODS))
    panels = (
        ("full", "(a) Full success", "Episode rate (%)", 45),
        ("collision", "(b) Robot collision", "Robot rate (%)", 40),
        ("timeout", "(c) Timeout", "Episode rate (%)", 45),
    )
    for panel_index, (ax, (metric, title, xlabel, upper)) in enumerate(zip(axes, panels)):
        values = [metrics[key][metric] for key, _, _ in METHODS]
        colors = [color for _, _, color in METHODS]
        bars = ax.barh(y, values, height=0.58, color=colors, edgecolor="none", zorder=2)
        for bar, value, (key, _, _) in zip(bars, values, METHODS):
            ax.text(
                value + 0.7,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.2f}",
                va="center",
                ha="left",
                fontsize=7.2,
                color=PINK_DARK if key == "piroute" else INK,
                fontweight="bold" if key == "piroute" else "normal",
            )
        ax.set_xlim(0, upper)
        ax.set_ylim(-0.65, len(METHODS) - 0.35)
        ax.invert_yaxis()
        ax.set_title(title, pad=7, fontweight="bold", color=INK)
        ax.set_xlabel(xlabel, color=INK)
        ax.set_yticks(y)
        if panel_index == 0:
            ax.set_yticklabels([label for _, label, _ in METHODS], color=INK)
            for label, (key, _, _) in zip(ax.get_yticklabels(), METHODS):
                if key == "piroute":
                    label.set_color(PINK_DARK)
                    label.set_fontweight("bold")
        else:
            ax.set_yticklabels([])
        clean_axis(ax)

    save(fig, "fig2_method_comparison")


def figure_paired_effects() -> None:
    fig, (ax_outcome, ax_steps) = plt.subplots(
        1, 2, figsize=(7.16, 2.75), gridspec_kw={"width_ratios": (1.55, 1.0)}
    )
    fig.subplots_adjust(left=0.19, right=0.985, bottom=0.25, top=0.86, wspace=0.38)

    outcome_rows = (
        (
            1,
            "Full success\n(higher is better)",
            24.61,
            38.54,
            "+13.93 pp",
            "95% CI [+9.51, +18.23]",
        ),
        (
            0,
            "Robot collision\n(lower is better)",
            31.72,
            22.63,
            "-9.09 pp",
            "95% CI [-11.25, -7.03]",
        ),
    )
    for y, label, base, method, effect, interval in outcome_rows:
        ax_outcome.plot([base, method], [y, y], color=GRAY_BLUE, linewidth=1.6, zorder=1)
        ax_outcome.scatter(base, y, s=48, color=BLUE, edgecolor="white", linewidth=0.7, zorder=3)
        ax_outcome.scatter(method, y, s=58, color=PINK, edgecolor="white", linewidth=0.7, zorder=3)
        ax_outcome.text(base, y + 0.19, f"{base:.2f}%", ha="center", fontsize=7.3, color=BLUE)
        ax_outcome.text(
            method,
            y + 0.19,
            f"{method:.2f}%",
            ha="center",
            fontsize=7.3,
            color=PINK_DARK,
            fontweight="bold",
        )
        ax_outcome.text(49.0, y + 0.04, effect, ha="right", va="center", fontsize=8.0, color=INK, fontweight="bold")
        ax_outcome.text(49.0, y - 0.17, interval, ha="right", va="center", fontsize=6.8, color=MUTED)
        ax_outcome.text(
            -0.04,
            y,
            label,
            transform=ax_outcome.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=7.8,
            color=INK,
        )
    ax_outcome.set_xlim(0, 50)
    ax_outcome.set_ylim(-0.55, 1.55)
    ax_outcome.set_yticks([])
    ax_outcome.set_title("(a) Task outcomes", pad=8, fontweight="bold", color=INK)
    ax_outcome.set_xlabel("Rate (%)", color=INK)
    clean_axis(ax_outcome)

    base_steps, method_steps, pair_count = paired_success_step_means()
    ax_steps.plot([base_steps, method_steps], [0, 0], color=GRAY_BLUE, linewidth=1.6, zorder=1)
    ax_steps.scatter(base_steps, 0, s=48, color=BLUE, edgecolor="white", linewidth=0.7, zorder=3)
    ax_steps.scatter(method_steps, 0, s=58, color=PINK, edgecolor="white", linewidth=0.7, zorder=3)
    ax_steps.text(base_steps, 0.19, f"{base_steps:.2f}", ha="center", fontsize=7.3, color=BLUE)
    ax_steps.text(
        method_steps,
        0.19,
        f"{method_steps:.2f}",
        ha="center",
        fontsize=7.3,
        color=PINK_DARK,
        fontweight="bold",
    )
    ax_steps.text(39.3, 0.05, "+12.90 steps", ha="right", fontsize=8.0, color=INK, fontweight="bold")
    ax_steps.text(39.3, -0.17, "95% CI [+10.06, +16.34]", ha="right", fontsize=6.8, color=MUTED)
    ax_steps.set_xlim(0, 40)
    ax_steps.set_ylim(-0.55, 0.55)
    ax_steps.set_yticks([])
    ax_steps.set_title("(b) Completion-step cost", pad=8, fontweight="bold", color=INK)
    ax_steps.set_xlabel("Paired-success steps (lower is better)", color=INK)
    clean_axis(ax_steps)

    handles = [
        plt.Line2D([], [], marker="o", linestyle="none", color=BLUE, label="Navigation Actor"),
        plt.Line2D([], [], marker="o", linestyle="none", color=PINK, label="PIRoute"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=2,
        frameon=False,
        fontsize=7.4,
        handletextpad=0.35,
        columnspacing=1.1,
    )
    if pair_count != 126:
        raise ValueError(f"expected 126 jointly successful pairs, found {pair_count}")

    save(fig, "fig3_paired_effects")


def main() -> None:
    set_style()
    metrics = load_metrics()
    figure_method_comparison(metrics)
    figure_paired_effects()
    print(f"Wrote current paper figures to {OUTPUT}")


if __name__ == "__main__":
    main()
