#!/usr/bin/env python3
"""Redraw the publication figures for ICRA layout.

The script reads only the frozen G25/Q1/E1 statistics and writes each figure
to its own folder under ``paper/generated``. It does not recompute any
experiment.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle, Wedge


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/03_保留专门化/02_论文主线"
OUT = ROOT / "paper/generated"
FIGURE_DIRS = {
    "piroute_overview": "fig1_overview",
    "g25_pareto": "fig2_tradeoff",
    "g25_primary_effects": "fig3_effects",
    "g26_e1_effects": "supplement",
}
G25 = BASE / "25_最终消融与Sealed评测/local_data/sealed/sealed_statistics.json"
Q1 = BASE / "26_数量泛化与外部切换基线/local_data/q1/results/q1_statistics.json"
E1 = BASE / "26_数量泛化与外部切换基线/local_data/e1/e1_statistics.json"

OKABE_ITO = {
    "blue": "#0072B2",
    "orange": "#D55E00",
    "green": "#009E73",
    "yellow": "#E69F00",
    "purple": "#CC79A7",
    "gray": "#7A8793",
    "red": "#B2182B",
    "ink": "#26323A",
    "grid": "#D9E0E5",
    "paper": "#F7F9FA",
}

METHODS = [
    ("5a", "5A", "blue"),
    ("r2b", "R2B", "gray"),
    ("ttc_cpa", "TTC/CPA", "green"),
    ("b2", "PIRoute", "red"),
    ("privileged_2m", "2 m oracle", "yellow"),
    ("epoch16", "always-on", "purple"),
    ("min_lidar", "min-LiDAR", "orange"),
]


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.4,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
    })


def save(fig, stem):
    target_dir = OUT / FIGURE_DIRS.get(stem, "supplement")
    target_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        path = target_dir / f"{stem}.{ext}"
        fig.savefig(path, dpi=600, bbox_inches="tight", facecolor="white")
        if ext == "svg":
            # Matplotlib emits trailing spaces in path data; normalize them so
            # git and the generation-record hashes remain deterministic.
            path.write_text(
                "\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()) + "\n",
                encoding="utf-8",
            )
    plt.close(fig)


def despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#71808B")
    ax.spines["bottom"].set_color("#71808B")
    ax.grid(axis="y", color=OKABE_ITO["grid"], linewidth=0.55, alpha=0.8)
    ax.set_axisbelow(True)


def rounded_box(ax, xy, width, height, title, detail, face, edge):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), width, height, boxstyle="round,pad=0.012,rounding_size=0.02",
        facecolor=face, edgecolor=edge, linewidth=0.9,
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height * 0.62, title, ha="center", va="center",
            fontsize=8.7, fontweight="bold", color=OKABE_ITO["ink"])
    ax.text(x + width / 2, y + height * 0.34, detail, ha="center", va="center",
            fontsize=7.2, color="#52616D")


def arrow(ax, start, end, color, dashed=False):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=9, linewidth=1.0,
        linestyle="--" if dashed else "-", color=color,
        connectionstyle="arc3,rad=0.0",
    ))


def figure_overview():
    """Reference-style overview with explicit branches and training lineage."""
    fig, ax = plt.subplots(figsize=(7.16, 4.20))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)

    ink, muted = OKABE_ITO["ink"], "#5B6872"
    blue, green, orange, purple = OKABE_ITO["blue"], OKABE_ITO["green"], OKABE_ITO["orange"], "#6E4F92"
    red, neutral = "#A33C3A", "#8295A0"

    def panel(x, y, w, h, face="white", edge="#C6D0D6", lw=.9):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=.006,rounding_size=.012",
                                    facecolor=face, edgecolor=edge, linewidth=lw))

    def node(x, y, w, h, title, detail, color, face="white", size=6.6):
        panel(x, y, w, h, face, color, .9)
        ax.text(x+w/2, y+h*.62, title, ha="center", va="center", fontsize=size,
                fontweight="bold", color=color)
        if detail:
            ax.text(x+w/2, y+h*.27, detail, ha="center", va="center", fontsize=size-1.0, color=muted)

    def lock(x, y, scale=.8):
        w, h = .014*scale, .011*scale
        ax.add_patch(Rectangle((x-w/2, y-h/2), w, h, facecolor="white", edgecolor=neutral, linewidth=.7))
        ax.add_patch(Arc((x, y+h/2), w*.75, h*1.2, theta1=0, theta2=180, color=neutral, linewidth=.7))
        ax.add_patch(Circle((x, y), .0014*scale, facecolor=neutral, edgecolor="none"))

    ax.text(.03, .965, "PIRoute", fontsize=14, fontweight="bold", color=ink, va="top")
    ax.text(.155, .965, "specialized policies with decentralized online routing", fontsize=8.2, color=muted, va="top")

    # Deployment: the local observation fans out to the perception branch and
    # the two policy branches; only their outputs meet at the Router/selector.
    ax.text(.03, .91, "DEPLOYMENT: per-robot online routing (decentralized)", fontsize=8.1,
            fontweight="bold", color=muted, va="center")
    panel(.025, .405, .95, .475, "#F7F9FA", "#AEBCC5", 1.0)
    node(.045, .515, .15, .275, "1  LOCAL VIEW", "LiDAR • motion • goal", blue, "#F3F8FB", 6.8)
    sx, sy = .105, .63
    ax.add_patch(Wedge((sx, sy), .052, 20, 160, facecolor="#DCEAF3", edgecolor="none", alpha=.9))
    for ang in np.linspace(30, 150, 5):
        ex = sx + .047*np.cos(np.deg2rad(ang)); ey = sy + .047*np.sin(np.deg2rad(ang))
        ax.plot([sx, ex], [sy, ey], color=blue, linewidth=.45)
    ax.add_patch(Circle((sx, sy), .014, facecolor="white", edgecolor=blue, linewidth=1.0, zorder=4))
    ax.add_patch(Circle((sx, sy), .004, facecolor=blue, edgecolor="none", zorder=5))
    ax.add_patch(Circle((.155, .67), .011, facecolor="white", edgecolor=orange, linewidth=.9, zorder=4))
    ax.add_patch(Circle((.155, .67), .0035, facecolor=orange, edgecolor="none", zorder=5))

    node(.255, .735, .16, .095, "2  FROZEN PERCEPTION", "G0 + tracking -> q_t", blue, "#F3F8FB", 6.3)
    panel(.255, .485, .18, .20, "#FFFDFB", "#D5A27C", .9)
    ax.text(.345, .665, "FROZEN POLICY BANK", ha="center", va="center", fontsize=6.5, fontweight="bold", color=muted)
    node(.27, .575, .15, .065, "3a  Actor N", "efficient progress", green, "#F3FAF7", 6.1)
    node(.27, .50, .15, .065, "3b  Actor I", "local interaction avoidance", orange, "#FFF7F0", 5.9)
    lock(.412, .59); lock(.412, .515)
    ax.text(.345, .468, "a_N, a_I", ha="center", fontsize=6.1, color=muted)
    ax.add_patch(FancyArrowPatch((.195, .70), (.255, .775), arrowstyle="-|>", mutation_scale=7, linewidth=.9, color=blue))
    ax.add_patch(FancyArrowPatch((.195, .625), (.255, .61), arrowstyle="-|>", mutation_scale=7, linewidth=.9, color=blue))

    node(.50, .55, .16, .20, "4  TEMPORAL ROUTER", "8-frame GRU\nq_t, a_N, a_I, Delta a", purple, "#F2ECF8", 6.7)
    node(.705, .59, .105, .12, "5  HYSTERESIS", "+ minimum hold", purple, "#F2ECF8", 6.0)
    node(.845, .55, .095, .16, "6  SELECT", "z_t chooses\na_N or a_I", ink, "white", 6.0)
    node(.92, .64, .055, .10, "7 EXEC.", "a_t", blue, "#F3F8FB", 5.0)
    ax.add_patch(FancyArrowPatch((.415, .79), (.50, .70), arrowstyle="-|>", mutation_scale=7, linewidth=.9, color=blue))
    ax.add_patch(FancyArrowPatch((.415, .61), (.50, .62), arrowstyle="-|>", mutation_scale=7, linewidth=.9, color=green))
    ax.add_patch(FancyArrowPatch((.415, .535), (.50, .59), arrowstyle="-|>", mutation_scale=7, linewidth=.9, color=orange))
    # Action disagreement is derived from the two candidate actions.
    panel(.405, .435, .095, .06, "white", neutral, .8)
    ax.text(.452, .475, "Delta a = a_I - a_N", ha="center", va="center", fontsize=5.8, color=ink)
    ax.add_patch(FancyArrowPatch((.35, .575), (.405, .475), arrowstyle="-|>", mutation_scale=6, linewidth=.7, color=green))
    ax.add_patch(FancyArrowPatch((.35, .50), (.405, .455), arrowstyle="-|>", mutation_scale=6, linewidth=.7, color=orange))
    ax.add_patch(FancyArrowPatch((.50, .47), (.50, .57), arrowstyle="-|>", mutation_scale=6, linewidth=.7, color=neutral))
    ax.add_patch(FancyArrowPatch((.66, .65), (.705, .65), arrowstyle="-|>", mutation_scale=7, linewidth=.9, color=purple))
    ax.text(.684, .67, "p_t", ha="center", fontsize=5.8, color=purple)
    ax.add_patch(FancyArrowPatch((.81, .65), (.845, .65), arrowstyle="-|>", mutation_scale=7, linewidth=.9, color=purple))
    ax.text(.827, .67, "z_t", ha="center", fontsize=5.8, color=purple)
    ax.add_patch(FancyArrowPatch((.94, .64), (.94, .58), arrowstyle="-|>", mutation_scale=7, linewidth=.9, color=blue))
    ax.plot([.42, .42, .80], [.605, .515, .515], color=green, linewidth=.8)
    ax.add_patch(FancyArrowPatch((.80, .515), (.845, .585), arrowstyle="-|>", mutation_scale=6, linewidth=.7, color=green))
    ax.plot([.42, .42, .80], [.535, .49, .49], color=orange, linewidth=.8)
    ax.add_patch(FancyArrowPatch((.80, .49), (.845, .555), arrowstyle="-|>", mutation_scale=6, linewidth=.7, color=orange))
    ax.plot([.947, .947, .20, .20], [.635, .425, .425, .515], color=neutral, linewidth=.8)
    ax.add_patch(FancyArrowPatch((.20, .515), (.195, .515), arrowstyle="-|>", mutation_scale=6, linewidth=.8, color=neutral))
    ax.text(.55, .425, "next local observation", ha="center", fontsize=5.8, color=muted)
    ax.text(.895, .445, "no communication", ha="center", fontsize=5.8, color=muted)

    # Training: actor lineage, perception pretraining, and Router supervision.
    ax.text(.03, .37, "TRAINING ONLY (offline)", fontsize=8.0, fontweight="bold", color=red, va="center")
    panel(.025, .07, .95, .275, "#FFFDFD", "#D9A0A0", 1.0)
    panel(.045, .10, .42, .205, "white", "#D7C1C1", .8)
    ax.text(.06, .315, "A. Actor pretraining (centralized critic)", fontsize=6.8, fontweight="bold", color=ink, va="top")
    node(.06, .15, .09, .105, "Gazebo", "shared state", red, "#FCF5F5", 6.0)
    node(.18, .205, .105, .05, "train N", "", red, "#FCF5F5", 5.7)
    node(.31, .205, .12, .05, "frozen N", "", green, "#F3FAF7", 5.7)
    node(.18, .125, .105, .05, "interaction update", "", red, "#FCF5F5", 5.2)
    node(.31, .125, .12, .05, "frozen I", "", orange, "#FFF7F0", 5.7)
    ax.add_patch(FancyArrowPatch((.15, .225), (.18, .23), arrowstyle="-|>", mutation_scale=6, linewidth=.8, color=red))
    ax.add_patch(FancyArrowPatch((.285, .23), (.31, .23), arrowstyle="-|>", mutation_scale=6, linewidth=.8, color=green))
    ax.plot([.37, .37, .295, .295], [.205, .18, .18, .175], color=orange, linewidth=.8)
    ax.add_patch(FancyArrowPatch((.295, .175), (.31, .15), arrowstyle="-|>", mutation_scale=6, linewidth=.8, color=orange))

    panel(.485, .10, .465, .205, "white", "#D7C1C1", .8)
    ax.text(.50, .315, "B. Perception pretraining", fontsize=6.8, fontweight="bold", color=ink, va="top")
    node(.50, .145, .09, .095, "robot pose", "ground truth", red, "#FCF5F5", 5.8)
    node(.615, .22, .105, .05, "G0 supervision", "", blue, "#F3F8FB", 5.5)
    node(.75, .22, .105, .05, "G0 pretraining", "", blue, "#F3F8FB", 5.5)
    node(.865, .22, .07, .05, "G0 + track", "frozen", blue, "#F3F8FB", 5.1)
    ax.add_patch(FancyArrowPatch((.59, .21), (.615, .245), arrowstyle="-|>", mutation_scale=6, linewidth=.8, color=red))
    arrow(ax, (.72, .245), (.75, .245), blue)
    arrow(ax, (.855, .245), (.865, .245), blue)
    node(.615, .125, .105, .05, "y_t = 1[d < 2 m]", "interaction label", red, "#FCF5F5", 5.1)
    ax.text(.615, .205, "C. ROUTER TRAINING", fontsize=5.0, fontweight="bold", color=ink, va="top")
    node(.77, .115, .14, .07, "train Router only", "x_t + y_t; N / I / G0 frozen", purple, "#F2ECF8", 5.4)
    arrow(ax, (.59, .17), (.615, .15), red)
    arrow(ax, (.72, .15), (.77, .15), red)
    ax.add_patch(FancyArrowPatch((.90, .22), (.90, .185), arrowstyle="-|>", mutation_scale=6,
                                 linewidth=.8, color=blue))
    ax.text(.915, .205, "x_t", fontsize=5.4, color=blue, va="center")
    ax.text(.72, .085, "deployable features + privileged labels", ha="center", fontsize=5.2, color=muted)

    ax.plot([.04, .06], [.035, .035], color=red, linewidth=2.2)
    ax.text(.065, .035, "training-only supervision", fontsize=5.9, color=red, va="center")
    lock(.285, .035, .7)
    ax.text(.30, .035, "frozen parameters", fontsize=5.9, color=neutral, va="center")
    ax.text(.70, .035, "deployment uses local observations only", fontsize=5.9, color=muted, va="center")
    save(fig, "piroute_overview")


def figure_overview_previous():
    """Fig. 1 method map: role specialization, Router training, and deployment."""
    fig, ax = plt.subplots(figsize=(7.16, 4.10))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)

    ink, muted = OKABE_ITO["ink"], "#5B6872"
    blue, green, orange, purple = OKABE_ITO["blue"], OKABE_ITO["green"], OKABE_ITO["orange"], "#6E4F92"
    red = "#A33C3A"
    neutral = "#8295A0"

    ax.text(.03, .965, "PIRoute", fontsize=14, fontweight="bold", color=ink, va="top")
    ax.text(.155, .965, "from specialized policies to local-observation routing", fontsize=8.6, color=muted, va="top")

    def panel(x, y, w, h, face, edge, lw=1.0):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=.008,rounding_size=.014",
                                    facecolor=face, edgecolor=edge, linewidth=lw))

    def box(x, y, w, h, title, detail, color, face="white", title_size=7.8, detail_size=6.5):
        panel(x, y, w, h, face, color, .95)
        ax.plot([x+.014, x+.014], [y+.015, y+h-.015], color=color, linewidth=2.8, solid_capstyle="round")
        ax.text(x+.032, y+h*.63, title, fontsize=title_size, fontweight="bold", color=color, va="center")
        ax.text(x+.032, y+h*.28, detail, fontsize=detail_size, color=muted, va="center")

    def lock_icon(x, y, color=neutral, scale=1.0):
        w, h = .018*scale, .014*scale
        ax.add_patch(Rectangle((x-w/2, y-h/2), w, h, facecolor="white",
                               edgecolor=color, linewidth=.8))
        ax.add_patch(Arc((x, y+h/2), w*.7, h*1.1, theta1=0, theta2=180,
                         color=color, linewidth=.8))
        ax.add_patch(Circle((x, y), .0018*scale, facecolor=color, edgecolor="none"))

    # Column headers and outer containers.
    ax.text(.03, .895, "TRAINING", fontsize=8.3, fontweight="bold", color=muted, va="center")
    ax.text(.69, .895, "DEPLOYMENT / INFERENCE", fontsize=8.3, fontweight="bold", color=muted, va="center")
    panel(.03, .13, .62, .72, "#F7F9FA", "#B8C4CC", 1.0)
    panel(.69, .13, .28, .72, "#F7F9FA", "#B8C4CC", 1.0)

    # Training stage 1: the two roles are learned under a centralized critic,
    # then both Actors are frozen before Router optimization.
    panel(.055, .555, .57, .245, "white", "#C6D0D6", .85)
    ax.text(.075, .775, "1  ROLE SPECIALIZATION", fontsize=7.3, fontweight="bold", color=ink, va="top")
    box(.075, .625, .145, .105, "centralized critic", "shared simulator state", purple, "#F2ECF8", 7.0, 6.0)
    box(.285, .645, .145, .095, "Actor N", "goal-directed", green, "#F2FAF7")
    box(.455, .645, .145, .095, "Actor I", "interaction avoidance", orange, "#FFF7F0")
    arrow(ax, (.22, .692), (.285, .692), purple)
    arrow(ax, (.43, .692), (.455, .692), orange)
    ax.text(.442, .605, "warm start + interaction-focused update", fontsize=5.6,
            color=orange, ha="center")
    ax.text(.365, .575, "freeze both Actors", fontsize=6.6, color=ink, ha="center", fontweight="bold")
    lock_icon(.342, .54, scale=.9)
    lock_icon(.513, .54, scale=.9)

    # Training stage 2: separate deployable input from privileged supervision.
    panel(.055, .205, .57, .285, "white", "#C6D0D6", .85)
    ax.text(.075, .465, "2  ROUTER SUPERVISION", fontsize=7.3, fontweight="bold", color=ink, va="top")
    box(.075, .30, .175, .115, "deployable features", "LiDAR + ego-motion\n8-frame history + action gap", blue, "#F3F8FB", 7.0, 5.9)
    box(.285, .30, .175, .115, "simulator labels", "positions -> interaction y_t", red, "#FCF5F5", 7.0, 6.0)
    panel(.505, .275, .095, .16, "#F2ECF8", purple, 1.0)
    ax.text(.552, .395, "GRU", fontsize=8.0, fontweight="bold", color=purple, ha="center")
    ax.text(.552, .355, "Router B2", fontsize=6.8, fontweight="bold", color=ink, ha="center")
    ax.text(.552, .315, "update only", fontsize=6.0, color=muted, ha="center")
    ax.plot([.25, .25, .49], [.43, .43, .43], color=blue, linewidth=1.0)
    ax.add_patch(FancyArrowPatch((.49, .43), (.505, .405), arrowstyle="-|>", mutation_scale=7,
                                 linewidth=1.0, color=blue))
    ax.plot([.46, .49], [.275, .275], color=red, linewidth=1.0)
    ax.add_patch(FancyArrowPatch((.49, .275), (.505, .30), arrowstyle="-|>", mutation_scale=7,
                                 linewidth=1.0, color=red))
    ax.text(.18, .245, "available at deployment", fontsize=6.1, color=blue, ha="center")
    ax.text(.39, .245, "training-only labels", fontsize=6.1, color=red, ha="center")
    ax.text(.545, .245, "freeze Router", fontsize=6.1, color=neutral, ha="center")
    lock_icon(.602, .245, scale=.65)

    # Deployment stage: one robot's local view produces two candidate actions;
    # Router selects the action actually executed by that robot.
    panel(.715, .625, .23, .155, "white", "#C6D0D6", .85)
    ax.text(.73, .762, "robot i: local view", fontsize=7.0, fontweight="bold", color=ink, va="top")
    sx, sy = .775, .685
    ax.add_patch(Wedge((sx, sy), .058, 18, 162, facecolor="#DCEAF3", edgecolor="none", alpha=.85))
    for ang in np.linspace(28, 152, 5):
        ex = sx + .05*np.cos(np.deg2rad(ang)); ey = sy + .05*np.sin(np.deg2rad(ang))
        ax.plot([sx, ex], [sy, ey], color=blue, linewidth=.45, alpha=.75)
    ax.add_patch(Circle((sx, sy), .015, facecolor="white", edgecolor=blue, linewidth=1.0, zorder=4))
    ax.add_patch(Circle((sx, sy), .005, facecolor=blue, edgecolor="none", zorder=5))
    ax.add_patch(Circle((.845, .715), .012, facecolor="white", edgecolor=orange, linewidth=1.0, zorder=4))
    ax.add_patch(Circle((.845, .715), .004, facecolor=orange, edgecolor="none", zorder=5))
    ax.text(.86, .685, "LiDAR, ego-motion,\nhistory", fontsize=5.8, color=blue, va="center")

    box(.715, .475, .095, .09, "Actor N", "a_N", green, "#F3FAF7", 6.8, 6.0)
    box(.85, .475, .095, .09, "Actor I", "a_I", orange, "#FFF7F0", 6.8, 6.0)
    arrow(ax, (.83, .67), (.76, .565), blue)
    arrow(ax, (.83, .67), (.895, .565), blue)
    ax.text(.83, .445, "candidate gap |a_N - a_I|", fontsize=6.0, color=muted, ha="center")
    panel(.755, .295, .15, .105, "#F2ECF8", purple, 1.0)
    ax.text(.83, .37, "TEMPORAL ROUTER", fontsize=6.9, fontweight="bold", color=purple, ha="center")
    ax.text(.83, .335, "GRU + hysteresis", fontsize=6.1, color=ink, ha="center")
    ax.text(.83, .31, "g_t in {N, I}", fontsize=6.6, fontweight="bold", color=purple, ha="center")
    arrow(ax, (.81, .475), (.80, .405), green)
    arrow(ax, (.895, .475), (.86, .405), orange)
    panel(.735, .17, .19, .075, "white", neutral, .9)
    ax.text(.83, .218, "execute selected a_t", fontsize=7.0, fontweight="bold", color=ink, ha="center")
    ax.text(.83, .185, "no communication", fontsize=6.1, color=muted, ha="center")
    arrow(ax, (.83, .295), (.83, .245), purple)
    ax.add_patch(FancyArrowPatch((.90, .155), (.76, .155), arrowstyle="-|>", mutation_scale=7,
                                 linewidth=.8, color=neutral, connectionstyle="arc3,rad=0"))

    # Minimal legend makes the semantic encoding explicit without another box maze.
    ax.plot([.04, .06], [.06, .06], color=red, linewidth=2.4)
    ax.text(.065, .06, "training-only supervision", fontsize=6.2, color=red, va="center")
    lock_icon(.27, .06, scale=.75)
    ax.text(.285, .06, "frozen parameters", fontsize=6.2, color=neutral, va="center")
    ax.text(.53, .06, "deployment uses local observations only", fontsize=6.2, color=muted, va="center")
    save(fig, "piroute_overview")


def figure_overview_compact():
    """Compact Fig. 1 overview: deployment chain above, training boundary below."""
    fig, ax = plt.subplots(figsize=(7.16, 3.55))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)

    ink, muted = OKABE_ITO["ink"], "#5B6872"
    blue, green, orange, purple = OKABE_ITO["blue"], OKABE_ITO["green"], OKABE_ITO["orange"], "#6E4F92"
    red = "#A33C3A"

    ax.text(.03, .965, "PIRoute", ha="left", va="top", fontsize=13, fontweight="bold", color=ink)
    ax.text(.155, .965, "local-observation policy routing", ha="left", va="top", fontsize=8.3, color=muted)

    # Deployment lane: one horizontal chain, with a small scene cue for the
    # multi-robot setting and no crossing arrows.
    ax.text(.03, .885, "DEPLOYMENT", ha="left", va="center", fontsize=8.2, fontweight="bold", color=muted)
    ax.text(.97, .885, "each robot acts independently", ha="right", va="center", fontsize=7.0, color=muted)
    ax.add_patch(FancyBboxPatch((.03, .39), .19, .43, boxstyle="round,pad=.01,rounding_size=.018",
                                facecolor="#F7F9FA", edgecolor="#C6D0D6", linewidth=.9))
    ax.text(.05, .785, "LOCAL OBSERVATION", fontsize=7.0, fontweight="bold", color=muted, va="top")
    sx, sy = .105, .565
    ax.add_patch(Wedge((sx, sy), .105, 18, 162, facecolor="#DCEAF3", edgecolor="none", alpha=.8))
    for ang in np.linspace(25, 155, 7):
        ex = sx + .09*np.cos(np.deg2rad(ang)); ey = sy + .09*np.sin(np.deg2rad(ang))
        ax.plot([sx, ex], [sy, ey], color=blue, linewidth=.55, alpha=.7)
    ax.add_patch(Circle((sx, sy), .026, facecolor="white", edgecolor=blue, linewidth=1.7, zorder=4))
    ax.add_patch(Circle((sx, sy), .008, facecolor=blue, edgecolor="none", zorder=5))
    ax.add_patch(FancyArrowPatch((sx, sy), (sx+.04, sy+.016), arrowstyle="-|>", mutation_scale=7,
                                 linewidth=1.0, color=blue, zorder=5))
    ax.add_patch(Circle((.17, .64), .021, facecolor="white", edgecolor=orange, linewidth=1.5, zorder=4))
    ax.add_patch(Circle((.17, .64), .007, facecolor=orange, edgecolor="none", zorder=5))
    ax.text(.105, .475, "LiDAR + ego-motion", ha="center", fontsize=6.8, color=blue)
    ax.text(.105, .425, "+ short history", ha="center", fontsize=6.8, color=muted)

    def overview_module(x, y, w, h, title, detail, color, face="white"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=.008,rounding_size=.014",
                                    facecolor=face, edgecolor=color, linewidth=1.1))
        ax.plot([x+.018, x+.018], [y+.018, y+h-.018], color=color, linewidth=3.0, solid_capstyle="round")
        ax.text(x+.038, y+h*.63, title, fontsize=8.3, fontweight="bold", color=color, va="center")
        ax.text(x+.038, y+h*.30, detail, fontsize=6.8, color=muted, va="center")

    overview_module(.285, .64, .17, .105, "Actor N", "goal-directed", green)
    overview_module(.285, .47, .17, .105, "Actor I", "interaction avoidance", orange)
    ax.text(.37, .415, "candidate actions", ha="center", fontsize=6.8, color=muted)
    ax.text(.37, .388, "a_N, a_I, |a_N - a_I|", ha="center", fontsize=6.5, color=muted)
    arrow(ax, (.22, .59), (.285, .69), blue)
    arrow(ax, (.22, .59), (.285, .52), blue)

    ax.add_patch(FancyBboxPatch((.53, .46), .18, .31, boxstyle="round,pad=.01,rounding_size=.018",
                                facecolor="#F2ECF8", edgecolor=purple, linewidth=1.15))
    ax.text(.62, .735, "TEMPORAL ROUTER", ha="center", fontsize=7.7, fontweight="bold", color=purple)
    ax.text(.62, .695, "GRU over 8-frame history", ha="center", fontsize=6.8, color=muted)
    for k, h in enumerate([.02, .035, .025, .045, .03, .052, .038, .048]):
        ax.add_patch(Rectangle((.55+k*.016, .61), .010, h, facecolor=purple, edgecolor="none", alpha=.4+.06*k))
    ax.text(.62, .565, "hysteresis + hold", ha="center", fontsize=7.1, fontweight="bold", color=ink)
    ax.text(.62, .515, "g_t in {N, I}", ha="center", fontsize=9.0, fontweight="bold", color=purple)
    ax.text(.62, .475, "route stride = 2", ha="center", fontsize=6.5, color=muted)
    arrow(ax, (.455, .69), (.53, .69), green)
    arrow(ax, (.455, .52), (.53, .52), orange)

    ax.add_patch(FancyBboxPatch((.78, .52), .18, .19, boxstyle="round,pad=.01,rounding_size=.018",
                                facecolor="#F7F9FA", edgecolor="#8295A0", linewidth=1.0))
    ax.text(.87, .665, "EXECUTE", ha="center", fontsize=7.7, fontweight="bold", color=ink)
    ax.text(.87, .615, "selected action a_t", ha="center", fontsize=7.0, color=muted)
    ax.text(.87, .565, "no communication", ha="center", fontsize=6.6, color=muted)
    arrow(ax, (.71, .615), (.78, .615), purple)
    ax.plot([.87, .87, .235, .235], [.515, .335, .335, .39], color="#8295A0", linewidth=.9)
    ax.add_patch(FancyArrowPatch((.235, .39), (.21, .39), arrowstyle="-|>", mutation_scale=8,
                                 linewidth=.9, color="#8295A0"))
    ax.text(.56, .347, "next local observation", ha="center", fontsize=6.6, color=muted)

    # Training lane is separated and explicitly marked as removed at deployment.
    ax.plot([.03, .97], [.30, .30], color="#C67C78", linewidth=1.0)
    ax.text(.03, .275, "TRAINING ONLY", fontsize=8.0, fontweight="bold", color=red, va="center")
    ax.text(.97, .275, "privileged state is removed before deployment", ha="right", fontsize=6.8, color=red, va="center")
    train = [(.08, "Gazebo state", "robot positions"), (.39, "supervision", "soft + interaction labels"), (.70, "optimize", "Router only; Actors frozen")]
    for i, (x, title, detail) in enumerate(train):
        overview_module(x, .08, .20, .105, title, detail, red, "#FCF5F5")
        if i < 2:
            arrow(ax, (x+.20, .132), (train[i+1][0], .132), red)
    ax.text(.50, .03, "training uses simulator-only labels; deployment uses local observations", ha="center", fontsize=6.8, color=muted)
    save(fig, "piroute_overview")


def figure_overview_legacy():
    fig, ax = plt.subplots(figsize=(7.16, 3.55))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)

    ink, muted = OKABE_ITO["ink"], "#5B6872"
    blue, green, orange, purple = OKABE_ITO["blue"], OKABE_ITO["green"], OKABE_ITO["orange"], "#6E4F92"

    # A compact scene carries the meaning of the method: one robot observes a
    # nearby robot, and the two frozen policies propose visibly different paths.
    ax.text(0.025, 0.965, "PIRoute", ha="left", va="top", fontsize=12.5, fontweight="bold", color=ink)
    ax.text(0.145, 0.965, "online policy routing for multi-robot navigation", ha="left", va="top", fontsize=8.5, color=muted)
    ax.text(0.975, 0.965, "DEPLOYMENT", ha="right", va="top", fontsize=7.5, fontweight="bold", color=muted)

    # Scene panel
    sx, sy, sw, sh = 0.025, 0.34, 0.375, 0.54
    ax.add_patch(FancyBboxPatch((sx, sy), sw, sh, boxstyle="round,pad=0.008,rounding_size=0.015",
                                facecolor="#F7F9FA", edgecolor="#C6D0D6", linewidth=0.9))
    ax.text(sx + 0.018, sy + sh - 0.025, "LOCAL VIEW", fontsize=7.2, fontweight="bold", color=muted, va="top")
    # subtle walls/obstacle blocks
    ax.add_patch(Rectangle((sx + .025, sy + .035), .06, .018, facecolor="#DCE3E7", edgecolor="none"))
    ax.add_patch(Rectangle((sx + .285, sy + .455), .045, .018, facecolor="#DCE3E7", edgecolor="none"))
    rx, ry = sx + .105, sy + .245
    nx, ny = sx + .275, sy + .335
    # LiDAR fan and rays for robot i
    ax.add_patch(Wedge((rx, ry), 0.12, 12, 168, facecolor="#DCEAF3", edgecolor="none", alpha=0.72))
    for ang in np.linspace(18, 162, 9):
        ex = rx + .105 * np.cos(np.deg2rad(ang)); ey = ry + .105 * np.sin(np.deg2rad(ang))
        ax.plot([rx, ex], [ry, ey], color=blue, linewidth=.55, alpha=.62)
    # Candidate paths: direct (N) and yielding (I)
    t = np.linspace(0, 1, 60)
    ax.plot(rx + .22*t, ry + .02*t + .06*np.sin(np.pi*t), color=green, linewidth=2.1, solid_capstyle="round")
    ax.add_patch(FancyArrowPatch((rx+.17, ry+.075), (rx+.22, ry+.02+.06*np.sin(np.pi)), arrowstyle="-|>",
                                 mutation_scale=8, linewidth=1.4, color=green))
    ax.plot(rx + .24*t, ry + .02*t - .11*np.sin(np.pi*t), color=orange, linewidth=2.1, solid_capstyle="round")
    ax.add_patch(FancyArrowPatch((rx+.19, ry-.08), (rx+.24, ry+.02), arrowstyle="-|>",
                                 mutation_scale=8, linewidth=1.4, color=orange))
    # Robot glyphs: body, heading, and labels
    def robot(x, y, color, label, heading=25, alpha=1.0, label_dy=-.047):
        ax.add_patch(Circle((x, y), .027, facecolor="white", edgecolor=color, linewidth=1.8, alpha=alpha, zorder=5))
        ax.add_patch(Circle((x, y), .009, facecolor=color, edgecolor="none", alpha=alpha, zorder=6))
        hx, hy = x + .04*np.cos(np.deg2rad(heading)), y + .04*np.sin(np.deg2rad(heading))
        ax.add_patch(FancyArrowPatch((x, y), (hx, hy), arrowstyle="-|>", mutation_scale=7, linewidth=1.1, color=color, alpha=alpha, zorder=6))
        ax.text(x, y + label_dy, label, ha="center", va="bottom" if label_dy > 0 else "top", fontsize=6.6, color=ink, alpha=alpha)
    robot(rx, ry, blue, "robot i", 20)
    robot(nx, ny, orange, "nearby robot", 205, label_dy=.048)
    # goal marker
    gx, gy = sx + .325, sy + .115
    ax.scatter([gx], [gy], marker="*", s=95, color="#E69F00", edgecolor="#8B6410", linewidth=.55, zorder=5)
    ax.text(gx, gy-.035, "goal", ha="center", va="top", fontsize=6.6, color=ink)
    ax.text(sx + .018, sy + .025, "LiDAR + ego-motion", fontsize=6.8, color=blue)

    # Policy candidates and disagreement
    ax.text(0.425, 0.845, "candidate actions", fontsize=7.2, fontweight="bold", color=muted, va="top")
    def actor_strip(y, color, title, detail):
        ax.add_patch(FancyBboxPatch((0.425, y), .135, .095, boxstyle="round,pad=0.006,rounding_size=0.012",
                                    facecolor="white", edgecolor=color, linewidth=1.25))
        ax.plot([0.44, 0.44], [y+.018, y+.077], color=color, linewidth=3.0, solid_capstyle="round")
        ax.text(.455, y+.062, title, fontsize=8.1, fontweight="bold", color=color, va="center")
        ax.text(.455, y+.030, detail, fontsize=6.6, color=muted, va="center")
    actor_strip(.705, green, "Actor N", "goal-directed")
    actor_strip(.575, orange, "Actor I", "local avoidance")
    arrow(ax, (sx+sw, sy+.30), (.425, .752), blue)
    arrow(ax, (sx+sw, sy+.30), (.425, .622), blue)
    ax.text(.492, .548, "|a_N - a_I|", ha="center", va="top", fontsize=7.0, color=muted)

    # Router and temporal context
    ax.add_patch(FancyBboxPatch((.605, .535), .175, .29, boxstyle="round,pad=0.01,rounding_size=0.018",
                                facecolor="#F2ECF8", edgecolor=purple, linewidth=1.2))
    ax.text(.622, .792, "TEMPORAL ROUTER", fontsize=7.4, fontweight="bold", color=purple, va="top")
    ax.text(.692, .744, "8-frame history", fontsize=6.7, color=muted, ha="center")
    for k, h in enumerate([.018, .032, .024, .042, .030, .050, .036, .046]):
        ax.add_patch(Rectangle((.627 + k*.014, .681), .009, h, facecolor=purple, edgecolor="none", alpha=.45 + .06*k))
    ax.text(.692, .655, "GRU + hysteresis", fontsize=7.2, color=ink, ha="center", fontweight="bold")
    ax.add_patch(Circle((.692, .594), .032, facecolor="white", edgecolor=purple, linewidth=1.2))
    ax.text(.692, .594, "g_t", ha="center", va="center", fontsize=10.5, fontweight="bold", color=purple)
    ax.text(.692, .548, "route stride 2", fontsize=6.5, color=muted, ha="center", va="top")
    arrow(ax, (.56, .752), (.605, .752), green)
    arrow(ax, (.56, .622), (.605, .622), orange)

    # Output action and closed-loop return
    ax.add_patch(FancyBboxPatch((.835, .585), .14, .19, boxstyle="round,pad=0.008,rounding_size=0.015",
                                facecolor="#F7F9FA", edgecolor="#8295A0", linewidth=1.0))
    ax.text(.905, .745, "SELECT", fontsize=7.2, fontweight="bold", color=ink, ha="center")
    ax.text(.905, .700, "g_t -> N / I", fontsize=7.4, color=purple, ha="center", fontweight="bold")
    ax.text(.905, .657, "execute a_t", fontsize=7.0, color=muted, ha="center")
    arrow(ax, (.78, .68), (.835, .68), purple)
    # Return path runs below the computation blocks, making the closed loop
    # explicit without crossing the candidate or Router modules.
    ax.plot([.905, .905, .405, .405], [.585, .455, .455, .35],
            color="#7D8A93", linewidth=1.0)
    ax.add_patch(FancyArrowPatch((.405, .35), (.375, .35), arrowstyle="-|>",
                                 mutation_scale=8, linewidth=1.0, color="#7D8A93"))
    ax.text(.67, .468, "next local observation", fontsize=6.8, color=muted, ha="center", va="bottom")
    ax.text(.905, .548, "no communication", fontsize=6.6, color=muted, ha="center")

    # Training-only privileged path. This is an explicit boundary, not a
    # second runtime loop, so it cannot be mistaken for deployable input.
    ax.plot([.025, .975], [.275, .275], color="#B94A48", linewidth=1.25)
    ax.text(.035, .292, "TRAINING ONLY", fontsize=7.2, fontweight="bold", color="#A33C3A", va="bottom")
    ax.text(.175, .292, "privileged simulator state creates supervision", fontsize=6.8, color="#A33C3A", va="bottom")
    def train_node(x, title, detail, icon):
        ax.add_patch(FancyBboxPatch((x, .105), .205, .105, boxstyle="round,pad=0.006,rounding_size=0.012",
                                    facecolor="#FCF5F5", edgecolor="#C67C78", linewidth=1.0))
        ax.text(x+.015, .174, icon, fontsize=11, color="#B94A48", va="center")
        ax.text(x+.045, .175, title, fontsize=7.5, fontweight="bold", color=ink, va="center")
        ax.text(x+.045, .137, detail, fontsize=6.5, color=muted, va="center")
    train_node(.06, "Gazebo positions", "robot ground truth", "o")
    train_node(.385, "supervision", "G0 soft evidence + d_min < 2 m", "[]")
    train_node(.71, "update Router", "Actors remain frozen", ">")
    arrow(ax, (.265, .157), (.385, .157), "#B94A48")
    arrow(ax, (.59, .157), (.71, .157), "#B94A48")
    ax.text(.5, .035, "At deployment, the privileged path is removed; only local observations remain.",
            ha="center", fontsize=6.9, color=muted)
    save(fig, "piroute_overview")


def figure_g25_tradeoff(g25):
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(7.16, 3.25), gridspec_kw={"width_ratios": [1.18, 0.82]})
    colors = {key: OKABE_ITO[color] for key, _, color in METHODS}
    for key, label, _ in METHODS:
        item = g25["pooled_descriptive"][key]
        x, y = 100 * item["collision"], 100 * item["full_success"]
        edge = "#111111" if key == "b2" else "white"
        size = 115 if key == "b2" else 65
        ax0.scatter(x, y, s=size, color=colors[key], edgecolor=edge, linewidth=1.0, zorder=3)
        dx, dy = {"5a": (0.25, -1.7), "r2b": (0.25, 1.0), "ttc_cpa": (0.25, 1.0),
                  "b2": (0.25, 1.2), "privileged_2m": (0.25, 1.0), "epoch16": (0.25, 1.0),
                  "min_lidar": (0.25, -1.6)}[key]
        ax0.annotate(label, (x, y), xytext=(x + dx, y + dy), fontsize=7.0, color=OKABE_ITO["ink"])
    ax0.set_xlabel("Robot collision rate (%)")
    ax0.set_ylabel("Full-episode success (%)")
    ax0.set_xlim(14, 34)
    ax0.set_ylim(20, 47)
    ax0.set_title("Success and safety", loc="left", fontweight="bold")
    despine(ax0)
    ax0.grid(color=OKABE_ITO["grid"], linewidth=0.55)

    order = ["5a", "r2b", "ttc_cpa", "b2", "privileged_2m", "epoch16", "min_lidar"]
    vals = [g25["pooled_descriptive"][key]["raw_steps"] for key in order]
    labels = {key: label for key, label, _ in METHODS}
    ys = np.arange(len(order))
    for y, key, value in zip(ys, order, vals):
        ax1.hlines(y, 0, value, color=colors[key], linewidth=3.0, alpha=0.85)
        ax1.scatter(value, y, color=colors[key], s=38, edgecolor="#111111" if key == "b2" else "white", linewidth=0.8, zorder=3)
        ax1.text(value + 2.0, y, f"{value:.1f}", va="center", fontsize=7.0, color=OKABE_ITO["ink"])
    ax1.set_yticks(ys)
    ax1.set_yticklabels([labels[key] for key in order])
    ax1.set_xlabel("Raw termination steps")
    ax1.set_xlim(0, 175)
    ax1.set_title("Completion-step cost", loc="left", fontweight="bold")
    ax1.invert_yaxis()
    despine(ax1)
    ax1.grid(axis="x", color=OKABE_ITO["grid"], linewidth=0.55)
    fig.suptitle("G25 sealed test: PIRoute improves success and safety with a step cost", fontsize=11, fontweight="bold")
    fig.text(0.01, 0.005, "PIRoute = frozen epoch16 Actor + B2 Router; 2 m oracle is privileged and not deployable.", fontsize=7.0, color="#52616D")
    fig.tight_layout(rect=(0, 0.06, 1, 0.91), w_pad=2.2)
    save(fig, "g25_pareto")


def forest_panel(ax, title, value, interval, xlim, unit, color):
    ax.axvline(0, color="#5A6770", linewidth=0.9, linestyle=(0, (3, 2)))
    ax.errorbar(value, 0, xerr=[[value - interval[0]], [interval[1] - value]], fmt="o",
                color=color, ecolor=color, elinewidth=2.2, capsize=3.5, markersize=6.5,
                markeredgecolor="#111111", markeredgewidth=0.5)
    ax.set_xlim(*xlim)
    ax.set_yticks([])
    ax.set_title(title, loc="left", fontsize=8.8, fontweight="bold")
    ax.set_xlabel(unit, fontsize=7.8)
    ax.grid(axis="x", color=OKABE_ITO["grid"], linewidth=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#71808B")


def figure_g25_effects(g25):
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.55), gridspec_kw={"wspace": 0.55})
    panels = [
        ("Full success", g25["primary_b2_minus_5a"]["mean_full_success_difference"] * 100,
         np.asarray(g25["primary_b2_minus_5a"]["scene_cluster_bca_95_ci"]) * 100, (-20, 20), "difference (pp)", OKABE_ITO["blue"]),
        ("Collision", g25["secondary_b2_minus_5a"]["collision"]["mean_difference"] * 100,
         np.asarray(g25["secondary_b2_minus_5a"]["collision"]["scene_cluster_bca_95_ci"]) * 100, (-14, 14), "difference (pp)", OKABE_ITO["orange"]),
        ("Paired-success steps", g25["secondary_b2_minus_5a"]["paired_success_steps"]["mean_difference"],
         np.asarray(g25["secondary_b2_minus_5a"]["paired_success_steps"]["scene_cluster_bca_95_ci"]), (-22, 22), "difference (steps)", OKABE_ITO["purple"]),
    ]
    for i, (title, value, interval, xlim, unit, color) in enumerate(panels):
        forest_panel(axes[i], title, value, interval, xlim, unit, color)
        axes[i].text(0.98, 0.82, f"{value:+.2f}\n[{interval[0]:+.2f}, {interval[1]:+.2f}]",
                     transform=axes[i].transAxes, ha="right", va="top", fontsize=7.2,
                     color=OKABE_ITO["ink"], fontweight="bold")
        axes[i].text(-0.08, 1.08, f"({chr(ord('a') + i)})", transform=axes[i].transAxes,
                     fontsize=8.5, fontweight="bold", va="top")
    fig.suptitle("G25 sealed test: PIRoute effects relative to 5A", fontsize=10.8, fontweight="bold")
    fig.text(0.01, 0.01, "Points are paired means; whiskers are scene-cluster BCa 95% CIs. Negative collision is favorable.", fontsize=7.0, color="#52616D")
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.25, top=0.73, wspace=0.48)
    save(fig, "g25_primary_effects")


def figure_e1(e1):
    fig, ax = plt.subplots(figsize=(5.4, 2.55))
    rows = [("NF-inspired", "nf_switch_minus_5a", OKABE_ITO["gray"]), ("PIRoute (B2)", "b2_minus_5a", OKABE_ITO["red"])]
    ys = np.array([1, 0])
    for y, (label, key, color) in zip(ys, rows):
        item = e1["comparisons"][key]["full_success"]
        value = 100 * item["mean_difference"]
        interval = np.asarray(item["scene_cluster_bca_95_ci"]) * 100
        ax.errorbar(value, y, xerr=[[value - interval[0]], [interval[1] - value]], fmt="o",
                    color=color, ecolor=color, elinewidth=2.6, capsize=4, markersize=7,
                    markeredgecolor="#111111", markeredgewidth=0.5)
        ax.text(15.7, y + 0.12, f"{value:+.2f} pp  [{interval[0]:+.2f}, {interval[1]:+.2f}]",
                ha="right", va="center", fontsize=7.4, color=OKABE_ITO["ink"])
    ax.axvline(0, color="#5A6770", linewidth=0.9, linestyle=(0, (3, 2)))
    ax.set_yticks(ys)
    ax.set_yticklabels([row[0] for row in rows])
    ax.set_xlim(-10, 16)
    ax.set_ylim(-0.55, 1.55)
    ax.set_xlabel("Full-success difference relative to 5A (percentage points)")
    ax.set_title("G26-E1 exploratory external switching baseline", loc="left", fontsize=9.5, fontweight="bold")
    despine(ax)
    ax.grid(axis="x", color=OKABE_ITO["grid"], linewidth=0.55)
    fig.text(0.02, 0.04, "Independent exploratory slice; intervals cross zero and do not enter G25 inference.",
             fontsize=7.0, color="#52616D")
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.28, top=0.80)
    save(fig, "g26_e1_effects")


def main():
    style()
    g25, _q1, e1 = load(G25), load(Q1), load(E1)
    figure_overview()
    figure_g25_tradeoff(g25)
    figure_g25_effects(g25)
    figure_e1(e1)
    print(f"Wrote revised figures to {OUT}")


if __name__ == "__main__":
    main()
