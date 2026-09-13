#!/usr/bin/env python3
"""Generate the editable PIRoute reward-aware overview figure."""

from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "paper/icra2027_template/figures/fig1_overview_reward_aware_v1.svg"

W, H = 1120, 660

INK = "#394A54"
MUTED = "#667780"
LIGHT_LINE = "#C9D4D9"
PANEL_BLUE = "#F5F9FA"
PANEL_PINK = "#FCF8FA"
BLUE = "#56A1B8"
BLUE_FILL = "#E2F0F4"
BLUE_DARK = "#2F6F82"
PINK = "#C47FA0"
PINK_FILL = "#F5E6ED"
PINK_DARK = "#8D506D"
TEAL = "#5E9D93"
TEAL_FILL = "#E5F1EF"
TEAL_DARK = "#356E66"
LAVENDER = "#817AA8"
LAVENDER_FILL = "#ECEAF4"
LAVENDER_DARK = "#514B78"
PRIV = "#B45B63"
PRIV_FILL = "#F8E8EA"
NEUTRAL_FILL = "#EEF3F5"


def attrs(**kwargs):
    return " ".join(f'{key.replace("_", "-")}="{escape(str(value))}"' for key, value in kwargs.items())


parts = [
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
    "<defs>",
    f'<marker id="arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="{INK}"/></marker>',
    f'<marker id="arrow-blue" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="{BLUE_DARK}"/></marker>',
    f'<marker id="arrow-pink" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="{PINK_DARK}"/></marker>',
    f'<marker id="arrow-teal" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="{TEAL_DARK}"/></marker>',
    f'<marker id="arrow-lavender" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="{LAVENDER_DARK}"/></marker>',
    f'<marker id="arrow-priv" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="{PRIV}"/></marker>',
    "</defs>",
    '<rect width="1120" height="660" fill="#FFFFFF"/>',
]


def add(tag, **kwargs):
    parts.append(f"<{tag} {attrs(**kwargs)}/>")


def text(x, y, value, size=15, weight=400, fill=INK, anchor="start", italic=False):
    style = "italic" if italic else "normal"
    parts.append(
        f'<text x="{x}" y="{y}" font-family="Arial, Helvetica, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" font-style="{style}" '
        f'fill="{fill}" text-anchor="{anchor}" letter-spacing="0">{escape(value)}</text>'
    )


def multiline(x, y, lines, size=15, weight=400, fill=INK, anchor="middle", line_gap=18):
    parts.append(
        f'<text x="{x}" y="{y}" font-family="Arial, Helvetica, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}" '
        f'text-anchor="{anchor}" letter-spacing="0">'
    )
    for idx, line in enumerate(lines):
        dy = 0 if idx == 0 else line_gap
        parts.append(f'<tspan x="{x}" dy="{dy}">{escape(line)}</tspan>')
    parts.append("</text>")


def box(x, y, width, height, fill, stroke, radius=6, stroke_width=1.5):
    add(
        "rect",
        x=x,
        y=y,
        width=width,
        height=height,
        rx=radius,
        fill=fill,
        stroke=stroke,
        stroke_width=stroke_width,
    )


def line(x1, y1, x2, y2, stroke=INK, width=1.6, marker="arrow", dash=None):
    kw = dict(x1=x1, y1=y1, x2=x2, y2=y2, stroke=stroke, stroke_width=width, marker_end=f"url(#{marker})")
    if dash:
        kw["stroke_dasharray"] = dash
    add("line", **kw)


def path(d, stroke=INK, width=1.6, marker="arrow", dash=None):
    kw = dict(d=d, fill="none", stroke=stroke, stroke_width=width, stroke_linejoin="round", stroke_linecap="round")
    if marker:
        kw["marker_end"] = f"url(#{marker})"
    if dash:
        kw["stroke_dasharray"] = dash
    add("path", **kw)


def lock_icon(x, y, color=BLUE_DARK):
    add("rect", x=x, y=y + 7, width=12, height=10, rx=2, fill="none", stroke=color, stroke_width=1.4)
    parts.append(f'<path d="M{x+3},{y+7} V{y+4.5} A3,3 0 0 1 {x+9},{y+4.5} V{y+7}" fill="none" stroke="{color}" stroke-width="1.4"/>')


# Panel backgrounds and headings
add("rect", x=18, y=18, width=1084, height=395, rx=6, fill=PANEL_BLUE, stroke=LIGHT_LINE, stroke_width=1.1)
add("rect", x=18, y=428, width=1084, height=214, rx=6, fill=PANEL_PINK, stroke=LIGHT_LINE, stroke_width=1.1)
text(34, 48, "(a) Decentralized deployment", size=20, weight=700, fill=BLUE_DARK)
text(1082, 47, "per-robot local observations; no communication", size=13, weight=700, fill=BLUE_DARK, anchor="end")
text(34, 458, "(b) Simulation training only", size=20, weight=700, fill=PINK_DARK)

# Deployment: local observation
box(36, 92, 142, 202, NEUTRAL_FILL, "#9BAAB1")
multiline(107, 121, ["Local", "observation"], size=17, weight=700, fill=INK, line_gap=19)
add("path", d="M57 170 Q69 154 81 170", fill="none", stroke=MUTED, stroke_width=1.5, stroke_dasharray="2.5 2.5")
add("circle", cx=69, cy=174, r=2.5, fill=MUTED)
text(92, 174, "LiDAR scan", size=13, weight=600, fill=MUTED)
add("path", d="M57 212 A12 12 0 1 1 73 223", fill="none", stroke=MUTED, stroke_width=1.5)
add("path", d="M70 218 L74 224 L77 217", fill="none", stroke=MUTED, stroke_width=1.5)
text(92, 219, "Motion state", size=13, weight=600, fill=MUTED)
add("path", d="M57 250 L57 271 M57 251 L74 258 L57 264", fill="none", stroke=MUTED, stroke_width=1.5)
text(92, 263, "Goal state", size=13, weight=600, fill=MUTED)

# Deployment: frozen components
text(218, 79, "Pretrained policy experts", size=15, weight=700, fill=BLUE_DARK)
lock_icon(394, 62)
box(216, 98, 192, 62, BLUE_FILL, BLUE)
multiline(312, 124, ["Navigation Actor", "πN(oₜ) → aN,ₜ"], size=15, weight=700, fill=BLUE_DARK, line_gap=19)
box(216, 181, 192, 70, BLUE_FILL, BLUE)
multiline(312, 207, ["Interaction Actor", "πI(oₜ) → aI,ₜ"], size=15, weight=700, fill=BLUE_DARK, line_gap=19)
box(216, 272, 192, 62, NEUTRAL_FILL, "#9BAAB1")
multiline(312, 297, ["Perception + tracking", "local representation qₜ"], size=14, weight=700, fill=INK, line_gap=18)

line(178, 136, 216, 129)
line(178, 193, 216, 216)
line(178, 250, 216, 303)

# Deployment: router inputs and dual temporal branches
box(446, 88, 410, 246, "#FFFFFF", "#C8A7B7", radius=6, stroke_width=1.5)
text(462, 113, "Two temporal predictors (separate parameters)", size=16, weight=700, fill=PINK_DARK)
box(464, 132, 158, 74, NEUTRAL_FILL, "#9BAAB1", radius=5)
multiline(543, 157, ["8-frame sequence", "[qₜ, aN,ₜ, aI,ₜ, Δaₜ]"], size=13, weight=700, fill=INK, line_gap=19)
box(662, 130, 164, 67, PINK_FILL, PINK, radius=5)
multiline(744, 155, ["Phase GRU", "interaction logit ℓₜ"], size=14, weight=700, fill=PINK_DARK, line_gap=18)
box(662, 236, 164, 67, PINK_FILL, PINK, radius=5)
multiline(744, 245, ["Reward GRU", "bounded one-step", "reward-difference", "estimate Âₜ"], size=12, weight=700, fill=PINK_DARK, line_gap=16)

path("M408 129 H424 V145 H446", stroke=BLUE_DARK, width=1.5, marker="arrow-blue")
path("M408 216 H432 V169 H446", stroke=BLUE_DARK, width=1.5, marker="arrow-blue")
path("M408 303 H440 V193 H446", stroke=INK, width=1.5, marker="arrow")
path("M622 169 H641 V164 H662", stroke=PINK_DARK, width=1.5, marker="arrow-pink")
path("M622 169 H641 V270 H662", stroke=PINK_DARK, width=1.5, marker="arrow-pink")

# Deployment: fusion, stabilized state, and action
box(882, 101, 194, 76, LAVENDER_FILL, LAVENDER, radius=5)
multiline(979, 127, ["Joint score", "gₜ = σ(ℓₜ + 0.25Âₜ)"], size=14, weight=700, fill=LAVENDER_DARK, line_gap=19)
path("M826 164 H846 V127 H882", stroke=PINK_DARK, width=1.5, marker="arrow-pink")
path("M826 270 H862 V151 H882", stroke=PINK_DARK, width=1.5, marker="arrow-pink")

box(882, 202, 194, 76, LAVENDER_FILL, LAVENDER, radius=5)
multiline(979, 211, ["Stable binary mode zₜ", "update every 2 env. steps", "minimum hold: 3 updates", "after entering Interaction"], size=12, weight=700, fill=LAVENDER_DARK, line_gap=16)
line(979, 177, 979, 202, stroke=LAVENDER_DARK, width=1.6, marker="arrow-lavender")

box(882, 302, 194, 70, TEAL_FILL, TEAL, radius=5)
multiline(979, 327, ["Hard actor selection", "execute selected action"], size=14, weight=700, fill=TEAL_DARK, line_gap=19)
line(979, 278, 979, 302, stroke=TEAL_DARK, width=1.6, marker="arrow-teal")

# Candidate-action bypass to the hard selector.
path("M408 129 H414 V346 H864 V329 H882", stroke=BLUE, width=1.35, marker="arrow-blue")
path("M408 216 H422 V362 H872 V351 H882", stroke=BLUE_DARK, width=1.35, marker="arrow-blue")
text(839, 340, "aN,ₜ", size=13, weight=700, fill=BLUE_DARK, anchor="end")
text(839, 360, "aI,ₜ", size=13, weight=700, fill=BLUE_DARK, anchor="end")

# Closed-loop feedback and deployment notes.
path("M1076 337 H1087 V393 H107 V294", stroke=INK, width=1.35, marker="arrow")
text(610, 407, "next local observation; selected actor recomputes the action every step", size=13, weight=600, fill=MUTED, anchor="middle")

# Distinct training sources for dense phase labels and sparse paired rewards.
box(38, 472, 166, 56, NEUTRAL_FILL, "#9BAAB1")
multiline(121, 494, ["Rollout frames", "all phase-labeled states"], size=13, weight=700, fill=INK, line_gap=17)
box(38, 550, 166, 68, NEUTRAL_FILL, "#9BAAB1")
multiline(121, 570, ["Audited aligned anchors", "smaller paired subset", "same simulator state"], size=12, weight=700, fill=INK, line_gap=16)

# Phase supervision lane.
box(242, 472, 176, 56, PRIV_FILL, PRIV, radius=5)
multiline(330, 494, ["Nearest active-robot", "distance dₜ"], size=13, weight=700, fill="#86434A", line_gap=17)
box(456, 472, 152, 56, PRIV_FILL, PRIV, radius=5)
multiline(532, 494, ["Phase target", "yₜ = 1[dₜ ≤ 2 m]"], size=13, weight=700, fill="#86434A", line_gap=17)
line(204, 500, 242, 500, stroke=PRIV, width=1.5, marker="arrow-priv", dash="5 4")
line(418, 500, 456, 500, stroke=PRIV, width=1.5, marker="arrow-priv", dash="5 4")

# Reward supervision lane.
box(242, 550, 176, 68, PRIV_FILL, PRIV, radius=5)
multiline(330, 570, ["Isolated one-step", "aN,ₜ branch", "aI,ₜ branch"], size=13, weight=700, fill="#86434A", line_gap=16)
box(456, 550, 152, 68, PRIV_FILL, PRIV, radius=5)
multiline(532, 570, ["Paired reward target", "Δrₜ = rI,ₜ − rN,ₜ", "bounded target Aₜ"], size=13, weight=700, fill="#86434A", line_gap=16)
line(204, 584, 242, 584, stroke=PRIV, width=1.5, marker="arrow-priv", dash="5 4")
line(418, 584, 456, 584, stroke=PRIV, width=1.5, marker="arrow-priv", dash="5 4")

# Losses and router-only update boundary.
box(650, 468, 181, 64, "#FFFFFF", PINK, radius=5)
multiline(740, 493, ["Weighted BCE", "supervises Phase GRU"], size=13, weight=700, fill=PINK_DARK, line_gap=18)
box(650, 550, 181, 68, "#FFFFFF", PINK, radius=5)
multiline(740, 575, ["Weighted Huber loss", "supervises Reward GRU"], size=13, weight=700, fill=PINK_DARK, line_gap=18)
line(608, 500, 650, 500, stroke=PRIV, width=1.5, marker="arrow-priv", dash="5 4")
line(608, 584, 650, 584, stroke=PRIV, width=1.5, marker="arrow-priv", dash="5 4")

box(870, 493, 206, 94, LAVENDER_FILL, LAVENDER, radius=5)
multiline(973, 520, ["Router-only updates", "Actors + perception fixed", "Targets absent at deployment"], size=13, weight=700, fill=LAVENDER_DARK, line_gap=18)
path("M831 500 H851 V524 H870", stroke=PRIV, width=1.5, marker="arrow-priv", dash="5 4")
path("M831 584 H851 V556 H870", stroke=PRIV, width=1.5, marker="arrow-priv", dash="5 4")

# Compact legend.
line(881, 615, 917, 615, stroke=INK, width=1.5, marker="arrow")
text(925, 620, "deployable flow", size=13, weight=600, fill=MUTED)
line(881, 635, 917, 635, stroke=PRIV, width=1.5, marker="arrow-priv", dash="5 4")
text(925, 640, "training-only supervision", size=13, weight=600, fill=MUTED)

parts.append("</svg>")
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text("\n".join(parts), encoding="utf-8")
print(OUTPUT)
