#!/usr/bin/env python3
"""Generate deterministic paper tables and dependency-free SVG figures."""

import csv
import hashlib
import html
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/03_保留专门化/02_论文主线"
G25_PATH = BASE / "25_最终消融与Sealed评测/local_data/sealed/sealed_statistics.json"
Q1_PATH = BASE / "26_数量泛化与外部切换基线/local_data/q1/results/q1_statistics.json"
E1_PATH = BASE / "26_数量泛化与外部切换基线/local_data/e1/e1_statistics.json"
OUTPUT = ROOT / "paper/generated"


G25_METHODS = (
    ("5a", "5A"),
    ("epoch16", "epoch16 always-on"),
    ("min_lidar", "min-LiDAR rule"),
    ("ttc_cpa", "TTC/CPA rule"),
    ("r2b", "R2B"),
    ("privileged_2m", "2 m privileged rule"),
    ("b2", "PIRoute (B2)"),
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pct(value, digits=1):
    return f"{100.0 * float(value):.{digits}f}%"


def num(value, digits=2):
    return f"{float(value):.{digits}f}"


def esc(value):
    return html.escape(str(value), quote=True)


def write_text(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def svg_open(width, height, title):
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f"<title id=\"title\">{esc(title)}</title>",
        '<desc id="desc">Generated from frozen experiment statistics.</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
    ]


def svg_close(lines):
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def svg_text(lines, x, y, value, size=14, anchor="start", fill="#20252b", weight="400"):
    lines.append(
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, sans-serif" font-size="{size}" '
        f'font-weight="{weight}" text-anchor="{anchor}" fill="{fill}">{esc(value)}</text>'
    )


def svg_line(lines, x1, y1, x2, y2, stroke="#cbd2d9", width=1, dash=None):
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    lines.append(
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{stroke}" stroke-width="{width}"{dash_attr}/>'
    )


def svg_circle(lines, x, y, radius, fill, stroke="#ffffff", width=1.5):
    lines.append(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{width}"/>'
    )


def svg_rect(lines, x, y, width, height, fill, stroke="none", radius=0):
    lines.append(
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" '
        f'rx="{radius}" fill="{fill}" stroke="{stroke}"/>'
    )


def axis(lines, left, top, right, bottom, x_ticks, y_ticks, x_map, y_map, x_label, y_label):
    svg_line(lines, left, bottom, right, bottom, stroke="#34404b", width=1.2)
    svg_line(lines, left, top, left, bottom, stroke="#34404b", width=1.2)
    for tick, label in x_ticks:
        x = x_map(tick)
        svg_line(lines, x, bottom, x, bottom + 6, stroke="#34404b")
        svg_text(lines, x, bottom + 24, label, size=12, anchor="middle", fill="#4d5965")
    for tick, label in y_ticks:
        y = y_map(tick)
        svg_line(lines, left - 6, y, left, y, stroke="#34404b")
        svg_line(lines, left, y, right, y, stroke="#e5e9ed", dash="3 4")
        svg_text(lines, left - 12, y + 4, label, size=12, anchor="end", fill="#4d5965")
    svg_text(lines, (left + right) / 2, bottom + 52, x_label, size=14, anchor="middle", weight="600")
    svg_text(lines, left - 58, (top + bottom) / 2, y_label, size=14, anchor="middle", weight="600")


def g25_table(data):
    fields = (
        ("Method", "method"),
        ("Full success", "full_success"),
        ("Agent success", "agent_success"),
        ("Collision", "collision"),
        ("Timeout", "timeout"),
        ("Raw steps", "raw_steps"),
        ("Interaction share", "interaction_share"),
    )
    rows = []
    for key, label in G25_METHODS:
        item = data["pooled_descriptive"][key]
        rows.append(
            {
                "method": label,
                "full_success": pct(item["full_success"]),
                "agent_success": pct(item["agent_success"]),
                "collision": pct(item["collision"]),
                "timeout": pct(item["timeout"]),
                "raw_steps": num(item["raw_steps"]),
                "interaction_share": pct(item["interaction_share"]),
            }
        )
    return fields, rows


def write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow([label for label, _ in fields])
        for row in rows:
            writer.writerow([row[key] for _, key in fields])


def write_markdown_table(path, title, fields, rows, note):
    header = "| " + " | ".join(label for label, _ in fields) + " |"
    divider = "|" + "|".join("---" for _ in fields) + "|"
    body = ["| " + " | ".join(row[key] for _, key in fields) + " |" for row in rows]
    content = [f"# {title}", "", header, divider, *body, "", f"> {note}", ""]
    write_text(path, "\n".join(content))


def g25_pareto(data):
    width, height = 1000, 660
    left, top, right, bottom = 145, 70, 930, 525
    x_min, x_max = 0.14, 0.34
    y_min, y_max = 0.20, 0.47
    x_map = lambda value: left + (value - x_min) / (x_max - x_min) * (right - left)
    y_map = lambda value: bottom - (value - y_min) / (y_max - y_min) * (bottom - top)
    lines = svg_open(width, height, "G25 sealed success collision steps Pareto")
    svg_text(lines, width / 2, 34, "G25 sealed: success, collision, and raw-step trade-off", 20, "middle", weight="700")
    axis(
        lines,
        left,
        top,
        right,
        bottom,
        [(0.14, "14%"), (0.18, "18%"), (0.22, "22%"), (0.26, "26%"), (0.30, "30%"), (0.34, "34%")],
        [(0.20, "20%"), (0.25, "25%"), (0.30, "30%"), (0.35, "35%"), (0.40, "40%"), (0.45, "45%")],
        x_map,
        y_map,
        "Collision rate (lower is better)",
        "Full success (higher)",
    )
    colors = {"5a": "#3b6ea8", "b2": "#d1495b", "privileged_2m": "#5a9367", "r2b": "#7b8794", "epoch16": "#9b6a9f", "min_lidar": "#c28f3d", "ttc_cpa": "#4f9d9d"}
    offsets = {"5a": (10, 5), "b2": (10, -10), "privileged_2m": (10, -9), "r2b": (10, 15), "epoch16": (10, -8), "min_lidar": (10, 14), "ttc_cpa": (10, 17)}
    for key, label in G25_METHODS:
        item = data["pooled_descriptive"][key]
        x = x_map(item["collision"])
        y = y_map(item["full_success"])
        radius = 5 + min(10, item["raw_steps"] / 20)
        svg_circle(lines, x, y, radius, colors[key])
        dx, dy = offsets[key]
        svg_text(lines, x + dx, y + dy, f"{label} ({item['raw_steps']:.1f})", 12, fill="#28333d")
    svg_text(lines, left, height - 35, "Bubble size is proportional to raw termination steps.", 12, fill="#596570")
    return svg_close(lines)


def g25_effects(data):
    width, height = 1000, 530
    left, right = 260, 820
    panels = [
        ("Full success difference", data["primary_b2_minus_5a"]["mean_full_success_difference"], data["primary_b2_minus_5a"]["scene_cluster_bca_95_ci"], "percentage points", 0.20),
        ("Collision difference", data["secondary_b2_minus_5a"]["collision"]["mean_difference"], data["secondary_b2_minus_5a"]["collision"]["scene_cluster_bca_95_ci"], "percentage points", 0.14),
        ("Paired-success steps", data["secondary_b2_minus_5a"]["paired_success_steps"]["mean_difference"], data["secondary_b2_minus_5a"]["paired_success_steps"]["scene_cluster_bca_95_ci"], "steps", 22.0),
    ]
    lines = svg_open(width, height, "G25 PIRoute effects relative to 5A")
    svg_text(lines, width / 2, 34, "G25 sealed: PIRoute effect relative to 5A", 20, "middle", weight="700")
    for index, (label, value, interval, unit, scale) in enumerate(panels):
        center = 125 + index * 150
        svg_text(lines, 30, center + 5, label, 14, weight="600")
        x0 = left
        x1 = right
        low = -scale if index < 2 else -scale
        high = scale
        x_map = lambda item, low=low, high=high: x0 + (item - low) / (high - low) * (x1 - x0)
        svg_line(lines, x0, center, x1, center, stroke="#d3d9df", width=2)
        svg_line(lines, x_map(0), center - 28, x_map(0), center + 28, stroke="#48545f", width=1.2, dash="4 3")
        svg_line(lines, x_map(interval[0]), center, x_map(interval[1]), center, stroke="#d1495b", width=5)
        svg_circle(lines, x_map(value), center, 7, "#d1495b")
        svg_text(lines, x0, center + 40, f"-{scale:g}", 11, fill="#596570")
        svg_text(lines, x_map(0), center + 40, "0", 11, anchor="middle", fill="#596570")
        svg_text(lines, x1, center + 40, f"+{scale:g}", 11, anchor="end", fill="#596570")
        value_text = f"{value * 100:+.2f} pp" if index < 2 else f"{value:+.2f} steps"
        interval_text = f"[{interval[0] * 100:+.2f}, {interval[1] * 100:+.2f}] pp" if index < 2 else f"[{interval[0]:+.2f}, {interval[1]:+.2f}]"
        svg_text(lines, right + 12, center - 5, value_text, 13, weight="600")
        svg_text(lines, right + 12, center + 14, interval_text, 11, fill="#596570")
    svg_text(lines, left, height - 28, "Point = mean paired difference; red line = scene-cluster BCa 95% CI.", 12, fill="#596570")
    return svg_close(lines)


def g26_table(q1, e1):
    fields = (("Supplement", "supplement"), ("Setting / method", "setting"), ("Full success", "full_success"), ("Agent success", "agent_success"), ("Collision", "collision"), ("Timeout", "timeout"), ("Raw steps", "raw_steps"), ("Interaction share", "interaction_share"))
    rows = []
    for count in ("3", "7"):
        for key, label in (("5a", "5A"), ("b2", "B2/PIRoute")):
            item = q1["vehicle_counts"][count]["pooled_descriptive"][key]
            rows.append({"supplement": "Q1", "setting": f"{count} robots / {label}", "full_success": pct(item["full_success"]), "agent_success": pct(item["agent_success"]), "collision": pct(item["collision"]), "timeout": pct(item["timeout"]), "raw_steps": num(item["raw_steps"]), "interaction_share": pct(item["interaction_share"])})
    for key, label in (("5a", "5A"), ("nf_switch", "NF-inspired"), ("b2", "B2/PIRoute")):
        item = e1["pooled_descriptive"][key]
        rows.append({"supplement": "E1", "setting": f"5 robots / {label}", "full_success": pct(item["full_success"]), "agent_success": pct(item["agent_success"]), "collision": pct(item["collision"]), "timeout": pct(item["timeout"]), "raw_steps": num(item["raw_steps"]), "interaction_share": pct(item["interaction_share"])})
    return fields, rows


def g26_effects(e1):
    width, height = 1000, 330
    left, right = 270, 880
    lines = svg_open(width, height, "G26 E1 supplemental effects relative to 5A")
    svg_text(lines, width / 2, 34, "G26-E1 exploratory effects relative to 5A", 20, "middle", weight="700")
    rows = (("NF-inspired", e1["comparisons"]["nf_switch_minus_5a"]["full_success"]), ("B2/PIRoute", e1["comparisons"]["b2_minus_5a"]["full_success"]))
    x_map = lambda item: left + (item + 0.10) / 0.25 * (right - left)
    for index, (label, item) in enumerate(rows):
        center = 120 + index * 105
        svg_text(lines, 35, center + 5, label, 14, weight="600")
        svg_line(lines, left, center, right, center, stroke="#d3d9df", width=2)
        svg_line(lines, x_map(0), center - 28, x_map(0), center + 28, stroke="#48545f", width=1.2, dash="4 3")
        interval = item["scene_cluster_bca_95_ci"]
        svg_line(lines, x_map(interval[0]), center, x_map(interval[1]), center, stroke="#3b6ea8", width=5)
        svg_circle(lines, x_map(item["mean_difference"]), center, 7, "#3b6ea8")
        svg_text(lines, right + 12, center - 5, f"{item['mean_difference'] * 100:+.2f} pp", 13, weight="600")
        svg_text(lines, right + 12, center + 14, f"[{interval[0] * 100:+.2f}, {interval[1] * 100:+.2f}] pp", 11, fill="#596570")
    for tick in (-0.10, -0.05, 0, 0.05, 0.10, 0.15):
        x = x_map(tick)
        svg_line(lines, x, 235, x, 241, stroke="#34404b")
        svg_text(lines, x, 262, f"{tick * 100:+.0f}%", 12, anchor="middle", fill="#4d5965")
    svg_text(lines, (left + right) / 2, 300, "Full-success difference (percentage points)", 14, anchor="middle", weight="600")
    return svg_close(lines)


def piroute_overview():
    """Conceptual overview of the training/deployment information boundary."""
    width, height = 1200, 640
    lines = svg_open(width, height, "PIRoute training and deployment overview")
    svg_text(lines, width / 2, 38, "PIRoute: privileged supervision, frozen policies, local-observation routing", 21, "middle", weight="700")

    # Three stages emphasize the information boundary rather than model novelty.
    stages = [
        (40, 92, 315, 470, "Training-time supervision", "#f4f7fb"),
        (442, 92, 315, 470, "Frozen policy library", "#f7f8f5"),
        (844, 92, 315, 470, "Deployment", "#fbf6f4"),
    ]
    for x, y, w, h, title, fill in stages:
        svg_rect(lines, x, y, w, h, fill, stroke="#c6ced6", radius=8)
        svg_text(lines, x + 18, y + 32, title, 17, weight="700")

    # Training column.
    svg_rect(lines, 68, 150, 258, 86, "#e2edf8", stroke="#5f86ad", radius=5)
    svg_text(lines, 197, 181, "Gazebo robot positions", 15, "middle", weight="600")
    svg_text(lines, 197, 207, "privileged labels only", 13, "middle", fill="#4d5965")
    svg_rect(lines, 68, 278, 258, 96, "#edf3fb", stroke="#5f86ad", radius=5)
    svg_text(lines, 197, 309, "G0 LiDAR soft evidence", 15, "middle", weight="600")
    svg_text(lines, 197, 335, "candidate shape + centre", 13, "middle", fill="#4d5965")
    svg_rect(lines, 68, 416, 258, 96, "#edf3fb", stroke="#5f86ad", radius=5)
    svg_text(lines, 197, 447, "2 m distance label", 15, "middle", weight="600")
    svg_text(lines, 197, 473, "interaction-state target", 13, "middle", fill="#4d5965")
    svg_line(lines, 197, 236, 197, 278, stroke="#5f86ad", width=1.8)
    svg_line(lines, 197, 374, 197, 416, stroke="#5f86ad", width=1.8)

    # Policy library column.
    svg_rect(lines, 470, 164, 124, 120, "#e6f0e7", stroke="#6d9877", radius=5)
    svg_text(lines, 532, 203, "Actor N", 16, "middle", weight="700")
    svg_text(lines, 532, 230, "general", 13, "middle", fill="#4d5965")
    svg_text(lines, 532, 253, "navigation", 13, "middle", fill="#4d5965")
    svg_rect(lines, 606, 164, 124, 120, "#f4e9df", stroke="#b4774d", radius=5)
    svg_text(lines, 668, 203, "Actor I", 16, "middle", weight="700")
    svg_text(lines, 668, 230, "conditional", 13, "middle", fill="#4d5965")
    svg_text(lines, 668, 253, "avoidance", 13, "middle", fill="#4d5965")
    svg_text(lines, 600, 326, "freeze both actors", 14, "middle", weight="600")
    svg_line(lines, 532, 284, 532, 360, stroke="#6d9877", width=1.8)
    svg_line(lines, 668, 284, 668, 360, stroke="#b4774d", width=1.8)
    svg_rect(lines, 500, 360, 200, 108, "#f2edf8", stroke="#8c6eaa", radius=5)
    svg_text(lines, 600, 397, "Temporal Router", 16, "middle", weight="700")
    svg_text(lines, 600, 424, "8-frame GRU + hysteresis", 13, "middle", fill="#4d5965")
    svg_text(lines, 600, 447, "hard policy selection", 13, "middle", fill="#4d5965")
    svg_line(lines, 700, 414, 844, 414, stroke="#8c6eaa", width=2)
    svg_text(lines, 772, 395, "trained once", 12, "middle", fill="#596570")

    # Deployment column.
    svg_rect(lines, 870, 148, 264, 98, "#e8f1f8", stroke="#5f86ad", radius=5)
    svg_text(lines, 1002, 180, "Local observation only", 15, "middle", weight="600")
    svg_text(lines, 1002, 207, "LiDAR + motion + history", 13, "middle", fill="#4d5965")
    svg_rect(lines, 870, 284, 264, 104, "#f2edf8", stroke="#8c6eaa", radius=5)
    svg_text(lines, 1002, 316, "Router chooses", 15, "middle", weight="600")
    svg_text(lines, 1002, 343, "Actor N or Actor I", 14, "middle", fill="#4d5965")
    svg_rect(lines, 870, 428, 264, 94, "#f7f7f7", stroke="#8b949e", radius=5)
    svg_text(lines, 1002, 460, "No communication", 15, "middle", weight="600")
    svg_text(lines, 1002, 486, "no robot ground truth", 13, "middle", fill="#4d5965")
    svg_line(lines, 1002, 246, 1002, 284, stroke="#5f86ad", width=1.8)
    svg_line(lines, 1002, 388, 1002, 428, stroke="#8c6eaa", width=1.8)

    # Boundary annotations.
    svg_line(lines, 326, 198, 442, 198, stroke="#c86b5e", width=1.6, dash="6 5")
    svg_text(lines, 384, 180, "labels removed", 12, "middle", fill="#a45045")
    svg_text(lines, 600, 590, "Dashed boundary: privileged simulator state is used for supervision, then removed before deployment.", 13, "middle", fill="#596570")
    return svg_close(lines)


def main():
    for path in (G25_PATH, Q1_PATH, E1_PATH):
        if not path.is_file():
            raise SystemExit(f"missing frozen statistics: {path}")
    g25 = json.loads(G25_PATH.read_text(encoding="utf-8"))
    q1 = json.loads(Q1_PATH.read_text(encoding="utf-8"))
    e1 = json.loads(E1_PATH.read_text(encoding="utf-8"))
    OUTPUT.mkdir(parents=True, exist_ok=True)

    g25_fields, g25_rows = g25_table(g25)
    write_csv(OUTPUT / "g25_main_table.csv", g25_fields, g25_rows)
    write_markdown_table(
        OUTPUT / "g25_main_table.md",
        "G25 Sealed Main Table",
        g25_fields,
        g25_rows,
        "Source: frozen G25 sealed statistics; 256 scenes, 3 repeats, 5 robots. Development and exploratory results are excluded.",
    )
    write_text(OUTPUT / "g25_pareto.svg", g25_pareto(g25))
    write_text(OUTPUT / "g25_primary_effects.svg", g25_effects(g25))
    write_text(OUTPUT / "piroute_overview.svg", piroute_overview())

    g26_fields, g26_rows = g26_table(q1, e1)
    write_csv(OUTPUT / "g26_supplement_table.csv", g26_fields, g26_rows)
    write_markdown_table(
        OUTPUT / "g26_supplement_table.md",
        "G26 Supplementary Table",
        g26_fields,
        g26_rows,
        "Q1 and E1 are exploratory supplements. E1 is a local literature-inspired baseline, not a strict reproduction of the IROS 2024 system.",
    )
    write_text(OUTPUT / "g26_e1_effects.svg", g26_effects(e1))

    inputs = {str(path.relative_to(ROOT)): sha256(path) for path in (G25_PATH, Q1_PATH, E1_PATH)}
    outputs = {}
    for path in sorted(OUTPUT.iterdir()):
        if path.name != "generation_record.json":
            outputs[path.name] = sha256(path)
    record = {
        "generator": "scripts/generate_paper_artifacts.py",
        "sources": inputs,
        "outputs": outputs,
        "notes": [
            "G25 is the only confirmatory result set.",
            "Q1 and E1 are exploratory supplements and must not modify G25 inference.",
            "Trajectory timelines require frame-level logs and are not generated from episode-level statistics.",
        ],
    }
    write_text(OUTPUT / "generation_record.json", json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
