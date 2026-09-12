#!/usr/bin/env python3
"""Promote the audited G27-P4 result into the manuscript.

The script is deliberately fail-closed: it edits the manuscript only when the
independent test is complete, all preregistered confirmation criteria pass, and
the two P4 figure PDFs exist in the Overleaf directory.  Until then the paper
continues to describe B2 as predecessor evidence.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
G27 = ROOT / "experiments/03_保留专门化/02_论文主线/27_反事实Reward增强Gate监督"
PAPER = ROOT / "paper/icra2027_template/piroute_overleaf_fonts_fixed_20260908"
STATS = G27 / "local_data/test/p4_statistics.json"
MANUSCRIPT = PAPER / "main.tex"
FIGURES = PAPER / "generated/g27_p4"


def pct(value, digits=2):
    return f"{100.0 * float(value):.{digits}f}\%"


def pp(value, digits=2, sign=False):
    number = 100.0 * float(value)
    return f"{number:+.{digits}f}" if sign else f"{number:.{digits}f}"


def interval(values, digits=2, unit="pp"):
    low, high = values
    if unit == "pp":
        return f"[{pp(low, digits, True)}, {pp(high, digits, True)}]~\text{{pp}}"
    return f"[{float(low):+.2f}, {float(high):+.2f}]~\text{{steps}}"


def load_and_validate():
    if not STATS.is_file():
        raise SystemExit("G27-P4 statistics are not available")
    data = json.loads(STATS.read_text(encoding="utf-8"))
    protocol = data.get("protocol", {})
    if (
        protocol.get("experiment_id") != "G27-P4-independent-test"
        or tuple(protocol.get("methods", ())) != ("5a", "b2", "b3")
        or protocol.get("scene_clusters") != 256
        or protocol.get("total_episodes") != 2304
        or protocol.get("sealed_test_read") is not True
        or protocol.get("actor_or_router_updated") is not False
    ):
        raise SystemExit("G27-P4 protocol guard failed")
    if data.get("confirmation_passed") is not True:
        raise SystemExit("G27-P4 did not pass all preregistered replacement criteria")
    for name in ("g27_p4_outcomes.pdf", "g27_p4_effects.pdf"):
        if not (FIGURES / name).is_file():
            raise SystemExit("missing audited P4 figure: %s" % (FIGURES / name))
    return data


def replace_once(text, pattern, replacement, label):
    updated, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE | re.DOTALL)
    if count != 1:
        raise SystemExit("could not replace manuscript block: %s" % label)
    return updated


def main():
    data = load_and_validate()
    text = MANUSCRIPT.read_text(encoding="utf-8")
    pooled = data["pooled_descriptive"]
    comparisons = data["comparisons"]
    b35 = comparisons["b3_minus_5a"]
    b32 = comparisons["b3_minus_b2"]

    abstract = (
        "In communication-free shared spaces, multiple robots must navigate using only onboard observations while responding to interactions that emerge and disappear during execution. A single policy must balance goal progress against conservative collision avoidance, and a proximity-only routing target does not directly indicate which available policy will produce the better action. We present \\method{}, an online framework that freezes a general navigation actor and an interaction actor and trains only a temporal router. The router uses each robot's local LiDAR, motion state, short observation history, and the two candidate actions. Its training signal has two parts: simulator positions define an interaction-phase label, while paired one-step branches execute the two candidate actions from the same replayed state and score them with one common task reward. A shared GRU predicts both interaction probability and relative one-step advantage, which are combined before stable hard selection. Deployment uses neither simulator states nor reward values and requires no inter-robot communication. On the independent G27-P4 test, reward-aware routing improves full-team success over 5A by "
        + pp(b35["full_success"]["mean_difference"], sign=True)
        + "~percentage points (scene-cluster BCa 95\\% CI "
        + interval(b35["full_success"]["scene_cluster_bca_95_ci"])
        + "), while robot-level collision changes by "
        + pp(b35["robot_collision"]["mean_difference"], sign=True)
        + "~points (BCa 95\\% CI "
        + interval(b35["robot_collision"]["scene_cluster_bca_95_ci"])
        + ")."
    )
    text = replace_once(
        text,
        r"(?<=\\begin\{abstract\}\n).*?(?=\n% Insert independent-test effect sizes only after development admission and the new sealed evaluation\.)",
        abstract,
        "abstract",
    )

    b3 = pooled["b3"]
    b2 = pooled["b2"]
    five_a = pooled["5a"]
    p_value = b35["full_success"]["sign_flip_two_sided_p"]
    confirmatory = (
        "The independent G27-P4 evaluation uses the previously unread dense-test slice and compares 5A, the proximity-only B2 predecessor, and reward-aware B3 over 768 episodes per method. "
        "Reward-aware B3 reaches "
        + pct(b3["full_success"])
        + " full-team success, compared with "
        + pct(five_a["full_success"])
        + " for 5A and "
        + pct(b2["full_success"])
        + " for B2. Its difference from 5A is "
        + pp(b35["full_success"]["mean_difference"], sign=True)
        + " percentage points (scene-cluster BCa 95\\% CI "
        + interval(b35["full_success"]["scene_cluster_bca_95_ci"])
        + "; two-sided sign-flip $p="
        + f"{p_value:.5f}"
        + "), and its robot-level collision difference is "
        + pp(b35["robot_collision"]["mean_difference"], sign=True)
        + " points (BCa 95\\% CI "
        + interval(b35["robot_collision"]["scene_cluster_bca_95_ci"])
        + "). Relative to B2, B3 changes full success by "
        + pp(b32["full_success"]["mean_difference"], sign=True)
        + " points, collision by "
        + pp(b32["robot_collision"]["mean_difference"], sign=True)
        + " points, and timeout by "
        + pp(b32["episode_timeout"]["mean_difference"], sign=True)
        + " points. All five preregistered success, safety, and replacement criteria pass; completion steps and interaction-policy selection are reported as operating costs."
    )
    table_rows = []
    for key, label in (("5a", "5A"), ("b2", "Proximity B2"), ("b3", "Reward-aware B3")):
        item = pooled[key]
        table_rows.append(
            "%s & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f \\\\" % (
                label,
                100 * float(item["full_success"]),
                100 * float(item["agent_success"]),
                100 * float(item["robot_collision"]),
                100 * float(item["robot_unresolved"]),
                100 * float(item["episode_timeout"]),
                float(item["raw_steps"]),
            )
        )
    main_block = (
        "\\subsection{Confirmatory Reward-Aware Result}\n"
        + confirmatory
        + "\n\n"
        + "\\begin{table}[t]\n"
        + "\\caption{G27-P4 independent test: 768 episodes per method. Full/timeout are episode rates; agent/collision/unresolved are robot rates. All entries are percentages except raw termination steps.}\n"
        + "\\label{tab:g27}\n\\centering\n\\footnotesize\n\\setlength{\\tabcolsep}{3pt}\n"
        + "\\begin{tabular}{lrrrrrr}\n\\toprule\nMethod & Full & Agent & Coll. & Unres. & Timeout & Steps\\\\\n\\midrule\n"
        + "\n".join(table_rows)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n\n"
        + "\\begin{figure*}[t]\n  \\centering\n  \\includegraphics[width=0.96\\textwidth]{generated/g27_p4/g27_p4_outcomes.pdf}\n"
        + "  \\caption{G27-P4 outcome rates for 5A, the proximity-only B2 predecessor, and reward-aware B3. Bars show pooled descriptive rates; the axes start at zero.}\n"
        + "  \\label{fig:p4outcomes}\n\\end{figure*}\n\n"
        + "\\begin{figure*}[t]\n  \\centering\n  \\includegraphics[width=0.96\\textwidth]{generated/g27_p4/g27_p4_effects.pdf}\n"
        + "  \\caption{G27-P4 paired effects. Points and horizontal intervals show scene-cluster BCa 95\\% confidence intervals for B3 relative to 5A and B2; zero is the no-difference reference.}\n"
        + "  \\label{fig:p4effects}\n\\end{figure*}\n\n"
        + "\\subsection{Predecessor Evidence for Frozen Policy Reuse}\n"
        + "The earlier G25 sealed evaluation established the predecessor operating point: proximity-only B2 raised full-team success from 24.61\\% to 38.54\\% and reduced robot-level collision from 31.72\\% to 22.47\\% on a separate dense-test slice. Those values are retained as historical evidence for the frozen policy pair and are not pooled with G27-P4 or used to select B3.\n\n"
        + "\\subsection{Completion Reliability and Time}\n"
        + "On the G27-P4 matched successful episode pairs, B3 changes paired-success steps relative to 5A by "
        + f"{b35['paired_success_steps']['mean_difference']:+.2f} steps (BCa 95\\% CI {interval(b35['paired_success_steps']['scene_cluster_bca_95_ci'], unit='steps')}). "
        + "Its pooled raw termination steps are "
        + f"{five_a['raw_steps']:.2f}, {b2['raw_steps']:.2f}, and {b3['raw_steps']:.2f} for 5A, B2, and B3, respectively. "
        + "The interaction-actor selection share for B3 is "
        + pct(b3["interaction_selection_share"])
        + "; this is a selection share, not a forward-call rate, because both actors are evaluated at routing instants.\n\n"
        + "\\subsection{Why Conditional Policy Reuse Matters}\n"
        + "The three-way outcome comparison in Fig.~\\ref{fig:p4outcomes} separates the revised supervision from the inherited proximity-only router. B3 is evaluated under the same frozen actors, perception stack, stride, thresholds, and termination rules as B2, so the B2--B3 contrast isolates the added paired-reward supervision within this protocol. The earlier always-on, local-rule, capacity, and privileged-switch results remain supporting evidence rather than components of the G27-P4 confirmatory comparison."
    )
    text = replace_once(
        text,
        r"\\subsection\{Predecessor Evidence for Frozen Policy Reuse\}.*?(?=\\subsection\{Development Evidence and Router Design\})",
        main_block + "\n\n",
        "results and predecessor block",
    )

    discussion = (
        "The independent test supports reward-aware routing as a refinement of frozen policy reuse. Under the fixed local-observation protocol, B3 improves team completion over 5A while satisfying the prespecified collision and replacement constraints relative to B2. The result does not show that the one-step reward target is a calibrated long-horizon value estimate; rather, it shows that combining this target with the interaction-phase target can improve the closed-loop operating point of the fixed policy library. Completion steps and interaction-policy selection remain explicit costs."
    )
    text = replace_once(
        text,
        r"The predecessor results identify policy selection as a useful adaptation mechanism.*?(?=\n\nThe current scope is simulation)",
        discussion,
        "discussion",
    )
    conclusion = (
        "We presented \\method{}, a reward-aware framework that reuses two frozen policies through per-robot local routing. A temporal Router combines local motion and candidate-action evidence with two training targets: a privileged interaction-phase label and the paired one-step reward difference between the frozen policies. Deployment uses neither supervision signal and requires no communication. On the independent G27-P4 test, this revised supervision improves full-team completion over 5A while preserving the prespecified collision and replacement criteria relative to the proximity-only predecessor; completion steps and interaction-policy selection remain operating costs."
    )
    text = replace_once(
        text,
        r"We presented \\method\{\}, a reward-aware framework.*?(?=\n\n\\IEEEtriggeratref)",
        conclusion,
        "conclusion",
    )
    MANUSCRIPT.write_text(text, encoding="utf-8")
    print("Promoted audited G27-P4 results into %s" % MANUSCRIPT)


if __name__ == "__main__":
    main()
