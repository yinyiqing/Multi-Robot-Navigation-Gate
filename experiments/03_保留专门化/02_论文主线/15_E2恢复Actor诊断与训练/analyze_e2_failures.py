#!/usr/bin/env python3
"""Diagnose E2 failure modes from archived admission logs."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


EPISODE_RE = re.compile(
    r"Episode (?P<episode>\d+) complete \| case=(?P<case>\S+) "
    r"\| env_steps=(?P<env_steps>\d+) \| agent_samples=(?P<agent_samples>\d+) "
    r"\| episode_env_steps=(?P<episode_env_steps>\d+) "
    r"\| episode_agent_samples=(?P<episode_agent_samples>\d+) "
    r"\| mean_reward=(?P<mean_reward>-?\d+(?:\.\d+)?) "
    r"\| success=(?P<success>\d+)/(?P<agents>\d+) "
    r"\| collision=(?P<collision>\d+)/(?P=agents) "
    r"\| unresolved=(?P<unresolved>\d+)/(?P=agents) "
    r"\| full_success=(?P<full_success>[01]) "
    r"\| timeout=(?P<timeout>[01]) "
    r"\| mean_final_distance=(?P<mean_final_distance>-?\d+(?:\.\d+)?)"
    r"(?: \| dense_action_share=(?P<dense_action_share>-?\d+(?:\.\d+)?))?"
)


def load_manifest(path: Path) -> dict[str, dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    scenarios = payload["scenarios"]
    return {case["scenario_id"]: case for case in scenarios}


def parse_log(path: Path, policy: str) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = EPISODE_RE.search(line)
        if not match:
            continue
        item = match.groupdict()
        item["policy"] = policy
        for key in (
            "episode",
            "env_steps",
            "agent_samples",
            "episode_env_steps",
            "episode_agent_samples",
            "success",
            "agents",
            "collision",
            "unresolved",
            "full_success",
            "timeout",
        ):
            item[key] = int(item[key])
        for key in ("mean_reward", "mean_final_distance"):
            item[key] = float(item[key])
        item["dense_action_share"] = (
            float(item["dense_action_share"])
            if item.get("dense_action_share") is not None
            else 0.0
        )
        rows.append(item)
    return rows


def failure_type(row: dict) -> str:
    if row["full_success"]:
        return "success"
    flags = []
    if row["collision"]:
        flags.append("collision")
    if row["timeout"]:
        flags.append("timeout")
    if row["unresolved"]:
        flags.append("unresolved")
    return "+".join(flags) if flags else "partial_success"


def summarize(rows: list[dict], label: str) -> dict:
    total = len(rows)
    agents = sum(row["agents"] for row in rows)
    success = sum(row["success"] for row in rows)
    collision = sum(row["collision"] for row in rows)
    unresolved = sum(row["unresolved"] for row in rows)
    full = sum(row["full_success"] for row in rows)
    timeout = sum(row["timeout"] for row in rows)
    return {
        "label": label,
        "episodes": total,
        "agent_success_rate": success / agents if agents else 0.0,
        "collision_rate": collision / agents if agents else 0.0,
        "unresolved_rate": unresolved / agents if agents else 0.0,
        "full_success_rate": full / total if total else 0.0,
        "timeout_episode_rate": timeout / total if total else 0.0,
        "mean_episode_steps": (
            sum(row["episode_env_steps"] for row in rows) / total if total else 0.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--e2-log", required=True, type=Path)
    parser.add_argument("--oracle-log", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(args.manifest)
    e2_rows = parse_log(args.e2_log, "e2")
    oracle_rows = parse_log(args.oracle_log, "e2_oracle_epoch16")

    if len(e2_rows) != len(oracle_rows):
        raise SystemExit(
            f"episode count mismatch: e2={len(e2_rows)} oracle={len(oracle_rows)}"
        )

    by_case = {}
    for row in e2_rows + oracle_rows:
        case = row["case"]
        if case not in manifest:
            raise SystemExit(f"case missing from manifest: {case}")
        meta = manifest[case]
        view = meta.get("view", {})
        row["pool"] = view.get("capacity_pool", meta.get("preset", "unknown"))
        row["topology"] = view.get("capacity_topology", "unknown")
        row["conflict_edge_count"] = int(
            meta.get("metrics", {}).get("conflict_edge_count", -1)
        )
        row["max_conflict_degree"] = meta.get("metrics", {}).get(
            "max_conflict_degree", None
        )
        row["failure_type"] = failure_type(row)
        by_case.setdefault(case, {})[row["policy"]] = row

    paired = []
    for case, policies in by_case.items():
        if "e2" not in policies or "e2_oracle_epoch16" not in policies:
            raise SystemExit(f"unpaired case: {case}")
        e2 = policies["e2"]
        oracle = policies["e2_oracle_epoch16"]
        delta = oracle["full_success"] - e2["full_success"]
        if delta > 0:
            relation = "oracle_improved"
        elif delta < 0:
            relation = "oracle_degraded"
        else:
            relation = "same"
        paired.append(
            {
                "episode": e2["episode"],
                "case": case,
                "pool": e2["pool"],
                "topology": e2["topology"],
                "conflict_edge_count": e2["conflict_edge_count"],
                "e2_full_success": e2["full_success"],
                "e2_success": e2["success"],
                "e2_collision": e2["collision"],
                "e2_unresolved": e2["unresolved"],
                "e2_timeout": e2["timeout"],
                "e2_steps": e2["episode_env_steps"],
                "e2_failure_type": e2["failure_type"],
                "oracle_full_success": oracle["full_success"],
                "oracle_success": oracle["success"],
                "oracle_collision": oracle["collision"],
                "oracle_unresolved": oracle["unresolved"],
                "oracle_timeout": oracle["timeout"],
                "oracle_steps": oracle["episode_env_steps"],
                "oracle_failure_type": oracle["failure_type"],
                "oracle_dense_action_share": oracle["dense_action_share"],
                "relation": relation,
            }
        )

    paired.sort(key=lambda row: row["episode"])
    fieldnames = list(paired[0].keys()) if paired else []
    with (args.out_dir / "paired_cases.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(paired)

    e2_failures = [row for row in paired if not row["e2_full_success"]]
    oracle_degraded = [row for row in paired if row["relation"] == "oracle_degraded"]
    oracle_improved = [row for row in paired if row["relation"] == "oracle_improved"]

    def grouped_counts(rows: list[dict], key: str) -> dict:
        return dict(Counter(row[key] for row in rows))

    def summarize_by(rows: list[dict], key: str, policy: str) -> dict:
        groups = defaultdict(list)
        for row in rows:
            groups[row[key]].append(
                {
                    "agents": 5,
                    "success": row[f"{policy}_success"],
                    "collision": row[f"{policy}_collision"],
                    "unresolved": row[f"{policy}_unresolved"],
                    "full_success": row[f"{policy}_full_success"],
                    "timeout": row[f"{policy}_timeout"],
                    "episode_env_steps": row[f"{policy}_steps"],
                }
            )
        return {name: summarize(items, name) for name, items in sorted(groups.items())}

    report = {
        "inputs": {
            "manifest": str(args.manifest),
            "e2_log": str(args.e2_log),
            "oracle_log": str(args.oracle_log),
        },
        "overall": {
            "e2": summarize(
                [
                    {
                        "agents": 5,
                        "success": row["e2_success"],
                        "collision": row["e2_collision"],
                        "unresolved": row["e2_unresolved"],
                        "full_success": row["e2_full_success"],
                        "timeout": row["e2_timeout"],
                        "episode_env_steps": row["e2_steps"],
                    }
                    for row in paired
                ],
                "e2",
            ),
            "e2_oracle_epoch16": summarize(
                [
                    {
                        "agents": 5,
                        "success": row["oracle_success"],
                        "collision": row["oracle_collision"],
                        "unresolved": row["oracle_unresolved"],
                        "full_success": row["oracle_full_success"],
                        "timeout": row["oracle_timeout"],
                        "episode_env_steps": row["oracle_steps"],
                    }
                    for row in paired
                ],
                "e2_oracle_epoch16",
            ),
        },
        "e2_failure_cases": len(e2_failures),
        "e2_failure_by_type": grouped_counts(e2_failures, "e2_failure_type"),
        "e2_failure_by_pool": grouped_counts(e2_failures, "pool"),
        "e2_failure_by_topology": grouped_counts(e2_failures, "topology"),
        "e2_metrics_by_pool": summarize_by(paired, "pool", "e2"),
        "e2_metrics_by_topology": summarize_by(paired, "topology", "e2"),
        "oracle_relation_counts": grouped_counts(paired, "relation"),
        "oracle_improved_cases": len(oracle_improved),
        "oracle_degraded_cases": len(oracle_degraded),
        "oracle_degraded_by_type": grouped_counts(oracle_degraded, "e2_failure_type"),
        "oracle_degraded_by_topology": grouped_counts(oracle_degraded, "topology"),
        "oracle_mean_dense_action_share": (
            sum(row["oracle_dense_action_share"] for row in paired) / len(paired)
            if paired
            else 0.0
        ),
        "oracle_mean_dense_action_share_on_degraded": (
            sum(row["oracle_dense_action_share"] for row in oracle_degraded)
            / len(oracle_degraded)
            if oracle_degraded
            else 0.0
        ),
        "oracle_mean_dense_action_share_on_improved": (
            sum(row["oracle_dense_action_share"] for row in oracle_improved)
            / len(oracle_improved)
            if oracle_improved
            else 0.0
        ),
        "candidate_recovery_cases": [
            row
            for row in paired
            if (
                not row["e2_full_success"]
                and (
                    row["e2_timeout"]
                    or row["e2_unresolved"]
                    or row["e2_steps"] >= 120
                    or row["topology"] == "multi"
                )
            )
        ],
    }
    with (args.out_dir / "diagnosis_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
