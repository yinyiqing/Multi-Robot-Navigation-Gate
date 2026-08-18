#!/usr/bin/env python3
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/03_保留专门化/02_论文主线"
GATE = BASE / "11_可部署在线Gate研究"
G25 = BASE / "25_最终消融与Sealed评测/local_data"


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def measured_elapsed(pattern):
    paths = sorted(ROOT.glob(pattern))
    if len(paths) != 1:
        raise ValueError("expected one timing log for %s, found %d" % (pattern, len(paths)))
    text = paths[0].read_text(encoding="utf-8", errors="replace")
    matches = re.findall(r"elapsed=([0-9]+):([0-9]+(?:\.[0-9]+)?)", text)
    if len(matches) != 1:
        raise ValueError("expected one elapsed record in %s" % paths[0])
    minutes, seconds = matches[0]
    return {
        "seconds": int(minutes) * 60 + float(seconds),
        "source": str(paths[0].relative_to(ROOT)),
        "measurement": "/usr/bin/time elapsed",
    }


def router_record(name, summary_path, timing_pattern):
    summary = load_json(summary_path)
    samples = summary["samples"]
    return {
        "device": summary["protocol"]["device"],
        "epochs_budget": summary["protocol"]["epochs"],
        "train_frames": samples["train_frames"],
        "train_ego_sequences": samples["train_ego_sequences"],
        "validation_frames": samples["validation_frames"],
        "validation_ego_sequences": samples["validation_ego_sequences"],
        "wall_clock": measured_elapsed(timing_pattern),
        "summary": str(summary_path.relative_to(ROOT)),
        "label": name,
    }


def main():
    a1_summary = GATE / "G11_A1_当前协议时序pilot/local_data/training/seed20260804/summary.json"
    b2_summary = GATE / "G11_B_student_rollout_v1/local_data/training/seed20260804/summary.json"
    v4_summary = G25 / "router_ablations/v4_single_frame/summary.json"
    v5_summary = G25 / "router_ablations/v5_no_action_difference/summary.json"
    b2_audit = load_json(
        GATE / "G11_B_student_rollout_v1/local_data/train_audit.json"
    )
    g0_history = load_json(
        BASE / "results/06_Gate开发/D5_G0_robot_detector_v1/local_data/model/pilot_v1/history.json"
    )
    if len(g0_history) != 20:
        raise ValueError("unexpected G0 epoch count")
    if b2_audit["shards"] != 640 or b2_audit["totals"]["frames"] != 42899:
        raise ValueError("unexpected B1 student rollout audit")

    result = {
        "protocol": {
            "experiment_id": "G25-training-cost-audit",
            "sealed_test_read": False,
            "unknown_values": "not recorded",
            "no_retraining_performed": True,
        },
        "actors": {
            "generalist_5a": {
                "selected_lineage_agent_samples_lower_bound": 190000,
                "final_5a_stage_agent_samples": 30000,
                "selected_checkpoint_near_stage_samples": 20000,
                "full_lineage_budget": "not recorded",
                "wall_clock": "not recorded",
                "device": "not recorded",
                "note": "The lineage starts from an undocumented TD3_velodyne_multi_v4 warm start; 190k is only the recoverable added path.",
                "source": "experiments/03_保留专门化/02_论文主线/12_参数匹配单Actor容量对照/R2_CURRICULUM_AUDIT.md",
            },
            "interaction_epoch16": {
                "agent_samples": 320000,
                "training_scenarios": 2560,
                "wall_clock": "not recorded",
                "device": "not recorded",
                "source": "experiments/03_保留专门化/02_论文主线/README.md",
            },
            "r2b_process_matched_capacity_control": {
                "agent_samples": 30374,
                "nominal_budget": 30000,
                "training_scenario_mode": "five-robot procedural standard",
                "wall_clock": "not directly measured; approximately 2h03m from launcher timestamp to final log mtime",
                "device": "GPU 0",
                "source": "experiments/03_保留专门化/02_论文主线/12_参数匹配单Actor容量对照/R2B_5A_RECIPE_RESULTS.md",
            },
        },
        "g0_detector": {
            "device": "not recorded",
            "epochs": 20,
            "train_scenarios": 100,
            "validation_scenarios": 100,
            "train_frames": 1855,
            "validation_frames": 1647,
            "train_candidates": 15425,
            "validation_candidates": 15433,
            "wall_clock": "not recorded",
            "source": "experiments/03_保留专门化/02_论文主线/results/06_Gate开发/D5_G0_robot_detector_v1/PILOT_REPORT.md",
        },
        "rollout_collection": {
            "a1_frozen_5a": {
                "scenarios": 640,
                "frames": 28082,
                "wall_clock": "not recorded",
            },
            "b1_student": {
                "scenarios": b2_audit["shards"],
                "frames": b2_audit["totals"]["frames"],
                "candidates": b2_audit["totals"]["candidates"],
                "wall_clock": "not recorded",
            },
        },
        "router_training": {
            "a1": router_record(
                "A1",
                a1_summary,
                "logs/archive/diagnostic/g11_a1/training/train_g11_a1_seed20260804_*.log",
            ),
            "b2": router_record(
                "B2",
                b2_summary,
                "logs/archive/diagnostic/g11_b/training/train_g11_b2_seed20260804_*.log",
            ),
            "v4_single_frame": router_record(
                "V4 single-frame",
                v4_summary,
                "logs/archive/training/g25_router_ablations/train_v4_single_frame.log",
            ),
            "v5_no_action_difference": router_record(
                "V5 no-action-difference",
                v5_summary,
                "logs/archive/training/g25_router_ablations/train_v5_no_action_difference.log",
            ),
        },
        "claim_boundary": "The method reuses frozen Actors without new Actor updates. Total historical system cost is not proven lower because several Actor/G0 wall-clock records are unavailable and B1 requires student rollout collection.",
    }
    output = G25 / "training_cost.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
