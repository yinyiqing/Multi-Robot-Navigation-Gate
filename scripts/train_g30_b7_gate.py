#!/usr/bin/env python3
"""Train B7 from the full Router corpus with a small auxiliary reward head."""
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "TD3"), str(ROOT / "scripts")]
from reward_aware_gate import MultiTaskRewardTemporalGate
from temporal_interaction_gate import TemporalInteractionGate
from train_g27_reward_aware_gate import load_p1_dataset, normalize_padded_windows
from train_g11_b_aggregated_gate import (
    A1_ROUTE, B_ROUTE, VIEW_DIR, build_dataset, concatenate_sources,
    normalize_source_balanced, source_scenario_sample_weights,
)
from train_temporal_interaction_gate import (
    build_dataset as build_single_dataset, make_temporal_windows,
    load_actor, reset_random_seed,
)
from robot_perception.dataset import list_shards

BASE = ROOT / "experiments/03_保留专门化/02_论文主线"
G27 = BASE / "27_反事实Reward增强Gate监督"
G30 = BASE / "30_B6单头辅助Reward监督"
B2 = BASE / "11_可部署在线Gate研究/G11_B_student_rollout_v1/local_data/training/seed20260804/any/T1/best.pt"


def main():
    out = G30 / "local_data/training/seed20260911_b7"
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    seed = 20260911
    reset_random_seed(seed)
    p1tr = load_p1_dataset(G27 / "local_data/counterfactual/audit/train.npz")
    p1va = load_p1_dataset(G27 / "local_data/counterfactual/audit/validation.npz")
    detector = torch.load(BASE / "results/06_Gate开发/D5_G0_robot_detector_v1/local_data/model/pilot_v1/best.pt", map_location="cpu", weights_only=False)
    from robot_perception.models import LocalRobotDetector
    det = LocalRobotDetector(**detector.get("model_config", {})); det.load_state_dict(detector["model_state_dict"]); det.eval()
    nav = load_actor(ROOT / "TD3/pytorch_models/TD3_velodyne_multi_v4_curriculum_stage2_to_5a_shared_from_3d2_guarded_best_actor.pth", "cpu")
    inter = load_actor(ROOT / "TD3/pytorch_models/interaction_focused_actor_from_5a_fullstrong_balanced_formal_s20260726_epoch_016_actor.pth", "cpu")
    ns = type("Args", (), {"batch_size": 256, "max_tracks": 4, "device": "cpu"})()
    a1 = build_single_dataset(list_shards(A1_ROUTE / "local_data/shards/train"), det, nav, inter, ns, "train")
    stu = build_single_dataset(list_shards(B_ROUTE / "local_data/student_shards/train"), det, nav, inter, ns, "train")
    va = build_single_dataset(list_shards(A1_ROUTE / "local_data/shards/validation"), det, nav, inter, ns, "validation")
    train = concatenate_sources((("a1_5a", a1), ("student", stu)))
    raw = np.concatenate((train.base_features, train.actor_features), axis=1)
    rawv = np.concatenate((va.base_features, va.actor_features), axis=1)
    norm, normv, mean, std = normalize_source_balanced(raw, rawv, train.scenarios)
    win, idx = make_temporal_windows(norm, train.sequence_indices, 8)
    winv, idxv = make_temporal_windows(normv, va.sequence_indices, 8)
    labels = train.labels["any"][idx].astype(np.float32)
    weights = source_scenario_sample_weights(train.labels["any"], train.scenarios)[idx]
    # P1 anchors are auxiliary-only; use the same normalization as deployment.
    aw = normalize_padded_windows(p1tr["gate_windows"], p1tr["history_lengths"], mean, std)
    av = normalize_padded_windows(p1va["gate_windows"], p1va["history_lengths"], mean, std)
    at = (p1tr["mean_delta_r"] > 0).astype(np.float32)
    avt = (p1va["mean_delta_r"] > 0).astype(np.float32)
    model = MultiTaskRewardTemporalGate(82, 64)
    src = torch.load(B2, map_location="cpu", weights_only=False)
    old = TemporalInteractionGate(**src["model_config"]); old.load_state_dict(src["model_state_dict"])
    model.gru.load_state_dict(old.gru.state_dict()); model.interaction_head.load_state_dict(old.head.state_dict())
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    X, Y, W = torch.from_numpy(win), torch.from_numpy(labels), torch.from_numpy(weights)
    AX, AY = torch.from_numpy(aw), torch.from_numpy(at)
    for epoch in range(1, 41):
        model.train(); order = rng.permutation(len(X)); opt.zero_grad(); total = 0.0
        for start in range(0, len(order), 256):
            b = order[start:start+256]; logits, _ = model(X[b])
            loss = (F.binary_cross_entropy_with_logits(logits, Y[b], reduction="none") * W[b]).mean()
            (loss / (len(order) / 256)).backward(); total += float(loss.detach())
        _, reward_logits = model(AX)
        reliable = torch.from_numpy(np.abs(p1tr["mean_delta_r"]) > 0.005)
        aux = F.binary_cross_entropy_with_logits(reward_logits[reliable], AY[reliable])
        (0.10 * aux).backward(); opt.step()
        with torch.no_grad():
            model.eval(); vl, _ = model(torch.from_numpy(winv)); vp = torch.sigmoid(vl); acc = float(((vp >= .5) == torch.from_numpy(va.labels["any"][idxv]).bool()).float().mean())
        print(f"epoch={epoch:02d} train_bce={total/len(order):.5f} aux={float(aux):.5f} val_acc={acc:.4f}", flush=True)
    out.mkdir(parents=True)
    ck = {"format_version":1,"model_id":"B7-multitask-reward-aware","label":"interaction+auxiliary-counterfactual-reward","model_state_dict":model.state_dict(),"model_config":{"input_dim":82,"hidden_dim":64},"feature_set":"base_and_actor_actions","feature_mean":mean,"feature_std":std,"threshold":0.43,"sequence_length":8,"max_tracks":4,"router_decision":{"protocol":"G30-B7-full-data-b2-init-v1","probability_band":[0.4,0.6],"reward_logit_confidence":0.25},"training_data":{"full_train_frames":int(len(win)),"full_validation_frames":int(len(winv)),"reward_train_anchors":int(len(aw)),"reward_auxiliary_weight":0.10,"reward_noise_threshold":0.005}}
    torch.save(ck, out / "best.pt")
    (out / "summary.json").write_text(json.dumps({"protocol":"G30-B7-full-data-b2-init-v1","seed":seed,"train_frames":int(len(win)),"validation_frames":int(len(winv)),"reward_anchors":int(len(aw)),"best_checkpoint":str(out/"best.pt")}, indent=2), encoding="utf-8")

if __name__ == "__main__": main()
