#!/usr/bin/env python3
"""Train only a reward-preference head on top of the frozen B2 Router."""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "TD3"), str(ROOT / "scripts")]
from reward_aware_gate import MultiTaskRewardTemporalGate
from train_g27_reward_aware_gate import load_p1_dataset, normalize_padded_windows

BASE = ROOT / "experiments/03_保留专门化/02_论文主线"
G27 = BASE / "27_反事实Reward增强Gate监督"
B2 = BASE / "11_可部署在线Gate研究/G11_B_student_rollout_v1/local_data/training/seed20260804/any/T1/best.pt"
OUT = BASE / "32_B8冻结B2独立RewardHead/local_data/training/seed20260911_b8"


def main():
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(OUT)
    b2 = torch.load(B2, map_location="cpu", weights_only=False)
    p1tr = load_p1_dataset(G27 / "local_data/counterfactual/audit/train.npz")
    p1va = load_p1_dataset(G27 / "local_data/counterfactual/audit/validation.npz")
    model = MultiTaskRewardTemporalGate(82, 64)
    from temporal_interaction_gate import TemporalInteractionGate
    old = TemporalInteractionGate(**b2["model_config"])
    old.load_state_dict(b2["model_state_dict"])
    model.gru.load_state_dict(old.gru.state_dict())
    model.interaction_head.load_state_dict(old.head.state_dict())
    for p in model.gru.parameters(): p.requires_grad = False
    for p in model.interaction_head.parameters(): p.requires_grad = False
    # B2 normalization is the deployment normalization and must remain fixed.
    x = normalize_padded_windows(p1tr["gate_windows"], p1tr["history_lengths"], b2["feature_mean"], b2["feature_std"])
    xv = normalize_padded_windows(p1va["gate_windows"], p1va["history_lengths"], b2["feature_mean"], b2["feature_std"])
    y = torch.from_numpy((p1tr["mean_delta_r"] > 0).astype(np.float32))
    yv = torch.from_numpy((p1va["mean_delta_r"] > 0).astype(np.float32))
    reliable = torch.from_numpy(np.abs(p1tr["mean_delta_r"]) > 0.005)
    reliable_v = torch.from_numpy(np.abs(p1va["mean_delta_r"]) > 0.005)
    opt = torch.optim.Adam(model.reward_head.parameters(), lr=1e-3, weight_decay=1e-4)
    X, XV = torch.from_numpy(x), torch.from_numpy(xv)
    for epoch in range(1, 41):
        opt.zero_grad(); _, logits = model(X); loss = F.binary_cross_entropy_with_logits(logits[reliable], y[reliable]); loss.backward(); opt.step()
        with torch.no_grad():
            model.eval(); phase, reward = model(XV); phase_b2 = old(XV); phase_delta = float(torch.max(torch.abs(phase - phase_b2)))
            reward_acc = float(((torch.sigmoid(reward[reliable_v]) >= .5) == yv[reliable_v].bool()).float().mean())
        print(f"epoch={epoch:02d} reward_bce={float(loss):.5f} reward_acc={reward_acc:.4f} phase_max_delta={phase_delta:.3g}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    ck = {"format_version":1,"model_id":"B8-frozen-b2-reward-aware","label":"frozen-b2+auxiliary-counterfactual-reward","model_state_dict":model.state_dict(),"model_config":{"input_dim":82,"hidden_dim":64},"feature_set":b2["feature_set"],"feature_mean":b2["feature_mean"],"feature_std":b2["feature_std"],"threshold":b2["threshold"],"sequence_length":b2["sequence_length"],"max_tracks":4,"router_decision":{"protocol":"G32-B8-frozen-b2-reward-head-v1","probability_band":[0.4,0.6],"reward_logit_confidence":0.25},"training_data":{"p1_train_anchors":int(len(x)),"p1_validation_anchors":int(len(xv)),"reliable_train_anchors":int(reliable.sum()),"reliable_validation_anchors":int(reliable_v.sum()),"reward_noise_threshold":0.005,"b2_checkpoint":str(B2)}}
    torch.save(ck, OUT / "best.pt")
    (OUT / "summary.json").write_text(json.dumps({"protocol":"G32-B8-frozen-b2-reward-head-v1","train_anchors":int(len(x)),"validation_anchors":int(len(xv)),"reliable_train":int(reliable.sum()),"reliable_validation":int(reliable_v.sum()),"phase_frozen":True}, indent=2), encoding="utf-8")


if __name__ == "__main__": main()
