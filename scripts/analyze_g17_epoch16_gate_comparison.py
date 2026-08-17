#!/usr/bin/env python3
import json, math
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"experiments/03_保留专门化/02_论文主线"
NEW=BASE/"23_G17_epoch16同场复测/local_data/results"
G17=BASE/"17_完整场景统一对比/local_data/results"
G22=BASE/"22_G17_Gate机制对照/local_data/results"
OUT=BASE/"23_G17_epoch16同场复测/local_data/summary.json"
SEEDS=(20260824,20260825)

def load(pattern, root):
    return np.concatenate([np.load(root/pattern.format(seed=s),allow_pickle=True) for s in SEEDS])
def metrics(x):
    n=len(x); total=n*5
    return {"episodes":n,"full_success_rate":float(x[:,8].astype(int).sum()/n),
            "agent_success_rate":float(x[:,6].astype(int).sum()/total),
            "collision_rate":float(x[:,7].astype(int).sum()/total),
            "unresolved_rate":float(x[:,10].astype(int).sum()/total),
            "timeout_episode_rate":float(x[:,11].astype(int).sum()/n),
            "mean_episode_steps":float(x[:,3].astype(float).mean()),
            "mean_interaction_share":float(x[:,14].astype(float).mean()),
            "mean_gate_switches":float(x[:,15].astype(float).mean())}
def paired(a,b):
    x,y=a[:,8].astype(int),b[:,8].astype(int)
    imp,deg=int((x>y).sum()),int((x<y).sum()); n=imp+deg
    p=min(1.0,2*sum(math.comb(n,k) for k in range(min(imp,deg)+1))/2**n) if n else 1.0
    half=len(a)//2
    if len(a)%2 or [str(v) for v in a[:half,12]] != [str(v) for v in a[half:,12]]:
        raise ValueError("clustered sign-flip requires two aligned repeats")
    differences=(x-y).reshape(2,half).sum(axis=0)
    observed=abs(int(differences.sum()))
    distribution={0:1.0}
    for difference in differences:
        updated={}
        for total,probability in distribution.items():
            updated[total+int(difference)]=updated.get(total+int(difference),0.0)+probability/2.0
            updated[total-int(difference)]=updated.get(total-int(difference),0.0)+probability/2.0
        distribution=updated
    cluster_p=sum(probability for total,probability in distribution.items() if abs(total)>=observed)
    return {"improved":imp,"degraded":deg,"tied":len(x)-n,"mcnemar_exact_p":p,
            "scenario_cluster_sign_flip_p":cluster_p}
def main():
    data={
      "5a":load("g17_5a_s{seed}.npy",G17),
      "epoch17_f_a1":load("g17_a1_s{seed}.npy",G17),
      "epoch17_rule_2m":load("g17_rule_2m_privileged_s{seed}.npy",G22),
      "epoch16_a1":load("g17_epoch16_a1_s{seed}.npy",NEW),
      "epoch16_b2":load("g17_epoch16_b2_s{seed}.npy",NEW),
      "epoch16_rule_2m":load("g17_epoch16_rule_2m_s{seed}.npy",NEW),
    }
    ids=[str(x) for x in data["5a"][:,12]]
    for name,x in data.items():
        if x.shape!=(240,17) or [str(v) for v in x[:,12]]!=ids:
            raise SystemExit(f"invalid aligned result: {name}")
        if sum(int(r[6])+int(r[7])+int(r[10]) for r in x)!=1200:
            raise SystemExit(f"terminal accounting mismatch: {name}")
    summary={"protocol":{"seeds":SEEDS,"episodes_per_policy":240,"sealed_test_read":False},
             "metrics":{k:metrics(v) for k,v in data.items()},
             "paired":{
               "epoch16_a1_vs_epoch17_f_a1":paired(data["epoch16_a1"],data["epoch17_f_a1"]),
               "epoch16_b2_vs_epoch17_f_a1":paired(data["epoch16_b2"],data["epoch17_f_a1"]),
               "epoch16_b2_vs_5a":paired(data["epoch16_b2"],data["5a"]),
               "epoch16_rule_vs_epoch17_rule":paired(data["epoch16_rule_2m"],data["epoch17_rule_2m"]),
             }}
    OUT.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=="__main__": main()
