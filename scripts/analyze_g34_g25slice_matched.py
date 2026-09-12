#!/usr/bin/env python3
"""Compare G34 matched results with the already sealed G25 5A results."""
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/03_保留专门化/02_论文主线"
RUN = BASE / "36_G34_G25slice_matched"
TEST = RUN / "local_data/matched"
MANIFEST = BASE / "25_最终消融与Sealed评测/local_data/sealed_manifest/dense_test_first256.json.gz"
G25_RESULTS = BASE / "25_最终消融与Sealed评测/local_data/sealed/results"
SEEDS = (20260901, 20260902, 20260903)
SCENES = 256
AGENTS = 5

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def aggregate(rows):
    return {
        "episodes": int(len(rows)),
        "full_success": float(rows[:,8].astype(float).mean()),
        "agent_success": float(rows[:,6].astype(float).sum()/(len(rows)*AGENTS)),
        "robot_collision": float(rows[:,7].astype(float).sum()/(len(rows)*AGENTS)),
        "robot_unresolved": float(rows[:,10].astype(float).sum()/(len(rows)*AGENTS)),
        "episode_timeout": float(rows[:,11].astype(float).mean()),
        "environment_steps": float(rows[:,3].astype(float).mean()),
        "interaction_selection_share": float(rows[:,14].astype(float).mean()),
        "switches_per_episode": float(rows[:,15].astype(float).mean()),
    }

def main():
    with gzip.open(MANIFEST, "rt", encoding="utf-8") as handle:
        expected=[str(item["scenario_id"]) for item in json.load(handle)["scenarios"]]
    if len(expected) != SCENES: raise ValueError("G25 manifest is not 256 scenes")
    runs={}
    hashes={}
    for seed in SEEDS:
        g34=TEST/"results"/f"g34_g25slice_s{seed}.npy"
        g25=G25_RESULTS/f"g25_sealed_5a_s{seed}.npy"
        a=np.load(g34,allow_pickle=True); b=np.load(g25,allow_pickle=True)
        for path,rows in ((g34,a),(g25,b)):
            if rows.shape != (SCENES,17): raise ValueError(f"wrong result shape: {path}")
            if [str(item) for item in rows[:,12]] != expected: raise ValueError(f"scenario order mismatch: {path}")
            if sum(int(row[6])+int(row[7])+int(row[10]) for row in rows) != SCENES*AGENTS:
                raise ValueError(f"terminal accounting mismatch: {path}")
        runs[('g34',seed)]=a; runs[('5a',seed)]=b
        hashes[f"g34_s{seed}"]=sha(g34)
        hashes[f"5a_s{seed}"]=sha(g25)
    overall={m:aggregate(np.concatenate([runs[(m,s)] for s in SEEDS])) for m in ('5a','g34')}
    diffs={}
    for name,col,scale in (("full_success",8,1.0),("agent_success",6,AGENTS),("robot_collision",7,AGENTS),("environment_steps",3,1.0),("interaction_selection_share",14,1.0),("switches_per_episode",15,1.0)):
        values=np.stack([(runs[('g34',s)][:,col].astype(float)-runs[('5a',s)][:,col].astype(float))/scale for s in SEEDS],axis=1)
        diffs[name]={"mean_scene_repeat_difference":float(values.mean()),"scene_mean_min":float(values.mean(axis=1).min()),"scene_mean_max":float(values.mean(axis=1).max())}
    output={
        "format_version":1,
        "experiment_id":"G34-G25-slice-matched-evaluation",
        "manifest_sha256":sha(MANIFEST),
        "result_sha256":hashes,
        "overall":overall,
        "g34_minus_5a":diffs,
        "interpretation_boundary":"Matched reuse of G25 slice [0:256]; [640:896] is not used.",
    }
    output["statistics_sha256"]=hashlib.sha256(json.dumps(output,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    (TEST/"statistics.json").write_text(json.dumps(output,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(output,ensure_ascii=False,indent=2))

if __name__ == "__main__": main()
