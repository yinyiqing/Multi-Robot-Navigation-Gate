#!/usr/bin/env python3
import json,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"experiments/03_保留专门化/02_论文主线"
NEW=BASE/"24_Dense256_epoch16_A1复测/local_data"
G18=BASE/"18_dense256当前方法复测/local_data/results"
OLD=BASE/"12_参数匹配单Actor容量对照/local_data/dense_first256_pilot/results"
PATHS={
 "5a":G18/"g18_dense256_5a_s20260810.npy",
 "epoch17_f_a1":G18/"g18_dense256_f_a1_s20260810.npy",
 "epoch17_rule_2m":G18/"g18_dense256_rule_2m_s20260810.npy",
 "epoch16_a1":NEW/"results/dense256_epoch16_a1_s20260810.npy",
 "epoch16_b2":OLD/"g12_dense256_b2_r1_s20260810.npy",
 "epoch16_rule_2m":OLD/"g12_dense256_oracle_r1_s20260810.npy"}
def metrics(x):
 n=len(x); total=n*5
 return {"full_success_rate":float(x[:,8].astype(int).mean()),"agent_success_rate":float(x[:,6].astype(int).sum()/total),"collision_rate":float(x[:,7].astype(int).sum()/total),"unresolved_rate":float(x[:,10].astype(int).sum()/total),"timeout_episode_rate":float(x[:,11].astype(int).mean()),"mean_episode_steps":float(x[:,3].astype(float).mean()),"mean_interaction_share":float(x[:,14].astype(float).mean())}
def paired(a,b):
 x,y=a[:,8].astype(int),b[:,8].astype(int); i,d=int((x>y).sum()),int((x<y).sum()); n=i+d
 p=min(1.,2*sum(math.comb(n,k) for k in range(min(i,d)+1))/2**n) if n else 1.
 return {"improved":i,"degraded":d,"tied":len(x)-n,"mcnemar_exact_p":p}
def main():
 d={k:np.load(v,allow_pickle=True) for k,v in PATHS.items()}; ids=[str(v) for v in d["5a"][:,12]]
 for k,x in d.items():
  if x.shape!=(256,17) or [str(v) for v in x[:,12]]!=ids: raise SystemExit(f"unaligned {k}")
 s={"metrics":{k:metrics(x) for k,x in d.items()},"paired":{"epoch16_a1_vs_5a":paired(d["epoch16_a1"],d["5a"]),"epoch16_a1_vs_epoch17_f_a1":paired(d["epoch16_a1"],d["epoch17_f_a1"]),"epoch16_a1_vs_epoch16_b2":paired(d["epoch16_a1"],d["epoch16_b2"]),"epoch16_a1_vs_epoch16_rule":paired(d["epoch16_a1"],d["epoch16_rule_2m"])}}
 (NEW/"summary.json").write_text(json.dumps(s,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
 print(json.dumps(s,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
