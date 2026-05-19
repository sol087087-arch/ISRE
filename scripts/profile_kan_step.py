"""Profile the KAN per-step cost BEFORE vectorizing (encoder-lesson:
measure, don't assume the hot path).

Splits one KAN training step into:
  featurize : building the [n_cand, FEATURE_DIM] matrix
              (Python list-comp of candidate_features + torch.tensor)
  kan_fwd   : pykan KAN.forward (spline eval)
  loss_bwd  : cross-entropy + backward + opt.step
"""
import sys, time, statistics
sys.stdout.reconfigure(encoding="utf-8")

import torch
from isre.training.train import load_trajectories
from isre.learning.kan_policy import KANPolicy
from isre.learning.features import candidate_features

N = 300
steps = load_trajectories("isre/trajectories_v6_bfs", max_files=200)[:N]
kp = KANPolicy(hidden=16, device="cpu")
opt = torch.optim.Adam(kp.parameters(), lr=1e-3)

t_feat, t_fwd, t_bwd, t_tot = [], [], [], []

# warmup
for s in steps[:10]:
    kp.score(s.ast, s.candidate_actions)

for s in steps:
    cand = s.candidate_actions
    if not cand:
        continue
    opt.zero_grad()
    a = time.perf_counter()
    rows = [candidate_features(s.ast, nid, act) for (nid, act) in cand]
    feats = torch.tensor(rows, dtype=torch.float32)
    b = time.perf_counter()
    out = kp.kan(feats).squeeze(-1)
    c = time.perf_counter()
    gi = None
    for i, (nid, act) in enumerate(cand):
        if nid == s.gold_node_id and act == s.gold_action:
            gi = i; break
    if gi is None:
        continue
    loss = torch.nn.functional.cross_entropy(
        out.unsqueeze(0), torch.tensor([gi]))
    loss.backward(); opt.step()
    d = time.perf_counter()
    t_feat.append(b - a); t_fwd.append(c - b); t_bwd.append(d - c)
    t_tot.append(d - a)


def st(x):
    return f"mean={statistics.fmean(x)*1000:.2f}ms p50={statistics.median(x)*1000:.2f}ms"


tot = statistics.fmean(t_tot)
print(f"n={len(t_tot)} steps  (KAN hidden=16, cpu)\n")
print(f"  featurize : {st(t_feat)}")
print(f"  kan_fwd   : {st(t_fwd)}")
print(f"  loss_bwd  : {st(t_bwd)}")
print(f"  TOTAL/step: {st(t_tot)}")
fe, fw, bw = (statistics.fmean(z) for z in (t_feat, t_fwd, t_bwd))
s = fe + fw + bw
print(f"\n  share: featurize={fe/s:.0%}  kan_fwd={fw/s:.0%}  loss_bwd={bw/s:.0%}")
print("\nLEVER: "
      + ("featurize dominates -> vectorize candidate_features (batch all "
         "candidates: one iter_preorder + tensor ops, no Python per-cand "
         "loop). Exact-equivalent by construction."
         if fe / s > 0.5 else
         "kan_fwd/backward dominates -> the pykan spline forward is the "
         "cost; featurize vectorization won't be enough; report options."))
