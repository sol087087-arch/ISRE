"""Profile where per-step train time goes BEFORE optimizing.

Splits one training step into: encoder.forward / policy.forward /
loss+backward / data-to-tensor. Tells us whether the lever is #1
(within-tree per-node vectorization, exact-equivalent, low risk) or #3
(cross-tree pad collator, risky ragged refactor).
"""
import sys, time, statistics
sys.stdout.reconfigure(encoding="utf-8")

import torch
from isre.training.train import load_trajectories
from isre.learning.encoder import ASTEncoder
from isre.learning.policy import PolicyNetwork

DEV = "cuda" if torch.cuda.is_available() else "cpu"
HID = 128
N = 400

steps = load_trajectories("isre/trajectories_v6_bfs", max_files=300)[:N]
enc = ASTEncoder(hidden_dim=HID, num_rounds=4).to(DEV)
pol = PolicyNetwork(node_emb_dim=HID * 2, variant="mlp", hidden_dim=HID).to(DEV)
enc.train(); pol.train()
opt = torch.optim.AdamW(list(enc.parameters()) + list(pol.parameters()), lr=1e-3)

t_enc = []; t_pol = []; t_bwd = []; t_tot = []


def sync():
    if DEV == "cuda":
        torch.cuda.synchronize()


# warmup
for s in steps[:20]:
    e, _ = enc(s.ast); sc = pol(e, s.candidate_actions)
sync()

for s in steps:
    opt.zero_grad()
    sync(); a = time.perf_counter()
    emb, _ = enc(s.ast)
    sync(); b = time.perf_counter()
    scores = pol(emb, s.candidate_actions)
    sync(); c = time.perf_counter()
    if scores.numel() == 0:
        continue
    gold = None
    for j, (nid, act) in enumerate(s.candidate_actions):
        if nid == s.gold_node_id and act == s.gold_action:
            gold = j; break
    if gold is None:
        continue
    loss = torch.nn.functional.cross_entropy(
        scores.unsqueeze(0), torch.tensor([gold], device=DEV))
    loss.backward(); opt.step()
    sync(); d = time.perf_counter()
    t_enc.append(b - a); t_pol.append(c - b); t_bwd.append(d - c)
    t_tot.append(d - a)


def stat(xs):
    return f"mean={statistics.fmean(xs)*1000:.2f}ms  p50={statistics.median(xs)*1000:.2f}ms"


print(f"Device={DEV}  hidden={HID}  n={len(t_tot)} steps\n")
print(f"  encoder.forward : {stat(t_enc)}")
print(f"  policy.forward  : {stat(t_pol)}")
print(f"  loss+backward   : {stat(t_bwd)}")
print(f"  TOTAL/step      : {stat(t_tot)}")
te, tp, tb = statistics.fmean(t_enc), statistics.fmean(t_pol), statistics.fmean(t_bwd)
tot = te + tp + tb
print(f"\n  share: encoder={te/tot:.0%}  policy={tp/tot:.0%}  backward={tb/tot:.0%}")
print(f"\nVERDICT: "
      + ("encoder dominates -> lever #1 (within-tree vectorization, "
         "exact-equivalent, low risk) is the right first move."
         if te / tot > 0.5 else
         "encoder NOT dominant -> cross-tree batching (#3) needed; "
         "profile the rest."))
