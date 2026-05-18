"""T2 — KANPolicy really trains: CE loss drops >30% and pykan KAN
parameters receive nonzero gradients (proves it's a live KAN, not a stub).

Run: cd C:\\GitHub\\ISRE; set PYTHONPATH=.; python scripts/test_kan_trains.py
"""
import sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import torch

from isre.symbolic.isre_ast import ASTNode
from isre.symbolic.symbolic_engine import SymbolicEngine, ActionType
from isre.learning.kan_policy import KANPolicy

torch.manual_seed(0)

DATA = Path("isre/trajectories_v6_bfs")
files = sorted(DATA.glob("traj_*.json"))[:50]
engine = SymbolicEngine()
action_map = {a.value: a for a in ActionType}

# Build ~200 (state, candidates, gold) tuples from first 50 trajectories.
samples = []
for f in files:
    traj = json.loads(f.read_text(encoding="utf-8"))
    for sd in traj["steps"]:
        try:
            ast = ASTNode.from_dict(sd["state"])
        except Exception:
            continue
        ga = sd["gold_action"]
        gn = sd["gold_node_id"]
        if ga not in action_map:
            continue
        ast.mark_dirty(); ast._rebuild_parents(); ast._ensure_metadata()
        craw = engine.get_candidates(ast)
        cand = [(nid, a) for nid, _, a in craw]
        if not any(nid == gn and a == action_map[ga] for nid, a in cand):
            continue
        samples.append((ast, cand, action_map[ga], gn))
        if len(samples) >= 200:
            break
    if len(samples) >= 200:
        break

print(f"built {len(samples)} (state,cands,gold) samples")
assert len(samples) >= 50, "too few samples"

pol = KANPolicy(hidden=8, seed=0, device="cpu")
opt = torch.optim.Adam(pol.parameters(), lr=1e-2)


def epoch_loss():
    tot, n = 0.0, 0
    for ast, cand, ga, gn in samples:
        loss = pol.compute_loss(ast, cand, ga, gn)
        tot += loss.item(); n += 1
    return tot / n


loss0 = epoch_loss()

grad_seen_nonzero = False
for step in range(30):
    opt.zero_grad()
    batch_loss = 0.0
    for ast, cand, ga, gn in samples:
        loss = pol.compute_loss(ast, cand, ga, gn)
        loss.backward()
        batch_loss += loss.item()
    if not grad_seen_nonzero:
        for p in pol.parameters():
            if p.grad is not None and p.grad.abs().sum().item() > 0:
                grad_seen_nonzero = True
                break
    torch.nn.utils.clip_grad_norm_(list(pol.parameters()), 1.0)
    opt.step()

lossN = epoch_loss()
drop = (loss0 - lossN) / loss0

print(f"loss[0]   = {loss0:.4f}")
print(f"loss[-1]  = {lossN:.4f}")
print(f"relative drop = {drop:.1%}")
print(f"nonzero KAN grad observed = {grad_seen_nonzero}")

assert grad_seen_nonzero, "NO nonzero gradient on any KAN param — dead stub"
assert drop > 0.30, f"CE loss did not drop >30% (got {drop:.1%})"

print("\nALL KAN TRAIN TESTS PASSED")
