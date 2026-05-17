"""Why does BFS report optimal > 1 for difficulty-1 trajectories?

A diff-1 trajectory has a 1-step gold path: start --gold--> canonical.
BFS from start MUST find a <=1 step path. If BFS returns >1, either:
  (a) the gold (node_id, action) does NOT reach canonical under current engine
      (trajectory/engine inconsistency), or
  (b) BFS misses the 1-step solution (BFS bug).

For each diff-1 trajectory where BFS != 1, print:
  - gold action + node_id, whether applying it reaches canonical
  - whether ANY candidate at start reaches canonical in 1 step
  - the BFS result
"""
import sys, json
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from isre.symbolic.isre_ast import ASTNode
from isre.symbolic.symbolic_engine import SymbolicEngine, ActionType
from isre.baselines.bfs_optimal import bfs_optimal

eng = SymbolicEngine()
files = sorted(Path("isre/trajectories").glob("traj_*.json"))[:500]

checked = 0
anomalies = 0
for f in files:
    t = json.loads(f.read_text(encoding="utf-8"))
    if t["difficulty"] != 1:
        continue
    checked += 1
    start = ASTNode.from_dict(t["original_ast"])
    canon = ASTNode.from_dict(t["canonical_ast"])
    canon_expr = canon.to_expr()

    outcome, opt = bfs_optimal(start, canon, eng, max_expansions=40000, max_depth=20)
    if outcome == "SUCCESS" and opt == 1:
        continue
    anomalies += 1
    if anomalies > 8:
        continue

    s = t["steps"][0]
    gold_nid = s["gold_node_id"]
    gold_av = s["gold_action"]
    gold_act = next(a for a in ActionType if a.value == gold_av)

    root = ASTNode.from_dict(t["original_ast"])
    root.mark_dirty(); root._rebuild_parents()

    # does the recorded gold step reach canonical?
    after_gold = eng.apply(root, gold_nid, gold_act)
    gold_reaches = after_gold.to_expr() == canon_expr

    # does ANY candidate at start reach canonical in 1 step?
    one_step = []
    for nid, _, act in eng.get_candidates(root):
        ch = eng.apply(root, nid, act)
        if ch.to_expr() == canon_expr:
            one_step.append((nid, act.value))

    print(f"--- {f.name}  BFS={outcome}/{opt} ---")
    print(f"  orig:  {t['original_expr']}")
    print(f"  canon: {t['canonical_expr']}")
    print(f"  gold:  node={gold_nid} action={gold_av}  reaches_canon={gold_reaches}")
    print(f"  gold-applied -> {after_gold.to_expr()}")
    print(f"  1-step candidates reaching canon: {one_step}")
    cands = [(nid, a.value) for nid, _, a in eng.get_candidates(root)]
    print(f"  all candidates at start: {cands}")
    print()

print(f"diff-1 checked={checked}  anomalies(BFS!=1)={anomalies}")
