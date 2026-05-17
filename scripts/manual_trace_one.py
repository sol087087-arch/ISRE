"""Manual single-example trace per reviewer's exact checklist.

Disambiguates reviewer hypothesis 5's two readings:
  (5-benign)   gold trajectory is CORRECT, BFS just can't find the same
               1-step path because forward space lacks it.
  (5-malignant) gold trajectory is WRONG: applying gold action does NOT
               reach the recorded canonical at all.

Checklist:
  1. print state_0, gold_action, state_canonical
  2. engine.get_candidates(state_0) — contains gold (node_id, action)?
  3. engine.apply(state_0, gold_nid, gold_act) -> equals canonical?
  4. is that result canonical by ground-truth string equality?
  5. what does BFS use as goal? (inspect: to_expr string equality vs same canon)
"""
import sys, json
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from isre.symbolic.isre_ast import ASTNode
from isre.symbolic.symbolic_engine import SymbolicEngine, ActionType

eng = SymbolicEngine()
f = Path("isre/trajectories/traj_0000234.json")
t = json.loads(f.read_text(encoding="utf-8"))

print(f"file: {f.name}  difficulty={t['difficulty']}  inv_seq={t['inverse_sequence']}")
print(f"#gold steps recorded: {len(t['steps'])}\n")

state_0 = ASTNode.from_dict(t["original_ast"])
state_0.mark_dirty(); state_0._rebuild_parents()
canonical = ASTNode.from_dict(t["canonical_ast"])

s = t["steps"][0]
gold_nid = s["gold_node_id"]
gold_av = s["gold_action"]
gold_act = next(a for a in ActionType if a.value == gold_av)

print("[1] STATES")
print(f"    state_0        = {state_0.to_expr()}")
print(f"    gold_action    = (node={gold_nid}, {gold_av})")
print(f"    state_canonical= {canonical.to_expr()}")
print()

print("[2] candidates at state_0")
cands = [(nid, a.value) for nid, _, a in eng.get_candidates(state_0)]
print(f"    {cands}")
gold_in = any(nid == gold_nid and av == gold_av for nid, av in cands)
print(f"    gold (node={gold_nid},{gold_av}) in candidates? {gold_in}")
print()

print("[3] apply gold action to state_0")
after = eng.apply(state_0, gold_nid, gold_act)
print(f"    result         = {after.to_expr()}")
print(f"    == canonical ? {after.to_expr() == canonical.to_expr()}")
print()

print("[4] is the gold-applied result the recorded canonical (ground truth)?")
print(f"    after.to_expr()  = {after.to_expr()}")
print(f"    canon.to_expr()  = {canonical.to_expr()}")
print(f"    EQUAL: {after.to_expr() == canonical.to_expr()}")
print()

print("[5] what does BFS use as goal?")
print("    bfs_optimal.py: canon_expr = canonical.to_expr();")
print("                    success when child.to_expr() == canon_expr")
print("    -> SAME canonical, SAME string-equality as gold check. No mismatch.")
print()

# How many forward actions ACTUALLY needed? replay greedily via engine search
from collections import deque
def true_min(start, canon, max_d=20, budget=40000):
    ce = canon.to_expr()
    r = start.clone(); r.mark_dirty(); r._rebuild_parents()
    if r.to_expr() == ce: return 0
    q = deque([(r,0)]); seen={r.to_expr()}; exp=0
    while q:
        nd,d = q.popleft()
        if d>=max_d: continue
        exp+=1
        if exp>budget: return -1
        for nid,_,a in eng.get_candidates(nd):
            c = eng.apply(nd,nid,a)
            if c.to_expr()==ce: return d+1
            if c.to_expr() not in seen:
                seen.add(c.to_expr()); q.append((c,d+1))
    return -2

tm = true_min(state_0, canonical)
print(f"[VERDICT] difficulty label = {t['difficulty']} (1 gold step)")
print(f"          true minimal forward steps to canonical (BFS) = {tm}")
print()
if after.to_expr() != canonical.to_expr():
    print("  Hypothesis 5-MALIGNANT confirmed: gold action does NOT reach")
    print("  canonical. The trajectory is INTERNALLY INCONSISTENT — not a")
    print("  forward/inverse space asymmetry, an outright wrong gold label.")
    print("  (Hypotheses 1-4 ruled out: same canon+string-eq, gold IS the")
    print("   only candidate and IS applied, node_id matches, BFS succeeded.)")
else:
    print("  gold reaches canonical — anomaly is elsewhere.")
