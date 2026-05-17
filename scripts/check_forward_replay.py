"""GENERATION CONSISTENCY GATE (was missing):
Replay the FULL gold action sequence forward from original_ast.
Does it reach canonical_ast EXACTLY?

If not, the trajectory is internally inconsistent: difficulty / gold
sequence claims a path that does not actually reach the recorded canonical.
Root cause class: an inverse transform whose forward direction needs >1
engine action (e.g. FACTOR_VARIABLE/FACTOR_PAIR -> EXPAND leaves products
un-merged), recorded as a single gold step.

This is the consistency check that should have gated generation from day 1.
"""
import sys, json
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from isre.symbolic.isre_ast import ASTNode
from isre.symbolic.symbolic_engine import SymbolicEngine, ActionType

eng = SymbolicEngine()
files = sorted(Path("isre/trajectories").glob("traj_*.json"))

n = 0
broken = 0
broken_by_diff = Counter()
broken_by_last_inv = Counter()
broken_by_gold_seq = Counter()
examples = []

for f in files:
    t = json.loads(f.read_text(encoding="utf-8"))
    n += 1
    root = ASTNode.from_dict(t["original_ast"])
    root.mark_dirty(); root._rebuild_parents()
    canon_expr = ASTNode.from_dict(t["canonical_ast"]).to_expr()

    ok = True
    try:
        for s in t["steps"]:
            act = next(a for a in ActionType if a.value == s["gold_action"])
            root = eng.apply(root, s["gold_node_id"], act)
        ok = (root.to_expr() == canon_expr)
    except Exception:
        ok = False

    if not ok:
        broken += 1
        broken_by_diff[t["difficulty"]] += 1
        inv_seq = t.get("inverse_sequence", [])
        broken_by_last_inv[inv_seq[0] if inv_seq else "?"] += 1
        gold_seq = tuple(s["gold_action"] for s in t["steps"])
        broken_by_gold_seq[gold_seq] += 1
        if len(examples) < 6:
            examples.append((f.name, t["difficulty"], inv_seq,
                             t["original_expr"], t["canonical_expr"],
                             root.to_expr()))

print(f"=== Forward-replay consistency on {n} trajectories ===\n")
print(f"BROKEN (gold sequence does NOT reach canonical): {broken}/{n} "
      f"({broken/n:.2%})\n")

print("By difficulty:")
for d in sorted(broken_by_diff):
    print(f"  diff={d}: {broken_by_diff[d]}")

print("\nBy first inverse transform (backward) that seeded the trajectory:")
for k, v in broken_by_last_inv.most_common():
    print(f"  {k:<25s} {v}")

print("\nBy gold action sequence (top 8):")
for k, v in broken_by_gold_seq.most_common(8):
    print(f"  {v:5d}  {' -> '.join(k)}")

print("\nExamples:")
for name, d, inv, o, c, got in examples:
    print(f"  {name} diff={d} inv={inv}")
    print(f"    orig:  {o}")
    print(f"    canon: {c}")
    print(f"    gold-replay ended at: {got}")
    print()

sys.exit(1 if broken else 0)
