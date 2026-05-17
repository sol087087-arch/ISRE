"""Pre-decision data for Option A vs C. Three things:

#1 Per-inverse-type break attribution (PRECISE, per-step):
   Replay gold step by step. The FIRST step where
   apply(steps[i].state, gold_i).to_expr() != expected_next  is the break.
   Attribute it to inverse_names[len-1-i] (forward step i undoes the
   i-th-from-last backward inverse — see generate_one: backward_records
   built backward, then reversed for steps; inverse_names parallel).

#2 Difficulty distribution: full vs clean (forward-consistent) subset.

(C-cost is a separate BFS-sample script.)
"""
import sys, json
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from isre.symbolic.isre_ast import ASTNode
from isre.symbolic.symbolic_engine import SymbolicEngine, ActionType

eng = SymbolicEngine()
files = sorted(Path("isre/trajectories").glob("traj_*.json"))

inv_total = Counter()      # how often each inverse appears (per trajectory)
inv_in_broken = Counter()  # appears in a trajectory that is broken
break_attributed = Counter()  # inverse blamed for the FIRST failing step
full_diff = Counter()
clean_diff = Counter()
n = 0
broken = 0

for f in files:
    t = json.loads(f.read_text(encoding="utf-8"))
    n += 1
    diff = t["difficulty"]
    full_diff[diff] += 1
    inv_seq = t.get("inverse_sequence", [])
    for iv in set(inv_seq):
        inv_total[iv] += 1

    steps = t["steps"]
    canon_expr = ASTNode.from_dict(t["canonical_ast"]).to_expr()

    root = ASTNode.from_dict(t["original_ast"])
    root.mark_dirty(); root._rebuild_parents()

    first_bad = None
    try:
        for i, s in enumerate(steps):
            act = next(a for a in ActionType if a.value == s["gold_action"])
            root = eng.apply(root, s["gold_node_id"], act)
            expected = (steps[i + 1]["state_expr"]
                        if i + 1 < len(steps) else canon_expr)
            if root.to_expr() != expected:
                first_bad = i
                break
        ok = (first_bad is None and root.to_expr() == canon_expr)
    except Exception:
        ok = False
        first_bad = first_bad if first_bad is not None else 0

    if ok:
        clean_diff[diff] += 1
    else:
        broken += 1
        for iv in set(inv_seq):
            inv_in_broken[iv] += 1
        if first_bad is not None and inv_seq:
            # forward step i undoes inverse_names[len-1-i]
            bw_idx = len(inv_seq) - 1 - first_bad
            if 0 <= bw_idx < len(inv_seq):
                break_attributed[inv_seq[bw_idx]] += 1

clean = n - broken
print(f"Trajectories: {n}  clean={clean} ({clean/n:.1%})  broken={broken} ({broken/n:.1%})\n")

print("#1 PER-INVERSE: appears_in / blamed_for_first_break")
print(f"{'inverse':<26}{'appears':>8}{'in_broken':>11}{'BLAMED':>8}{'blame%':>8}")
print("-" * 61)
for iv in sorted(inv_total, key=lambda k: -break_attributed[k]):
    tot = inv_total[iv]
    blamed = break_attributed[iv]
    print(f"{iv:<26}{tot:>8}{inv_in_broken[iv]:>11}{blamed:>8}"
          f"{blamed/max(tot,1):>7.0%}")

print("\n#2 DIFFICULTY: full -> clean (kept%)")
for d in sorted(full_diff):
    fk, ck = full_diff[d], clean_diff[d]
    print(f"  diff={d}: {fk:>6} -> {ck:>6}  ({ck/fk:.0%} kept)")
fa = sum(d*c for d, c in full_diff.items()) / n
ca = sum(d*c for d, c in clean_diff.items()) / max(clean, 1)
print(f"  avg difficulty: full={fa:.2f}  clean={ca:.2f}  (shift {ca-fa:+.2f})")

print("\nINTERPRETATION:")
top = break_attributed.most_common(1)[0]
print(f"  Dominant break source: {top[0]} ({top[1]} of {broken} breaks)")
if ca - fa > 0.3 or ca - fa < -0.3:
    print(f"  Difficulty SHIFT {ca-fa:+.2f} after dropping broken -> A skews dist.")
else:
    print(f"  Difficulty shift {ca-fa:+.2f} — mild.")
