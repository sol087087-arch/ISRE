"""Verify the claim: SORT_COMMUTATIVE is the ONLY action whose application
can leave canonical_cycle_key() unchanged while changing to_expr().

If any OTHER action exhibits this (key-invariant + expr-changed), the
loop-detection rationale is incomplete AND it likely signals an action
that is secretly a commutative-reorder no-op (a bug).

Scans real dataset states: for every (state, candidate action), apply it,
compare canonical_cycle_key and to_expr before/after.
"""
import sys, json
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from isre.symbolic.isre_ast import ASTNode
from isre.symbolic.symbolic_engine import SymbolicEngine

eng = SymbolicEngine()
files = sorted(Path("isre/trajectories").glob("traj_*.json"))[:3000]

# For each action: count (key_same & expr_diff), (key_diff), (both_same=noop)
key_same_expr_diff = Counter()
key_diff = Counter()
both_same = Counter()
examples = defaultdict(list)
n_states = 0

for f in files:
    t = json.loads(f.read_text(encoding="utf-8"))
    for s in t["steps"]:
        root = ASTNode.from_dict(s["state"]) if "state" in s else None
        if root is None:
            continue
        root.mark_dirty()
        root._rebuild_parents()
        n_states += 1
        for nid, _, action in eng.get_candidates(root):
            before_key = root.canonical_cycle_key()
            before_expr = root.to_expr()
            try:
                after = eng.apply(root, nid, action)
            except Exception:
                continue
            after_key = after.canonical_cycle_key()
            after_expr = after.to_expr()
            av = action.value
            if before_key == after_key and before_expr != after_expr:
                key_same_expr_diff[av] += 1
                if len(examples[av]) < 3:
                    examples[av].append((before_expr, after_expr))
            elif before_key == after_key and before_expr == after_expr:
                both_same[av] += 1
            else:
                key_diff[av] += 1

print(f"Scanned {n_states} states from {len(files)} trajectories\n")
print(f"{'ACTION':<20} {'key=  expr!=':>14} {'noop':>8} {'key!=':>10}")
print("-" * 56)
all_actions = sorted(set(key_same_expr_diff) | set(key_diff) | set(both_same))
for a in all_actions:
    print(f"{a:<20} {key_same_expr_diff[a]:>14} {both_same[a]:>8} {key_diff[a]:>10}")

print("\n=== ACTIONS with key-invariant + expr-changed (the SORT property) ===")
offenders = {a: c for a, c in key_same_expr_diff.items() if c > 0}
for a, c in offenders.items():
    print(f"  {a}: {c}")
    for b, af in examples[a]:
        print(f"     {b}  ->  {af}")

only_sort = set(offenders) <= {"SORT_COMMUTATIVE"}
print(f"\nCLAIM (only SORT_COMMUTATIVE has this property): {only_sort}")
sys.exit(0 if only_sort else 1)
