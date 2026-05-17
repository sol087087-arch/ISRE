"""Acceptance gate for the PER-INDEX path (generate_one_indexed).

Two things, both must pass before the ~100 CPU-h parallel v6 regen:

  A) BFS-arm correctness (same criteria as cbfs_acceptance):
     0% forward-replay broken, 0 cycles, all 10 actions, EXPAND healthy.

  B) PAIRING INVARIANT (the make-or-break ablation check):
     for each idx, generate_one_indexed(idx, "recorded") and
     generate_one_indexed(idx, "bfs") MUST have byte-identical
     original_ast (same scrambled start). If this fails, the ablation
     CE-on-recorded vs CE-on-BFS is confounded by different starts.
"""
import sys, json
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")

from isre.data.trajectory_generator import TrajectoryGenerator
from isre.symbolic.isre_ast import ASTNode
from isre.symbolic.symbolic_engine import SymbolicEngine, ActionType

N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
SEED = 42
eng = SymbolicEngine()

broken = cycles = 0
actions = Counter()
diffs = Counter()
produced = 0
pair_checked = pair_mismatch = 0
recorded_produced = 0

for idx in range(N):
    g_bfs = TrajectoryGenerator(seed=SEED, max_ast_depth=6,
                                max_trajectory_length=6, bfs_budget=100000)
    t_bfs = g_bfs.generate_one_indexed(idx, seed_base=SEED, gold_mode="bfs")

    g_rec = TrajectoryGenerator(seed=SEED, max_ast_depth=6,
                                max_trajectory_length=6, bfs_budget=100000)
    t_rec = g_rec.generate_one_indexed(idx, seed_base=SEED, gold_mode="recorded")

    # B) pairing invariant — only meaningful when BOTH produced
    if t_bfs is not None and t_rec is not None:
        pair_checked += 1
        if json.dumps(t_bfs.original_ast, sort_keys=True) != \
           json.dumps(t_rec.original_ast, sort_keys=True):
            pair_mismatch += 1
    if t_rec is not None:
        recorded_produced += 1

    # A) BFS-arm correctness
    if t_bfs is None:
        continue
    produced += 1
    diffs[t_bfs.difficulty] += 1
    for s in t_bfs.steps:
        actions[s.gold_action] += 1
    root = ASTNode.from_dict(t_bfs.original_ast)
    root.mark_dirty(); root._rebuild_parents()
    canon = ASTNode.from_dict(t_bfs.canonical_ast).to_expr()
    seen = set(); cyc = False
    try:
        for s in t_bfs.steps:
            e = root.to_expr()
            if e in seen: cyc = True
            seen.add(e)
            act = next(a for a in ActionType if a.value == s.gold_action)
            root = eng.apply(root, s.gold_node_id, act)
        if root.to_expr() != canon:
            broken += 1
    except Exception:
        broken += 1
    if cyc:
        cycles += 1

tot = sum(actions.values())
ALL10 = ["FOLD_CONST","REMOVE_ZERO","REMOVE_ONE","FLATTEN_ADD","FLATTEN_MUL",
         "COMBINE_COEFF","COLLECT_TERMS","MERGE_POWER","EXPAND","SORT_COMMUTATIVE"]

print(f"=== PER-INDEX acceptance (N={N}, seed={SEED}) ===\n")
print(f"[A] BFS-arm correctness:")
print(f"    produced            : {produced}/{N}  (drop {(N-produced)/N:.1%})")
print(f"    forward-replay broken: {broken}/{produced} ({broken/max(produced,1):.2%})")
print(f"    cycles              : {cycles}/{produced}")
print(f"    EXPAND              : {actions['EXPAND']}/{tot} "
      f"({actions['EXPAND']/max(tot,1):.2%})")
missing = [a for a in ALL10 if actions[a] == 0]
print(f"    missing actions     : {missing if missing else 'none'}")
print(f"    difficulty          : {dict(sorted(diffs.items()))}")
print(f"\n[B] PAIRING INVARIANT (ablation make-or-break):")
print(f"    recorded produced   : {recorded_produced}/{N}")
print(f"    pairs checked       : {pair_checked}")
print(f"    start MISMATCHES    : {pair_mismatch}")

a_ok = (broken == 0 and cycles == 0 and not missing
        and actions["EXPAND"]/max(tot,1) >= 0.008 and produced/N >= 0.85)
b_ok = (pair_mismatch == 0 and pair_checked >= int(0.8 * N))
print("\n" + "=" * 52)
if a_ok and b_ok:
    print("ACCEPTANCE: PASS — per-index path clear for parallel v6 regen")
else:
    print("ACCEPTANCE: FAIL")
    if not a_ok:
        print(f"  [A] broken={broken} cycles={cycles} missing={missing} "
              f"expand={actions['EXPAND']/max(tot,1):.2%} drop={(N-produced)/N:.1%}")
    if not b_ok:
        print(f"  [B] PAIRING BROKEN: {pair_mismatch} start mismatches "
              f"-> ablation would be confounded. DO NOT REGEN.")
sys.exit(0 if (a_ok and b_ok) else 1)
