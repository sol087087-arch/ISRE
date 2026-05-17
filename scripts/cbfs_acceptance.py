"""C-BFS acceptance gate on a sample. MUST pass before full v6 regen.

Criteria:
  1. forward-replay consistency = 100% (0% broken) — BY CONSTRUCTION
  2. all 10 gold actions present; EXPAND healthy (NOT gutted — the whole
     reason we chose C over A)
  3. difficulty range sane (1..~8)
  4. drop rate (BFS timeout/unreachable) reported
"""
import sys, json, tempfile
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from isre.data.trajectory_generator import generate_dataset
from isre.symbolic.isre_ast import ASTNode
from isre.symbolic.symbolic_engine import SymbolicEngine, ActionType

N = int(sys.argv[1]) if len(sys.argv) > 1 else 500
d = tempfile.mkdtemp(prefix="isre_cbfs_acc_")
gen = generate_dataset(count=N, seed=42, output_dir=d)

files = sorted(Path(d).glob("traj_*.json"))
eng = SymbolicEngine()

broken = 0
actions = Counter()
diffs = Counter()
cycle = 0

for f in files:
    t = json.loads(f.read_text(encoding="utf-8"))
    diffs[t["difficulty"]] += 1
    for s in t["steps"]:
        actions[s["gold_action"]] += 1
    # forward-replay consistency
    root = ASTNode.from_dict(t["original_ast"])
    root.mark_dirty(); root._rebuild_parents()
    canon = ASTNode.from_dict(t["canonical_ast"]).to_expr()
    seen = set()
    cyc = False
    try:
        for s in t["steps"]:
            e = root.to_expr()
            if e in seen:
                cyc = True
            seen.add(e)
            act = next(a for a in ActionType if a.value == s["gold_action"])
            root = eng.apply(root, s["gold_node_id"], act)
        if root.to_expr() != canon:
            broken += 1
    except Exception:
        broken += 1
    if cyc:
        cycle += 1

tot = sum(actions.values())
print(f"\n=== C-BFS acceptance (requested N={N}, produced {len(files)}) ===")
print(f"drop rate (BFS timeout/unreachable+other): "
      f"{(N-len(files))/N:.1%}  [{N-len(files)} dropped]")
print(f"forward-replay BROKEN: {broken}/{len(files)} ({broken/len(files):.2%})")
print(f"trajectories with cycle: {cycle}/{len(files)} ({cycle/len(files):.2%})")

print(f"\nGold action distribution ({tot} pairs):")
ALL10 = ["FOLD_CONST","REMOVE_ZERO","REMOVE_ONE","FLATTEN_ADD","FLATTEN_MUL",
         "COMBINE_COEFF","COLLECT_TERMS","MERGE_POWER","EXPAND","SORT_COMMUTATIVE"]
for a in sorted(ALL10, key=lambda k: -actions[k]):
    c = actions[a]
    flag = "  <-- MISSING" if c == 0 else ("  <-- thin" if c/tot < 0.005 else "")
    print(f"  {a:<20}{c:>6}{c/tot:>8.1%}{flag}")

print(f"\nDifficulty: {dict(sorted(diffs.items()))}")

missing = [a for a in ALL10 if actions[a] == 0]
expand_ok = actions["EXPAND"] / tot >= 0.01 if tot else False
gate_pass = (broken == 0 and not missing and expand_ok
             and cycle == 0 and len(files) / N >= 0.85)

print("\n" + "=" * 50)
if gate_pass:
    print("ACCEPTANCE: PASS — clear for full v6 regeneration")
else:
    print("ACCEPTANCE: FAIL")
    if broken: print(f"  - {broken} broken (must be 0 by construction!)")
    if missing: print(f"  - missing actions: {missing}")
    if not expand_ok: print(f"  - EXPAND too thin: {actions['EXPAND']/tot:.2%} (need >=1%)")
    if cycle: print(f"  - {cycle} cycles")
    if len(files)/N < 0.85: print(f"  - drop rate {(N-len(files))/N:.0%} too high")
sys.exit(0 if gate_pass else 1)
