"""Option A cost estimate: if we keep ONLY forward-consistent trajectories
(gold sequence replays exactly to canonical), what's left?

Decision-relevant question: does the clean subset still cover all 10 gold
actions with healthy proportions, or does dropping broken trajectories gut
a type (esp. EXPAND, the most-affected, already only 1.8%)? If a type
craters -> Option A-lite (drop) repeats failure #3; need Option C instead.

Reports gold-action distribution on FULL vs CLEAN subset, side by side.
"""
import sys, json
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from isre.symbolic.isre_ast import ASTNode
from isre.symbolic.symbolic_engine import SymbolicEngine, ActionType

eng = SymbolicEngine()
files = sorted(Path("isre/trajectories").glob("traj_*.json"))

full_actions = Counter()
clean_actions = Counter()
full_traj = 0
clean_traj = 0
clean_by_diff = Counter()
full_by_diff = Counter()

for f in files:
    t = json.loads(f.read_text(encoding="utf-8"))
    full_traj += 1
    full_by_diff[t["difficulty"]] += 1
    for s in t["steps"]:
        full_actions[s["gold_action"]] += 1

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

    if ok:
        clean_traj += 1
        clean_by_diff[t["difficulty"]] += 1
        for s in t["steps"]:
            clean_actions[s["gold_action"]] += 1

ft = sum(full_actions.values())
ct = sum(clean_actions.values())

print(f"Trajectories : full={full_traj}  clean={clean_traj}  "
      f"({clean_traj/full_traj:.1%} kept, {1-clean_traj/full_traj:.1%} dropped)")
print(f"Training pairs: full={ft}  clean={ct}  ({ct/ft:.1%} kept)\n")

print(f"{'ACTION':<20}{'FULL %':>10}{'CLEAN %':>10}{'kept%':>9}")
print("-" * 49)
for a in sorted(full_actions, key=lambda k: -full_actions[k]):
    fp = full_actions[a] / ft
    cp = clean_actions[a] / ct if ct else 0
    kept = clean_actions[a] / full_actions[a] if full_actions[a] else 0
    flag = "  <-- GUTTED" if kept < 0.5 else ("  <-- thin" if cp < 0.005 else "")
    print(f"{a:<20}{fp:>9.1%}{cp:>10.1%}{kept:>8.0%}{flag}")

print(f"\nPer-difficulty kept:")
for d in sorted(full_by_diff):
    fk = full_by_diff[d]; ck = clean_by_diff[d]
    print(f"  diff={d}: {ck}/{fk}  ({ck/fk:.0%} kept)")

print("\nVERDICT:")
worst = min((clean_actions[a]/full_actions[a]) for a in full_actions)
worst_a = min(full_actions, key=lambda a: clean_actions[a]/full_actions[a])
if worst < 0.5:
    print(f"  Option A-lite GUTS '{worst_a}' (only {worst:.0%} kept). Dropping")
    print(f"  broken trajectories repeats failure #3. Need Option C")
    print(f"  (record true multi-step forward), not A.")
else:
    print(f"  All actions retain >={worst:.0%} (worst: {worst_a}). Option A")
    print(f"  leaves a balanced dataset. Coverage loss acceptable.")
