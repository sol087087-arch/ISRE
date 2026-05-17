"""Smoke test: replay one trajectory by hand, verify it reaches canonical."""
import sys, json
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from isre.symbolic.isre_ast import ASTNode
from isre.symbolic.symbolic_engine import SymbolicEngine, ActionType

idx = int(sys.argv[1]) if len(sys.argv) > 1 else 1234
f = sorted(Path("isre/trajectories").glob("traj_*.json"))[idx]
t = json.loads(f.read_text(encoding="utf-8"))

print(f"file: {f.name}  difficulty={t['difficulty']}")
print(f"canonical: {t['canonical_expr']}")
print(f"original:  {t['original_expr']}")
print(f"inverse_seq: {t['inverse_sequence']}")
print()

eng = SymbolicEngine()
root = ASTNode.from_dict(t["original_ast"])
root.mark_dirty()
root._rebuild_parents()

print("FORWARD REPLAY (apply each gold action):")
for i, s in enumerate(t["steps"]):
    print(f"  [{i}] {s['gold_action']:<18} node={s['gold_node_id']:<2} {root.to_expr()}")
    act = next(a for a in ActionType if a.value == s["gold_action"])
    root = eng.apply(root, s["gold_node_id"], act)

final = root.to_expr()
canon = t["canonical_expr"]
print(f"  [END] result: {final}")
print(f"  canonical:    {canon}")
print(f"  MATCH: {final == canon}")
sys.exit(0 if final == canon else 1)
