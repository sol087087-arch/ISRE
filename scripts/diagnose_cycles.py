"""Diagnose state cycles in trajectories."""
import json, sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

files = sorted(Path("isre/trajectories").glob("traj_*.json"))[:5000]
cycle_examples = []

for f in files:
    t = json.loads(f.read_text(encoding="utf-8"))
    states  = [s["state_expr"]  for s in t["steps"]]
    actions = [s["gold_action"] for s in t["steps"]]
    seen = {}
    for i, s in enumerate(states):
        if s in seen:
            if len(cycle_examples) < 8:
                cycle_examples.append({
                    "file":     f.name,
                    "inv_seq":  t.get("inverse_sequence", []),
                    "states":   states,
                    "actions":  actions,
                    "first":    seen[s],
                    "repeat":   i,
                })
            break
        seen[s] = i

print(f"Cycle examples (first {len(cycle_examples)}):\n")
for ex in cycle_examples:
    print(f"  file:    {ex['file']}")
    print(f"  inv_seq: {ex['inv_seq']}")
    print(f"  actions: {ex['actions']}")
    print(f"  states:")
    for i, state in enumerate(ex["states"]):
        action = ex["actions"][i] if i < len(ex["actions"]) else "[END]"
        tag = ""
        if i == ex["first"]:  tag = "  <-- FIRST"
        if i == ex["repeat"]: tag = "  <-- REPEAT"
        print(f"    [{i}] {action:<25s}  {state}{tag}")
    print()
