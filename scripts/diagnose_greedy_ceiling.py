"""Why is greedy 100%? Variant A (trivial distribution) vs B (soft metric/bug).

Metric already confirmed ground-truth (to_expr == canonical). This script
attacks Variant A: per-difficulty, does greedy follow the OPTIMAL (teacher)
path or a longer one? If greedy steps == teacher steps everywhere, the
distribution is greedy-trivial (no headroom). If greedy steps > teacher
steps, there is EFFICIENCY headroom even at 100% success.

Also dumps a few long-trajectory (diff 5-6) cases where greedy succeeds,
for manual eyeballing.
"""
import sys, json
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from isre.symbolic.isre_ast import ASTNode
from isre.symbolic.symbolic_engine import SymbolicEngine
from isre.baselines.greedy_heuristic import rollout

eng = SymbolicEngine()
files = sorted(Path("isre/trajectories").glob("traj_*.json"))[:3000]

# per difficulty: [n, n_success, sum_greedy_steps_on_success, sum_teacher_steps]
agg = defaultdict(lambda: [0, 0, 0, 0])
strictly_longer = 0          # greedy took MORE steps than teacher (suboptimal)
exactly_optimal = 0          # greedy == teacher steps
shorter = 0                  # greedy < teacher (teacher was not optimal)
long_success_examples = []

for f in files:
    t = json.loads(f.read_text(encoding="utf-8"))
    diff = t["difficulty"]
    start = ASTNode.from_dict(t["original_ast"])
    canon = ASTNode.from_dict(t["canonical_ast"])
    success, steps, _ = rollout(start, canon, eng, max_steps=20)
    a = agg[diff]
    a[0] += 1
    if success:
        a[1] += 1
        a[2] += steps
        a[3] += diff  # teacher steps == difficulty (gold path length)
        if steps > diff:
            strictly_longer += 1
        elif steps == diff:
            exactly_optimal += 1
        else:
            shorter += 1
        if diff >= 5 and len(long_success_examples) < 5:
            long_success_examples.append(
                (f.name, diff, steps, t["original_expr"], t["canonical_expr"])
            )

print(f"Greedy per-difficulty (n={len(files)} trajectories):\n")
print(f"{'diff':>4} {'n':>5} {'succ%':>7} {'greedy_steps':>13} {'teacher_steps':>14} {'overhead':>9}")
print("-" * 60)
tot_n = tot_s = 0
for d in sorted(agg):
    n, s, gs, ts = agg[d]
    tot_n += n; tot_s += s
    if s:
        print(f"{d:>4} {n:>5} {s/n:>6.1%} {gs/s:>13.2f} {ts/s:>14.2f} {(gs-ts)/s:>+9.2f}")
    else:
        print(f"{d:>4} {n:>5} {0:>6.1%} {'-':>13} {'-':>14} {'-':>9}")

print(f"\nTotal: {tot_s}/{tot_n} = {tot_s/tot_n:.1%}")
print(f"\nPath optimality (on greedy successes):")
tot = strictly_longer + exactly_optimal + shorter
print(f"  greedy LONGER than teacher (suboptimal):  {strictly_longer:5d}  ({strictly_longer/tot:.1%})")
print(f"  greedy EQUAL to teacher (optimal):        {exactly_optimal:5d}  ({exactly_optimal/tot:.1%})")
print(f"  greedy SHORTER (teacher not optimal):     {shorter:5d}  ({shorter/tot:.1%})")

print(f"\nLong (diff>=5) greedy-success examples for eyeball:")
for name, d, st, o, c in long_success_examples:
    print(f"  {name} diff={d} greedy_steps={st}")
    print(f"    orig:  {o}")
    print(f"    canon: {c}")

# Verdict
if strictly_longer / tot > 0.10:
    print(f"\nVERDICT: greedy is SUBOPTIMAL on {strictly_longer/tot:.0%} of successes "
          f"-> EFFICIENCY headroom exists. Neural can win on steps (Option 2).")
else:
    print(f"\nVERDICT: greedy is near-optimal everywhere "
          f"-> distribution is greedy-trivial. Need harder dist (Option 1) for v1.")
