"""#3 Option C cost: if gold = BFS-optimal forward path from start, what is
the new dataset shape?

For a sample, compute BFS true-minimal forward length vs recorded difficulty.
Project: new avg trajectory length, new training-pair count vs Option A.
"""
import sys, json
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from isre.symbolic.isre_ast import ASTNode
from isre.baselines.bfs_optimal import bfs_optimal
from isre.symbolic.symbolic_engine import SymbolicEngine

eng = SymbolicEngine()
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
files = sorted(Path("isre/trajectories").glob("traj_*.json"))[:N]

rec_diff_sum = 0
bfs_len_sum = 0
ok = 0
to = 0
by_diff = {}

for f in files:
    t = json.loads(f.read_text(encoding="utf-8"))
    start = ASTNode.from_dict(t["original_ast"])
    canon = ASTNode.from_dict(t["canonical_ast"])
    d = t["difficulty"]
    outcome, opt = bfs_optimal(start, canon, eng, max_expansions=40000, max_depth=20)
    if outcome == "SUCCESS":
        ok += 1
        rec_diff_sum += d
        bfs_len_sum += opt
        r = by_diff.setdefault(d, [0, 0])
        r[0] += 1
        r[1] += opt
    elif outcome == "BFS_TIMEOUT":
        to += 1

print(f"Sample n={len(files)}  BFS success={ok}  timeout={to}\n")
print(f"{'rec_diff':>9}{'n':>6}{'bfs_optimal':>13}{'overhead':>10}")
print("-" * 38)
for d in sorted(by_diff):
    cnt, ssum = by_diff[d]
    print(f"{d:>9}{cnt:>6}{ssum/cnt:>13.2f}{ssum/cnt-d:>+10.2f}")

if ok:
    avg_rec = rec_diff_sum / ok
    avg_bfs = bfs_len_sum / ok
    print(f"\navg recorded difficulty (current gold) : {avg_rec:.2f}")
    print(f"avg BFS-optimal length  (C-BFS gold)   : {avg_bfs:.2f}")
    print(f"per-traj step delta C-BFS vs current   : {avg_bfs-avg_rec:+.2f}")
    # projected pair counts on 50k
    cur_pairs = 148832
    a_pairs = 134792  # clean subset (option A-lite)
    c_pairs = int(50000 * avg_bfs)
    print(f"\nProjected training pairs @50k trajectories:")
    print(f"  current (broken incl.) : {cur_pairs}")
    print(f"  Option A (drop broken) : {a_pairs}   "
          f"(EXPAND gutted 1.8%->0.1% — known)")
    print(f"  Option C-BFS           : ~{c_pairs}  "
          f"(all 50k kept, gold=optimal)")
