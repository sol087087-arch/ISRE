"""Full-scale ablation pairing verifier: v6_recorded <-> v6_bfs.

The ablation CE-on-recorded-teacher vs CE-on-BFS-teacher is only valid if
the two arms share IDENTICAL scrambled starts (only gold differs). The
acceptance gate checked 300; this checks ALL paired indices.

For every trajectory_id present in BOTH dirs:
  assert original_ast byte-identical (json, sort_keys) AND
         canonical_ast byte-identical.
Any mismatch => ablation confounded by different inputs. MUST be 0.
"""
import sys, json
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

REC = Path("isre/trajectories_v6_recorded")
BFS = Path("isre/trajectories_v6_bfs")

rec_ids = {p.name for p in REC.glob("traj_*.json")}
bfs_ids = {p.name for p in BFS.glob("traj_*.json")}

inter = sorted(rec_ids & bfs_ids)
only_rec = rec_ids - bfs_ids
only_bfs = bfs_ids - rec_ids

print(f"v6_recorded : {len(rec_ids)}")
print(f"v6_bfs      : {len(bfs_ids)}")
print(f"paired (∩)  : {len(inter)}")
print(f"only_recorded: {len(only_rec)}  (expected: bfs-dropped via BFS timeout)")
print(f"only_bfs     : {len(only_bfs)}  (MUST be 0: bfs ⊆ recorded by scramble logic)")

start_mismatch = 0
canon_mismatch = 0
gold_same = 0          # sanity: gold SHOULD differ (that's the point)
examples = []

for tid in inter:
    r = json.loads((REC / tid).read_text(encoding="utf-8"))
    b = json.loads((BFS / tid).read_text(encoding="utf-8"))
    rs = json.dumps(r["original_ast"], sort_keys=True)
    bs = json.dumps(b["original_ast"], sort_keys=True)
    rc = json.dumps(r["canonical_ast"], sort_keys=True)
    bc = json.dumps(b["canonical_ast"], sort_keys=True)
    if rs != bs:
        start_mismatch += 1
        if len(examples) < 5:
            examples.append((tid, r["original_expr"], b["original_expr"]))
    if rc != bc:
        canon_mismatch += 1
    rg = [s["gold_action"] for s in r["steps"]]
    bg = [s["gold_action"] for s in b["steps"]]
    if rg == bg:
        gold_same += 1

print(f"\nstart  MISMATCHES: {start_mismatch}  (MUST be 0)")
print(f"canon  MISMATCHES: {canon_mismatch}  (MUST be 0)")
print(f"identical gold seq: {gold_same}/{len(inter)} "
      f"({gold_same/max(len(inter),1):.1%}) — "
      f"the rest is the ablation signal (recorded vs BFS-optimal gold)")

if examples:
    print("\nstart-mismatch examples (DESIGN FAILURE if any):")
    for tid, ro, bo in examples:
        print(f"  {tid}\n    recorded: {ro}\n    bfs     : {bo}")

ok = (start_mismatch == 0 and canon_mismatch == 0 and len(only_bfs) == 0)
print("\n" + "=" * 52)
print("PAIRING: PASS — ablation arms share identical starts"
      if ok else "PAIRING: FAIL — ablation confounded, DO NOT proceed")
sys.exit(0 if ok else 1)
