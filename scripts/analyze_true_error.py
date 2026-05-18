"""Where do the 128-full true_errors live? Data-coverage vs irreducible.

Hypothesis (reviewer): the residual ~19% true_error is a HOLE IN DATA
COVERAGE of specific compositional patterns, not an architecture limit.
If true_error concentrates in one provenance bucket (e.g. FACTOR_PAIR at
depth>=4) -> data-coverage bottleneck. If ~uniform -> not here.

Method: load 128-full (the saturation operating point — its residual
error IS the real signal; 512 is worse/overfit). Reuse eval_neural's
model-load + _argmax_action (no interface drift). Teacher-force the BFS
gold path on the IDENTICAL leak-free held-out (val_traj_ids.json,
split_seed=1234). A step is true_error iff mlp != bfs_gold AND
mlp != recorded_teacher (same definition as eval_neural MODE B).

NORMALIZED: per bucket report true_error RATE = te_steps / categorizable
_steps, vs the marginal rate. Concentration = rate >> marginal, NOT raw
share (60% of errors on X is meaningless if X is 60% of data).
"""
import sys, json
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from isre.symbolic.isre_ast import ASTNode
from isre.symbolic.symbolic_engine import SymbolicEngine
from scripts.eval_neural import load_model, _argmax_action, _ACT

CKPT = "checkpoints_128_full/best.pt"
HID = 128
BFS = Path("isre/trajectories_v6_bfs")
REC = Path("isre/trajectories_v6_recorded")

vpath = Path(CKPT).parent / "val_traj_ids.json"
meta = json.loads(vpath.read_text(encoding="utf-8"))
val_ids = sorted(meta["val_traj_ids"])
print(f"128-full true_error provenance analysis")
print(f"held-out: {len(val_ids)} trajs (split_seed={meta['split_seed']})\n")

import torch
dev = "cuda" if torch.cuda.is_available() else "cpu"
enc, pol = load_model(CKPT, HID, 4, dev)
engine = SymbolicEngine()

# bucket -> [categorizable_steps, true_error_steps]
def _b(): return [0, 0]
by_firstinv = defaultdict(_b)
by_hasfactor = defaultdict(_b)
by_difficulty = defaultdict(_b)
by_missed_gold = defaultdict(_b)
tot_cat = tot_te = 0

for i, tid in enumerate(val_ids):
    f = tid if tid.endswith(".json") else f"{tid}.json"
    bp, rp = BFS / f, REC / f
    if not bp.exists():
        continue
    b = json.loads(bp.read_text(encoding="utf-8"))
    inv_seq = b.get("inverse_sequence", [])
    first_inv = inv_seq[0] if inv_seq else "(none)"
    has_factor = any("FACTOR" in x for x in inv_seq)
    diff = b["difficulty"]

    rec_by_expr = {}
    if rp.exists():
        for s in json.loads(rp.read_text(encoding="utf-8"))["steps"]:
            rec_by_expr[s["state_expr"]] = s["gold_action"]

    start = ASTNode.from_dict(b["original_ast"])
    root = start.clone(); root.mark_dirty(); root._rebuild_parents()
    for s in b["steps"]:
        bfs_act = s["gold_action"]
        nid, act, _ = _argmax_action(enc, pol, root, engine)
        if act is None:
            break
        mlp_act = act.value
        teach = rec_by_expr.get(s["state_expr"])
        # categorizable = teacher defined (matches eval_neural true_error def)
        if teach is not None:
            is_te = (mlp_act != bfs_act and mlp_act != teach)
            tot_cat += 1
            if is_te:
                tot_te += 1
            for d, key in ((by_firstinv, first_inv),
                           (by_hasfactor, "FACTOR" if has_factor else "no_factor"),
                           (by_difficulty, diff),
                           (by_missed_gold, bfs_act if is_te else None)):
                if key is None:
                    continue
                d[key][0] += 1
                if is_te:
                    d[key][1] += 1
        root = engine.apply(root, s["gold_node_id"], _ACT[s["gold_action"]])

marg = tot_te / max(tot_cat, 1)
print(f"MARGINAL true_error rate: {tot_te}/{tot_cat} = {marg:.1%}\n")


def show(title, d, minn=30):
    print(f"--- {title} (rate vs marginal {marg:.1%}; conc = rate >> marg) ---")
    rows = []
    for k, (n, te) in d.items():
        if n < minn:
            continue
        rows.append((te / n, n, te, k))
    for rate, n, te, k in sorted(rows, reverse=True):
        flag = "  <== CONCENTRATED" if rate > 1.5 * marg else (
               "  (low)" if rate < 0.5 * marg else "")
        print(f"  {str(k):<26s} rate={rate:5.1%}  ({te:5d}/{n:5d}){flag}")
    print()


show("by FIRST inverse (scrambler seed op)", by_firstinv)
show("by has-FACTOR provenance", by_hasfactor, minn=1)
show("by difficulty (BFS-optimal len)", by_difficulty)
print("--- which gold action the model MISSES on true_error steps ---")
for k, (_, te) in sorted(by_missed_gold.items(), key=lambda x: -x[1][1]):
    if te:
        print(f"  miss {k:<18s} {te}")

print("\nVERDICT:")
concentrated = [k for k, (n, te) in by_firstinv.items()
                if n >= 30 and te / n > 1.5 * marg]
hf = by_hasfactor.get("FACTOR", [0, 0])
hf_rate = hf[1] / hf[0] if hf[0] else 0
if concentrated or hf_rate > 1.5 * marg:
    print(f"  CONCENTRATED -> data-coverage bottleneck. Hot: "
          f"{concentrated} ; FACTOR-rate={hf_rate:.1%} vs marg {marg:.1%}.")
    print("  => fix is MORE/BETTER trajectories of that pattern, not capacity.")
else:
    print(f"  ~UNIFORM across provenance (no bucket >> {1.5*marg:.0%}). "
          f"Residual is not a single data hole; likely irreducible "
          f"subgradient-equivalent ambiguity. Bigger data of one type "
          f"won't fix it; this IS the KAN-interpretability target.")
