"""Is the residual error beam-search-recoverable?

Reviewer's compounding model: difficulty gradient (0.2->24.6%) implies a
~constant per-step error ~4-5% (after diff=2); bottleneck = greedy
decoding compounding, NOT capacity/data/curriculum/encoder.

Decisive cheap test BEFORE coding beam search: on the steps where the
model's argmax != BFS-optimal gold, what RANK does the model assign to
the gold action? (1 = model's top pick.)
  - median rank 2, p75 3        -> beam-2/3 recovers most -> beam search wins
  - median rank >=5, long tail  -> model is far from gold -> beam won't fix
  - rank 2 + long tail to 10+   -> beam partial + a separate true-confusion

128-full (saturation operating point), identical leak-free held-out
(val_traj_ids.json, split_seed=1234). Teacher-forced BFS-gold walk,
reuse eval_neural model-load (no interface drift). No retraining.
"""
import sys, json, statistics
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import torch
from isre.symbolic.isre_ast import ASTNode
from isre.symbolic.symbolic_engine import SymbolicEngine
from scripts.eval_neural import load_model, _ACT

CKPT, HID = "checkpoints_128_full/best.pt", 128
BFS = Path("isre/trajectories_v6_bfs")

meta = json.loads((Path(CKPT).parent / "val_traj_ids.json").read_text())
val_ids = sorted(meta["val_traj_ids"])
dev = "cuda" if torch.cuda.is_available() else "cpu"
enc, pol = load_model(CKPT, HID, 4, dev)
engine = SymbolicEngine()
print(f"gold-rank diagnostic | 128-full | held-out {len(val_ids)} "
      f"(split_seed={meta['split_seed']})\n")


@torch.no_grad()
def gold_rank(root, gold_nid, gold_av):
    """Rank of the (gold_nid, gold_action) candidate in model score order.
    Returns (rank, n_candidates, argmax_is_gold). rank 1 = model's top."""
    craw = engine.get_candidates(root)
    if not craw:
        return None, 0, False
    cand = [(nid, a) for nid, _, a in craw]
    emb, _ = enc(root)
    scores = pol(emb, cand)
    if scores.numel() == 0:
        return None, 0, False
    order = torch.argsort(scores, descending=True).tolist()
    gi = None
    for j, (nid, a) in enumerate(cand):
        if nid == gold_nid and a.value == gold_av:
            gi = j
            break
    if gi is None:
        return None, len(cand), False           # gold not a candidate (rare)
    rank = order.index(gi) + 1
    return rank, len(cand), (rank == 1)


all_ranks, err_ranks = [], []
err_by_diff = Counter()
n_steps = 0

for tid in val_ids:
    f = tid if tid.endswith(".json") else f"{tid}.json"
    p = BFS / f
    if not p.exists():
        continue
    b = json.loads(p.read_text(encoding="utf-8"))
    diff = b["difficulty"]
    root = ASTNode.from_dict(b["original_ast"])
    root.mark_dirty(); root._rebuild_parents()
    for s in b["steps"]:
        r, ncand, is_top = gold_rank(root, s["gold_node_id"], s["gold_action"])
        if r is not None:
            n_steps += 1
            all_ranks.append(r)
            if not is_top:                       # model erred at this step
                err_ranks.append(r)
                err_by_diff[diff] += 1
        root = engine.apply(root, s["gold_node_id"], _ACT[s["gold_action"]])


def dist(xs):
    xs = sorted(xs)
    q = lambda p: xs[min(len(xs) - 1, int(p * len(xs)))]
    return (f"n={len(xs)} mean={statistics.fmean(xs):.2f} "
            f"median={statistics.median(xs):.0f} p75={q(.75)} "
            f"p90={q(.90)} p95={q(.95)} max={xs[-1]}")


per_step_err = len(err_ranks) / max(n_steps, 1)
print(f"steps={n_steps}  per-step error (argmax != gold) = "
      f"{len(err_ranks)}/{n_steps} = {per_step_err:.1%}\n")
print(f"gold rank, ALL steps   : {dist(all_ranks)}")
print(f"gold rank, ERROR steps : {dist(err_ranks)}\n")

e = err_ranks
n = len(e)
for k in (2, 3, 5):
    c = sum(1 for r in e if r <= k)
    print(f"  on error steps: gold rank <= {k}: {c}/{n} = {c/n:.1%}")
tail = sum(1 for r in e if r > 5)
print(f"  on error steps: gold rank  > 5: {tail}/{n} = {tail/n:.1%}\n")

# beam-k projected per-step error = errors NOT recovered by keeping top-k
for k in (2, 3, 5):
    unrec = sum(1 for r in e if r > k) / max(n_steps, 1)
    print(f"  projected per-step err with beam-{k} (gold must be in top-{k}): "
          f"{unrec:.1%}  (greedy={per_step_err:.1%})")

print("\nVERDICT:")
med = statistics.median(e) if e else 0
r2 = sum(1 for r in e if r <= 2) / n if n else 0
r3 = sum(1 for r in e if r <= 3) / n if n else 0
if med <= 2 and r3 >= 0.6:
    print(f"  BEAM-RECOVERABLE: median err-rank={med:.0f}, "
          f"{r3:.0%} of errors have gold in top-3. beam-2/3 should cut "
          f"compounding sharply. Code beam search (Experiment #3), no retrain.")
elif med >= 5:
    print(f"  NOT beam-recoverable: median err-rank={med:.0f} — model is far "
          f"from gold on errors. Beam won't fix; true confusion / different lever.")
else:
    print(f"  PARTIAL: median err-rank={med:.0f}, top-3={r3:.0%}. Beam helps "
          f"some; a residual true-confusion class remains. Worth beam + "
          f"separate analysis of the rank>5 tail.")
