"""Neural policy evaluation harness.

Two modes, both per-trajectory on a held-out split:

MODE A — free rollout (PRIMARY metric):
  model picks argmax action until canonical or max_steps.
  bfs_steps = the v6_bfs trajectory difficulty (BFS-provably-optimal by
  construction of v6_bfs gold). overhead = mlp_steps - bfs_steps.
  Reports: success_rate, overhead {mean,p25,p50,p75,p95},
           bfs_optimal_rate (% overhead==0), catastrophic_rate (% >=3).
  Distribution (not just mean) distinguishes sharp-competence from
  heavy-tail models — needed for KAN collapse diagnostics.

MODE B — teacher-forced step agreement + 3-way divergence:
  walk the BFS-optimal gold path; at each state compare model argmax to
  bfs_action (the gold there) and to teacher_action (the NAIVE recorded
  arm's action at the SAME state, looked up by state_expr in the paired
  v6_recorded trajectory — no off-path BFS needed; the arms are pairing-
  verified to share the start).
  Categories per divergence:
    smarter_than_recorded : mlp == bfs  != recorded
    imitated_recorded     : mlp == recorded != bfs
    true_error            : mlp != bfs  and mlp != recorded
  + per-action confusion (gold_action -> what model picked).
  This is the instrument that decides v1.5 (BFS-relabel) vs v2 (ranking/RL):
  dominant 'imitated_recorded' => teacher signal is the problem (relabel
  helps); dominant 'true_error' => model capacity/training (relabel won't).

Usage:
  python scripts/eval_neural.py --ckpt checkpoints_x/best.pt --hidden-dim 128 \
      --bfs-data isre/trajectories_v6_bfs --recorded-data isre/trajectories_v6_recorded \
      --val-split 0.1 --seed 0 --max-steps 30 --n 2000
"""
from __future__ import annotations

import argparse, json, random, statistics, sys
from collections import Counter, defaultdict
from pathlib import Path

import torch

from isre.symbolic.isre_ast import ASTNode
from isre.symbolic.symbolic_engine import SymbolicEngine, ActionType
from isre.learning.encoder import ASTEncoder
from isre.learning.policy import PolicyNetwork

_ACT = {a.value: a for a in ActionType}


def load_model(ckpt_path: str, hidden_dim: int, num_rounds: int, device: str):
    enc = ASTEncoder(hidden_dim=hidden_dim, num_rounds=num_rounds).to(device)
    pol = PolicyNetwork(node_emb_dim=hidden_dim * 2, variant="mlp",
                        hidden_dim=hidden_dim).to(device)
    ck = torch.load(ckpt_path, map_location=device)
    enc.load_state_dict(ck["encoder"])
    pol.load_state_dict(ck["policy"])
    enc.eval(); pol.eval()
    return enc, pol


@torch.no_grad()
def _argmax_action(enc, pol, root: ASTNode, engine: SymbolicEngine):
    cands_raw = engine.get_candidates(root)
    if not cands_raw:
        return None, None, []
    cand = [(nid, a) for nid, _, a in cands_raw]
    emb, _ = enc(root)
    scores = pol(emb, cand)
    if scores.numel() == 0:
        return None, None, cand
    idx = int(torch.argmax(scores).item())
    return cand[idx][0], cand[idx][1], cand


@torch.no_grad()
def free_rollout(enc, pol, start: ASTNode, canon_expr: str,
                 engine: SymbolicEngine, max_steps: int):
    root = start.clone(); root.mark_dirty(); root._rebuild_parents()
    seen = set()
    for step in range(max_steps):
        if root.to_expr() == canon_expr:
            return True, step
        nid, act, cand = _argmax_action(enc, pol, root, engine)
        if act is None:
            return False, step          # engine dead-end (should be ~0 on v6)
        e = root.to_expr()
        if e in seen:                   # order-sensitive loop guard (POSTMORTEM)
            return False, step
        seen.add(e)
        root = engine.apply(root, nid, act)
    return root.to_expr() == canon_expr, max_steps


def quantiles(xs):
    if not xs:
        return {}
    s = sorted(xs)
    def q(p): return s[min(len(s) - 1, int(p * len(s)))]
    return {"mean": round(statistics.fmean(s), 3),
            "p25": q(.25), "p50": q(.50), "p75": q(.75), "p95": q(.95)}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--hidden-dim", type=int, required=True)
    ap.add_argument("--num-rounds", type=int, default=4)
    ap.add_argument("--bfs-data", default="isre/trajectories_v6_bfs")
    ap.add_argument("--recorded-data", default="isre/trajectories_v6_recorded")
    ap.add_argument("--val-split", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=30)
    ap.add_argument("--n", type=int, default=2000, help="cap eval trajectories")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else args.device
    print(f"Device: {device}  ckpt: {args.ckpt}  hidden: {args.hidden_dim}")

    enc, pol = load_model(args.ckpt, args.hidden_dim, args.num_rounds, device)
    engine = SymbolicEngine()

    bfs_dir = Path(args.bfs_data)
    rec_dir = Path(args.recorded_data)

    # EXPLICIT held-out dependency: read the exact trajectory ids the
    # trainer held out (written next to the checkpoint as val_traj_ids.json).
    # Fail LOUD if absent — never re-derive / shuffle-guess (the old
    # "reproduce split by reshuffling" was a lie: it shuffled id strings
    # while the trainer shuffled steps -> contaminated eval, POSTMORTEM #7).
    vpath = Path(args.ckpt).parent / "val_traj_ids.json"
    if not vpath.exists():
        sys.exit(f"FATAL: {vpath} missing. This checkpoint was trained "
                 f"without the trajectory-level split (or pre-#7). A clean "
                 f"held-out eval is impossible; re-train with current "
                 f"train.py. NOT falling back to a guessed split.")
    meta = json.loads(vpath.read_text(encoding="utf-8"))
    val_set = set(meta["val_traj_ids"])
    # Map ids (stored as traj.trajectory_id, e.g. 'traj_0001234') to files.
    all_files = {p.stem: p for p in bfs_dir.glob("traj_*.json")}
    val_ids = sorted(tid for tid in val_set if tid in all_files)
    if args.n:
        val_ids = val_ids[: args.n]
    print(f"Held-out: {len(val_set)} val trajectories "
          f"(split_seed={meta.get('split_seed')}); evaluating {len(val_ids)}\n")

    succ = 0
    overheads = []
    fail_diff = Counter()
    # Mode B accumulators
    cat = Counter()                       # divergence categories
    confusion = defaultdict(Counter)      # bfs_action -> model pick
    gold_steps_total = 0
    gold_steps_match = 0

    for tid in val_ids:
        fname = tid if tid.endswith(".json") else f"{tid}.json"
        b = json.loads((bfs_dir / fname).read_text(encoding="utf-8"))
        start = ASTNode.from_dict(b["original_ast"])
        canon = b["canonical_expr"]
        bfs_len = b["difficulty"]         # = BFS-optimal by construction

        # ---- MODE A: free rollout ----
        ok, steps = free_rollout(enc, pol, start, canon, engine, args.max_steps)
        if ok:
            succ += 1
            overheads.append(steps - bfs_len)
        else:
            fail_diff[bfs_len] += 1

        # ---- MODE B: teacher-forced step agreement on BFS gold path ----
        rec_path = rec_dir / fname
        rec_by_expr = {}
        if rec_path.exists():
            r = json.loads(rec_path.read_text(encoding="utf-8"))
            for s in r["steps"]:
                rec_by_expr[s["state_expr"]] = s["gold_action"]

        root = start.clone(); root.mark_dirty(); root._rebuild_parents()
        for s in b["steps"]:
            bfs_act = s["gold_action"]
            nid, act, cand = _argmax_action(enc, pol, root, engine)
            if act is None:
                break
            mlp_act = act.value
            gold_steps_total += 1
            if mlp_act == bfs_act:
                gold_steps_match += 1
            confusion[bfs_act][mlp_act] += 1
            # 3-way categorization (teacher = recorded action at THIS state)
            teach = rec_by_expr.get(s["state_expr"])
            if mlp_act != bfs_act:
                if teach is None:
                    cat["div_teacher_undefined"] += 1
                elif mlp_act == teach and teach != bfs_act:
                    cat["imitated_recorded"] += 1
                elif mlp_act != teach:
                    cat["true_error"] += 1
            else:  # mlp == bfs
                if teach is not None and teach != bfs_act:
                    cat["smarter_than_recorded"] += 1
                else:
                    cat["agree_optimal"] += 1
            # advance ALONG THE GOLD path (teacher-forced), not model's pick
            root = engine.apply(root, s["gold_node_id"],
                                _ACT[s["gold_action"]])

    n = len(val_ids)
    print("=== MODE A: free rollout (primary) ===")
    print(f"  success_rate      : {succ}/{n} = {succ/n:.1%}")
    print(f"  overhead vs BFS   : {quantiles(overheads)}")
    if overheads:
        opt = sum(1 for o in overheads if o == 0) / len(overheads)
        cat3 = sum(1 for o in overheads if o >= 3) / len(overheads)
        print(f"  bfs_optimal_rate  : {opt:.1%}  (overhead==0 on successes)")
        print(f"  catastrophic_rate : {cat3:.1%}  (overhead>=3 on successes)")
    if fail_diff:
        print(f"  failures by bfs_len: {dict(sorted(fail_diff.items()))}")

    print("\n=== MODE B: gold-path step agreement ===")
    if gold_steps_total:
        print(f"  step_match (mlp==bfs_optimal): "
              f"{gold_steps_match}/{gold_steps_total} = "
              f"{gold_steps_match/gold_steps_total:.1%}")
    tot_div = sum(v for k, v in cat.items() if k != "agree_optimal")
    print(f"  divergence categories (of {tot_div} divergent steps):")
    for k in ["smarter_than_recorded", "imitated_recorded", "true_error",
              "div_teacher_undefined"]:
        c = cat.get(k, 0)
        print(f"    {k:<22s} {c:6d}  "
              f"({c/tot_div:.1%})" if tot_div else f"    {k}: 0")
    print("\n  per-action confusion (bfs_gold -> model pick), "
          "top mismatches:")
    for ga in sorted(confusion):
        row = confusion[ga]
        miss = sum(v for k, v in row.items() if k != ga)
        if miss:
            top = sorted(((v, k) for k, v in row.items() if k != ga),
                         reverse=True)[:3]
            tops = ", ".join(f"{k}:{v}" for v, k in top)
            print(f"    {ga:<18s} {miss} mismatches -> {tops}")


if __name__ == "__main__":
    main()
