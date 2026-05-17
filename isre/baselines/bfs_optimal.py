"""BFS baseline — TRUE minimal step count (optimal path).

Why this exists:
  greedy reaches canonical 100% but is suboptimal on ~19% of trajectories
  (and 11.6% of gold/teacher paths are themselves non-minimal). So neither
  greedy steps nor teacher steps are a valid optimality reference. BFS over
  engine states gives the provably-minimal step count, which is the anchor
  for the v1 efficiency metric (step-overhead vs optimal).

Design:
- State identity = to_expr() (ORDER-SENSITIVE, same rationale as loop
  detection: engine dynamics are a function of the exact ordered AST;
  commutatively-equal-but-reordered ASTs are DIFFERENT engine states).
- Branching factor = all (node_id, action) from engine.get_candidates.
- BFS guarantees shortest path in #actions (uniform edge cost).
- Hard expansion budget per trajectory: BFS can blow up. On budget
  exhaustion we report BFS_TIMEOUT — NOT success and NOT failure. Honest
  about completeness limits; never claim optimal when search was truncated.

Usage:
    python -m isre.baselines.bfs_optimal --data isre/trajectories --n 500
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path

from isre.symbolic.isre_ast import ASTNode
from isre.symbolic.symbolic_engine import SymbolicEngine

SUCCESS = "SUCCESS"
UNREACHABLE = "UNREACHABLE"   # search space exhausted, canonical never found
BFS_TIMEOUT = "BFS_TIMEOUT"   # expansion budget hit before exhaustion


def bfs_optimal(
    start: ASTNode,
    canonical: ASTNode,
    engine: SymbolicEngine,
    max_expansions: int = 40000,
    max_depth: int = 20,
) -> tuple[str, int]:
    """Return (outcome, optimal_steps).

    outcome = SUCCESS  -> optimal_steps is the provably-minimal path length
            = UNREACHABLE -> exhausted without finding canonical (steps=-1)
            = BFS_TIMEOUT  -> budget hit, optimum unknown (steps=-1)
    """
    canon_expr = canonical.to_expr()

    root = start.clone()
    root.mark_dirty()
    root._rebuild_parents()

    if root.to_expr() == canon_expr:
        return SUCCESS, 0

    # queue of (node, depth); visited keyed by order-sensitive expr
    q: deque = deque()
    q.append((root, 0))
    visited: set[str] = {root.to_expr()}
    expansions = 0

    while q:
        node, depth = q.popleft()
        if depth >= max_depth:
            continue

        expansions += 1
        if expansions > max_expansions:
            return BFS_TIMEOUT, -1

        for nid, _, action in engine.get_candidates(node):
            child = engine.apply(node, nid, action)
            ce = child.to_expr()
            if ce == canon_expr:
                return SUCCESS, depth + 1
            if ce not in visited:
                visited.add(ce)
                q.append((child, depth + 1))

    return UNREACHABLE, -1


def bfs_optimal_path(
    start: ASTNode,
    canonical: ASTNode,
    engine: SymbolicEngine,
    max_expansions: int = 100000,
    max_depth: int = 20,
):
    """Return (outcome, path) where path is the BFS-shortest forward action
    sequence from start to canonical.

    path = [(state_ASTNode, node_id, action), ...]  — state is BEFORE the
    action; applying action to state yields the next state; the last action's
    result equals canonical (by order-sensitive to_expr()).

    Determinism: candidates are expanded in sorted (node_id, action.value)
    order, so the discovered shortest path is reproducible across runs —
    required for reproducible dataset regeneration.

    Memory: back-pointers keyed by expr string only (O(#states), not
    O(#states * AST)). The path is rematerialized by replaying the recorded
    actions from start (engine.apply is deterministic).

    outcome = SUCCESS | UNREACHABLE | BFS_TIMEOUT (path = [] unless SUCCESS).
    """
    canon_expr = canonical.to_expr()

    root = start.clone()
    root.mark_dirty()
    root._rebuild_parents()
    start_expr = root.to_expr()

    if start_expr == canon_expr:
        return SUCCESS, []

    # back[child_expr] = (parent_expr, node_id, action)
    back: dict[str, tuple[str, int, object]] = {}
    q: deque = deque()
    q.append((root, 0))
    visited: set[str] = {start_expr}
    expansions = 0
    found = False

    while q:
        node, depth = q.popleft()
        if depth >= max_depth:
            continue

        expansions += 1
        if expansions > max_expansions:
            return BFS_TIMEOUT, []

        cands = sorted(
            engine.get_candidates(node),
            key=lambda x: (x[0], x[2].value),
        )
        for nid, _, action in cands:
            child = engine.apply(node, nid, action)
            ce = child.to_expr()
            if ce not in visited:
                back[ce] = (node.to_expr(), nid, action)
                if ce == canon_expr:
                    found = True
                    break
                visited.add(ce)
                q.append((child, depth + 1))
        if found:
            break

    if not found:
        return UNREACHABLE, []

    # Reconstruct action sequence: walk back canon -> start, then reverse.
    seq: list[tuple[int, object]] = []
    cur = canon_expr
    while cur != start_expr:
        parent_expr, nid, action = back[cur]
        seq.append((nid, action))
        cur = parent_expr
    seq.reverse()

    # Rematerialize states by replaying actions from start (deterministic).
    path = []
    state = start.clone()
    state.mark_dirty()
    state._rebuild_parents()
    for nid, action in seq:
        path.append((state, nid, action))
        state = engine.apply(state, nid, action)
    # sanity: final state must equal canonical
    if state.to_expr() != canon_expr:
        return UNREACHABLE, []  # should never happen; defensive
    return SUCCESS, path


def evaluate(
    data_dir: str,
    n: int = 500,
    max_expansions: int = 40000,
    max_depth: int = 20,
) -> dict:
    engine = SymbolicEngine()
    files = sorted(Path(data_dir).glob("traj_*.json"))[:n]
    if not files:
        raise FileNotFoundError(f"No trajectories in {data_dir}")

    n_eval = len(files)
    counts = {SUCCESS: 0, UNREACHABLE: 0, BFS_TIMEOUT: 0}
    opt_steps_sum = 0
    by_diff: dict[int, list] = {}   # diff -> [n, n_success, sum_opt, sum_teacher]

    for f in files:
        with f.open(encoding="utf-8") as fh:
            traj = json.load(fh)
        start = ASTNode.from_dict(traj["original_ast"])
        canon = ASTNode.from_dict(traj["canonical_ast"])
        diff = traj["difficulty"]

        outcome, opt = bfs_optimal(
            start, canon, engine,
            max_expansions=max_expansions, max_depth=max_depth,
        )
        counts[outcome] += 1
        rec = by_diff.setdefault(diff, [0, 0, 0, 0])
        rec[0] += 1
        if outcome == SUCCESS:
            opt_steps_sum += opt
            rec[1] += 1
            rec[2] += opt
            rec[3] += diff  # teacher (gold path) length

    succ = counts[SUCCESS]
    return {
        "policy": "bfs_optimal",
        "n_evaluated": n_eval,
        "success_rate": succ / n_eval,
        "unreachable_rate": counts[UNREACHABLE] / n_eval,
        "timeout_rate": counts[BFS_TIMEOUT] / n_eval,
        "avg_optimal_steps": opt_steps_sum / max(succ, 1),
        "by_difficulty": {
            d: {
                "n": v[0],
                "success": v[1],
                "avg_optimal": v[2] / max(v[1], 1),
                "avg_teacher": v[3] / max(v[1], 1),
            }
            for d, v in sorted(by_diff.items())
        },
    }


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="isre/trajectories")
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--max-expansions", type=int, default=40000)
    parser.add_argument("--max-depth", type=int, default=20)
    args = parser.parse_args()

    r = evaluate(args.data, n=args.n,
                 max_expansions=args.max_expansions, max_depth=args.max_depth)
    print(f"Policy:            {r['policy']}")
    print(f"Evaluated:         {r['n_evaluated']}")
    print(f"Success rate:      {r['success_rate']:.1%}")
    print(f"Unreachable:       {r['unreachable_rate']:.1%}")
    print(f"BFS timeout:       {r['timeout_rate']:.1%}  "
          f"(budget={args.max_expansions} expansions)")
    print(f"Avg OPTIMAL steps: {r['avg_optimal_steps']:.2f}  (provably minimal)")
    print(f"\nPer-difficulty (optimal vs teacher/gold path length):")
    print(f"  {'diff':>4} {'n':>5} {'succ%':>7} {'optimal':>8} {'teacher':>8} {'gold_overhead':>14}")
    for d, v in r["by_difficulty"].items():
        if v["success"]:
            ov = v["avg_teacher"] - v["avg_optimal"]
            print(f"  {d:>4} {v['n']:>5} {v['success']/v['n']:>6.1%} "
                  f"{v['avg_optimal']:>8.2f} {v['avg_teacher']:>8.2f} {ov:>+14.2f}")
        else:
            print(f"  {d:>4} {v['n']:>5} {0:>6.1%} {'-':>8} {'-':>8} {'-':>14}")
