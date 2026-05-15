"""Greedy heuristic baseline: at each step picks the action that reduces
AST complexity the most. Fully deterministic — no rng.

Design notes:
- Success verified against canonical_ast ground truth (to_expr() comparison).
- avg_steps_on_success: steps only on successful rollouts.
- avg_steps_all: total steps / n_eval (includes max_steps for failures).
- failure_rate: explicit.
- Loop detection: visited set tracks state expressions. loop_detected=True is
  a diagnostic flag only — rollout continues to max_steps regardless.
  Greedy does NOT escape loops by design: it is a dumb baseline.
  Changing this would make it "greedy + anti-loop heuristic", a different agent.
  Oscillating trajectories are counted as failures, loop_rate is reported.
- Tiebreak tuple: (complexity, priority, node_id, action.value) — fully
  deterministic even if node_id ever ties across different actions.

Usage:
    python -m isre.baselines.greedy_heuristic --data isre/trajectories --n 500
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from isre.symbolic.isre_ast import ASTNode
from isre.symbolic.symbolic_engine import SymbolicEngine

_ACTION_PRIORITY = [
    "FOLD_CONST",
    "REMOVE_ZERO",
    "REMOVE_ONE",
    "FLATTEN_ADD",
    "FLATTEN_MUL",
    "COMBINE_COEFF",
    "COLLECT_TERMS",
    "MERGE_POWER",
    "EXPAND",
    "SORT_COMMUTATIVE",
]


def _priority(action_name: str) -> int:
    try:
        return _ACTION_PRIORITY.index(action_name)
    except ValueError:
        return len(_ACTION_PRIORITY)


def _canonical_reached(root: ASTNode, canonical: ASTNode) -> bool:
    """Ground-truth check against known canonical expression."""
    return root.to_expr() == canonical.to_expr()


def rollout(
    start: ASTNode,
    canonical: ASTNode,
    engine: SymbolicEngine,
    max_steps: int = 20,
) -> tuple[bool, int, bool]:
    """Run greedy policy until canonical or step limit.

    Greedy criterion: minimise complexity after apply.
    Tiebreak: (priority, node_id, action.value) — fully deterministic.

    Returns (success, n_steps_taken, loop_detected).
    loop_detected is diagnostic only — rollout is never interrupted by it.
    """
    root = start.clone()
    root.mark_dirty()
    root._rebuild_parents()

    visited: set[str] = set()
    loop_detected = False

    for step in range(max_steps):
        if _canonical_reached(root, canonical):
            return True, step, loop_detected

        state_key = root.to_expr()
        if state_key in visited:
            loop_detected = True   # diagnostic flag — we keep going
        visited.add(state_key)

        candidates = engine.get_candidates(root)
        if not candidates:
            break

        best = min(
            candidates,
            key=lambda x: (
                engine.apply(root, x[0], x[2]).complexity(),
                _priority(x[2].value),
                x[0],
                x[2].value,
            ),
        )
        root = engine.apply(root, best[0], best[2])

    return _canonical_reached(root, canonical), max_steps, loop_detected


def evaluate(
    data_dir: str,
    n: int = 500,
    max_steps: int = 20,
) -> dict:
    engine = SymbolicEngine()
    files = sorted(Path(data_dir).glob("traj_*.json"))[:n]
    if not files:
        raise FileNotFoundError(f"No trajectories in {data_dir}")

    successes = 0
    success_steps = 0
    total_steps = 0
    teacher_steps_sum = 0
    steps_below_teacher = 0
    loops = 0
    by_diff: dict[int, list[bool]] = {}
    n_eval = len(files)

    for f in files:
        with f.open(encoding="utf-8") as fh:
            traj = json.load(fh)

        start = ASTNode.from_dict(traj["original_ast"])
        canonical = ASTNode.from_dict(traj["canonical_ast"])
        difficulty = traj["difficulty"]

        success, steps, loop = rollout(start, canonical, engine, max_steps=max_steps)

        total_steps += steps
        if success:
            successes += 1
            success_steps += steps
            if steps < difficulty:
                steps_below_teacher += 1
        if loop:
            loops += 1
        teacher_steps_sum += difficulty
        by_diff.setdefault(difficulty, []).append(success)

    return {
        "policy": "greedy_heuristic",
        "n_seeds": 1,
        "n_evaluated": n_eval,
        "success_rate": successes / n_eval,
        "failure_rate": (n_eval - successes) / n_eval,
        "avg_steps_on_success": success_steps / max(successes, 1),
        "avg_steps_all": total_steps / n_eval,
        "avg_teacher_steps": teacher_steps_sum / n_eval,
        "pct_faster_than_teacher": steps_below_teacher / max(successes, 1),
        "loop_rate": loops / n_eval,
        "by_difficulty": {d: (sum(v), len(v)) for d, v in sorted(by_diff.items())},
    }


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="isre/trajectories")
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=20)
    args = parser.parse_args()

    r = evaluate(args.data, n=args.n, max_steps=args.max_steps)
    print(f"Policy:              {r['policy']}  (n_seeds=1, deterministic)")
    print(f"Evaluated:           {r['n_evaluated']}")
    print(f"Success rate:        {r['success_rate']:.1%}")
    print(f"Failure rate:        {r['failure_rate']:.1%}")
    print(f"Avg steps (success): {r['avg_steps_on_success']:.2f}")
    print(f"Avg steps (all):     {r['avg_steps_all']:.2f}")
    print(f"Avg teacher steps:   {r['avg_teacher_steps']:.2f}")
    print(f"Faster than teacher: {r['pct_faster_than_teacher']:.1%} of successes")
    print(f"Loop rate:           {r['loop_rate']:.1%}")
    print(f"Per-difficulty:")
    for diff, (s, total) in r["by_difficulty"].items():
        print(f"  diff={diff}  {s:4d}/{total:4d}  ({s/total:.0%})")
