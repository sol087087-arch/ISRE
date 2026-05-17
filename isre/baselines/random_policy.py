"""Random baseline: picks a uniformly random valid candidate at each step.

Design notes:
- Per-trajectory rng: Random(seed + traj_idx). Different traj_idx values
  always produce different RNG states (Python ints don't overflow, so
  seed+0 != seed+1 etc. is guaranteed). Any single trajectory can be
  re-run in isolation and reproduce the same result.
- Candidates sorted before rng.choice to guarantee reproducibility even if
  engine internals change iteration order.
- Success verified against canonical_ast ground truth (to_expr() comparison),
  not indirectly via empty candidate list (which false-positives on engine gaps).
- avg_steps_on_success: steps only on successful rollouts.
- avg_steps_all: total steps over all rollouts / n_eval (includes max_steps
  for failures). Measures compute cost of the baseline.
- failure_rate: explicit, not inferred.
- loop_rate: random rarely loops (uniform choice over candidates), reported
  for completeness and symmetry with greedy.
- Multi-seed report uses median±std (median more robust than mean at n=5).

Usage:
    python -m isre.baselines.random_policy --data isre/trajectories --n 500
    python -m isre.baselines.random_policy --data isre/trajectories --n 500 --seeds 5
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

from isre.symbolic.isre_ast import ASTNode
from isre.symbolic.symbolic_engine import SymbolicEngine


def _canonical_reached(root: ASTNode, canonical: ASTNode) -> bool:
    """Ground-truth check: compare against known canonical, not candidate count."""
    return root.to_expr() == canonical.to_expr()


def rollout(
    start: ASTNode,
    canonical: ASTNode,
    engine: SymbolicEngine,
    rng: random.Random,
    max_steps: int = 20,
) -> tuple[bool, int, bool]:
    """Run random policy from start until canonical or step limit.

    Returns (success, n_steps_taken, loop_detected).
    n_steps_taken = steps to canonical on success, max_steps on failure.
    loop_detected is diagnostic only — does not alter policy behaviour.
    """
    root = start.clone()
    root.mark_dirty()
    root._rebuild_parents()

    visited: set[str] = set()
    loop_detected = False

    for step in range(max_steps):
        if _canonical_reached(root, canonical):
            return True, step, loop_detected

        # Order-SENSITIVE key (to_expr), deliberately NOT canonicalized.
        # SORT_COMMUTATIVE is a real action whose entire purpose is to move
        # between commutatively-equivalent orderings (1+x -> x+1). An
        # order-invariant key would flag every legitimate SORT as a loop
        # (empirically: random SUCCESS 98%->36%, LOOP 0%->64% — all false).
        # A genuine loop closes when an EXACT ordered state repeats, which
        # to_expr() catches correctly.
        state_key = root.to_expr()
        if state_key in visited:
            loop_detected = True
        visited.add(state_key)

        candidates = sorted(
            engine.get_candidates(root),
            key=lambda x: (x[0], x[2].value),
        )
        if not candidates:
            break

        node_id, _, action = rng.choice(candidates)
        root = engine.apply(root, node_id, action)

    return _canonical_reached(root, canonical), max_steps, loop_detected


def evaluate(
    data_dir: str,
    n: int = 500,
    max_steps: int = 20,
    seed: int = 0,
) -> dict:
    engine = SymbolicEngine()
    files = sorted(Path(data_dir).glob("traj_*.json"))[:n]
    if not files:
        raise FileNotFoundError(f"No trajectories in {data_dir}")

    successes = 0
    success_steps = 0    # steps only on successful rollouts
    total_steps = 0      # steps over all rollouts (max_steps for failures)
    teacher_steps_sum = 0
    steps_below_teacher = 0
    loops = 0
    n_eval = len(files)

    for traj_idx, f in enumerate(files):
        with f.open(encoding="utf-8") as fh:
            traj = json.load(fh)

        start = ASTNode.from_dict(traj["original_ast"])
        canonical = ASTNode.from_dict(traj["canonical_ast"])
        difficulty = traj["difficulty"]

        # Per-trajectory rng: seed+traj_idx is unique for every trajectory
        # (Python ints have no overflow, so no hash collisions possible).
        traj_rng = random.Random(seed + traj_idx)

        success, steps, loop = rollout(
            start, canonical, engine, traj_rng, max_steps=max_steps
        )

        total_steps += steps
        if success:
            successes += 1
            success_steps += steps
            if steps < difficulty:
                steps_below_teacher += 1
        if loop:
            loops += 1
        teacher_steps_sum += difficulty

    return {
        "policy": "random",
        "seed": seed,
        "n_evaluated": n_eval,
        "success_rate": successes / n_eval,
        "failure_rate": (n_eval - successes) / n_eval,
        "avg_steps_on_success": success_steps / max(successes, 1),
        "avg_steps_all": total_steps / n_eval,
        "avg_teacher_steps": teacher_steps_sum / n_eval,
        "pct_faster_than_teacher": steps_below_teacher / max(successes, 1),
        "loop_rate": loops / n_eval,
    }


def evaluate_multi_seed(
    data_dir: str,
    n: int = 500,
    max_steps: int = 20,
    seeds: list[int] | None = None,
) -> dict:
    """Run evaluate() over multiple seeds. Report median±std."""
    if seeds is None:
        seeds = list(range(5))

    runs = [evaluate(data_dir, n=n, max_steps=max_steps, seed=s) for s in seeds]

    def med_std(key: str) -> tuple[float, float]:
        vals = [r[key] for r in runs]
        return statistics.median(vals), statistics.stdev(vals) if len(vals) > 1 else 0.0

    sr_med, sr_std   = med_std("success_rate")
    st_med, st_std   = med_std("avg_steps_on_success")
    sa_med, sa_std   = med_std("avg_steps_all")
    ptr_med, _       = med_std("pct_faster_than_teacher")
    lr_med, _        = med_std("loop_rate")

    return {
        "policy": "random",
        "n_seeds": len(seeds),
        "seeds": seeds,
        "n_evaluated": runs[0]["n_evaluated"],
        "success_rate_med": sr_med,
        "success_rate_std": sr_std,
        "failure_rate_med": 1.0 - sr_med,
        "avg_steps_on_success_med": st_med,
        "avg_steps_on_success_std": st_std,
        "avg_steps_all_med": sa_med,
        "avg_steps_all_std": sa_std,
        "avg_teacher_steps": runs[0]["avg_teacher_steps"],
        "pct_faster_than_teacher_med": ptr_med,
        "loop_rate_med": lr_med,
    }


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="isre/trajectories")
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, default=1,
                        help="Number of seeds (uses seed, seed+1, ...)")
    args = parser.parse_args()

    if args.seeds > 1:
        seeds = list(range(args.seed, args.seed + args.seeds))
        r = evaluate_multi_seed(args.data, n=args.n, max_steps=args.max_steps, seeds=seeds)
        print(f"Policy:              {r['policy']}  (n_seeds={r['n_seeds']}, seeds={r['seeds']})")
        print(f"Evaluated:           {r['n_evaluated']}")
        print(f"Success rate:        {r['success_rate_med']:.1%}  (median) ± {r['success_rate_std']:.1%}")
        print(f"Failure rate:        {r['failure_rate_med']:.1%}  (median)")
        print(f"Avg steps (success): {r['avg_steps_on_success_med']:.2f} ± {r['avg_steps_on_success_std']:.2f}")
        print(f"Avg steps (all):     {r['avg_steps_all_med']:.2f} ± {r['avg_steps_all_std']:.2f}")
        print(f"Avg teacher steps:   {r['avg_teacher_steps']:.2f}")
        print(f"Faster than teacher: {r['pct_faster_than_teacher_med']:.1%} of successes")
        print(f"Loop rate:           {r['loop_rate_med']:.1%}")
    else:
        r = evaluate(args.data, n=args.n, max_steps=args.max_steps, seed=args.seed)
        print(f"Policy:              {r['policy']}  (seed={r['seed']})")
        print(f"Evaluated:           {r['n_evaluated']}")
        print(f"Success rate:        {r['success_rate']:.1%}")
        print(f"Failure rate:        {r['failure_rate']:.1%}")
        print(f"Avg steps (success): {r['avg_steps_on_success']:.2f}")
        print(f"Avg steps (all):     {r['avg_steps_all']:.2f}")
        print(f"Avg teacher steps:   {r['avg_teacher_steps']:.2f}")
        print(f"Faster than teacher: {r['pct_faster_than_teacher']:.1%} of successes")
        print(f"Loop rate:           {r['loop_rate']:.1%}")
