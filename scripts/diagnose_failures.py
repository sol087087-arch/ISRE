"""Diagnose why random policy fails.

Classifies each failure into one of three categories:
  ENGINE_GAP   — get_candidates() returned [] but state != canonical.
                 This is a symbolic engine bug: valid transforms exist but
                 engine doesn't know them. Contaminates training data.
  LOOP         — policy revisited an already-seen state (oscillation).
                 Healthy failure mode: random is expected to loop.
  MAX_STEPS    — hit step limit without looping or hitting a dead-end.
                 Policy just wandered too long.

Also checks both baselines for engine gaps since gaps are policy-independent.

Usage:
    python scripts/diagnose_failures.py --data isre/trajectories --n 500
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

from isre.symbolic.isre_ast import ASTNode
from isre.symbolic.symbolic_engine import SymbolicEngine


ENGINE_GAP = "ENGINE_GAP"
LOOP       = "LOOP"
MAX_STEPS  = "MAX_STEPS"
SUCCESS    = "SUCCESS"


def classify_rollout(
    start: ASTNode,
    canonical: ASTNode,
    engine: SymbolicEngine,
    rng: random.Random,
    max_steps: int = 20,
) -> tuple[str, str]:
    """Run random policy, return (outcome, failure_class).

    outcome: SUCCESS | ENGINE_GAP | LOOP | MAX_STEPS
    On success: failure_class = ""
    """
    root = start.clone()
    root.mark_dirty()
    root._rebuild_parents()

    visited: set[str] = set()

    for step in range(max_steps):
        expr = root.to_expr()

        if expr == canonical.to_expr():
            return SUCCESS, ""

        candidates = sorted(
            engine.get_candidates(root),
            key=lambda x: (x[0], x[2].value),
        )

        if not candidates:
            return ENGINE_GAP, expr  # dead-end: engine has no move but not canonical

        # Order-SENSITIVE: SORT_COMMUTATIVE legitimately moves between
        # commutatively-equivalent orderings; an order-invariant key would
        # false-flag every SORT step as a loop.
        if expr in visited:
            # detected before picking next move — policy will loop
            return LOOP, expr

        visited.add(expr)
        node_id, _, action = rng.choice(candidates)
        root = engine.apply(root, node_id, action)

    # Hit max_steps — check final state
    final_expr = root.to_expr()
    if final_expr == canonical.to_expr():
        return SUCCESS, ""
    final_candidates = engine.get_candidates(root)
    if not final_candidates:
        return ENGINE_GAP, final_expr
    return MAX_STEPS, final_expr


def classify_greedy_failure(
    start: ASTNode,
    canonical: ASTNode,
    engine: SymbolicEngine,
    max_steps: int = 20,
) -> tuple[str, str]:
    """Run greedy policy, classify failure."""
    from isre.baselines.greedy_heuristic import _priority

    root = start.clone()
    root.mark_dirty()
    root._rebuild_parents()

    visited: set[str] = set()

    for step in range(max_steps):
        expr = root.to_expr()

        if expr == canonical.to_expr():
            return SUCCESS, ""

        candidates = engine.get_candidates(root)
        if not candidates:
            return ENGINE_GAP, expr

        # Order-SENSITIVE (see classify_rollout note).
        if expr in visited:
            return LOOP, expr

        visited.add(expr)
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

    final_expr = root.to_expr()
    if final_expr == canonical.to_expr():
        return SUCCESS, ""
    if not engine.get_candidates(root):
        return ENGINE_GAP, final_expr
    return MAX_STEPS, final_expr


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="isre/trajectories")
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--show-examples", type=int, default=3,
                        help="Number of examples to show per failure class")
    args = parser.parse_args()

    engine = SymbolicEngine()
    files = sorted(Path(args.data).glob("traj_*.json"))[:args.n]
    if not files:
        print(f"No trajectories in {args.data}")
        sys.exit(1)

    n = len(files)

    # ── Random diagnosis ─────────────────────────────────────────────────────
    rand_counts: Counter = Counter()
    rand_examples: dict[str, list] = {}

    for traj_idx, f in enumerate(files):
        with f.open(encoding="utf-8") as fh:
            traj = json.load(fh)

        start     = ASTNode.from_dict(traj["original_ast"])
        canonical = ASTNode.from_dict(traj["canonical_ast"])
        traj_rng  = random.Random(args.seed + traj_idx)

        outcome, stuck_expr = classify_rollout(
            start, canonical, engine, traj_rng, max_steps=args.max_steps
        )
        rand_counts[outcome] += 1

        if outcome != SUCCESS and outcome not in rand_examples:
            rand_examples[outcome] = {
                "file": f.name,
                "canonical": traj["canonical_expr"],
                "original": traj["original_expr"],
                "stuck_at": stuck_expr,
                "difficulty": traj["difficulty"],
            }

    # ── Greedy diagnosis ─────────────────────────────────────────────────────
    greedy_counts: Counter = Counter()
    greedy_examples: dict[str, list] = {}

    for f in files:
        with f.open(encoding="utf-8") as fh:
            traj = json.load(fh)

        start     = ASTNode.from_dict(traj["original_ast"])
        canonical = ASTNode.from_dict(traj["canonical_ast"])

        outcome, stuck_expr = classify_greedy_failure(
            start, canonical, engine, max_steps=args.max_steps
        )
        greedy_counts[outcome] += 1

        if outcome != SUCCESS and outcome not in greedy_examples:
            greedy_examples[outcome] = {
                "file": f.name,
                "canonical": traj["canonical_expr"],
                "original": traj["original_expr"],
                "stuck_at": stuck_expr,
                "difficulty": traj["difficulty"],
            }

    # ── Report ────────────────────────────────────────────────────────────────
    print(f"=== Failure diagnosis on {n} trajectories ===\n")

    print("--- RANDOM policy ---")
    for cls in [SUCCESS, ENGINE_GAP, LOOP, MAX_STEPS]:
        c = rand_counts[cls]
        print(f"  {cls:12s}  {c:5d}  ({c/n:.1%})")

    engine_gap_rand = rand_counts[ENGINE_GAP]
    if engine_gap_rand == 0:
        print("\n  [OK] No engine gaps detected for random policy.")
    else:
        print(f"\n  [!!] {engine_gap_rand} engine gaps ({engine_gap_rand/n:.1%}) — "
              f"engine dead-ends that are NOT canonical. REQUIRES INVESTIGATION.")

    print("\n--- GREEDY policy ---")
    for cls in [SUCCESS, ENGINE_GAP, LOOP, MAX_STEPS]:
        c = greedy_counts[cls]
        print(f"  {cls:12s}  {c:5d}  ({c/n:.1%})")

    engine_gap_greedy = greedy_counts[ENGINE_GAP]
    if engine_gap_greedy == 0:
        print("\n  [OK] No engine gaps detected for greedy policy.")
    else:
        print(f"\n  [!!] {engine_gap_greedy} engine gaps ({engine_gap_greedy/n:.1%}) — "
              f"same dead-ends hit by greedy. Policy-independent bug.")

    # Engine gap rate (policy-independent lower bound)
    # If both policies hit same gap states, the gap rate is an inherent property
    gap_rate_estimate = min(engine_gap_rand, engine_gap_greedy) / n
    print(f"\n  Estimated policy-independent engine gap rate: "
          f"~{gap_rate_estimate:.1%} (lower bound)")
    print(f"  Random failure attributable to policy:  "
          f"~{(rand_counts[LOOP] + rand_counts[MAX_STEPS])/n:.1%}")

    # Examples
    print("\n--- ENGINE_GAP examples (random) ---")
    for cls, ex in rand_examples.items():
        if cls == ENGINE_GAP:
            print(f"  file:      {ex['file']}")
            print(f"  canonical: {ex['canonical']}")
            print(f"  original:  {ex['original']}")
            print(f"  stuck at:  {ex['stuck_at']}")
            print(f"  difficulty:{ex['difficulty']}")
            print()

    print("--- ENGINE_GAP examples (greedy) ---")
    for cls, ex in greedy_examples.items():
        if cls == ENGINE_GAP:
            print(f"  file:      {ex['file']}")
            print(f"  canonical: {ex['canonical']}")
            print(f"  original:  {ex['original']}")
            print(f"  stuck at:  {ex['stuck_at']}")
            print(f"  difficulty:{ex['difficulty']}")
            print()


if __name__ == "__main__":
    main()
