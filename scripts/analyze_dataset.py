"""Diversity + dedup analysis for a generated trajectory dataset.

Reports:
  - distribution of gold forward actions (what the model actually learns)
  - distribution of inverse_sequence elements (what the generator produced)
  - distribution of difficulty (trajectory length)
  - distribution of canonical polynomial structure (degree, n_terms)
  - duplicate (state, gold_node_id, gold_action) training triples
  - state cycles within trajectories

Usage:
    python scripts/analyze_dataset.py [--data isre/trajectories]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def state_key(state_dict: dict) -> str:
    """Stable canonical key for an AST dict — used for dedup and cycle detection."""
    return json.dumps(state_dict, sort_keys=True, ensure_ascii=False)


def main(data_dir: str) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    path = Path(data_dir)
    files = sorted(path.glob("traj_*.json"))
    if not files:
        print(f"No trajectories found in {data_dir}")
        return 1

    gold_actions: Counter = Counter()
    inverse_uses: Counter = Counter()
    difficulties: Counter = Counter()
    canonical_degrees: Counter = Counter()
    canonical_n_terms: Counter = Counter()

    train_triples: Counter = Counter()  # (state_key, gold_node_id, gold_action) → count
    canonical_states: Counter = Counter()
    scrambled_states: Counter = Counter()

    n_trajectories_with_cycles = 0
    n_total_pairs = 0

    for f in files:
        with f.open(encoding="utf-8") as fh:
            traj = json.load(fh)

        difficulties[traj["difficulty"]] += 1
        for inv in traj["inverse_sequence"]:
            inverse_uses[inv] += 1

        canonical_states[state_key(traj["canonical_ast"])] += 1
        scrambled_states[state_key(traj["original_ast"])] += 1

        # Canonical structure: top-level is Add(terms...) or single term
        canon = traj["canonical_ast"]
        if canon["type"] == "ADD":
            canonical_n_terms[len(canon["children"])] += 1
        else:
            canonical_n_terms[1] += 1
        canonical_degrees[_max_degree(canon)] += 1

        # Per-step training triples + cycle detection
        seen_states_in_traj: set[str] = set()
        had_cycle = False
        for step in traj["steps"]:
            sk = state_key(step["state"])
            if sk in seen_states_in_traj:
                had_cycle = True
            seen_states_in_traj.add(sk)

            triple = (sk, step["gold_node_id"], step["gold_action"])
            train_triples[triple] += 1
            gold_actions[step["gold_action"]] += 1
            n_total_pairs += 1

        if had_cycle:
            n_trajectories_with_cycles += 1

    # ── Report ──────────────────────────────────────────
    print(f"=== Dataset summary: {len(files)} trajectories, {n_total_pairs} training pairs ===\n")

    _print_dist("Gold forward actions (what the policy learns)", gold_actions, total=n_total_pairs)
    _print_dist("Inverse transforms used during generation", inverse_uses)
    _print_dist("Trajectory difficulty (= len(steps))", difficulties, total=len(files))
    _print_dist("Canonical polynomial degree", canonical_degrees, total=len(files))
    _print_dist("Canonical polynomial #terms", canonical_n_terms, total=len(files))

    # Dedup
    n_unique_canonical = len(canonical_states)
    n_unique_scrambled = len(scrambled_states)
    n_unique_triples = len(train_triples)
    n_dup_triples = n_total_pairs - n_unique_triples

    print("=== Dedup ===")
    print(f"  Unique canonical expressions:  {n_unique_canonical:6d} / {len(files)} "
          f"({n_unique_canonical/len(files):.1%})")
    print(f"  Unique scrambled expressions:  {n_unique_scrambled:6d} / {len(files)} "
          f"({n_unique_scrambled/len(files):.1%})")
    print(f"  Unique training triples:       {n_unique_triples:6d} / {n_total_pairs} "
          f"({n_unique_triples/n_total_pairs:.1%})")
    print(f"  Duplicate training pairs:      {n_dup_triples:6d} ({n_dup_triples/n_total_pairs:.1%})")

    # Top 10 most-duplicated triples — usually the "easy" canonical-adjacent states
    most_dup = train_triples.most_common(10)
    print(f"\n  Top 10 duplicated training triples:")
    for (sk, nid, action), count in most_dup:
        if count == 1:
            break
        # Decode the state for human view
        try:
            state = json.loads(sk)
            expr = _expr_from_dict(state)
        except Exception:
            expr = "<unparseable>"
        print(f"    x{count:4d}  node={nid:2d}  action={action:18s}  state={expr}")

    print(f"\n=== Cycles ===")
    print(f"  Trajectories containing a state cycle (A→B→A): {n_trajectories_with_cycles} "
          f"({n_trajectories_with_cycles/len(files):.1%})")

    return 0


def _print_dist(title: str, counter: Counter, total: int | None = None) -> None:
    if total is None:
        total = sum(counter.values())
    print(f"=== {title} ===")
    for k, v in sorted(counter.items(), key=lambda x: -x[1]):
        pct = v / total if total else 0.0
        print(f"  {str(k):28s} {v:6d}  ({pct:.1%})")
    print()


def _max_degree(ast_dict: dict) -> int:
    """Walk the AST and find the largest exponent on a Pow over x."""
    best = 1 if ast_dict["type"] in ("VARIABLE",) else 0
    if ast_dict["type"] == "POW":
        try:
            exp = int(float(ast_dict["children"][1]["value"]))
            best = max(best, exp)
        except (ValueError, KeyError, IndexError, TypeError):
            pass
    for c in ast_dict.get("children", []):
        best = max(best, _max_degree(c))
    return best


def _expr_from_dict(d: dict) -> str:
    t = d["type"]
    if t in ("NUMBER", "VARIABLE", "CONST"):
        return d["value"]
    if t == "ADD":
        return "(" + " + ".join(_expr_from_dict(c) for c in d["children"]) + ")"
    if t == "MUL":
        return "(" + " * ".join(_expr_from_dict(c) for c in d["children"]) + ")"
    if t == "POW":
        return f"({_expr_from_dict(d['children'][0])})^{_expr_from_dict(d['children'][1])}"
    return "?"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="isre/trajectories", help="Trajectory directory")
    args = parser.parse_args()
    sys.exit(main(args.data))
