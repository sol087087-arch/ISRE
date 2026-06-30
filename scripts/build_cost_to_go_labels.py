"""Build offline cost-to-go candidate labels for ranking training.

For each training state, every valid candidate action is applied once and the
remaining BFS distance to the canonical form is measured from that child state.
All candidates with the minimal finite remaining distance are marked optimal.

The output is JSONL so it can be appended, inspected, cached, and reused without
changing the original trajectory files.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from isre.baselines.bfs_optimal import SUCCESS, bfs_optimal
from isre.symbolic.isre_ast import ASTNode
from isre.symbolic.symbolic_engine import ActionType, SymbolicEngine


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _candidate_costs(
    *,
    engine: SymbolicEngine,
    state: ASTNode,
    canonical: ASTNode,
    canonical_expr: str,
    candidates: list[tuple[int, ActionType]],
    cache: dict[tuple[str, str], tuple[str, int]],
    max_expansions: int,
    max_depth: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for index, (node_id, action) in enumerate(candidates):
        child = engine.apply(state, node_id, action)
        child_expr = child.to_expr()

        if child_expr == canonical_expr:
            outcome, distance = SUCCESS, 0
        else:
            key = (child_expr, canonical_expr)
            if key in cache:
                outcome, distance = cache[key]
            else:
                outcome, distance = bfs_optimal(
                    child,
                    canonical,
                    engine,
                    max_expansions=max_expansions,
                    max_depth=max_depth,
                )
                cache[key] = (outcome, distance)

        rows.append({
            "index": index,
            "node_id": node_id,
            "action": action.value,
            "child_expr": child_expr,
            "bfs_outcome": outcome,
            "remaining_bfs_distance": distance if outcome == SUCCESS else None,
        })

    finite = [
        row["remaining_bfs_distance"]
        for row in rows
        if row["remaining_bfs_distance"] is not None
    ]
    best = min(finite) if finite else None
    for row in rows:
        row["is_optimal"] = (
            best is not None and row["remaining_bfs_distance"] == best
        )
    return rows


def build_labels(
    *,
    data_dir: Path,
    output: Path,
    manifest: Path,
    max_files: int | None,
    max_steps: int | None,
    max_expansions: int,
    max_depth: int,
) -> dict[str, Any]:
    files = sorted(data_dir.glob("traj_*.json"))
    if max_files is not None:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"No trajectory JSON files found in {data_dir}")

    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)

    engine = SymbolicEngine()
    action_map = {a.value: a for a in ActionType}
    cache: dict[tuple[str, str], tuple[str, int]] = {}

    started = time.time()
    state_count = 0
    candidate_count = 0
    labeled_state_count = 0
    multi_optimal_count = 0
    fallback_gold_not_optimal = 0
    skipped_steps = 0

    with output.open("w", encoding="utf-8", newline="\n") as out:
        for file_index, path in enumerate(files, start=1):
            traj = _load_json(path)
            canonical = ASTNode.from_dict(traj["canonical_ast"])
            canonical_expr = traj["canonical_expr"]
            trajectory_id = traj.get("trajectory_id", path.stem)

            for step_index, step_data in enumerate(traj["steps"]):
                if max_steps is not None and state_count >= max_steps:
                    break

                try:
                    state = ASTNode.from_dict(step_data["state"])
                    state.mark_dirty()
                    state._rebuild_parents()
                    raw_candidates = engine.get_candidates(state)
                    candidates = [
                        (node_id, action)
                        for node_id, _, action in raw_candidates
                    ]
                    gold_action = action_map[step_data["gold_action"]]
                    gold_node_id = int(step_data["gold_node_id"])
                except Exception:
                    skipped_steps += 1
                    continue

                rows = _candidate_costs(
                    engine=engine,
                    state=state,
                    canonical=canonical,
                    canonical_expr=canonical_expr,
                    candidates=candidates,
                    cache=cache,
                    max_expansions=max_expansions,
                    max_depth=max_depth,
                )
                optimal_indices = [
                    row["index"] for row in rows if row["is_optimal"]
                ]
                gold_index = next(
                    (
                        i for i, (node_id, action) in enumerate(candidates)
                        if node_id == gold_node_id and action == gold_action
                    ),
                    None,
                )

                record = {
                    "trajectory_file": path.name,
                    "trajectory_id": trajectory_id,
                    "step_index": step_index,
                    "state_expr": step_data.get("state_expr", state.to_expr()),
                    "canonical_expr": canonical_expr,
                    "difficulty": traj.get("difficulty"),
                    "complexity": step_data.get("complexity"),
                    "gold_index": gold_index,
                    "gold_node_id": gold_node_id,
                    "gold_action": gold_action.value,
                    "optimal_indices": optimal_indices,
                    "candidate_costs": rows,
                }
                out.write(json.dumps(record, ensure_ascii=True) + "\n")

                state_count += 1
                candidate_count += len(rows)
                if optimal_indices:
                    labeled_state_count += 1
                if len(optimal_indices) > 1:
                    multi_optimal_count += 1
                if (
                    gold_index is not None
                    and optimal_indices
                    and gold_index not in optimal_indices
                ):
                    fallback_gold_not_optimal += 1

            if max_steps is not None and state_count >= max_steps:
                break
            if file_index % 100 == 0:
                elapsed = time.time() - started
                rate = state_count / max(elapsed, 1e-9)
                print(
                    f"processed files={file_index} states={state_count} "
                    f"rate={rate:.2f} states/s cache={len(cache)}",
                    flush=True,
                )

    elapsed = time.time() - started
    summary = {
        "data_dir": str(data_dir),
        "output": str(output),
        "files_seen": len(files),
        "states_written": state_count,
        "candidates_written": candidate_count,
        "states_with_finite_label": labeled_state_count,
        "multi_optimal_states": multi_optimal_count,
        "recorded_gold_not_cost_optimal": fallback_gold_not_optimal,
        "skipped_steps": skipped_steps,
        "cache_entries": len(cache),
        "max_expansions": max_expansions,
        "max_depth": max_depth,
        "elapsed_seconds": elapsed,
        "states_per_second": state_count / max(elapsed, 1e-9),
    }
    manifest.write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build offline cost-to-go labels for all candidate actions."
    )
    parser.add_argument("--data", required=True, help="Trajectory directory.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument(
        "--manifest",
        default=None,
        help="Summary JSON path. Defaults to <output>.manifest.json",
    )
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-expansions", type=int, default=20000)
    parser.add_argument("--max-depth", type=int, default=20)
    args = parser.parse_args()

    output = Path(args.output)
    manifest = (
        Path(args.manifest)
        if args.manifest
        else output.with_suffix(output.suffix + ".manifest.json")
    )
    summary = build_labels(
        data_dir=Path(args.data),
        output=output,
        manifest=manifest,
        max_files=args.max_files,
        max_steps=args.max_steps,
        max_expansions=args.max_expansions,
        max_depth=args.max_depth,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
