"""Mine hard states for cost-to-go ranking fine-tuning.

This script rolls a trained policy over BFS-generated trajectories, computes
cost-to-go labels for each visited state, and keeps states where the current
model is wrong or low-margin. It writes two append-only artifacts:

1. A JSONL cost-label cache compatible with train.py --cost-labels.
2. A tiny trajectory directory where each hard state is one trainable step.

Use --split train for fine-tuning data. The split is read from the checkpoint's
val_traj_ids.json, so held-out evaluation trajectories are not mined by default.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch

from isre.baselines.bfs_optimal import SUCCESS
from isre.symbolic.isre_ast import ASTNode
from isre.symbolic.symbolic_engine import SymbolicEngine
from scripts.build_cost_to_go_labels import _candidate_costs
from scripts.eval_neural import _score, load_model


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _softmax(xs: list[float]) -> list[float]:
    if not xs:
        return []
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    z = sum(exps)
    return [x / z for x in exps]


def _read_split_ids(ckpt: Path) -> set[str]:
    split_path = ckpt.parent / "val_traj_ids.json"
    if not split_path.exists():
        raise FileNotFoundError(
            f"{split_path} missing. Refusing to guess held-out split."
        )
    meta = _load_json(split_path)
    return set(meta["val_traj_ids"])


def _candidate_pairs(rows: list[dict[str, Any]]) -> list[tuple[int, str]]:
    return [(int(row["node_id"]), row["action"]) for row in rows]


def _gold_from_optimal(
    rows: list[dict[str, Any]],
    optimal_indices: list[int],
) -> tuple[int, str, int]:
    """Pick a deterministic representative gold action for trajectory format.

    The ranking loss still uses all optimal candidates from the JSONL cache.
    This single gold is only a compatibility target for CE fallback.
    """
    if not optimal_indices:
        raise ValueError("Cannot export hard trajectory without an optimal action.")
    best = sorted(
        (
            int(rows[i]["node_id"]),
            rows[i]["action"],
            i,
        )
        for i in optimal_indices
    )[0]
    node_id, action, index = best
    return node_id, action, index


@torch.no_grad()
def mine_hard_states(
    *,
    ckpt: Path,
    data_dir: Path,
    output_labels: Path,
    output_trajectories: Path,
    manifest: Path,
    hidden_dim: int,
    num_rounds: int,
    policy: str,
    policy_hidden_dim: int | None,
    action_emb_dim: int | None,
    kan_hidden: int,
    kan_action_emb_dim: int | None,
    kan_bottleneck_dim: int | None,
    kan_depth: int | None,
    kan_grid: int | None,
    kan_spline_order: int | None,
    device: str,
    split: str,
    max_trajectories: int | None,
    max_steps_per_rollout: int,
    max_hard_states: int | None,
    max_expansions: int,
    max_depth: int,
    margin_threshold: float,
    include_low_margin: bool,
    include_correct_multi_optimal: bool,
) -> dict[str, Any]:
    output_labels.parent.mkdir(parents=True, exist_ok=True)
    output_trajectories.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)

    device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    enc, pol = load_model(
        str(ckpt),
        hidden_dim,
        num_rounds,
        device,
        policy_kind=policy,
        kan_hidden=kan_hidden,
        policy_hidden_dim=policy_hidden_dim,
        action_emb_dim=action_emb_dim,
        kan_action_emb_dim=kan_action_emb_dim,
        kan_bottleneck_dim=kan_bottleneck_dim,
        kan_depth=kan_depth,
        kan_grid=kan_grid,
        kan_spline_order=kan_spline_order,
    )
    engine = SymbolicEngine()
    val_ids = _read_split_ids(ckpt)
    files = sorted(data_dir.glob("traj_*.json"))
    selected = []
    for path in files:
        tid = path.stem
        is_val = tid in val_ids
        if split == "train" and is_val:
            continue
        if split == "val" and not is_val:
            continue
        selected.append(path)
    if max_trajectories is not None:
        selected = selected[:max_trajectories]
    if not selected:
        raise FileNotFoundError(f"No trajectories selected from {data_dir}")

    started = time.time()
    bfs_cache: dict[tuple[str, str], tuple[str, int]] = {}
    seen_states: set[tuple[str, str]] = set()

    trajectories_seen = 0
    rollout_states_seen = 0
    labeled_states = 0
    hard_states_written = 0
    model_wrong = 0
    low_margin = 0
    multi_optimal = 0
    loops = 0
    dead_ends = 0
    successes = 0
    timeouts = 0
    reason_counts: dict[str, int] = {}

    with output_labels.open("w", encoding="utf-8", newline="\n") as labels:
        for path in selected:
            if max_hard_states is not None and hard_states_written >= max_hard_states:
                break
            traj = _load_json(path)
            trajectories_seen += 1
            trajectory_id = traj.get("trajectory_id", path.stem)
            canonical = ASTNode.from_dict(traj["canonical_ast"])
            canonical_expr = traj["canonical_expr"]
            root = ASTNode.from_dict(traj["original_ast"])
            root.mark_dirty()
            root._rebuild_parents()
            visited = set()

            for rollout_step in range(max_steps_per_rollout):
                state_expr = root.to_expr()
                if state_expr == canonical_expr:
                    successes += 1
                    break
                if state_expr in visited:
                    loops += 1
                    break
                visited.add(state_expr)
                rollout_states_seen += 1

                raw_candidates = engine.get_candidates(root)
                if not raw_candidates:
                    dead_ends += 1
                    break
                candidates = [(node_id, action) for node_id, _, action in raw_candidates]
                scores_t = _score(enc, pol, root, candidates)
                if scores_t.numel() == 0:
                    dead_ends += 1
                    break
                scores = [float(x) for x in scores_t.detach().cpu().tolist()]
                probs = _softmax(scores)
                top_index = int(scores_t.argmax().item())

                state_key = (state_expr, canonical_expr)
                if state_key in seen_states:
                    node_id, action = candidates[top_index]
                    root = engine.apply(root, node_id, action)
                    continue
                seen_states.add(state_key)

                rows = _candidate_costs(
                    engine=engine,
                    state=root,
                    canonical=canonical,
                    canonical_expr=canonical_expr,
                    candidates=candidates,
                    cache=bfs_cache,
                    max_expansions=max_expansions,
                    max_depth=max_depth,
                )
                optimal_indices = [row["index"] for row in rows if row["is_optimal"]]
                if not optimal_indices:
                    node_id, action = candidates[top_index]
                    root = engine.apply(root, node_id, action)
                    continue

                labeled_states += 1
                if len(optimal_indices) > 1:
                    multi_optimal += 1

                best_opt_score = max(scores[i] for i in optimal_indices)
                finite_non_opt = [
                    i for i, row in enumerate(rows)
                    if i not in optimal_indices
                    and row["remaining_bfs_distance"] is not None
                ]
                best_non_opt_score = (
                    max(scores[i] for i in finite_non_opt)
                    if finite_non_opt
                    else None
                )
                margin = (
                    best_opt_score - best_non_opt_score
                    if best_non_opt_score is not None
                    else None
                )

                reasons = []
                if top_index not in optimal_indices:
                    reasons.append("model_top_not_cost_optimal")
                    model_wrong += 1
                if (
                    include_low_margin
                    and margin is not None
                    and margin <= margin_threshold
                ):
                    reasons.append("low_margin")
                    low_margin += 1
                if include_correct_multi_optimal and len(optimal_indices) > 1:
                    reasons.append("multi_optimal")

                keep = bool(reasons)
                if keep:
                    for reason in reasons:
                        reason_counts[reason] = reason_counts.get(reason, 0) + 1
                    gold_node_id, gold_action, gold_index = _gold_from_optimal(
                        rows,
                        optimal_indices,
                    )
                    record = {
                        "trajectory_file": path.name,
                        "trajectory_id": trajectory_id,
                        "source": "hard_state_rollout",
                        "source_split": split,
                        "rollout_step": rollout_step,
                        "state_expr": state_expr,
                        "canonical_expr": canonical_expr,
                        "difficulty": traj.get("difficulty"),
                        "complexity": None,
                        "gold_index": gold_index,
                        "gold_node_id": gold_node_id,
                        "gold_action": gold_action,
                        "optimal_indices": optimal_indices,
                        "model_top_index": top_index,
                        "model_top_node_id": int(candidates[top_index][0]),
                        "model_top_action": candidates[top_index][1].value,
                        "model_top_is_optimal": top_index in optimal_indices,
                        "best_opt_score": best_opt_score,
                        "best_non_opt_score": best_non_opt_score,
                        "score_margin": margin,
                        "hard_reasons": reasons,
                        "candidate_scores": scores,
                        "candidate_probs": probs,
                        "candidate_costs": rows,
                    }
                    labels.write(json.dumps(record, ensure_ascii=True) + "\n")
                    labels.flush()

                    hard_id = f"hard_{hard_states_written:07d}"
                    hard_traj = {
                        "trajectory_id": hard_id,
                        "canonical_expr": canonical_expr,
                        "canonical_ast": canonical.to_dict(),
                        "original_expr": state_expr,
                        "original_ast": root.to_dict(),
                        "steps": [{
                            "state": root.to_dict(),
                            "state_expr": state_expr,
                            "candidate_actions": [
                                [int(node_id), action.value]
                                for node_id, action in candidates
                            ],
                            "gold_action": gold_action,
                            "gold_node_id": gold_node_id,
                            "complexity": 0,
                            "hard_reasons": reasons,
                            "source_trajectory_id": trajectory_id,
                            "source_rollout_step": rollout_step,
                        }],
                        "difficulty": max(
                            1,
                            min(
                                row["remaining_bfs_distance"]
                                for row in rows
                                if row["remaining_bfs_distance"] is not None
                            ) + 1,
                        ),
                        "inverse_sequence": [],
                        "source": "hard_state_rollout",
                    }
                    out_path = output_trajectories / f"traj_{hard_states_written:07d}.json"
                    out_path.write_text(
                        json.dumps(hard_traj, indent=2, ensure_ascii=True) + "\n",
                        encoding="utf-8",
                    )
                    hard_states_written += 1

                if max_hard_states is not None and hard_states_written >= max_hard_states:
                    break

                node_id, action = candidates[top_index]
                root = engine.apply(root, node_id, action)
            else:
                timeouts += 1

            if trajectories_seen % 50 == 0:
                elapsed = time.time() - started
                print(
                    f"trajectories={trajectories_seen} hard={hard_states_written} "
                    f"labeled={labeled_states} rate={rollout_states_seen / max(elapsed, 1e-9):.2f} states/s",
                    flush=True,
                )

    elapsed = time.time() - started
    summary = {
        "ckpt": str(ckpt),
        "data_dir": str(data_dir),
        "split": split,
        "output_labels": str(output_labels),
        "output_trajectories": str(output_trajectories),
        "trajectories_seen": trajectories_seen,
        "rollout_states_seen": rollout_states_seen,
        "unique_states_seen": len(seen_states),
        "labeled_states": labeled_states,
        "hard_states_written": hard_states_written,
        "model_top_not_cost_optimal": model_wrong,
        "low_margin_states": low_margin,
        "multi_optimal_labeled_states": multi_optimal,
        "reason_counts": reason_counts,
        "successes": successes,
        "loops": loops,
        "dead_ends": dead_ends,
        "timeouts": timeouts,
        "bfs_cache_entries": len(bfs_cache),
        "max_expansions": max_expansions,
        "max_depth": max_depth,
        "margin_threshold": margin_threshold,
        "elapsed_seconds": elapsed,
        "rollout_states_per_second": rollout_states_seen / max(elapsed, 1e-9),
    }
    manifest.write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    ap = argparse.ArgumentParser(description="Mine hard states for ranking fine-tune.")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--output-labels", required=True)
    ap.add_argument("--output-trajectories", required=True)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--hidden-dim", type=int, required=True)
    ap.add_argument("--num-rounds", type=int, default=4)
    ap.add_argument("--policy", choices=["mlp", "kan", "kan_ae", "encoder_kan", "encoder_deep_kan"], default="mlp")
    ap.add_argument("--policy-hidden-dim", type=int, default=None)
    ap.add_argument("--action-emb-dim", type=int, default=None)
    ap.add_argument("--kan-hidden", type=int, default=16)
    ap.add_argument("--kan-action-emb-dim", type=int, default=None)
    ap.add_argument("--kan-bottleneck-dim", type=int, default=None)
    ap.add_argument("--kan-depth", type=int, default=None)
    ap.add_argument("--kan-grid", type=int, default=None)
    ap.add_argument("--kan-spline-order", type=int, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--split", choices=["train", "val", "all"], default="train")
    ap.add_argument("--max-trajectories", type=int, default=None)
    ap.add_argument("--max-steps-per-rollout", type=int, default=30)
    ap.add_argument("--max-hard-states", type=int, default=None)
    ap.add_argument("--max-expansions", type=int, default=20000)
    ap.add_argument("--max-depth", type=int, default=20)
    ap.add_argument("--margin-threshold", type=float, default=0.25)
    ap.add_argument("--no-low-margin", action="store_true")
    ap.add_argument("--include-correct-multi-optimal", action="store_true")
    args = ap.parse_args()

    output_labels = Path(args.output_labels)
    manifest = (
        Path(args.manifest)
        if args.manifest
        else output_labels.with_suffix(output_labels.suffix + ".manifest.json")
    )
    summary = mine_hard_states(
        ckpt=Path(args.ckpt),
        data_dir=Path(args.data),
        output_labels=output_labels,
        output_trajectories=Path(args.output_trajectories),
        manifest=manifest,
        hidden_dim=args.hidden_dim,
        num_rounds=args.num_rounds,
        policy=args.policy,
        policy_hidden_dim=args.policy_hidden_dim,
        action_emb_dim=args.action_emb_dim,
        kan_hidden=args.kan_hidden,
        kan_action_emb_dim=args.kan_action_emb_dim,
        kan_bottleneck_dim=args.kan_bottleneck_dim,
        kan_depth=args.kan_depth,
        kan_grid=args.kan_grid,
        kan_spline_order=args.kan_spline_order,
        device=args.device,
        split=args.split,
        max_trajectories=args.max_trajectories,
        max_steps_per_rollout=args.max_steps_per_rollout,
        max_hard_states=args.max_hard_states,
        max_expansions=args.max_expansions,
        max_depth=args.max_depth,
        margin_threshold=args.margin_threshold,
        include_low_margin=not args.no_low_margin,
        include_correct_multi_optimal=args.include_correct_multi_optimal,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
