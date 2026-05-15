"""Compare baselines (and eventually neural policy) on a trajectory dataset.

Fixed table format — columns are frozen so neural policy slots in without
restructuring. All columns present for every policy; placeholder "—" for
metrics that don't apply (e.g. std for deterministic greedy).

Columns:
  policy | n_seeds | success_rate | failure_rate | avg_steps_success |
  avg_steps_all | avg_teacher_steps | pct_faster_teacher | loop_rate

Usage:
    python scripts/compare_baselines.py [--data isre/trajectories] [--n 1000]
    python scripts/compare_baselines.py --n 1000 --seeds 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isre.baselines.random_policy import evaluate_multi_seed as random_multi
from isre.baselines.greedy_heuristic import evaluate as greedy_eval


_COLUMNS = [
    ("policy",              "Policy",              "<20"),
    ("n_seeds",             "Seeds",               ">6"),
    ("success_rate",        "Succ%",               ">9"),
    ("failure_rate",        "Fail%",               ">9"),
    ("avg_steps_success",   "Steps|Succ",          ">11"),
    ("avg_steps_all",       "Steps|All",           ">11"),
    ("avg_teacher_steps",   "Teacher",             ">9"),
    ("pct_faster_teacher",  "FasterTchr",          ">11"),
    ("loop_rate",           "LoopRate",            ">10"),
]


def _pct(v) -> str:
    if v == "—":
        return "—"
    return f"{v:.1%}"


def _f2(v) -> str:
    if v == "—":
        return "—"
    return f"{v:.2f}"


def _row_from_random(r: dict) -> dict:
    sr   = r["success_rate_med"]
    sr_s = r["success_rate_std"]
    st   = r["avg_steps_on_success_med"]
    st_s = r["avg_steps_on_success_std"]
    sa   = r["avg_steps_all_med"]
    sa_s = r["avg_steps_all_std"]
    return {
        "policy":             f"random",
        "n_seeds":            f"{r['n_seeds']} seeds",
        "success_rate":       f"{sr:.1%}±{sr_s:.1%}",
        "failure_rate":       f"{1-sr:.1%}",
        "avg_steps_success":  f"{st:.2f}±{st_s:.2f}",
        "avg_steps_all":      f"{sa:.2f}±{sa_s:.2f}",
        "avg_teacher_steps":  f"{r['avg_teacher_steps']:.2f}",
        "pct_faster_teacher": f"{r['pct_faster_than_teacher_med']:.1%}",
        "loop_rate":          f"{r['loop_rate_med']:.1%}",
    }


def _row_from_greedy(r: dict) -> dict:
    return {
        "policy":             "greedy",
        "n_seeds":            "1 (det.)",
        "success_rate":       f"{r['success_rate']:.1%}",
        "failure_rate":       f"{r['failure_rate']:.1%}",
        "avg_steps_success":  f"{r['avg_steps_on_success']:.2f}",
        "avg_steps_all":      f"{r['avg_steps_all']:.2f}",
        "avg_teacher_steps":  f"{r['avg_teacher_steps']:.2f}",
        "pct_faster_teacher": f"{r['pct_faster_than_teacher']:.1%}",
        "loop_rate":          f"{r['loop_rate']:.1%}",
    }


def _neural_placeholder() -> dict:
    return {
        "policy":             "neural (MLP)",
        "n_seeds":            "—",
        "success_rate":       "—",
        "failure_rate":       "—",
        "avg_steps_success":  "—",
        "avg_steps_all":      "—",
        "avg_teacher_steps":  "—",
        "pct_faster_teacher": "—",
        "loop_rate":          "—",
    }


def print_table(rows: list[dict]) -> None:
    # Header
    header = ""
    sep = ""
    for key, label, fmt in _COLUMNS:
        header += format(label, fmt) + "  "
        sep    += "-" * int(fmt[1:]) + "  "
    print(header)
    print(sep)
    for row in rows:
        line = ""
        for key, label, fmt in _COLUMNS:
            line += format(row.get(key, "—"), fmt) + "  "
        print(line)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="isre/trajectories")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, default=5,
                        help="Number of seeds for random policy")
    args = parser.parse_args()

    print(f"Evaluating {args.n} trajectories  (max_steps={args.max_steps})\n")

    seeds = list(range(args.seed, args.seed + args.seeds))

    print("Running random (multi-seed)...")
    r_rand = random_multi(args.data, n=args.n, max_steps=args.max_steps, seeds=seeds)

    print("Running greedy (deterministic)...")
    r_greedy = greedy_eval(args.data, n=args.n, max_steps=args.max_steps)

    rows = [
        _row_from_random(r_rand),
        _row_from_greedy(r_greedy),
        _neural_placeholder(),
    ]

    print()
    print_table(rows)

    # Per-difficulty breakdown for greedy
    print(f"\nGreedy per-difficulty:")
    for diff, (s, total) in r_greedy["by_difficulty"].items():
        print(f"  diff={diff}  {s:4d}/{total:4d}  ({s/total:.0%})")


if __name__ == "__main__":
    main()
