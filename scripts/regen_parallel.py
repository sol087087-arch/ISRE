"""Parallel per-index trajectory regeneration.

Per-index seeding (generate_one_indexed) makes generation embarrassingly
parallel: each worker owns a disjoint index range and computes it
independently. Used for BOTH ablation arms:

  --gold-mode bfs       -> v6 BFS-optimal teacher (expensive: BFS-at-gen)
  --gold-mode recorded  -> naive recorded teacher (cheap, no BFS) — the
                           ablation's contaminated control arm

Same --seed + same idx => identical scrambled start in both modes
(perfect pairing for CE-on-recorded vs CE-on-BFS).

Usage:
  python scripts/regen_parallel.py --out isre/trajectories_v6_bfs \
      --n 50000 --workers 12 --seed 42 --gold-mode bfs
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from multiprocessing import Pool
from pathlib import Path

from isre.data.trajectory_generator import TrajectoryGenerator


def _worker(args):
    idx_lo, idx_hi, seed, gold_mode, max_degree, out_dir, max_depth, max_traj, bfs_budget = args
    gen = TrajectoryGenerator(
        seed=seed,
        max_ast_depth=max_depth,
        max_trajectory_length=max_traj,
        bfs_budget=bfs_budget,
    )
    out = Path(out_dir)
    written = 0
    dropped = 0
    for idx in range(idx_lo, idx_hi):
        path = out / f"traj_{idx:07d}.json"
        if path.exists():            # resumable: skip already-done indices
            written += 1
            continue
        traj = gen.generate_one_indexed(
            idx, seed_base=seed, gold_mode=gold_mode, max_degree=max_degree
        )
        if traj is None:
            dropped += 1
            continue
        tmp = out / f".tmp_{idx:07d}_{os.getpid()}.json"
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(asdict(traj), f, indent=2, ensure_ascii=False)
        tmp.replace(path)            # atomic: no half-written files on kill
        written += 1
    return (idx_lo, idx_hi, written, dropped, dict(gen.inverse_skip_counts))


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=50000)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gold-mode", choices=["bfs", "recorded"], default="bfs")
    ap.add_argument("--max-degree", type=int, default=4)
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--max-traj", type=int, default=6)
    ap.add_argument("--bfs-budget", type=int, default=100000)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    W = args.workers
    n = args.n
    chunk = (n + W - 1) // W
    tasks = []
    for w in range(W):
        lo = w * chunk
        hi = min(n, lo + chunk)
        if lo >= hi:
            break
        tasks.append((lo, hi, args.seed, args.gold_mode, args.max_degree,
                      str(out), args.max_depth, args.max_traj, args.bfs_budget))

    print(f"Regen: n={n} mode={args.gold_mode} workers={len(tasks)} "
          f"seed={args.seed} bfs_budget={args.bfs_budget} -> {out}")
    t0 = time.time()

    total_w = total_d = 0
    agg_skips: dict = {}
    with Pool(len(tasks)) as pool:
        for lo, hi, w_, d_, sk in pool.imap_unordered(_worker, tasks):
            total_w += w_
            total_d += d_
            for k, v in sk.items():
                agg_skips[k] = agg_skips.get(k, 0) + v
            el = time.time() - t0
            print(f"  chunk[{lo}:{hi}] done  written={w_} dropped={d_}  "
                  f"(elapsed {el/3600:.2f}h)")

    el = time.time() - t0
    print(f"\nDONE in {el/3600:.2f}h  written={total_w} dropped={total_d} "
          f"({total_d/max(n,1):.2%} drop)")
    if agg_skips:
        print("skip/drop counters:")
        for k, v in sorted(agg_skips.items(), key=lambda x: -x[1]):
            print(f"  {k:<32s} {v}")


if __name__ == "__main__":
    main()
