"""Overnight orchestrator.

Sequence:
  1. Poll GPU until utilization < GPU_UTIL_THRESHOLD %
  2. Run full training (50K, CUDA, hidden_dim=128, 20 epochs)
  3. Run compare_baselines on val trajectories
  4. Print summary

Usage:
    python scripts/orchestrate.py
"""
from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

GPU_UTIL_THRESHOLD = 20   # % — wait until GPU is this idle
GPU_POLL_INTERVAL  = 120  # seconds between checks
TRAIN_DATA         = "isre/trajectories"
TRAIN_EPOCHS       = 20
TRAIN_HIDDEN_DIM   = 128
TRAIN_NUM_ROUNDS   = 4
TRAIN_SAVE_DIR     = "checkpoints/mlp_v1_gpu"
BASELINE_N         = 2000


def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def gpu_utilization() -> int | None:
    """Return current GPU utilization %, or None if nvidia-smi unavailable."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            return int(result.stdout.strip().split("\n")[0])
    except Exception:
        pass
    return None


def wait_for_gpu() -> None:
    log(f"Waiting for GPU utilization < {GPU_UTIL_THRESHOLD}%...")
    while True:
        util = gpu_utilization()
        if util is None:
            log("nvidia-smi unavailable — skipping GPU wait, will use CPU.")
            return
        log(f"GPU utilization: {util}%")
        if util < GPU_UTIL_THRESHOLD:
            log("GPU is free.")
            return
        time.sleep(GPU_POLL_INTERVAL)


def run(cmd: list[str], label: str) -> int:
    log(f"START: {label}")
    log(f"  cmd: {' '.join(cmd)}")
    # Use sys.executable + inherit stdout/stderr so output flows to log
    result = subprocess.run(cmd, timeout=None)
    log(f"END:   {label}  (exit={result.returncode})")
    return result.returncode


def main() -> None:
    log("=== Overnight orchestrator started ===")

    # Step 1: wait for GPU
    wait_for_gpu()

    # Step 2: full training
    rc = run([
        sys.executable, "-m", "isre.training.train",
        "--data",             TRAIN_DATA,
        "--epochs",           str(TRAIN_EPOCHS),
        "--hidden-dim",       str(TRAIN_HIDDEN_DIM),
        "--num-rounds",       str(TRAIN_NUM_ROUNDS),
        "--device",           "auto",
        "--save-dir",         TRAIN_SAVE_DIR,
        "--accumulation-steps", "8",
    ], label=f"MLP training ({TRAIN_EPOCHS} epochs, 50K, hidden={TRAIN_HIDDEN_DIM})")

    if rc != 0:
        log(f"ERROR: training failed with exit code {rc}. Stopping.")
        sys.exit(rc)

    # Step 3: baseline comparison (on first BASELINE_N trajectories as proxy val set)
    run([
        sys.executable, "scripts/compare_baselines.py",
        "--data", TRAIN_DATA,
        "--n",    str(BASELINE_N),
    ], label=f"Baseline comparison (n={BASELINE_N})")

    # Step 4: summary
    best_pt = Path(TRAIN_SAVE_DIR) / "best.pt"
    log("=== Orchestrator done ===")
    log(f"  Checkpoint: {best_pt} ({'exists' if best_pt.exists() else 'MISSING'})")
    log("  Next step: implement KAN head and run Experiment 1.")


if __name__ == "__main__":
    main()
