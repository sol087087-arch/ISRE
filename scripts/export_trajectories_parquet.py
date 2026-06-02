"""Export ISRE trajectory JSON files to sharded Parquet for dataset hosting.

The raw trajectory folders contain one JSON file per trajectory. That is nice
for generation/resume, but inconvenient for Hugging Face Datasets. This script
keeps the trajectory as a single row and stores nested fields as JSON strings,
so no information is lost and the export stays simple to load.

Example:
  python scripts/export_trajectories_parquet.py \
      --data isre/trajectories_v7_bfs \
      --out hf_exports/isre_v7_bfs \
      --rows-per-shard 5000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


SCHEMA = pa.schema([
    ("trajectory_id", pa.string()),
    ("canonical_expr", pa.string()),
    ("canonical_ast_json", pa.string()),
    ("original_expr", pa.string()),
    ("original_ast_json", pa.string()),
    ("steps_json", pa.string()),
    ("difficulty", pa.int32()),
    ("inverse_sequence_json", pa.string()),
    ("n_steps", pa.int32()),
])


def _row_from_traj(traj: dict) -> dict:
    steps = traj.get("steps", [])
    return {
        "trajectory_id": traj.get("trajectory_id", ""),
        "canonical_expr": traj.get("canonical_expr", ""),
        "canonical_ast_json": json.dumps(
            traj.get("canonical_ast"), ensure_ascii=False, sort_keys=True
        ),
        "original_expr": traj.get("original_expr", ""),
        "original_ast_json": json.dumps(
            traj.get("original_ast"), ensure_ascii=False, sort_keys=True
        ),
        "steps_json": json.dumps(steps, ensure_ascii=False),
        "difficulty": int(traj.get("difficulty", len(steps))),
        "inverse_sequence_json": json.dumps(
            traj.get("inverse_sequence", []), ensure_ascii=False
        ),
        "n_steps": len(steps),
    }


def _write_shard(rows: list[dict], out_dir: Path, shard_idx: int) -> Path:
    table = pa.Table.from_pylist(rows, schema=SCHEMA)
    path = out_dir / f"train-{shard_idx:05d}.parquet"
    pq.write_table(table, path, compression="zstd")
    return path


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Folder with traj_*.json")
    ap.add_argument("--out", required=True, help="Output folder for parquet shards")
    ap.add_argument("--rows-per-shard", type=int, default=5000)
    args = ap.parse_args()

    data_dir = Path(args.data)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(data_dir.glob("traj_*.json"))
    if not files:
        print(f"No trajectory JSON files found in {data_dir}", file=sys.stderr)
        return 1

    rows: list[dict] = []
    shards: list[dict] = []
    total_rows = 0
    total_steps = 0

    for path in files:
        traj = json.loads(path.read_text(encoding="utf-8"))
        row = _row_from_traj(traj)
        rows.append(row)
        total_rows += 1
        total_steps += row["n_steps"]

        if len(rows) >= args.rows_per_shard:
            shard_path = _write_shard(rows, out_dir, len(shards))
            shards.append({"file": shard_path.name, "rows": len(rows)})
            print(f"wrote {shard_path.name}: {len(rows)} rows")
            rows = []

    if rows:
        shard_path = _write_shard(rows, out_dir, len(shards))
        shards.append({"file": shard_path.name, "rows": len(rows)})
        print(f"wrote {shard_path.name}: {len(rows)} rows")

    manifest = {
        "source_dir": str(data_dir),
        "num_trajectories": total_rows,
        "num_training_pairs": total_steps,
        "rows_per_shard": args.rows_per_shard,
        "format": "parquet",
        "compression": "zstd",
        "schema": [(field.name, str(field.type)) for field in SCHEMA],
        "shards": shards,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(
        f"\nDONE: {total_rows} trajectories, {total_steps} training pairs, "
        f"{len(shards)} shards -> {out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
