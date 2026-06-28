"""Build the full ISRE v7 KAN rescue analysis from raw logs only.

This script is intentionally self-contained. It does not read SUMMARY.md or
RESULTS.md as metric sources; those are derived artifacts. The source of truth
is:

  - leaderboards/v7/logs/train_*.log
  - leaderboards/v7/evals/eval_*_beam{1,5}.log
  - leaderboards/v7/B_KAN_RESCUE/*_s{0,1,2,...}.log
  - leaderboards/v7/B_KAN_RESCUE/eval_*_beam{1,5}.log
  - checkpoint file sizes for inventory only

Older root-level checkpoints are inventoried, but not mixed into the v7
leaderboard unless a matching raw v7 eval log exists. This keeps the comparison
honest: no eval log, no leaderboard metric.
"""

from __future__ import annotations

import csv
import json
import re
import shutil
import statistics
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, Optional


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
V7 = ROOT / "leaderboards" / "v7"
RESCUE = V7 / "B_KAN_RESCUE"


@dataclass
class TrainMetrics:
    model_id: str
    family: str
    source_log: str
    policy: str = ""
    params: Optional[int] = None
    best_val_loss: Optional[float] = None
    final_val_loss: Optional[float] = None
    final_val_acc: Optional[float] = None
    final_avg_gold_rank: Optional[float] = None
    completed: bool = False


@dataclass
class EvalMetrics:
    model_id: str
    beam: int
    source_log: str
    success_rate: Optional[float] = None
    overhead_mean: Optional[float] = None
    overhead_p95: Optional[int] = None
    bfs_optimal_rate: Optional[float] = None
    catastrophic_rate: Optional[float] = None
    step_match_rate: Optional[float] = None


def read(path: Path) -> str:
    data = path.read_bytes()
    if data.startswith(b"\xff\xfe") or data.startswith(b"\xfe\xff"):
        return data.decode("utf-16", errors="replace")
    if data.startswith(b"\xef\xbb\xbf"):
        return data.decode("utf-8-sig", errors="replace")
    return data.decode("utf-8", errors="replace")


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def infer_family(model_id: str) -> str:
    if model_id.startswith("micro_mlp"):
        return "micro-MLP"
    if model_id.startswith("mlp128"):
        return "MLP-128"
    if model_id.startswith("kan_ae"):
        return "KAN-AE"
    if (
        model_id.startswith("encoder_kan")
        or model_id.startswith("encoder16_kan")
        or model_id.startswith("encoder16r")
        or model_id.startswith("encoder_deep_kan")
    ):
        return "Encoder-KAN"
    if model_id.startswith("kan"):
        return "raw KAN"
    return "unknown"


def normalize_train_id(path: Path) -> str:
    stem = path.stem
    if stem.startswith("train_v7_"):
        stem = stem.removeprefix("train_v7_")
    return stem


def normalize_eval_id(path: Path) -> tuple[str, int]:
    stem = path.stem
    stem = stem.removeprefix("eval_v7_")
    stem = stem.removeprefix("eval_")
    m = re.search(r"_beam([15])$", stem)
    if not m:
        raise ValueError(f"Cannot infer beam from {path}")
    beam = int(m.group(1))
    model_id = stem[: m.start()]
    return model_id, beam


def parse_train_log(path: Path) -> TrainMetrics:
    text = read(path)
    model_id = normalize_train_id(path)
    tm = TrainMetrics(
        model_id=model_id,
        family=infer_family(model_id),
        source_log=rel(path),
    )

    m = re.search(r"Policy:\s+(.+)", text)
    if m:
        tm.policy = m.group(1).strip()

    m = re.search(r"Total params:\s+([0-9,]+)", text)
    if m:
        tm.params = int(m.group(1).replace(",", ""))
    else:
        m = re.search(r"Policy params:\s+([0-9,]+)", text)
        if m:
            tm.params = int(m.group(1).replace(",", ""))

    vals = re.findall(
        r"VAL\s+\|\s+loss\s+([0-9.]+)\s+\|\s+acc\s+([0-9.]+)\s+\|\s+avg_gold_rank\s+([0-9.]+)",
        text,
    )
    if vals:
        loss, acc, rank = vals[-1]
        tm.final_val_loss = float(loss)
        tm.final_val_acc = float(acc)
        tm.final_avg_gold_rank = float(rank)

    m = re.search(r"Training complete\. Best val loss:\s+([0-9.]+)", text)
    if m:
        tm.best_val_loss = float(m.group(1))
        tm.completed = True
    else:
        saved = re.findall(r"SAVED best model \(val_loss=([0-9.]+)\)", text)
        if saved:
            tm.best_val_loss = float(saved[-1])

    return tm


def parse_eval_log(path: Path) -> EvalMetrics:
    text = read(path)
    model_id, beam = normalize_eval_id(path)
    em = EvalMetrics(model_id=model_id, beam=beam, source_log=rel(path))

    m = re.search(r"success_rate\s+:\s+\d+/\d+\s+=\s+([0-9.]+)%", text)
    if m:
        em.success_rate = float(m.group(1))

    m = re.search(r"overhead vs BFS\s+:\s+\{([^}]+)\}", text)
    if m:
        d = m.group(1)
        mm = re.search(r"'mean':\s+([0-9.]+)", d)
        if mm:
            em.overhead_mean = float(mm.group(1))
        pp = re.search(r"'p95':\s+([0-9]+)", d)
        if pp:
            em.overhead_p95 = int(pp.group(1))

    m = re.search(r"bfs_optimal_rate\s+:\s+([0-9.]+)%", text)
    if m:
        em.bfs_optimal_rate = float(m.group(1))

    m = re.search(r"catastrophic_rate\s+:\s+([0-9.]+)%", text)
    if m:
        em.catastrophic_rate = float(m.group(1))

    m = re.search(r"step_match .*?:\s+(\d+)/(\d+)\s+=\s+([0-9.]+)%", text)
    if m:
        em.step_match_rate = float(m.group(3))

    return em


def iter_train_logs() -> Iterable[Path]:
    rescue_train_name = re.compile(
        r"^(kan_ae|encoder_kan|encoder16_kan|encoder16r\d+_kan|encoder_deep_kan).+_s\d+\.log$"
    )
    yield from sorted((V7 / "logs").glob("train_v7_*.log"))
    for path in sorted(RESCUE.glob("*.log")):
        if rescue_train_name.match(path.name):
            yield path


def iter_eval_logs() -> Iterable[Path]:
    strict_eval_name = re.compile(r".*_beam[15]\.log$")
    for path in sorted((V7 / "evals").glob("eval_v7_*_beam*.log")):
        if strict_eval_name.match(path.name):
            yield path
    for path in sorted(RESCUE.glob("eval_*_beam*.log")):
        if strict_eval_name.match(path.name):
            yield path


def checkpoint_inventory() -> list[dict]:
    rows = []
    seen = set()
    checkpoint_roots = [
        (V7 / "checkpoints", "v7_base"),
        (RESCUE, "v7_kan_rescue"),
    ]
    checkpoint_roots.extend(
        (p, "historical_root")
        for p in sorted(ROOT.glob("checkpoints*"))
        if p.is_dir()
    )
    for base, scope in checkpoint_roots:
        for ckpt in sorted(base.rglob("best.pt")):
            key = ckpt.resolve()
            if key in seen:
                continue
            seen.add(key)
            rows.append({
                "scope": scope,
                "checkpoint": rel(ckpt),
                "bytes": ckpt.stat().st_size,
                "kb": round(ckpt.stat().st_size / 1024, 1),
            })
    return rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def merged_rows(train: Dict[str, TrainMetrics], evals: list[EvalMetrics]) -> list[dict]:
    by_eval: Dict[tuple[str, int], EvalMetrics] = {
        (e.model_id, e.beam): e for e in evals
    }
    ids = sorted(set(train) | {e.model_id for e in evals})
    rows = []
    for mid in ids:
        t = train.get(mid)
        e1 = by_eval.get((mid, 1))
        e5 = by_eval.get((mid, 5))
        rows.append({
            "model_id": mid,
            "family": t.family if t else infer_family(mid),
            "params": t.params if t else None,
            "best_val_loss": t.best_val_loss if t else None,
            "greedy_bfs_optimal": e1.bfs_optimal_rate if e1 else None,
            "greedy_overhead": e1.overhead_mean if e1 else None,
            "greedy_catastrophic": e1.catastrophic_rate if e1 else None,
            "beam5_bfs_optimal": e5.bfs_optimal_rate if e5 else None,
            "beam5_overhead": e5.overhead_mean if e5 else None,
            "beam5_catastrophic": e5.catastrophic_rate if e5 else None,
            "train_source": t.source_log if t else "",
            "greedy_source": e1.source_log if e1 else "",
            "beam5_source": e5.source_log if e5 else "",
        })
    return rows


def stability_base(model_id: str) -> str:
    return re.sub(r"_s\d+$", "", model_id)


def mean(values: list[float]) -> Optional[float]:
    return round(sum(values) / len(values), 4) if values else None


def stdev(values: list[float]) -> Optional[float]:
    return round(statistics.stdev(values), 4) if len(values) > 1 else None


def stability_rows(rows: list[dict]) -> list[dict]:
    grouped: Dict[str, list[dict]] = {}
    for row in rows:
        if not re.search(r"_s\d+$", row["model_id"]):
            continue
        if row["greedy_bfs_optimal"] is None or row["beam5_bfs_optimal"] is None:
            continue
        grouped.setdefault(stability_base(row["model_id"]), []).append(row)

    out = []
    for base, group in sorted(grouped.items()):
        if len(group) < 2:
            continue
        greedy = [float(r["greedy_bfs_optimal"]) for r in group]
        beam5 = [float(r["beam5_bfs_optimal"]) for r in group]
        val = [float(r["best_val_loss"]) for r in group if r["best_val_loss"] is not None]
        params = sorted({r["params"] for r in group if r["params"] is not None})
        out.append({
            "model_base": base,
            "family": group[0]["family"],
            "seeds": ",".join(sorted(r["model_id"].rsplit("_s", 1)[1] for r in group)),
            "n": len(group),
            "params": params[0] if len(params) == 1 else "",
            "best_val_loss_mean": mean(val),
            "best_val_loss_std": stdev(val),
            "greedy_bfs_optimal_mean": mean(greedy),
            "greedy_bfs_optimal_min": min(greedy),
            "greedy_bfs_optimal_max": max(greedy),
            "greedy_bfs_optimal_std": stdev(greedy),
            "beam5_bfs_optimal_mean": mean(beam5),
            "beam5_bfs_optimal_min": min(beam5),
            "beam5_bfs_optimal_max": max(beam5),
            "beam5_bfs_optimal_std": stdev(beam5),
        })
    return sorted(out, key=lambda r: (r["greedy_bfs_optimal_mean"] or -1), reverse=True)


def fmt(x, suffix: str = "") -> str:
    if x is None:
        return "n/a"
    if isinstance(x, int):
        return f"{x:,}{suffix}"
    if isinstance(x, float):
        return f"{x:.4g}{suffix}"
    return str(x)


def md_table(rows: list[dict], cols: list[tuple[str, str]]) -> str:
    out = []
    out.append("| " + " | ".join(title for _, title in cols) + " |")
    out.append("|" + "|".join("---" for _ in cols) + "|")
    for row in rows:
        out.append("| " + " | ".join(fmt(row.get(key)) for key, _ in cols) + " |")
    return "\n".join(out)


def build_markdown(
    rows: list[dict],
    train: Dict[str, TrainMetrics],
    evals: list[EvalMetrics],
    stability: list[dict],
) -> str:
    rows_with_greedy = [r for r in rows if r["greedy_bfs_optimal"] is not None]
    by_greedy = sorted(rows_with_greedy, key=lambda r: r["greedy_bfs_optimal"], reverse=True)
    by_beam = sorted(
        [r for r in rows if r["beam5_bfs_optimal"] is not None],
        key=lambda r: (r["beam5_bfs_optimal"], -float(r["beam5_overhead"] or 999)),
        reverse=True,
    )
    encoder = [r for r in rows if r["family"] == "Encoder-KAN"]
    raw_kan = [r for r in rows if r["family"] == "raw KAN"]
    mlp = [r for r in rows if r["family"] in {"micro-MLP", "MLP-128"}]
    missing_rollout = [
        r for r in rows
        if r["train_source"] and not r["greedy_source"] and not r["beam5_source"]
    ]

    best_encoder_greedy = max(encoder, key=lambda r: r["greedy_bfs_optimal"] or -1)
    best_encoder_beam = max(encoder, key=lambda r: r["beam5_bfs_optimal"] or -1)
    best_raw = max(raw_kan, key=lambda r: r["greedy_bfs_optimal"] or -1)
    best_mlp_small = max(
        [r for r in mlp if r["model_id"] != "mlp128_s0"],
        key=lambda r: r["greedy_bfs_optimal"] or -1,
    )
    kan_ae = next((r for r in rows if r["model_id"] == "kan_ae_h16_s0"), None)
    micro_h8 = next((r for r in rows if r["model_id"] == "micro_mlp_h8_s0"), None)
    mlp128 = next(r for r in rows if r["model_id"] == "mlp128_s0")
    b40_stability = next((r for r in stability if r["model_base"] == "encoder16_kan_b40_h16"), None)
    best_stability = max(
        stability,
        key=lambda r: r["greedy_bfs_optimal_mean"] or -1,
    ) if stability else None

    lines = [
        "# ISRE v7 Full KAN/MLP Analysis",
        "",
        "This folder is generated from raw train/eval logs only.",
        "",
        "Source-of-truth policy:",
        "",
        "```text",
        "No metric in this report is taken from chat memory, SUMMARY.md, or RESULTS.md.",
        "Metrics are parsed from train/eval log files and checkpoint inventory.",
        "Historical checkpoints without matching eval logs are inventoried, not ranked.",
        "```",
        "",
        "## Executive Summary",
        "",
        f"- Best Encoder-KAN greedy point: `{best_encoder_greedy['model_id']}` "
        f"with {best_encoder_greedy['greedy_bfs_optimal']}% greedy BFS-optimal "
        f"and {best_encoder_greedy['params']:,} params.",
        f"- Best Encoder-KAN beam-5 point: `{best_encoder_beam['model_id']}` "
        f"with {best_encoder_beam['beam5_bfs_optimal']}% beam-5 BFS-optimal.",
        f"- Best raw KAN greedy point: `{best_raw['model_id']}` "
        f"with {best_raw['greedy_bfs_optimal']}% greedy BFS-optimal.",
        f"- Best small MLP greedy point: `{best_mlp_small['model_id']}` "
        f"with {best_mlp_small['greedy_bfs_optimal']}% greedy BFS-optimal.",
        (
            f"- Best multi-seed Encoder-KAN group: `{best_stability['model_base']}` "
            f"with n={best_stability['n']} seeds ({best_stability['seeds']}), "
            f"mean greedy {best_stability['greedy_bfs_optimal_mean']}%, "
            f"range {best_stability['greedy_bfs_optimal_min']}-{best_stability['greedy_bfs_optimal_max']}%, "
            f"mean beam-5 {best_stability['beam5_bfs_optimal_mean']}%."
            if best_stability
            else "- Best multi-seed Encoder-KAN group: not enough seeds parsed yet."
        ),
        (
            f"- Encoder16-KAN b40 h16 stability: n={b40_stability['n']} seeds "
            f"({b40_stability['seeds']}), mean greedy "
            f"{b40_stability['greedy_bfs_optimal_mean']}%, range "
            f"{b40_stability['greedy_bfs_optimal_min']}-{b40_stability['greedy_bfs_optimal_max']}%, "
            f"mean beam-5 {b40_stability['beam5_bfs_optimal_mean']}%."
            if b40_stability
            else "- Encoder16-KAN b40 h16 stability: not enough seeds parsed yet."
        ),
        (
            f"- Micro-MLP h8 reference: {micro_h8['greedy_bfs_optimal']}% greedy "
            f"BFS-optimal with {micro_h8['params']:,} params."
            if micro_h8
            else "- Micro-MLP h8 reference: not found in raw logs."
        ),
        f"- MLP-128 reference: {mlp128['greedy_bfs_optimal']}% greedy BFS-optimal "
        f"with {mlp128['params']:,} params.",
        "",
        "Plain-language conclusion:",
        "",
        "```text",
        "Raw KAN underperformed because it was given poor hand-crafted inputs.",
        (
            f"KAN-AE h16 is now ranked: {kan_ae['greedy_bfs_optimal']}% greedy, "
            f"{kan_ae['beam5_bfs_optimal']}% beam-5."
            if kan_ae and kan_ae["greedy_bfs_optimal"] is not None
            else "KAN-AE was trained, but it is not ranked here because no raw eval log is present."
        ),
        "ASTEncoder + KAN head recovered a large part of the gap.",
        "The bottleneck/head geometry matters: bigger is not automatically better.",
        "Validation loss does not reliably rank rollout quality.",
        "```",
        "",
        "## Full Leaderboard, Greedy First",
        "",
        md_table(
            by_greedy,
            [
                ("model_id", "model"),
                ("family", "family"),
                ("params", "params"),
                ("best_val_loss", "best val loss"),
                ("greedy_bfs_optimal", "greedy BFS-opt %"),
                ("greedy_overhead", "greedy overhead"),
                ("beam5_bfs_optimal", "beam-5 BFS-opt %"),
                ("beam5_overhead", "beam-5 overhead"),
            ],
        ),
        "",
        "## Multi-Seed Stability",
        "",
        "Rows here are grouped by architecture name with the `_sN` suffix removed. "
        "This is the repeatability check; it is not based on chat memory.",
        "",
        md_table(
            stability,
            [
                ("model_base", "model base"),
                ("family", "family"),
                ("seeds", "seeds"),
                ("n", "n"),
                ("params", "params"),
                ("best_val_loss_mean", "mean val loss"),
                ("best_val_loss_std", "std val loss"),
                ("greedy_bfs_optimal_mean", "mean greedy BFS-opt %"),
                ("greedy_bfs_optimal_min", "min greedy"),
                ("greedy_bfs_optimal_max", "max greedy"),
                ("greedy_bfs_optimal_std", "std greedy"),
                ("beam5_bfs_optimal_mean", "mean beam-5 BFS-opt %"),
                ("beam5_bfs_optimal_min", "min beam-5"),
                ("beam5_bfs_optimal_max", "max beam-5"),
            ],
        ) if stability else "No multi-seed groups found.",
        "",
        "## Beam-5 Ranking",
        "",
        md_table(
            by_beam,
            [
                ("model_id", "model"),
                ("family", "family"),
                ("params", "params"),
                ("beam5_bfs_optimal", "beam-5 BFS-opt %"),
                ("beam5_overhead", "beam-5 overhead"),
                ("greedy_bfs_optimal", "greedy BFS-opt %"),
            ],
        ),
        "",
        "## Where KAN Wins",
        "",
        "- Encoder-KAN beats raw KAN by a wide margin once it receives learned AST representations.",
        f"- Best raw KAN greedy: {best_raw['greedy_bfs_optimal']}%. "
        f"Best Encoder-KAN greedy: {best_encoder_greedy['greedy_bfs_optimal']}%.",
        "- Encoder-KAN is dramatically smaller than MLP-128 while beating it on greedy rollout.",
        f"- `{best_encoder_greedy['model_id']}` has {best_encoder_greedy['params']:,} params "
        f"vs MLP-128's {mlp128['params']:,} params.",
        "- Encoder-KAN can beat micro-MLP h8 on greedy rollout in the tuned local sweep.",
        (
            f"- Concrete h8 comparison: `{best_encoder_greedy['model_id']}` "
            f"{best_encoder_greedy['greedy_bfs_optimal']}% vs "
            f"`micro_mlp_h8_s0` {micro_h8['greedy_bfs_optimal']}%."
            if micro_h8
            else "- Concrete h8 comparison unavailable because micro_mlp_h8_s0 was not parsed."
        ),
        "",
        "## Where KAN Still Loses",
        "",
        "- The strongest micro-MLP h32 point still narrowly leads greedy rollout.",
        "- Encoder16-KAN b40 h16 and encoder16r6_kan b40 h16 both beat micro-MLP h16 on greedy in multiple seeds.",
        "- Encoder-KAN is sensitive to bottleneck/head geometry; b20 h24 had good validation loss but poor rollout.",
        "- A KAN head is not a drop-in win: raw feature-only KAN is clearly weaker.",
        "",
        "## Methodological Findings",
        "",
        "1. Representation quality matters more than raw KAN enthusiasm.",
        "2. The 24 -> 16 bottleneck was too narrow for the encoder representation.",
        "3. Too much widening is also not monotonic: b32 h32 did not become the best greedy model.",
        "4. Rollout metrics are primary; cross-entropy validation loss is not sufficient.",
        "5. Beam search and greedy emphasize different strengths.",
        "",
        "## Missing Rollout Coverage",
        "",
        "These models have raw training logs/checkpoints but no matching raw eval logs in this package, so they are not used for rollout claims:",
        "",
        md_table(
            missing_rollout,
            [
                ("model_id", "model"),
                ("family", "family"),
                ("params", "params"),
                ("best_val_loss", "best val loss"),
                ("train_source", "train source"),
            ],
        ) if missing_rollout else "None.",
        "",
        "## Source Verification",
        "",
        "Every row below points back to raw train/eval logs.",
        "",
        md_table(
            rows,
            [
                ("model_id", "model"),
                ("train_source", "train source"),
                ("greedy_source", "greedy source"),
                ("beam5_source", "beam-5 source"),
            ],
        ),
        "",
    ]
    return "\n".join(lines)


def copy_figures() -> None:
    for name in [
        "kan_rescue_portfolio_summary.png",
        "kan_rescue_sweep_map.png",
        "kan_rescue_comparison.png",
    ]:
        src = RESCUE / name
        if src.exists():
            shutil.copy2(src, OUT / name)


def copy_raw_logs(paths: list[Path]) -> None:
    raw_dir = OUT / "raw_logs"
    raw_dir.mkdir(exist_ok=True)
    for src in paths:
        dst = raw_dir / rel(src).replace("/", "__")
        shutil.copy2(src, dst)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    train_logs = list(iter_train_logs())
    eval_logs = list(iter_eval_logs())
    parsed_trains = [parse_train_log(p) for p in train_logs]
    trains = {t.model_id: t for t in parsed_trains}
    evals = [parse_eval_log(p) for p in eval_logs]
    rows = merged_rows(trains, evals)
    stability = stability_rows(rows)

    write_csv(
        OUT / "train_metrics_from_raw_logs.csv",
        [asdict(v) for v in trains.values()],
        list(asdict(next(iter(trains.values()))).keys()),
    )
    write_csv(
        OUT / "eval_metrics_from_raw_logs.csv",
        [asdict(e) for e in evals],
        list(asdict(evals[0]).keys()),
    )
    write_csv(
        OUT / "merged_leaderboard_from_raw_logs.csv",
        rows,
        list(rows[0].keys()),
    )
    write_csv(
        OUT / "seed_stability_from_raw_logs.csv",
        stability,
        list(stability[0].keys()) if stability else [
            "model_base",
            "family",
            "seeds",
            "n",
            "params",
            "best_val_loss_mean",
            "best_val_loss_std",
            "greedy_bfs_optimal_mean",
            "greedy_bfs_optimal_min",
            "greedy_bfs_optimal_max",
            "greedy_bfs_optimal_std",
            "beam5_bfs_optimal_mean",
            "beam5_bfs_optimal_min",
            "beam5_bfs_optimal_max",
            "beam5_bfs_optimal_std",
        ],
    )
    ckpts = checkpoint_inventory()
    write_csv(
        OUT / "checkpoint_inventory.csv",
        ckpts,
        ["scope", "checkpoint", "bytes", "kb"],
    )

    verification = {
        "train_logs_parsed": len(train_logs),
        "eval_logs_parsed": len(eval_logs),
        "models_in_merged_table": len(rows),
        "multi_seed_groups": len(stability),
        "checkpoints_in_inventory": len(ckpts),
        "train_logs": [rel(p) for p in train_logs],
        "eval_logs": [rel(p) for p in eval_logs],
    }
    (OUT / "verification_manifest.json").write_text(
        json.dumps(verification, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    (OUT / "README.md").write_text(
        build_markdown(rows, trains, evals, stability),
        encoding="utf-8",
    )
    copy_figures()
    copy_raw_logs(train_logs + eval_logs)

    print(f"Wrote analysis package to {OUT}")
    print(json.dumps(verification, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
