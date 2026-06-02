$ErrorActionPreference = "Continue"
$repo = "C:\GitHub\ISRE"
$base = Join-Path $repo "leaderboards\v7"
$logs = Join-Path $base "logs"
$evals = Join-Path $base "evals"
Set-Location $repo
$env:PYTHONPATH = "."
$env:PYTHONIOENCODING = "utf-8"

"[$(Get-Date -Format s)] watcher started; waiting for train cmd PIDs: 65448, 67324" | Tee-Object -FilePath (Join-Path $logs "watch_and_eval.log") -Append
Wait-Process -Id 65448,67324 -ErrorAction SilentlyContinue
"[$(Get-Date -Format s)] training processes exited; starting evals" | Tee-Object -FilePath (Join-Path $logs "watch_and_eval.log") -Append

$micro = "leaderboards/v7/checkpoints/v7_micro_mlp_h8_s0/best.pt"
$kan = "leaderboards/v7/checkpoints/v7_kan_h16_s0/best.pt"
$noTeacher = "leaderboards/v7/no_recorded_teacher"

if (Test-Path $micro) {
  python scripts/eval_neural.py --policy mlp --ckpt $micro --hidden-dim 8 --policy-hidden-dim 16 --action-emb-dim 8 --bfs-data isre/trajectories_v7_bfs --recorded-data $noTeacher --n 2000 --device cpu --beam 1 *> (Join-Path $evals "eval_v7_micro_mlp_h8_s0_beam1.log")
  python scripts/eval_neural.py --policy mlp --ckpt $micro --hidden-dim 8 --policy-hidden-dim 16 --action-emb-dim 8 --bfs-data isre/trajectories_v7_bfs --recorded-data $noTeacher --n 2000 --device cpu --beam 5 *> (Join-Path $evals "eval_v7_micro_mlp_h8_s0_beam5.log")
  "[$(Get-Date -Format s)] micro-MLP evals done" | Tee-Object -FilePath (Join-Path $logs "watch_and_eval.log") -Append
} else {
  "[$(Get-Date -Format s)] micro checkpoint missing: $micro" | Tee-Object -FilePath (Join-Path $logs "watch_and_eval.log") -Append
}

if (Test-Path $kan) {
  python scripts/eval_neural.py --policy kan --kan-hidden 16 --ckpt $kan --hidden-dim 128 --bfs-data isre/trajectories_v7_bfs --recorded-data $noTeacher --n 2000 --device cpu --beam 1 *> (Join-Path $evals "eval_v7_kan_h16_s0_beam1.log")
  python scripts/eval_neural.py --policy kan --kan-hidden 16 --ckpt $kan --hidden-dim 128 --bfs-data isre/trajectories_v7_bfs --recorded-data $noTeacher --n 2000 --device cpu --beam 5 *> (Join-Path $evals "eval_v7_kan_h16_s0_beam5.log")
  "[$(Get-Date -Format s)] KAN evals done" | Tee-Object -FilePath (Join-Path $logs "watch_and_eval.log") -Append
} else {
  "[$(Get-Date -Format s)] KAN checkpoint missing: $kan" | Tee-Object -FilePath (Join-Path $logs "watch_and_eval.log") -Append
}

"[$(Get-Date -Format s)] watcher finished" | Tee-Object -FilePath (Join-Path $logs "watch_and_eval.log") -Append
