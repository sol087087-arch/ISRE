$ErrorActionPreference = 'Continue'
$repo = 'C:\GitHub\ISRE'
$base = Join-Path $repo 'leaderboards\v7'
$logs = Join-Path $base 'logs'
$evals = Join-Path $base 'evals'
Set-Location $repo
$env:PYTHONPATH = '.'
$env:PYTHONIOENCODING = 'utf-8'
"[$(Get-Date -Format s)] watcher started; waiting for KAN h8/h32 cmd PIDs: 48276, 31336" | Tee-Object -FilePath (Join-Path $logs 'watch_and_eval_kan_h8_h32.log') -Append
Wait-Process -Id 48276,31336 -ErrorAction SilentlyContinue
"[$(Get-Date -Format s)] KAN h8/h32 training processes exited; starting evals" | Tee-Object -FilePath (Join-Path $logs 'watch_and_eval_kan_h8_h32.log') -Append
$noTeacher = 'leaderboards/v7/no_recorded_teacher'
foreach ($h in @(8,32)) {
  $ckpt = "leaderboards/v7/checkpoints/v7_kan_h${h}_s0/best.pt"
  if (Test-Path $ckpt) {
    python scripts/eval_neural.py --policy kan --kan-hidden $h --ckpt $ckpt --hidden-dim 128 --bfs-data isre/trajectories_v7_bfs --recorded-data $noTeacher --n 2000 --device cpu --beam 1 *> (Join-Path $evals "eval_v7_kan_h${h}_s0_beam1.log")
    python scripts/eval_neural.py --policy kan --kan-hidden $h --ckpt $ckpt --hidden-dim 128 --bfs-data isre/trajectories_v7_bfs --recorded-data $noTeacher --n 2000 --device cpu --beam 5 *> (Join-Path $evals "eval_v7_kan_h${h}_s0_beam5.log")
    "[$(Get-Date -Format s)] KAN h$h evals done" | Tee-Object -FilePath (Join-Path $logs 'watch_and_eval_kan_h8_h32.log') -Append
  } else {
    "[$(Get-Date -Format s)] KAN h$h checkpoint missing: $ckpt" | Tee-Object -FilePath (Join-Path $logs 'watch_and_eval_kan_h8_h32.log') -Append
  }
}
"[$(Get-Date -Format s)] watcher finished" | Tee-Object -FilePath (Join-Path $logs 'watch_and_eval_kan_h8_h32.log') -Append
