$ErrorActionPreference = 'Continue'
$repo = 'C:\GitHub\ISRE'
$base = Join-Path $repo 'leaderboards\v7'
$logs = Join-Path $base 'logs'
$evals = Join-Path $base 'evals'
Set-Location $repo
$env:PYTHONPATH = '.'
$env:PYTHONIOENCODING = 'utf-8'
"[$(Get-Date -Format s)] watcher started; waiting for MLP-128 train cmd PID: 45240" | Tee-Object -FilePath (Join-Path $logs 'watch_and_eval_mlp128.log') -Append
Wait-Process -Id 45240 -ErrorAction SilentlyContinue
"[$(Get-Date -Format s)] MLP-128 training process exited; starting evals" | Tee-Object -FilePath (Join-Path $logs 'watch_and_eval_mlp128.log') -Append
$ckpt = 'leaderboards/v7/checkpoints/v7_mlp128_s0/best.pt'
$noTeacher = 'leaderboards/v7/no_recorded_teacher'
if (Test-Path $ckpt) {
  python scripts/eval_neural.py --policy mlp --ckpt $ckpt --hidden-dim 128 --bfs-data isre/trajectories_v7_bfs --recorded-data $noTeacher --n 2000 --device cpu --beam 1 *> (Join-Path $evals 'eval_v7_mlp128_s0_beam1.log')
  python scripts/eval_neural.py --policy mlp --ckpt $ckpt --hidden-dim 128 --bfs-data isre/trajectories_v7_bfs --recorded-data $noTeacher --n 2000 --device cpu --beam 5 *> (Join-Path $evals 'eval_v7_mlp128_s0_beam5.log')
  "[$(Get-Date -Format s)] MLP-128 evals done" | Tee-Object -FilePath (Join-Path $logs 'watch_and_eval_mlp128.log') -Append
} else {
  "[$(Get-Date -Format s)] MLP-128 checkpoint missing: $ckpt" | Tee-Object -FilePath (Join-Path $logs 'watch_and_eval_mlp128.log') -Append
}
"[$(Get-Date -Format s)] watcher finished" | Tee-Object -FilePath (Join-Path $logs 'watch_and_eval_mlp128.log') -Append
