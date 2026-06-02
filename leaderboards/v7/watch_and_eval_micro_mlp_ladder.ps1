$ErrorActionPreference = 'Continue'
$repo = 'C:\GitHub\ISRE'
$base = Join-Path $repo 'leaderboards\v7'
$logs = Join-Path $base 'logs'
$evals = Join-Path $base 'evals'
Set-Location $repo
$env:PYTHONPATH = '.'
$env:PYTHONIOENCODING = 'utf-8'
"[$(Get-Date -Format s)] watcher started; waiting for micro-MLP ladder cmd PIDs: 57724,49132,14404" | Tee-Object -FilePath (Join-Path $logs 'watch_and_eval_micro_mlp_ladder.log') -Append
Wait-Process -Id 57724,49132,14404 -ErrorAction SilentlyContinue
"[$(Get-Date -Format s)] micro-MLP ladder training processes exited; starting evals" | Tee-Object -FilePath (Join-Path $logs 'watch_and_eval_micro_mlp_ladder.log') -Append
$noTeacher = 'leaderboards/v7/no_recorded_teacher'
foreach ($h in @(4,16,32)) {
  $ph = 2 * $h
  $ae = $h
  $ckpt = "leaderboards/v7/checkpoints/v7_micro_mlp_h$($h)_s0/best.pt"
  if (Test-Path $ckpt) {
    python scripts/eval_neural.py --policy mlp --ckpt $ckpt --hidden-dim $h --policy-hidden-dim $ph --action-emb-dim $ae --bfs-data isre/trajectories_v7_bfs --recorded-data $noTeacher --n 2000 --device cpu --beam 1 *> (Join-Path $evals "eval_v7_micro_mlp_h$($h)_s0_beam1.log")
    python scripts/eval_neural.py --policy mlp --ckpt $ckpt --hidden-dim $h --policy-hidden-dim $ph --action-emb-dim $ae --bfs-data isre/trajectories_v7_bfs --recorded-data $noTeacher --n 2000 --device cpu --beam 5 *> (Join-Path $evals "eval_v7_micro_mlp_h$($h)_s0_beam5.log")
    "[$(Get-Date -Format s)] micro-MLP h$h evals done" | Tee-Object -FilePath (Join-Path $logs 'watch_and_eval_micro_mlp_ladder.log') -Append
  } else {
    "[$(Get-Date -Format s)] micro-MLP h$h checkpoint missing: $ckpt" | Tee-Object -FilePath (Join-Path $logs 'watch_and_eval_micro_mlp_ladder.log') -Append
  }
}
"[$(Get-Date -Format s)] watcher finished" | Tee-Object -FilePath (Join-Path $logs 'watch_and_eval_micro_mlp_ladder.log') -Append
