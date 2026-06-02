# ISRE v7 leaderboard runbook. Additive only.

$env:PYTHONPATH='.'
$env:PYTHONIOENCODING='utf-8'

# Train micro-MLP h8
python -u -m isre.training.train `
  --data isre/trajectories_v7_bfs `
  --policy mlp `
  --hidden-dim 8 `
  --policy-hidden-dim 16 `
  --action-emb-dim 8 `
  --epochs 15 `
  --device cpu `
  --seed 0 `
  --save-dir leaderboards/v7/checkpoints/v7_micro_mlp_h8_s0

# Train KAN h16
python -u -m isre.training.train `
  --data isre/trajectories_v7_bfs `
  --policy kan `
  --kan-hidden 16 `
  --epochs 15 `
  --device cpu `
  --seed 0 `
  --save-dir leaderboards/v7/checkpoints/v7_kan_h16_s0

# Eval examples after training completes:
python scripts/eval_neural.py --policy mlp --ckpt leaderboards/v7/checkpoints/v7_micro_mlp_h8_s0/best.pt --hidden-dim 8 --policy-hidden-dim 16 --action-emb-dim 8 --bfs-data isre/trajectories_v7_bfs --recorded-data leaderboards/v7/no_recorded_teacher --n 2000 --device cpu --beam 5
python scripts/eval_neural.py --policy kan --kan-hidden 16 --ckpt leaderboards/v7/checkpoints/v7_kan_h16_s0/best.pt --hidden-dim 128 --bfs-data isre/trajectories_v7_bfs --recorded-data leaderboards/v7/no_recorded_teacher --n 2000 --device cpu --beam 5


# Train MLP-128 reference baseline
python -u -m isre.training.train `
  --data isre/trajectories_v7_bfs `
  --policy mlp `
  --hidden-dim 128 `
  --epochs 15 `
  --device cpu `
  --seed 0 `
  --save-dir leaderboards/v7/checkpoints/v7_mlp128_s0

# Eval MLP-128 reference baseline
python scripts/eval_neural.py --policy mlp --ckpt leaderboards/v7/checkpoints/v7_mlp128_s0/best.pt --hidden-dim 128 --bfs-data isre/trajectories_v7_bfs --recorded-data leaderboards/v7/no_recorded_teacher --n 2000 --device cpu --beam 1
python scripts/eval_neural.py --policy mlp --ckpt leaderboards/v7/checkpoints/v7_mlp128_s0/best.pt --hidden-dim 128 --bfs-data isre/trajectories_v7_bfs --recorded-data leaderboards/v7/no_recorded_teacher --n 2000 --device cpu --beam 5

# Train KAN compression ladder on v7
python -u -m isre.training.train `
  --data isre/trajectories_v7_bfs `
  --policy kan `
  --kan-hidden 4 `
  --epochs 15 `
  --device cpu `
  --seed 0 `
  --save-dir leaderboards/v7/checkpoints/v7_kan_h4_s0

python -u -m isre.training.train `
  --data isre/trajectories_v7_bfs `
  --policy kan `
  --kan-hidden 2 `
  --epochs 15 `
  --device cpu `
  --seed 0 `
  --save-dir leaderboards/v7/checkpoints/v7_kan_h2_s0

# Train micro-MLP width ladder on v7
# recipe: encoder hidden=H, policy hidden=2H, action emb=H
python -u -m isre.training.train --data isre/trajectories_v7_bfs --policy mlp --hidden-dim 4 --policy-hidden-dim 8 --action-emb-dim 4 --epochs 15 --device cpu --seed 0 --save-dir leaderboards/v7/checkpoints/v7_micro_mlp_h4_s0
python -u -m isre.training.train --data isre/trajectories_v7_bfs --policy mlp --hidden-dim 16 --policy-hidden-dim 32 --action-emb-dim 16 --epochs 15 --device cpu --seed 0 --save-dir leaderboards/v7/checkpoints/v7_micro_mlp_h16_s0
python -u -m isre.training.train --data isre/trajectories_v7_bfs --policy mlp --hidden-dim 32 --policy-hidden-dim 64 --action-emb-dim 32 --epochs 15 --device cpu --seed 0 --save-dir leaderboards/v7/checkpoints/v7_micro_mlp_h32_s0
