# B: KAN Rescue Campaign

Append-only experiment folder. Nothing here replaces the locked v7 raw-KAN
leaderboard. This campaign asks a narrower fairness question:

> Did raw KAN lose because KAN is weak here, or because our KAN arm had less
> learned representation machinery than the MLP arm?

## Existing Control

`--policy kan` remains the locked raw interpretable baseline:

```text
candidate_features[27] -> efficient_kan KAN([27,H,1]) -> score
```

It has no AST encoder and no learned action embedding.

## B1: Action-Embedding KAN

New policy name:

```text
--policy kan_ae
```

Architecture:

```text
candidate_features[27] + action_embedding[A]
-> efficient_kan KAN([27+A,H,1])
-> score
```

Purpose:

This isolates the action-embedding asymmetry. The micro-MLP has a learned
action embedding; raw KAN only had explicit action one-hot features.

Primary run:

```powershell
python -u -m isre.training.train --data isre/trajectories_v7_bfs --policy kan_ae --kan-hidden 16 --kan-action-emb-dim 8 --epochs 15 --device cpu --seed 0 --save-dir leaderboards/v7/B_KAN_RESCUE/checkpoints_kan_ae_h16_s0
```

Eval:

```powershell
python scripts/eval_neural.py --policy kan_ae --ckpt leaderboards/v7/B_KAN_RESCUE/checkpoints_kan_ae_h16_s0/best.pt --hidden-dim 128 --bfs-data isre/trajectories_v7_bfs --recorded-data isre/trajectories_v6_recorded --n 2000 --device cpu --beam 5
```

## B2: Optimizer Rescue

Status: planned, not mixed into B1 by default.

Rationale:

KAN papers often rely on KAN-specific optimizer and grid recipes. The current
training loop uses AdamW for every model. That is protocol-clean, but may be
suboptimal for KAN.

Honest rescue order:

```text
AdamW baseline -> lower-LR continuation -> grid/k sweep -> optional LBFGS polish
```

Do not combine optimizer rescue with architecture rescue until B1 has a clean
baseline. Otherwise we cannot tell what helped.

## B3: Bottleneck Encoder-KAN

New policy name:

```text
--policy encoder_kan
```

Architecture:

```text
ASTEncoder
node_embedding + action_embedding[A]
-> Linear bottleneck[B]
-> efficient_kan KAN([B,H,1])
-> score
```

Purpose:

This is the matched-learned-representation question. It is less directly
interpretable than raw KAN, but fairer against micro-MLP because both arms get
the AST encoder and learned action embedding.

Primary run:

```powershell
python -u -m isre.training.train --data isre/trajectories_v7_bfs --policy encoder_kan --hidden-dim 8 --num-rounds 4 --kan-hidden 16 --kan-action-emb-dim 8 --kan-bottleneck-dim 16 --epochs 15 --device cpu --seed 0 --save-dir leaderboards/v7/B_KAN_RESCUE/checkpoints_encoder_kan_b16_h16_s0
```

Eval:

```powershell
python scripts/eval_neural.py --policy encoder_kan --ckpt leaderboards/v7/B_KAN_RESCUE/checkpoints_encoder_kan_b16_h16_s0/best.pt --hidden-dim 8 --bfs-data isre/trajectories_v7_bfs --recorded-data isre/trajectories_v6_recorded --n 2000 --device cpu --beam 5
```

## Reporting Rule

Keep regimes separate:

```text
raw KAN               = maximum interpretability, least learned machinery
Action-Embedding KAN  = raw features plus learned action identity
Encoder-KAN           = matched representation family, weaker interpretability
micro-MLP             = learned encoder plus MLP scorer
```

Never merge these into one headline without naming the regime.
