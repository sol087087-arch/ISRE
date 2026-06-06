# B: KAN Rescue Campaign

Portfolio-ready summary of the ISRE v7 KAN rescue experiment.

## Executive Summary

We tested whether KAN underperformed because KAN itself was weak on symbolic
algebra simplification, or because the original KAN arm received a much poorer
input representation than the MLP baselines.

The answer is nuanced and useful:

```text
Raw feature-only KAN was underpowered.
Action embeddings helped a little.
ASTEncoder + Encoder-KAN helped a lot.
The original bottleneck was too narrow.
The best Encoder-KAN point now beats micro-MLP h8 on greedy rollout.
```

Best current Encoder-KAN point:

```text
Encoder-KAN b28 h16
params:               7,444
greedy BFS-optimal:   92.2%
beam-5 BFS-optimal:   99.2%
mean greedy overhead: 0.111
```

Reference points:

```text
raw KAN h16:       84.5% greedy BFS-optimal
micro-MLP h8:      91.1% greedy BFS-optimal
micro-MLP h16:     93.7% greedy BFS-optimal
MLP-128 reference: 89.4% greedy BFS-optimal
```

Main takeaway:

> KAN did not simply fail. It failed when deprived of learned AST
> representation. With encoder-matched inputs and a tuned bottleneck,
> Encoder-KAN becomes competitive while staying far smaller than the large MLP
> reference.

## Figures

![KAN rescue portfolio summary](kan_rescue_portfolio_summary.png)

![Encoder-KAN local sweep map](kan_rescue_sweep_map.png)

Historical early-campaign chart:

![KAN rescue comparison](kan_rescue_comparison.png)

## Final Local Sweep

All runs use `isre/trajectories_v7_bfs`, seed 0, the trajectory-level held-out
split from `val_traj_ids.json`, and Mode A free rollout on 2,000 held-out
trajectories.

| Model | Params | Best val loss | Greedy BFS-optimal | Greedy overhead | Beam-5 BFS-optimal | Beam-5 overhead |
|---|---:|---:|---:|---:|---:|---:|
| Encoder-KAN b24 h12 | 5,704 | 0.4570 | 91.9% | 0.114 | 99.2% | 0.013 |
| Encoder-KAN b24 h16 | 6,704 | 0.4338 | 91.4% | 0.119 | 99.1% | 0.011 |
| Encoder-KAN b28 h16 | 7,444 | 0.4464 | 92.2% | 0.111 | 99.2% | 0.010 |
| Encoder-KAN b24 h24 | 8,704 | 0.4464 | 90.3% | 0.137 | 99.1% | 0.012 |
| Encoder-KAN b24 h32 | 10,704 | 0.4697 | 90.8% | 0.132 | 99.3% | 0.009 |

Important methodological point:

```text
Validation loss did not perfectly predict rollout quality.
b24 h16 has the best val loss.
b28 h16 has the best greedy rollout.
b24 h32 has the best beam-5 rollout.
```

This is why the leaderboard ranks by free rollout rather than validation loss
alone.

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
