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
The original bottleneck/encoder channel was too narrow.
Increasing encoder hidden size from 8 to 16 closed most of the gap to micro-MLP h16.
Widening the Encoder16 bottleneck from 32 to 40 closed almost all of the
remaining greedy gap to micro-MLP h32 and beat the micro-MLP line under beam-5.
Increasing AST encoder message-passing rounds from 4 to 6 gave the best
multi-seed Encoder-KAN result found so far.
```

Best current Encoder-KAN point:

```text
Encoder16r6-KAN b40 h16
encoder hidden dim:   16
encoder rounds:       6
bottleneck:           40
KAN hidden:           16
params:               15,528
single-seed best greedy: 94.8%  (seeds 0 and 2)
single-seed best beam-5: 99.6%  (seed 1)
3-seed mean greedy:     94.63% (seeds 0,1,2)
3-seed mean beam-5:     99.53% (seeds 0,1,2)
```

Reference points:

```text
raw KAN h16:       84.5% greedy BFS-optimal
micro-MLP h8:      91.1% greedy BFS-optimal
micro-MLP h16:     93.7% greedy BFS-optimal
micro-MLP h32:     95.0% greedy BFS-optimal
MLP-128 reference: 89.4% greedy BFS-optimal
```

Main takeaway:

> KAN did not simply fail. It failed when deprived of learned AST
> representation. With encoder-matched inputs, a wider bottleneck, and a larger
> encoder hidden state, Encoder-KAN reproducibly beats micro-MLP h16. With
> encoder rounds increased to 6, it nearly matches micro-MLP h32 on greedy
> rollout, matches the micro-MLP beam-5 regime, and remains far smaller than the
> large MLP-128 reference.

## Latest Result: Why It Worked

The key improvement was not just "make KAN bigger." The winning change was to
increase the representation that reaches KAN:

```text
old best local sweep:
  hidden_dim=8, bottleneck=28, kan_hidden=16
  params=7,444
  greedy BFS-optimal=92.2%

new run:
  hidden_dim=16, bottleneck=32, kan_hidden=16
  params=13,920
  greedy BFS-optimal=93.3%

latest run:
  hidden_dim=16, bottleneck=40, kan_hidden=16
  params=15,528
  seed 0: greedy=94.7%, beam-5=99.8%, best_val_loss=0.3123
  seed 1: greedy=94.8%, beam-5=99.2%, best_val_loss=0.3506
  seed 2: greedy=93.8%, beam-5=99.5%, best_val_loss=0.3358
  mean:   greedy=94.43%, beam-5=99.50%, best_val_loss=0.3329

encoder-depth run:
  hidden_dim=16, num_rounds=6, bottleneck=40, kan_hidden=16
  params=15,528
  seed 0: greedy=94.8%, beam-5=99.5%, best_val_loss=0.3401
  seed 1: greedy=94.3%, beam-5=99.6%, best_val_loss=0.3370
  seed 2: greedy=94.8%, beam-5=99.5%, best_val_loss=0.3341
  mean:   greedy=94.63%, beam-5=99.53%, best_val_loss=0.3371
```

Why this helped:

1. The AST encoder now has more room to represent expression structure.
2. The KAN head receives a wider learned state instead of a too-compressed one.
3. Widening the bottleneck from 32 to 40 helped even with the same KAN hidden
   size, which points to representation bandwidth rather than just head size.
4. KAN hidden size stayed moderate, so the gain is mostly representation quality,
   not uncontrolled parameter inflation.
5. Beam-5 was already near saturated; the improvement mainly targets greedy
   top-1 ranking, which is the harder metric.
6. Encoder rounds 6 improved the 3-seed rollout mean; rounds 8 did not, so
   deeper tree message passing helps only up to a point.

This narrows the important gap:

```text
micro-MLP h16:        93.7% greedy, 10,065 params
Encoder16-KAN b32h16: 93.3% greedy, 13,920 params
Encoder16-KAN b40h16 r4: 94.43% mean greedy, 15,528 params
Encoder16-KAN b40h16 r6: 94.63% mean greedy, 15,528 params
micro-MLP h32:        95.0% greedy, 38,049 params
```

So the updated conclusion is:

> Encoder-KAN is now a competitive architecture, not merely a smaller curiosity.
> Across three seeds, the r6 variant beats micro-MLP h16 on mean greedy rollout,
> comes within 0.37 percentage points of micro-MLP h32, and reaches the same
> 99.5% beam-5 regime as the micro-MLP baselines.

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
| Encoder16r6-KAN b40 h16 | 15,528 | 0.3401 | 94.8% | 0.072 | 99.5% | 0.007 |
| Encoder16r8-KAN b40 h16 | 15,528 | 0.3507 | 93.9% | 0.080 | 99.5% | 0.007 |
| Encoder16-KAN b40 h16 | 15,528 | 0.3123 | 94.7% | 0.079 | 99.8% | 0.004 |
| Encoder16-KAN b44 h16 | 16,332 | 0.3412 | 94.8% | 0.073 | 99.5% | 0.007 |
| Encoder16-KAN b40 h20 | 17,168 | 0.3184 | 94.1% | 0.081 | 99.7% | 0.004 |
| Encoder16-KAN b40 h24 | 18,808 | 0.3452 | 94.0% | 0.084 | 99.7% | 0.005 |
| Encoder16-KAN b48 h16 | 17,136 | 0.3625 | 92.2% | 0.111 | 99.2% | 0.011 |
| Encoder-Deep-KAN b40 h16 d2 | 18,088 | 0.3357 | 94.3% | 0.081 | 99.4% | 0.009 |
| Encoder16-KAN b32 h16 | 13,920 | 0.3322 | 93.3% | 0.102 | 99.4% | 0.009 |
| Encoder-KAN b24 h12 | 5,704 | 0.4570 | 91.9% | 0.114 | 99.2% | 0.013 |
| Encoder-KAN b24 h16 | 6,704 | 0.4338 | 91.4% | 0.119 | 99.1% | 0.011 |
| Encoder-KAN b28 h16 | 7,444 | 0.4464 | 92.2% | 0.111 | 99.2% | 0.010 |
| Encoder-KAN b24 h24 | 8,704 | 0.4464 | 90.3% | 0.137 | 99.1% | 0.012 |
| Encoder-KAN b24 h32 | 10,704 | 0.4697 | 90.8% | 0.132 | 99.3% | 0.009 |

## Multi-Seed Stability

The strongest candidates were rerun for seeds `0,1,2` on the same v7 held-out
protocol. This is the robustness check against a lucky single seed.

| Model | Seeds | Params | Mean val loss | Greedy BFS-optimal | Greedy range | Beam-5 BFS-optimal | Beam-5 range |
|---|---|---:|---:|---:|---:|---:|---:|
| Encoder16r6-KAN b40 h16 | 0,1,2 | 15,528 | 0.3371 | 94.63% | 94.3-94.8% | 99.53% | 99.5-99.6% |
| Encoder16-KAN b40 h16 | 0,1,2 | 15,528 | 0.3329 | 94.43% | 93.8-94.8% | 99.50% | 99.2-99.8% |

Raw-backed source:

```text
portfolio/kan_rescue_full_analysis/seed_stability_from_raw_logs.csv
```

Important methodological point:

```text
Validation loss did not perfectly predict rollout quality within the hidden=8 sweep.
b24 h16 had the best hidden=8 val loss.
b28 h16 had the best hidden=8 greedy rollout.
b24 h32 had the best hidden=8 beam-5 rollout.
After increasing encoder hidden dim to 16, Encoder16-KAN b32 h16 became the new
best Encoder-KAN point. Widening that Encoder16 bottleneck to 40 then became the
new best point across val loss, greedy rollout, and beam-5 rollout. Seeds 1 and
2 then reproduced the key finding: b40 h16 stays above micro-MLP h16 on greedy
rollout in the 3-seed mean. Increasing encoder rounds to 6 gave the current best
3-seed Encoder-KAN mean; increasing rounds to 8 or adding a second KAN layer did
not improve the frontier. Micro-MLP h32 remains the narrow greedy leader.
```

This is why the leaderboard ranks by free rollout rather than validation loss
alone.

## Next Stage: Cost-to-Go Ranking Labels

The next training target is no longer "imitate one recorded first action".
For each state, we can score every valid candidate action by:

```text
candidate action -> child state -> remaining BFS distance to canonical form
```

Then every candidate with the minimal finite remaining BFS distance is treated
as correct. This matters because symbolic simplification can have multiple
equally short first moves. A single-label cross-entropy target can punish a
different but still optimal first step; the new ranking target does not.

Implemented path:

```text
--loss rank_cost_to_go
```

This uses:

```text
multi-positive listwise loss over all optimal candidates
pairwise margin loss: optimal candidates should outrank finite non-optimal ones
optional offline JSONL labels via --cost-labels
```

Smoke result:

```text
states labeled:                 20 / 20
candidate actions labeled:      104
multi-optimal states:           10 / 20
recorded gold not cost-optimal: 0 / 20
```

The smoke is small, but it already shows that multi-optimal first moves are
common. It also shows why offline labels are necessary: online BFS labels are
too expensive to recompute for every state on every epoch. The next honest
experiment is therefore:

```text
1. build a larger offline cost-to-go label cache
2. mine hard states from failed or non-optimal greedy rollouts
3. fine-tune the best Encoder-KAN frontier with rank_cost_to_go
4. evaluate on the same v7 held-out split, greedy and beam-5
```

## Hard-State Mining / Probe Cache

The probe-cache path is implemented as an append-only data-mining stage:

```text
scripts/mine_hard_states.py
```

It rolls out a checkpoint, labels each visited state with cost-to-go candidate
labels, and keeps states where the model's top action is not cost-optimal or
where the optimal/non-optimal score margin is small. It writes both:

```text
hard_labels.jsonl       # multi-positive cost-to-go labels
hard_trajectories/      # one trainable trajectory per hard state
```

Important leakage guard:

```text
Default split is train.
The held-out trajectory ids are read from the checkpoint's val_traj_ids.json.
The script refuses to guess the split.
```

Smoke result on a small public checkpoint:

```text
checkpoint: checkpoints_encoder_kan_b28_h16_s0
source split: train
trajectories scanned: 10
rollout states labeled: 24
hard states written: 1
model top not cost-optimal: 1
low-margin states: 1
multi-optimal labeled states: 12 / 24
```

This confirms two things:

```text
1. true hard states are sparse, so targeted mining is the right tool;
2. multi-optimal first moves are common, so single-label CE is too narrow.
```

Warm-start fine-tune is also supported:

```text
--init-ckpt <best.pt>
```

This loads encoder/policy weights only and starts a fresh optimizer for the new
loss/data. A smoke fine-tune on the mined hard trajectory successfully loaded
the checkpoint, consumed `hard_labels.jsonl`, and trained with
`--loss rank_cost_to_go`.

## B4: Encoder Capacity + Bottleneck Rescue

Latest run:

```powershell
python -u -m isre.training.train --data isre/trajectories_v7_bfs --policy encoder_kan --hidden-dim 16 --kan-bottleneck-dim 40 --kan-hidden 16 --epochs 15 --device cpu --seed 0 --save-dir leaderboards/v7/B_KAN_RESCUE/checkpoints_encoder16_kan_b40_h16_s0
```

Eval:

```powershell
python scripts/eval_neural.py --policy encoder_kan --kan-hidden 16 --kan-bottleneck-dim 40 --ckpt leaderboards/v7/B_KAN_RESCUE/checkpoints_encoder16_kan_b40_h16_s0/best.pt --hidden-dim 16 --bfs-data isre/trajectories_v7_bfs --recorded-data isre/trajectories_v7_recorded --n 2000 --device cpu --beam 1
python scripts/eval_neural.py --policy encoder_kan --kan-hidden 16 --kan-bottleneck-dim 40 --ckpt leaderboards/v7/B_KAN_RESCUE/checkpoints_encoder16_kan_b40_h16_s0/best.pt --hidden-dim 16 --bfs-data isre/trajectories_v7_bfs --recorded-data isre/trajectories_v7_recorded --n 2000 --device cpu --beam 5
```

Raw logs:

```text
encoder16_kan_b40_h16_s0.log
eval_encoder16_kan_b40_h16_s0_beam1.log
eval_encoder16_kan_b40_h16_s0_beam5.log
```

Previous encoder-capacity run:

```powershell
python -u -m isre.training.train --data isre/trajectories_v7_bfs --policy encoder_kan --hidden-dim 16 --kan-bottleneck-dim 32 --kan-hidden 16 --epochs 15 --device cpu --seed 0 --save-dir leaderboards/v7/B_KAN_RESCUE/checkpoints_encoder16_kan_b32_h16_s0
```

Eval:

```powershell
python scripts/eval_neural.py --policy encoder_kan --kan-hidden 16 --kan-bottleneck-dim 32 --ckpt leaderboards/v7/B_KAN_RESCUE/checkpoints_encoder16_kan_b32_h16_s0/best.pt --hidden-dim 16 --bfs-data isre/trajectories_v7_bfs --recorded-data isre/trajectories_v7_recorded --n 2000 --device cpu --beam 1
python scripts/eval_neural.py --policy encoder_kan --kan-hidden 16 --kan-bottleneck-dim 32 --ckpt leaderboards/v7/B_KAN_RESCUE/checkpoints_encoder16_kan_b32_h16_s0/best.pt --hidden-dim 16 --bfs-data isre/trajectories_v7_bfs --recorded-data isre/trajectories_v7_recorded --n 2000 --device cpu --beam 5
```

Raw logs:

```text
encoder16_kan_b32_h16_s0.log
eval_encoder16_kan_b32_h16_s0_beam1.log
eval_encoder16_kan_b32_h16_s0_beam5.log
```

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
