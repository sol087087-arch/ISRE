# Hard-State Probe: Encoder24r6-KAN b40 h16 s1

This probe tests whether cost-to-go hard-state mining can improve the strongest
current Encoder-KAN frontier without changing the held-out evaluation protocol.

## Setup

Base checkpoint:

```text
checkpoints_encoder24r6_kan_b40_h16_s1/best.pt
```

Mining protocol:

```text
split: train
max_trajectories: 120
max_steps_per_rollout: 30
max_expansions: 2000
max_depth: 12
margin_threshold: 0.25
```

Mining result:

```text
rollout states seen:       334
unique states seen:        328
labeled states:            328
hard states written:       2
model top not optimal:     2
low-margin states:         2
multi-optimal states:      183 / 328
successes:                 120 / 120
loops/dead-ends/timeouts:  0 / 0 / 0
```

Interpretation:

```text
Hard states are sparse for this strong checkpoint.
Multi-optimal first moves are common.
Targeted ranking fine-tuning is a plausible next-stage tool, but the mined set
must be much larger than two states for a stable training signal.
```

## Fine-Tune Probes

Both probes warm-start from the base checkpoint and train only on the two mined
hard states with `--loss rank_cost_to_go`.

| Variant | Hard states fixed | Greedy BFS-optimal | Greedy overhead | Beam-5 BFS-optimal | Beam-5 overhead |
|---|---:|---:|---:|---:|---:|
| Base s1 | 0/2 | 96.0% | 0.050 | 99.7% | 0.005 |
| FT lr=1e-4, 120 epochs | 0/2 | 96.1% | 0.049 | 99.7% | 0.005 |
| FT lr=1e-3, 120 epochs | 1/2 | 96.7% | 0.043 | 99.6% | 0.006 |

Held-out eval:

```text
dataset: isre/trajectories_v7_bfs
split: original checkpoint val_traj_ids.json
n: 2000 held-out trajectories
device: CPU
metrics: Mode A free rollout, greedy and beam-5
```

Raw logs:

```text
eval_base_s1_beam1.log
eval_base_s1_beam5.log
eval_ft_last_beam1.log
eval_ft_last_beam5.log
eval_ft_lr1e3_last_beam1.log
eval_ft_lr1e3_last_beam5.log
```

## Takeaway

The first hard-state probe is positive but not conclusive:

```text
lr=1e-3 fine-tuning improved greedy path optimality from 96.0% to 96.7%
on the same 2,000 held-out trajectories, while preserving 100% success and
0% catastrophic rate.
```

Beam-5 stayed essentially saturated:

```text
99.7% -> 99.6% BFS-optimal
```

This suggests the ranking signal is useful for greedy top-1 ordering, which is
exactly where the remaining gap lives. The next experiment should mine a larger
hard-state cache, then fine-tune with a mixed objective instead of only two hard
states.
