# Mixed Hard-State Fine-Tune Probe: h21 snapshot

This run uses the long hard-state mining process for
`encoder24r6_kan_b40_h16_s1`, snapshotted after the first 21 hard states were
found.

## Mining Snapshot

Source checkpoint:

```text
checkpoints_encoder24r6_kan_b40_h16_s1/best.pt
```

Snapshot:

```text
hard states:       21
rehearsal trajs:   105
mixed ratio:       1 hard : 5 rehearsal
mixed trajectories: 126
```

The long mining job is intentionally separate from this snapshot and can keep
running toward 1000-2000 train trajectories / 50 hard states.

## Mixed Fine-Tune

Training command shape:

```text
--loss rank_cost_to_go
--cost-labels hard_labels.jsonl
--init-ckpt checkpoints_encoder24r6_kan_b40_h16_s1/best.pt
--lr 0.001
--epochs 40
```

Mixed dataset:

```text
21 mined hard one-step trajectories
105 ordinary train-split rehearsal trajectories
```

## Held-Out Rollout Results

All evaluations use the original checkpoint held-out split:

```text
dataset: isre/trajectories_v7_bfs
n: 2000 held-out trajectories
device: CPU
```

| Model | Greedy BFS-optimal | Greedy overhead | Beam-5 BFS-optimal | Beam-5 overhead | Catastrophic |
|---|---:|---:|---:|---:|---:|
| Base s1 | 96.0% | 0.050 | 99.7% | 0.005 | 0.0% |
| Mixed FT best.pt | 96.4% | 0.043 | 99.7% | 0.005 | 0.0% |
| Mixed FT last.pt | 95.2% | 0.059 | 99.6% | 0.005 | 0.1% |

## Hard-State Fix Rate

Measured directly on the 21 mined hard states:

| Model | Fixed hard states | Mean top-minus-best-optimal score | Max top-minus-best-optimal score |
|---|---:|---:|---:|
| Base s1 | 0/21 | 3.2967 | 9.9142 |
| Mixed FT best.pt | 11/21 | 1.4951 | 7.8341 |
| Mixed FT last.pt | 11/21 | 0.9567 | 5.2198 |

Lower score margin is better here because these are states where the model's
top action is initially non-optimal.

## Takeaway

The mixed ranking signal is real:

```text
Base fixes 0/21 mined hard states.
Mixed fine-tune fixes 11/21 and improves greedy BFS-optimality from 96.0% to
96.4% on the same held-out eval.
```

But over-training hurts:

```text
last.pt continues to reduce hard-state margins, but held-out greedy rollout
drops to 95.2% and introduces 0.1% catastrophic overhead.
```

Conclusion:

```text
Use mixed hard-state fine-tuning with early stopping.
Do not train to exhaustion on the small hard cache.
The next useful run is a larger hard cache, then the same mixed protocol with
best.pt selection, not last.pt.
```

Raw logs:

```text
eval_mixed_ft_best_beam1.log
eval_mixed_ft_best_beam5.log
eval_mixed_ft_last_beam1.log
eval_mixed_ft_last_beam5.log
```
