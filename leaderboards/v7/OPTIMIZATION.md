# ISRE v7 Model Optimization Campaign

Policy: append-only experimental ledger. Do not delete runs, checkpoints,
logs, summaries, or negative results.

Dataset: `isre/trajectories_v7_bfs`

Primary metric:
- Mode A free rollout on 2,000 held-out v7 trajectories
- rank by Beam-5 BFS-optimal rate
- tie-breakers: Beam-5 overhead, greedy BFS-optimal, parameter count

Secondary metrics:
- success rate
- catastrophic rate
- validation loss
- gold-path step agreement

## Current Points

| Family | Model | Params | Status |
|---|---|---:|---|
| micro-MLP | h8, policy h16, action emb 8 | 2,793 | done |
| MLP | h128 reference | 474,753 | done |
| KAN | h2 | 560 | done |
| KAN | h4 | 1,120 | done |
| KAN | h16 | 4,480 | done |

## Missing Points

### KAN Width Ladder

Run these to locate the KAN optimum and saturation point:

| Model | Reason |
|---|---|
| KAN h8 | midpoint between h4 and h16 |
| KAN h32 | tests whether h16 is saturated or under-sized |
| KAN h1 | optional lower failure boundary |

### micro-MLP Width Ladder

Use the same compressed recipe as h8:

`encoder hidden = H`, `policy hidden = 2H`, `action embedding = H`.

| Model | Reason |
|---|---|
| micro-MLP h4 | lower compression boundary |
| micro-MLP h16 | checks whether h8 is saturated or under-sized |
| micro-MLP h32 | checks overcapacity / diminishing returns |

## Current Hypothesis

The best v7 point is not the largest model. The likely optimum is a small
structurally aligned model trained on the larger BFS-optimal dataset.

Working hypothesis:
- micro-MLP h8 is near the Pareto optimum.
- KAN h16 is the strongest KAN point seen so far.
- KAN h2/h4 are useful compression points, not the quality optimum.

## Next Additive Runs

Phase 1:
- KAN h8
- KAN h32

Phase 2:
- micro-MLP h4
- micro-MLP h16
- micro-MLP h32

Phase 3:
- multi-seed replication of the best point in each family.
