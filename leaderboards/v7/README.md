# ISRE v7 Leaderboard

Status: building. This is an additive leaderboard layer; v6 artifacts are not removed or overwritten.

Dataset: `isre/trajectories_v7_bfs`

Known dataset shape:
- trajectories: 98,472
- training pairs: 293,631
- generation: harder/bigger v7 BFS corpus

Primary ranking metrics:
- success_rate on held-out trajectories
- BFS-optimal rate
- overhead vs BFS
- catastrophic_rate

Initial models queued:

| Run | Model | Params target | Train dataset | Status |
|---|---|---:|---|---|
| v7_micro_mlp_h8_s0 | micro-MLP h8 | 2,793 | v7 BFS | training queued |
| v7_kan_h16_s0 | KAN h16 | 4,480 | v7 BFS | training queued |

Reference to add later:
- MLP-128 reference baseline, if we decide the long runtime is worth it.
- KAN h4/h2 compression probes after h16 result lands.

No deletion policy: append new runs and summaries only.

## Sad Conclusions

The interpretability hypothesis did not win as cleanly as hoped.

KAN remains useful, but not as the best-quality policy on v7. The best KAN
point is `KAN h16`: 4,480 parameters, 98.5% Beam-5 BFS-optimality, and about
2.7 hours of CPU training. That is a strong, fast, interpretable baseline.

But the best micro-MLP points are better:

| Model | Params | Beam-5 BFS-optimal | Beam-5 overhead |
|---|---:|---:|---:|
| micro-MLP h16 | 10,065 | 99.5% | 0.007 |
| micro-MLP h32 | 38,049 | 99.5% | 0.011 |
| KAN h16 | 4,480 | 98.5% | 0.024 |

So the painful result is:

**Interpretability is not free.**

KAN buys interpretability and much faster training, but it gives up about one
percentage point of path optimality compared with the best micro-MLP.

Another painful result:

**Bigger is not automatically better.**

The 474,753-parameter MLP-128 reference does not beat the micro-MLP h16/h32
models, despite being much larger. The useful signal is not "make the model
bigger"; it is "match the architecture capacity to the structured dataset."

Current honest framing:

- micro-MLP is the quality winner.
- KAN is the interpretable and fast-to-train baseline.
- KAN is not dead, but it is not the absolute winner.
- The v7 dataset exposes a real capacity/quality frontier rather than a
  single magic architecture.

This is disappointing for the strongest KAN hope, but it is a better
scientific result: the project now has a measured trade-off between
interpretability, training speed, parameter count, and path optimality.
