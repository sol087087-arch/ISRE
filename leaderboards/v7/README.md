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
