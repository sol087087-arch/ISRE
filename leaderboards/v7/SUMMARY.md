# ISRE v7 Leaderboard Summary

Status: v7 leaderboard built for micro-MLP h8, KAN h16, and MLP-128 reference.

Dataset: `isre/trajectories_v7_bfs`

Dataset shape:
- trajectories: 98,472
- total training pairs: 293,631
- train pairs: 264,265
- validation pairs: 29,366
- held-out validation trajectories: 9,847
- max BFS difficulty: 12
- split seed: 1234

Primary metric: Mode A free rollout on 2,000 held-out v7 trajectories.

| Rank | Model | Params | Greedy success | Greedy BFS-optimal | Greedy overhead | Beam-5 success | Beam-5 BFS-optimal | Beam-5 overhead | Best val loss |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | micro-MLP h16 seed0 | 10,065 | 100.0% | 93.7% | 0.087 | 100.0% | 99.5% | 0.007 | 0.3326 |
| 1 | micro-MLP h32 seed0 | 38,049 | 100.0% | 95.0% | 0.076 | 100.0% | 99.5% | 0.011 | 0.3110 |
| 3 | micro-MLP h8 seed0 | 2,793 | 100.0% | 91.1% | 0.123 | 100.0% | 99.1% | 0.013 | 0.4967 |
| 4 | KAN h16 seed0 | 4,480 | 100.0% | 84.5% | 0.247 | 100.0% | 98.5% | 0.024 | 0.5877 |
| 4 | MLP-128 seed0 reference | 474,753 | 100.0% | 89.4% | 0.176 | 100.0% | 98.5% | 0.024 | 0.4996 |
| 6 | KAN h8 seed0 | 2,240 | 100.0% | 87.4% | 0.214 | 100.0% | 98.4% | 0.025 | 0.6171 |
| 6 | KAN h32 seed0 | 8,960 | 100.0% | 86.0% | 0.245 | 100.0% | 98.4% | 0.024 | 0.5802 |
| 8 | micro-MLP h4 seed0 | 837 | 100.0% | 88.8% | 0.159 | 100.0% | 98.2% | 0.024 | 0.7065 |
| 9 | KAN h4 seed0 | 1,120 | 100.0% | 83.2% | 0.272 | 100.0% | 97.2% | 0.043 | 0.6561 |
| 10 | KAN h2 seed0 | 560 | 100.0% | 81.2% | 0.306 | 100.0% | 96.5% | 0.043 | 0.7356 |

Interpretation:
- micro-MLP h16/h32 are the current v7 leaders by Beam-5 BFS-optimal rate.
- h16 wins the beam-overhead tie-break; h32 wins greedy BFS-optimal and validation loss.
- micro-MLP h8 remains the smaller Pareto point: 2,793 parameters and 99.1% Beam-5 BFS-optimal.
- MLP-128 does not beat the micro-MLP reference on v7 despite being ~170x larger.
- KAN h16 and MLP-128 tie on beam-5 BFS-optimal and overhead, but KAN is much smaller and MLP-128 is better on greedy path optimality.
- KAN h8/h16/h32 show KAN saturation around h16. h32 improves validation loss but not rollout optimality.
- KAN h4 and h2 confirm a smooth compression ladder: even 560 parameters keeps 100% success and 96.5% beam-5 BFS-optimal, but the gap to micro-MLP becomes visible.
- The next critical control is multi-seed replication for micro-MLP and the best KAN point.

Notes:
- Mode B recorded-teacher categories are intentionally not used for v7 ranking because v7 has no paired recorded trajectory arm.
- `div_teacher_undefined = 100%` is expected in v7 eval logs.
- No v6 artifacts are overwritten; this leaderboard is additive only.

Log links:
- `evals/eval_v7_micro_mlp_h8_s0_beam1.log`
- `evals/eval_v7_micro_mlp_h8_s0_beam5.log`
- `evals/eval_v7_kan_h16_s0_beam1.log`
- `evals/eval_v7_kan_h16_s0_beam5.log`
- `evals/eval_v7_mlp128_s0_beam1.log`
- `evals/eval_v7_mlp128_s0_beam5.log`
- `evals/eval_v7_kan_h4_s0_beam1.log`
- `evals/eval_v7_kan_h4_s0_beam5.log`
- `evals/eval_v7_kan_h2_s0_beam1.log`
- `evals/eval_v7_kan_h2_s0_beam5.log`
- `evals/eval_v7_kan_h8_s0_beam1.log`
- `evals/eval_v7_kan_h8_s0_beam5.log`
- `evals/eval_v7_kan_h32_s0_beam1.log`
- `evals/eval_v7_kan_h32_s0_beam5.log`
- `evals/eval_v7_micro_mlp_h4_s0_beam1.log`
- `evals/eval_v7_micro_mlp_h4_s0_beam5.log`
- `evals/eval_v7_micro_mlp_h16_s0_beam1.log`
- `evals/eval_v7_micro_mlp_h16_s0_beam5.log`
- `evals/eval_v7_micro_mlp_h32_s0_beam1.log`
- `evals/eval_v7_micro_mlp_h32_s0_beam5.log`
