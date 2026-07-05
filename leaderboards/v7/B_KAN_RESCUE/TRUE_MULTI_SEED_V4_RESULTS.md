# True Multi-Seed V4 Fine-Tune Results

This run checks whether the cost-to-go hard-state rescue result is repeatable across independently mined seeds, not just a single lucky probe.

## Setup

- Base policy: `encoder_kan`
- Encoder: hidden dim 24, 6 rounds
- KAN head: bottleneck 40, hidden 16
- Loss: `rank_cost_to_go`
- Fine-tune data per seed: 50 mined hard states plus 250 rehearsal trajectories
- Held-out evaluation: v7 split, 2,000 trajectories
- Checkpoints evaluated: `best.pt` and `last.pt`
- Search modes: greedy (`beam=1`) and beam search (`beam=5`)

## Held-Out Rollout Metrics

| Seed | Checkpoint | Beam | Success | BFS-optimal | Mean overhead | p95 | Catastrophic | Step match |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| s0 | best | 1 | 100.0% | 96.6% | 0.045 | 0 | 0.2% | 71.3% |
| s0 | best | 5 | 100.0% | 99.7% | 0.004 | 0 | 0.1% | 71.3% |
| s0 | last | 1 | 100.0% | 96.5% | 0.048 | 0 | 0.2% | 71.1% |
| s0 | last | 5 | 100.0% | 99.8% | 0.002 | 0 | 0.0% | 71.1% |
| s2 | best | 1 | 100.0% | 96.6% | 0.045 | 0 | 0.2% | 71.2% |
| s2 | best | 5 | 100.0% | 99.6% | 0.004 | 0 | 0.0% | 71.2% |
| s2 | last | 1 | 100.0% | 97.0% | 0.040 | 0 | 0.2% | 71.2% |
| s2 | last | 5 | 100.0% | 99.7% | 0.004 | 0 | 0.0% | 71.2% |

## Takeaway

The improvement is repeatable across true per-seed hard-state mining. Greedy rollouts reach 96.5-97.0% BFS-optimality with 100% success, and beam-5 reaches 99.6-99.8% BFS-optimality with near-zero overhead.

The strongest greedy point in this batch is `s2/last.pt` at 97.0% BFS-optimality. The strongest beam-5 point is `s0/last.pt` at 99.8% BFS-optimality with 0.002 mean overhead and 0.0% catastrophic rate.

These results support the main hypothesis: the Encoder-KAN policy benefits from targeted cost-to-go hard-state mining, and the gain is not a one-seed artifact.

## Raw Logs

The metrics above were extracted directly from these append-only eval logs:

- `mixed_probe_encoder24r6_b40_h16_s0_true_seed_v4/eval_true_seed_v4_best_beam1_v7split_20260703_092055.log`
- `mixed_probe_encoder24r6_b40_h16_s0_true_seed_v4/eval_true_seed_v4_best_beam5_v7split_20260703_092055.log`
- `mixed_probe_encoder24r6_b40_h16_s0_true_seed_v4/eval_true_seed_v4_last_beam1_v7split_20260703_092055.log`
- `mixed_probe_encoder24r6_b40_h16_s0_true_seed_v4/eval_true_seed_v4_last_beam5_v7split_20260703_092055.log`
- `mixed_probe_encoder24r6_b40_h16_s2_true_seed_v4/eval_true_seed_v4_best_beam1_v7split_20260703_092055.log`
- `mixed_probe_encoder24r6_b40_h16_s2_true_seed_v4/eval_true_seed_v4_best_beam5_v7split_20260703_092055.log`
- `mixed_probe_encoder24r6_b40_h16_s2_true_seed_v4/eval_true_seed_v4_last_beam1_v7split_20260703_092055.log`
- `mixed_probe_encoder24r6_b40_h16_s2_true_seed_v4/eval_true_seed_v4_last_beam5_v7split_20260703_092055.log`
