# ISRE v7 Full KAN/MLP Analysis

This folder is generated from raw train/eval logs only.

Source-of-truth policy:

```text
No metric in this report is taken from chat memory, SUMMARY.md, or RESULTS.md.
Metrics are parsed from train/eval log files and checkpoint inventory.
Historical checkpoints without matching eval logs are inventoried, not ranked.
```

## Executive Summary

- Best Encoder-KAN greedy point: `encoder16_kan_b40_h16_s1` with 94.8% greedy BFS-optimal and 15,528 params.
- Best Encoder-KAN beam-5 point: `encoder16_kan_b40_h16_s0` with 99.8% beam-5 BFS-optimal.
- Best raw KAN greedy point: `kan_h8_s0` with 87.4% greedy BFS-optimal.
- Best small MLP greedy point: `micro_mlp_h32_s0` with 95.0% greedy BFS-optimal.
- Best multi-seed Encoder-KAN group: `encoder16r6_kan_b40_h16` with n=3 seeds (0,1,2), mean greedy 94.6333%, range 94.3-94.8%, mean beam-5 99.5333%.
- Encoder16-KAN b40 h16 stability: n=3 seeds (0,1,2), mean greedy 94.4333%, range 93.8-94.8%, mean beam-5 99.5%.
- Micro-MLP h8 reference: 91.1% greedy BFS-optimal with 2,793 params.
- MLP-128 reference: 89.4% greedy BFS-optimal with 474,753 params.

Plain-language conclusion:

```text
Raw KAN underperformed because it was given poor hand-crafted inputs.
KAN-AE h16 is now ranked: 85.9% greedy, 98.6% beam-5.
ASTEncoder + KAN head recovered a large part of the gap.
The bottleneck/head geometry matters: bigger is not automatically better.
Validation loss does not reliably rank rollout quality.
```

## Full Leaderboard, Greedy First

| model | family | params | best val loss | greedy BFS-opt % | greedy overhead | beam-5 BFS-opt % | beam-5 overhead |
|---|---|---|---|---|---|---|---|
| micro_mlp_h32_s0 | micro-MLP | 38,049 | 0.311 | 95 | 0.076 | 99.5 | 0.011 |
| encoder16_kan_b40_h16_s1 | Encoder-KAN | 15,528 | 0.3506 | 94.8 | 0.077 | 99.2 | 0.013 |
| encoder16_kan_b44_h16_s0 | Encoder-KAN | 16,332 | 0.3412 | 94.8 | 0.073 | 99.5 | 0.007 |
| encoder16r6_kan_b40_h16_s0 | Encoder-KAN | 15,528 | 0.3401 | 94.8 | 0.072 | 99.5 | 0.007 |
| encoder16r6_kan_b40_h16_s2 | Encoder-KAN | 15,528 | 0.3341 | 94.8 | 0.077 | 99.5 | 0.006 |
| encoder16_kan_b40_h16_s0 | Encoder-KAN | 15,528 | 0.3123 | 94.7 | 0.079 | 99.8 | 0.004 |
| encoder16_kan_b36_h16_s0 | Encoder-KAN | 14,724 | 0.3425 | 94.3 | 0.083 | 99.7 | 0.005 |
| encoder16r6_kan_b40_h16_s1 | Encoder-KAN | 15,528 | 0.337 | 94.3 | 0.074 | 99.6 | 0.005 |
| encoder_deep_kan_b40_h16_d2_s0 | Encoder-KAN | 18,088 | 0.3357 | 94.3 | 0.081 | 99.4 | 0.009 |
| encoder16_kan_b40_h20_s0 | Encoder-KAN | 17,168 | 0.3184 | 94.1 | 0.081 | 99.7 | 0.004 |
| encoder16_kan_b40_h24_s0 | Encoder-KAN | 18,808 | 0.3452 | 94 | 0.084 | 99.7 | 0.005 |
| encoder16r8_kan_b40_h16_s0 | Encoder-KAN | 15,528 | 0.3507 | 93.9 | 0.08 | 99.5 | 0.007 |
| encoder16_kan_b40_h16_s2 | Encoder-KAN | 15,528 | 0.3358 | 93.8 | 0.082 | 99.5 | 0.006 |
| micro_mlp_h16_s0 | micro-MLP | 10,065 | 0.3326 | 93.7 | 0.087 | 99.5 | 0.007 |
| encoder16_kan_b32_h16_s0 | Encoder-KAN | 13,920 | 0.3322 | 93.3 | 0.102 | 99.4 | 0.009 |
| encoder16_kan_b48_h16_s0 | Encoder-KAN | 17,136 | 0.3625 | 92.2 | 0.111 | 99.2 | 0.011 |
| encoder_kan_b28_h16_s0 | Encoder-KAN | 7,444 | 0.4464 | 92.2 | 0.111 | 99.2 | 0.01 |
| encoder_kan_b24_h12_s0 | Encoder-KAN | 5,704 | 0.457 | 91.9 | 0.114 | 99.2 | 0.013 |
| encoder_kan_b24_h16_s0 | Encoder-KAN | 6,704 | 0.4338 | 91.4 | 0.119 | 99.1 | 0.011 |
| micro_mlp_h8_s0 | micro-MLP | 2,793 | 0.4967 | 91.1 | 0.123 | 99.1 | 0.013 |
| encoder_kan_b16_h32_s0 | Encoder-KAN | 7,944 | 0.4599 | 90.9 | 0.139 | 99.2 | 0.011 |
| encoder_kan_b24_h32_s0 | Encoder-KAN | 10,704 | 0.4697 | 90.8 | 0.132 | 99.3 | 0.009 |
| encoder_kan_b32_h32_s0 | Encoder-KAN | 13,464 | 0.4464 | 90.5 | 0.128 | 99.1 | 0.011 |
| encoder_kan_b24_h24_s0 | Encoder-KAN | 8,704 | 0.4464 | 90.3 | 0.137 | 99.1 | 0.012 |
| encoder_kan_b16_h16_s0 | Encoder-KAN | 5,224 | 0.4763 | 89.5 | 0.168 | 98.8 | 0.018 |
| mlp128_s0 | MLP-128 | 474,753 | 0.4996 | 89.4 | 0.176 | 98.5 | 0.024 |
| micro_mlp_h4_s0 | micro-MLP | 837 | 0.7065 | 88.8 | 0.159 | 98.2 | 0.024 |
| encoder_kan_b20_h24_s0 | Encoder-KAN | 7,644 | 0.4387 | 88.3 | 0.173 | 98.8 | 0.016 |
| kan_h8_s0 | raw KAN | 2,240 | 0.6171 | 87.4 | 0.214 | 98.4 | 0.025 |
| kan_h32_s0 | raw KAN | 8,960 | 0.5802 | 86 | 0.245 | 98.4 | 0.024 |
| kan_ae_h16_s0 | KAN-AE | 5,840 | 0.5924 | 85.9 | 0.234 | 98.6 | 0.021 |
| kan_h16_s0 | raw KAN | 4,480 | 0.5877 | 84.5 | 0.247 | 98.5 | 0.024 |
| kan_h4_s0 | raw KAN | 1,120 | 0.6561 | 83.2 | 0.272 | 97.2 | 0.043 |
| kan_h2_s0 | raw KAN | 560 | 0.7356 | 81.2 | 0.306 | 96.5 | 0.043 |

## Multi-Seed Stability

Rows here are grouped by architecture name with the `_sN` suffix removed. This is the repeatability check; it is not based on chat memory.

| model base | family | seeds | n | params | mean val loss | std val loss | mean greedy BFS-opt % | min greedy | max greedy | std greedy | mean beam-5 BFS-opt % | min beam-5 | max beam-5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| encoder16r6_kan_b40_h16 | Encoder-KAN | 0,1,2 | 3 | 15,528 | 0.3371 | 0.003 | 94.63 | 94.3 | 94.8 | 0.2887 | 99.53 | 99.5 | 99.6 |
| encoder16_kan_b40_h16 | Encoder-KAN | 0,1,2 | 3 | 15,528 | 0.3329 | 0.0193 | 94.43 | 93.8 | 94.8 | 0.5508 | 99.5 | 99.2 | 99.8 |

## Beam-5 Ranking

| model | family | params | beam-5 BFS-opt % | beam-5 overhead | greedy BFS-opt % |
|---|---|---|---|---|---|
| encoder16_kan_b40_h16_s0 | Encoder-KAN | 15,528 | 99.8 | 0.004 | 94.7 |
| encoder16_kan_b40_h20_s0 | Encoder-KAN | 17,168 | 99.7 | 0.004 | 94.1 |
| encoder16_kan_b36_h16_s0 | Encoder-KAN | 14,724 | 99.7 | 0.005 | 94.3 |
| encoder16_kan_b40_h24_s0 | Encoder-KAN | 18,808 | 99.7 | 0.005 | 94 |
| encoder16r6_kan_b40_h16_s1 | Encoder-KAN | 15,528 | 99.6 | 0.005 | 94.3 |
| encoder16_kan_b40_h16_s2 | Encoder-KAN | 15,528 | 99.5 | 0.006 | 93.8 |
| encoder16r6_kan_b40_h16_s2 | Encoder-KAN | 15,528 | 99.5 | 0.006 | 94.8 |
| encoder16_kan_b44_h16_s0 | Encoder-KAN | 16,332 | 99.5 | 0.007 | 94.8 |
| encoder16r6_kan_b40_h16_s0 | Encoder-KAN | 15,528 | 99.5 | 0.007 | 94.8 |
| encoder16r8_kan_b40_h16_s0 | Encoder-KAN | 15,528 | 99.5 | 0.007 | 93.9 |
| micro_mlp_h16_s0 | micro-MLP | 10,065 | 99.5 | 0.007 | 93.7 |
| micro_mlp_h32_s0 | micro-MLP | 38,049 | 99.5 | 0.011 | 95 |
| encoder16_kan_b32_h16_s0 | Encoder-KAN | 13,920 | 99.4 | 0.009 | 93.3 |
| encoder_deep_kan_b40_h16_d2_s0 | Encoder-KAN | 18,088 | 99.4 | 0.009 | 94.3 |
| encoder_kan_b24_h32_s0 | Encoder-KAN | 10,704 | 99.3 | 0.009 | 90.8 |
| encoder_kan_b28_h16_s0 | Encoder-KAN | 7,444 | 99.2 | 0.01 | 92.2 |
| encoder16_kan_b48_h16_s0 | Encoder-KAN | 17,136 | 99.2 | 0.011 | 92.2 |
| encoder_kan_b16_h32_s0 | Encoder-KAN | 7,944 | 99.2 | 0.011 | 90.9 |
| encoder16_kan_b40_h16_s1 | Encoder-KAN | 15,528 | 99.2 | 0.013 | 94.8 |
| encoder_kan_b24_h12_s0 | Encoder-KAN | 5,704 | 99.2 | 0.013 | 91.9 |
| encoder_kan_b24_h16_s0 | Encoder-KAN | 6,704 | 99.1 | 0.011 | 91.4 |
| encoder_kan_b32_h32_s0 | Encoder-KAN | 13,464 | 99.1 | 0.011 | 90.5 |
| encoder_kan_b24_h24_s0 | Encoder-KAN | 8,704 | 99.1 | 0.012 | 90.3 |
| micro_mlp_h8_s0 | micro-MLP | 2,793 | 99.1 | 0.013 | 91.1 |
| encoder_kan_b20_h24_s0 | Encoder-KAN | 7,644 | 98.8 | 0.016 | 88.3 |
| encoder_kan_b16_h16_s0 | Encoder-KAN | 5,224 | 98.8 | 0.018 | 89.5 |
| kan_ae_h16_s0 | KAN-AE | 5,840 | 98.6 | 0.021 | 85.9 |
| kan_h16_s0 | raw KAN | 4,480 | 98.5 | 0.024 | 84.5 |
| mlp128_s0 | MLP-128 | 474,753 | 98.5 | 0.024 | 89.4 |
| kan_h32_s0 | raw KAN | 8,960 | 98.4 | 0.024 | 86 |
| kan_h8_s0 | raw KAN | 2,240 | 98.4 | 0.025 | 87.4 |
| micro_mlp_h4_s0 | micro-MLP | 837 | 98.2 | 0.024 | 88.8 |
| kan_h4_s0 | raw KAN | 1,120 | 97.2 | 0.043 | 83.2 |
| kan_h2_s0 | raw KAN | 560 | 96.5 | 0.043 | 81.2 |

## Where KAN Wins

- Encoder-KAN beats raw KAN by a wide margin once it receives learned AST representations.
- Best raw KAN greedy: 87.4%. Best Encoder-KAN greedy: 94.8%.
- Encoder-KAN is dramatically smaller than MLP-128 while beating it on greedy rollout.
- `encoder16_kan_b40_h16_s1` has 15,528 params vs MLP-128's 474,753 params.
- Encoder-KAN can beat micro-MLP h8 on greedy rollout in the tuned local sweep.
- Concrete h8 comparison: `encoder16_kan_b40_h16_s1` 94.8% vs `micro_mlp_h8_s0` 91.1%.

## Where KAN Still Loses

- The strongest micro-MLP h32 point still narrowly leads greedy rollout.
- Encoder16-KAN b40 h16 and encoder16r6_kan b40 h16 both beat micro-MLP h16 on greedy in multiple seeds.
- Encoder-KAN is sensitive to bottleneck/head geometry; b20 h24 had good validation loss but poor rollout.
- A KAN head is not a drop-in win: raw feature-only KAN is clearly weaker.

## Methodological Findings

1. Representation quality matters more than raw KAN enthusiasm.
2. The 24 -> 16 bottleneck was too narrow for the encoder representation.
3. Too much widening is also not monotonic: b32 h32 did not become the best greedy model.
4. Rollout metrics are primary; cross-entropy validation loss is not sufficient.
5. Beam search and greedy emphasize different strengths.

## Missing Rollout Coverage

These models have raw training logs/checkpoints but no matching raw eval logs in this package, so they are not used for rollout claims:

None.

## Source Verification

Every row below points back to raw train/eval logs.

| model | train source | greedy source | beam-5 source |
|---|---|---|---|
| encoder16_kan_b32_h16_s0 | leaderboards/v7/B_KAN_RESCUE/encoder16_kan_b32_h16_s0.log | leaderboards/v7/B_KAN_RESCUE/eval_encoder16_kan_b32_h16_s0_beam1.log | leaderboards/v7/B_KAN_RESCUE/eval_encoder16_kan_b32_h16_s0_beam5.log |
| encoder16_kan_b36_h16_s0 | leaderboards/v7/B_KAN_RESCUE/encoder16_kan_b36_h16_s0.log | leaderboards/v7/B_KAN_RESCUE/eval_encoder16_kan_b36_h16_s0_beam1.log | leaderboards/v7/B_KAN_RESCUE/eval_encoder16_kan_b36_h16_s0_beam5.log |
| encoder16_kan_b40_h16_s0 | leaderboards/v7/B_KAN_RESCUE/encoder16_kan_b40_h16_s0.log | leaderboards/v7/B_KAN_RESCUE/eval_encoder16_kan_b40_h16_s0_beam1.log | leaderboards/v7/B_KAN_RESCUE/eval_encoder16_kan_b40_h16_s0_beam5.log |
| encoder16_kan_b40_h16_s1 | leaderboards/v7/B_KAN_RESCUE/encoder16_kan_b40_h16_s1.log | leaderboards/v7/B_KAN_RESCUE/eval_encoder16_kan_b40_h16_s1_beam1.log | leaderboards/v7/B_KAN_RESCUE/eval_encoder16_kan_b40_h16_s1_beam5.log |
| encoder16_kan_b40_h16_s2 | leaderboards/v7/B_KAN_RESCUE/encoder16_kan_b40_h16_s2.log | leaderboards/v7/B_KAN_RESCUE/eval_encoder16_kan_b40_h16_s2_beam1.log | leaderboards/v7/B_KAN_RESCUE/eval_encoder16_kan_b40_h16_s2_beam5.log |
| encoder16_kan_b40_h20_s0 | leaderboards/v7/B_KAN_RESCUE/encoder16_kan_b40_h20_s0.log | leaderboards/v7/B_KAN_RESCUE/eval_encoder16_kan_b40_h20_s0_beam1.log | leaderboards/v7/B_KAN_RESCUE/eval_encoder16_kan_b40_h20_s0_beam5.log |
| encoder16_kan_b40_h24_s0 | leaderboards/v7/B_KAN_RESCUE/encoder16_kan_b40_h24_s0.log | leaderboards/v7/B_KAN_RESCUE/eval_encoder16_kan_b40_h24_s0_beam1.log | leaderboards/v7/B_KAN_RESCUE/eval_encoder16_kan_b40_h24_s0_beam5.log |
| encoder16_kan_b44_h16_s0 | leaderboards/v7/B_KAN_RESCUE/encoder16_kan_b44_h16_s0.log | leaderboards/v7/B_KAN_RESCUE/eval_encoder16_kan_b44_h16_s0_beam1.log | leaderboards/v7/B_KAN_RESCUE/eval_encoder16_kan_b44_h16_s0_beam5.log |
| encoder16_kan_b48_h16_s0 | leaderboards/v7/B_KAN_RESCUE/encoder16_kan_b48_h16_s0.log | leaderboards/v7/B_KAN_RESCUE/eval_encoder16_kan_b48_h16_s0_beam1.log | leaderboards/v7/B_KAN_RESCUE/eval_encoder16_kan_b48_h16_s0_beam5.log |
| encoder16r6_kan_b40_h16_s0 | leaderboards/v7/B_KAN_RESCUE/encoder16r6_kan_b40_h16_s0.log | leaderboards/v7/B_KAN_RESCUE/eval_encoder16r6_kan_b40_h16_s0_beam1.log | leaderboards/v7/B_KAN_RESCUE/eval_encoder16r6_kan_b40_h16_s0_beam5.log |
| encoder16r6_kan_b40_h16_s1 | leaderboards/v7/B_KAN_RESCUE/encoder16r6_kan_b40_h16_s1.log | leaderboards/v7/B_KAN_RESCUE/eval_encoder16r6_kan_b40_h16_s1_beam1.log | leaderboards/v7/B_KAN_RESCUE/eval_encoder16r6_kan_b40_h16_s1_beam5.log |
| encoder16r6_kan_b40_h16_s2 | leaderboards/v7/B_KAN_RESCUE/encoder16r6_kan_b40_h16_s2.log | leaderboards/v7/B_KAN_RESCUE/eval_encoder16r6_kan_b40_h16_s2_beam1.log | leaderboards/v7/B_KAN_RESCUE/eval_encoder16r6_kan_b40_h16_s2_beam5.log |
| encoder16r8_kan_b40_h16_s0 | leaderboards/v7/B_KAN_RESCUE/encoder16r8_kan_b40_h16_s0.log | leaderboards/v7/B_KAN_RESCUE/eval_encoder16r8_kan_b40_h16_s0_beam1.log | leaderboards/v7/B_KAN_RESCUE/eval_encoder16r8_kan_b40_h16_s0_beam5.log |
| encoder_deep_kan_b40_h16_d2_s0 | leaderboards/v7/B_KAN_RESCUE/encoder_deep_kan_b40_h16_d2_s0.log | leaderboards/v7/B_KAN_RESCUE/eval_encoder_deep_kan_b40_h16_d2_s0_beam1.log | leaderboards/v7/B_KAN_RESCUE/eval_encoder_deep_kan_b40_h16_d2_s0_beam5.log |
| encoder_kan_b16_h16_s0 | leaderboards/v7/B_KAN_RESCUE/encoder_kan_b16_h16_s0.log | leaderboards/v7/B_KAN_RESCUE/eval_encoder_kan_b16_h16_s0_beam1.log | leaderboards/v7/B_KAN_RESCUE/eval_encoder_kan_b16_h16_s0_beam5.log |
| encoder_kan_b16_h32_s0 | leaderboards/v7/B_KAN_RESCUE/encoder_kan_b16_h32_s0.log | leaderboards/v7/B_KAN_RESCUE/eval_encoder_kan_b16_h32_s0_beam1.log | leaderboards/v7/B_KAN_RESCUE/eval_encoder_kan_b16_h32_s0_beam5.log |
| encoder_kan_b20_h24_s0 | leaderboards/v7/B_KAN_RESCUE/encoder_kan_b20_h24_s0.log | leaderboards/v7/B_KAN_RESCUE/eval_encoder_kan_b20_h24_s0_beam1.log | leaderboards/v7/B_KAN_RESCUE/eval_encoder_kan_b20_h24_s0_beam5.log |
| encoder_kan_b24_h12_s0 | leaderboards/v7/B_KAN_RESCUE/encoder_kan_b24_h12_s0.log | leaderboards/v7/B_KAN_RESCUE/eval_encoder_kan_b24_h12_s0_beam1.log | leaderboards/v7/B_KAN_RESCUE/eval_encoder_kan_b24_h12_s0_beam5.log |
| encoder_kan_b24_h16_s0 | leaderboards/v7/B_KAN_RESCUE/encoder_kan_b24_h16_s0.log | leaderboards/v7/B_KAN_RESCUE/eval_encoder_kan_b24_h16_s0_beam1.log | leaderboards/v7/B_KAN_RESCUE/eval_encoder_kan_b24_h16_s0_beam5.log |
| encoder_kan_b24_h24_s0 | leaderboards/v7/B_KAN_RESCUE/encoder_kan_b24_h24_s0.log | leaderboards/v7/B_KAN_RESCUE/eval_encoder_kan_b24_h24_s0_beam1.log | leaderboards/v7/B_KAN_RESCUE/eval_encoder_kan_b24_h24_s0_beam5.log |
| encoder_kan_b24_h32_s0 | leaderboards/v7/B_KAN_RESCUE/encoder_kan_b24_h32_s0.log | leaderboards/v7/B_KAN_RESCUE/eval_encoder_kan_b24_h32_s0_beam1.log | leaderboards/v7/B_KAN_RESCUE/eval_encoder_kan_b24_h32_s0_beam5.log |
| encoder_kan_b28_h16_s0 | leaderboards/v7/B_KAN_RESCUE/encoder_kan_b28_h16_s0.log | leaderboards/v7/B_KAN_RESCUE/eval_encoder_kan_b28_h16_s0_beam1.log | leaderboards/v7/B_KAN_RESCUE/eval_encoder_kan_b28_h16_s0_beam5.log |
| encoder_kan_b32_h32_s0 | leaderboards/v7/B_KAN_RESCUE/encoder_kan_b32_h32_s0.log | leaderboards/v7/B_KAN_RESCUE/eval_encoder_kan_b32_h32_s0_beam1.log | leaderboards/v7/B_KAN_RESCUE/eval_encoder_kan_b32_h32_s0_beam5.log |
| kan_ae_h16_s0 | leaderboards/v7/B_KAN_RESCUE/kan_ae_h16_s0.log | leaderboards/v7/B_KAN_RESCUE/eval_kan_ae_h16_s0_beam1.log | leaderboards/v7/B_KAN_RESCUE/eval_kan_ae_h16_s0_beam5.log |
| kan_h16_s0 | leaderboards/v7/logs/train_v7_kan_h16_s0.log | leaderboards/v7/evals/eval_v7_kan_h16_s0_beam1.log | leaderboards/v7/evals/eval_v7_kan_h16_s0_beam5.log |
| kan_h2_s0 | leaderboards/v7/logs/train_v7_kan_h2_s0.log | leaderboards/v7/evals/eval_v7_kan_h2_s0_beam1.log | leaderboards/v7/evals/eval_v7_kan_h2_s0_beam5.log |
| kan_h32_s0 | leaderboards/v7/logs/train_v7_kan_h32_s0.log | leaderboards/v7/evals/eval_v7_kan_h32_s0_beam1.log | leaderboards/v7/evals/eval_v7_kan_h32_s0_beam5.log |
| kan_h4_s0 | leaderboards/v7/logs/train_v7_kan_h4_s0.log | leaderboards/v7/evals/eval_v7_kan_h4_s0_beam1.log | leaderboards/v7/evals/eval_v7_kan_h4_s0_beam5.log |
| kan_h8_s0 | leaderboards/v7/logs/train_v7_kan_h8_s0.log | leaderboards/v7/evals/eval_v7_kan_h8_s0_beam1.log | leaderboards/v7/evals/eval_v7_kan_h8_s0_beam5.log |
| micro_mlp_h16_s0 | leaderboards/v7/logs/train_v7_micro_mlp_h16_s0.log | leaderboards/v7/evals/eval_v7_micro_mlp_h16_s0_beam1.log | leaderboards/v7/evals/eval_v7_micro_mlp_h16_s0_beam5.log |
| micro_mlp_h32_s0 | leaderboards/v7/logs/train_v7_micro_mlp_h32_s0.log | leaderboards/v7/evals/eval_v7_micro_mlp_h32_s0_beam1.log | leaderboards/v7/evals/eval_v7_micro_mlp_h32_s0_beam5.log |
| micro_mlp_h4_s0 | leaderboards/v7/logs/train_v7_micro_mlp_h4_s0.log | leaderboards/v7/evals/eval_v7_micro_mlp_h4_s0_beam1.log | leaderboards/v7/evals/eval_v7_micro_mlp_h4_s0_beam5.log |
| micro_mlp_h8_s0 | leaderboards/v7/logs/train_v7_micro_mlp_h8_s0.log | leaderboards/v7/evals/eval_v7_micro_mlp_h8_s0_beam1.log | leaderboards/v7/evals/eval_v7_micro_mlp_h8_s0_beam5.log |
| mlp128_s0 | leaderboards/v7/logs/train_v7_mlp128_s0.log | leaderboards/v7/evals/eval_v7_mlp128_s0_beam1.log | leaderboards/v7/evals/eval_v7_mlp128_s0_beam5.log |
