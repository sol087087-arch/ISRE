# ISRE Eval Pipeline Specification

## Overview

After training, we need to answer three questions:
1. Does the model beat greedy heuristic? (go/no-go for the project)
2. Where does the model fail? (collapse taxonomy)
3. What did the model learn? (KAN interpretability, if applicable)

This document specifies the eval pipeline that answers all three.

---

## 1. Eval Modes

### 1.1 Offline Eval (primary, day 5-7)

Input: trained checkpoint + held-out test trajectories.
Process: for each trajectory step, run encoder + policy, record scores.
Output: `eval_results.json` — feeds into trajectory viewer eval mode.

This is the main eval. Fast, reproducible, no symbolic engine in the loop.

### 1.2 Online Eval (day 7+)

Input: trained checkpoint + canonical polynomials (no trajectories).
Process: start from scrambled expression, let model choose actions step-by-step until canonical form or max_steps.
Output: full reasoning trace with model decisions.

This is the real test — model drives the reasoning loop end-to-end.
Requires symbolic engine in the loop (generate candidates → model picks → apply → repeat).

### 1.3 Baseline Comparison (day 5)

Run the same test set with:
- Random policy (uniform over valid candidates)
- Greedy heuristic (pick action that maximally reduces node_count + operator_count)
- BFS optimal (exhaustive search, depth ≤ 10) — only for short trajectories

Generate same `eval_results.json` format for each baseline.
Compare all three + model in one summary.

---

## 2. Offline Eval Detail

### 2.1 Input Format

```
test_trajectories/
  traj_0050000.json
  traj_0050001.json
  ...
```

Same format as training data. Held out during training (val split or separate generation).

### 2.2 Per-Step Output

For each step in each trajectory:

```json
{
  "scores": [0.82, 0.11, 0.04, ...],
  "predicted_action": "EXPAND",
  "predicted_node_id": 3,
  "gold_action": "MERGE_POWER",
  "gold_node_id": 7,
  "gold_rank": 2,
  "top1_correct": false,
  "entropy_raw": 1.23,
  "entropy_norm": 0.71,
  "num_candidates": 6
}
```

`scores` order matches candidate order from symbolic engine on that state.
`gold_rank` = 1-indexed rank of gold action in model's sorted scores.
`entropy_norm` = H(policy) / log(num_candidates).

### 2.3 Per-Trajectory Output

```json
{
  "trajectory_id": "traj_0050000",
  "difficulty": 4,
  "num_steps_evaluated": 4,
  "top1_accuracy": 0.75,
  "avg_gold_rank": 1.5,
  "avg_entropy_norm": 0.34,
  "collapse_type": "NONE",
  "progress_trace": [12, 9, 8, 6]
}
```

`collapse_type` — one of: NONE, STALL, OSCILLATION, LATE_COLLAPSE, ENTROPY_COLLAPSE, INSTANT_FAILURE.
`progress_trace` — complexity at each step (for progress chart in viewer).

### 2.4 Full Output: eval_results.json

```json
{
  "model_name": "mlp_baseline_v1",
  "checkpoint": "checkpoints/best.pt",
  "eval_date": "2026-03-14",
  "num_trajectories": 500,
  "num_steps": 1842,
  "summary": {
    "top1_accuracy": 0.723,
    "avg_gold_rank": 1.41,
    "avg_entropy_norm": 0.38,
    "collapse_distribution": {
      "NONE": 412,
      "STALL": 35,
      "OSCILLATION": 22,
      "LATE_COLLAPSE": 8,
      "ENTROPY_COLLAPSE": 18,
      "INSTANT_FAILURE": 5
    },
    "action_accuracy": {
      "MERGE_POWER": 0.89,
      "REMOVE_ONE": 0.95,
      "REMOVE_ZERO": 0.97,
      "COMBINE_COEFF": 0.62,
      "COLLECT_TERMS": 0.58,
      "FOLD_CONST": 0.91,
      "FLATTEN_ADD": 0.78,
      "FLATTEN_MUL": 0.80,
      "EXPAND": 0.45,
      "SORT_COMMUTATIVE": 0.72
    }
  },
  "per_trajectory": {
    "traj_0050000": { ... },
    "traj_0050001": { ... }
  }
}
```

`action_accuracy` — per-action breakdown. If EXPAND accuracy is 0.45 and REMOVE_ZERO is 0.97, we know the model struggles with strategy, not pattern matching. This is the most diagnostic single table in the eval.

---

## 3. Online Eval Detail

### 3.1 Process

```
for each test expression:
    state = scrambled expression
    trace = []
    seen_states = set()
    
    for step in range(max_steps):
        candidates = engine.get_candidates(state)
        if not candidates:
            break  # stuck
            
        scores = model.score(state, candidates)
        action = argmax(scores)  # greedy at eval time
        
        trace.append({state, scores, action, complexity})
        
        new_state = engine.apply(state, action)
        
        if new_state in seen_states:
            mark OSCILLATION, break
        if new_state == canonical:
            mark SUCCESS, break
            
        seen_states.add(hash(new_state))
        state = new_state
    
    if not success:
        classify collapse type from trace
```

### 3.2 Collapse Classification Logic

```
Given progress_trace = [c0, c1, c2, ...]:

NONE:           reached canonical form
INSTANT_FAILURE: c1 >= c0 and c2 >= c1 (never improves)
STALL:          exists i where c[i] == c[i+1] == c[i+2] (plateau ≥ 3 steps)
OSCILLATION:    state cycle detected via seen_states hash set
LATE_COLLAPSE:  min(progress_trace) < c0, but final complexity > min (got worse after improving)
ENTROPY_COLLAPSE: entropy_norm increases for ≥ 3 consecutive steps in second half of trace
```

Priority: OSCILLATION > LATE_COLLAPSE > ENTROPY_COLLAPSE > STALL > INSTANT_FAILURE.
If multiple apply, pick highest priority.

### 3.3 Output

Same format as offline eval, plus:

```json
{
  "online_trace": [
    {
      "step": 0,
      "state_expr": "((3+1) * x^1 * x^1 ...)",
      "state_ast": { ... },
      "action_taken": "MERGE_POWER",
      "action_node_id": 7,
      "complexity": 24,
      "entropy_norm": 0.31
    },
    ...
  ],
  "reached_canonical": true,
  "steps_used": 5,
  "collapse_type": "NONE"
}
```

This feeds directly into trajectory viewer — same format as training trajectories, but with model decisions instead of gold actions.

---

## 4. Greedy Heuristic Baseline

```
for each step:
    candidates = engine.get_candidates(state)
    for each candidate:
        simulated = engine.apply(state, candidate)
        score = state.complexity() - simulated.complexity()  # complexity reduction
    pick candidate with max score (biggest complexity drop)
    ties broken by: FOLD_CONST > REMOVE_ZERO > REMOVE_ONE > others (cheapest first)
```

This is the bar to beat. If the neural model doesn't beat greedy on online eval success_rate — the neural net adds no value.

Expected: greedy solves most simple trajectories (depth 1-3) but fails on:
- Non-monotonic paths (EXPAND then COLLECT)
- Tie-breaking where wrong choice leads to dead end
- Long trajectories where myopic strategy isn't optimal

---

## 5. Summary Report

After running all evals, generate a single comparison table:

```
                  | Random | Greedy | MLP    | KAN
------------------+--------+--------+--------+--------
success_rate      | 0.12   | 0.78   | 0.85   | 0.83
avg_steps         | -      | 4.2    | 3.8    | 3.9
top1_accuracy     | 0.10   | 0.61   | 0.72   | 0.70
avg_gold_rank     | 3.2    | 1.8    | 1.4    | 1.5
collapse_rate     | 0.88   | 0.22   | 0.15   | 0.17
EXPAND_accuracy   | 0.10   | 0.20   | 0.45   | 0.42
```

Key comparisons:
- MLP vs Greedy: does the neural net add value?
- MLP vs KAN: accuracy tradeoff for interpretability
- Per-action accuracy: where does each approach struggle?
- Collapse distribution: do they fail differently?

This table is the core result of the project.

---

## 6. Integration with Trajectory Viewer

Trajectory viewer already supports eval mode:

```bash
python trajectory_viewer.py \
    --data test_trajectories/ \
    --predictions eval_results.json \
    --output viewer_eval/
```

Viewer reads `per_trajectory.{traj_id}.steps.{idx}.scores` and renders:
- Score column in candidate table
- Rank column
- Gold row highlighting (correct answer)
- Top row highlighting (model's choice)
- Gold rank in metrics panel
- Entropy in metrics panel

For online eval traces, generate trajectory JSONs in standard format and feed to viewer.

---

## 7. File Structure

```
eval/
  eval_offline.py     # run checkpoint on test trajectories, output eval_results.json
  eval_online.py      # model drives reasoning loop end-to-end
  baselines.py        # random + greedy + BFS baselines
  classify_collapse.py # collapse taxonomy classification
  summary.py          # generate comparison table
```

---

## 8. Timeline

| Day | Eval task |
|-----|-----------|
| 5   | Greedy + random baselines on test set |
| 6   | eval_offline.py — first model results |
| 7   | Compare: model vs greedy. Go/no-go decision |
| 8   | KAN policy head training |
| 9   | eval_online.py — model drives reasoning. Collapse analysis |
| 10  | Summary table. Viewer eval mode. KAN curve extraction if applicable |
