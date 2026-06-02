# v7 Evaluation Notes

Mode A is the leaderboard metric for v7:
- success_rate
- overhead vs BFS
- BFS-optimal rate
- catastrophic_rate

Mode B's recorded-teacher comparison is intentionally disabled for v7 by using a non-existent recorded-data directory:
`leaderboards/v7/no_recorded_teacher`.

Reason: v7 has BFS-optimal trajectories but no paired recorded-human/heuristic arm. Reusing v6 recorded trajectories would mix datasets and could create accidental state-expression matches. We keep Mode B output only as gold-path step agreement and treat `div_teacher_undefined` as expected.
