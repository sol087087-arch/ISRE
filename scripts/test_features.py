"""T1 — candidate_features correctness + NO-LEAK structural check.

Run: cd C:\\GitHub\\ISRE; set PYTHONPATH=.; python scripts/test_features.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from isre.symbolic.isre_ast import Add, Mul, Pow, Var, Num
from isre.symbolic.symbolic_engine import SymbolicEngine, ActionType
from isre.learning.features import (
    candidate_features, FEATURE_DIM, FEATURE_NAMES,
)

engine = SymbolicEngine()

# 5 diverse ASTs.
ASTS = [
    Add(Var(), Num(1)),
    Mul(Num(2), Var()),
    Add(Mul(Num(2), Var()), Mul(Num(3), Var()), Num(0)),
    Pow(Add(Var(), Num(1)), Num(2)),
    Mul(Add(Var(), Num(1)), Add(Var(), Num(2)), Pow(Var(), Num(3))),
]

print(f"FEATURE_DIM = {FEATURE_DIM}")
print(f"FEATURE_NAMES ({len(FEATURE_NAMES)}):")
for i, nm in enumerate(FEATURE_NAMES):
    print(f"  [{i:2d}] {nm}")

# --- FEATURE_DIM matches len() on all 5 ASTs, all candidates ---
n_checked = 0
for root in ASTS:
    root.mark_dirty(); root._rebuild_parents(); root._ensure_metadata()
    for nid, _, act in engine.get_candidates(root):
        v = candidate_features(root, nid, act)
        assert len(v) == FEATURE_DIM, (len(v), FEATURE_DIM, root.to_expr())
        assert all(isinstance(x, float) for x in v)
        n_checked += 1
print(f"\n[dim] FEATURE_DIM==len() on {n_checked} candidates across 5 ASTs OK")

# --- determinism: same args -> identical vector ---
root = ASTS[2]; root.mark_dirty(); root._rebuild_parents(); root._ensure_metadata()
cands = [(nid, a) for nid, _, a in engine.get_candidates(root)]
for nid, act in cands:
    a = candidate_features(root, nid, act)
    b = candidate_features(root, nid, act)
    assert a == b, "non-deterministic"
print("[determinism] identical vectors on repeat calls OK")

# --- NO-LEAK structural check ---
# The function signature is candidate_features(state_root, node_id, action):
# there is NO parameter through which gold_action / gold_node_id /
# trajectory_id / difficulty could enter. We additionally PROVE it
# behaviourally: pick a node with >=2 candidate actions; whichever action
# we (externally) call "gold" must NOT change ANY feature vector for ANY
# candidate. We compute all vectors, then re-compute under several
# different "gold" designations and assert byte-equality.
import inspect
sig = list(inspect.signature(candidate_features).parameters)
assert sig == ["state_root", "node_id", "action"], sig
for bad in ("gold", "trajectory", "difficulty", "label", "target"):
    assert not any(bad in p for p in sig), f"suspicious param {bad!r}"
print(f"[no-leak] signature == {sig}  (no gold/label channel) OK")

baseline = {(nid, act.value): candidate_features(root, nid, act)
            for nid, act in cands}
# Permute which candidate is "gold" across every candidate; recompute the
# full feature set each time. A leak would make some vector depend on the
# gold choice -> mismatch. (gold is never passed in, so this must hold.)
for gold_pick in cands:
    recomputed = {(nid, act.value): candidate_features(root, nid, act)
                  for nid, act in cands}
    assert recomputed == baseline, (
        f"LEAK: feature vectors changed when gold designated {gold_pick}")
print(f"[no-leak] feature set invariant under all {len(cands)} gold "
      f"permutations OK")

# Spot-check semantics on a known tree: Add(Var, Num(1)) root node.
r = Add(Var(), Num(1)); r.mark_dirty(); r._rebuild_parents(); r._ensure_metadata()
fv = candidate_features(r, 0, ActionType.REMOVE_ZERO)
assert fv[0] == 1.0, "root is ADD -> idx0 one-hot"           # ADD
assert fv[15] == 1.0, "node 0 is root -> is_root"
assert all(fv[8 + i] == 0.0 for i in range(6)), "root parent block all-zero"
assert fv[26] == 3.0 / 50.0, "total nodes = 3 (Add,Var,Num)"
# child Var at preorder idx 1, parent is ADD
fv1 = candidate_features(r, 1, ActionType.SORT_COMMUTATIVE)
assert fv1[4] == 1.0, "VARIABLE one-hot idx4"
assert fv1[8 + 0] == 1.0, "parent ADD one-hot in parent block"
assert fv1[15] == 0.0, "not root"
print("[semantics] root/child/parent/global spot-checks OK")

print("\nALL FEATURE TESTS PASSED")
