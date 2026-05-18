"""
Hand-crafted per-candidate features for the KAN policy arm.

The KAN-vs-MLP experiment is matched-PROTOCOL, not matched-input:
  - MLP arm: GRU encoder embeddings -> MLP (existing, frozen design)
  - KAN arm: THESE hand-crafted features -> KAN

`candidate_features(state_root, node_id, action)` is a PURE function of
(state_root, node_id, action). It returns a fixed-length list[float] of
length FEATURE_DIM.

================= CRITICAL NO-LEAK INVARIANT =================
This function MUST NOT see gold_action, gold_node_id, trajectory_id,
difficulty, or ANY "is this candidate the gold one" signal. Its entire
input is the structural state, the index of the candidate node, and the
ActionType being scored. There is deliberately NO parameter through which
a gold/label signal could enter. The KAN scores every candidate with the
same featuriser; the supervised CE target is applied OUTSIDE this module
(in KANPolicy.compute_loss) by position, exactly as PolicyNetwork does.
A test (scripts/test_features.py) permutes which candidate is "gold" and
asserts every feature vector is byte-identical, proving no leak.
=============================================================

Node indexing convention:
  The symbolic engine indexes nodes via enumerate(root.iter_preorder())
  (see SymbolicEngine.get_candidates). We resolve node_id the SAME way:
      list(state_root.iter_preorder())[node_id]
  so feature node identity matches the candidate node the engine produced.

Action ordering convention:
  ActionType one-hot uses sorted(ActionType, key=lambda a: a.value),
  the SAME stable-alphabetical ordering as PolicyNetwork._action_to_idx.
  Reordering the enum source does not remap features.
"""

from typing import List

from isre.symbolic.isre_ast import ASTNode, NodeType
from isre.symbolic.symbolic_engine import ActionType


# Stable NodeType order (matches ASTEncoder._type_index ordering intent:
# ADD, MUL, POW, NUMBER, VARIABLE, CONST). Used for both target-node and
# parent-node one-hot blocks.
_NODE_TYPE_ORDER: List[NodeType] = [
    NodeType.ADD,
    NodeType.MUL,
    NodeType.POW,
    NodeType.NUMBER,
    NodeType.VARIABLE,
    NodeType.CONST,
]
_NODE_TYPE_IDX = {nt: i for i, nt in enumerate(_NODE_TYPE_ORDER)}
_N_NODE_TYPES = len(_NODE_TYPE_ORDER)  # 6

# Stable ActionType order — SAME convention as policy.py _action_to_idx.
_ACTION_ORDER: List[ActionType] = sorted(ActionType, key=lambda a: a.value)
_ACTION_IDX = {a: i for i, a in enumerate(_ACTION_ORDER)}
_N_ACTIONS = len(_ACTION_ORDER)  # 10

# Normalisation constants (documented; chosen to keep features ~O(1)).
_DEPTH_NORM = 12.0
_SUBTREE_NORM = 30.0
_NUM_CHILDREN_NORM = 5.0
_TOTAL_NODES_NORM = 50.0


# ---------------------------------------------------------------------------
# FEATURE LAYOUT  (FEATURE_DIM = 6 + 1 + 1 + 6 + 1 + 1 + 10 + 1 = 27)
#
#  idx  0.. 5 : target node node_type one-hot [ADD,MUL,POW,NUMBER,VARIABLE,CONST]
#  idx      6 : target node depth / 12.0
#  idx      7 : target node subtree_size / 30.0
#  idx  8..13 : parent node_type one-hot (all zeros if target is root)
#  idx     14 : target num_children / 5.0
#  idx     15 : is_root (1.0 / 0.0)
#  idx 16..25 : action one-hot, sorted(ActionType, key=value):
#               [COLLECT_TERMS, COMBINE_COEFF, EXPAND, FLATTEN_ADD,
#                FLATTEN_MUL, FOLD_CONST, MERGE_POWER, REMOVE_ONE,
#                REMOVE_ZERO, SORT_COMMUTATIVE]
#  idx     26 : total node count of state_root / 50.0
# ---------------------------------------------------------------------------
FEATURE_DIM = _N_NODE_TYPES + 1 + 1 + _N_NODE_TYPES + 1 + 1 + _N_ACTIONS + 1
assert FEATURE_DIM == 27, FEATURE_DIM

# Human-readable per-index legend (used by test + phi sweep plots).
FEATURE_NAMES: List[str] = (
    [f"tnode_type[{nt.name}]" for nt in _NODE_TYPE_ORDER]
    + ["tnode_depth_n", "tnode_subtree_n"]
    + [f"parent_type[{nt.name}]" for nt in _NODE_TYPE_ORDER]
    + ["tnode_num_children_n", "is_root"]
    + [f"action[{a.value}]" for a in _ACTION_ORDER]
    + ["total_nodes_n"]
)
assert len(FEATURE_NAMES) == FEATURE_DIM


def candidate_features(
    state_root: ASTNode,
    node_id: int,
    action: ActionType,
) -> List[float]:
    """Fixed-length feature vector for one (state, node_id, action) candidate.

    PURE function. No gold/label signal is reachable from this signature
    (see module docstring NO-LEAK INVARIANT). Length == FEATURE_DIM.
    """
    # Preorder list — same indexing the engine uses for node_id.
    preorder = list(state_root.iter_preorder())
    total_nodes = len(preorder)

    # node_id is engine-produced and always in range on clean data; clamp
    # defensively so a malformed id degrades rather than crashes (it never
    # injects a label — still a pure function of the inputs).
    if not (0 <= node_id < total_nodes):
        node_id = max(0, min(total_nodes - 1, node_id))
    target = preorder[node_id]

    feats: List[float] = [0.0] * FEATURE_DIM

    # --- target node_type one-hot (idx 0..5) ---
    ti = _NODE_TYPE_IDX.get(target.node_type)
    if ti is not None:
        feats[ti] = 1.0

    # --- normalized depth / subtree_size (idx 6, 7) ---
    # .depth / .subtree_size lazily recompute from the (dirty) root; callers
    # in train/eval mark_dirty + rebuild on the root before featurising.
    feats[6] = float(target.depth) / _DEPTH_NORM
    feats[7] = float(target.subtree_size) / _SUBTREE_NORM

    # --- parent node_type one-hot (idx 8..13), zeros if root ---
    parent = target.parent
    if parent is not None:
        pi = _NODE_TYPE_IDX.get(parent.node_type)
        if pi is not None:
            feats[8 + pi] = 1.0

    # --- num_children, is_root (idx 14, 15) ---
    feats[14] = float(len(target.children)) / _NUM_CHILDREN_NORM
    feats[15] = 1.0 if parent is None else 0.0

    # --- action one-hot (idx 16..25), stable alphabetical order ---
    ai = _ACTION_IDX.get(action)
    if ai is not None:
        feats[16 + ai] = 1.0

    # --- global: total node count (idx 26) ---
    feats[26] = float(total_nodes) / _TOTAL_NODES_NORM

    return feats
