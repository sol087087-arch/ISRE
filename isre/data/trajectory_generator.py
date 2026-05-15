"""
Backward trajectory generation pipeline v1.
Generates training data: canonical polynomial → apply inverse transforms → record trajectory.

Usage:
    python trajectory_gen.py --count 1000 --output trajectories/
"""

from __future__ import annotations

import json
import random
import argparse
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from tqdm import tqdm

from isre.symbolic.isre_ast import ASTNode, NodeType, Num, Var, Add, Mul, Pow
from isre.symbolic.symbolic_engine import SymbolicEngine, ActionType


# ====================== DATA STRUCTURES ======================

@dataclass
class TrajectoryStep:
    """One step of the forward (solving) trajectory."""
    state: dict                     # AST as dict (the messy expression)
    state_expr: str                 # human-readable expression
    candidate_actions: List[List]    # [[node_id, action_name], ...] — full candidates
    gold_action: str                # correct next forward action
    gold_node_id: int               # which node to apply it to
    complexity: int                 # node_count + operator_count


@dataclass
class Trajectory:
    trajectory_id: str
    canonical_expr: str
    canonical_ast: dict
    original_expr: str              # the scrambled starting expression
    original_ast: dict
    steps: List[TrajectoryStep]
    difficulty: int                 # len(steps)
    inverse_sequence: List[str]     # which inverse transforms were applied


# ====================== CANONICAL POLYNOMIAL GENERATOR ======================

def sample_canonical_polynomial(
    rng: random.Random,
    max_degree: int = 4,
    coeff_range: Tuple[int, int] = (-9, 9),
    min_terms: int = 2,
    max_terms: int = 4,
    constant_prob: float = 0.5,
) -> ASTNode:
    """Generate a canonical polynomial in collected form, sorted by degree descending.
    
    Guarantees at least one term with degree ≥ 1 (so inverse transforms have targets).
    Constant term included with probability constant_prob.
    Examples: 3x^2 + 2x + 1, -x^3 + 5x, 4x^4 - 2x^2 + 7
    """
    # First pick the polynomial's leading degree uniformly in [1, max_degree].
    # Then pick non-constant degrees as a subset of [1, leading_degree], always
    # including leading_degree itself. This avoids the previous bias toward
    # max_degree=4 (which appeared in 62% of polys).
    leading_degree = rng.randint(1, max_degree)
    available_lower = list(range(1, leading_degree))  # may be empty if leading=1
    extra_count = rng.randint(0, min(max_terms - 1, len(available_lower)))
    extra_degrees = rng.sample(available_lower, extra_count) if extra_count else []
    non_const_degrees = sorted([leading_degree, *extra_degrees], reverse=True)
    # Honor min_terms if specified (count includes optional constant added below)
    while len(non_const_degrees) + 1 < min_terms and available_lower:
        # need more terms — pull additional unique lower degrees if available
        remaining = [d for d in available_lower if d not in non_const_degrees]
        if not remaining:
            break
        non_const_degrees.append(rng.choice(remaining))
        non_const_degrees = sorted(set(non_const_degrees), reverse=True)

    # Optionally add constant term
    include_constant = rng.random() < constant_prob
    degrees = non_const_degrees + ([0] if include_constant else [])

    terms = []
    for deg in degrees:
        coeff = 0
        while coeff == 0:  # no zero coefficients
            coeff = rng.randint(coeff_range[0], coeff_range[1])
        terms.append(_make_term(coeff, deg))

    if len(terms) == 1:
        return terms[0]
    return Add(*terms)


def _make_term(coeff: int, degree: int) -> ASTNode:
    """Build a single polynomial term: coeff * x^degree."""
    if degree == 0:
        return Num(coeff)
    if degree == 1:
        var_part = Var()
    else:
        var_part = Pow(Var(), Num(degree))

    if coeff == 1:
        return var_part
    if coeff == -1:
        return Mul(Num(-1), var_part)
    return Mul(Num(coeff), var_part)


# ====================== INVERSE TRANSFORMS ======================
# Each takes (rng, root_clone, node, node_id) → Optional[Tuple[ASTNode, int]]
# Returns (new_root, target_node_id_for_forward_action) or None if not applicable.
# The node is already inside root_clone (it's a reference into the cloned tree).

InverseResult = Optional[Tuple[ASTNode, int]]  # (new_root, forward_node_id)


def inverse_uncollect_terms(rng: random.Random, root: ASTNode, node: ASTNode, node_id: int) -> InverseResult:
    """Mul(k, x) → Add(x, x, ..., x)  [k copies, k ≤ 3]
    Forward: COLLECT_TERMS"""
    if node.node_type != NodeType.MUL:
        return None
    if len(node.children) != 2:
        return None
    if node.children[0].node_type != NodeType.NUMBER:
        return None
    try:
        k = int(float(node.children[0].value))
    except (ValueError, TypeError):
        return None
    if k < 2 or k > 3:  # constraint from spec
        return None
    var_part = node.children[1]
    if not _is_bare_term(var_part):
        return None

    # Build Add(x, x, ..., x)
    copies = [var_part.clone() for _ in range(k)]
    replacement = Add(*copies)

    # Replace node in tree
    new_root = _replace_node(root, node, replacement)
    if new_root is None:
        return None
    new_root.mark_dirty()

    # Find node_id of the new Add node for forward action
    fwd_id = _find_node_id(new_root, replacement)
    return (new_root, fwd_id) if fwd_id is not None else None


def inverse_split_coefficient(rng: random.Random, root: ASTNode, node: ASTNode, node_id: int) -> InverseResult:
    """Mul(k, x) → Add(Mul(a, x), Mul(b, x)) where a+b=k, always binary split.
    Forward: COMBINE_COEFF"""
    if node.node_type != NodeType.MUL:
        return None
    if len(node.children) != 2:
        return None
    if node.children[0].node_type != NodeType.NUMBER:
        return None
    try:
        k = int(float(node.children[0].value))
    except (ValueError, TypeError):
        return None
    if abs(k) < 2 or abs(k) > 6:  # constraint from spec
        return None
    var_part = node.children[1]
    if not _is_bare_term(var_part):
        return None

    # Binary split: pick a in [1, k-1] (positive k) or [-k+1, -1] (negative k)
    if k > 0:
        a = rng.randint(1, k - 1)
    else:
        a = rng.randint(k + 1, -1)
    b = k - a

    replacement = Add(
        Mul(Num(a), var_part.clone()),
        Mul(Num(b), var_part.clone()),
    )

    new_root = _replace_node(root, node, replacement)
    if new_root is None:
        return None
    new_root.mark_dirty()

    fwd_id = _find_node_id(new_root, replacement)
    return (new_root, fwd_id) if fwd_id is not None else None


def inverse_unflatten_add(rng: random.Random, root: ASTNode, node: ASTNode, node_id: int) -> InverseResult:
    """Add(a, b, c) → Add(Add(a, b), c).  Only if arity ≥ 3.
    Forward: FLATTEN_ADD"""
    if node.node_type != NodeType.ADD:
        return None
    if len(node.children) < 3:
        return None

    # Pick split point
    split = rng.randint(2, len(node.children) - 1)
    left_group = [c.clone() for c in node.children[:split]]
    right_group = [c.clone() for c in node.children[split:]]

    inner = Add(*left_group)
    replacement = Add(inner, *right_group)

    new_root = _replace_node(root, node, replacement)
    if new_root is None:
        return None
    new_root.mark_dirty()

    fwd_id = _find_node_id(new_root, replacement)
    return (new_root, fwd_id) if fwd_id is not None else None


def inverse_unflatten_mul(rng: random.Random, root: ASTNode, node: ASTNode, node_id: int) -> InverseResult:
    """Mul(a, b, c) → Mul(Mul(a, b), c).  Only if arity ≥ 3.
    Forward: FLATTEN_MUL"""
    if node.node_type != NodeType.MUL:
        return None
    if len(node.children) < 3:
        return None

    split = rng.randint(2, len(node.children) - 1)
    left_group = [c.clone() for c in node.children[:split]]
    right_group = [c.clone() for c in node.children[split:]]

    inner = Mul(*left_group)
    replacement = Mul(inner, *right_group)

    new_root = _replace_node(root, node, replacement)
    if new_root is None:
        return None
    new_root.mark_dirty()

    fwd_id = _find_node_id(new_root, replacement)
    return (new_root, fwd_id) if fwd_id is not None else None


def inverse_unfold_const(rng: random.Random, root: ASTNode, node: ASTNode, node_id: int) -> InverseResult:
    """5 → Add(2, 3). Only NUMBER, value ≥ 2 and ≤ 10.
    Forward: FOLD_CONST"""
    if node.node_type != NodeType.NUMBER:
        return None
    try:
        val = int(float(node.value))
    except (ValueError, TypeError):
        return None
    if val < 2 or val > 10:
        return None

    # Split into two positive parts (both ≥ 1)
    if val > 1:
        a = rng.randint(1, val - 1)
    else:
        return None
    b = val - a

    replacement = Add(Num(a), Num(b))

    new_root = _replace_node(root, node, replacement)
    if new_root is None:
        return None
    new_root.mark_dirty()

    fwd_id = _find_node_id(new_root, replacement)
    return (new_root, fwd_id) if fwd_id is not None else None


def inverse_split_power(rng: random.Random, root: ASTNode, node: ASTNode, node_id: int) -> InverseResult:
    """Pow(x, n) → Mul(Pow(x, a1), Pow(x, a2), ..., Pow(x, ak)) where Σai = n, n ≤ 4.
    Splits into arity 2..n parts (uniform over k, then random composition).
    Forward: MERGE_POWER (after FLATTEN_MUL if arity ≥ 3 got nested by UNFLATTEN_MUL)."""
    if node.node_type != NodeType.POW:
        return None
    if node.children[0].node_type != NodeType.VARIABLE:
        return None
    if node.children[1].node_type != NodeType.NUMBER:
        return None
    try:
        n = int(float(node.children[1].value))
    except (ValueError, TypeError):
        return None
    if n < 2 or n > 4:
        return None

    # Pick number of parts k ∈ [2, n]. Each part ≥ 1, sum = n.
    k = rng.randint(2, n)
    # Random composition: pick k-1 cut points in [1, n-1] without replacement, then take diffs.
    cuts = sorted(rng.sample(range(1, n), k - 1)) if k > 1 else []
    parts = []
    prev = 0
    for c in cuts:
        parts.append(c - prev)
        prev = c
    parts.append(n - prev)
    # Sanity: all parts ≥ 1, sum = n
    assert all(p >= 1 for p in parts) and sum(parts) == n

    base_var = node.children[0].value
    factors = [
        Pow(ASTNode(NodeType.VARIABLE, value=base_var), Num(p))
        for p in parts
    ]
    replacement = Mul(*factors)

    new_root = _replace_node(root, node, replacement)
    if new_root is None:
        return None
    new_root.mark_dirty()

    fwd_id = _find_node_id(new_root, replacement)
    return (new_root, fwd_id) if fwd_id is not None else None


def inverse_factor_pair(rng: random.Random, root: ASTNode, node: ASTNode, node_id: int) -> InverseResult:
    """Add(Mul(c1,p1), Mul(c2,p2), ...) → Mul(g, Add(Mul(c1/g,p1), Mul(c2/g,p2), ...))

    Pulls out a common positive numeric factor g (with 2 ≤ g ≤ 3 per spec §4.3
    so that forward expand→collect is a net complexity decrease).

    Forward action: EXPAND on the new outer Mul.

    Constraints:
      - All children of the Add must be either:
          * Num(c) with c > 0, or
          * Mul(Num(c), bare_monomial) with c > 0
      - The integer GCD of all child coefficients must be ≥ 2 and ≤ 3.
      - Bare terms (e.g. plain x) have implicit coeff 1, which kills the GCD;
        such Adds are skipped. This is intentional — keeps the case clean.
    """
    if node.node_type != NodeType.ADD:
        return None
    if len(node.children) < 2:
        return None

    # Extract (coeff, monomial_or_None) for every child. Bail on any non-conforming child.
    parts: List[Tuple[int, Optional[ASTNode]]] = []
    for child in node.children:
        if child.node_type == NodeType.NUMBER:
            try:
                c = int(float(child.value))
            except (ValueError, TypeError):
                return None
            if c <= 0:
                return None
            parts.append((c, None))
        elif (child.node_type == NodeType.MUL
                and len(child.children) == 2
                and child.children[0].node_type == NodeType.NUMBER
                and _is_bare_term(child.children[1])):
            try:
                c = int(float(child.children[0].value))
            except (ValueError, TypeError):
                return None
            if c <= 0:
                return None
            parts.append((c, child.children[1]))
        else:
            return None

    # Compute GCD of all coefficients
    from math import gcd
    from functools import reduce as _reduce
    g = _reduce(gcd, (c for c, _ in parts))
    if g < 2 or g > 3:
        return None

    # Build factored children
    new_add_children: List[ASTNode] = []
    for c, mono in parts:
        new_c = c // g
        if mono is None:
            # pure constant
            new_add_children.append(Num(new_c))
        else:
            if new_c == 1:
                new_add_children.append(mono.clone())
            else:
                new_add_children.append(Mul(Num(new_c), mono.clone()))

    inner_add = Add(*new_add_children)
    replacement = Mul(Num(g), inner_add)

    new_root = _replace_node(root, node, replacement)
    if new_root is None:
        return None
    new_root.mark_dirty()

    # Forward action (EXPAND) targets the new outer Mul
    fwd_id = _find_node_id(new_root, replacement)
    return (new_root, fwd_id) if fwd_id is not None else None


def inverse_factor_variable(rng: random.Random, root: ASTNode, node: ASTNode, node_id: int) -> InverseResult:
    """Add(c1·x^a1, c2·x^a2, ..., ck·x^ak) where min(ai) ≥ 1
    → Mul(x^min, Add(c1·x^(a1-min), c2·x^(a2-min), ...))

    Pulls out a common variable factor x^k where k = min variable degree across
    all children of an Add. Every child must be a monomial in x with degree ≥ 1.
    Pure-constant terms break the factoring → bail.

    Forward action: EXPAND on the new outer Mul.
    """
    if node.node_type != NodeType.ADD:
        return None
    if len(node.children) < 2:
        return None

    # Decompose every child into (coeff, var_degree); bail if anything doesn't fit.
    parts: List[Tuple[int, int]] = []  # (coeff, degree)
    for child in node.children:
        info = _monomial_decomp(child)
        if info is None:
            return None
        coeff, deg = info
        if deg < 1:
            return None  # pure constant kills the variable factor
        parts.append((coeff, deg))

    min_deg = min(d for _, d in parts)
    if min_deg < 1:
        return None
    # Don't bother factoring out x^4 — leaves trivial inner Add of constants only.
    # Require at least one child to keep degree ≥ 1 after factoring.
    if not any(d - min_deg >= 1 for _, d in parts):
        # All children would collapse to constants → still valid algebraically,
        # but the inner Add becomes Add(c1, c2, ...) which is just FOLD_CONST
        # territory and not what EXPAND would naturally re-derive cleanly.
        return None

    # Build factored inner Add
    base_var = "x"
    inner_terms: List[ASTNode] = []
    for coeff, deg in parts:
        new_deg = deg - min_deg
        if new_deg == 0:
            # pure coefficient
            inner_terms.append(Num(coeff))
        elif new_deg == 1:
            if coeff == 1:
                inner_terms.append(ASTNode(NodeType.VARIABLE, value=base_var))
            elif coeff == -1:
                inner_terms.append(Mul(Num(-1), ASTNode(NodeType.VARIABLE, value=base_var)))
            else:
                inner_terms.append(Mul(Num(coeff), ASTNode(NodeType.VARIABLE, value=base_var)))
        else:
            mono = Pow(ASTNode(NodeType.VARIABLE, value=base_var), Num(new_deg))
            if coeff == 1:
                inner_terms.append(mono)
            elif coeff == -1:
                inner_terms.append(Mul(Num(-1), mono))
            else:
                inner_terms.append(Mul(Num(coeff), mono))

    inner_add = Add(*inner_terms)
    factor = (
        ASTNode(NodeType.VARIABLE, value=base_var) if min_deg == 1
        else Pow(ASTNode(NodeType.VARIABLE, value=base_var), Num(min_deg))
    )
    replacement = Mul(factor, inner_add)

    new_root = _replace_node(root, node, replacement)
    if new_root is None:
        return None
    new_root.mark_dirty()

    fwd_id = _find_node_id(new_root, replacement)
    return (new_root, fwd_id) if fwd_id is not None else None


def _monomial_decomp(node: ASTNode) -> Optional[Tuple[int, int]]:
    """Decompose a child of an Add into (integer_coefficient, variable_degree).
    Returns None if the node isn't a clean monomial in a single variable.

    Recognized shapes:
      Num(c)                                    → (c, 0)
      Var()                                     → (1, 1)
      Pow(Var, Num(n))                          → (1, n)
      Mul(Num(c), Var())                        → (c, 1)
      Mul(Num(c), Pow(Var, Num(n)))             → (c, n)
    """
    if node.node_type == NodeType.NUMBER:
        try:
            return (int(float(node.value)), 0)
        except (ValueError, TypeError):
            return None
    if node.node_type == NodeType.VARIABLE:
        return (1, 1)
    if node.node_type == NodeType.POW:
        if (node.children[0].node_type == NodeType.VARIABLE
                and node.children[1].node_type == NodeType.NUMBER):
            try:
                return (1, int(float(node.children[1].value)))
            except (ValueError, TypeError):
                return None
        return None
    if node.node_type == NodeType.MUL and len(node.children) == 2:
        c, m = node.children
        if c.node_type != NodeType.NUMBER:
            return None
        try:
            coeff = int(float(c.value))
        except (ValueError, TypeError):
            return None
        if m.node_type == NodeType.VARIABLE:
            return (coeff, 1)
        if (m.node_type == NodeType.POW
                and m.children[0].node_type == NodeType.VARIABLE
                and m.children[1].node_type == NodeType.NUMBER):
            try:
                return (coeff, int(float(m.children[1].value)))
            except (ValueError, TypeError):
                return None
    return None


def inverse_introduce_redundant_one(rng: random.Random, root: ASTNode, node: ASTNode, node_id: int) -> InverseResult:
    """x → Mul(1, x). Only bare terms.
    Forward: REMOVE_ONE"""
    if not _is_bare_term(node):
        return None
    replacement = Mul(Num(1), node.clone())

    new_root = _replace_node(root, node, replacement)
    if new_root is None:
        return None
    new_root.mark_dirty()

    fwd_id = _find_node_id(new_root, replacement)
    return (new_root, fwd_id) if fwd_id is not None else None


def inverse_introduce_redundant_zero(rng: random.Random, root: ASTNode, node: ASTNode, node_id: int) -> InverseResult:
    """x → Add(x, 0). Only bare terms.
    Forward: REMOVE_ZERO"""
    if not _is_bare_term(node):
        return None
    replacement = Add(node.clone(), Num(0))

    new_root = _replace_node(root, node, replacement)
    if new_root is None:
        return None
    new_root.mark_dirty()

    fwd_id = _find_node_id(new_root, replacement)
    return (new_root, fwd_id) if fwd_id is not None else None


def inverse_shuffle_commutative(
    rng: random.Random, root: ASTNode, node: ASTNode, node_id: int
) -> InverseResult:
    """Shuffle children of ADD/MUL to be out of canonical sorted order.

    Forward: SORT_COMMUTATIVE.

    Applicability guard: node must be ADD or MUL with ≥ 2 children and at
    least 2 *distinct* child expressions (otherwise every permutation is
    semantically equivalent and the engine may not recognise any order as
    'unsorted').

    After shuffling we do NOT check whether the result is already sorted —
    the validation step in generate_one (engine.get_candidates → SORT_COMMUTATIVE
    present?) filters that case and increments inverse_skip_counts.
    For a 2-child node with distinct children exactly 1/2 shuffles land on
    sorted order; for 3+ children the miss rate drops quickly.
    """
    if node.node_type not in (NodeType.ADD, NodeType.MUL):
        return None
    if len(node.children) < 2:
        return None
    # Require ≥ 2 distinct child expressions — otherwise shuffle is a no-op.
    exprs = [c.to_expr() for c in node.children]
    if len(set(exprs)) < 2:
        return None

    # Shuffle in-place (node is already a clone from _pick_and_apply_inverse).
    children = list(node.children)
    rng.shuffle(children)
    node.children = children
    # _rebuild_parents on node: children's parent pointers all still point to node
    # (reorder doesn't change parent), so this is a no-op for parent links but
    # also harmless — call it anyway to stay consistent with other inverses.
    node._rebuild_parents()
    root.mark_dirty()

    return (root, node_id)


# ====================== INVERSE TRANSFORM REGISTRY ======================

INVERSE_REGISTRY: List[Tuple[str, Callable, ActionType, float]] = [
    # (name, function, forward_action, weight) — weights match spec §4.3
    ("SPLIT_COEFFICIENT",       inverse_split_coefficient,         ActionType.COMBINE_COEFF,  0.20),
    ("FACTOR_PAIR",             inverse_factor_pair,               ActionType.EXPAND,         0.10),
    ("FACTOR_VARIABLE",         inverse_factor_variable,           ActionType.EXPAND,         0.10),
    ("UNCOLLECT_TERMS",         inverse_uncollect_terms,           ActionType.COLLECT_TERMS,  0.15),
    ("SPLIT_POWER",             inverse_split_power,               ActionType.MERGE_POWER,    0.15),
    ("UNFLATTEN_ADD",           inverse_unflatten_add,             ActionType.FLATTEN_ADD,    0.10),
    ("UNFLATTEN_MUL",           inverse_unflatten_mul,             ActionType.FLATTEN_MUL,    0.10),
    ("UNFOLD_CONST",            inverse_unfold_const,              ActionType.FOLD_CONST,     0.05),
    ("INTRODUCE_REDUNDANT_ONE", inverse_introduce_redundant_one,   ActionType.REMOVE_ONE,          0.025),
    ("INTRODUCE_REDUNDANT_ZERO",inverse_introduce_redundant_zero,  ActionType.REMOVE_ZERO,         0.025),
    ("SHUFFLE_COMMUTATIVE",     inverse_shuffle_commutative,       ActionType.SORT_COMMUTATIVE,    0.03),
]

# Compositional constraints: (prev_inverse, current_inverse) pairs that are forbidden
FORBIDDEN_SEQUENCES = {
    ("UNFOLD_CONST", "UNFOLD_CONST"),
    ("SPLIT_COEFFICIENT", "UNCOLLECT_TERMS"),
}

# Max once per expression
ONCE_PER_EXPRESSION = {
    "INTRODUCE_REDUNDANT_ONE",
    "INTRODUCE_REDUNDANT_ZERO",
    "UNFOLD_CONST",
    "SHUFFLE_COMMUTATIVE",   # one shuffle per trajectory is enough; prevents dominance
}


# ====================== HELPERS ======================

def _is_bare_term(node: ASTNode) -> bool:
    """Variable or Pow(variable, number)."""
    if node.node_type == NodeType.VARIABLE:
        return True
    if (node.node_type == NodeType.POW
            and node.children[0].node_type == NodeType.VARIABLE
            and node.children[1].node_type == NodeType.NUMBER):
        return True
    return False


def _replace_node(root: ASTNode, old: ASTNode, new: ASTNode) -> Optional[ASTNode]:
    """Replace old node with new in the tree. Returns new root."""
    if root is old:
        new._parent = None  # guard: new may have had a parent from a previous context
        return new
    if old.parent is None:
        return None  # can't find old in tree
    old.parent.replace_child(old, new)
    return root


def _find_node_id(root: ASTNode, target: ASTNode) -> Optional[int]:
    """Find preorder index of a node (by identity, not equality)."""
    for i, node in enumerate(root.iter_preorder()):
        if node is target:
            return i
    return None


def _max_depth(root: ASTNode) -> int:
    """Maximum depth of any node in the tree."""
    return max(n.depth for n in root.iter_preorder())


def _augment_commutative(rng: random.Random, root: ASTNode) -> ASTNode:
    """50% chance to shuffle children of commutative nodes. Data augmentation, not a trajectory step."""
    for node in root.iter_preorder():
        if node.node_type in (NodeType.ADD, NodeType.MUL) and len(node.children) >= 2:
            if rng.random() < 0.5:
                rng.shuffle(node.children)
                node._rebuild_parents()
    return root


# ====================== TRAJECTORY GENERATOR ======================

class TrajectoryGenerator:
    def __init__(
        self,
        seed: int = 42,
        max_ast_depth: int = 6,
        max_trajectory_length: int = 6,
    ) -> None:
        self.rng = random.Random(seed)
        self.engine = SymbolicEngine()
        self.max_ast_depth = max_ast_depth
        self.max_trajectory_length = max_trajectory_length
        # Diagnostic: counts how many times each inverse was skipped because
        # the corresponding forward action wasn't in engine.get_candidates().
        # Non-zero entries reveal inverse ops that produce engine-invisible states.
        # TODO(perf #2): _pick_and_apply_inverse clones N×K ASTs per backward step.
        #   Refactor: run applicability predicates first (no clone), clone only
        #   survivors. ~10x speedup on deep trees; defer until 1M-traj scale.
        self.inverse_skip_counts: Counter = Counter()

    def generate_one(self, canonical_ast: ASTNode, trajectory_id: str) -> Optional[Trajectory]:
        """Generate one backward trajectory from a canonical polynomial.
        
        Records training pairs directly during backward pass.
        No forward replay — avoids node_id drift between inverse/forward.
        
        Steps are stored in forward order (most scrambled first → canonical).
        """

        current = canonical_ast.clone()
        current.mark_dirty()
        current._rebuild_parents()

        # Each record: (scrambled_state, fwd_action, fwd_node_id, candidates)
        # Recorded during backward pass, then reversed for forward order.
        backward_records: list = []
        inverse_names: List[str] = []
        used_once: set = set()
        prev_inverse: Optional[str] = None

        for step_idx in range(self.rng.randint(1, self.max_trajectory_length)):
            result = self._pick_and_apply_inverse(current, prev_inverse, used_once)
            if result is None:
                break

            inv_name, new_root, fwd_node_id = result
            fwd_action = self._get_forward_action(inv_name)

            # Validate: forward action must exist in candidates of scrambled state
            candidates = self.engine.get_candidates(new_root)
            # Store full (node_id, action_name) pairs — not deduplicated
            candidate_pairs = [[nid, a.value] for nid, _, a in candidates]

            if not any(a.value == fwd_action.value for _, _, a in candidates):
                self.inverse_skip_counts[inv_name] += 1
                continue  # skip this step, try next inverse

            # Check depth constraint
            if _max_depth(new_root) > self.max_ast_depth:
                return None

            backward_records.append((
                new_root.clone(),
                fwd_action,
                fwd_node_id,
                candidate_pairs,
            ))
            inverse_names.append(inv_name)

            current = new_root
            prev_inverse = inv_name

            if inv_name in ONCE_PER_EXPRESSION:
                used_once.add(inv_name)

        if not backward_records:
            return None

        # Build steps in forward order (most scrambled → canonical)
        steps: List[TrajectoryStep] = []
        for state, fwd_action, fwd_node_id, cand_actions in reversed(backward_records):
            steps.append(TrajectoryStep(
                state=state.to_dict(),
                state_expr=state.to_expr(),
                candidate_actions=cand_actions,
                gold_action=fwd_action.value,
                gold_node_id=fwd_node_id,
                complexity=state.complexity(),
            ))

        scrambled = backward_records[-1][0]

        return Trajectory(
            trajectory_id=trajectory_id,
            canonical_expr=canonical_ast.to_expr(),
            canonical_ast=canonical_ast.to_dict(),
            original_expr=scrambled.to_expr(),
            original_ast=scrambled.to_dict(),
            steps=steps,
            difficulty=len(steps),
            inverse_sequence=inverse_names,
        )

    def _pick_and_apply_inverse(
        self,
        root: ASTNode,
        prev_inverse: Optional[str],
        used_once: set,
    ) -> Optional[Tuple[str, ASTNode, int]]:
        """Pick a valid inverse transform, apply it, return (name, new_root, forward_node_id).
        Pre-filters by actually trying each inverse, then weighted-samples from successes."""

        # Collect all actually-successful applications
        successes: List[Tuple[str, ASTNode, int, float]] = []  # (name, new_root, fwd_id, weight)

        for inv_name, fn, _, weight in INVERSE_REGISTRY:
            # compositional constraint
            if prev_inverse and (prev_inverse, inv_name) in FORBIDDEN_SEQUENCES:
                continue
            # once-per-expression constraint
            if inv_name in ONCE_PER_EXPRESSION and inv_name in used_once:
                continue

            for node_id, node in enumerate(root.iter_preorder()):
                cloned_root = root.clone()
                cloned_root._rebuild_parents()
                cloned_node = cloned_root.get_node_by_id(node_id)
                if cloned_node is None:
                    continue

                result = fn(self.rng, cloned_root, cloned_node, node_id)
                if result is not None:
                    new_root, fwd_node_id = result
                    successes.append((inv_name, new_root, fwd_node_id, weight))

        if not successes:
            return None

        # Weighted random selection from verified successes
        weights = [w for _, _, _, w in successes]
        choice = self.rng.choices(successes, weights=weights, k=1)[0]
        inv_name, new_root, fwd_node_id, _ = choice
        return (inv_name, new_root, fwd_node_id)

    def _get_forward_action(self, inv_name: str) -> ActionType:
        for name, _, fwd_action, _ in INVERSE_REGISTRY:
            if name == inv_name:
                return fwd_action
        raise ValueError(f"Unknown inverse: {inv_name}")


# ====================== BATCH GENERATION ======================

def generate_dataset(
    count: int = 1000,
    seed: int = 42,
    max_degree: int = 4,
    max_depth: int = 6,
    max_traj_len: int = 6,
    output_dir: str = "trajectories",
    max_attempts_per_poly: int = 5,
) -> List[Trajectory]:
    """Generate a dataset of backward trajectories."""
    rng = random.Random(seed)
    gen = TrajectoryGenerator(seed=seed, max_ast_depth=max_depth, max_trajectory_length=max_traj_len)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    trajectories: List[Trajectory] = []
    failed = 0

    for i in tqdm(range(count), desc="Generating trajectories"):
        poly = sample_canonical_polynomial(rng, max_degree=max_degree)
        traj = None

        for attempt in range(max_attempts_per_poly):
            traj = gen.generate_one(poly, trajectory_id=f"traj_{i:07d}")
            if traj is not None:
                break

        if traj is not None:
            trajectories.append(traj)
            path = out_path / f"{traj.trajectory_id}.json"
            with path.open("w", encoding="utf-8") as f:
                json.dump(asdict(traj), f, indent=2, ensure_ascii=False)
        else:
            failed += 1

    print(f"\nGenerated: {len(trajectories)}, Failed: {failed}")

    # Diagnostic: report which inverse ops were skipped due to missing forward action.
    # Non-zero counts = states produced by that inverse where engine couldn't find
    # the corresponding forward action — potential engine gap source.
    if gen.inverse_skip_counts:
        import sys
        total_skips = sum(gen.inverse_skip_counts.values())
        print(f"\n[diag] inverse_skip_counts (total={total_skips}):", file=sys.stderr)
        for inv, cnt in sorted(gen.inverse_skip_counts.items(), key=lambda x: -x[1]):
            print(f"  {inv:<35s}  {cnt:6d}", file=sys.stderr)
    else:
        import sys
        print("\n[diag] inverse_skip_counts: all zeros (engine covers all inverse ops)",
              file=sys.stderr)

    return trajectories


# ====================== TESTS ======================

if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Windows console: pretty() uses ├─/└─

    # Test canonical polynomial generation
    rng = random.Random(42)
    print("=== Sample canonical polynomials ===")
    for i in range(10):
        poly = sample_canonical_polynomial(rng)
        print(f"  {poly}")

    # Test trajectory generation (small batch)
    print("\n=== Trajectory generation test ===")
    import tempfile
    test_dir = tempfile.mkdtemp(prefix="isre_traj_test_")
    print(f"  output dir: {test_dir}")
    trajs = generate_dataset(count=50, seed=42, output_dir=test_dir)

    if trajs:
        print(f"\n=== Sample trajectory ===")
        t = trajs[0]
        print(f"  canonical: {t.canonical_expr}")
        print(f"  original:  {t.original_expr}")
        print(f"  difficulty: {t.difficulty}")
        print(f"  inverse_sequence: {t.inverse_sequence}")
        for i, step in enumerate(t.steps):
            print(f"  step {i}: {step.state_expr}")
            print(f"    candidates: {step.candidate_actions}")
            print(f"    gold: {step.gold_action} at node {step.gold_node_id}")
            print(f"    complexity: {step.complexity}")
    else:
        print("  No trajectories generated — inverse transforms may not match any nodes.")
        print("  This is expected if canonical polynomials don't have Mul(k,x) nodes for uncollect/split.")

    # Diagnostic: check what canonical polys look like
    print("\n=== Diagnostic: canonical poly structure ===")
    rng2 = random.Random(42)
    for i in range(5):
        poly = sample_canonical_polynomial(rng2)
        print(f"\n  poly: {poly}")
        print(f"  tree:\n{poly.pretty()}")
        candidates = SymbolicEngine().get_candidates(poly)
        print(f"  forward candidates: {[(nid, a.value) for nid, _, a in candidates]}")