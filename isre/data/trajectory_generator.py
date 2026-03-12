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
    # Pick non-constant degrees first (guaranteed ≥ 1 term with degree ≥ 1)
    non_const_count = rng.randint(max(1, min_terms - 1), min(max_terms, max_degree))
    non_const_degrees = sorted(
        rng.sample(range(1, max_degree + 1), non_const_count),
        reverse=True,
    )

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
    """Pow(x, n) → Mul(Pow(x, a), Pow(x, b)) where a+b=n, n ≤ 4.
    Forward: MERGE_POWER"""
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

    a = rng.randint(1, n - 1)
    b = n - a
    base_var = node.children[0].value

    replacement = Mul(
        Pow(ASTNode(NodeType.VARIABLE, value=base_var), Num(a)),
        Pow(ASTNode(NodeType.VARIABLE, value=base_var), Num(b)),
    )

    new_root = _replace_node(root, node, replacement)
    if new_root is None:
        return None
    new_root.mark_dirty()

    fwd_id = _find_node_id(new_root, replacement)
    return (new_root, fwd_id) if fwd_id is not None else None


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


# ====================== INVERSE TRANSFORM REGISTRY ======================

INVERSE_REGISTRY: List[Tuple[str, Callable, ActionType, float]] = [
    # (name, function, forward_action, weight)
    ("SPLIT_COEFFICIENT",       inverse_split_coefficient,         ActionType.COMBINE_COEFF,  0.20),
    ("UNCOLLECT_TERMS",         inverse_uncollect_terms,           ActionType.COLLECT_TERMS,  0.15),
    ("UNFLATTEN_ADD",           inverse_unflatten_add,             ActionType.FLATTEN_ADD,    0.10),
    ("UNFLATTEN_MUL",           inverse_unflatten_mul,             ActionType.FLATTEN_MUL,    0.10),
    ("UNFOLD_CONST",            inverse_unfold_const,              ActionType.FOLD_CONST,     0.05),
    ("SPLIT_POWER",             inverse_split_power,               ActionType.MERGE_POWER,    0.15),
    ("INTRODUCE_REDUNDANT_ONE", inverse_introduce_redundant_one,   ActionType.REMOVE_ONE,     0.025),
    ("INTRODUCE_REDUNDANT_ZERO",inverse_introduce_redundant_zero,  ActionType.REMOVE_ZERO,    0.025),
    # FACTOR_PAIR deferred — requires more complex pattern matching
]

# Compositional constraints: (prev_inverse, current_inverse) pairs that are forbidden
FORBIDDEN_SEQUENCES = {
    ("UNFOLD_CONST", "UNFOLD_CONST"),
    ("SPLIT_COEFFICIENT", "UNCOLLECT_TERMS"),
}

# Max once per expression
ONCE_PER_EXPRESSION = {"INTRODUCE_REDUNDANT_ONE", "INTRODUCE_REDUNDANT_ZERO", "UNFOLD_CONST"}


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
    return trajectories


# ====================== TESTS ======================

if __name__ == "__main__":
    # Test canonical polynomial generation
    rng = random.Random(42)
    print("=== Sample canonical polynomials ===")
    for i in range(10):
        poly = sample_canonical_polynomial(rng)
        print(f"  {poly}")

    # Test trajectory generation (small batch)
    print("\n=== Trajectory generation test ===")
    trajs = generate_dataset(count=50, seed=42, output_dir="/tmp/test_trajectories")

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