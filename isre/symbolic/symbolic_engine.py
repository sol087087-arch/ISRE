"""
Symbolic Engine v1 — deterministic candidate generation and transform application.
The neural network does NOT generate transforms — it only ranks candidates from this module.
"""

from collections import Counter
from enum import Enum
from typing import List, Tuple, Optional
from .isre_ast import ASTNode, NodeType, Num, Var, Add, Mul, Pow


class ActionType(str, Enum):
    """All forward transforms for v1."""
    COLLECT_TERMS = "COLLECT_TERMS"       # Add(x, x) → Mul(2, x)
    COMBINE_COEFF = "COMBINE_COEFF"       # Add(Mul(2,x), Mul(3,x)) → Mul(5, x)
    EXPAND = "EXPAND"                     # Mul(a, Add(b,c)) → Add(Mul(a,b), Mul(a,c))
    FLATTEN_ADD = "FLATTEN_ADD"           # Add(Add(a,b), c) → Add(a,b,c)
    FLATTEN_MUL = "FLATTEN_MUL"           # Mul(Mul(a,b), c) → Mul(a,b,c)
    FOLD_CONST = "FOLD_CONST"             # Add(2,3) → 5
    MERGE_POWER = "MERGE_POWER"           # Mul(Pow(x,a), Pow(x,b)) → Pow(x, a+b)
    REMOVE_ONE = "REMOVE_ONE"             # Mul(1, x) → x
    REMOVE_ZERO = "REMOVE_ZERO"           # Add(x, 0) → x
    SORT_COMMUTATIVE = "SORT_COMMUTATIVE" # canonical child ordering


# Return type: (node_id, node_ref, action)
Candidate = Tuple[int, ASTNode, ActionType]


class SymbolicEngine:
    """Deterministic module. Generates valid candidates. Neural net only ranks them."""

    # ====================== PUBLIC API ======================

    def get_candidates(self, root: ASTNode) -> List[Candidate]:
        """Return all valid (node_id, node_ref, action) for current AST.
        Single preorder pass. node_ref avoids repeated lookups."""
        candidates: List[Candidate] = []

        checks = [
            (ActionType.COLLECT_TERMS, self._can_collect_terms),
            (ActionType.COMBINE_COEFF, self._can_combine_coeff),
            (ActionType.EXPAND, self._can_expand),
            (ActionType.FLATTEN_ADD, self._can_flatten_add),
            (ActionType.FLATTEN_MUL, self._can_flatten_mul),
            (ActionType.FOLD_CONST, self._can_fold_const),
            (ActionType.MERGE_POWER, self._can_merge_power),
            (ActionType.REMOVE_ONE, self._can_remove_one),
            (ActionType.REMOVE_ZERO, self._can_remove_zero),
            (ActionType.SORT_COMMUTATIVE, self._can_sort_commutative),
        ]

        for node_id, node in enumerate(root.iter_preorder()):
            for action, check in checks:
                if check(node):
                    candidates.append((node_id, node, action))

        return candidates

    def apply(self, root: ASTNode, node_id: int, action: ActionType) -> ASTNode:
        """Apply transform to a clone of the AST. Returns new root.
        Original AST is not modified (safe for trajectory recording)."""
        new_root = root.clone()
        target = new_root.get_node_by_id(node_id)
        if target is None:
            raise ValueError(f"node_id {node_id} not found in AST")

        apply_fn = {
            ActionType.COLLECT_TERMS: self._apply_collect_terms,
            ActionType.COMBINE_COEFF: self._apply_combine_coeff,
            ActionType.EXPAND: self._apply_expand,
            ActionType.FLATTEN_ADD: self._apply_flatten_add,
            ActionType.FLATTEN_MUL: self._apply_flatten_mul,
            ActionType.FOLD_CONST: self._apply_fold_const,
            ActionType.MERGE_POWER: self._apply_merge_power,
            ActionType.REMOVE_ONE: self._apply_remove_one,
            ActionType.REMOVE_ZERO: self._apply_remove_zero,
            ActionType.SORT_COMMUTATIVE: self._apply_sort_commutative,
        }[action]

        result = apply_fn(target)

        # result may replace the target node entirely
        if target is new_root:
            # root was replaced
            result._parent = None
            result.mark_dirty()
            return result

        if result is not target:
            target.parent.replace_child(target, result)

        new_root.mark_dirty()
        return new_root

    # ====================== MATCH FUNCTIONS ======================

    def _can_collect_terms(self, node: ASTNode) -> bool:
        """Add with 2+ bare children that have the same variable part.
        For: Add(x, x, x) → Mul(3, x).
        Bare = VARIABLE or Pow(VARIABLE, NUMBER). No coefficient wrapper.
        Does NOT handle Mul(coeff, x) — that's COMBINE_COEFF."""
        if node.node_type != NodeType.ADD:
            return False
        bare_keys = [self._var_key(c) for c in node.children if self._is_bare_term(c)]
        return any(count >= 2 for count in Counter(bare_keys).values())

    def _can_combine_coeff(self, node: ASTNode) -> bool:
        """Add with 2+ children that are Mul(number, expr) with same expr.
        For: Add(Mul(2,x), Mul(3,x)) → Mul(5,x).
        Also handles mix of bare x and Mul(k,x)."""
        if node.node_type != NodeType.ADD:
            return False
        keys = []
        for child in node.children:
            key = self._coeff_key(child)
            if key is not None:
                keys.append(key)
        return any(count >= 2 for count in Counter(keys).values())

    def _can_expand(self, node: ASTNode) -> bool:
        """Mul where at least one child is Add."""
        if node.node_type != NodeType.MUL:
            return False
        return any(c.node_type == NodeType.ADD for c in node.children)

    def _can_flatten_add(self, node: ASTNode) -> bool:
        """Add with at least one Add child."""
        if node.node_type != NodeType.ADD:
            return False
        return any(c.node_type == NodeType.ADD for c in node.children)

    def _can_flatten_mul(self, node: ASTNode) -> bool:
        """Mul with at least one Mul child."""
        if node.node_type != NodeType.MUL:
            return False
        return any(c.node_type == NodeType.MUL for c in node.children)

    def _can_fold_const(self, node: ASTNode) -> bool:
        """Add or Mul where ALL children are NUMBER.
        Only NUMBER, not CONST (we don't fold pi + e)."""
        if node.node_type not in (NodeType.ADD, NodeType.MUL):
            return False
        if len(node.children) < 2:
            return False
        return all(c.node_type == NodeType.NUMBER for c in node.children)

    def _can_remove_zero(self, node: ASTNode) -> bool:
        """Add with at least one zero child."""
        if node.node_type != NodeType.ADD:
            return False
        return any(self._is_zero(c) for c in node.children)

    def _can_remove_one(self, node: ASTNode) -> bool:
        """Mul with at least one one child."""
        if node.node_type != NodeType.MUL:
            return False
        return any(self._is_one(c) for c in node.children)

    def _can_merge_power(self, node: ASTNode) -> bool:
        """Mul with 2+ Pow children sharing the same simple base.
        Simple base = single VARIABLE node (v1 constraint).
        Exponents must be numeric (NUMBER) — not compound expressions."""
        if node.node_type != NodeType.MUL:
            return False
        powers = [c for c in node.children if c.node_type == NodeType.POW]
        if len(powers) < 2:
            return False
        # v1: only simple variable bases, numeric exponents
        first_base = powers[0].children[0]
        if first_base.node_type != NodeType.VARIABLE:
            return False
        return all(
            p.children[0].node_type == NodeType.VARIABLE
            and p.children[0].value == first_base.value
            and p.children[1].node_type == NodeType.NUMBER
            for p in powers
        )

    def _can_sort_commutative(self, node: ASTNode) -> bool:
        """Add or Mul whose children are not in canonical order."""
        if node.node_type not in (NodeType.ADD, NodeType.MUL):
            return False
        if len(node.children) < 2:
            return False
        keys = [self._sort_key(c) for c in node.children]
        return keys != sorted(keys)

    # ====================== APPLY FUNCTIONS ======================

    def _apply_collect_terms(self, node: ASTNode) -> ASTNode:
        """Add(x, x, x) → Mul(3, x) (for the largest group of bare duplicates)."""
        keys = {}
        for child in node.children:
            if self._is_bare_term(child):
                k = self._var_key(child)
                keys.setdefault(k, []).append(child)

        # find largest group
        best_key = max(keys, key=lambda k: len(keys[k]))
        group = keys[best_key]
        if len(group) < 2:
            return node

        # build replacement: Mul(count, representative)
        representative = group[0].clone()
        replacement = Mul(Num(len(group)), representative)

        # rebuild children: replacement + everything not in group
        group_set = set(id(c) for c in group)
        new_children = [c for c in node.children if id(c) not in group_set]
        new_children.append(replacement)

        if len(new_children) == 1:
            return new_children[0]
        node.children = new_children
        node._rebuild_parents()
        return node

    def _apply_combine_coeff(self, node: ASTNode) -> ASTNode:
        """Add(Mul(2,x), Mul(3,x)) → Mul(5,x). Also handles bare x as coeff 1."""
        groups: dict[str, list[tuple[float, ASTNode, ASTNode]]] = {}
        for child in node.children:
            info = self._coeff_info(child)
            if info is not None:
                key, coeff, var_part = info
                groups.setdefault(key, []).append((coeff, var_part, child))

        # find a group with 2+ members
        target_key = None
        for k, v in groups.items():
            if len(v) >= 2:
                target_key = k
                break
        if target_key is None:
            return node

        group = groups[target_key]
        total_coeff = sum(c for c, _, _ in group)
        var_part = group[0][1].clone()

        if total_coeff == 1:
            replacement = var_part
        elif total_coeff == 0:
            replacement = Num(0)
        else:
            coeff_val = int(total_coeff) if total_coeff == int(total_coeff) else total_coeff
            replacement = Mul(Num(coeff_val), var_part)

        group_ids = set(id(orig) for _, _, orig in group)
        new_children = [c for c in node.children if id(c) not in group_ids]
        new_children.append(replacement)

        if len(new_children) == 1:
            return new_children[0]
        node.children = new_children
        node._rebuild_parents()
        return node

    def _apply_expand(self, node: ASTNode) -> ASTNode:
        """Mul(a, Add(b,c)) → Add(Mul(a,b), Mul(a,c)).
        Expands first Add child found."""
        add_idx = None
        for i, child in enumerate(node.children):
            if child.node_type == NodeType.ADD:
                add_idx = i
                break
        if add_idx is None:
            return node

        add_node = node.children[add_idx]
        other_children = [c for i, c in enumerate(node.children) if i != add_idx]

        # distribute: for each term in Add, create Mul(other_children..., term)
        new_add_children = []
        for term in add_node.children:
            if len(other_children) == 1:
                mul_node = Mul(other_children[0].clone(), term.clone())
            else:
                mul_node = Mul(*[c.clone() for c in other_children], term.clone())
            new_add_children.append(mul_node)

        result = Add(*new_add_children)
        return result

    def _apply_flatten_add(self, node: ASTNode) -> ASTNode:
        """Add(Add(a,b), c) → Add(a,b,c). Flattens all nested Add children."""
        new_children = []
        for child in node.children:
            if child.node_type == NodeType.ADD:
                new_children.extend(child.children)
            else:
                new_children.append(child)
        node.children = new_children
        node._rebuild_parents()
        return node

    def _apply_flatten_mul(self, node: ASTNode) -> ASTNode:
        """Mul(Mul(a,b), c) → Mul(a,b,c). Flattens all nested Mul children."""
        new_children = []
        for child in node.children:
            if child.node_type == NodeType.MUL:
                new_children.extend(child.children)
            else:
                new_children.append(child)
        node.children = new_children
        node._rebuild_parents()
        return node

    def _apply_fold_const(self, node: ASTNode) -> ASTNode:
        """Add(2,3) → 5, Mul(2,3) → 6. Folds ALL numeric children."""
        values = []
        for child in node.children:
            try:
                values.append(float(child.value))
            except (ValueError, TypeError):
                return node

        if node.node_type == NodeType.ADD:
            result_val = sum(values)
        elif node.node_type == NodeType.MUL:
            result_val = 1
            for v in values:
                result_val *= v
        else:
            return node

        int_val = int(result_val) if result_val == int(result_val) else result_val
        return Num(int_val)

    def _apply_remove_zero(self, node: ASTNode) -> ASTNode:
        """Add(x, 0) → x. Removes all zero children."""
        new_children = [c for c in node.children if not self._is_zero(c)]
        if len(new_children) == 0:
            return Num(0)
        if len(new_children) == 1:
            return new_children[0]
        node.children = new_children
        node._rebuild_parents()
        return node

    def _apply_remove_one(self, node: ASTNode) -> ASTNode:
        """Mul(1, x) → x. Removes all one children."""
        new_children = [c for c in node.children if not self._is_one(c)]
        if len(new_children) == 0:
            return Num(1)
        if len(new_children) == 1:
            return new_children[0]
        node.children = new_children
        node._rebuild_parents()
        return node

    def _apply_merge_power(self, node: ASTNode) -> ASTNode:
        """Mul(Pow(x,a), Pow(x,b)) → Pow(x, a+b). Merges all matching powers."""
        powers = [c for c in node.children if c.node_type == NodeType.POW]
        non_powers = [c for c in node.children if c.node_type != NodeType.POW]

        if len(powers) < 2:
            return node

        # sum exponents (v1: numeric exponents only)
        total_exp = 0
        base = powers[0].children[0].clone()
        for p in powers:
            try:
                total_exp += float(p.children[1].value)
            except (ValueError, TypeError):
                return node  # non-numeric exponent, bail

        int_exp = int(total_exp) if total_exp == int(total_exp) else total_exp
        merged = Pow(base, Num(int_exp))

        new_children = non_powers + [merged]
        if len(new_children) == 1:
            return new_children[0]
        node.children = new_children
        node._rebuild_parents()
        return node

    def _apply_sort_commutative(self, node: ASTNode) -> ASTNode:
        """Sort children of Add/Mul into canonical order."""
        node.children.sort(key=self._sort_key)
        node._rebuild_parents()
        return node

    # ====================== HELPERS ======================

    def _is_zero(self, node: ASTNode) -> bool:
        return node.node_type == NodeType.NUMBER and node.value == "0"

    def _is_one(self, node: ASTNode) -> bool:
        return node.node_type == NodeType.NUMBER and node.value == "1"

    def _is_bare_term(self, node: ASTNode) -> bool:
        """Is this a bare variable or Pow(var, n) — no coefficient wrapper."""
        if node.node_type == NodeType.VARIABLE:
            return True
        if node.node_type == NodeType.POW and node.children[0].node_type == NodeType.VARIABLE:
            return True
        return False

    def _var_key(self, node: ASTNode) -> str:
        """Variable-part key for a bare term (no coefficient).
        x → 'x', Pow(x,2) → 'x^2'."""
        if node.node_type == NodeType.VARIABLE:
            return node.value
        if node.node_type == NodeType.POW:
            return f"{node.children[0].value}^{node.children[1].value}"
        return node.to_json()  # fallback — stable, clone-safe

    def _coeff_key(self, node: ASTNode) -> Optional[str]:
        """Variable-part key for coefficient extraction.
        Returns None if node is not a recognizable monomial.
        x → 'x', Mul(3,x) → 'x', Mul(2, Pow(x,2)) → 'x^2', Pow(x,3) → 'x^3'."""
        if self._is_bare_term(node):
            return self._var_key(node)
        if (node.node_type == NodeType.MUL
                and len(node.children) == 2
                and node.children[0].node_type == NodeType.NUMBER
                and self._is_bare_term(node.children[1])):
            return self._var_key(node.children[1])
        return None

    def _coeff_info(self, node: ASTNode) -> Optional[tuple[str, float, ASTNode]]:
        """Extract (var_key, coefficient, var_part) from a monomial.
        x → ('x', 1, x), Mul(3, x) → ('x', 3, x), Pow(x,2) → ('x^2', 1, Pow(x,2))."""
        if self._is_bare_term(node):
            return (self._var_key(node), 1.0, node)
        if (node.node_type == NodeType.MUL
                and len(node.children) == 2
                and node.children[0].node_type == NodeType.NUMBER
                and self._is_bare_term(node.children[1])):
            try:
                coeff = float(node.children[0].value)
            except ValueError:
                return None
            return (self._var_key(node.children[1]), coeff, node.children[1])
        return None

    def _sort_key(self, node: ASTNode) -> tuple:
        """Canonical sort key for commutative children.
        Order: numbers first, then variables, then complex expressions.
        Within numbers: by value. Within variables: by name then degree."""
        if node.node_type == NodeType.NUMBER:
            try:
                return (0, float(node.value), "")
            except ValueError:
                return (0, 0, node.value)
        if node.node_type == NodeType.VARIABLE:
            return (1, 0, node.value)
        if node.node_type == NodeType.POW:
            try:
                deg = float(node.children[1].value)
            except (ValueError, TypeError, IndexError):
                deg = 0
            return (2, deg, "")
        if node.node_type == NodeType.MUL:
            # Mul(coeff, var_part) — sort by var_part then coeff
            return (3, 0, str(node._structural_tuple()))
        # everything else
        return (9, 0, str(node._structural_tuple()))


# ====================== TESTS ======================

if __name__ == "__main__":
    engine = SymbolicEngine()

    def test(name, root, expected_actions=None):
        candidates = engine.get_candidates(root)
        actions = [(nid, a.value) for nid, _, a in candidates]
        print(f"\n{'='*60}")
        print(f"TEST: {name}")
        print(f"expr: {root}")
        print(f"candidates: {actions}")
        if expected_actions:
            for ea in expected_actions:
                assert any(a == ea for _, a in actions), f"  MISSING: {ea}"
            print(f"  ✓ all expected actions found")
        return candidates

    def test_apply(name, root, node_id, action, expected_expr):
        result = engine.apply(root, node_id, action)
        print(f"\n{'='*60}")
        print(f"APPLY: {name}")
        print(f"  before: {root}")
        print(f"  action: {action.value} on node {node_id}")
        print(f"  after:  {result}")
        print(f"  expect: {expected_expr}")
        assert str(result) == expected_expr, f"  ✗ got {result}, expected {expected_expr}"
        print(f"  ✓ OK")
        return result

    # --- candidate detection ---

    # REMOVE_ZERO: Add(x, 0)
    test("Add(x, 0)", Add(Var(), Num(0)), ["REMOVE_ZERO"])

    # REMOVE_ONE: Mul(1, x)
    test("Mul(1, x)", Mul(Num(1), Var()), ["REMOVE_ONE"])

    # FOLD_CONST: Add(2, 3)
    test("Add(2, 3)", Add(Num(2), Num(3)), ["FOLD_CONST"])

    # FOLD_CONST: Mul(4, 5)
    test("Mul(4, 5)", Mul(Num(4), Num(5)), ["FOLD_CONST"])

    # EXPAND: Mul(x, Add(x, 1))
    test("Mul(x, Add(x,1))", Mul(Var(), Add(Var(), Num(1))), ["EXPAND"])

    # FLATTEN_ADD: Add(Add(x, 1), 2)
    test("Add(Add(x,1), 2)", Add(Add(Var(), Num(1)), Num(2)), ["FLATTEN_ADD"])

    # FLATTEN_MUL: Mul(Mul(x, 2), 3)
    test("Mul(Mul(x,2), 3)", Mul(Mul(Var(), Num(2)), Num(3)), ["FLATTEN_MUL"])

    # COLLECT_TERMS: Add(x, x)
    test("Add(x, x)", Add(Var(), Var()), ["COLLECT_TERMS"])

    # COMBINE_COEFF: Add(Mul(2,x), Mul(3,x))
    test("Add(Mul(2,x), Mul(3,x))",
         Add(Mul(Num(2), Var()), Mul(Num(3), Var())),
         ["COMBINE_COEFF"])

    # MERGE_POWER: Mul(Pow(x,2), Pow(x,3))
    test("Mul(Pow(x,2), Pow(x,3))",
         Mul(Pow(Var(), Num(2)), Pow(Var(), Num(3))),
         ["MERGE_POWER"])

    # SORT_COMMUTATIVE: Add(x, 1) — x before 1, should sort to 1, x
    test("Add(x, 1)", Add(Var(), Num(1)), ["SORT_COMMUTATIVE"])

    # --- apply transforms ---

    test_apply("fold Add(2,3)", Add(Num(2), Num(3)),
               0, ActionType.FOLD_CONST, "5")

    test_apply("fold Mul(4,5)", Mul(Num(4), Num(5)),
               0, ActionType.FOLD_CONST, "20")

    test_apply("remove zero", Add(Var(), Num(0)),
               0, ActionType.REMOVE_ZERO, "x")

    test_apply("remove one", Mul(Num(1), Var()),
               0, ActionType.REMOVE_ONE, "x")

    test_apply("flatten add", Add(Add(Var(), Num(1)), Num(2)),
               0, ActionType.FLATTEN_ADD, "(x + 1 + 2)")

    test_apply("flatten mul", Mul(Mul(Var(), Num(2)), Num(3)),
               0, ActionType.FLATTEN_MUL, "(x * 2 * 3)")

    test_apply("expand", Mul(Num(2), Add(Var(), Num(3))),
               0, ActionType.EXPAND, "((2 * x) + (2 * 3))")

    test_apply("collect x+x", Add(Var(), Var()),
               0, ActionType.COLLECT_TERMS, "(2 * x)")

    test_apply("combine 2x+3x",
               Add(Mul(Num(2), Var()), Mul(Num(3), Var())),
               0, ActionType.COMBINE_COEFF, "(5 * x)")

    test_apply("merge Pow(x,2)*Pow(x,3)",
               Mul(Pow(Var(), Num(2)), Pow(Var(), Num(3))),
               0, ActionType.MERGE_POWER, "(x)^5")

    test_apply("sort Add(x, 1)", Add(Var(), Num(1)),
               0, ActionType.SORT_COMMUTATIVE, "(1 + x)")

    print(f"\n{'='*60}")
    print("ALL TESTS PASSED")