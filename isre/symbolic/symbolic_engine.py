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
        """Fold numeric children. Both ADD and MUL: fire when >= 2 NUMBER
        children present; mixed expressions are allowed and the non-numeric
        children are preserved by _apply_fold_const.
          ADD: Add(1, 2, x^4) -> Add(x^4, 3)
          MUL: Mul(3, 3, x)   -> Mul(9, x)
        Only NUMBER, not CONST (we don't fold pi + e)."""
        if node.node_type not in (NodeType.ADD, NodeType.MUL):
            return False
        if len(node.children) < 2:
            return False
        num_count = sum(1 for c in node.children if c.node_type == NodeType.NUMBER)
        # For both ADD and MUL: fold when 2+ NUMBER children present (may be mixed).
        # MUL(3, 3, x) → MUL(9, x); Add(1, 2, x^4) → Add(x^4, 3).
        return num_count >= 2

    def _can_remove_zero(self, node: ASTNode) -> bool:
        """Add with at least one zero child, OR Mul with at least one zero child.
        Mul(a, 0, b) = 0 — zero absorbs the product."""
        if node.node_type == NodeType.ADD:
            return any(self._is_zero(c) for c in node.children)
        if node.node_type == NodeType.MUL:
            return any(self._is_zero(c) for c in node.children)
        return False

    def _can_remove_one(self, node: ASTNode) -> bool:
        """Mul with at least one 1 child, OR Pow(expr, 1) — trivial exponent."""
        if node.node_type == NodeType.MUL:
            return any(self._is_one(c) for c in node.children)
        if node.node_type == NodeType.POW:
            return (len(node.children) == 2
                    and self._is_one(node.children[1]))
        return False

    def _can_merge_power(self, node: ASTNode) -> bool:
        """Mul with 2+ power-like children sharing the same simple variable base.
        Power-like = POW(x, n) with numeric n, OR bare VARIABLE(x) (treated as x^1).
        v1 constraint: simple variable base only."""
        if node.node_type != NodeType.MUL:
            return False
        power_like = self._collect_power_like(node)
        return len(power_like) >= 2

    def _collect_power_like(self, node: ASTNode):
        """Return list of (base_var_name, exponent, child_node) for all
        power-like children of a MUL node (POW(x,n) or bare VARIABLE)."""
        result = []
        for c in node.children:
            if c.node_type == NodeType.VARIABLE:
                result.append((c.value, 1.0, c))
            elif (c.node_type == NodeType.POW
                    and c.children[0].node_type == NodeType.VARIABLE
                    and c.children[1].node_type == NodeType.NUMBER):
                try:
                    exp = float(c.children[1].value)
                    result.append((c.children[0].value, exp, c))
                except (ValueError, TypeError):
                    pass
        # Only valid if all share the same variable name
        if not result:
            return []
        base = result[0][0]
        if not all(r[0] == base for r in result):
            return []
        return result

    def _can_sort_commutative(self, node: ASTNode) -> bool:
        """Add or Mul whose children are not in canonical order.
        ADD and MUL use different sort keys — canonical order differs:
          ADD: polynomial terms first (high degree → low), constants last.
          MUL: NUMBER coefficient first, then variable parts by descending degree.
        """
        if node.node_type not in (NodeType.ADD, NodeType.MUL):
            return False
        if len(node.children) < 2:
            return False
        key_fn = self._sort_key_add if node.node_type == NodeType.ADD else self._sort_key_mul
        keys = [key_fn(c) for c in node.children]
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

        # Pick the LARGEST group (symmetric with _apply_collect_terms, which
        # uses max-by-size). Picking the first 2+ group made the engine combine
        # a 2-term group while a 3-term group waited, inflating trajectory length
        # and making the greedy tiebreak order-dependent.
        if not groups:
            return node
        target_key = max(groups, key=lambda k: len(groups[k]))
        if len(groups[target_key]) < 2:
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
        """Fold numeric children.
        ADD: fold all NUMBER children into one, keep non-numbers.
             Add(1, 2, x^4) -> Add(x^4, 3)
        MUL: fold the NUMBER subset into a coefficient, keep non-numbers,
             place coefficient first.
             Mul(2, 3) -> 6 ; Mul(3, 3, x) -> Mul(9, x)
        """
        if node.node_type == NodeType.ADD:
            nums = [c for c in node.children if c.node_type == NodeType.NUMBER]
            non_nums = [c for c in node.children if c.node_type != NodeType.NUMBER]
            try:
                total = sum(float(n.value) for n in nums)
            except (ValueError, TypeError):
                return node
            int_val = int(total) if total == int(total) else total
            folded = Num(int_val)
            if not non_nums:
                return folded
            # Keep non-numeric children, append folded constant at end
            node.children = non_nums + [folded]
            node._rebuild_parents()
            return node
        elif node.node_type == NodeType.MUL:
            nums = [c for c in node.children if c.node_type == NodeType.NUMBER]
            non_nums = [c for c in node.children if c.node_type != NodeType.NUMBER]
            try:
                product = 1.0
                for n in nums:
                    product *= float(n.value)
            except (ValueError, TypeError):
                return node
            int_val = int(product) if product == int(product) else product
            folded = Num(int_val)
            if not non_nums:
                return folded
            node.children = [folded] + non_nums  # coefficient first
            node._rebuild_parents()
            return node
        return node

    def _apply_remove_zero(self, node: ASTNode) -> ASTNode:
        """Add(x, 0) → x. Removes all zero children from Add.
        Mul(a, 0, b) → 0. Any zero in Mul absorbs the product."""
        if node.node_type == NodeType.MUL:
            return Num(0)
        # ADD case
        new_children = [c for c in node.children if not self._is_zero(c)]
        if len(new_children) == 0:
            return Num(0)
        if len(new_children) == 1:
            return new_children[0]
        node.children = new_children
        node._rebuild_parents()
        return node

    def _apply_remove_one(self, node: ASTNode) -> ASTNode:
        """Mul(1, x) → x. Pow(x, 1) → x."""
        if node.node_type == NodeType.POW:
            return node.children[0].clone()
        # MUL case
        new_children = [c for c in node.children if not self._is_one(c)]
        if len(new_children) == 0:
            return Num(1)
        if len(new_children) == 1:
            return new_children[0]
        node.children = new_children
        node._rebuild_parents()
        return node

    def _apply_merge_power(self, node: ASTNode) -> ASTNode:
        """Merge all power-like children (POW(x,n) or VARIABLE(x)) into one Pow.
        Mul(x, x, 8) → Mul(8, x^2). Mul(x^2, x^3) → x^5.
        If merged exponent == 1, returns VARIABLE(x) instead of Pow(x,1).
        """
        power_like = self._collect_power_like(node)
        if len(power_like) < 2:
            return node

        power_like_nodes = {id(pl[2]) for pl in power_like}
        non_powers = [c for c in node.children if id(c) not in power_like_nodes]

        base_var = power_like[0][0]
        total_exp = sum(pl[1] for pl in power_like)
        int_exp = int(total_exp) if total_exp == int(total_exp) else total_exp

        if int_exp == 1:
            merged = Var(base_var)
        else:
            merged = Pow(Var(base_var), Num(int_exp))

        new_children = non_powers + [merged]
        if len(new_children) == 1:
            return new_children[0]
        node.children = new_children
        node._rebuild_parents()
        return node

    def _apply_sort_commutative(self, node: ASTNode) -> ASTNode:
        """Sort children of Add/Mul into canonical order."""
        key_fn = self._sort_key_add if node.node_type == NodeType.ADD else self._sort_key_mul
        node.children.sort(key=key_fn)
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

    def _split_coeff_term(self, node: ASTNode) -> Optional[tuple[ASTNode, ASTNode]]:
        """For a 2-child MUL, return (number_child, term_child) regardless of
        operand order. Mul(2, x) and Mul(x, 2) both yield (Num(2), x).
        Returns None if node is not NUMBER * bare_term in either orientation.

        Order-agnostic by design: requiring NUMBER first forced an artificial
        SORT_COMMUTATIVE -> COMBINE_COEFF ordering and skewed the action-sequence
        distribution in generated trajectories."""
        if node.node_type != NodeType.MUL or len(node.children) != 2:
            return None
        a, b = node.children
        if a.node_type == NodeType.NUMBER and self._is_bare_term(b):
            return (a, b)
        if b.node_type == NodeType.NUMBER and self._is_bare_term(a):
            return (b, a)
        return None

    def _coeff_key(self, node: ASTNode) -> Optional[str]:
        """Variable-part key for coefficient extraction.
        Returns None if node is not a recognizable monomial.
        x → 'x', Mul(3,x) → 'x', Mul(x,3) → 'x', Mul(2,Pow(x,2)) → 'x^2'."""
        if self._is_bare_term(node):
            return self._var_key(node)
        split = self._split_coeff_term(node)
        if split is not None:
            return self._var_key(split[1])
        return None

    def _coeff_info(self, node: ASTNode) -> Optional[tuple[str, float, ASTNode]]:
        """Extract (var_key, coefficient, var_part) from a monomial.
        x → ('x', 1, x), Mul(3,x) → ('x', 3, x), Mul(x,3) → ('x', 3, x)."""
        if self._is_bare_term(node):
            return (self._var_key(node), 1.0, node)
        split = self._split_coeff_term(node)
        if split is not None:
            num_child, term_child = split
            try:
                coeff = float(num_child.value)
            except ValueError:
                return None
            return (self._var_key(term_child), coeff, term_child)
        return None

    def _sort_key_add(self, node: ASTNode) -> tuple:
        """Sort key for ADD children.
        Canonical ADD order: high-degree terms first, constants last.
          x^4 term → x^2 term → x term → constant

        Tier 0 (first): polynomial terms, by descending degree (-deg sorts asc).
        Tier 1 (last):  NUMBER constants.
        """
        if node.node_type == NodeType.NUMBER:
            return (1, 0.0, node.value)
        if node.node_type == NodeType.VARIABLE:
            return (0, -1.0, node.value)
        if node.node_type == NodeType.POW:
            try:
                deg = float(node.children[1].value)
            except (ValueError, TypeError, IndexError):
                deg = 0.0
            return (0, -deg, "")
        if node.node_type == NodeType.MUL:
            deg = self._mul_degree(node)
            return (0, -deg, str(node._structural_tuple()))
        return (0, 0.0, str(node._structural_tuple()))

    def _sort_key_mul(self, node: ASTNode) -> tuple:
        """Sort key for MUL children.
        Canonical MUL order: NUMBER coefficient first, then variable parts
        by descending degree.
          (-1 * x)     → NUMBER(-1) before VARIABLE(x)
          (8 * (x)^2)  → NUMBER(8)  before POW(x,2)
          (8 * x * x)  → NUMBER(8)  before VARIABLE, VARIABLE

        Tier 0 (first): NUMBER.
        Tier 1 (rest):  VARIABLE/POW/MUL by descending degree.
        """
        if node.node_type == NodeType.NUMBER:
            return (0, 0.0, node.value)
        if node.node_type == NodeType.VARIABLE:
            return (1, -1.0, node.value)
        if node.node_type == NodeType.POW:
            try:
                deg = float(node.children[1].value)
            except (ValueError, TypeError, IndexError):
                deg = 0.0
            return (1, -deg, "")
        if node.node_type == NodeType.MUL:
            deg = self._mul_degree(node)
            return (1, -deg, str(node._structural_tuple()))
        return (1, 0.0, str(node._structural_tuple()))

    # _sort_key kept as alias for any external callers; prefer _sort_key_add/_sort_key_mul
    def _sort_key(self, node: ASTNode) -> tuple:
        return self._sort_key_add(node)

    def _mul_degree(self, node: ASTNode) -> float:
        """Extract polynomial degree from Mul(coeff, var_part). Returns 0 if unrecognized."""
        if len(node.children) == 2 and node.children[0].node_type == NodeType.NUMBER:
            var_part = node.children[1]
            if var_part.node_type == NodeType.VARIABLE:
                return 1.0
            if (var_part.node_type == NodeType.POW
                    and var_part.children[1].node_type == NodeType.NUMBER):
                try:
                    return float(var_part.children[1].value)
                except (ValueError, TypeError):
                    pass
        return 0.0


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

    # SORT_COMMUTATIVE: canonical ADD = high-degree terms first, constants last.
    # Add(x, 1) is ALREADY sorted (x is tier-0, 1 is tier-1) -> NOT a candidate.
    # Add(1, x) is OUT of order -> SORT_COMMUTATIVE applies, sorts to (x + 1).
    test("Add(1, x) [unsorted]", Add(Num(1), Var()), ["SORT_COMMUTATIVE"])
    _sorted_add = engine.get_candidates(Add(Var(), Num(1)))
    assert not any(a.value == "SORT_COMMUTATIVE" for _, _, a in _sorted_add), \
        "Add(x, 1) is already canonical — SORT_COMMUTATIVE must NOT fire"

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

    # Canonical ADD: terms first, constants last. Add(1, x) -> (x + 1).
    test_apply("sort Add(1, x)", Add(Num(1), Var()),
               0, ActionType.SORT_COMMUTATIVE, "(x + 1)")

    # ── Loop-detection invariant ──────────────────────────────────────
    # SORT_COMMUTATIVE is the ONLY action that can leave
    # canonical_cycle_key() unchanged while changing to_expr() (it merely
    # reorders commutative children). Every other action materially
    # changes the canonical structure. This is a SUPPORTING invariant for
    # order-sensitive loop detection (a future action exhibiting the SORT
    # property would be a secret reorder-noop = bug), NOT the load-bearing
    # reason. The load-bearing reason: engine dynamics (get_candidates,
    # _coeff_key, _sort_key) are functions of the EXACT ordered AST, so
    # the correct loop key is the exact engine state = to_expr().
    # Full empirical scan: scripts/verify_sort_uniqueness.py
    _sort_in  = Add(Num(1), Var())
    _sort_out = engine.apply(_sort_in, 0, ActionType.SORT_COMMUTATIVE)
    assert _sort_in.canonical_cycle_key() == _sort_out.canonical_cycle_key()
    assert _sort_in.to_expr() != _sort_out.to_expr()
    # A representative non-SORT action must change the canonical key.
    _fold_in  = Add(Num(2), Num(3))
    _fold_out = engine.apply(_fold_in, 0, ActionType.FOLD_CONST)
    assert _fold_in.canonical_cycle_key() != _fold_out.canonical_cycle_key()
    _rm_in  = Mul(Num(1), Var())
    _rm_out = engine.apply(_rm_in, 0, ActionType.REMOVE_ONE)
    assert _rm_in.canonical_cycle_key() != _rm_out.canonical_cycle_key()
    print("\nloop-detection invariant (SORT-unique commutative reorder): OK")

    print(f"\n{'='*60}")
    print("ALL TESTS PASSED")