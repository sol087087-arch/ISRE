from dataclasses import dataclass, field
from typing import List, Optional, Iterator, Tuple
from enum import Enum, auto
from copy import deepcopy
import json


class NodeType(Enum):
    ADD = auto()
    MUL = auto()
    POW = auto()
    NUMBER = auto()
    VARIABLE = auto()
    CONST = auto()


@dataclass
class ASTNode:
    node_type: NodeType
    children: List["ASTNode"] = field(default_factory=list)
    value: Optional[str] = None  # "x", "2", "pi" etc.

    # metadata — computed lazily from root
    _depth: int = 0
    _subtree_size: int = 1
    _dirty: bool = True
    # Parent pointer: navigational only, NOT part of tree identity.
    # Excluded from __eq__, __hash__, _structural_tuple, repr.
    _parent: Optional["ASTNode"] = field(default=None, repr=False)

    def __post_init__(self):
        if self.value is None and self.node_type in {
            NodeType.NUMBER,
            NodeType.VARIABLE,
            NodeType.CONST,
        }:
            raise ValueError(f"{self.node_type} requires value")
        # set parent on any children passed to constructor
        for child in self.children:
            child._parent = self

    @property
    def parent(self) -> Optional["ASTNode"]:
        return self._parent

    # ── metadata ──────────────────────────────────────────

    @property
    def depth(self) -> int:
        self._ensure_metadata()
        return self._depth

    @property
    def subtree_size(self) -> int:
        self._ensure_metadata()
        return self._subtree_size

    def _ensure_metadata(self):
        """Call on root to propagate depth/size top-down.
        If called on non-root, depth will be relative to this node (0)."""
        if not self._dirty:
            return
        self._recompute_metadata(current_depth=self._depth)

    def _recompute_metadata(self, current_depth: int = 0):
        self._depth = current_depth
        self._subtree_size = 1
        for child in self.children:
            child._recompute_metadata(current_depth + 1)
            self._subtree_size += child._subtree_size
        self._dirty = False

    def mark_dirty(self):
        """Mark entire subtree as needing metadata recomputation.
        Call on root after any structural transform."""
        self._dirty = True
        for child in self.children:
            child.mark_dirty()

    def _rebuild_parents(self):
        """Rebuild parent pointers top-down. Call on root after
        deserialization (from_dict/from_json) or any structural change."""
        for child in self.children:
            child._parent = self
            child._rebuild_parents()

    def replace_child(self, old: "ASTNode", new: "ASTNode"):
        """Replace a direct child. O(k) where k = number of children.
        Updates parent pointers and marks tree dirty."""
        for i, child in enumerate(self.children):
            if child is old:
                self.children[i] = new
                new._parent = self
                old._parent = None
                self.mark_dirty()
                return
        raise ValueError("old node is not a child of this node")

    # ── traversal ─────────────────────────────────────────

    def iter_preorder(self) -> Iterator["ASTNode"]:
        yield self
        for child in self.children:
            yield from child.iter_preorder()

    def iter_postorder(self) -> Iterator["ASTNode"]:
        for child in self.children:
            yield from child.iter_postorder()
        yield self

    def indexed_nodes(self) -> list["ASTNode"]:
        """Single preorder pass → list indexed by position.
        Use this instead of repeated get_node_by_id."""
        return list(self.iter_preorder())

    def get_node_by_id(self, node_id: int) -> Optional["ASTNode"]:
        nodes = self.indexed_nodes()
        if 0 <= node_id < len(nodes):
            return nodes[node_id]
        return None

    # ── metrics ───────────────────────────────────────────

    def complexity(self) -> int:
        """distance metric from spec: node_count + operator_count"""
        node_count = self.subtree_size
        op_count = sum(
            1
            for n in self.iter_preorder()
            if n.node_type not in {NodeType.NUMBER, NodeType.VARIABLE, NodeType.CONST}
        )
        return node_count + op_count

    # ── equality / hashing (for cycle detection) ──────────

    def _structural_tuple(self) -> tuple:
        """Canonical tuple repr for eq/hash. Hashable, recursive."""
        return (
            self.node_type,
            self.value,
            tuple(c._structural_tuple() for c in self.children),
        )

    def __eq__(self, other) -> bool:
        if not isinstance(other, ASTNode):
            return NotImplemented
        return self._structural_tuple() == other._structural_tuple()

    def __hash__(self) -> int:
        return hash(self._structural_tuple())

    # ── copy ──────────────────────────────────────────────

    def clone(self) -> "ASTNode":
        """Deep copy for trajectory recording.
        Without this, all trajectory steps point to final state."""
        return deepcopy(self)

    # ── display ───────────────────────────────────────────

    def pretty(self) -> str:
        """ASCII tree for debugging / trajectory viewer."""
        lines: list[str] = []
        self._pretty_rec(lines, prefix="", connector="")
        return "\n".join(lines)

    def _pretty_rec(self, lines: list[str], prefix: str, connector: str):
        label = self.node_type.name
        if self.value is not None:
            label += f"({self.value})"
        lines.append(f"{prefix}{connector}{label}")
        # prefix for children: add continuation of current connector
        if connector == "├─ ":
            child_prefix = prefix + "│  "
        elif connector == "└─ ":
            child_prefix = prefix + "   "
        else:
            child_prefix = prefix
        for i, child in enumerate(self.children):
            is_last = i == len(self.children) - 1
            child_connector = "└─ " if is_last else "├─ "
            child._pretty_rec(lines, child_prefix, child_connector)

    def to_expr(self) -> str:
        """Human-readable algebraic expression."""
        if self.node_type == NodeType.NUMBER:
            return self.value
        if self.node_type == NodeType.VARIABLE:
            return self.value
        if self.node_type == NodeType.CONST:
            return self.value
        if self.node_type == NodeType.ADD:
            parts = [c.to_expr() for c in self.children]
            return "(" + " + ".join(parts) + ")"
        if self.node_type == NodeType.MUL:
            parts = [c.to_expr() for c in self.children]
            return "(" + " * ".join(parts) + ")"
        if self.node_type == NodeType.POW:
            base = self.children[0].to_expr()
            exp = self.children[1].to_expr()
            return f"({base})^{exp}"
        return f"??{self.node_type}??"

    # ── serialization ─────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "type": self.node_type.name,
            "value": self.value,
            "children": [c.to_dict() for c in self.children],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ASTNode":
        node = cls(node_type=NodeType[d["type"]], value=d.get("value"))
        node.children = [cls.from_dict(c) for c in d.get("children", [])]
        node._rebuild_parents()
        return node

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, s: str) -> "ASTNode":
        return cls.from_dict(json.loads(s))

    def __repr__(self) -> str:
        return self.to_expr()


# ── convenience constructors ──────────────────────────────

def Num(v: int | float) -> ASTNode:
    return ASTNode(NodeType.NUMBER, value=str(v))

def Var(name: str = "x") -> ASTNode:
    return ASTNode(NodeType.VARIABLE, value=name)

def Add(*children: ASTNode) -> ASTNode:
    return ASTNode(NodeType.ADD, children=list(children))

def Mul(*children: ASTNode) -> ASTNode:
    return ASTNode(NodeType.MUL, children=list(children))

def Pow(base: ASTNode, exp: ASTNode) -> ASTNode:
    return ASTNode(NodeType.POW, children=[base, exp])


# ── quick test ────────────────────────────────────────────

if __name__ == "__main__":
    # (x+1)(x+2)
    expr = Mul(
        Add(Var(), Num(1)),
        Add(Var(), Num(2)),
    )

    print("=== pretty ===")
    print(expr.pretty())
    print()
    print(f"=== expr: {expr} ===")
    print(f"complexity: {expr.complexity()}")
    print(f"root depth: {expr.depth}, subtree_size: {expr.subtree_size}")
    print()

    # clone test
    expr2 = expr.clone()
    assert expr == expr2
    assert expr is not expr2
    print("clone + eq: OK")

    # hash test (for cycle detection)
    seen = {expr}
    assert expr2 in seen
    print("hash cycle detection: OK")

    # serialization roundtrip
    expr3 = ASTNode.from_json(expr.to_json())
    assert expr == expr3
    print("json roundtrip: OK")

    # parent pointer tests
    assert expr.parent is None  # root has no parent
    left_add = expr.children[0]
    assert left_add.parent is expr
    assert left_add.children[0].parent is left_add  # x's parent is Add
    print("parent pointers: OK")

    # parent pointers survive from_json
    assert expr3.children[0].parent is expr3
    assert expr3.children[1].children[0].parent is expr3.children[1]
    print("parent pointers after deserialization: OK")

    # replace_child
    new_node = Num(99)
    expr.children[0].replace_child(expr.children[0].children[1], new_node)
    assert new_node.parent is expr.children[0]
    assert expr.to_expr() == "((x + 99) * (x + 2))"
    print(f"replace_child: OK → {expr}")