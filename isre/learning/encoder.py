"""
AST Encoder v1 — operator-conditioned Tree GRU with bottom-up aggregation.

Architecture (from spec):
    AST leaf nodes → node embeddings → N rounds of bottom-up GRU aggregation
    → root embedding = global context → broadcast concat(h_node, h_root) per node

Output: tensor [num_nodes, hidden_dim * 2] for policy network input.
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional
from isre.symbolic.isre_ast import ASTNode, NodeType


# Coefficient range from spec: [-9, 9]. We embed discrete values.
COEFF_MIN = -9
COEFF_MAX = 9
NUM_COEFF_VALUES = COEFF_MAX - COEFF_MIN + 1  # 19 values
# Exponent range from spec: [0, 4]. Also discrete.
EXP_MAX = 4
NUM_EXP_VALUES = EXP_MAX + 1  # 5 values: 0,1,2,3,4


class ASTEncoder(nn.Module):
    """
    Tree GRU encoder with operator-conditioned aggregation.

    Output: (node_embeddings, global_context)
        node_embeddings: [num_nodes, hidden_dim * 2]  (local + global for each node)
        global_context:  [hidden_dim]

    Multi-round message passing: each round = one bottom-up pass over the tree.
    Information propagates one tree level per round.
    N rounds covers trees of depth N.
    """

    def __init__(
        self,
        num_node_types: int = 6,
        hidden_dim: int = 256,
        num_rounds: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_rounds = num_rounds

        # ── Node embeddings ──────────────────────────────
        self.type_embedding = nn.Embedding(num_node_types, hidden_dim)

        # Discrete value embeddings (not continuous projection)
        self.coeff_embedding = nn.Embedding(NUM_COEFF_VALUES, hidden_dim)  # for NUMBER nodes
        self.exp_embedding = nn.Embedding(NUM_EXP_VALUES, hidden_dim)      # reusable if needed

        # Variable name embedding (v1: single variable x, but extensible)
        self.var_embedding = nn.Embedding(4, hidden_dim)  # x, y, z, other

        # Projection: combine type + value embeddings → hidden_dim
        self.init_proj = nn.Linear(hidden_dim * 2, hidden_dim)

        # ── Operator-aware aggregation ───────────────────
        # Commutative (Add, Mul): mean + max pooling → project
        self.agg_commutative = nn.Linear(hidden_dim * 2, hidden_dim)
        # Non-commutative binary (Pow): concat(left, right) → project
        self.agg_binary = nn.Linear(hidden_dim * 2, hidden_dim)

        # ── Operator-conditioned GRU cells ───────────────
        # Different weights per operator category (spec: W_op[type(node)])
        self.gru_commutative = nn.GRUCell(hidden_dim, hidden_dim)
        self.gru_binary = nn.GRUCell(hidden_dim, hidden_dim)
        self.gru_leaf = nn.GRUCell(hidden_dim, hidden_dim)

        # ── Normalization and regularization ──────────────
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # ── Global context projection ────────────────────
        self.global_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, ast: ASTNode) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            ast: root ASTNode

        Returns:
            node_embeddings: [num_nodes, hidden_dim * 2] — concat(local, global) per node
            global_context:  [hidden_dim]
        """
        device = self.device

        # ── Step 0: collect nodes in post-order (leaves first) ────
        postorder: List[ASTNode] = list(ast.iter_postorder())
        node_to_id = {node: i for i, node in enumerate(ast.iter_preorder())}
        num_nodes = len(postorder)

        # ── Step 1: initial embeddings for all nodes ──────────────
        h = torch.zeros(num_nodes, self.hidden_dim, device=device)

        for node in postorder:
            nid = node_to_id[node]
            type_idx = self._type_index(node.node_type)
            type_emb = self.type_embedding(torch.tensor(type_idx, device=device))
            val_emb = self._value_embedding(node, device)
            init = self.init_proj(torch.cat([type_emb, val_emb]))
            h[nid] = init

        # ── Step 2: multi-round bottom-up message passing ─────────
        # Each round: propagate information one level up the tree.
        # After N rounds, root sees information from depth N.
        for round_idx in range(self.num_rounds):
            h_new = h.clone()

            for node in postorder:
                nid = node_to_id[node]
                children_ids = [node_to_id[c] for c in node.children]

                if not children_ids:
                    # Leaf: GRU with zero aggregation input
                    agg = torch.zeros(self.hidden_dim, device=device)
                    gru = self.gru_leaf
                elif node.node_type in (NodeType.ADD, NodeType.MUL):
                    # Commutative: mean + max pooling
                    child_h = h[children_ids]  # [num_children, hidden_dim]
                    mean_pool = child_h.mean(dim=0)
                    max_pool = child_h.max(dim=0).values
                    agg = self.agg_commutative(torch.cat([mean_pool, max_pool]))
                    gru = self.gru_commutative
                elif node.node_type == NodeType.POW and len(children_ids) == 2:
                    # Non-commutative binary: concat(left, right)
                    agg = self.agg_binary(torch.cat([h[children_ids[0]], h[children_ids[1]]]))
                    gru = self.gru_binary
                else:
                    # Fallback: mean pooling + commutative GRU
                    child_h = h[children_ids]
                    agg = child_h.mean(dim=0)
                    gru = self.gru_commutative

                # GRU update: agg = input, h[nid] = previous hidden state
                updated = gru(agg.unsqueeze(0), h[nid].unsqueeze(0)).squeeze(0)

                # Residual connection (spec: h_out = GRU(...) + h_in)
                updated = updated + h[nid]

                # Normalize and dropout
                updated = self.layer_norm(updated)
                updated = self.dropout(updated)

                h_new[nid] = updated

            h = h_new

        # ── Step 3: global context from root ──────────────────────
        root_id = node_to_id[ast]
        global_context = self.global_proj(h[root_id])

        # ── Step 4: broadcast — concat(local, global) per node ────
        global_expanded = global_context.unsqueeze(0).expand(num_nodes, -1)
        node_embeddings = torch.cat([h, global_expanded], dim=1)

        return node_embeddings, global_context

    # ── Helpers ────────────────────────────────────────────────

    def _value_embedding(self, node: ASTNode, device: torch.device) -> torch.Tensor:
        """Discrete value embedding for leaf nodes. Zero for operators."""
        if node.node_type == NodeType.NUMBER:
            try:
                val = int(float(node.value))
                idx = max(0, min(NUM_COEFF_VALUES - 1, val - COEFF_MIN))
            except (ValueError, TypeError):
                idx = NUM_COEFF_VALUES // 2  # map unknown to 0
            return self.coeff_embedding(torch.tensor(idx, device=device))

        if node.node_type == NodeType.VARIABLE:
            var_map = {"x": 0, "y": 1, "z": 2}
            idx = var_map.get(node.value, 3)
            return self.var_embedding(torch.tensor(idx, device=device))

        if node.node_type == NodeType.CONST:
            # Constants like pi, e — use zero embedding for v1
            return torch.zeros(self.hidden_dim, device=device)

        # Operators: no value, zero embedding
        return torch.zeros(self.hidden_dim, device=device)

    def _type_index(self, node_type: NodeType) -> int:
        mapping = {
            NodeType.ADD: 0,
            NodeType.MUL: 1,
            NodeType.POW: 2,
            NodeType.NUMBER: 3,
            NodeType.VARIABLE: 4,
            NodeType.CONST: 5,
        }
        return mapping.get(node_type, 0)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device


# ====================== TESTS ======================

if __name__ == "__main__":
    from isre.symbolic.isre_ast import Num, Var, Add, Mul, Pow

    encoder = ASTEncoder(hidden_dim=128, num_rounds=4)
    encoder.eval()

    # Test 1: simple expression (x + 1)
    expr1 = Add(Var(), Num(1))
    emb1, ctx1 = encoder(expr1)
    print(f"(x + 1): nodes={emb1.shape[0]}, emb_dim={emb1.shape[1]}, global={ctx1.shape}")
    assert emb1.shape == (3, 256), f"Expected (3, 256), got {emb1.shape}"
    assert ctx1.shape == (128,), f"Expected (128,), got {ctx1.shape}"
    print("  ✓ shapes OK")

    # Test 2: deeper expression 3x^2 + 2x - 5
    expr2 = Add(
        Mul(Num(3), Pow(Var(), Num(2))),
        Mul(Num(2), Var()),
        Num(-5),
    )
    emb2, ctx2 = encoder(expr2)
    print(f"3x²+2x-5: nodes={emb2.shape[0]}, emb_dim={emb2.shape[1]}")
    assert emb2.shape[0] == 9  # Add, Mul, 3, Pow, x, 2, Mul, 2, x, -5... count manually
    print("  ✓ shapes OK")

    # Test 3: gradient flow
    encoder.train()
    emb3, ctx3 = encoder(expr2)
    loss = emb3.sum()
    loss.backward()
    grad_norms = {name: p.grad.norm().item() for name, p in encoder.named_parameters() if p.grad is not None}
    print(f"  gradient norms: {len(grad_norms)} params with gradients")
    assert len(grad_norms) > 0, "No gradients!"
    print("  ✓ gradients flow")

    # Test 4: deterministic output
    encoder.eval()
    emb4a, _ = encoder(expr2)
    emb4b, _ = encoder(expr2)
    assert torch.allclose(emb4a, emb4b), "Non-deterministic in eval mode!"
    print("  ✓ deterministic in eval mode")

    # Test 5: different trees produce different embeddings
    expr_a = Add(Var(), Num(1))
    expr_b = Add(Var(), Num(2))
    emb_a, _ = encoder(expr_a)
    emb_b, _ = encoder(expr_b)
    assert not torch.allclose(emb_a, emb_b), "Different trees should produce different embeddings!"
    print("  ✓ different trees → different embeddings")

    print("\nALL ENCODER TESTS PASSED")