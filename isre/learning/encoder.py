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
        # NOTE: no exp_embedding. v1 deliberately does NOT embed POW exponents
        # as a discrete token — POW nodes take the zero-embedding value path
        # and the exponent is carried structurally (Pow has a NUMBER child).
        # A dead nn.Embedding was removed here: it added parameters that would
        # inflate the reported param count in the matched KAN-vs-MLP
        # comparison without ever contributing to the forward pass.
        # (NUM_EXP_VALUES kept as a constant for a future v2 wiring.)

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
        Vectorized Tree-GRU encoder. Public interface.

        Numerically equivalent to _forward_reference (the equivalence oracle):
        in eval mode the two agree to torch.allclose(atol=1e-5, rtol=1e-4) on
        both outputs; the only differences are float non-associativity from
        batched reductions (~1e-6) and, in train mode only, dropout sampling.

        Args:
            ast: root ASTNode

        Returns:
            node_embeddings: [num_nodes, hidden_dim * 2] — concat(local, global) per node
            global_context:  [hidden_dim]
        """
        device = self.device
        H = self.hidden_dim

        # ── Step 0: collect nodes (id scheme MUST match reference) ──────────
        # node_to_id keys by structural value (ASTNode __eq__/__hash__ are
        # structural), exactly as the reference does. Index i corresponds to
        # list(ast.iter_preorder())[i].
        # NOTE: ASTNode __eq__/__hash__ are structural, so node_to_id collapses
        # structurally-equal nodes to a single id (the LAST preorder index seen
        # for that structural key) — identical to the reference. num_nodes is
        # the full postorder/preorder count (matches reference len(postorder));
        # ids that never get written stay zero in both paths.
        postorder: List[ASTNode] = list(ast.iter_postorder())
        node_to_id = {node: i for i, node in enumerate(ast.iter_preorder())}
        num_nodes = len(postorder)

        # ── Step 1: initial embeddings, fully batched ──────────────────────
        # Build per-id arrays. Because node_to_id collapses structurally-equal
        # nodes, we resolve each id back to a representative node the same way
        # the reference's h[nid] = ... last-writer-wins does: it iterates
        # postorder and writes h[node_to_id[node]], so for each id the LAST
        # postorder node mapping to it wins. We mirror that exactly. ids that
        # never appear (collapsed away) keep rep[i] = None -> zero rows, which
        # also matches the reference (those h rows are never written).

        # Representative node per id following reference postorder last-write.
        rep: List[Optional[ASTNode]] = [None] * num_nodes
        for node in postorder:
            rep[node_to_id[node]] = node

        # Only ids that actually appear get a Step-1 embedding; the rest stay
        # zero (matches the reference, where unwritten h rows remain zero).
        active_ids = [i for i in range(num_nodes) if rep[i] is not None]
        n_active = len(active_ids)
        t_active = torch.tensor(active_ids, device=device, dtype=torch.long)

        type_idx = torch.tensor(
            [self._type_index(rep[i].node_type) for i in active_ids],
            device=device, dtype=torch.long,
        )
        type_mat = self.type_embedding(type_idx)  # [n_active, H]

        # Value embedding matrix by category, same index math as _value_embedding.
        val_mat = torch.zeros(n_active, H, device=device)
        num_rows, num_idx_list = [], []
        var_rows, var_idx_list = [], []
        for row, i in enumerate(active_ids):
            n = rep[i]
            if n.node_type == NodeType.NUMBER:
                num_rows.append(row)
                num_idx_list.append(self._coeff_index(n))
            elif n.node_type == NodeType.VARIABLE:
                var_rows.append(row)
                var_idx_list.append(self._var_index(n))
            # CONST / operators -> zeros (already set)
        if num_rows:
            val_mat[torch.tensor(num_rows, device=device)] = self.coeff_embedding(
                torch.tensor(num_idx_list, device=device, dtype=torch.long)
            )
        if var_rows:
            val_mat[torch.tensor(var_rows, device=device)] = self.var_embedding(
                torch.tensor(var_idx_list, device=device, dtype=torch.long)
            )

        h = torch.zeros(num_nodes, H, device=device)
        h[t_active] = self.init_proj(torch.cat([type_mat, val_mat], dim=1))

        # ── Step 2 precompute: category groups + child structure ───────────
        # Categories are fixed across rounds (node types don't change).
        #
        # The reference loops postorder writing h_new[node_to_id[node]]; with
        # structural id collapse multiple postorder nodes can share an id and
        # the LAST one wins. rep[id] already holds that last-postorder node,
        # so we key every group by UNIQUE id and use rep[id] for both the
        # category decision and its children — this exactly reproduces the
        # reference's last-writer-wins net result per id.
        leaf_ids: List[int] = []          # no children
        comm_ids: List[int] = []          # ADD/MUL with children
        pow_ids: List[int] = []           # POW with exactly 2 children
        fb_ids: List[int] = []            # fallback: other with children
        pow_left: List[int] = []
        pow_right: List[int] = []

        for i in active_ids:
            node = rep[i]
            children_ids = [node_to_id[c] for c in node.children]
            if not children_ids:
                leaf_ids.append(i)
            elif node.node_type in (NodeType.ADD, NodeType.MUL):
                comm_ids.append(i)
            elif node.node_type == NodeType.POW and len(children_ids) == 2:
                pow_ids.append(i)
                pow_left.append(children_ids[0])
                pow_right.append(children_ids[1])
            else:
                fb_ids.append(i)

        # Ragged child segments for commutative pool (mean+max).
        seg_parent_rows: List[int] = []
        seg_child_global: List[int] = []
        for row, nid in enumerate(comm_ids):
            for c in rep[nid].children:
                seg_parent_rows.append(row)
                seg_child_global.append(node_to_id[c])
        n_comm = len(comm_ids)

        # Ragged child segments for fallback pool (mean only).
        fb_seg_parent_rows: List[int] = []
        fb_seg_child_global: List[int] = []
        for row, nid in enumerate(fb_ids):
            for c in rep[nid].children:
                fb_seg_parent_rows.append(row)
                fb_seg_child_global.append(node_to_id[c])
        n_fb = len(fb_ids)

        # Tensors for scatter/index ops (built once).
        t_leaf = torch.tensor(leaf_ids, device=device, dtype=torch.long) if leaf_ids else None
        t_comm = torch.tensor(comm_ids, device=device, dtype=torch.long) if comm_ids else None
        t_pow = torch.tensor(pow_ids, device=device, dtype=torch.long) if pow_ids else None
        t_fb = torch.tensor(fb_ids, device=device, dtype=torch.long) if fb_ids else None
        t_pow_l = torch.tensor(pow_left, device=device, dtype=torch.long) if pow_ids else None
        t_pow_r = torch.tensor(pow_right, device=device, dtype=torch.long) if pow_ids else None

        if seg_child_global:
            t_seg_child = torch.tensor(seg_child_global, device=device, dtype=torch.long)
            t_seg_parent = torch.tensor(seg_parent_rows, device=device, dtype=torch.long)
            comm_counts = torch.zeros(n_comm, 1, device=device)
            comm_counts.index_add_(
                0, t_seg_parent, torch.ones(len(seg_parent_rows), 1, device=device)
            )
        else:
            t_seg_child = t_seg_parent = comm_counts = None

        if fb_seg_child_global:
            t_fb_child = torch.tensor(fb_seg_child_global, device=device, dtype=torch.long)
            t_fb_parent = torch.tensor(fb_seg_parent_rows, device=device, dtype=torch.long)
            fb_counts = torch.zeros(n_fb, 1, device=device)
            fb_counts.index_add_(
                0, t_fb_parent, torch.ones(len(fb_seg_parent_rows), 1, device=device)
            )
        else:
            t_fb_child = t_fb_parent = fb_counts = None

        # ── Step 2: multi-round bottom-up message passing (batched) ────────
        for _ in range(self.num_rounds):
            h_new = h.clone()

            # Leaf category: zero aggregation input, gru_leaf.
            if t_leaf is not None:
                agg_leaf = torch.zeros(t_leaf.shape[0], H, device=device)
                h_prev = h.index_select(0, t_leaf)
                upd = self.gru_leaf(agg_leaf, h_prev)
                upd = upd + h_prev
                upd = self.layer_norm(upd)
                upd = self.dropout(upd)
                h_new[t_leaf] = upd

            # Commutative category: mean + max pool over ragged children.
            if t_comm is not None:
                child_h = h.index_select(0, t_seg_child)  # [E, H]
                mean_sum = torch.zeros(n_comm, H, device=device)
                mean_sum.index_add_(0, t_seg_parent, child_h)
                mean_pool = mean_sum / comm_counts
                # segment max (every commutative node has >=2 children, so
                # every row receives >=1 value; -inf init is always overwritten)
                max_pool = torch.full((n_comm, H), float("-inf"), device=device)
                idx_exp = t_seg_parent.unsqueeze(1).expand(-1, H)
                max_pool = max_pool.scatter_reduce(
                    0, idx_exp, child_h, reduce="amax", include_self=True
                )
                agg = self.agg_commutative(torch.cat([mean_pool, max_pool], dim=1))
                h_prev = h.index_select(0, t_comm)
                upd = self.gru_commutative(agg, h_prev)
                upd = upd + h_prev
                upd = self.layer_norm(upd)
                upd = self.dropout(upd)
                h_new[t_comm] = upd

            # POW category: concat(left, right), order preserved.
            if t_pow is not None:
                left = h.index_select(0, t_pow_l)
                right = h.index_select(0, t_pow_r)
                agg = self.agg_binary(torch.cat([left, right], dim=1))
                h_prev = h.index_select(0, t_pow)
                upd = self.gru_binary(agg, h_prev)
                upd = upd + h_prev
                upd = self.layer_norm(upd)
                upd = self.dropout(upd)
                h_new[t_pow] = upd

            # Fallback category: mean pool only, gru_commutative.
            if t_fb is not None:
                child_h = h.index_select(0, t_fb_child)
                mean_sum = torch.zeros(n_fb, H, device=device)
                mean_sum.index_add_(0, t_fb_parent, child_h)
                agg = mean_sum / fb_counts
                h_prev = h.index_select(0, t_fb)
                upd = self.gru_commutative(agg, h_prev)
                upd = upd + h_prev
                upd = self.layer_norm(upd)
                upd = self.dropout(upd)
                h_new[t_fb] = upd

            h = h_new

        # ── Step 3: global context from root ───────────────────────────────
        root_id = node_to_id[ast]
        global_context = self.global_proj(h[root_id])

        # ── Step 4: broadcast — concat(local, global) per node ─────────────
        global_expanded = global_context.unsqueeze(0).expand(num_nodes, -1)
        node_embeddings = torch.cat([h, global_expanded], dim=1)

        return node_embeddings, global_context

    def _forward_reference(self, ast: ASTNode) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Golden per-node reference. Kept as the equivalence oracle for the
        vectorized forward. Do not modify.

        Args:
            ast: root ASTNode

        Returns:
            node_embeddings: [num_nodes, hidden_dim * 2] — concat(local, global) per node
            global_context:  [hidden_dim]
        """
        device = self.device

        # ── Step 0: collect nodes ─────────────────────────────────
        # node_to_id assigns unique stable IDs (preorder indices) to each node.
        # Traversal for message passing is postorder (leaves before parents) — correct
        # for bottom-up GRU. The ID scheme (preorder) is independent of traversal order;
        # lookup is by object identity, so IDs are consistent across rounds.
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
        # TODO(perf): h_new = h.clone() allocates O(N×rounds) tensors.
        #   Replace with double-buffer: pre-allocate h_new once, use h_new.copy_(h)
        #   at start of each round. Minor at current scale (<200 nodes/tree).
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

    @staticmethod
    def _coeff_index(node: ASTNode) -> int:
        """Coefficient embedding index for a NUMBER node.
        Pure extraction of the NUMBER index math from _value_embedding —
        behavior-identical, shared by reference and vectorized paths."""
        try:
            val = int(float(node.value))
            return max(0, min(NUM_COEFF_VALUES - 1, val - COEFF_MIN))
        except (ValueError, TypeError):
            return NUM_COEFF_VALUES // 2  # map unknown to 0

    @staticmethod
    def _var_index(node: ASTNode) -> int:
        """Variable embedding index for a VARIABLE node.
        Pure extraction of the VARIABLE index math from _value_embedding —
        behavior-identical, shared by reference and vectorized paths."""
        var_map = {"x": 0, "y": 1, "z": 2}
        return var_map.get(node.value, 3)

    def _value_embedding(self, node: ASTNode, device: torch.device) -> torch.Tensor:
        """Discrete value embedding for leaf nodes. Zero for operators."""
        if node.node_type == NodeType.NUMBER:
            idx = self._coeff_index(node)
            return self.coeff_embedding(torch.tensor(idx, device=device))

        if node.node_type == NodeType.VARIABLE:
            idx = self._var_index(node)
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
    # 10 nodes: Add, Mul, Num(3), Pow, Var(x), Num(2)=exp, Mul, Num(2)=coeff, Var(x), Num(-5)
    assert emb2.shape[0] == 10, f"expected 10 nodes, got {emb2.shape[0]}"
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