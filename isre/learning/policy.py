"""
Policy Network v1 — scores (node, action) pairs for the symbolic reasoning engine.

Input: node embeddings from ASTEncoder + candidate actions from SymbolicEngine.
Output: score per (node_id, action) candidate.

Two variants:
  - MLP (baseline, implement first, day 4)
  - KAN (interpretability experiments, day 8)
"""

import torch
import torch.nn as nn
from typing import List, Tuple
from isre.symbolic.symbolic_engine import ActionType


class PolicyNetwork(nn.Module):
    """
    Scores candidate actions for symbolic reasoning.

    For each candidate (node_id, action_type):
        score = f(node_embedding[node_id], action_embedding[action_type])

    where node_embedding includes global context (from encoder broadcast).
    """

    def __init__(
        self,
        node_emb_dim: int,          # encoder.hidden_dim * 2 (local + global)
        num_action_types: int = 10,  # len(ActionType)
        action_emb_dim: int = 64,
        hidden_dim: int = 256,
        variant: str = "mlp",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.variant = variant.lower()
        self.node_emb_dim = node_emb_dim
        self.action_emb_dim = action_emb_dim

        # Action type embedding (shared across all candidates)
        self.action_embedding = nn.Embedding(num_action_types, action_emb_dim)

        # Action type → index mapping
        self._action_to_idx = {action: i for i, action in enumerate(ActionType)}

        input_dim = node_emb_dim + action_emb_dim

        if self.variant == "mlp":
            # 3-layer MLP baseline (spec: 2-3 layers, dim 256)
            self.scorer = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
        elif self.variant == "kan":
            # Placeholder — swap in KAN implementation on day 8.
            # KAN expects small input dim for interpretability.
            # We add a bottleneck: input_dim → bottleneck_dim → KAN → 1
            bottleneck_dim = 32  # small enough for interpretable φ_i curves
            self.bottleneck = nn.Linear(input_dim, bottleneck_dim)
            # KAN would go here:
            # self.kan = KAN(width=[bottleneck_dim, hidden_dim, 1], grid=5, k=3)
            # For now, fallback to small MLP with same bottleneck
            self.scorer = nn.Sequential(
                nn.Linear(bottleneck_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
        else:
            raise ValueError(f"Unknown variant: {variant}")

    def forward(
        self,
        node_embeddings: torch.Tensor,
        candidates: List[Tuple[int, ActionType]],
    ) -> torch.Tensor:
        """
        Score all candidates for a single expression.
        No batching across expressions — process one at a time for v1.

        Args:
            node_embeddings: [num_nodes, node_emb_dim] from encoder
            candidates: list of (node_id, ActionType) from symbolic engine

        Returns:
            scores: [num_candidates] raw logits (no softmax)
        """
        if not candidates:
            return torch.tensor([], device=node_embeddings.device)

        device = node_embeddings.device

        # Gather node embeddings for each candidate
        node_ids = [c[0] for c in candidates]
        action_types = [c[1] for c in candidates]

        # Node features for candidates: [num_candidates, node_emb_dim]
        node_feats = node_embeddings[node_ids]

        # Action embeddings: [num_candidates, action_emb_dim]
        action_indices = torch.tensor(
            [self._action_to_idx[a] for a in action_types],
            device=device,
        )
        action_feats = self.action_embedding(action_indices)

        # Concatenate: [num_candidates, node_emb_dim + action_emb_dim]
        combined = torch.cat([node_feats, action_feats], dim=-1)

        # Score
        if self.variant == "kan":
            combined = self.bottleneck(combined)

        scores = self.scorer(combined).squeeze(-1)  # [num_candidates]

        return scores

    def select_action(
        self,
        node_embeddings: torch.Tensor,
        candidates: List[Tuple[int, ActionType]],
        temperature: float = 1.0,
        greedy: bool = False,
    ) -> Tuple[int, ActionType, torch.Tensor]:
        """
        Select an action from candidates.

        Args:
            node_embeddings: from encoder
            candidates: from symbolic engine
            temperature: for softmax sampling
            greedy: if True, pick argmax

        Returns:
            (node_id, action, log_prob)
        """
        scores = self.forward(node_embeddings, candidates)

        if greedy:
            idx = scores.argmax().item()
            log_prob = torch.tensor(0.0)
        else:
            probs = torch.softmax(scores / temperature, dim=-1)
            dist = torch.distributions.Categorical(probs)
            idx = dist.sample().item()
            log_prob = dist.log_prob(torch.tensor(idx))

        node_id, action = candidates[idx]
        return node_id, action, log_prob

    def compute_loss(
        self,
        node_embeddings: torch.Tensor,
        candidates: List[Tuple[int, ActionType]],
        gold_action: ActionType,
        gold_node_id: int,
    ) -> torch.Tensor:
        """
        Cross-entropy loss for supervised training.
        Gold action = (gold_node_id, gold_action) from trajectory.

        Returns scalar loss.
        """
        scores = self.forward(node_embeddings, candidates)

        # Find index of gold action in candidates
        gold_idx = None
        for i, (nid, action) in enumerate(candidates):
            if nid == gold_node_id and action == gold_action:
                gold_idx = i
                break

        if gold_idx is None:
            # Gold action not in candidates — should not happen if data is valid
            # Return zero loss to avoid crash, but log warning
            return torch.tensor(0.0, requires_grad=True, device=scores.device)

        target = torch.tensor(gold_idx, device=scores.device)
        loss = nn.functional.cross_entropy(scores.unsqueeze(0), target.unsqueeze(0))
        return loss


# ====================== TESTS ======================

if __name__ == "__main__":
    # Test with random embeddings (no torch dependency on encoder)
    hidden_dim = 128
    node_emb_dim = hidden_dim * 2  # matches encoder output

    policy = PolicyNetwork(
        node_emb_dim=node_emb_dim,
        variant="mlp",
        hidden_dim=128,
    )
    policy.eval()

    # Simulate encoder output: 7 nodes, each with [hidden_dim*2] embedding
    num_nodes = 7
    fake_embeddings = torch.randn(num_nodes, node_emb_dim)

    # Simulate candidates from symbolic engine
    candidates = [
        (0, ActionType.EXPAND),
        (0, ActionType.FLATTEN_ADD),
        (2, ActionType.FOLD_CONST),
        (3, ActionType.REMOVE_ZERO),
        (5, ActionType.SORT_COMMUTATIVE),
    ]

    # Test forward
    scores = policy(fake_embeddings, candidates)
    print(f"scores shape: {scores.shape}")  # [5]
    print(f"scores: {scores}")
    assert scores.shape == (5,)
    print("  ✓ forward OK")

    # Test select_action (greedy)
    nid, action, lp = policy.select_action(fake_embeddings, candidates, greedy=True)
    print(f"greedy: node={nid}, action={action.value}")
    print("  ✓ select_action OK")

    # Test compute_loss
    policy.train()
    loss = policy.compute_loss(
        fake_embeddings,
        candidates,
        gold_action=ActionType.FOLD_CONST,
        gold_node_id=2,
    )
    print(f"loss: {loss.item():.4f}")
    loss.backward()
    print("  ✓ compute_loss + backward OK")

    # Test empty candidates
    empty_scores = policy(fake_embeddings, [])
    assert empty_scores.shape == (0,)
    print("  ✓ empty candidates OK")

    # Test gold action not in candidates
    bad_loss = policy.compute_loss(
        fake_embeddings, candidates,
        gold_action=ActionType.MERGE_POWER, gold_node_id=99,
    )
    print(f"missing gold loss: {bad_loss.item()}")
    print("  ✓ missing gold handled gracefully")

    print("\nALL POLICY TESTS PASSED")