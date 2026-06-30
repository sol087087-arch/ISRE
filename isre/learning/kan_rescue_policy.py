"""
KAN rescue policy variants.

These are append-only experimental arms. They do NOT replace the locked
raw-feature KAN in kan_policy.py.

Variants:
  - ActionEmbeddingKANPolicy:
      candidate_features(state, node, action) + learned action embedding
      -> efficient_kan KAN -> score.

  - EncoderKANPolicy:
      ASTEncoder node embedding + learned action embedding
      -> linear bottleneck -> efficient_kan KAN -> score.

  - EncoderDeepKANPolicy:
      Same representation path as EncoderKANPolicy, but the efficient_kan
      scorer has multiple KAN hidden layers.

The point is to separate fairness regimes:
  raw KAN       = interpretable feature-only baseline.
  kan_ae        = feature-only baseline plus the action embedding that MLP has.
  encoder_kan   = same learned AST representation family as MLP, KAN scorer.
  encoder_deep_kan = depth rescue: same inputs, deeper KAN scorer.
"""

from typing import List, Tuple

import torch
import torch.nn as nn

from isre.symbolic.isre_ast import ASTNode
from isre.symbolic.symbolic_engine import ActionType
from isre.learning.features import candidate_features, FEATURE_DIM


_ACTION_ORDER = sorted(ActionType, key=lambda a: a.value)
_ACTION_TO_IDX = {a: i for i, a in enumerate(_ACTION_ORDER)}


class ActionEmbeddingKANPolicy(nn.Module):
    """KAN over hand-crafted candidate features plus learned action embedding."""

    def __init__(
        self,
        hidden: int = 16,
        action_emb_dim: int = 8,
        grid: int = 5,
        k: int = 3,
        seed: int = 0,
        device: str = "cpu",
    ):
        super().__init__()
        from efficient_kan import KAN

        torch.manual_seed(seed)
        self.feature_dim = FEATURE_DIM
        self.action_emb_dim = action_emb_dim
        self.input_dim = FEATURE_DIM + action_emb_dim
        self.hidden = hidden
        self._device = device

        self.action_embedding = nn.Embedding(len(_ACTION_ORDER), action_emb_dim)
        self.kan = KAN(
            [self.input_dim, hidden, 1],
            grid_size=grid,
            spline_order=k,
        )
        self.to(device)

    def _feature_matrix(
        self,
        state_root: ASTNode,
        candidates: List[Tuple[int, ActionType]],
    ) -> torch.Tensor:
        state_root.mark_dirty()
        state_root._ensure_metadata()
        rows = [
            candidate_features(state_root, nid, act)
            for (nid, act) in candidates
        ]
        return torch.tensor(rows, dtype=torch.float32, device=self._device)

    def score(
        self,
        state_root: ASTNode,
        candidates: List[Tuple[int, ActionType]],
    ) -> torch.Tensor:
        if not candidates:
            return torch.tensor([], device=self._device)

        feats = self._feature_matrix(state_root, candidates)
        action_indices = torch.tensor(
            [_ACTION_TO_IDX[action] for _, action in candidates],
            device=self._device,
        )
        action_feats = self.action_embedding(action_indices)
        x = torch.cat([feats, action_feats], dim=-1)
        return self.kan(x).squeeze(-1)

    def forward(self, state_root, candidates):
        return self.score(state_root, candidates)

    def select_action(
        self,
        state_root: ASTNode,
        candidates: List[Tuple[int, ActionType]],
        temperature: float = 1.0,
        greedy: bool = False,
    ) -> Tuple[int, ActionType, torch.Tensor]:
        scores = self.score(state_root, candidates)
        log_probs_all = torch.log_softmax(scores / temperature, dim=-1)
        if greedy:
            idx = int(scores.argmax().item())
        else:
            dist = torch.distributions.Categorical(logits=scores / temperature)
            idx = int(dist.sample().item())
        node_id, action = candidates[idx]
        return node_id, action, log_probs_all[idx]

    def compute_loss(
        self,
        state_root: ASTNode,
        candidates: List[Tuple[int, ActionType]],
        gold_action: ActionType,
        gold_node_id: int,
    ) -> torch.Tensor:
        scores = self.score(state_root, candidates)
        gold_idx = None
        for i, (nid, action) in enumerate(candidates):
            if nid == gold_node_id and action == gold_action:
                gold_idx = i
                break
        if gold_idx is None:
            cand_str = [(nid, a.value) for nid, a in candidates]
            raise ValueError(
                f"Gold action ({gold_node_id}, {gold_action.value}) not found "
                f"in candidates: {cand_str}. Check trajectory validity."
            )
        target = torch.tensor(gold_idx, device=scores.device)
        return nn.functional.cross_entropy(scores.unsqueeze(0), target.unsqueeze(0))


class EncoderKANPolicy(nn.Module):
    """KAN scorer over ASTEncoder node embeddings plus learned action embedding."""

    def __init__(
        self,
        node_emb_dim: int,
        hidden: int = 16,
        action_emb_dim: int = 8,
        bottleneck_dim: int = 16,
        grid: int = 5,
        k: int = 3,
        seed: int = 0,
    ):
        super().__init__()
        from efficient_kan import KAN

        torch.manual_seed(seed)
        self.node_emb_dim = node_emb_dim
        self.action_emb_dim = action_emb_dim
        self.bottleneck_dim = bottleneck_dim
        self.hidden = hidden

        self.action_embedding = nn.Embedding(len(_ACTION_ORDER), action_emb_dim)
        self.bottleneck = nn.Linear(node_emb_dim + action_emb_dim, bottleneck_dim)
        self.kan = KAN(
            [bottleneck_dim, hidden, 1],
            grid_size=grid,
            spline_order=k,
        )

    def forward(
        self,
        node_embeddings: torch.Tensor,
        candidates: List[Tuple[int, ActionType]],
    ) -> torch.Tensor:
        if not candidates:
            return torch.tensor([], device=node_embeddings.device)

        device = node_embeddings.device
        node_ids = [node_id for node_id, _ in candidates]
        action_indices = torch.tensor(
            [_ACTION_TO_IDX[action] for _, action in candidates],
            device=device,
        )
        node_feats = node_embeddings[node_ids]
        action_feats = self.action_embedding(action_indices)
        x = torch.cat([node_feats, action_feats], dim=-1)
        x = self.bottleneck(x)
        return self.kan(x).squeeze(-1)

    def select_action(
        self,
        node_embeddings: torch.Tensor,
        candidates: List[Tuple[int, ActionType]],
        temperature: float = 1.0,
        greedy: bool = False,
    ) -> Tuple[int, ActionType, torch.Tensor]:
        scores = self.forward(node_embeddings, candidates)
        log_probs_all = torch.log_softmax(scores / temperature, dim=-1)
        if greedy:
            idx = int(scores.argmax().item())
        else:
            dist = torch.distributions.Categorical(logits=scores / temperature)
            idx = int(dist.sample().item())
        node_id, action = candidates[idx]
        return node_id, action, log_probs_all[idx]

    def compute_loss(
        self,
        node_embeddings: torch.Tensor,
        candidates: List[Tuple[int, ActionType]],
        gold_action: ActionType,
        gold_node_id: int,
    ) -> torch.Tensor:
        scores = self.forward(node_embeddings, candidates)
        gold_idx = None
        for i, (nid, action) in enumerate(candidates):
            if nid == gold_node_id and action == gold_action:
                gold_idx = i
                break
        if gold_idx is None:
            cand_str = [(nid, a.value) for nid, a in candidates]
            raise ValueError(
                f"Gold action ({gold_node_id}, {gold_action.value}) not found "
                f"in candidates: {cand_str}. Check trajectory validity."
            )
        target = torch.tensor(gold_idx, device=scores.device)
        return nn.functional.cross_entropy(scores.unsqueeze(0), target.unsqueeze(0))


class EncoderDeepKANPolicy(EncoderKANPolicy):
    """Encoder-KAN with a deeper efficient_kan scorer.

    This is an append-only depth probe. The input representation is identical to
    EncoderKANPolicy; only the KAN scorer changes from [B,H,1] to
    [B,H,...,H,1]. This keeps the experiment about depth rather than changing
    the encoder or candidate features.
    """

    def __init__(
        self,
        node_emb_dim: int,
        hidden: int = 16,
        action_emb_dim: int = 8,
        bottleneck_dim: int = 16,
        depth: int = 2,
        grid: int = 5,
        k: int = 3,
        seed: int = 0,
    ):
        nn.Module.__init__(self)
        from efficient_kan import KAN

        if depth < 2:
            raise ValueError("EncoderDeepKANPolicy requires depth >= 2.")

        torch.manual_seed(seed)
        self.node_emb_dim = node_emb_dim
        self.action_emb_dim = action_emb_dim
        self.bottleneck_dim = bottleneck_dim
        self.hidden = hidden
        self.depth = depth

        self.action_embedding = nn.Embedding(len(_ACTION_ORDER), action_emb_dim)
        self.bottleneck = nn.Linear(node_emb_dim + action_emb_dim, bottleneck_dim)
        self.kan = KAN(
            [bottleneck_dim] + [hidden] * depth + [1],
            grid_size=grid,
            spline_order=k,
        )
