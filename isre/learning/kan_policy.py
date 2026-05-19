"""
KAN policy head over hand-crafted per-candidate features.

This is the KAN arm of the matched-PROTOCOL KAN-vs-MLP experiment. It does
NOT use the GRU encoder. Each candidate (node_id, ActionType) is turned
into a fixed FEATURE_DIM vector by isre.learning.features.candidate_features,
and an efficient_kan KAN with layers=[FEATURE_DIM, hidden, 1] scores it.

BACKEND: efficient_kan (NOT pykan). The swap is matched-PROTOCOL, not
matched-params: efficient_kan KAN([27,16,1]) is ~4480 params vs pykan's
7752 at hidden=16 — a DIFFERENT spline parameterization/init, EXPECTED
and locked. allclose-vs-pykan is the wrong gate by construction; the
verification regime is T1+T2+Day7-on-efficient-kan+matched-protocol+
re-profile. efficient_kan KAN is a plain nn.Module: forward
[N,FEATURE_DIM]->[N,1], native .parameters()/.state_dict().

Usage mirrors PolicyNetwork so train/eval can be model-agnostic:
  - .score(state_root, candidates)            -> Tensor [n_candidates]
  - .compute_loss(state_root, candidates,
                   gold_action, gold_node_id)  -> scalar CE loss
  - .parameters()                              -> efficient_kan params (optimizer)
  - .select_action(...)                        -> (node_id, action, log_prob)
  - .phi_curves(out_dir)                       -> interpretability plots

compute_loss RAISES ValueError if the gold candidate is absent — the SAME
contract as policy.py (never silent-zero, which would drop the example
from the gradient graph and mask a generator/engine bug).
"""

from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn

from isre.symbolic.isre_ast import ASTNode
from isre.symbolic.symbolic_engine import ActionType
from isre.learning.features import (
    candidate_features,
    FEATURE_DIM,
    FEATURE_NAMES,
)


class KANPolicy(nn.Module):
    """efficient_kan KAN scorer over hand-crafted candidate features.

    layers = [FEATURE_DIM, hidden, 1]. The KAN is a real efficient_kan
    model (plain nn.Module); its parameters flow through a standard torch
    optimizer (verified by scripts/test_kan_trains.py: loss decreases and
    KAN params get grads).
    """

    def __init__(
        self,
        hidden: int = 16,
        grid: int = 5,
        k: int = 3,
        seed: int = 0,
        device: str = "cpu",
        ckpt_path: str = "./.kan_ckpt",
    ):
        super().__init__()
        # Import here so the rest of the project (MLP path) never pays the
        # efficient_kan import cost / dependency unless KAN is actually
        # requested. Signature preserved (grid/k/seed/ckpt_path) so
        # train.py / eval_neural.py need ZERO changes; grid->grid_size,
        # k->spline_order map onto efficient_kan kwargs (defaults 5/3
        # match the old pykan defaults).
        from efficient_kan import KAN

        # Deterministic init from `seed` (efficient_kan has no seed kw; it
        # inits with torch RNG at construction time).
        torch.manual_seed(seed)

        self.feature_dim = FEATURE_DIM
        self.hidden = hidden
        self._device = device

        # efficient_kan KAN takes a layers-list as first positional arg
        # (layers_hidden) plus grid_size/spline_order kwargs. _ = ckpt_path
        # kept for call-surface compatibility; we checkpoint via
        # torch.save(state_dict) ourselves (train.py / eval_neural.py).
        self.kan = KAN(
            [FEATURE_DIM, hidden, 1],
            grid_size=grid,
            spline_order=k,
        ).to(device)

    # ---- internal: build feature matrix for candidates --------------------

    def _feature_matrix(
        self,
        state_root: ASTNode,
        candidates: List[Tuple[int, ActionType]],
    ) -> torch.Tensor:
        # Ensure lazy depth/subtree metadata is consistent for this root.
        state_root.mark_dirty()
        state_root._ensure_metadata()
        rows = [
            candidate_features(state_root, nid, act)
            for (nid, act) in candidates
        ]
        return torch.tensor(rows, dtype=torch.float32, device=self._device)

    # ---- scoring (mirrors PolicyNetwork.forward usage) --------------------

    def score(
        self,
        state_root: ASTNode,
        candidates: List[Tuple[int, ActionType]],
    ) -> torch.Tensor:
        """Score every candidate. Returns [n_candidates] raw logits."""
        if not candidates:
            return torch.tensor([], device=self._device)
        feats = self._feature_matrix(state_root, candidates)  # [n, FEATURE_DIM]
        out = self.kan(feats)                                 # [n, 1]
        return out.squeeze(-1)                                # [n]

    # Alias so eval/training code that does `model(state, cands)` still works.
    def forward(self, state_root, candidates):  # noqa: D401
        return self.score(state_root, candidates)

    def select_action(
        self,
        state_root: ASTNode,
        candidates: List[Tuple[int, ActionType]],
        temperature: float = 1.0,
        greedy: bool = False,
    ) -> Tuple[int, ActionType, torch.Tensor]:
        """Same semantics/contract as PolicyNetwork.select_action."""
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
        """CE over candidates; gold = position where (nid, action) matches.

        RAISES ValueError if gold not in candidates (same contract as
        PolicyNetwork.compute_loss — do NOT silent-zero).
        """
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
        return nn.functional.cross_entropy(
            scores.unsqueeze(0), target.unsqueeze(0)
        )

    # ---- parameters: expose the pykan KAN params for the optimizer --------

    def parameters(self, recurse: bool = True):
        # self.kan is a registered submodule (nn.Module), so the default
        # nn.Module.parameters() already yields its params. We override
        # explicitly to make the contract obvious and to guarantee the
        # optimizer trains the KAN. efficient_kan KAN is a native
        # nn.Module so this delegates directly.
        return self.kan.parameters(recurse=recurse)

    # ---- checkpoint helpers ----------------------------------------------

    def state_dict(self, *args, **kwargs):
        return self.kan.state_dict(*args, **kwargs)

    def load_state_dict(self, sd, *args, **kwargs):
        return self.kan.load_state_dict(sd, *args, **kwargs)

    # ---- interpretability: phi curves ------------------------------------

    def phi_curves(self, out_dir: str, n_repr: int = 512):
        """Save interpretability artifacts to out_dir.

        efficient_kan has NO .plot()/symbolic/auto-cache. We use ONLY the
        backend-INDEPENDENT per-feature sweep (the Day-7-validated
        phi_recovery approach): vary one feature across its range with all
        others held at their mean, forward the KAN, plot output vs that
        feature. Saves phi_<idx>_<featurename>.png per feature plus a
        combined grid figure. Zero pykan calls; cannot hit the
        std-dof / alpha-nan path (that was pykan-only).
        """
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Representative batch: features are mostly in [0,1] (one-hots) with
        # a few normalized scalars; sample uniformly in [0,1] as a generic
        # representative distribution for the "others at mean" baseline.
        repr_x = torch.rand(n_repr, FEATURE_DIM, device=self._device)
        mean_x = repr_x.mean(dim=0, keepdim=True)

        plot_note = (f"efficient_kan backend: backend-independent "
                     f"per-feature phi sweeps in {out}")

        # Backend-independent per-feature sweeps + combined grid figure.
        import math
        n_cols = 6
        n_rows = math.ceil(FEATURE_DIM / n_cols)
        cfig, caxes = plt.subplots(n_rows, n_cols,
                                   figsize=(3 * n_cols, 2.4 * n_rows))
        caxes = caxes.reshape(-1)
        sweep = torch.linspace(0.0, 1.0, 100, device=self._device)
        for fi in range(FEATURE_DIM):
            X = mean_x.repeat(100, 1).clone()
            X[:, fi] = sweep
            with torch.no_grad():
                y = self.kan(X).squeeze(-1).cpu().numpy()
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.plot(sweep.cpu().numpy(), y, "C0")
            ax.set_title(f"phi[{fi}] {FEATURE_NAMES[fi]}")
            ax.set_xlabel("feature value (others at mean)")
            ax.set_ylabel("KAN output")
            fig.tight_layout()
            safe = FEATURE_NAMES[fi].replace("[", "_").replace("]", "") \
                .replace("/", "_")
            fig.savefig(out / f"phi_{fi:02d}_{safe}.png", dpi=110,
                        bbox_inches="tight")
            plt.close(fig)

            cax = caxes[fi]
            cax.plot(sweep.cpu().numpy(), y, "C0")
            cax.set_title(f"[{fi}] {FEATURE_NAMES[fi]}", fontsize=7)
            cax.tick_params(labelsize=6)

        for j in range(FEATURE_DIM, len(caxes)):
            caxes[j].axis("off")
        cfig.suptitle("efficient_kan phi sweeps (per-feature, others@mean)")
        cfig.tight_layout()
        cfig.savefig(out / "phi_combined.png", dpi=110,
                     bbox_inches="tight")
        plt.close(cfig)

        return plot_note


# ====================== TESTS ======================

if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    from isre.symbolic.isre_ast import Add, Mul, Var, Num, Pow
    from isre.symbolic.symbolic_engine import SymbolicEngine

    engine = SymbolicEngine()
    root = Add(Mul(Num(2), Var()), Mul(Num(3), Var()), Num(0))
    root.mark_dirty(); root._rebuild_parents()
    cands_raw = engine.get_candidates(root)
    cands = [(nid, a) for nid, _, a in cands_raw]
    print(f"candidates: {len(cands)}")

    pol = KANPolicy(hidden=8, seed=0)
    scores = pol.score(root, cands)
    print(f"scores shape: {scores.shape}")
    assert scores.shape == (len(cands),)
    print("  score OK")

    nid, act, lp = pol.select_action(root, cands, greedy=True)
    print(f"  greedy pick: node={nid} action={act.value}")

    gnid, gact = cands[0]
    loss = pol.compute_loss(root, cands, gact, gnid)
    loss.backward()
    grads = [p.grad for p in pol.parameters() if p.grad is not None]
    nz = any(g.abs().sum().item() > 0 for g in grads)
    print(f"  loss={loss.item():.4f}  nonzero KAN grad={nz}")
    assert nz, "no nonzero KAN gradient"

    try:
        pol.compute_loss(root, cands, ActionType.MERGE_POWER, 999)
        raise AssertionError("expected ValueError")
    except ValueError:
        print("  missing-gold raises ValueError OK")

    print("\nALL KAN_POLICY TESTS PASSED")
