"""
KAN policy head over hand-crafted per-candidate features.

This is the KAN arm of the matched-PROTOCOL KAN-vs-MLP experiment. It does
NOT use the GRU encoder. Each candidate (node_id, ActionType) is turned
into a fixed FEATURE_DIM vector by isre.learning.features.candidate_features,
and a pykan KAN with width=[FEATURE_DIM, hidden, 1] scores it.

Usage mirrors PolicyNetwork so train/eval can be model-agnostic:
  - .score(state_root, candidates)            -> Tensor [n_candidates]
  - .compute_loss(state_root, candidates,
                   gold_action, gold_node_id)  -> scalar CE loss
  - .parameters()                              -> pykan KAN params (optimizer)
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
    """pykan KAN scorer over hand-crafted candidate features.

    width = [FEATURE_DIM, hidden, 1]. The KAN is a real pykan model; its
    parameters flow through a standard torch optimizer (verified by
    scripts/test_kan_trains.py: loss decreases and KAN params get grads).
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
        # pykan import cost / dependency unless KAN is actually requested.
        from kan import KAN

        self.feature_dim = FEATURE_DIM
        self.hidden = hidden
        self._device = device

        # auto_save=False: we checkpoint via torch.save(state_dict) ourselves
        # (train.py / eval_neural.py), not pykan's versioned auto-saver.
        self.kan = KAN(
            width=[FEATURE_DIM, hidden, 1],
            grid=grid,
            k=k,
            seed=seed,
            device=device,
            auto_save=False,
            ckpt_path=ckpt_path,
        )

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
        # optimizer trains the KAN even if pykan changes its registration.
        return self.kan.parameters(recurse=recurse)

    # ---- checkpoint helpers ----------------------------------------------

    def state_dict(self, *args, **kwargs):
        return self.kan.state_dict(*args, **kwargs)

    def load_state_dict(self, sd, *args, **kwargs):
        return self.kan.load_state_dict(sd, *args, **kwargs)

    # ---- interpretability: phi curves ------------------------------------

    def phi_curves(self, out_dir: str, n_repr: int = 512):
        """Save interpretability artifacts to out_dir:

          1. pykan native model.plot() edge figure. NOTE: pykan .plot()
             needs a representative LARGE-batch forward IMMEDIATELY before
             it, else it raises alpha(nan) from std dof<=0 on cached
             postacts (known; see scripts/day7_kan_synthetic.py). We do
             that forward here then call model.plot(folder=out_dir).
          2. Backend-INDEPENDENT per-feature sweep plots phi_<name>.png:
             vary one feature across its range with all others at their
             mean, plot KAN output. Does not depend on pykan .plot().
        """
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Representative batch: features are mostly in [0,1] (one-hots) with
        # a few normalized scalars; sample uniformly in [0,1] as a generic
        # representative distribution for plotting statistics.
        repr_x = torch.rand(n_repr, FEATURE_DIM, device=self._device)
        mean_x = repr_x.mean(dim=0, keepdim=True)

        # 1) pykan native plot — large forward immediately before .plot().
        plot_note = ""
        try:
            with torch.no_grad():
                self.kan(repr_x)  # repopulate cached acts (valid std)
            self.kan.plot(folder=str(out))
            plt.savefig(out / "pykan_native.png", dpi=110,
                        bbox_inches="tight")
            plt.close("all")
            plot_note = f"pykan native: {out / 'pykan_native.png'}"
        except Exception as e:  # nice-to-have, never fatal
            plot_note = f"pykan native .plot() failed: {e}"

        # 2) Backend-independent per-feature sweeps.
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
