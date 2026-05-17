"""
Training loop v1 — supervised learning on backward-generated trajectories.

Pipeline: trajectory JSON → AST reconstruction → encoder → policy → cross-entropy loss.
One expression at a time (no cross-expression batching for v1).

Usage:
    python train.py --data trajectories/ --epochs 20 --lr 1e-3
"""

import json
import math
import random
import time
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim

from isre.symbolic.isre_ast import ASTNode
from isre.symbolic.symbolic_engine import SymbolicEngine, ActionType

from isre.learning.encoder import ASTEncoder
from isre.learning.policy import PolicyNetwork


# ====================== DATA LOADING ======================

@dataclass
class TrainingStep:
    """One (state, candidates, gold) training pair."""
    ast: ASTNode
    candidate_actions: List[Tuple[int, ActionType]]
    gold_action: ActionType
    gold_node_id: int
    complexity: int
    difficulty: int  # trajectory length (for curriculum)


def load_trajectories(data_dir: str, max_files: int = None) -> List[TrainingStep]:
    """Load trajectory JSONs → flat list of training steps."""
    data_path = Path(data_dir)
    files = sorted(data_path.glob("traj_*.json"))
    if max_files:
        files = files[:max_files]

    steps: List[TrainingStep] = []
    action_map = {a.value: a for a in ActionType}
    engine = SymbolicEngine()
    skipped = 0

    for f in files:
        with open(f) as fh:
            traj = json.load(fh)

        difficulty = traj["difficulty"]

        for step_data in traj["steps"]:
            try:
                ast = ASTNode.from_dict(step_data["state"])
            except Exception:
                skipped += 1
                continue

            gold_action_str = step_data["gold_action"]
            gold_node_id = step_data["gold_node_id"]

            if gold_action_str not in action_map:
                skipped += 1
                continue

            gold_action = action_map[gold_action_str]

            # Reconstruct candidates from symbolic engine (ground truth)
            # We regenerate rather than trust stored candidates — ensures consistency
            candidates_raw = engine.get_candidates(ast)
            candidate_actions = [(nid, action) for nid, _, action in candidates_raw]

            # Verify gold is in candidates
            gold_present = any(
                nid == gold_node_id and action == gold_action
                for nid, action in candidate_actions
            )
            if not gold_present:
                skipped += 1
                continue

            steps.append(TrainingStep(
                ast=ast,
                candidate_actions=candidate_actions,
                gold_action=gold_action,
                gold_node_id=gold_node_id,
                complexity=step_data.get("complexity", 0),
                difficulty=difficulty,
            ))

    if skipped > 0:
        print(f"  Skipped {skipped} invalid steps")

    return steps


# ====================== METRICS ======================

@dataclass
class EpochMetrics:
    """Accumulated metrics for one epoch."""
    total_loss: float = 0.0
    total_steps: int = 0
    correct_top1: int = 0
    gold_rank_sum: int = 0
    action_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    @property
    def avg_loss(self) -> float:
        return self.total_loss / max(self.total_steps, 1)

    @property
    def accuracy(self) -> float:
        return self.correct_top1 / max(self.total_steps, 1)

    @property
    def avg_gold_rank(self) -> float:
        return self.gold_rank_sum / max(self.total_steps, 1)

    def report(self, epoch: int, elapsed: float) -> str:
        lines = [
            f"Epoch {epoch:3d} | "
            f"loss {self.avg_loss:.4f} | "
            f"acc {self.accuracy:.3f} | "
            f"avg_gold_rank {self.avg_gold_rank:.2f} | "
            f"steps {self.total_steps} | "
            f"{elapsed:.1f}s"
        ]
        # Action distribution
        if self.action_counts:
            total = sum(self.action_counts.values())
            dist = {k: f"{v/total:.1%}" for k, v in sorted(self.action_counts.items())}
            lines.append(f"  gold action dist: {dist}")
        return "\n".join(lines)


# ====================== CURRICULUM ======================
#
# Temperature-weighted sampling, NOT a hard 1->6 stage filter.
#
# Why the old design was a methodological hazard:
#   - hard filter (diff <= max) -> stepwise distribution shift -> loss
#     jumps at transitions, not smooth learning.
#   - cap at 6 -> v6 trajectories with diff 7-10 (BFS-true length) were
#     NEVER trained -> any eval collapse on long trajectories would be a
#     CURRICULUM ARTIFACT misread as an architecture weakness, and KAN
#     phi-curves on depth 7+ would be extrapolation noise, not signal.
#   - auto-advance keyed on noisy val accuracy -> non-reproducible
#     curriculum trajectory across the 5 seeds (seeds must differ only
#     in init/shuffle, not in what data they saw).
#
# Replacement: every step keeps a soft weight
#   w(diff) = exp(-(diff - target(epoch))^2 / (2*sigma^2))
# target(epoch) ramps deterministically 1.5 -> max_difficulty over the
# run; sigma is wide so SHORT and LONG tails always retain nonzero weight
# (no catastrophic forgetting, no untrained tail). Schedule is a pure
# function of (epoch, total_epochs) -> identical across seeds.


CURRICULUM_LO = 1.0       # start target (genuinely easy-skewed early)
CURRICULUM_SIGMA = 2.0    # narrow enough for a sharp early low-diff peak
CURRICULUM_TRUNC = 3.0    # hard cutoff at |diff-target| > TRUNC*sigma
CURRICULUM_HI = 7.0       # target ramp CAP. NOT max_diff.
# Why cap at 7, not max_diff (~11): the v6 population is mid-skewed and
# the tail is thin (d7:1.4%, d11:0.05%). Ramping target onto near-empty
# d10-11 bins centred the Gaussian where there is no population; the
# convolution (weight rising toward 11) x (population falling toward 11)
# produced a BIMODAL final-epoch histogram (verify_curriculum check 1
# failed at e20: peak d6, dip d7, second bump d9). With target capped at
# 7 and sigma 2, the 3sigma window late = [~1,13] still fully covers the
# d7-11 tail — in fact d7 sits AT the target (weight 1.0), so the tail is
# trained MORE strongly than when target overshot to 11. Restores the
# originally-specified "ramp 1 -> 6/7" intent.


def curriculum_target(epoch: int, total_epochs: int, max_diff: int,
                      lo: float = CURRICULUM_LO,
                      hi: float = CURRICULUM_HI) -> float:
    """Deterministic difficulty target. Ramps lo -> min(hi, max_diff)
    over ~80% of epochs. Capped (see CURRICULUM_HI) so the Gaussian never
    centres on the near-empty difficulty tail."""
    top = min(float(max_diff), hi)
    if total_epochs <= 1:
        return top
    ramp_end = max(1, int(0.8 * total_epochs))
    progress = min(1.0, (epoch - 1) / ramp_end)
    return lo + (top - lo) * progress


def difficulty_weights(steps: List["TrainingStep"], target: float,
                       sigma: float = CURRICULUM_SIGMA,
                       trunc: float = CURRICULUM_TRUNC) -> List[float]:
    """TRUNCATED Gaussian soft weight per step around `target`.

    Why truncated (reviewer's methodologically-cleanest option, verified):
    an untruncated wide Gaussian gave w(diff=10)≈0.003 even at target=1.5,
    and empirically the epoch-1 sampled histogram PEAKED at diff 3-4
    (56.8%) rather than being easy-skewed to diff 1-2 (31.3%) — because
    sigma was wide relative to the 1-4 range and the v6 population is
    mid-skewed. Hard 3σ truncation removes the far-tail noise EARLY (clean
    curriculum, no diff-10 extrapolation polluting early KAN signal) while
    the ramping target guarantees the full tail is covered LATE (cutoff
    moves up with target; by the final epochs diff 7-10 are in-window).
    """
    inv = 1.0 / (2.0 * sigma * sigma)
    cut = trunc * sigma
    out = []
    for s in steps:
        d = s.difficulty - target
        out.append(0.0 if abs(d) > cut else math.exp(-(d * d) * inv))
    return out


# ====================== TRAINING ======================

class Trainer:
    def __init__(
        self,
        encoder: ASTEncoder,
        policy: PolicyNetwork,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        device: str = "cpu",
    ):
        self.encoder = encoder.to(device)
        self.policy = policy.to(device)
        self.device = device

        # Single optimizer for both encoder and policy
        self.optimizer = optim.AdamW(
            list(encoder.parameters()) + list(policy.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=3,
        )

    def train_epoch(
        self,
        steps: List[TrainingStep],
        epoch: int,
        total_epochs: int,
        accumulation_steps: int = 8,
    ) -> EpochMetrics:
        """Train one epoch with temperature-weighted curriculum sampling."""
        self.encoder.train()
        self.policy.train()

        metrics = EpochMetrics()
        self.optimizer.zero_grad()

        if not steps:
            print("  WARNING: no training steps")
            return metrics

        # Temperature-weighted curriculum: soft Gaussian weight around a
        # deterministic per-epoch difficulty target. No hard cutoff, no
        # untrained tail, schedule reproducible across seeds.
        max_diff = max(s.difficulty for s in steps)
        target = curriculum_target(epoch, total_epochs, max_diff)
        weights = difficulty_weights(steps, target)
        # Resample an epoch-sized multiset (with replacement) by weight.
        filtered = random.choices(steps, weights=weights, k=len(steps))

        # Verification instrumentation: histogram of SAMPLED difficulties,
        # so the ramp is auditable (smooth shift, tail always covered).
        sampled_hist: Dict[int, int] = defaultdict(int)
        for s in filtered:
            sampled_hist[s.difficulty] += 1
        hist_str = " ".join(f"d{d}:{sampled_hist[d]}"
                             for d in sorted(sampled_hist))
        print(f"  CURRICULUM e{epoch}/{total_epochs} target={target:.2f} "
              f"max_diff={max_diff} sampled[{hist_str}]")

        for i, step in enumerate(filtered):
            loss, top1_correct, gold_rank = self._train_step(step)

            if loss is None:
                continue

            # Gradient accumulation
            scaled_loss = loss / accumulation_steps
            scaled_loss.backward()

            if (i + 1) % accumulation_steps == 0:
                nn.utils.clip_grad_norm_(
                    list(self.encoder.parameters()) + list(self.policy.parameters()),
                    max_norm=1.0,
                )
                self.optimizer.step()
                self.optimizer.zero_grad()

            # Track metrics
            metrics.total_loss += loss.item()
            metrics.total_steps += 1
            metrics.correct_top1 += int(top1_correct)
            metrics.gold_rank_sum += gold_rank
            metrics.action_counts[step.gold_action.value] += 1

        # Final optimizer step for remaining gradients
        if metrics.total_steps % accumulation_steps != 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

        return metrics

    def _train_step(
        self, step: TrainingStep
    ) -> Tuple[Optional[torch.Tensor], bool, int]:
        """Process one training step. Returns (loss, top1_correct, gold_rank)."""
        try:
            # Encode AST
            node_embeddings, _ = self.encoder(step.ast)

            # Score candidates
            scores = self.policy(node_embeddings, step.candidate_actions)

            if scores.numel() == 0:
                return None, False, 0

            # Find gold index
            gold_idx = None
            for j, (nid, action) in enumerate(step.candidate_actions):
                if nid == step.gold_node_id and action == step.gold_action:
                    gold_idx = j
                    break

            if gold_idx is None:
                return None, False, 0

            # Cross-entropy loss
            target = torch.tensor(gold_idx, device=self.device)
            loss = nn.functional.cross_entropy(scores.unsqueeze(0), target.unsqueeze(0))

            # Metrics
            top1 = scores.argmax().item() == gold_idx
            ranked = scores.argsort(descending=True).tolist()
            gold_rank = ranked.index(gold_idx) + 1  # 1-indexed

            return loss, top1, gold_rank

        except Exception as e:
            # Don't crash training on one bad example
            print(f"  WARNING: step failed: {e}")
            return None, False, 0

    @torch.no_grad()
    def evaluate(self, steps: List[TrainingStep]) -> EpochMetrics:
        """Evaluate on a set of steps (no gradient)."""
        self.encoder.eval()
        self.policy.eval()

        metrics = EpochMetrics()

        for step in steps:
            loss, top1_correct, gold_rank = self._train_step(step)
            if loss is None:
                continue

            metrics.total_loss += loss.item()
            metrics.total_steps += 1
            metrics.correct_top1 += int(top1_correct)
            metrics.gold_rank_sum += gold_rank
            metrics.action_counts[step.gold_action.value] += 1

        return metrics

    # maybe_advance_curriculum REMOVED: noisy-val-accuracy auto-advance made
    # the curriculum trajectory non-reproducible across seeds and capped at
    # diff 6 (untrained 7-10 tail). Replaced by deterministic
    # curriculum_target(epoch) used inside train_epoch.


# ====================== MAIN ======================

def train(
    data_dir: str = "trajectories",
    epochs: int = 20,
    lr: float = 1e-3,
    hidden_dim: int = 128,
    num_rounds: int = 4,
    val_split: float = 0.1,
    device: str = "auto",
    max_files: int = None,
    accumulation_steps: int = 8,
    save_dir: str = "checkpoints",
    seed: int = 0,
):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  Seed: {seed}")

    # Global seeding — REQUIRED for the 5-seed campaign to be meaningful.
    # curriculum_target is already a pure function of epoch (identical
    # across seeds); seeding makes the curriculum SAMPLING (random.choices)
    # and weight init / shuffle reproducible: same seed -> identical run,
    # different seed -> controlled variance (only init+sampling differ,
    # never the curriculum schedule). Without this, "5 seeds" is undefined.
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # ── Load data ──────────────────────────────────────
    print(f"Loading trajectories from {data_dir}...")
    all_steps = load_trajectories(data_dir, max_files=max_files)
    print(f"  Total training steps: {len(all_steps)}")

    if not all_steps:
        print("No training data found. Run trajectory_gen.py first.")
        return

    # Train/val split
    random.shuffle(all_steps)
    split_idx = max(1, int(len(all_steps) * (1 - val_split)))
    train_steps = all_steps[:split_idx]
    val_steps = all_steps[split_idx:]
    print(f"  Train: {len(train_steps)}, Val: {len(val_steps)}")

    # Dataset bias check
    action_dist = defaultdict(int)
    for s in train_steps:
        action_dist[s.gold_action.value] += 1
    total = sum(action_dist.values())
    print(f"  Gold action distribution:")
    for action, count in sorted(action_dist.items(), key=lambda x: -x[1]):
        print(f"    {action}: {count} ({count/total:.1%})")

    # ── Build model ────────────────────────────────────
    encoder = ASTEncoder(hidden_dim=hidden_dim, num_rounds=num_rounds)
    policy = PolicyNetwork(
        node_emb_dim=hidden_dim * 2,
        variant="mlp",
        hidden_dim=hidden_dim,
    )

    total_params = sum(p.numel() for p in encoder.parameters()) + \
                   sum(p.numel() for p in policy.parameters())
    print(f"  Encoder params: {sum(p.numel() for p in encoder.parameters()):,}")
    print(f"  Policy params:  {sum(p.numel() for p in policy.parameters()):,}")
    print(f"  Total params:   {total_params:,}")

    # ── Trainer ────────────────────────────────────────
    trainer = Trainer(
        encoder=encoder,
        policy=policy,
        lr=lr,
        device=device,
    )

    # ── Training loop ──────────────────────────────────
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # Train
        train_metrics = trainer.train_epoch(
            train_steps, epoch=epoch, total_epochs=epochs,
            accumulation_steps=accumulation_steps,
        )
        elapsed = time.time() - t0
        print(train_metrics.report(epoch, elapsed))

        # Validate
        val_metrics = trainer.evaluate(val_steps)
        print(f"  VAL  | loss {val_metrics.avg_loss:.4f} | "
              f"acc {val_metrics.accuracy:.3f} | "
              f"avg_gold_rank {val_metrics.avg_gold_rank:.2f}")

        # LR schedule
        trainer.scheduler.step(val_metrics.avg_loss)

        # (curriculum is deterministic per-epoch inside train_epoch — no
        #  noisy-signal advancement step here anymore)

        # Save best
        if val_metrics.avg_loss < best_val_loss:
            best_val_loss = val_metrics.avg_loss
            torch.save({
                "epoch": epoch,
                "encoder": encoder.state_dict(),
                "policy": policy.state_dict(),
                "optimizer": trainer.optimizer.state_dict(),
                "val_loss": best_val_loss,
            }, save_path / "best.pt")
            print(f"  SAVED best model (val_loss={best_val_loss:.4f})")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="ISRE Training")
    parser.add_argument("--data", default="trajectories", help="Trajectory directory")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-rounds", type=int, default=4)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--accumulation-steps", type=int, default=8)
    parser.add_argument("--save-dir", default="checkpoints")
    parser.add_argument("--seed", type=int, default=0,
                        help="Global seed (random+torch+cuda). The 5-seed "
                             "campaign varies ONLY this; curriculum schedule "
                             "stays identical across seeds.")
    args = parser.parse_args()

    train(
        data_dir=args.data,
        epochs=args.epochs,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        num_rounds=args.num_rounds,
        val_split=args.val_split,
        device=args.device,
        max_files=args.max_files,
        accumulation_steps=args.accumulation_steps,
        save_dir=args.save_dir,
        seed=args.seed,
    )