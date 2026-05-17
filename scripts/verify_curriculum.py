"""Curriculum verification panel (reviewer's 4 standard checks).

Pure simulation of the sampling — no model training — over the real
v6_bfs difficulty population. Covers:

  1. Per-epoch histogram at e1, N/4, N/2, 3N/4, N — each UNIMODAL, peak
     moves monotonically left -> right.
  2. Tail coverage: every epoch after 50% has diff>=7 count > 0.
  4. Reproducibility: same seed twice -> identical sampled IDs; different
     seed -> different (controlled variance).

(Check 3, per-batch loss smoothness, needs a training run — done as a
separate short smoke; structurally the target is a continuous linear ramp
with no discrete stage switch, so there are no boundaries to jump at.)

Usage: python scripts/verify_curriculum.py [data_dir] [total_epochs]
"""
import sys, json, random
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from isre.training.train import curriculum_target, difficulty_weights

DATA = sys.argv[1] if len(sys.argv) > 1 else "isre/trajectories_v6_bfs"
N_EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 20


class _S:  # shim: difficulty_weights only needs .difficulty
    __slots__ = ("difficulty", "tid")
    def __init__(self, d, t): self.difficulty = d; self.tid = t


print(f"Loading difficulties from {DATA} ...")
steps = []
for f in sorted(Path(DATA).glob("traj_*.json")):
    t = json.loads(f.read_text(encoding="utf-8"))
    d = t["difficulty"]
    for _ in t["steps"]:               # one Step per gold step (matches trainer)
        steps.append(_S(d, t["trajectory_id"]))
max_diff = max(s.difficulty for s in steps)
pop = Counter(s.difficulty for s in steps)
print(f"  {len(steps)} steps, max_diff={max_diff}")
print(f"  population by diff: {dict(sorted(pop.items()))}\n")

checkpoints = sorted(set([1, N_EPOCHS // 4, N_EPOCHS // 2,
                          3 * N_EPOCHS // 4, N_EPOCHS]))
checkpoints = [e for e in checkpoints if e >= 1]


def hist_at(epoch, seed=12345):
    random.seed(seed)
    tgt = curriculum_target(epoch, N_EPOCHS, max_diff)
    w = difficulty_weights(steps, tgt)
    samp = random.choices(steps, weights=w, k=len(steps))
    h = Counter(s.difficulty for s in samp)
    return tgt, h


def unimodal(h, lo, hi):
    seq = [h.get(d, 0) for d in range(lo, hi + 1)]
    peak = seq.index(max(seq))
    # non-decreasing up to peak, non-increasing after (allow flats)
    up = all(seq[i] <= seq[i + 1] for i in range(peak))
    down = all(seq[i] >= seq[i + 1] for i in range(peak, len(seq) - 1))
    return up and down, lo + peak


print("=== [1] per-epoch histograms (unimodal, peak L->R) ===")
peaks = []
ok1 = True
for e in checkpoints:
    tgt, h = hist_at(e)
    uni, peak = unimodal(h, 1, max_diff)
    peaks.append(peak)
    bar = " ".join(f"d{d}:{h.get(d,0)}" for d in range(1, max_diff + 1))
    flag = "" if uni else "  <-- NOT UNIMODAL"
    print(f"  e{e:>2}/{N_EPOCHS} tgt={tgt:5.2f} peak=d{peak}  [{bar}]{flag}")
    if not uni:
        ok1 = False
mono = all(peaks[i] <= peaks[i + 1] for i in range(len(peaks) - 1))
print(f"  unimodal-all={ok1}  peak-monotone-LtoR={mono}  peaks={peaks}")

print("\n=== [2] tail coverage (diff>=7 > 0 after 50% epochs) ===")
ok2 = True
for e in range(1, N_EPOCHS + 1):
    if e / N_EPOCHS < 0.5:
        continue
    _, h = hist_at(e)
    tail = sum(h.get(d, 0) for d in range(7, max_diff + 1))
    if tail == 0:
        print(f"  e{e}: TAIL EMPTY  <-- FAIL")
        ok2 = False
print(f"  tail-always-covered-after-50%={ok2}"
      + ("" if ok2 else "  (widen sigma / shift target earlier)"))

print("\n=== [4] reproducibility (same seed -> identical sample) ===")
random.seed(777)
t1 = curriculum_target(3, N_EPOCHS, max_diff)
a = [id_.tid for id_ in random.choices(steps, weights=difficulty_weights(steps, t1), k=2000)]
random.seed(777)
b = [id_.tid for id_ in random.choices(steps, weights=difficulty_weights(steps, t1), k=2000)]
random.seed(778)
c = [id_.tid for id_ in random.choices(steps, weights=difficulty_weights(steps, t1), k=2000)]
same = (a == b)
diff = (a != c)
print(f"  same seed identical : {same}")
print(f"  diff seed differs   : {diff}  (controlled variance)")
ok4 = same and diff

print("\n" + "=" * 52)
allok = ok1 and mono and ok2 and ok4
print("CURRICULUM VERIFICATION: PASS" if allok else "CURRICULUM VERIFICATION: FAIL")
sys.exit(0 if allok else 1)
