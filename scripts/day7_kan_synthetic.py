"""Day-7 KAN feasibility gate (BLOCKING for KAN-vs-MLP).

Falsifiable check: give KAN a task with KNOWN additive analytic
structure and verify phi_i recover it.

  target  f(x,y) = sin(x) + |y|     (two univariate ground-truth funcs)

PASS (quantitative, backend-agnostic — NOT eyeballing .plot()):
  1. learns the function:  test MSE small
  2. x-marginal recovers sin:  m_x(x)=model(x,0)  (|0|=0 => f=sin(x))
       R^2( m_x , sin(x) ) > 0.95
  3. y-marginal recovers |.|:  m_y(y)=model(0,y)  (sin(0)=0 => f=|y|)
       R^2( m_y , |y| ) > 0.95
  4. ADDITIVE-separable (the interpretable structure, not an entangled
     2D blob):  model(x,y) ~= model(x,0)+model(0,y)-model(0,0)
       interaction RMSE / signal-std  < 0.05

PASS => interpretability methodology works on this infra => proceed to
KAN-vs-MLP@128. FAIL => debug on THIS synthetic, never on the real task.
.plot() figures saved as the visual artifact, but the gate is the numbers.
"""
import sys, math
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import torch
from kan import KAN

torch.manual_seed(0)
DEV = "cpu"                       # 2-input toy; speed irrelevant
OUT = Path("day7_kan_out"); OUT.mkdir(exist_ok=True)


def f(xy):                        # ground truth
    return torch.sin(xy[:, 0:1]) + torch.abs(xy[:, 1:2])


def make(n, xr=3.0, yr=2.0):
    x = (torch.rand(n, 1) * 2 - 1) * xr
    y = (torch.rand(n, 1) * 2 - 1) * yr
    inp = torch.cat([x, y], 1)
    return inp, f(inp)


tr_i, tr_l = make(3000)
te_i, te_l = make(1000)
dataset = {"train_input": tr_i, "train_label": tr_l,
           "test_input": te_i, "test_label": te_l}

# Additive separable f(x,y)=g1(x)+g2(y): canonical KAN width [2,1,1].
model = KAN(width=[2, 1, 1], grid=7, k=3, seed=0,
            grid_range=[-3, 3], device=DEV)
print("training KAN [2,1,1] grid=7 k=3 on sin(x)+|y| ...")
model.fit(dataset, opt="LBFGS", steps=60)

with torch.no_grad():
    pred = model(te_i)
    test_mse = torch.mean((pred - te_l) ** 2).item()
    sig_std = torch.std(te_l).item()

    grid = torch.linspace(-3, 3, 200).unsqueeze(1)
    zerox = torch.zeros_like(grid)
    # x-marginal at y=0
    mx = model(torch.cat([grid, zerox], 1)).squeeze()
    sinx = torch.sin(grid.squeeze())
    # y-marginal at x=0  (use y-range -2..2)
    gy = torch.linspace(-2, 2, 200).unsqueeze(1)
    my = model(torch.cat([torch.zeros_like(gy), gy], 1)).squeeze()
    absy = torch.abs(gy.squeeze())

    def r2(a, b):
        a, b = a - a.mean(), b - b.mean()
        return (torch.sum(a * b) ** 2 /
                (torch.sum(a * a) * torch.sum(b * b) + 1e-12)).item()

    r2_x = r2(mx, sinx)
    r2_y = r2(my, absy)

    # additivity / interaction residual on a 30x30 grid
    gx = torch.linspace(-3, 3, 30)
    gyy = torch.linspace(-2, 2, 30)
    XX, YY = torch.meshgrid(gx, gyy, indexing="ij")
    G = torch.stack([XX.reshape(-1), YY.reshape(-1)], 1)
    m_xy = model(G).squeeze()
    m_x0 = model(torch.cat([G[:, 0:1], torch.zeros(G.size(0), 1)], 1)).squeeze()
    m_0y = model(torch.cat([torch.zeros(G.size(0), 1), G[:, 1:2]], 1)).squeeze()
    m_00 = model(torch.zeros(1, 2)).squeeze()
    inter = m_xy - (m_x0 + m_0y - m_00)
    inter_rmse = torch.sqrt(torch.mean(inter ** 2)).item()
    inter_ratio = inter_rmse / (sig_std + 1e-12)

try:
    model.plot(folder=str(OUT))
    import matplotlib.pyplot as plt
    plt.savefig(OUT / "phi_curves.png", dpi=110, bbox_inches="tight")
    plot_note = f"phi_i figures: {OUT/'phi_curves.png'}"
except Exception as e:
    plot_note = f"(plot skipped: {e})"

print(f"\ntest MSE        : {test_mse:.5f}  (signal std {sig_std:.3f})")
print(f"R^2 x-marginal vs sin(x) : {r2_x:.4f}   (PASS > 0.95)")
print(f"R^2 y-marginal vs |y|    : {r2_y:.4f}   (PASS > 0.95)")
print(f"interaction RMSE / sig   : {inter_ratio:.4f} (PASS < 0.05  "
      f"=> additive-separable)")
print(plot_note)

ok = (test_mse < 0.05 * sig_std ** 2 + 0.02 and
      r2_x > 0.95 and r2_y > 0.95 and inter_ratio < 0.05)
print("\n" + "=" * 56)
if ok:
    print("DAY-7: PASS — KAN recovers known additive structure; phi_i "
          "interpretability methodology is valid on this infra. "
          "Clear to proceed to KAN-vs-MLP@128 (hand-crafted features).")
else:
    print("DAY-7: FAIL — KAN did NOT cleanly recover sin/|.| / not "
          "additive-separable. Debug HERE (synthetic), not on the real "
          "task. KAN-vs-MLP is NOT meaningful until this passes.")
sys.exit(0 if ok else 1)
