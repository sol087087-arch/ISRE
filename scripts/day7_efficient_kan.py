"""Day-7 KAN feasibility gate ON THE efficient_kan BACKEND (BLOCKING).

Copy of scripts/day7_kan_synthetic.py with the KAN backend swapped from
pykan to efficient_kan. SAME synthetic, SAME quantitative falsifiable
gate. This is the interpretability falsifiable gate on the NEW backend:
if it FAILs, efficient-kan cannot recover known additive structure and
the backend swap is NOT viable.

  target  f(x,y) = sin(x) + |y|     (two univariate ground-truth funcs)

PASS (quantitative, backend-agnostic):
  1. learns the function:  test MSE small  (< 0.05*sig^2 + 0.02)
  2. x-marginal recovers sin:  R^2( m_x , sin(x) ) > 0.95
  3. y-marginal recovers |.|:  R^2( m_y , |y| )   > 0.95
  4. ADDITIVE-separable:  interaction RMSE / signal-std  < 0.05

ALSO report the roughness coefficient (std of 2nd-difference / std) of
the recovered marginals — the same smoothness metric used in the pykan
smoke phi-health check (it was <0.5% on pykan). Cross-backend evidence
that marginal smoothness is a task property, not a pykan artifact.

efficient_kan trains with plain backprop -> standard torch Adam loop.
Exit 0 ONLY if all four gate criteria pass.
"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import torch
from efficient_kan import KAN

torch.manual_seed(0)
DEV = "cpu"                       # 2-input toy; speed irrelevant
OUT = Path("day7_efficient_kan_out"); OUT.mkdir(exist_ok=True)


def f(xy):                        # ground truth
    return torch.sin(xy[:, 0:1]) + torch.abs(xy[:, 1:2])


def make(n, xr=3.0, yr=2.0):
    x = (torch.rand(n, 1) * 2 - 1) * xr
    y = (torch.rand(n, 1) * 2 - 1) * yr
    inp = torch.cat([x, y], 1)
    return inp, f(inp)


tr_i, tr_l = make(3000)
te_i, te_l = make(1000)

# Additive separable f(x,y)=g1(x)+g2(y). efficient_kan KAN takes a
# layers-list; use a small hidden to give it capacity for the two
# univariate funcs. grid_range covers the x-domain [-3,3].
model = KAN([2, 4, 1], grid_size=7, spline_order=3,
            grid_range=[-3, 3]).to(DEV)
print("training efficient_kan KAN [2,4,1] grid=7 k=3 on sin(x)+|y| ...")

opt = torch.optim.Adam(model.parameters(), lr=1e-2)
loss_fn = torch.nn.MSELoss()
STEPS = 800
for step in range(STEPS):
    opt.zero_grad()
    pred = model(tr_i)
    loss = loss_fn(pred, tr_l)
    loss.backward()
    opt.step()
    if step == 0 or (step + 1) % 200 == 0:
        print(f"  step {step + 1:4d}  train MSE = {loss.item():.5f}")

model.eval()
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

    def roughness(v):
        # std of discrete 2nd-difference / std of the signal. Same metric
        # used by the pykan smoke phi-health check (<0.5% on pykan).
        d2 = v[2:] - 2 * v[1:-1] + v[:-2]
        return (torch.std(d2) / (torch.std(v) + 1e-12)).item()

    rough_x = roughness(mx)
    rough_y = roughness(my)

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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Backend-INDEPENDENT phi_i artifact: learned marginal vs ground truth.
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(grid.squeeze().numpy(), sinx.numpy(), "k--", label="true sin(x)")
ax[0].plot(grid.squeeze().numpy(), mx.numpy(), "C0",
           label="KAN phi_x  (y=0)")
ax[0].set_title(f"phi_x vs sin(x)   R2={r2_x:.4f}"); ax[0].legend()
ax[1].plot(gy.squeeze().numpy(), absy.numpy(), "k--", label="true |y|")
ax[1].plot(gy.squeeze().numpy(), my.numpy(), "C1",
           label="KAN phi_y  (x=0)")
ax[1].set_title(f"phi_y vs |y|   R2={r2_y:.4f}"); ax[1].legend()
fig.suptitle("Day-7 efficient_kan: learned univariate funcs vs ground truth")
fig.tight_layout()
fig.savefig(OUT / "phi_recovery.png", dpi=120, bbox_inches="tight")
plt.close(fig)
plot_note = (f"phi_i artifact (backend-independent): "
             f"{OUT / 'phi_recovery.png'}")

print(f"\ntest MSE        : {test_mse:.5f}  (signal std {sig_std:.3f})")
print(f"R^2 x-marginal vs sin(x) : {r2_x:.4f}   (PASS > 0.95)")
print(f"R^2 y-marginal vs |y|    : {r2_y:.4f}   (PASS > 0.95)")
print(f"interaction RMSE / sig   : {inter_ratio:.4f} (PASS < 0.05  "
      f"=> additive-separable)")
print(f"roughness phi_x          : {rough_x:.4%}  "
      f"(pykan smoke ref: <0.5%)")
print(f"roughness phi_y          : {rough_y:.4%}  "
      f"(pykan smoke ref: <0.5%)")
print(plot_note)

ok = (test_mse < 0.05 * sig_std ** 2 + 0.02 and
      r2_x > 0.95 and r2_y > 0.95 and inter_ratio < 0.05)
print("\n" + "=" * 56)
if ok:
    print("DAY-7 (efficient_kan): PASS — efficient-kan recovers known "
          "additive structure; phi_i interpretability methodology is "
          "valid on the NEW backend. Backend swap is viable.")
else:
    print("DAY-7 (efficient_kan): FAIL — efficient-kan did NOT cleanly "
          "recover sin/|.| / not additive-separable. The backend swap "
          "is NOT viable on this gate. Debug HERE (synthetic).")
sys.exit(0 if ok else 1)
