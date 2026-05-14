"""
Synthetic data evaluation: compare Heston PINN variants against GL reference.

Both models evaluated on the same params and S grid.

Test params: kappa=1.0, theta=0.08, xi=0.39, rho=-0.7, v0=0.04, r=0.05
  - Chosen to be within Hainaut training range AND close to HESTON_INDEP
  - Hainaut ranges: S∈[20,180], r∈[0.01,0.07], kappa∈[0.5,2.0],
    theta∈[0.062,0.42], xi∈[0.1,0.9], rho∈[-0.8,0.8]
  - heston_pinn trained on rho=-0.93, r=0.1 — these test params are slightly OOD for it
  - hainaut_orig: all params in-distribution

Hainaut prices PUT at K=100; price() returns absolute put price.
Call price via put-call parity: C = P + S - K*exp(-rT)
S grid clipped to [50, 170] to stay within Hainaut S_RANGE=[20,180].
"""
import sys, os
import numpy as np
import torch

BASE = "/home/yz2026/zhuwl2022/uni_pinn/option_pinn"
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "independent"))

from ref_solvers import heston_call
from heston_hainaut import HestonHainaut
from heston_pinn import Heston_PINN

K_SYN  = 100.0
T_SYN  = 1.0
# Stay within Hainaut S_RANGE=[20,180] and heston_pinn S_max=400
S_GRID = np.linspace(50, 170, 50)

# Params within Hainaut training range; close to HESTON_INDEP where possible
PARAMS = dict(kappa=1.0, theta=0.08, xi=0.39, rho=-0.7, v0=0.04)
r_VAL  = 0.05


def mse(pred, ref):
    return float(np.mean((np.array(pred) - np.array(ref)) ** 2))


def relmse(pred, ref):
    ref, pred = np.array(ref), np.array(pred)
    mask = np.abs(ref) > 0.01
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(((pred[mask] - ref[mask]) / ref[mask]) ** 2))


def relmae(pred, ref):
    ref, pred = np.array(ref), np.array(pred)
    mask = np.abs(ref) > 0.01
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((pred[mask] - ref[mask]) / ref[mask])))


# ── Load models ───────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading heston_pinn (fixed-param)...")
ckpt = torch.load(os.path.join(BASE, "results/indep_heston.pt"), map_location=DEVICE)
heston_pinn = Heston_PINN(**ckpt["params"], device=DEVICE)
heston_pinn.aux_net.load_state_dict(ckpt["aux_state"])
heston_pinn.main_net.load_state_dict(ckpt["main_state"])
heston_pinn.aux_net.eval()
heston_pinn.main_net.eval()
print(f"  params: {ckpt['params']}")

print("Loading hainaut_orig (parametric, original range)...")
hainaut = HestonHainaut(device=DEVICE)
hainaut.load(os.path.join(BASE, "results/hainaut.pt"))
print("  Loaded.")


# ── Reference prices ──────────────────────────────────────────────────────────
print("\nComputing GL reference prices...")
ref = np.array([heston_call(S, K_SYN, T_SYN, r_VAL, **PARAMS) for S in S_GRID])


# ── heston_pinn inference ─────────────────────────────────────────────────────
pinn_prices = np.array([heston_pinn.price(S) for S in S_GRID])


# ── hainaut inference ─────────────────────────────────────────────────────────
# price() returns absolute put price; convert to call via put-call parity
hainaut_prices = []
for S in S_GRID:
    put_abs = hainaut.price(
        S=S, V=PARAMS["v0"], t=0.0, T=T_SYN,
        r=r_VAL,
        kappa=PARAMS["kappa"],
        theta=PARAMS["theta"],
        xi=PARAMS["xi"],
        rho=PARAMS["rho"],
    )
    call = put_abs + S - K_SYN * np.exp(-r_VAL * T_SYN)
    hainaut_prices.append(call)
hainaut_prices = np.array(hainaut_prices)


# ── Results ───────────────────────────────────────────────────────────────────
print(f"\nTest params: kappa={PARAMS['kappa']} theta={PARAMS['theta']} "
      f"xi={PARAMS['xi']} rho={PARAMS['rho']} v0={PARAMS['v0']} r={r_VAL}")
print(f"S grid: [{S_GRID[0]:.0f}, {S_GRID[-1]:.0f}], n={len(S_GRID)}\n")

print(f"{'Model':<20} {'MSE':>12} {'RelMSE':>10} {'RelMAE':>10}  Notes")
print("-" * 80)
print(f"{'heston_pinn':<20} {mse(pinn_prices, ref):>12.4e} "
      f"{relmse(pinn_prices, ref):>10.4f} "
      f"{relmae(pinn_prices, ref):>10.4f}  rho/r slightly OOD")
print(f"{'hainaut_orig':<20} {mse(hainaut_prices, ref):>12.4e} "
      f"{relmse(hainaut_prices, ref):>10.4f} "
      f"{relmae(hainaut_prices, ref):>10.4f}  in-distribution")

print("\nSample prices (S, ref, heston_pinn, hainaut_call):")
for i in [5, 15, 25, 35, 45]:
    S = S_GRID[i]
    print(f"  S={S:6.1f}  ref={ref[i]:8.4f}  "
          f"pinn={pinn_prices[i]:8.4f}  hainaut={hainaut_prices[i]:8.4f}")
