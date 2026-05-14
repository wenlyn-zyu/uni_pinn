"""
Synthetic data evaluation: compare Heston PINN variants against GL reference.

Models:
  - heston_pinn (fixed-param, HESTON_INDEP): evaluated on its own training params
  - hainaut_orig (parametric): evaluated on params within its training range

Hainaut training ranges (from heston_hainaut.py):
  S∈[20,180], V∈[0.032,0.52], r∈[0.01,0.07]
  kappa∈[0.5,2.0], theta∈[0.062,0.42], xi∈[0.1,0.9], rho∈[-0.8,0.8], T∈[0.1,5.0]

heston_pinn training params (from checkpoint):
  kappa=1.0, theta=0.08, xi=0.39, rho=-0.93, v0=0.04, r=0.1, K=100, T=1.0
  Note: rho=-0.93 and r=0.1 are OOD for Hainaut.
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
# S grid within Hainaut training range [20, 180]
S_GRID_HAINAUT = np.linspace(30, 170, 50)
# S grid for heston_pinn (trained with S_max=400, K=100)
S_GRID_INDEP   = np.linspace(50, 250, 50)

# HESTON_INDEP: exact params from checkpoint
HESTON_INDEP = dict(kappa=1.0, theta=0.08, xi=0.39, rho=-0.93, v0=0.04)
r_INDEP = 0.1

# Hainaut in-distribution params: all within training ranges
# kappa∈[0.5,2.0], theta∈[0.062,0.42], xi∈[0.1,0.9], rho∈[-0.8,0.8], r∈[0.01,0.07]
HESTON_HAINAUT_ID = dict(kappa=1.15, theta=0.202, xi=0.20, rho=-0.40, v0=0.04)
r_HAINAUT = 0.04


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


# ── Eval 1: heston_pinn on its own training params ────────────────────────────
print("\n=== heston_pinn: in-distribution (HESTON_INDEP params) ===")
ref_indep = np.array([heston_call(S, K_SYN, T_SYN, r_INDEP, **HESTON_INDEP)
                      for S in S_GRID_INDEP])
pinn_indep = np.array([heston_pinn.price(S) for S in S_GRID_INDEP])

print(f"  MSE={mse(pinn_indep, ref_indep):.4e}  "
      f"RelMSE={relmse(pinn_indep, ref_indep):.4f}  "
      f"RelMAE={relmae(pinn_indep, ref_indep):.4f}")


# ── Eval 2: hainaut on in-distribution params ─────────────────────────────────
# Hainaut prices PUT at K=100 fixed; output = put/K_STRIKE
# put_abs = price() * K_STRIKE; call = put_abs + S - K*exp(-rT)
print("\n=== hainaut_orig: in-distribution (Hainaut ID params) ===")
print(f"  params: {HESTON_HAINAUT_ID}, r={r_HAINAUT}")
ref_hainaut_id = np.array([heston_call(S, K_SYN, T_SYN, r_HAINAUT, **HESTON_HAINAUT_ID)
                            for S in S_GRID_HAINAUT])

hainaut_id_prices = []
for S in S_GRID_HAINAUT:
    put_norm = hainaut.price(
        S=S, V=HESTON_HAINAUT_ID["v0"], t=0.0, T=T_SYN,
        r=r_HAINAUT,
        kappa=HESTON_HAINAUT_ID["kappa"],
        theta=HESTON_HAINAUT_ID["theta"],
        xi=HESTON_HAINAUT_ID["xi"],
        rho=HESTON_HAINAUT_ID["rho"],
    )
    put_abs = put_norm * HestonHainaut.K_STRIKE
    call = put_abs + S - K_SYN * np.exp(-r_HAINAUT * T_SYN)
    hainaut_id_prices.append(call)
hainaut_id_prices = np.array(hainaut_id_prices)

print(f"  MSE={mse(hainaut_id_prices, ref_hainaut_id):.4e}  "
      f"RelMSE={relmse(hainaut_id_prices, ref_hainaut_id):.4f}  "
      f"RelMAE={relmae(hainaut_id_prices, ref_hainaut_id):.4f}")

# Also show a few sample values for sanity check
print("\n  Sample prices (S, ref_call, hainaut_call, put_norm):")
for i in [10, 20, 30, 40, 49]:
    S = S_GRID_HAINAUT[i]
    put_norm = hainaut.price(
        S=S, V=HESTON_HAINAUT_ID["v0"], t=0.0, T=T_SYN,
        r=r_HAINAUT,
        kappa=HESTON_HAINAUT_ID["kappa"],
        theta=HESTON_HAINAUT_ID["theta"],
        xi=HESTON_HAINAUT_ID["xi"],
        rho=HESTON_HAINAUT_ID["rho"],
    )
    put_abs = put_norm * HestonHainaut.K_STRIKE
    call = put_abs + S - K_SYN * np.exp(-r_HAINAUT * T_SYN)
    ref = heston_call(S, K_SYN, T_SYN, r_HAINAUT, **HESTON_HAINAUT_ID)
    print(f"    S={S:.1f}  ref={ref:.4f}  hainaut={call:.4f}  put_norm={put_norm:.4f}")


# ── Summary ───────────────────────────────────────────────────────────────────
print("\n=== Summary ===")
print(f"{'Model':<20} {'Test params':<20} {'RelMSE':>8} {'RelMAE':>8} {'In-dist?'}")
print("-" * 70)
print(f"{'heston_pinn':<20} {'HESTON_INDEP':<20} "
      f"{relmse(pinn_indep, ref_indep):>8.4f} "
      f"{relmae(pinn_indep, ref_indep):>8.4f}  YES")
print(f"{'hainaut_orig':<20} {'Hainaut ID':<20} "
      f"{relmse(hainaut_id_prices, ref_hainaut_id):>8.4f} "
      f"{relmae(hainaut_id_prices, ref_hainaut_id):>8.4f}  YES")
