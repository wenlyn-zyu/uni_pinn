"""
Synthetic data evaluation: compare Heston PINN variants against GL reference.

Models:
  - heston_pinn.py  (fixed-param, HESTON_INDEP: kappa=1.0, xi=0.39)
  - heston_hainaut.py (parametric, original range, checkpoint: results/hainaut.pt)

Reference: GL 96-point Gauss-Legendre semi-analytical Heston call price.

Test grid: S in linspace(50, 250, 50), K=100, T=1.0, r=0.05
  - For heston_pinn: uses HESTON_INDEP params (kappa=1.0, theta=0.08, xi=0.39, rho=-0.93, v0=0.04)
  - For hainaut: uses HESTON_INDEP params (within its training range? kappa=1.0 is OOD for Hainaut)
    Also tested with Hainaut's own default params (kappa=0.2, theta=0.202, xi=0.20, rho=-0.40)
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

# ── Parameters ────────────────────────────────────────────────────────────────
K_SYN = 100.0
T_SYN = 1.0
S_GRID = np.linspace(50, 250, 50)

# HESTON_INDEP: parameters used when training heston_pinn.py
HESTON_INDEP = dict(kappa=1.0, theta=0.08, xi=0.39, rho=-0.93, v0=0.04)
r_INDEP = 0.1

# Hainaut paper default eval params (Table 6) — within its training range
HESTON_HAINAUT_DEFAULT = dict(kappa=0.2, theta=0.202, xi=0.20, rho=-0.40, v0=0.04)
r_HAINAUT = 0.04


def mse(pred, ref):
    return float(np.mean((np.array(pred) - np.array(ref)) ** 2))


def relmse(pred, ref):
    ref = np.array(ref)
    pred = np.array(pred)
    mask = ref > 0.01
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(((pred[mask] - ref[mask]) / ref[mask]) ** 2))


def relmae(pred, ref):
    ref = np.array(ref)
    pred = np.array(pred)
    mask = np.abs(ref) > 0.01
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((pred[mask] - ref[mask]) / ref[mask])))


# ── Load models ───────────────────────────────────────────────────────────────
print("Loading heston_pinn (fixed-param, HESTON_INDEP)...")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = torch.load(os.path.join(BASE, "results/indep_heston.pt"), map_location=DEVICE)
heston_pinn = Heston_PINN(**ckpt["params"], device=DEVICE)
heston_pinn.aux_net.load_state_dict(ckpt["aux_state"])
heston_pinn.main_net.load_state_dict(ckpt["main_state"])
heston_pinn.aux_net.eval()
heston_pinn.main_net.eval()
print("  Loaded. params:", ckpt["params"])

print("Loading hainaut (parametric, original range)...")
hainaut = HestonHainaut()
hainaut.load(os.path.join(BASE, "results/hainaut.pt"))
print("  Loaded.")


# ── Eval 1: HESTON_INDEP params ───────────────────────────────────────────────
print("\n=== Test set: HESTON_INDEP params (kappa=1.0, xi=0.39, r=0.1) ===")
print("  (heston_pinn trained on these; hainaut kappa=1.0 is OOD for original range)")

ref_indep = np.array([
    heston_call(S, K_SYN, T_SYN, r_INDEP, **HESTON_INDEP) for S in S_GRID
])

# heston_pinn inference
pinn_indep = np.array([heston_pinn.price(S) for S in S_GRID])

# hainaut inference: price PUT then convert to CALL via put-call parity
# hainaut.price(S, V, t, T, r, kappa, theta, xi, rho) -> put price (normalised)
# actual put = put_n * (S_spot / K_REF), but here S_spot=K_REF=100 so no scaling
hainaut_indep = []
for S in S_GRID:
    put_n = hainaut.price(
        S=K_SYN, V=HESTON_INDEP["v0"], t=0.0, T=T_SYN,
        r=r_INDEP,
        kappa=HESTON_INDEP["kappa"],
        theta=HESTON_INDEP["theta"],
        xi=HESTON_INDEP["xi"],
        rho=HESTON_INDEP["rho"],
    )
    # put-call parity: C = P + S - K*exp(-rT)
    call = put_n * (S / K_SYN) + S - K_SYN * np.exp(-r_INDEP * T_SYN)
    hainaut_indep.append(call)
hainaut_indep = np.array(hainaut_indep)

print(f"\n  heston_pinn  MSE={mse(pinn_indep, ref_indep):.4e}  "
      f"RelMSE={relmse(pinn_indep, ref_indep):.4f}  "
      f"RelMAE={relmae(pinn_indep, ref_indep):.4f}")
print(f"  hainaut_orig MSE={mse(hainaut_indep, ref_indep):.4e}  "
      f"RelMSE={relmse(hainaut_indep, ref_indep):.4f}  "
      f"RelMAE={relmae(hainaut_indep, ref_indep):.4f}")

# ── Eval 2: Hainaut default params (within its training range) ────────────────
print("\n=== Test set: Hainaut default params (kappa=0.2, xi=0.20, r=0.04) ===")
print("  (both models evaluated; heston_pinn is OOD for these params)")

ref_hainaut = np.array([
    heston_call(S, K_SYN, T_SYN, r_HAINAUT, **HESTON_HAINAUT_DEFAULT) for S in S_GRID
])

# heston_pinn: OOD (different params), but run anyway for comparison
pinn_hainaut = np.array([
    heston_pinn.price(S, v=HESTON_HAINAUT_DEFAULT["v0"]) for S in S_GRID
])

hainaut_default = []
for S in S_GRID:
    put_n = hainaut.price(
        S=K_SYN, V=HESTON_HAINAUT_DEFAULT["v0"], t=0.0, T=T_SYN,
        r=r_HAINAUT,
        kappa=HESTON_HAINAUT_DEFAULT["kappa"],
        theta=HESTON_HAINAUT_DEFAULT["theta"],
        xi=HESTON_HAINAUT_DEFAULT["xi"],
        rho=HESTON_HAINAUT_DEFAULT["rho"],
    )
    call = put_n * (S / K_SYN) + S - K_SYN * np.exp(-r_HAINAUT * T_SYN)
    hainaut_default.append(call)
hainaut_default = np.array(hainaut_default)

print(f"\n  heston_pinn  MSE={mse(pinn_hainaut, ref_hainaut):.4e}  "
      f"RelMSE={relmse(pinn_hainaut, ref_hainaut):.4f}  "
      f"RelMAE={relmae(pinn_hainaut, ref_hainaut):.4f}  [OOD]")
print(f"  hainaut_orig MSE={mse(hainaut_default, ref_hainaut):.4e}  "
      f"RelMSE={relmse(hainaut_default, ref_hainaut):.4f}  "
      f"RelMAE={relmae(hainaut_default, ref_hainaut):.4f}  [in-distribution]")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n=== Summary ===")
print("Model            | Test params      | RelMSE  | RelMAE  | In-dist?")
print("-" * 70)
print(f"heston_pinn      | HESTON_INDEP     | {relmse(pinn_indep, ref_indep):.4f}  | {relmae(pinn_indep, ref_indep):.4f}  | YES")
print(f"hainaut_orig     | HESTON_INDEP     | {relmse(hainaut_indep, ref_indep):.4f}  | {relmae(hainaut_indep, ref_indep):.4f}  | NO (kappa OOD)")
print(f"heston_pinn      | Hainaut default  | {relmse(pinn_hainaut, ref_hainaut):.4f}  | {relmae(pinn_hainaut, ref_hainaut):.4f}  | NO")
print(f"hainaut_orig     | Hainaut default  | {relmse(hainaut_default, ref_hainaut):.4f}  | {relmae(hainaut_default, ref_hainaut):.4f}  | YES")
