"""
eval_compare.py — Compare independent PINNs vs unified PINN on the same parameter set.

Unified evaluation parameters (all models):
  BSM:    K=100, T=1, r=0.05, sigma=0.20
  CEV:    K=100, T=1, r=0.05, sigma=0.20, beta=0.5
  Heston: K=100, T=1, r=0.05, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, v0=0.04

Reference solutions:
  BSM:    Black-Scholes analytical formula
  CEV:    Schroder (1989) non-central chi-square analytical solution
  Heston: GL characteristic function semi-analytical solution

Eval grid: S in [60, 160], 50 uniformly spaced points, t=0

Usage (from project root):
  python eval_compare.py

Checkpoints expected at:
  results/indep_bsm.pt
  results/indep_cev.pt
  results/indep_heston.pt       (or indep_heston_unified.pt)
  results/unified_v15.pt
"""

import os
import sys
import numpy as np

_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _root)

RESULTS_DIR = os.path.join(_root, "results")

# ── Reference prices ──────────────────────────────────────────────────────────

def bsm_ref(spots):
    from independent.bsm_pinn import bs_call_price
    return np.array([bs_call_price(S, 100, 1.0, 0.05, 0.20) for S in spots])


def cev_ref(spots):
    from independent.cev_pinn import cev_analytical_call
    return np.array([cev_analytical_call(S, 100, 1.0, 0.05, 0.20, 0.5) for S in spots])


def heston_ref(spots):
    from independent.heston_pinn import heston_call_price
    return np.array([
        heston_call_price(S, 100, 1.0, 0.05, 2.0, 0.04, 0.3, -0.7, 0.04)
        for S in spots
    ])


# ── Independent PINN prices ───────────────────────────────────────────────────

def indep_bsm_prices(spots):
    from independent.bsm_pinn import BSM_PINN
    ckpt = os.path.join(RESULTS_DIR, "indep_bsm.pt")
    m = BSM_PINN(K=100, T=1.0, r=0.05, sigma=0.20, S_max=300)
    m.load(ckpt)
    return np.array([m.price(S, t=0.0) for S in spots])


def indep_cev_prices(spots):
    from independent.cev_pinn import CEV_PINN
    ckpt = os.path.join(RESULTS_DIR, "indep_cev.pt")
    m = CEV_PINN(K=100, T=1.0, r=0.05, sigma=0.20, beta=0.5, S_max=300)
    m.load(ckpt)
    return np.array([m.price(S, t=0.0) for S in spots])


def indep_heston_prices(spots):
    from independent.heston_pinn import Heston_PINN
    # Try unified-params checkpoint first, fall back to default name
    for name in ("indep_heston_unified.pt", "indep_heston.pt"):
        ckpt = os.path.join(RESULTS_DIR, name)
        if os.path.exists(ckpt):
            break
    m = Heston_PINN(K=100, T=1.0, r=0.05,
                    kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, v0=0.04,
                    S_max=300, v_max=1.0)
    m.load(ckpt)
    return np.array([m.price(S, v=0.04, t=0.0) for S in spots])


# ── Unified PINN prices ───────────────────────────────────────────────────────

def load_unified():
    # Import from server-side unified_pinn_v2 or local parametric_pinn
    try:
        from unified_pinn_v2 import ModelParams, UnifiedPINN
    except ImportError:
        raise ImportError(
            "unified_pinn_v2 not found. Run this script on the server where "
            "unified_pinn_v2.py is available, or copy it to the project root."
        )
    param_list = []
    for sigma in [0.13, 0.15, 0.17, 0.20, 0.25, 0.30]:
        param_list.append(ModelParams.from_bsm(K=100., T=1., r=0.05, sigma=sigma))
    for beta in [0.1, 0.3, 0.5, 0.7, 0.9]:
        for sigma in [0.15, 0.20]:
            param_list.append(ModelParams.from_cev(K=100., T=1., r=0.05, sigma=sigma, beta=beta))
    for kappa in [0.5, 2.0, 5.0, 8.0]:
        for xi in [0.1, 0.3, 0.5]:
            for rho in [-0.9, -0.7, -0.5]:
                param_list.append(ModelParams.from_heston(
                    K=100., T=1., r=0.05, kappa=kappa, theta=0.04,
                    xi=xi, rho=rho, v0=0.04))
    ckpt = os.path.join(RESULTS_DIR, "unified_v15.pt")
    m = UnifiedPINN(param_list, hidden=128, depth=6)
    m.load(ckpt)
    return m


def unified_bsm_prices(spots, m):
    from unified_pinn_v2 import ModelParams
    p = ModelParams.from_bsm(K=100., T=1., r=0.05, sigma=0.20)
    return np.array([m.price(p, S=S, t=0.0) for S in spots])


def unified_cev_prices(spots, m):
    from unified_pinn_v2 import ModelParams
    p = ModelParams.from_cev(K=100., T=1., r=0.05, sigma=0.20, beta=0.5)
    return np.array([m.price(p, S=S, t=0.0) for S in spots])


def unified_heston_prices(spots, m):
    from unified_pinn_v2 import ModelParams
    p = ModelParams.from_heston(K=100., T=1., r=0.05,
                                kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, v0=0.04)
    return np.array([m.price(p, S=S, v=0.04, t=0.0) for S in spots])


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(pred, ref, atm_lo=80, atm_hi=120, spots=None):
    err = np.abs(pred - ref)
    mae  = float(np.mean(err))
    rmse = float(np.sqrt(np.mean((pred - ref) ** 2)))
    if spots is not None:
        atm_mask = (spots >= atm_lo) & (spots <= atm_hi)
        ref_atm = ref[atm_mask]
        err_atm = err[atm_mask]
        relmae = float(np.mean(err_atm / (np.abs(ref_atm) + 1e-8)))
    else:
        mask = ref > 0.5
        relmae = float(np.mean(err[mask] / ref[mask])) if mask.any() else float("nan")
    return mae, rmse, relmae


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    spots = np.linspace(60, 160, 50)

    print("Loading unified PINN v15...", flush=True)
    try:
        uni = load_unified()
        has_unified = True
    except ImportError as e:
        print(f"  Warning: {e}")
        print("  Skipping unified PINN comparison.")
        has_unified = False

    header = f"{'Method':<28}  {'MAE':>8}  {'RMSE':>8}  {'RelMAE(ATM)':>12}"
    sep = "-" * 62

    for model_name, ref_fn, indep_fn, unified_fn in [
        ("BSM",    bsm_ref,    indep_bsm_prices,    unified_bsm_prices    if has_unified else None),
        ("CEV",    cev_ref,    indep_cev_prices,    unified_cev_prices    if has_unified else None),
        ("Heston", heston_ref, indep_heston_prices, unified_heston_prices if has_unified else None),
    ]:
        print(f"\n{'─'*20} {model_name} {'─'*20}")
        print(header)
        print(sep)

        ref = ref_fn(spots)

        try:
            pred = indep_fn(spots)
            mae, rmse, relmae = compute_metrics(pred, ref, spots=spots)
            print(f"  {'Indep-PINN-' + model_name:<26}  {mae:8.4f}  {rmse:8.4f}  {100*relmae:10.2f}%")
        except FileNotFoundError as e:
            print(f"  Indep-PINN-{model_name}: checkpoint not found ({e})")

        if has_unified and unified_fn is not None:
            try:
                pred = unified_fn(spots, uni)
                mae, rmse, relmae = compute_metrics(pred, ref, spots=spots)
                print(f"  {'Unified-PINN-' + model_name:<26}  {mae:8.4f}  {rmse:8.4f}  {100*relmae:10.2f}%")
            except Exception as e:
                print(f"  Unified-PINN-{model_name}: error ({e})")


if __name__ == "__main__":
    main()
