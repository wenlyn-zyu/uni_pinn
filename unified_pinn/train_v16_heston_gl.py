"""
train_v16_heston_gl.py — Upgraded Heston data anchors for unified PINN.

Key changes from v15_schr:
  - Heston anchor density increased from 200 to 800 per variant (100 S × 8 v)
  - Denser ATM coverage (log-spaced strikes around K)
  - More anchor points at t=0.5*T for time-domain regularization
  - Higher data loss weight for Heston variants

Anchors:
  BSM    -> Black-Scholes analytical
  CEV    -> Schroder 1989 non-central chi-squared
  Heston -> Gauss-Legendre semi-analytical (denser grid)

Usage:
  python train_v16_heston_gl.py --epochs 30000 --out results/unified_v16_gl.pt
"""

import argparse
import os
import sys
import numpy as np
import torch
from scipy.stats import norm
from numpy.polynomial.legendre import leggauss

sys.path.insert(0, os.path.dirname(__file__))
from unified_pinn_v2 import ModelParams, UnifiedPINN, cev_schroder_call

# ---------------------------------------------------------------------------
# GL quadrature for Heston reference
# ---------------------------------------------------------------------------
_N_GL = 96
_GL_NODES, _GL_WEIGHTS = leggauss(_N_GL)
_GL_PHI_MAX = 100.0
_GL_PHI = (_GL_NODES + 1.0) * 0.5 * _GL_PHI_MAX
_GL_W   = _GL_WEIGHTS * 0.5 * _GL_PHI_MAX


def _heston_cf_batch(phi_arr, S, Ks, T, r, kappa, theta, xi, rho, v0, j):
    i = 1j
    u, b = (0.5, kappa - rho * xi) if j == 1 else (-0.5, kappa)
    a = kappa * theta
    x = np.log(S / Ks)
    phi = phi_arr[:, None]; x2d = x[None, :]
    d_sqrt = np.sqrt((rho * xi * i * phi - b)**2 - xi**2 * (2 * u * i * phi - phi**2))
    num = b - rho * xi * i * phi + d_sqrt
    g = num / (b - rho * xi * i * phi - d_sqrt)
    exp_dT = np.exp(d_sqrt * T)
    C = (r * i * phi * T
         + (a / xi**2) * (num * T - 2.0 * np.log((1.0 - g * exp_dT) / (1.0 - g))))
    D = (num / xi**2) * ((1.0 - exp_dT) / (1.0 - g * exp_dT))
    return np.exp(C + D * v0 + i * phi * x2d)


def heston_price_gl(S, K, T, r, kappa, theta, xi, rho, v0):
    if T < 1e-6:
        return max(S - K, 0.0)
    Ks = np.array([K])
    phi, w = _GL_PHI, _GL_W
    cf1 = _heston_cf_batch(phi, S, Ks, T, r, kappa, theta, xi, rho, v0, 1)
    cf2 = _heston_cf_batch(phi, S, Ks, T, r, kappa, theta, xi, rho, v0, 2)
    phi2d = phi[:, None]
    I1 = (w[:, None] * np.real(cf1 / (1j * phi2d))).sum(axis=0)
    I2 = (w[:, None] * np.real(cf2 / (1j * phi2d))).sum(axis=0)
    price = S * (0.5 + I1[0] / np.pi) - K * np.exp(-r * T) * (0.5 + I2[0] / np.pi)
    return float(max(price, 0.0))


def bsm_prices(S_arr, K, T, r, sigma):
    sqt = sigma * np.sqrt(max(T, 1e-10))
    d1 = (np.log(S_arr / K) + (r + 0.5 * sigma**2) * T) / sqt
    d2 = d1 - sqt
    return S_arr * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


# ---------------------------------------------------------------------------
# Build param list (same as v15)
# ---------------------------------------------------------------------------
def build_param_list():
    params = []
    for sigma in [0.13, 0.15, 0.17, 0.20, 0.25, 0.30]:
        params.append(ModelParams.from_bsm(K=100., T=1., r=0.05, sigma=sigma))
    for beta in [0.1, 0.3, 0.5, 0.7, 0.9]:
        for sigma in [0.15, 0.20]:
            params.append(ModelParams.from_cev(K=100., T=1., r=0.05, sigma=sigma, beta=beta))
    for kappa in [0.5, 2.0, 5.0, 8.0]:
        for xi in [0.1, 0.3, 0.5]:
            for rho in [-0.9, -0.7, -0.5]:
                params.append(ModelParams.from_heston(
                    K=100., T=1., r=0.05, kappa=kappa, theta=0.04,
                    xi=xi, rho=rho, v0=0.04))
    return params


# ---------------------------------------------------------------------------
# Build data anchors (upgraded Heston)
# ---------------------------------------------------------------------------
def build_ref_data(param_list, n_s_anchor=200, n_heston_s=100, n_heston_v=8):
    """Generate reference prices for data anchor loss.

    BSM/CEV → 200 S-points at t=0 (unchanged from v15)
    Heston → 100 S × 8 v = 800 points at t=0 (4× denser than v15)
             + 50 S × 4 v = 200 points at t=0.5T (time-domain anchors)
    """
    ref_data = {}
    for idx, p in enumerate(param_list):
        model_type = 'BSM' if p.xi == 0 and p.beta == 1.0 else \
                     'CEV' if p.xi == 0 else 'Heston'
        print(f"  Generating anchors [{idx+1}/{len(param_list)}] {model_type} "
              f"σ={p.sigma:.2f} β={p.beta:.1f} κ={p.kappa:.1f} ξ={p.xi:.1f} ρ={p.rho:.1f}")

        if p.xi == 0:  # BSM or CEV (1D — unchanged)
            S_arr = np.linspace(1.0, p.S_max * 0.98, n_s_anchor)
            v_arr = np.full_like(S_arr, p.v0)
            t_arr = np.zeros_like(S_arr)
            if p.beta == 1.0:
                V_arr = bsm_prices(S_arr, p.K, p.T, p.r, p.sigma)
            else:
                V_arr = np.array([cev_schroder_call(s, p.K, p.T, p.r, p.sigma, p.beta)
                                  for s in S_arr])
        else:  # Heston — upgraded anchors
            # t=0 anchors: denser S-v grid
            S_pos = np.linspace(max(p.K * 0.3, 1.0), p.S_max * 0.98, n_heston_s)
            v_pts = np.logspace(-3, np.log10(p.v_max * 0.95), n_heston_v)
            S_grid, v_grid = np.meshgrid(S_pos, v_pts)
            S_arr = S_grid.ravel()
            v_arr = v_grid.ravel()
            t_arr = np.zeros_like(S_arr)

            V_arr = np.array([
                heston_price_gl(s, p.K, p.T, p.r,
                                p.kappa, p.theta, p.xi, p.rho,
                                p.v0 if v < 1e-6 else v)
                for s, v in zip(S_arr, v_arr)
            ])

            # t=0.5*T anchors: additional time-domain regularization
            t_mid = 0.5 * p.T
            S_mid = np.linspace(max(p.K * 0.5, 1.0), p.S_max * 0.95, 50)
            v_mid = np.logspace(-3, np.log10(p.v_max * 0.95), 4)
            Sg, vg = np.meshgrid(S_mid, v_mid)
            S_arr_mid = Sg.ravel()
            v_arr_mid = vg.ravel()
            t_arr_mid = np.full_like(S_arr_mid, t_mid)

            # For t>0 anchors, use the GL price at time-to-maturity = T - t
            V_arr_mid = np.array([
                heston_price_gl(s, p.K, p.T - t_mid, p.r,
                                p.kappa, p.theta, p.xi, p.rho,
                                p.v0 if v < 1e-6 else v)
                for s, v in zip(S_arr_mid, v_arr_mid)
            ])

            S_arr = np.concatenate([S_arr, S_arr_mid])
            v_arr = np.concatenate([v_arr, v_arr_mid])
            t_arr = np.concatenate([t_arr, t_arr_mid])
            V_arr = np.concatenate([V_arr, V_arr_mid])

        ref_data[idx] = (S_arr, v_arr, t_arr, V_arr)

    return ref_data


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", type=str, default="results/unified_v16_gl.pt")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-every", type=int, default=0,
                        help="Save checkpoint every N epochs")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    param_list = build_param_list()
    print(f"\nTotal model variants: {len(param_list)} "
          f"(BSM: 6, CEV: 10, Heston: 36)\n")

    print("Building reference data anchors (upgraded Heston)...")
    ref_data = build_ref_data(param_list)
    n_total = sum(len(arr) for arr, _, _, _ in ref_data.values())
    n_heston_total = sum(len(arr) for i, (arr, _, _, _) in ref_data.items()
                         if param_list[i].xi > 0)
    print(f"  Total anchor points: {n_total}")
    print(f"  Heston anchor points: {n_heston_total} (upgraded from 7200 in v15)")
    print()

    print("Initializing UnifiedPINN...")
    model = UnifiedPINN(param_list, hidden=128, depth=6, lr=args.lr,
                        ref_data=ref_data, device=device)

    print(f"\nTraining {args.epochs} epochs...")
    history = model.train(
        epochs=args.epochs,
        n_per_model=5000,
        w_pde=1.0, w_bc=10.0, w_ic=10.0, w_data=100.0,
        log_every=500,
        save_every=args.save_every if args.save_every > 0 else 0,
        save_path=args.out if args.save_every > 0 else None,
    )

    model.save(args.out)
    print(f"\nCheckpoint saved: {args.out}")

    import json
    hist_out = args.out.replace(".pt", "_history.json")
    with open(hist_out, "w") as f:
        json.dump(history, f, indent=2)
    print(f"History saved: {hist_out}")


if __name__ == "__main__":
    main()
