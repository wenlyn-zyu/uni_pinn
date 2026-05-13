"""
train_v15_schr.py — Train unified PINN with CEV Schroder 1989 data anchor.

Key change from v15: CEV reference prices use non-central chi-squared
analytical formula (Schroder 1989) instead of Crank-Nicolson FD.

Anchors:
  BSM    -> Black-Scholes analytical
  CEV    -> Schroder 1989 non-central chi-squared
  Heston -> Gauss-Legendre semi-analytical (characteristic function)

Usage:
  python train_v15_schr.py [--epochs 30000] [--out results/unified_v15_schr.pt]
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


# ---------------------------------------------------------------------------
# BSM analytical reference
# ---------------------------------------------------------------------------

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
    # BSM: 6 sigma variants
    for sigma in [0.13, 0.15, 0.17, 0.20, 0.25, 0.30]:
        params.append(ModelParams.from_bsm(K=100., T=1., r=0.05, sigma=sigma))

    # CEV: 10 (sigma, beta) variants
    for beta in [0.1, 0.3, 0.5, 0.7, 0.9]:
        for sigma in [0.15, 0.20]:
            params.append(ModelParams.from_cev(K=100., T=1., r=0.05, sigma=sigma, beta=beta))

    # Heston: 36 (kappa, xi, rho) variants
    for kappa in [0.5, 2.0, 5.0, 8.0]:
        for xi in [0.1, 0.3, 0.5]:
            for rho in [-0.9, -0.7, -0.5]:
                params.append(ModelParams.from_heston(
                    K=100., T=1., r=0.05, kappa=kappa, theta=0.04,
                    xi=xi, rho=rho, v0=0.04))
    return params


# ---------------------------------------------------------------------------
# Build data anchors
# ---------------------------------------------------------------------------

def build_ref_data(param_list, n_s_anchor=200, n_heston=200):
    """Generate reference prices for data anchor loss.

    BSM  → Black-Scholes analytical
    CEV  → Schroder 1989 non-central chi-squared
    Heston → Gauss-Legendre semi-analytical
    """
    ref_data = {}
    for idx, p in enumerate(param_list):
        print(f"  Generating anchors [{idx+1}/{len(param_list)}] "
              f"{'BSM' if p.xi==0 and p.beta==1.0 else 'CEV' if p.xi==0 else 'Heston'} "
              f"σ={p.sigma:.2f} β={p.beta:.1f} κ={p.kappa:.1f} ξ={p.xi:.1f} ρ={p.rho:.1f}")

        if p.xi == 0:  # BSM or CEV (1D problem)
            S_arr = np.linspace(1.0, p.S_max * 0.98, n_s_anchor)
            v_arr = np.full_like(S_arr, p.v0)
            t_arr = np.zeros_like(S_arr)  # t=0 (today, pricing time)

            if p.beta == 1.0:  # BSM
                V_arr = bsm_prices(S_arr, p.K, p.T, p.r, p.sigma)
            else:  # CEV
                V_arr = np.array([
                    cev_schroder_call(s, p.K, p.T, p.r, p.sigma, p.beta)
                    for s in S_arr
                ])
        else:  # Heston (2D problem)
            S_pts = np.linspace(max(p.K * 0.5, 1.0), p.S_max * 0.98, n_heston // 4)
            v_pts = np.linspace(0.001, p.v_max * 0.95, 4)
            S_grid, v_grid = np.meshgrid(S_pts, v_pts)
            S_arr = S_grid.ravel()
            v_arr = v_grid.ravel()
            t_arr = np.zeros_like(S_arr)

            V_arr = np.array([
                heston_price_gl(s, p.K, p.T, p.r,
                               p.kappa, p.theta, p.xi, p.rho, p.v0 if v < 1e-6 else v)
                for s, v in zip(S_arr, v_arr)
            ])

        ref_data[idx] = (S_arr, v_arr, t_arr, V_arr)

    return ref_data


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", type=str, default="results/unified_v15_schr.pt")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"])
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    param_list = build_param_list()
    print(f"\nTotal model variants: {len(param_list)} "
          f"(BSM: 6, CEV: 10, Heston: 36)\n")

    print("Building reference data anchors...")
    ref_data = build_ref_data(param_list)
    n_total = sum(len(arr) for arr, _, _, _ in ref_data.values())
    print(f"  Total anchor points: {n_total}\n")

    print("Initializing UnifiedPINN...")
    model = UnifiedPINN(param_list, hidden=128, depth=6, lr=args.lr,
                        ref_data=ref_data, device=device)

    print(f"\nTraining {args.epochs} epochs...")
    history = model.train(
        epochs=args.epochs,
        n_per_model=5000,
        w_pde=1.0, w_bc=10.0, w_ic=10.0, w_data=100.0,
        log_every=500,
    )

    model.save(args.out)
    print(f"\nCheckpoint saved: {args.out}")

    # Save training history
    import json
    hist_out = args.out.replace(".pt", "_history.json")
    with open(hist_out, "w") as f:
        json.dump(history, f, indent=2)
    print(f"History saved: {hist_out}")


if __name__ == "__main__":
    main()
