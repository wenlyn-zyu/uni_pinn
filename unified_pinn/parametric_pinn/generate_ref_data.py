"""
generate_ref_data.py

Pre-generate reference price anchors covering the full parameter space.
Uses:
  BSM    → Black-Scholes analytical
  CEV    → Schroder 1989 non-central chi-squared
  Heston → Gauss-Legendre (96-node) semi-analytical

The anchor set is saved to disk and loaded during training.
"""

import argparse
import os
import sys
import numpy as np
import pickle
from scipy.stats import norm
from numpy.polynomial.legendre import leggauss

sys.path.insert(0, os.path.dirname(__file__))

# ---------------------------------------------------------------------------
# GL quadrature setup
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
    phi = phi_arr[:, None]
    x2d = x[None, :]
    d_sqrt = np.sqrt((rho * xi * i * phi - b)**2 - xi**2 * (2 * u * i * phi - phi**2))
    num = b - rho * xi * i * phi + d_sqrt
    g = num / (b - rho * xi * i * phi - d_sqrt)
    exp_dT = np.exp(d_sqrt * T)
    C = (r * i * phi * T
         + (a / xi**2) * (num * T - 2.0 * np.log((1.0 - g * exp_dT) / (1.0 - g))))
    D = (num / xi**2) * ((1.0 - exp_dT) / (1.0 - g * exp_dT))
    return np.exp(C + D * v0 + i * phi * x2d)


def heston_price_gl(S, K, T, r, kappa, theta, xi, rho, v0):
    """Vectorized Heston call price via Gauss-Legendre integration."""
    if T < 1e-6:
        return max(S - K, 0.0)
    try:
        Ks = np.array([K])
        phi, w = _GL_PHI, _GL_W
        cf1 = _heston_cf_batch(phi, S, Ks, T, r, kappa, theta, xi, rho, v0, 1)
        cf2 = _heston_cf_batch(phi, S, Ks, T, r, kappa, theta, xi, rho, v0, 2)
        phi2d = phi[:, None]
        I1 = (w[:, None] * np.real(cf1 / (1j * phi2d))).sum(axis=0)
        I2 = (w[:, None] * np.real(cf2 / (1j * phi2d))).sum(axis=0)
        price = S * (0.5 + I1[0] / np.pi) - K * np.exp(-r * T) * (0.5 + I2[0] / np.pi)
        return float(max(price, 0.0))
    except Exception:
        return np.nan


# ---------------------------------------------------------------------------
# BSM analytical
# ---------------------------------------------------------------------------

def bsm_price(S, K, T, r, sigma):
    if sigma <= 0 or T < 1e-8:
        return max(S - K * np.exp(-r * T), 0.0)
    sqt = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / sqt
    d2 = d1 - sqt
    return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))


# ---------------------------------------------------------------------------
# CEV analytical (Schroder 1989)
# ---------------------------------------------------------------------------

def cev_schroder_call(S, K, T, r, sigma, beta):
    from scipy.stats import ncx2
    if abs(beta - 1.0) < 1e-9:
        return bsm_price(S, K, T, r, sigma)
    delta = 1.0 - beta
    nu    = 1.0 / delta
    lam   = (2.0 * r) / (sigma**2 * delta * (np.exp(2.0 * r * delta * T) - 1.0))
    x     = lam * S**(2.0 * delta) * np.exp(2.0 * r * delta * T)
    y     = lam * K**(2.0 * delta)
    d     = 2.0 + nu
    call  = (S * (1.0 - ncx2.cdf(y, df=d,   nc=x))
             - K * np.exp(-r * T) * ncx2.cdf(x, df=d - 2, nc=y))
    return float(max(call, max(S - K * np.exp(-r * T), 0.0)))


# ---------------------------------------------------------------------------
# Anchor generation
# ---------------------------------------------------------------------------

def generate_anchors(n_s_per_model: int = 30,
                     n_tau_per_model: int = 5,
                     n_k_per_model: int = 6,
                     n_r_per_model: int = 3,
                     n_heston_v: int = 5,
                     seed: int = 42):
    """Generate reference price anchors covering the full parameter space.

    Args:
        n_s_per_model:   S (moneyness) points per anchor
        n_tau_per_model: tau values per anchor
        n_k_per_model:   K values
        n_r_per_model:   r values
        n_heston_v:      v points for Heston anchors

    Returns:
        list of (S_arr, v_arr, tau_arr, K_val, r_val, lam_arr, V_arr) tuples
    """
    rng = np.random.RandomState(seed)
    anchors = []

    # Parameter grids
    K_vals  = np.linspace(60, 180, n_k_per_model)
    r_vals  = np.linspace(0.01, 0.08, n_r_per_model)
    tau_vals = np.logspace(-2, np.log10(2.5), n_tau_per_model)  # 0.01 to 2.5 years

    # =====================================================================
    # BSM anchors
    # =====================================================================
    print("Generating BSM anchors...")
    bsm_count = 0
    sigma_vals = np.linspace(0.08, 0.45, 10)
    for K in K_vals:
        for r in r_vals:
            for sigma in sigma_vals:
                for tau in tau_vals:
                    # S grid: moneyness from 0.5 to 2.0
                    S_arr = np.linspace(K * 0.5, K * 2.0, n_s_per_model)
                    S_arr = np.clip(S_arr, 1.0, 500.0)
                    v_arr = np.full_like(S_arr, sigma**2)
                    tau_arr = np.full_like(S_arr, tau)
                    V_arr = np.array([bsm_price(s, K, tau, r, sigma) for s in S_arr])
                    lam_arr = np.tile([sigma, 1.0, 0.0, 0.0, 0.0, 0.0], (len(S_arr), 1))

                    anchors.append((S_arr.copy(), v_arr.copy(), tau_arr.copy(),
                                    K, r, lam_arr.copy(), V_arr.copy()))
                    bsm_count += len(S_arr)

    print(f"  BSM: {bsm_count} anchor points from {len(anchors)} configs")

    # =====================================================================
    # CEV anchors
    # =====================================================================
    print("Generating CEV anchors...")
    cev_count = 0
    sigma_cev_vals = [0.15, 0.20]
    beta_vals = np.linspace(0.15, 0.85, 8)
    for K in K_vals:
        for r in r_vals:
            for sigma in sigma_cev_vals:
                for beta in beta_vals:
                    for tau in tau_vals:
                        S_arr = np.linspace(K * 0.4, K * 2.5, n_s_per_model)
                        S_arr = np.clip(S_arr, 1.0, 500.0)
                        v_arr = np.full_like(S_arr, sigma**2)
                        tau_arr = np.full_like(S_arr, tau)
                        V_arr = np.array([
                            cev_schroder_call(s, K, tau, r, sigma, beta)
                            for s in S_arr
                        ])
                        lam_arr = np.tile([sigma, beta, 0.0, 0.0, 0.0, 0.0], (len(S_arr), 1))

                        anchors.append((S_arr.copy(), v_arr.copy(), tau_arr.copy(),
                                        K, r, lam_arr.copy(), V_arr.copy()))
                        cev_count += len(S_arr)

    print(f"  CEV: {cev_count} anchor points from {len(anchors)} configs")

    # =====================================================================
    # Heston anchors
    # =====================================================================
    print("Generating Heston anchors (this is the slow part)...")
    heston_count = 0
    heston_configs = []

    # Representative Heston parameter combinations
    kappa_vals  = [0.5, 2.0, 5.0, 8.0]
    theta_vals  = [0.02, 0.04, 0.08]
    xi_vals     = [0.10, 0.30, 0.50]
    rho_vals    = [-0.9, -0.7, -0.5, -0.3]

    for kappa in kappa_vals:
        for theta in theta_vals:
            for xi in xi_vals:
                for rho in rho_vals:
                    heston_configs.append((kappa, theta, xi, rho))

    # Also add random Heston configs for better coverage
    for _ in range(50):
        kappa = 10 ** rng.uniform(np.log10(0.5), np.log10(10))
        theta = rng.uniform(0.01, 0.10)
        xi    = 10 ** rng.uniform(np.log10(0.05), np.log10(0.50))
        rho   = rng.uniform(-0.95, -0.10)
        heston_configs.append((kappa, theta, xi, rho))

    total_heston = len(heston_configs)
    for i, (kappa, theta, xi, rho) in enumerate(heston_configs):
        if i % 20 == 0:
            print(f"  Heston [{i+1}/{total_heston}]...")

        for K in K_vals:
            for r in r_vals:
                for tau in tau_vals:
                    v0 = theta
                    S_arr = np.linspace(K * 0.4, K * 2.5, n_s_per_model)
                    S_arr = np.clip(S_arr, 1.0, 500.0)

                    # For Heston, we have v as a true state variable
                    # Generate a grid of (S, v, tau) points
                    S_list, v_list, tau_list, V_list = [], [], [], []

                    # At each S, sample multiple v values
                    v_pts = np.logspace(-3, np.log10(0.95), n_heston_v)
                    for s in S_arr:
                        for v_val in v_pts:
                            S_list.append(s)
                            v_list.append(v_val)
                            tau_list.append(tau)
                            V_list.append(
                                heston_price_gl(s, K, tau, r, kappa, theta, xi, rho, v_val)
                            )

                    S_a = np.array(S_list)
                    v_a = np.array(v_list)
                    tau_a = np.array(tau_list)
                    V_a = np.array(V_list)

                    # Filter NaN
                    valid = np.isfinite(V_a)
                    if valid.sum() < 10:
                        continue

                    lam_a = np.tile([0.0, 1.0, kappa, theta, xi, rho],
                                    (len(S_a), 1))
                    anchors.append((S_a[valid].copy(), v_a[valid].copy(),
                                    tau_a[valid].copy(), K, r,
                                    lam_a[valid].copy(), V_a[valid].copy()))
                    heston_count += valid.sum()

    print(f"  Heston: {heston_count} anchor points")
    print(f"\nTotal: {sum(len(a[0]) for a in anchors)} anchor points from {len(anchors)} configs")
    return anchors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="results/ref_data.pkl")
    parser.add_argument("--n-s", type=int, default=30,
                        help="S (moneyness) points per tau-slice")
    parser.add_argument("--n-tau", type=int, default=5,
                        help="tau values per config")
    parser.add_argument("--n-k", type=int, default=5,
                        help="K values")
    parser.add_argument("--n-r", type=int, default=3,
                        help="r values")
    parser.add_argument("--n-v", type=int, default=5,
                        help="v points for Heston")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    anchors = generate_anchors(
        n_s_per_model=args.n_s,
        n_tau_per_model=args.n_tau,
        n_k_per_model=args.n_k,
        n_r_per_model=args.n_r,
        n_heston_v=args.n_v,
        seed=args.seed,
    )

    print(f"\nSaving to {args.out}...")
    with open(args.out, "wb") as f:
        pickle.dump(anchors, f)
    print("Done.")


if __name__ == "__main__":
    main()
