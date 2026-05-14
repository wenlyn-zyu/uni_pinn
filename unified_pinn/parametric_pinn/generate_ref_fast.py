"""
generate_ref_fast.py

Fast reference data generation using random parameter sampling.
Focuses on diversity over density — fewer points, broader coverage.

Strategy:
  - BSM:    200 random configs × 30 S-points = 6,000 anchors (fast)
  - CEV:    300 random configs × 30 S-points = 9,000 anchors (reasonable)
  - Heston: 400 random configs × 20 (S,v) points = 8,000 anchors (slowest)

Total: ~23,000 anchors, takes ~3-5 minutes.
"""

import argparse
import os
import sys
import numpy as np
import pickle
from scipy.stats import norm
from numpy.polynomial.legendre import leggauss

# ---------------------------------------------------------------------------
# GL quadrature for Heston
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


def heston_price_batch(S_arr, K, T, r, kappa, theta, xi, rho, v_arr):
    """Price a batch of (S, v) pairs for given Heston parameters."""
    if T < 1e-6:
        return np.maximum(S_arr - K, 0.0)
    try:
        Ks = np.array([K])
        phi, w = _GL_PHI, _GL_W
        prices = np.full(len(S_arr), np.nan)
        for i, (S, v0) in enumerate(zip(S_arr, v_arr)):
            cf1 = _heston_cf_batch(phi, S, Ks, T, r, kappa, theta, xi, rho, v0, 1)
            cf2 = _heston_cf_batch(phi, S, Ks, T, r, kappa, theta, xi, rho, v0, 2)
            phi2d = phi[:, None]
            I1 = (w[:, None] * np.real(cf1 / (1j * phi2d))).sum(axis=0)
            I2 = (w[:, None] * np.real(cf2 / (1j * phi2d))).sum(axis=0)
            price = S * (0.5 + I1[0] / np.pi) - K * np.exp(-r * T) * (0.5 + I2[0] / np.pi)
            prices[i] = float(max(price, 0.0))
        return prices
    except Exception:
        return np.full(len(S_arr), np.nan)


# ---------------------------------------------------------------------------
# Fast pricers
# ---------------------------------------------------------------------------

def bsm_prices(S_arr, K, T, r, sigma):
    if sigma <= 0 or T < 1e-8:
        return np.maximum(S_arr - K * np.exp(-r * T), 0.0)
    sqt = sigma * np.sqrt(T)
    d1 = (np.log(S_arr / K) + (r + 0.5 * sigma**2) * T) / sqt
    d2 = d1 - sqt
    return S_arr * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def cev_schroder_call(S, K, T, r, sigma, beta):
    from scipy.stats import ncx2
    if abs(beta - 1.0) < 1e-9:
        sqt = sigma * np.sqrt(max(T, 1e-10))
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / sqt
        d2 = d1 - sqt
        return float(max(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2), 0.0))
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
# Main generation
# ---------------------------------------------------------------------------

def generate(rng, n_bsm=200, n_cev=300, n_heston=400, n_s=30, n_heston_v=4):
    anchors = []

    # ---- BSM ----
    print(f"BSM: {n_bsm} random configs...")
    for i in range(n_bsm):
        K   = rng.uniform(50, 200)
        r   = rng.uniform(0.01, 0.10)
        tau = rng.uniform(0.02, 2.5)
        sigma = rng.uniform(0.08, 0.45)
        S_arr = rng.uniform(K * 0.5, K * 2.0, n_s)
        S_arr = np.clip(S_arr, 1.0, 500.0)
        v_arr = np.full(n_s, sigma**2)
        tau_arr = np.full(n_s, tau)
        V_arr = bsm_prices(S_arr, K, tau, r, sigma)
        lam_arr = np.tile([sigma, 1.0, 0.0, 0.0, 0.0, 0.0], (n_s, 1))
        anchors.append((S_arr.copy(), v_arr.copy(), tau_arr.copy(),
                        K, r, lam_arr.copy(), V_arr.copy()))
    print(f"  {n_bsm * n_s} points")

    # ---- CEV ----
    print(f"CEV: {n_cev} random configs...")
    for i in range(n_cev):
        K   = rng.uniform(50, 200)
        r   = rng.uniform(0.01, 0.10)
        tau = rng.uniform(0.02, 2.5)
        sigma = rng.uniform(0.10, 0.30)
        beta  = rng.uniform(0.10, 0.90)
        S_arr = rng.uniform(K * 0.4, K * 2.5, n_s)
        S_arr = np.clip(S_arr, 1.0, 500.0)
        v_arr = np.full(n_s, sigma**2)
        tau_arr = np.full(n_s, tau)
        V_arr = np.array([cev_schroder_call(s, K, tau, r, sigma, beta) for s in S_arr])
        lam_arr = np.tile([sigma, beta, 0.0, 0.0, 0.0, 0.0], (n_s, 1))
        anchors.append((S_arr.copy(), v_arr.copy(), tau_arr.copy(),
                        K, r, lam_arr.copy(), V_arr.copy()))
    print(f"  {n_cev * n_s} points")

    # ---- Heston ----
    print(f"Heston: {n_heston} random configs...")
    for i in range(n_heston):
        if i % 50 == 0:
            print(f"  [{i}/{n_heston}]...")

        K   = rng.uniform(50, 200)
        r   = rng.uniform(0.01, 0.10)
        tau = rng.uniform(0.02, 2.5)
        kappa = 10 ** rng.uniform(np.log10(0.5), np.log10(10))
        theta = rng.uniform(0.01, 0.10)
        xi    = 10 ** rng.uniform(np.log10(0.05), np.log10(0.50))
        rho   = rng.uniform(-0.95, -0.10)

        S_arr = rng.uniform(K * 0.5, K * 2.0, n_s)
        S_arr = np.clip(S_arr, 1.0, 500.0)
        v_arr = 10 ** rng.uniform(-3, np.log10(0.95), n_s * n_heston_v)
        S_arr = np.repeat(S_arr, n_heston_v)
        tau_arr = np.full(len(S_arr), tau)

        V_arr = heston_price_batch(S_arr, K, tau, r, kappa, theta, xi, rho, v_arr)
        valid = np.isfinite(V_arr)
        if valid.sum() < 5:
            continue

        lam_arr = np.tile([0.0, 1.0, kappa, theta, xi, rho], (len(S_arr), 1))
        anchors.append((S_arr[valid].copy(), v_arr[valid].copy(),
                        tau_arr[valid].copy(), K, r,
                        lam_arr[valid].copy(), V_arr[valid].copy()))

    total = sum(len(a[0]) for a in anchors)
    print(f"\nTotal: {total} anchor points from {len(anchors)} configs")
    return anchors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="results/ref_data_fast.pkl")
    parser.add_argument("--n-bsm", type=int, default=200)
    parser.add_argument("--n-cev", type=int, default=300)
    parser.add_argument("--n-heston", type=int, default=400)
    parser.add_argument("--n-s", type=int, default=30)
    parser.add_argument("--n-v", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    rng = np.random.RandomState(args.seed)

    anchors = generate(rng,
                       n_bsm=args.n_bsm,
                       n_cev=args.n_cev,
                       n_heston=args.n_heston,
                       n_s=args.n_s,
                       n_heston_v=args.n_v)

    print(f"Saving to {args.out}...")
    with open(args.out, "wb") as f:
        pickle.dump(anchors, f)
    print("Done.")


if __name__ == "__main__":
    main()
