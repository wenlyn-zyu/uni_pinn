"""
spy_backtest_parametric.py

Real-market backtest: FullyParametricPINN vs analytical solutions.

Analytical benchmarks:
  BSM    -- Black-Scholes closed-form
  CEV    -- Schroder (1989) non-central chi-squared formula
  Heston -- Gil-Pelaez inversion with Gauss-Legendre quadrature

Usage:
  python spy_backtest_parametric.py \
      --data data/spy_quotedata.csv \
      --ckpt parametric_pinn/results/fully_param_v1.pt \
      --out  results/spy_backtest_parametric.csv
"""

import argparse
import sys
import os
import re
import datetime
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize
from scipy.stats import norm, ncx2
from scipy.integrate import quad
from numpy.polynomial.legendre import leggauss

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

# Pre-compute GL nodes for Heston pricing
_N_GL = 96
_GL_NODES, _GL_WEIGHTS = leggauss(_N_GL)
_GL_PHI_MAX = 100.0
_GL_PHI = (_GL_NODES + 1.0) * 0.5 * _GL_PHI_MAX
_GL_W   = _GL_WEIGHTS * 0.5 * _GL_PHI_MAX


# ---------------------------------------------------------------------------
# BSM
# ---------------------------------------------------------------------------

def bs_call_scalar(S, K, T, r, sigma):
    if sigma <= 0 or T < 1e-8:
        return max(S - K * np.exp(-r * T), 0.0)
    sqt = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / sqt
    d2 = d1 - sqt
    return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))


def implied_vol_bsm(mkt, S, K, T, r):
    intrinsic = max(S - K * np.exp(-r * T), 0.0)
    if mkt <= intrinsic + 1e-4 or mkt <= 0:
        return float("nan")
    try:
        return brentq(lambda s: bs_call_scalar(S, K, T, r, s) - mkt,
                      1e-4, 10.0, xtol=1e-7, maxiter=300)
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# CEV -- Schroder (1989) exact formula via non-central chi-squared
# ---------------------------------------------------------------------------

def cev_schroder_call(S, K, T, r, sigma, beta):
    """CEV call price via Schroder (1989) non-central chi-squared formula."""
    if abs(beta - 1.0) < 1e-6:
        return bs_call_scalar(S, K, T, r, sigma)
    if T < 1e-8 or sigma <= 0:
        return max(S - K * np.exp(-r * T), 0.0)

    b = 1.0 - beta
    # Schroder parameterization
    kappa_s = 2.0 * r / (sigma**2 * b * (np.exp(2.0 * r * b * T) - 1.0))
    x = kappa_s * S**(2.0 * b) * np.exp(2.0 * r * b * T)
    y = kappa_s * K**(2.0 * b)
    nu = 1.0 / b  # degrees of freedom parameter (= 1/(1-beta))

    # For beta < 1: call = S*exp(rT)*[1 - F(2y; 2+nu, 2x)] - K*F(2x; nu, 2y)
    # where F(z; df, nc) is CDF of non-central chi-squared
    try:
        disc = np.exp(-r * T)
        # P(chi^2(df=2+nu, nc=2x) <= 2y)
        p1 = ncx2.cdf(2.0 * y, df=2.0 + nu, nc=2.0 * x)
        # P(chi^2(df=nu, nc=2y) <= 2x)
        p2 = ncx2.cdf(2.0 * x, df=nu, nc=2.0 * y)
        call = S * (1.0 - p1) - K * disc * p2
        return float(max(call, max(S - K * disc, 0.0)))
    except Exception:
        # Fallback to BS approximation
        sigma_loc = sigma * (S / K) ** (1.0 - beta)
        return bs_call_scalar(S, K, T, r, max(sigma_loc, 1e-4))


def calibrate_cev(calls, S, Ks, T, r):
    weights = np.array([1.0 / (c + 0.5) for c in calls])

    def obj(params):
        sigma, beta = params
        if sigma <= 0 or beta <= 0 or beta > 1.0:
            return 1e9
        errs = np.array([cev_schroder_call(S, K, T, r, sigma, beta) - c
                         for K, c in zip(Ks, calls)])
        return float(np.sum(weights * errs**2))

    best_val, best_p = 1e9, (0.2, 0.5)
    for s0, b0 in [(0.15, 0.5), (0.20, 0.7), (0.10, 0.3), (0.25, 0.9)]:
        res = minimize(obj, [s0, b0],
                       bounds=[(0.01, 1.0), (0.01, 1.0)],
                       method="L-BFGS-B",
                       options={"maxiter": 100, "ftol": 1e-8})
        if res.fun < best_val:
            best_val, best_p = res.fun, res.x
    return float(best_p[0]), float(best_p[1])


# ---------------------------------------------------------------------------
# Heston -- GL semi-analytical
# ---------------------------------------------------------------------------

def _heston_cf_batch(phi_arr, S, Ks, T, r, kappa, theta, xi, rho, v0, j):
    i = 1j
    u, b = (0.5, kappa - rho * xi) if j == 1 else (-0.5, kappa)
    a = kappa * theta
    x = np.log(S / Ks)
    phi = phi_arr[:, None]
    x2d = x[None, :]
    d = np.sqrt((rho * xi * i * phi - b)**2 - xi**2 * (2 * u * i * phi - phi**2))
    num = b - rho * xi * i * phi + d
    den = b - rho * xi * i * phi - d
    g = num / den
    exp_dT = np.exp(d * T)
    C = (r * i * phi * T
         + (a / xi**2) * (num * T - 2.0 * np.log((1.0 - g * exp_dT) / (1.0 - g))))
    D = (num / xi**2) * ((1.0 - exp_dT) / (1.0 - g * exp_dT))
    return np.exp(C + D * v0 + i * phi * x2d)


def heston_prices_gl(S, Ks, T, r, kappa, theta, xi, rho, v0):
    if T < 1e-6:
        return np.maximum(S - Ks, 0.0)
    try:
        phi, w = _GL_PHI, _GL_W
        cf1 = _heston_cf_batch(phi, S, Ks, T, r, kappa, theta, xi, rho, v0, 1)
        cf2 = _heston_cf_batch(phi, S, Ks, T, r, kappa, theta, xi, rho, v0, 2)
        phi2d = phi[:, None]
        I1 = (w[:, None] * np.real(cf1 / (1j * phi2d))).sum(axis=0)
        I2 = (w[:, None] * np.real(cf2 / (1j * phi2d))).sum(axis=0)
        P1 = 0.5 + I1 / np.pi
        P2 = 0.5 + I2 / np.pi
        prices = S * P1 - Ks * np.exp(-r * T) * P2
        return np.where(np.isfinite(prices), prices, np.nan)
    except Exception:
        return np.full(len(Ks), np.nan)


def calibrate_heston(calls, S, Ks, T, r):
    calls_arr, Ks_arr = np.array(calls), np.array(Ks)
    n = len(Ks_arr)
    if n > 20:
        idx = np.round(np.linspace(0, n - 1, 20)).astype(int)
        Ks_cal, calls_cal = Ks_arr[idx], calls_arr[idx]
    else:
        Ks_cal, calls_cal = Ks_arr, calls_arr
    weights = 1.0 / (calls_cal + 0.5)

    def obj(params):
        kappa, theta, xi, rho, v0 = params
        if kappa <= 0 or theta <= 0 or xi <= 0 or rho <= -1 or rho >= 0 or v0 <= 0:
            return 1e9
        prices = heston_prices_gl(S, Ks_cal, T, r, kappa, theta, xi, rho, v0)
        if np.any(np.isnan(prices)):
            return 1e9
        return float(np.sum(weights * (prices - calls_cal)**2))

    bounds = [(0.05, 20), (0.001, 0.5), (0.01, 2.5), (-0.98, -0.01), (0.001, 0.5)]
    starts = [
        [2.0, 0.04, 0.3, -0.7, 0.04],
        [1.0, 0.02, 0.2, -0.5, 0.02],
        [5.0, 0.06, 0.5, -0.8, 0.06],
        [0.5, 0.03, 0.1, -0.6, 0.03],
    ]
    best_val, best_p = 1e9, starts[0]
    for x0 in starts:
        res = minimize(obj, x0, bounds=bounds, method="L-BFGS-B",
                       options={"maxiter": 150, "ftol": 1e-9})
        if res.fun < best_val:
            best_val, best_p = res.fun, res.x
    return tuple(float(x) for x in best_p)


# ---------------------------------------------------------------------------
# CBOE CSV parser (same format as spy_backtest_v15.py)
# ---------------------------------------------------------------------------

def parse_cboe_csv(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    m = re.search(r"Last:\s*([\d.]+)", lines[1] if len(lines) > 1 else lines[0])
    S = float(m.group(1)) if m else None
    rows = []
    for line in lines[4:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) < 20:
            continue
        try:
            expiry_str = parts[0].strip()
            call_last  = float(parts[2])  if parts[2].strip()  else float("nan")
            call_iv    = float(parts[7])  if parts[7].strip()  else float("nan")
            strike     = float(parts[11])
            rows.append({"expiry_str": expiry_str,
                         "strike": strike,
                         "call_last": call_last,
                         "call_iv": call_iv})
        except (ValueError, IndexError):
            continue
    df = pd.DataFrame(rows)
    df["expiry_date"] = df["expiry_str"].apply(
        lambda s: datetime.datetime.strptime(s.strip(), "%a %b %d %Y").date())
    return S, df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/spy_quotedata.csv")
    parser.add_argument("--ckpt", default="parametric_pinn/results/fully_param_v1.pt")
    parser.add_argument("--r",    type=float, default=0.043)
    parser.add_argument("--out",  default="results/spy_backtest_parametric.csv")
    args = parser.parse_args()

    today = datetime.date.today()
    r = args.r

    # --- load market data ---
    print(f"Parsing {args.data} ...")
    S, df = parse_cboe_csv(args.data)
    print(f"  SPY spot: {S:.2f}")

    df["T"] = df["expiry_date"].apply(lambda d: (d - today).days / 365.0)
    df = df[df["T"] > 1/365].copy()
    df = df[
        df["call_iv"].notna() & (df["call_iv"] > 0.01) & (df["call_iv"] < 2.0)
        & df["call_last"].notna() & (df["call_last"] > 0.10)
        & (df["strike"] >= 0.85 * S) & (df["strike"] <= 1.15 * S)
    ].copy()
    print(f"  Filtered contracts: {len(df)}")

    # --- load Parametric PINN ---
    from parametric_pinn.fully_parametric_pinn import FullyParametricPINN
    model = FullyParametricPINN(hidden=256, depth=8)
    model.load(args.ckpt)
    print(f"  Loaded model: {args.ckpt}\n")

    summary_rows = []
    all_rows = []

    expiries = sorted(df["expiry_date"].unique())
    for exp_date in expiries:
        grp = df[df["expiry_date"] == exp_date].copy()
        T_val   = grp["T"].iloc[0]
        Ks      = grp["strike"].values
        calls   = grp["call_last"].values
        n       = len(grp)
        exp_str = grp["expiry_str"].iloc[0]

        print(f"  {exp_str}  T={T_val:.3f}y  n={n}")

        # --- BSM calibration: median implied vol ---
        ivs = np.array([implied_vol_bsm(c, S, K, T_val, r)
                        for c, K in zip(calls, Ks)])
        valid_iv = ivs[~np.isnan(ivs)]
        sigma_bsm = float(np.median(valid_iv)) if len(valid_iv) > 0 else 0.15
        bsm_prices = np.array([bs_call_scalar(S, K, T_val, r, sigma_bsm) for K in Ks])

        # --- CEV calibration (Schroder exact) ---
        sigma_cev, beta_cev = calibrate_cev(calls, S, Ks, T_val, r)
        cev_prices = np.array([cev_schroder_call(S, K, T_val, r, sigma_cev, beta_cev)
                               for K in Ks])

        # --- Heston calibration (GL semi-analytical) ---
        if T_val < 0.05:
            kappa_h, theta_h, xi_h, rho_h, v0_h = 2.0, sigma_bsm**2, 0.3, -0.7, sigma_bsm**2
            heston_prices = bsm_prices.copy()
        else:
            kappa_h, theta_h, xi_h, rho_h, v0_h = calibrate_heston(
                calls, S, Ks, T_val, r)
            heston_prices = heston_prices_gl(
                S, Ks, T_val, r, kappa_h, theta_h, xi_h, rho_h, v0_h)

        # --- Parametric PINN pricing (batch per expiry) ---
        scale  = S / 100.0
        T_safe = max(T_val, 0.01)
        Ks_n   = 100.0 * Ks / S   # normalized strikes

        def pinn_batch(sigma, beta, kappa, theta, xi, rho, v0):
            """Price all strikes for one expiry in a single forward pass."""
            import torch
            nb = len(Ks_n)
            dev = model.device
            S_t   = torch.full((nb, 1), 100.0,   dtype=torch.float32, device=dev)
            v_t   = torch.full((nb, 1), v0,       dtype=torch.float32, device=dev)
            tau_t = torch.full((nb, 1), T_safe,   dtype=torch.float32, device=dev)
            K_t   = torch.tensor(Ks_n, dtype=torch.float32, device=dev).reshape(-1, 1)
            r_t   = torch.full((nb, 1), r,        dtype=torch.float32, device=dev)
            lam_t = torch.tensor(
                [[sigma, beta, kappa, theta, xi, rho]] * nb,
                dtype=torch.float32, device=dev)
            model.net.eval()
            with torch.no_grad():
                out = model.net(S_t, v_t, tau_t, K_t, r_t, lam_t)
            return out.cpu().numpy().flatten() * scale

        v0_bsm = sigma_bsm ** 2
        pinn_bsm_prices    = pinn_batch(sigma_bsm, 1.0,      0.0,     0.0,     0.0,   0.0,   v0_bsm)
        pinn_cev_prices    = pinn_batch(sigma_cev,  beta_cev, 0.0,     0.0,     0.0,   0.0,   sigma_cev**2)
        pinn_heston_prices = pinn_batch(0.0,        1.0,      kappa_h, theta_h, xi_h,  rho_h, v0_h)

        # --- MAE ---
        def mae(pred):
            return float(np.mean(np.abs(np.array(pred) - calls)))

        def rel_mae(pred, ref):
            return float(np.mean(np.abs(np.array(pred) - np.array(ref)) / (np.array(ref) + 1e-6)))

        summary_rows.append({
            "expiry":              exp_str,
            "T_years":             round(T_val, 3),
            "n":                   n,
            "sigma_bsm":           round(sigma_bsm, 4),
            "sigma_cev":           round(sigma_cev, 4),
            "beta_cev":            round(beta_cev, 3),
            "kappa_h":             round(kappa_h, 3),
            "theta_h":             round(theta_h, 4),
            "xi_h":                round(xi_h, 3),
            "rho_h":               round(rho_h, 3),
            "v0_h":                round(v0_h, 4),
            # MAE vs market
            "MAE_BSM":             round(mae(bsm_prices), 4),
            "MAE_CEV":             round(mae(cev_prices), 4),
            "MAE_Heston":          round(mae(heston_prices), 4),
            "MAE_PINN_BSM":        round(mae(pinn_bsm_prices), 4),
            "MAE_PINN_CEV":        round(mae(pinn_cev_prices), 4),
            "MAE_PINN_Heston":     round(mae(pinn_heston_prices), 4),
            # RelMAE vs analytical (PINN vs its own analytical benchmark)
            "RelMAE_PINN_vs_BSM":    round(rel_mae(pinn_bsm_prices, bsm_prices), 6),
            "RelMAE_PINN_vs_CEV":    round(rel_mae(pinn_cev_prices, cev_prices), 6),
            "RelMAE_PINN_vs_Heston": round(rel_mae(pinn_heston_prices, heston_prices), 6),
        })

        for i, K in enumerate(Ks):
            all_rows.append({
                "expiry":       exp_str,
                "T_years":      round(T_val, 4),
                "strike":       K,
                "market":       calls[i],
                "BSM":          round(bsm_prices[i], 4),
                "CEV":          round(cev_prices[i], 4),
                "Heston":       round(float(heston_prices[i]), 4),
                "PINN_BSM":     round(pinn_bsm_prices[i], 4),
                "PINN_CEV":     round(pinn_cev_prices[i], 4),
                "PINN_Heston":  round(pinn_heston_prices[i], 4),
            })

    # --- print summary ---
    print()
    hdr = (f"{'到期日':<22} {'T':>5} {'n':>4}  "
           f"{'BSM':>6} {'CEV':>6} {'Heston':>7}  "
           f"{'P-BSM':>7} {'P-CEV':>7} {'P-Hes':>7}  "
           f"{'P/BSM%':>7} {'P/CEV%':>7} {'P/Hes%':>7}")
    print(hdr)
    print("-" * len(hdr))
    for row in summary_rows:
        print(
            f"{row['expiry']:<22} {row['T_years']:>5.3f} {row['n']:>4}  "
            f"{row['MAE_BSM']:>6.2f} {row['MAE_CEV']:>6.2f} {row['MAE_Heston']:>7.2f}  "
            f"{row['MAE_PINN_BSM']:>7.2f} {row['MAE_PINN_CEV']:>7.2f} {row['MAE_PINN_Heston']:>7.2f}  "
            f"{row['RelMAE_PINN_vs_BSM']*100:>7.2f} "
            f"{row['RelMAE_PINN_vs_CEV']*100:>7.2f} "
            f"{row['RelMAE_PINN_vs_Heston']*100:>7.2f}"
        )

    # --- save ---
    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else ".", exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(args.out, index=False, encoding="utf-8-sig")
    detail_path = args.out.replace(".csv", "_detail.csv")
    pd.DataFrame(all_rows).to_csv(detail_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved summary -> {args.out}")
    print(f"Saved detail  -> {detail_path}")


if __name__ == "__main__":
    main()
