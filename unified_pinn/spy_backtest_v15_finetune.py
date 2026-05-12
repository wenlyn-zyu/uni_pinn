"""
spy_backtest_v15_finetune.py
Same as spy_backtest_v15.py but adds PINN-Heston-FT:
  per-expiry online fine-tuning of the last 2 layers using market prices.

Fine-tune protocol (per expiry):
  - Freeze layers 0..4, unfreeze layer 5 + output layer (last Linear)
  - Use calibrated Heston params for that expiry
  - Subsample up to FT_N strikes from market data as supervision
  - 300 Adam steps, lr=1e-4
  - Restore original weights after pricing (no cross-expiry contamination)

Usage:
  python spy_backtest_v15_finetune.py --data data/spy_quotedata.csv \
                                       --ckpt results/unified_v15.pt \
                                       --out  results/spy_backtest_v15_ft.csv
"""

import argparse
import copy
import sys
import os
import re
import datetime
import warnings
import numpy as np
import pandas as pd
import torch
from scipy.optimize import brentq, minimize
from scipy.stats import norm
from scipy.integrate import quad
from numpy.polynomial.legendre import leggauss

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

# Pre-compute GL nodes
_N_GL = 96
_GL_NODES, _GL_WEIGHTS = leggauss(_N_GL)
_GL_PHI_MAX = 100.0
_GL_PHI = (_GL_NODES + 1.0) * 0.5 * _GL_PHI_MAX
_GL_W   = _GL_WEIGHTS * 0.5 * _GL_PHI_MAX

FT_N     = 20    # max strikes used for fine-tuning supervision
FT_STEPS = 300   # fine-tuning steps per expiry
FT_LR    = 1e-4  # fine-tuning learning rate


# ---------------------------------------------------------------------------
# BS / CEV / Heston helpers (identical to v15)
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


def cev_call_approx(S, K, T, r, sigma, beta):
    if beta == 1.0:
        return bs_call_scalar(S, K, T, r, sigma)
    sigma_loc = sigma * (S / K) ** (1.0 - beta)
    sigma_loc = max(sigma_loc, 1e-4)
    return bs_call_scalar(S, K, T, r, sigma_loc)


def calibrate_cev(calls, S, Ks, T, r):
    weights = np.array([1.0 / (c + 0.5) for c in calls])

    def obj(params):
        sigma, beta = params
        if sigma <= 0 or beta <= 0 or beta > 1.0:
            return 1e9
        errs = np.array([cev_call_approx(S, K, T, r, sigma, beta) - c
                         for K, c in zip(Ks, calls)])
        return float(np.sum(weights * errs**2))

    best_val, best_p = 1e9, (0.2, 0.5)
    for s0 in [0.10, 0.15, 0.20, 0.25]:
        for b0 in [0.3, 0.5, 0.7, 0.9]:
            res = minimize(obj, [s0, b0],
                           bounds=[(0.01, 1.0), (0.01, 1.0)],
                           method="L-BFGS-B",
                           options={"maxiter": 200, "ftol": 1e-10})
            if res.fun < best_val:
                best_val, best_p = res.fun, res.x
    return float(best_p[0]), float(best_p[1])


def heston_char_func(phi, S, K, T, r, kappa, theta, xi, rho, v0, j):
    i = complex(0, 1)
    if j == 1:
        u, b = 0.5, kappa - rho * xi
    else:
        u, b = -0.5, kappa
    a = kappa * theta
    x = np.log(S / K)
    d = np.sqrt((rho * xi * i * phi - b)**2 - xi**2 * (2 * u * i * phi - phi**2))
    g = (b - rho * xi * i * phi + d) / (b - rho * xi * i * phi - d)
    C = r * i * phi * T + (a / xi**2) * (
        (b - rho * xi * i * phi + d) * T
        - 2.0 * np.log((1.0 - g * np.exp(d * T)) / (1.0 - g))
    )
    D = ((b - rho * xi * i * phi + d) / xi**2) * (
        (1.0 - np.exp(d * T)) / (1.0 - g * np.exp(d * T))
    )
    return np.exp(C + D * v0 + i * phi * x)


def heston_call_price(S, K, T, r, kappa, theta, xi, rho, v0):
    if T < 1e-6:
        return max(S - K, 0.0)

    def integrand(phi, j):
        cf = heston_char_func(phi, S, K, T, r, kappa, theta, xi, rho, v0, j)
        return np.real(cf / (complex(0, 1) * phi))

    try:
        P1 = 0.5 + (1.0 / np.pi) * quad(integrand, 0, 100, args=(1,), limit=100,
                                          epsabs=1e-4, epsrel=1e-4)[0]
        P2 = 0.5 + (1.0 / np.pi) * quad(integrand, 0, 100, args=(2,), limit=100,
                                          epsabs=1e-4, epsrel=1e-4)[0]
        return float(S * P1 - K * np.exp(-r * T) * P2)
    except Exception:
        return float("nan")


def _heston_cf_batch(phi_arr, S, Ks, T, r, kappa, theta, xi, rho, v0, j):
    i = 1j
    if j == 1:
        u, b = 0.5, kappa - rho * xi
    else:
        u, b = -0.5, kappa
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
    calls_arr = np.array(calls)
    Ks_arr    = np.array(Ks)
    n = len(Ks_arr)
    if n > 20:
        idx = np.round(np.linspace(0, n - 1, 20)).astype(int)
        Ks_cal    = Ks_arr[idx]
        calls_cal = calls_arr[idx]
    else:
        Ks_cal    = Ks_arr
        calls_cal = calls_arr
    weights = 1.0 / (calls_cal + 0.5)

    def obj(params):
        kappa, theta, xi, rho, v0 = params
        if (kappa <= 0 or theta <= 0 or xi <= 0
                or rho <= -1 or rho >= 0 or v0 <= 0):
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
        [8.0, 0.04, 0.4, -0.9, 0.04],
        [0.5, 0.03, 0.1, -0.6, 0.03],
        [3.0, 0.05, 0.6, -0.75, 0.05],
        [10.0, 0.08, 0.8, -0.85, 0.08],
        [1.5, 0.025, 0.15, -0.55, 0.025],
    ]
    best_val, best_p = 1e9, starts[0]
    for x0 in starts:
        res = minimize(obj, x0, bounds=bounds, method="L-BFGS-B",
                       options={"maxiter": 500, "ftol": 1e-12})
        if res.fun < best_val:
            best_val, best_p = res.fun, res.x
    return tuple(float(x) for x in best_p)


# ---------------------------------------------------------------------------
# CBOE CSV parser
# ---------------------------------------------------------------------------

def parse_cboe_csv(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    m = re.search(r"Last:\s*([\d.]+)", lines[1])
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
# Per-expiry fine-tuning
# ---------------------------------------------------------------------------

def finetune_heston(net_orig, device, S_spot, Ks, calls, T_val, r,
                    kappa_h, theta_h, xi_h, rho_h, v0_h,
                    scale, ft_n=FT_N, ft_steps=FT_STEPS, ft_lr=FT_LR):
    """
    Clone the network, freeze first 5 layers, fine-tune last layer + output
    using market prices for this expiry. Returns fine-tuned prices (numpy).
    Original net_orig is NOT modified.
    """
    from unified_pinn_v2 import ModelParams

    net = copy.deepcopy(net_orig)
    net.train()

    # Freeze all layers, then unfreeze last hidden layer + output layer.
    # net.net is Sequential: [Linear,Tanh, Linear,Tanh, ...(6 pairs), Linear(output)]
    # depth=6 → indices 0..11 are hidden (Linear+Tanh pairs), index 12 is output Linear
    for param in net.parameters():
        param.requires_grad_(False)
    # Unfreeze last hidden Linear (index 10) and output Linear (index 12)
    for idx in [10, 12]:
        for param in net.net[idx].parameters():
            param.requires_grad_(True)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, net.parameters()), lr=ft_lr
    )

    # Subsample strikes for fine-tuning
    n = len(Ks)
    if n > ft_n:
        idx_ft = np.round(np.linspace(0, n - 1, ft_n)).astype(int)
        Ks_ft    = Ks[idx_ft]
        calls_ft = calls[idx_ft]
    else:
        Ks_ft    = Ks
        calls_ft = calls

    # Normalize to PINN training domain (S'=100)
    Ks_n     = 100.0 * Ks_ft / S_spot          # normalized strikes
    calls_n  = calls_ft / scale                  # normalized prices

    S_max, v_max = 300.0, 1.0
    K_ref = 100.0
    T_use = max(T_val, 0.01)

    # Build input tensors: S=100 (ATM reference), v=v0, t=0
    S_t   = torch.full((len(Ks_n), 1), 100.0,  dtype=torch.float32, device=device)
    v_t   = torch.full((len(Ks_n), 1), v0_h,   dtype=torch.float32, device=device)
    t_t   = torch.zeros(len(Ks_n), 1,           dtype=torch.float32, device=device)
    V_tgt = torch.tensor(calls_n, dtype=torch.float32, device=device).reshape(-1, 1)

    # Build lambda for each strike (K varies per row)
    lam_rows = []
    for K_n in Ks_n:
        lam_rows.append([0.0, 1.0, kappa_h, theta_h, xi_h, rho_h])
    lam_t = torch.tensor(lam_rows, dtype=torch.float32, device=device)

    # K tensor (per-strike)
    K_t = torch.tensor(Ks_n, dtype=torch.float32, device=device).reshape(-1, 1)

    for _ in range(ft_steps):
        optimizer.zero_grad()
        pred = net(S_t / S_max, v_t / v_max, t_t / T_use,
                   lam_t, S_t, t_t, K_t, T_use, r)
        loss = torch.mean(((pred - V_tgt) / (V_tgt.abs() + K_ref * 0.01))**2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            filter(lambda p: p.requires_grad, net.parameters()), max_norm=1.0)
        optimizer.step()

    # Price all strikes with fine-tuned net
    net.eval()
    ft_prices = []
    with torch.no_grad():
        for K in Ks:
            K_n = 100.0 * K / S_spot
            S_i   = torch.tensor([[100.0]], dtype=torch.float32, device=device)
            v_i   = torch.tensor([[v0_h]],  dtype=torch.float32, device=device)
            t_i   = torch.zeros(1, 1,        dtype=torch.float32, device=device)
            lam_i = torch.tensor([[0.0, 1.0, kappa_h, theta_h, xi_h, rho_h]],
                                  dtype=torch.float32, device=device)
            K_i   = torch.tensor([[K_n]], dtype=torch.float32, device=device)
            p_n   = net(S_i / S_max, v_i / v_max, t_i / T_use,
                        lam_i, S_i, t_i, K_i, T_use, r)
            ft_prices.append(p_n.item() * scale)

    return np.array(ft_prices)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/spy_quotedata.csv")
    parser.add_argument("--ckpt", default="results/unified_v15.pt")
    parser.add_argument("--r",    type=float, default=0.043)
    parser.add_argument("--out",  default="results/spy_backtest_v15_ft.csv")
    args = parser.parse_args()

    today = datetime.date.today()
    r = args.r

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

    from unified_pinn_v2 import ModelParams, UnifiedPINN

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
    model = UnifiedPINN(param_list, hidden=128, depth=6)
    model.load(args.ckpt)
    device = model.device
    print(f"  Loaded model: {args.ckpt}  device={device}\n")

    summary_rows = []
    all_rows = []

    expiries = sorted(df["expiry_date"].unique())
    for exp_date in expiries:
        grp    = df[df["expiry_date"] == exp_date].copy()
        T_val  = grp["T"].iloc[0]
        Ks     = grp["strike"].values
        calls  = grp["call_last"].values
        n      = len(grp)
        exp_str = grp["expiry_str"].iloc[0]

        print(f"  {exp_str}  T={T_val:.3f}y  n={n}")

        # BSM
        ivs = np.array([implied_vol_bsm(c, S, K, T_val, r)
                        for c, K in zip(calls, Ks)])
        valid_iv  = ivs[~np.isnan(ivs)]
        sigma_bsm = float(np.median(valid_iv)) if len(valid_iv) > 0 else 0.15
        bsm_prices = np.array([bs_call_scalar(S, K, T_val, r, sigma_bsm) for K in Ks])

        # CEV
        sigma_cev, beta_cev = calibrate_cev(calls, S, Ks, T_val, r)
        cev_prices = np.array([cev_call_approx(S, K, T_val, r, sigma_cev, beta_cev)
                               for K in Ks])

        # Heston
        if T_val < 0.05:
            kappa_h, theta_h, xi_h, rho_h, v0_h = 2.0, 0.04, 0.3, -0.7, sigma_bsm**2
            heston_prices = bsm_prices.copy()
        else:
            kappa_h, theta_h, xi_h, rho_h, v0_h = calibrate_heston(
                calls, S, Ks, T_val, r)
            heston_prices = heston_prices_gl(
                S, Ks, T_val, r, kappa_h, theta_h, xi_h, rho_h, v0_h)

        scale = S / 100.0

        # PINN-BSM (unchanged)
        pinn_bsm_prices = []
        for K in Ks:
            K_n = 100.0 * K / S
            p = ModelParams.from_bsm(K=K_n, T=max(T_val, 0.01), r=r, sigma=sigma_bsm)
            pinn_bsm_prices.append(model.price(p, S=100.0) * scale)
        pinn_bsm_prices = np.array(pinn_bsm_prices)

        # PINN-Heston (no fine-tuning, baseline)
        pinn_heston_prices = []
        for K in Ks:
            K_n = 100.0 * K / S
            p = ModelParams.from_heston(K=K_n, T=max(T_val, 0.01), r=r,
                                        kappa=kappa_h, theta=theta_h,
                                        xi=xi_h, rho=rho_h, v0=v0_h)
            pinn_heston_prices.append(model.price(p, S=100.0) * scale)
        pinn_heston_prices = np.array(pinn_heston_prices)

        # PINN-Heston-FT (per-expiry fine-tuning)
        if T_val >= 0.05:
            pinn_heston_ft_prices = finetune_heston(
                model.net, device, S, Ks, calls, T_val, r,
                kappa_h, theta_h, xi_h, rho_h, v0_h, scale)
        else:
            pinn_heston_ft_prices = pinn_heston_prices.copy()

        def mae(pred): return float(np.mean(np.abs(pred - calls)))
        summary_rows.append({
            "expiry":            exp_str,
            "T_years":           round(T_val, 3),
            "n":                 n,
            "sigma_bsm":         round(sigma_bsm, 4),
            "beta_cev":          round(beta_cev, 3),
            "kappa_h":           round(kappa_h, 3),
            "xi_h":              round(xi_h, 3),
            "rho_h":             round(rho_h, 3),
            "MAE_BSM":           round(mae(bsm_prices), 4),
            "MAE_CEV":           round(mae(cev_prices), 4),
            "MAE_Heston":        round(mae(heston_prices), 4),
            "MAE_PINN_BSM":      round(mae(pinn_bsm_prices), 4),
            "MAE_PINN_Heston":   round(mae(pinn_heston_prices), 4),
            "MAE_PINN_Heston_FT": round(mae(pinn_heston_ft_prices), 4),
        })

        for i, K in enumerate(Ks):
            all_rows.append({
                "expiry": exp_str, "T_years": round(T_val, 4),
                "strike": K, "market": calls[i],
                "BSM":            round(bsm_prices[i], 4),
                "CEV":            round(cev_prices[i], 4),
                "Heston":         round(float(heston_prices[i]), 4),
                "PINN_BSM":       round(pinn_bsm_prices[i], 4),
                "PINN_Heston":    round(pinn_heston_prices[i], 4),
                "PINN_Heston_FT": round(pinn_heston_ft_prices[i], 4),
            })

    print()
    hdr = (f"{'到期日':<22} {'T':>5} {'n':>4}  "
           f"{'BSM':>6} {'CEV':>6} {'Heston':>7} "
           f"{'PINN-BSM':>9} {'PINN-Hes':>9} {'PINN-Hes-FT':>12}")
    print(hdr)
    print("-" * len(hdr))
    for row in summary_rows:
        print(f"{row['expiry']:<22} {row['T_years']:>5.3f} {row['n']:>4}  "
              f"{row['MAE_BSM']:>6.2f} {row['MAE_CEV']:>6.2f} "
              f"{row['MAE_Heston']:>7.2f} {row['MAE_PINN_BSM']:>9.2f} "
              f"{row['MAE_PINN_Heston']:>9.2f} {row['MAE_PINN_Heston_FT']:>12.2f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(args.out, index=False, encoding="utf-8-sig")
    pd.DataFrame(all_rows).to_csv(
        args.out.replace(".csv", "_detail.csv"), index=False, encoding="utf-8-sig")
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
