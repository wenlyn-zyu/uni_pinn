# option_pinn/exploration/eval_improved.py
"""Comprehensive evaluation of improved models vs baselines.

Tests:
  1. Synthetic price accuracy (BSM/CEV/Heston, MAE, RelMAE, RMSE)
  2. Greeks accuracy (Delta, Gamma, Vega via autograd)
  3. Market data accuracy (SPY option chain)

Models evaluated:
  - indep: Independent PINN per model
  - unified_v2: Baseline unified PINN
  - parametric: Original parametric PINN v1
  - imp_param: Improved parametric PINN v2  (if checkpoint exists)
  - unified_ft: Fine-tuned unified v2         (if checkpoint exists)

Usage:
  python eval_improved.py --imp-param results/imp_param_v2.pt --unified-ft results/unified_v2_ft_v2.pt
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from scipy.optimize import brentq, minimize
from scipy.stats import norm
from numpy.polynomial.legendre import leggauss

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ref_solvers import (bsm_call, bsm_greeks, cev_call,
                          heston_call, heston_greeks_fd)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE   = os.path.dirname(os.path.abspath(__file__))


# ============================================================================
# Reference solvers (duplicated for standalone use)
# ============================================================================

def _bsm_greeks(S, K, T, r, sigma):
    sqt = sigma * np.sqrt(max(T, 1e-8))
    d1  = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / sqt
    d2  = d1 - sqt
    return {
        "delta": float(norm.cdf(d1)),
        "gamma": float(norm.pdf(d1) / (S * sqt)),
        "vega":  float(S * norm.pdf(d1) * np.sqrt(T)),
    }


def _heston_greeks_fd(S, K, T, r, kappa, theta, xi, rho, v0):
    dS = S * 0.001
    dv = max(v0 * 0.01, 1e-5)
    Vp  = heston_call(S + dS, K, T, r, kappa, theta, xi, rho, v0)
    Vm  = heston_call(S - dS, K, T, r, kappa, theta, xi, rho, v0)
    V0  = heston_call(S,      K, T, r, kappa, theta, xi, rho, v0)
    Vvp = heston_call(S,      K, T, r, kappa, theta, xi, rho, v0 + dv)
    Vvm = heston_call(S,      K, T, r, kappa, theta, xi, rho, v0 - dv)
    return {
        "delta": float((Vp - Vm) / (2 * dS)),
        "gamma": float((Vp - 2 * V0 + Vm) / (dS ** 2)),
        "vega":  float((Vvp - Vvm) / (2 * dv)),
    }


# ============================================================================
# Metrics
# ============================================================================

def _mae(pred, ref):
    return float(np.mean(np.abs(np.array(pred) - np.array(ref))))


def _mse_relmse(pred, ref, mask_threshold=0.01):
    pred, ref = np.array(pred, dtype=float), np.array(ref, dtype=float)
    mse = float(np.mean((pred - ref) ** 2))
    mask = np.abs(ref) > mask_threshold
    if mask.sum() == 0:
        return mse, float("nan")
    relmse = float(np.mean(((pred[mask] - ref[mask]) / np.abs(ref[mask])) ** 2))
    return mse, relmse


# ============================================================================
# Model loading
# ============================================================================

def load_unified():
    sys.path.insert(0, os.path.join(BASE, ".."))
    from unified_pinn_v2 import UnifiedPINN, ModelParams
    params = []
    for sigma in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35]:
        params.append(ModelParams.from_bsm(sigma=sigma))
    for sigma in [0.15, 0.2, 0.25]:
        for beta in [0.3, 0.5, 0.7, 0.9]:
            params.append(ModelParams.from_cev(sigma=sigma, beta=beta))
    for kappa in [1.0, 2.0, 3.0]:
        for theta in [0.02, 0.04, 0.06]:
            for xi in [0.2, 0.3, 0.4]:
                for rho in [-0.7, -0.5]:
                    params.append(ModelParams.from_heston(
                        kappa=kappa, theta=theta, xi=xi, rho=rho, v0=theta))
    pinn = UnifiedPINN(params, hidden=128, depth=6, device=DEVICE)
    pinn.load(os.path.join(BASE, "..", "results", "unified_v16_gl.pt"))
    pinn.net.eval()
    return pinn


def load_parametric():
    sys.path.insert(0, os.path.join(BASE, "..", "parametric_pinn"))
    from fully_parametric_pinn import FullyParametricPINN
    pinn = FullyParametricPINN(device=DEVICE)
    pinn.load(os.path.join(BASE, "..", "parametric_pinn", "results", "fully_param_v1.pt"))
    pinn.net.eval()
    return pinn


def load_imp_parametric(ckpt_path):
    from improved_parametric_pinn import ImprovedParametricPINN
    pinn = ImprovedParametricPINN(device=DEVICE)
    pinn.load(ckpt_path)
    pinn.net.eval()
    return pinn


def load_hainaut():
    sys.path.insert(0, os.path.join(BASE, "..", "independent"))
    from heston_hainaut import HestonHainaut
    model = HestonHainaut(device=DEVICE)
    model.load(os.path.join(BASE, "..", "results", "hainaut.pt"))
    return model


# ============================================================================
# Greeks via autograd (parametric models)
# ============================================================================

def _parametric_greeks(pinn, price_fn, S, K, T, r, sigma, beta,
                       kappa, theta, xi, rho, v0):
    """Compute Delta, Gamma, Vega for parametric PINN via FD on price_fn."""
    dS = S * 0.005
    dv = max(v0 * 0.05, 1e-4)
    p_mid = price_fn(S, K, T, r, sigma, beta, kappa, theta, xi, rho, v0)
    p_up  = price_fn(S + dS, K, T, r, sigma, beta, kappa, theta, xi, rho, v0)
    p_dn  = price_fn(S - dS, K, T, r, sigma, beta, kappa, theta, xi, rho, v0)
    p_vup = price_fn(S, K, T, r, sigma, beta, kappa, theta, xi, rho, v0 + dv)
    p_vdn = price_fn(S, K, T, r, sigma, beta, kappa, theta, xi, rho, v0 - dv)
    return {
        "delta": (p_up - p_dn) / (2 * dS),
        "gamma": (p_up - 2 * p_mid + p_dn) / (dS ** 2),
        "vega":  (p_vup - p_vdn) / (2 * dv),
    }


# ============================================================================
# Market data helpers
# ============================================================================

def load_spy():
    import re, datetime
    csv_path = os.path.join(BASE, "..", "data", "spy_quotedata.csv")
    with open(csv_path, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    m = re.search(r"Last:\s*([\d.]+)", lines[1] if len(lines) > 1 else lines[0])
    S_spot = float(m.group(1)) if m else None
    today = datetime.date.today()
    rows = []
    for line in lines[4:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) < 12:
            continue
        try:
            expiry_str = parts[0].strip()
            bid  = float(parts[4]) if parts[4].strip() else float("nan")
            ask  = float(parts[5]) if parts[5].strip() else float("nan")
            K    = float(parts[11])
            expiry_date = datetime.datetime.strptime(expiry_str, "%a %b %d %Y").date()
            tau = (expiry_date - today).days / 365.0
            rows.append({"S": S_spot, "K": K, "tau": tau, "bid": bid, "ask": ask,
                         "expiry_date": expiry_date})
        except (ValueError, IndexError):
            continue
    df = pd.DataFrame(rows)
    df["mid"] = (df["bid"] + df["ask"]) / 2.0
    df = df[(df["bid"] > 0) & (df["ask"] > 0)]
    df = df[df["tau"] > 0.05]
    df["moneyness"] = df["S"] / df["K"]
    df = df[(df["moneyness"] >= 0.8) & (df["moneyness"] <= 1.2)]
    return df.reset_index(drop=True)


def calibrate_heston_mkt(calls, S, Ks, T, r, sigma_bsm=0.2):
    if T < 0.05:
        v0 = sigma_bsm ** 2
        return (2.0, v0, 0.3, -0.7, v0)
    def loss(p):
        kappa, theta, xi, rho, v0 = p
        total = 0.0
        for call, K in zip(calls, Ks):
            pred = heston_call(S, K, T, r, kappa, theta, xi, rho, v0)
            total += (pred - call) ** 2
        return total / max(len(calls), 1)
    bounds = [(0.05, 20), (0.001, 0.5), (0.01, 2.0), (-0.99, -0.01), (0.001, 0.5)]
    starts = [
        [2.0, 0.04, 0.3, -0.7, 0.04],
        [1.0, 0.02, 0.2, -0.5, 0.02],
        [3.0, 0.06, 0.4, -0.9, 0.06],
    ]
    best_val, best_params = np.inf, starts[0]
    for x0 in starts:
        try:
            res = minimize(loss, x0, method="L-BFGS-B", bounds=bounds,
                          options={"maxiter": 200, "ftol": 1e-9})
            if res.fun < best_val:
                best_val, best_params = res.fun, res.x.tolist()
        except Exception:
            pass
    return tuple(best_params)


# ============================================================================
# Synthetic evaluation
# ============================================================================

K_SYN, T_SYN, r_syn = 100.0, 1.0, 0.05
S_GRID = np.linspace(60, 250, 50)

BSM_EVAL    = dict(sigma=0.20)
CEV_EVAL    = dict(sigma=0.20, beta=0.5)
HESTON_EVAL = dict(kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, v0=0.04)


def eval_synthetic(models: dict):
    print("\n" + "=" * 70)
    print("SYNTHETIC DATA EVALUATION")
    print("=" * 70)

    # Reference prices
    ref_bsm    = np.array([bsm_call(s, K_SYN, T_SYN, r_syn, **BSM_EVAL) for s in S_GRID])
    ref_cev    = np.array([cev_call(s, K_SYN, T_SYN, r_syn, **CEV_EVAL) for s in S_GRID])
    ref_heston = np.array([heston_call(s, K_SYN, T_SYN, r_syn, **HESTON_EVAL) for s in S_GRID])

    rows = []

    for name, price_fn in models.items():
        if price_fn is None:
            continue

        # BSM price
        pred_bsm = np.array([price_fn("bsm", s, K_SYN, T_SYN, r_syn,
                                      BSM_EVAL["sigma"], 1.0,
                                      0, 0, 0, 0, BSM_EVAL["sigma"]**2)
                            for s in S_GRID])
        bsm_mse, bsm_rel = _mse_relmse(pred_bsm, ref_bsm)

        # CEV price
        pred_cev = np.array([price_fn("cev", s, K_SYN, T_SYN, r_syn,
                                      CEV_EVAL["sigma"], CEV_EVAL["beta"],
                                      0, 0, 0, 0, CEV_EVAL["sigma"]**2)
                            for s in S_GRID])
        cev_mse, cev_rel = _mse_relmse(pred_cev, ref_cev)

        # Heston price
        pred_heston = np.array([price_fn("heston", s, K_SYN, T_SYN, r_syn,
                                        0, 1.0,
                                        HESTON_EVAL["kappa"],
                                        HESTON_EVAL["theta"],
                                        HESTON_EVAL["xi"],
                                        HESTON_EVAL["rho"],
                                        HESTON_EVAL["v0"])
                                for s in S_GRID])
        heston_mse, heston_rel = _mse_relmse(pred_heston, ref_heston)

        print(f"\n{name}:")
        print(f"  BSM:    MSE={bsm_mse:.2e}  RelMSE={bsm_rel:.4%}")
        print(f"  CEV:    MSE={cev_mse:.4f}  RelMSE={cev_rel:.4%}")
        print(f"  Heston: MSE={heston_mse:.4f}  RelMSE={heston_rel:.4%}")

        rows.append({
            "model": name,
            "bsm_mse": bsm_mse, "bsm_relmse": bsm_rel,
            "cev_mse": cev_mse, "cev_relmse": cev_rel,
            "heston_mse": heston_mse, "heston_relmse": heston_rel,
        })

    return pd.DataFrame(rows)


def eval_greeks_parametric(name: str, price_fn):
    """Evaluate Greeks for a parametric model."""
    print(f"\n--- {name} Greeks ---")

    # BSM Greeks
    greeks_bsm = []
    ref_bsm_g = []
    for s in [90.0, 100.0, 110.0]:
        sig = BSM_EVAL["sigma"]
        ref_bsm_g.append(_bsm_greeks(s, K_SYN, T_SYN, r_syn, sig))
        greeks_bsm.append(_parametric_greeks(
            None, price_fn, s, K_SYN, T_SYN, r_syn,
            sig, 1.0, 0, 0, 0, 0, sig**2, option_type="call"))

    delta_mae = _mae([g["delta"] for g in greeks_bsm],
                     [g["delta"] for g in ref_bsm_g])
    gamma_mae = _mae([g["gamma"] for g in greeks_bsm],
                     [g["gamma"] for g in ref_bsm_g])
    vega_mae  = _mae([g["vega"]  for g in greeks_bsm],
                     [g["vega"]  for g in ref_bsm_g])
    print(f"  BSM:  delta_mae={delta_mae:.4f}  gamma_mae={gamma_mae:.6f}  vega_mae={vega_mae:.2f}")

    # Heston Greeks
    greeks_heston = []
    ref_heston_g  = []
    for s in [90.0, 100.0, 110.0]:
        h = HESTON_EVAL
        ref_heston_g.append(_heston_greeks_fd(
            s, K_SYN, T_SYN, r_syn, h["kappa"], h["theta"],
            h["xi"], h["rho"], h["v0"]))
        greeks_heston.append(_parametric_greeks(
            None, price_fn, s, K_SYN, T_SYN, r_syn,
            0, 1.0, h["kappa"], h["theta"], h["xi"], h["rho"], h["v0"],
            option_type="call"))

    h_delta_mae = _mae([g["delta"] for g in greeks_heston],
                       [g["delta"] for g in ref_heston_g])
    h_gamma_mae = _mae([g["gamma"] for g in greeks_heston],
                       [g["gamma"] for g in ref_heston_g])
    h_vega_mae  = _mae([g["vega"]  for g in greeks_heston],
                       [g["vega"]  for g in ref_heston_g])
    print(f"  Heston: delta_mae={h_delta_mae:.4f}  gamma_mae={h_gamma_mae:.6f}  vega_mae={h_vega_mae:.2f}")

    return {
        "model": name,
        "bsm_delta_mae": delta_mae, "bsm_gamma_mae": gamma_mae, "bsm_vega_mae": vega_mae,
        "heston_delta_mae": h_delta_mae, "heston_gamma_mae": h_gamma_mae,
        "heston_vega_mae": h_vega_mae,
    }


# ============================================================================
# Market evaluation
# ============================================================================

r_mkt = 0.043
K_REF = 100.0


def eval_market(imp_param_pinn=None, unified_ft_pinn=None):
    print("\n" + "=" * 70)
    print("MARKET DATA EVALUATION")
    print("=" * 70)

    df = load_spy()
    S_spot = float(df["S"].iloc[0])
    scale  = S_spot / K_REF
    print(f"Spot={S_spot:.2f}, contracts={len(df)}")

    unified = load_unified()
    param   = load_parametric()

    rows = []
    models_available = {
        "bsm_analytical": "analytical",
        "heston_analytical": "analytical",
        "unified_v2": unified,
        "parametric": param,
    }
    if imp_param_pinn is not None:
        models_available["imp_param"] = imp_param_pinn
    if unified_ft_pinn is not None:
        models_available["unified_ft"] = unified_ft_pinn

    for exp_date, grp in df.groupby("expiry_date"):
        T_val = float(grp["tau"].iloc[0])
        Ks    = grp["K"].values.astype(float)
        calls = grp["mid"].values.astype(float)

        ivs = [brentq(lambda sig: bsm_call(S_spot, K, T_val, r_mkt, sig) - c,
                      1e-4, 5.0, xtol=1e-6, maxiter=100)
               for K, c in zip(Ks, calls)]
        ivs_clean = [iv for iv in ivs if 0.01 < iv < 2.0]
        sigma_bsm = float(np.median(ivs_clean)) if ivs_clean else 0.2
        kappa_h, theta_h, xi_h, rho_h, v0_h = calibrate_heston_mkt(
            calls, S_spot, Ks, T_val, r_mkt, sigma_bsm)

        errs = {name: [] for name in models_available}

        for K, call in zip(Ks, calls):
            K_n = K_REF * K / S_spot
            # BSM analytical
            errs["bsm_analytical"].append(
                abs(bsm_call(S_spot, K, T_val, r_mkt, sigma_bsm) - call))
            # Heston analytical
            errs["heston_analytical"].append(abs(
                heston_call(S_spot, K, T_val, r_mkt,
                            kappa_h, theta_h, xi_h, rho_h, v0_h) - call))
            # Unified v2
            from unified_pinn_v2 import ModelParams as MP
            p_u = MP.from_heston(K=K_n, T=max(T_val, 0.01), r=r_mkt,
                                 kappa=kappa_h, theta=theta_h,
                                 xi=xi_h, rho=rho_h, v0=v0_h, S_max=500)
            errs["unified_v2"].append(
                abs(unified.price(p_u, S=K_REF) * scale - call))
            # Parametric v1
            errs["parametric"].append(abs(param.price(
                S=K_REF, K=K_n, T=max(T_val, 0.01), r=r_mkt,
                kappa=kappa_h, theta=theta_h, xi=xi_h, rho=rho_h, v0=v0_h) * scale - call))
            # Improved parametric
            if imp_param_pinn is not None:
                errs["imp_param"].append(abs(imp_param_pinn.price(
                    S=K_REF, K=K_n, T=max(T_val, 0.01), r=r_mkt,
                    kappa=kappa_h, theta=theta_h, xi=xi_h, rho=rho_h, v0=v0_h,
                    option_type="call") * scale - call))
            # Unified FT
            if unified_ft_pinn is not None:
                errs["unified_ft"].append(
                    abs(unified_ft_pinn.price(p_u, S=K_REF) * scale - call))

        for name, err_list in errs.items():
            if err_list:
                rows.append({
                    "expiry": str(exp_date), "T": round(T_val, 4), "n": len(Ks),
                    "model": name,
                    "MAE": round(float(np.mean(err_list)), 4),
                })

    df_r = pd.DataFrame(rows)
    summary = df_r.groupby("model")["MAE"].mean().reset_index()
    summary.columns = ["model", "avg_MAE"]
    print("\nMarket MAE summary:")
    for _, r in summary.iterrows():
        print(f"  {r['model']:20s}: {r['avg_MAE']:.4f}")

    return summary


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--imp-param", type=str, default=None,
                        help="Path to improved parametric v2 checkpoint")
    parser.add_argument("--unified-ft", type=str, default=None,
                        help="Path to fine-tuned unified v2 checkpoint")
    parser.add_argument("--out-dir", type=str, default="results")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load improved parametric if available
    imp_param = None
    if args.imp_param and os.path.exists(args.imp_param):
        print(f"Loading improved parametric: {args.imp_param}")
        imp_param = load_imp_parametric(args.imp_param)

    # Load unified FT if available
    unified_ft = None
    if args.unified_ft and os.path.exists(args.unified_ft):
        print(f"Loading unified FT: {args.unified_ft}")
        from unified_pinn_v2 import UnifiedPINN as UP
        params = []
        for sigma in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35]:
            from unified_pinn_v2 import ModelParams as MP
            params.append(MP.from_bsm(sigma=sigma))
        for sigma in [0.15, 0.2, 0.25]:
            for beta in [0.3, 0.5, 0.7, 0.9]:
                params.append(MP.from_cev(sigma=sigma, beta=beta))
        for kappa in [1.0, 2.0, 3.0]:
            for theta in [0.02, 0.04, 0.06]:
                for xi in [0.2, 0.3, 0.4]:
                    for rho in [-0.7, -0.5]:
                        params.append(MP.from_heston(
                            kappa=kappa, theta=theta, xi=xi, rho=rho, v0=theta))
        unified_ft = UP(params, hidden=128, depth=6, device=DEVICE)
        unified_ft.load(args.unified_ft)
        unified_ft.net.eval()

    # Build model price functions for synthetic eval
    def _make_price_wrapper(pinn, is_parametric=False):
        def fn(model, S, K, T, r, sigma, beta, kappa, theta, xi, rho, v0):
            if is_parametric:
                return pinn.price(S=S, K=K, T=T, r=r,
                                 sigma=sigma, beta=beta,
                                 kappa=kappa, theta=theta,
                                 xi=xi, rho=rho, v0=v0,
                                 option_type="call")
            else:
                from unified_pinn_v2 import ModelParams as MP
                if model == "bsm":
                    p = MP.from_bsm(K=K, T=T, r=r, sigma=sigma, S_max=500)
                elif model == "cev":
                    p = MP.from_cev(K=K, T=T, r=r, sigma=sigma, beta=beta, S_max=500)
                else:
                    p = MP.from_heston(K=K, T=T, r=r, kappa=kappa, theta=theta,
                                      xi=xi, rho=rho, v0=v0, S_max=500)
                return pinn.price(p, S=S, v=v0, t=0.0)
        return fn

    unified_pinn = load_unified()
    param_pinn   = load_parametric()

    models = {
        "unified_v2": _make_price_wrapper(unified_pinn),
        "parametric": _make_price_wrapper(param_pinn, is_parametric=True),
    }
    if imp_param is not None:
        def imp_fn(model, S, K, T, r, sigma, beta, kappa, theta, xi, rho, v0):
            return imp_param.price(S=S, K=K, T=T, r=r,
                                  sigma=sigma, beta=beta,
                                  kappa=kappa, theta=theta,
                                  xi=xi, rho=rho, v0=v0,
                                  option_type="call")
        models["imp_param"] = imp_fn

    # Run synthetic eval
    df_price = eval_synthetic(models)

    # Run Greeks for parametric models
    greeks_rows = []
    param_price_fn = lambda S, K, T, r, sigma, beta, kappa, theta, xi, rho, v0, **kw: param_pinn.price(
        S=S, K=K, T=T, r=r, sigma=sigma, beta=beta,
        kappa=kappa, theta=theta, xi=xi, rho=rho, v0=v0, option_type="call")
    greeks_rows.append(eval_greeks_parametric("parametric", param_price_fn))

    if imp_param is not None:
        imp_price_fn = lambda S, K, T, r, sigma, beta, kappa, theta, xi, rho, v0, **kw: imp_param.price(
            S=S, K=K, T=T, r=r, sigma=sigma, beta=beta,
            kappa=kappa, theta=theta, xi=xi, rho=rho, v0=v0, option_type="call")
        greeks_rows.append(eval_greeks_parametric("imp_param", imp_price_fn))

    # Run market eval
    df_market = eval_market(imp_param, unified_ft)

    # Save
    df_price.to_csv(os.path.join(args.out_dir, "eval_improved_price.csv"), index=False)
    pd.DataFrame(greeks_rows).to_csv(
        os.path.join(args.out_dir, "eval_improved_greeks.csv"), index=False)
    df_market.to_csv(os.path.join(args.out_dir, "eval_improved_market.csv"), index=False)
    print(f"\nResults saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
