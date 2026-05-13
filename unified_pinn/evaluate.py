"""
evaluate.py — 评估统一PINN在三个模型上的精度

修复内容：
  1. 导入改为 unified_pinn_v2
  2. BSM -> Black-Scholes 解析解
  3. CEV -> Schroder 1989 非中心卡方解析解
  4. Heston -> Gauss-Legendre 半解析解

用法：
  python evaluate.py [--ckpt results/unified_v15.pt] [--out results/]
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
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
    phi = phi_arr[:, None]
    x2d = x[None, :]
    d = np.sqrt((rho * xi * i * phi - b)**2 - xi**2 * (2 * u * i * phi - phi**2))
    num = b - rho * xi * i * phi + d
    g = num / (b - rho * xi * i * phi - d)
    exp_dT = np.exp(d * T)
    C = (r * i * phi * T
         + (a / xi**2) * (num * T - 2.0 * np.log((1.0 - g * exp_dT) / (1.0 - g))))
    D = (num / xi**2) * ((1.0 - exp_dT) / (1.0 - g * exp_dT))
    return np.exp(C + D * v0 + i * phi * x2d)


def heston_prices_gl(S_scalar, Ks, T, r, kappa, theta, xi, rho, v0):
    if T < 1e-6:
        return np.maximum(S_scalar - Ks, 0.0)
    try:
        phi, w = _GL_PHI, _GL_W
        cf1 = _heston_cf_batch(phi, S_scalar, Ks, T, r, kappa, theta, xi, rho, v0, 1)
        cf2 = _heston_cf_batch(phi, S_scalar, Ks, T, r, kappa, theta, xi, rho, v0, 2)
        phi2d = phi[:, None]
        I1 = (w[:, None] * np.real(cf1 / (1j * phi2d))).sum(axis=0)
        I2 = (w[:, None] * np.real(cf2 / (1j * phi2d))).sum(axis=0)
        prices = S_scalar * (0.5 + I1 / np.pi) - Ks * np.exp(-r * T) * (0.5 + I2 / np.pi)
        return np.where(np.isfinite(prices), np.maximum(prices, 0.0), np.nan)
    except Exception:
        return np.full(len(Ks), np.nan)


# ---------------------------------------------------------------------------
# BSM analytical reference
# ---------------------------------------------------------------------------

def bsm_prices(S_arr, K, T, r, sigma):
    if sigma <= 0 or T < 1e-8:
        return np.maximum(S_arr - K * np.exp(-r * T), 0.0)
    sqt = sigma * np.sqrt(T)
    d1 = (np.log(S_arr / K) + (r + 0.5 * sigma**2) * T) / sqt
    d2 = d1 - sqt
    return S_arr * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def metrics(pred, ref):
    err = np.abs(pred - ref)
    mask = ref > 0.5
    rel = float(np.mean(err[mask] / ref[mask])) if mask.any() else float("nan")
    return {
        "MAE":    float(np.mean(err)),
        "RMSE":   float(np.sqrt(np.mean(err**2))),
        "MaxErr": float(np.max(err)),
        "RelMAE": rel,
    }


# ---------------------------------------------------------------------------
# Build full param list (same as training)
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
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="results/unified_v15.pt")
    parser.add_argument("--out",  type=str, default="results/")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    param_list = build_param_list()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = UnifiedPINN(param_list, hidden=128, depth=6, device=device)
    model.load(args.ckpt)
    print(f"已加载模型: {args.ckpt}  device={device}\n")

    S_grid = np.linspace(60, 160, 50)

    rows = []

    # ------------------------------------------------------------------
    # BSM
    # ------------------------------------------------------------------
    print("=== BSM ===")
    bsm_params = [p for p in param_list if p.xi == 0 and p.beta == 1.0]
    for p in bsm_params:
        pred = np.array([model.price(p, S) for S in S_grid])
        ref  = bsm_prices(S_grid, p.K, p.T, p.r, p.sigma)
        m = metrics(pred, ref)
        label = f"BSM(σ={p.sigma:.2f})"
        print(f"  {label:<20}  MAE={m['MAE']:.4f}  RMSE={m['RMSE']:.4f}  "
              f"MaxErr={m['MaxErr']:.4f}  RelMAE={m['RelMAE']:.4f}")
        rows.append({"model": "BSM", "params": label, **m})

    # ------------------------------------------------------------------
    # CEV
    # ------------------------------------------------------------------
    print("\n=== CEV ===")
    cev_params = [p for p in param_list if p.xi == 0 and p.beta != 1.0]
    for p in cev_params:
        pred = np.array([model.price(p, S) for S in S_grid])
        ref  = np.array([cev_schroder_call(s, p.K, p.T, p.r, p.sigma, p.beta) for s in S_grid])
        m = metrics(pred, ref)
        label = f"CEV(σ={p.sigma:.2f},β={p.beta:.1f})"
        print(f"  {label:<25}  MAE={m['MAE']:.4f}  RMSE={m['RMSE']:.4f}  "
              f"MaxErr={m['MaxErr']:.4f}  RelMAE={m['RelMAE']:.4f}")
        rows.append({"model": "CEV", "params": label, **m})

    # ------------------------------------------------------------------
    # Heston
    # ------------------------------------------------------------------
    print("\n=== Heston ===")
    heston_params = [p for p in param_list if p.xi > 0]
    for p in heston_params:
        pred = np.array([model.price(p, S, v=p.v0) for S in S_grid])
        ref  = heston_prices_gl(S_grid, np.full(len(S_grid), p.K),
                                p.T, p.r, p.kappa, p.theta, p.xi, p.rho, p.v0)
        if np.any(np.isnan(ref)):
            print(f"  Heston(κ={p.kappa},ξ={p.xi},ρ={p.rho})  [GL ref NaN, skipped]")
            continue
        m = metrics(pred, ref)
        label = f"Heston(κ={p.kappa:.1f},ξ={p.xi:.1f},ρ={p.rho:.1f})"
        print(f"  {label:<35}  MAE={m['MAE']:.4f}  RMSE={m['RMSE']:.4f}  "
              f"MaxErr={m['MaxErr']:.4f}  RelMAE={m['RelMAE']:.4f}")
        rows.append({"model": "Heston", "params": label, **m})

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    df = pd.DataFrame(rows)
    print("\n=== 汇总 ===")
    for model_type in ["BSM", "CEV", "Heston"]:
        sub = df[df["model"] == model_type]
        if len(sub) == 0:
            continue
        print(f"  {model_type:<8}  avg MAE={sub['MAE'].mean():.4f}  "
              f"avg RelMAE={sub['RelMAE'].mean():.4f}  "
              f"max RelMAE={sub['RelMAE'].max():.4f}")

    out_csv = os.path.join(args.out, "eval_synthetic.csv")
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n结果已保存至 {out_csv}")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # BSM: pick sigma=0.20
        p_bsm = next(p for p in bsm_params if abs(p.sigma - 0.20) < 1e-4)
        pred_b = np.array([model.price(p_bsm, S) for S in S_grid])
        ref_b  = bsm_prices(S_grid, p_bsm.K, p_bsm.T, p_bsm.r, p_bsm.sigma)
        axes[0].plot(S_grid, pred_b, label="PINN", lw=2)
        axes[0].plot(S_grid, ref_b, "--", label="BSM ref", lw=1.5)
        axes[0].set_title("BSM (σ=0.20)")

        # CEV: pick sigma=0.20, beta=0.5
        p_cev = next(p for p in cev_params
                     if abs(p.sigma - 0.20) < 1e-4 and abs(p.beta - 0.5) < 1e-4)
        pred_c = np.array([model.price(p_cev, S) for S in S_grid])
        ref_c  = np.array([cev_schroder_call(s, p_cev.K, p_cev.T, p_cev.r, p_cev.sigma, p_cev.beta) for s in S_grid])
        axes[1].plot(S_grid, pred_c, label="PINN", lw=2)
        axes[1].plot(S_grid, ref_c, "--", label="Schroder ref", lw=1.5)
        axes[1].set_title("CEV (σ=0.20, β=0.5)")

        # Heston: pick kappa=2.0, xi=0.3, rho=-0.7
        p_hes = next(p for p in heston_params
                     if abs(p.kappa - 2.0) < 1e-4 and abs(p.xi - 0.3) < 1e-4
                     and abs(p.rho + 0.7) < 1e-4)
        pred_h = np.array([model.price(p_hes, S, v=p_hes.v0) for S in S_grid])
        ref_h  = heston_prices_gl(S_grid, np.full(len(S_grid), p_hes.K),
                                  p_hes.T, p_hes.r, p_hes.kappa, p_hes.theta,
                                  p_hes.xi, p_hes.rho, p_hes.v0)
        axes[2].plot(S_grid, pred_h, label="PINN", lw=2)
        axes[2].plot(S_grid, ref_h, "--", label="GL ref", lw=1.5)
        axes[2].set_title("Heston (κ=2,ξ=0.3,ρ=-0.7)")

        for ax in axes:
            ax.set_xlabel("S")
            ax.set_ylabel("V")
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path = os.path.join(args.out, "eval_synthetic.pdf")
        plt.savefig(fig_path)
        print(f"图表已保存至 {fig_path}")
    except Exception as e:
        print(f"[绘图跳过] {e}")


if __name__ == "__main__":
    main()
