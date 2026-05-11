"""
calibrate_compare.py — 对 SPY 期权链做 BSM/CEV/Heston 参数校准，
对比三个模型和 PINN 的定价精度。

用法：
  python calibrate_compare.py --data data/spy_quotedata.csv \
      --expiry "Tue Jun 30 2026" --ckpt results/unified_v14.pt
"""

import argparse
import sys
import os
import re
import datetime
import numpy as np
import pandas as pd
from scipy.optimize import minimize, brentq
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(__file__))
from unified_pinn import ModelParams, UnifiedPINN
from evaluate import cn_cev_price


# ---------------------------------------------------------------------------
# BS 解析解（标量）
# ---------------------------------------------------------------------------

def bs_call(S, K, T, r, sigma):
    if sigma <= 0 or T <= 1e-6:
        return max(S - K * np.exp(-r * T), 0.0)
    sqt = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / sqt
    d2 = d1 - sqt
    return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))


# ---------------------------------------------------------------------------
# Heston 半解析解（特征函数法）
# ---------------------------------------------------------------------------

def heston_call(S, K, T, r, kappa, theta, xi, rho, v0):
    """Heston (1993) 半解析看涨价格，Little Heston Trap 稳定公式（Albrecher 2007）。"""
    from scipy.integrate import quad

    def char_func(phi, j):
        if j == 1:
            u, b = 0.5, kappa - rho * xi
        else:
            u, b = -0.5, kappa
        a = kappa * theta
        x = np.log(S)   # Little Heston Trap: x=log(S), strike handled by exp(-i*phi*log(K))
        d = np.sqrt((b - rho * xi * 1j * phi)**2
                    - xi**2 * (2 * u * 1j * phi - phi**2))
        g = (b - rho * xi * 1j * phi - d) / (b - rho * xi * 1j * phi + d)
        exp_neg_dT = np.exp(-d * T)
        denom = 1.0 - g * exp_neg_dT
        if abs(denom) < 1e-12:
            return 0.0 + 0.0j
        C = r * 1j * phi * T + (a / xi**2) * (
            (b - rho * xi * 1j * phi - d) * T
            - 2.0 * np.log(denom / (1.0 - g))
        )
        D = ((b - rho * xi * 1j * phi - d) / xi**2) * (
            (1.0 - exp_neg_dT) / denom
        )
        return np.exp(C + D * v0 + 1j * phi * x)

    def integrand(phi, j):
        try:
            val = char_func(phi, j)
            return np.real(np.exp(-1j * phi * np.log(K)) * val / (1j * phi))
        except Exception:
            return 0.0

    try:
        P1 = 0.5 + (1 / np.pi) * quad(integrand, 1e-6, 100,
                                        args=(1,), limit=100)[0]
        P2 = 0.5 + (1 / np.pi) * quad(integrand, 1e-6, 100,
                                        args=(2,), limit=100)[0]
        return float(max(S * P1 - K * np.exp(-r * T) * P2, 0.0))
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# 解析 CBOE CSV
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
            rows.append({
                "expiry_str": parts[0].strip(),
                "strike":     float(parts[11]),
                "call_last":  float(parts[2])  if parts[2].strip()  else np.nan,
                "call_bid":   float(parts[4])  if parts[4].strip()  else np.nan,
                "call_ask":   float(parts[5])  if parts[5].strip()  else np.nan,
                "call_vol":   float(parts[6])  if parts[6].strip()  else 0,
                "call_iv":    float(parts[7])  if parts[7].strip()  else np.nan,
            })
        except (ValueError, IndexError):
            continue
    return S, pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 参数校准
# ---------------------------------------------------------------------------

def calibrate_bsm(S, strikes, market_prices, T, r):
    """BSM 校准：用 IV 均值作为 σ*。"""
    ivs = []
    for K, mp in zip(strikes, market_prices):
        try:
            iv = brentq(lambda s: bs_call(S, K, T, r, s) - mp,
                        1e-4, 5.0, xtol=1e-6)
            ivs.append(iv)
        except Exception:
            pass
    sigma = float(np.median(ivs)) if ivs else 0.2
    prices = np.array([bs_call(S, K, T, r, sigma) for K in strikes])
    return {"sigma": sigma}, prices


def calibrate_cev(S, strikes, market_prices, T, r):
    """CEV 校准：最小化加权 MSE，参数 (sigma, beta)，beta ∈ (0,1]。
    CEV 局部波动率：σ_local(K) = sigma * (S/K)^(1-beta)
    """
    def objective(params):
        sigma, beta = params
        if sigma <= 0.01 or beta <= 0.01 or beta > 1.0:
            return 1e10
        try:
            preds = np.array([bs_call(S, K, T, r, sigma * (S/K)**(1-beta))
                               for K in strikes])
            w = 1.0 / (np.array(market_prices) + 1.0)
            return float(np.mean(w * (preds - market_prices)**2))
        except Exception:
            return 1e10

    best, best_val = None, 1e10
    for s0 in [0.10, 0.15, 0.20, 0.25]:
        for b0 in [0.3, 0.5, 0.7, 0.9]:
            res = minimize(objective, [s0, b0], method="L-BFGS-B",
                           bounds=[(0.01, 2.0), (0.01, 1.0)],
                           options={"maxiter": 1000, "ftol": 1e-10})
            if res.fun < best_val:
                best_val = res.fun
                best = res.x
    sigma, beta = best
    prices = np.array([bs_call(S, K, T, r, sigma * (S/K)**(1-beta))
                        for K in strikes])
    return {"sigma": sigma, "beta": beta}, prices


def calibrate_heston(S, strikes, market_prices, T, r):
    """Heston 校准：多起点随机搜索 + L-BFGS-B 精化。"""
    def objective(params):
        kappa, theta, xi, rho, v0 = params
        if (kappa <= 0.05 or theta <= 0.001 or xi <= 0.01 or
                abs(rho) >= 0.99 or v0 <= 0.001):
            return 1e10
        try:
            preds = np.array([heston_call(S, K, T, r, kappa, theta, xi, rho, v0)
                               for K in strikes])
            bad = np.isnan(preds) | (preds < 0)
            if bad.sum() > len(preds) * 0.3:
                return 1e10
            preds = np.where(bad, np.array(market_prices), preds)
            w = 1.0 / (np.array(market_prices) + 1.0)
            return float(np.mean(w * (preds - market_prices)**2))
        except Exception:
            return 1e10

    bounds = [(0.05, 20), (0.001, 0.5), (0.01, 2.5), (-0.98, 0.0), (0.001, 0.5)]
    # 多起点搜索，避免局部最优
    best, best_val = None, 1e10
    starts = [
        [2.0, 0.04, 0.30, -0.70, 0.04],
        [1.0, 0.02, 0.50, -0.50, 0.02],
        [3.0, 0.06, 0.40, -0.80, 0.06],
        [5.0, 0.03, 0.60, -0.60, 0.03],
        [1.5, 0.05, 0.20, -0.40, 0.05],
        [4.0, 0.04, 0.80, -0.75, 0.04],
        [0.5, 0.03, 0.35, -0.65, 0.03],
        [8.0, 0.02, 0.45, -0.55, 0.02],
    ]
    for x0 in starts:
        res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 1000, "ftol": 1e-10, "gtol": 1e-7})
        if res.fun < best_val:
            best_val = res.fun
            best = res.x
    if best is None:
        print("  警告：Heston 校准失败，使用默认参数")
        best = [2.0, 0.04, 0.3, -0.7, 0.04]
    kappa, theta, xi, rho, v0 = best
    prices = np.array([heston_call(S, K, T, r, kappa, theta, xi, rho, v0)
                        for K in strikes])
    return {"kappa": kappa, "theta": theta, "xi": xi,
            "rho": rho, "v0": v0}, prices


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   type=str, default="data/spy_quotedata.csv")
    parser.add_argument("--expiry", type=str, default="Tue Jun 30 2026")
    parser.add_argument("--ckpt",   type=str, default="results/unified_v14.pt")
    parser.add_argument("--r",      type=float, default=0.043)
    parser.add_argument("--out",    type=str, default="results/calibrate_compare.csv")
    args = parser.parse_args()

    today = datetime.date.today()

    # 1. 解析数据，筛选指定到期日
    print(f"解析数据，到期日: {args.expiry}")
    S, df = parse_cboe_csv(args.data)
    print(f"  SPY 现价: S = {S:.2f}")

    sub = df[df["expiry_str"] == args.expiry].copy()
    if len(sub) == 0:
        print(f"  错误：找不到到期日 '{args.expiry}'")
        print("  可用到期日:", df["expiry_str"].unique()[:10])
        return

    # 过滤：ATM ±15%，流动性好，Heston 数值积分在此范围稳定
    sub = sub[
        sub["call_iv"].notna() & (sub["call_iv"] > 0.01) & (sub["call_iv"] < 2.0) &
        (sub["call_last"] > 0.10) &
        (sub["strike"] >= S * 0.85) & (sub["strike"] <= S * 1.15)
    ].copy()
    sub = sub.sort_values("strike").reset_index(drop=True)
    print(f"  有效合约数: {len(sub)}（行权价 {sub['strike'].min():.0f}~{sub['strike'].max():.0f}）")

    expiry_date = datetime.datetime.strptime(args.expiry, "%a %b %d %Y").date()
    T = max((expiry_date - today).days / 365.0, 1/365)
    r = args.r
    print(f"  T = {T:.4f} 年（{(expiry_date-today).days} 天），r = {r}")

    strikes = sub["strike"].values
    market  = sub["call_last"].values

    # 2. 三个模型校准
    print("\n校准 BSM ...")
    bsm_params, bsm_prices = calibrate_bsm(S, strikes, market, T, r)
    print(f"  σ* = {bsm_params['sigma']:.4f}")

    print("校准 CEV ...")
    cev_params, cev_prices = calibrate_cev(S, strikes, market, T, r)
    print(f"  σ* = {cev_params['sigma']:.4f}, β* = {cev_params['beta']:.4f}")

    print("校准 Heston ...")
    heston_params, heston_prices = calibrate_heston(S, strikes, market, T, r)
    print(f"  κ={heston_params['kappa']:.3f}, θ={heston_params['theta']:.4f}, "
          f"ξ={heston_params['xi']:.3f}, ρ={heston_params['rho']:.3f}, "
          f"v₀={heston_params['v0']:.4f}")

    # 3. PINN 定价（用各模型校准参数）
    print("\n加载 PINN ...")
    param_list = [
        ModelParams.from_bsm(),
        ModelParams.from_cev(beta=0.3), ModelParams.from_cev(beta=0.5),
        ModelParams.from_cev(beta=0.7), ModelParams.from_cev(beta=0.9),
        ModelParams.from_heston(),
    ]
    model = UnifiedPINN(param_list)
    model.load(args.ckpt)

    scale = S / 100.0

    # PINN-BSM
    pinn_bsm = np.array([
        model.price(ModelParams.from_bsm(
            K=100.*K/S, T=T, r=r, sigma=bsm_params["sigma"]),
            S=100.) * scale
        for K in strikes
    ])

    # PINN-Heston
    hp = heston_params
    pinn_heston = np.array([
        model.price(ModelParams.from_heston(
            K=100.*K/S, T=T, r=r,
            kappa=hp["kappa"], theta=hp["theta"],
            xi=hp["xi"], rho=hp["rho"], v0=hp["v0"]),
            S=100.) * scale
        for K in strikes
    ])

    # 4. 汇总结果
    result = pd.DataFrame({
        "strike":       strikes,
        "moneyness":    (strikes / S).round(4),
        "market":       market,
        "bsm":          bsm_prices.round(4),
        "cev":          cev_prices.round(4),
        "heston":       heston_prices.round(4),
        "pinn_bsm":     pinn_bsm.round(4),
        "pinn_heston":  pinn_heston.round(4),
        "err_bsm":      (bsm_prices    - market).round(4),
        "err_cev":      (cev_prices     - market).round(4),
        "err_heston":   (heston_prices  - market).round(4),
        "err_pinn_bsm": (pinn_bsm       - market).round(4),
        "err_pinn_hes": (pinn_heston    - market).round(4),
    })
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    result.to_csv(args.out, index=False, encoding="utf-8-sig")

    # 5. 打印摘要
    def mae(col): return result[col].abs().mean()
    def rmse(col): return np.sqrt((result[col]**2).mean())

    print(f"\n{'模型':<14} {'MAE':>8} {'RMSE':>8}")
    print("-" * 32)
    for name, col in [("BSM",       "err_bsm"),
                       ("CEV",       "err_cev"),
                       ("Heston",    "err_heston"),
                       ("PINN-BSM",  "err_pinn_bsm"),
                       ("PINN-Heston","err_pinn_hes")]:
        print(f"{name:<14} {mae(col):>8.4f} {rmse(col):>8.4f}")

    print(f"\n结果已保存至 {args.out}")
    print(f"\n校准参数摘要:")
    print(f"  BSM:    σ = {bsm_params['sigma']:.4f}")
    print(f"  CEV:    σ = {cev_params['sigma']:.4f}, β = {cev_params['beta']:.4f}")
    print(f"  Heston: κ={hp['kappa']:.3f}, θ={hp['theta']:.4f}, "
          f"ξ={hp['xi']:.3f}, ρ={hp['rho']:.3f}, v₀={hp['v0']:.4f}")


if __name__ == "__main__":
    main()
