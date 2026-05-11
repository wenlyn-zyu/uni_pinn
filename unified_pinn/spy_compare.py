"""
spy_compare.py — 解析 CBOE SPY 期权链，计算隐含波动率，与 PINN 定价对比。

用法：
  python spy_compare.py --data data/spy_quotedata.csv --ckpt results/unified_v14.pt
"""

import argparse
import sys
import os
import re
import datetime
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(__file__))
from unified_pinn import ModelParams, UnifiedPINN


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def bs_call_scalar(S, K, T, r, sigma):
    if sigma <= 0 or T <= 1e-6:
        return max(S - K * np.exp(-r * T), 0.0)
    sqt = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / sqt
    d2 = d1 - sqt
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def implied_vol(market_price, S, K, T, r):
    """从看涨市场价格反解隐含波动率，失败返回 nan。"""
    intrinsic = max(S - K * np.exp(-r * T), 0.0)
    if market_price <= intrinsic + 1e-4 or market_price <= 0:
        return float("nan")
    try:
        return brentq(
            lambda sig: bs_call_scalar(S, K, T, r, sig) - market_price,
            1e-4, 10.0, xtol=1e-6, maxiter=200
        )
    except (ValueError, RuntimeError):
        return float("nan")


def parse_expiry_date(date_str):
    """把 'Mon May 11 2026' 解析为 datetime.date。"""
    return datetime.datetime.strptime(date_str.strip(), "%a %b %d %Y").date()


# ---------------------------------------------------------------------------
# 解析 CBOE CSV
# ---------------------------------------------------------------------------

def parse_cboe_csv(path):
    """
    返回 DataFrame，列：expiry_str, expiry_date, strike,
                        call_last, call_iv, put_last, put_iv
    """
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    # 第2行：现价
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
            expiry_str  = parts[0].strip()
            call_last   = float(parts[2])  if parts[2].strip()  else float("nan")
            call_iv     = float(parts[7])  if parts[7].strip()  else float("nan")
            strike      = float(parts[11])
            put_last    = float(parts[13]) if parts[13].strip() else float("nan")
            put_iv      = float(parts[18]) if parts[18].strip() else float("nan")
            rows.append({
                "expiry_str": expiry_str,
                "strike":     strike,
                "call_last":  call_last,
                "call_iv":    call_iv,
                "put_last":   put_last,
                "put_iv":     put_iv,
            })
        except (ValueError, IndexError):
            continue

    df = pd.DataFrame(rows)
    df["expiry_date"] = df["expiry_str"].apply(parse_expiry_date)
    return S, df


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",  type=str, default="data/spy_quotedata.csv")
    parser.add_argument("--ckpt",  type=str, default="results/unified_v14.pt")
    parser.add_argument("--r",     type=float, default=0.043,
                        help="无风险利率（美国1年期国债收益率约4.3%）")
    parser.add_argument("--out",   type=str, default="results/spy_compare.csv")
    # 只取流动性好的合约：volume > min_vol，IV 在合理范围
    parser.add_argument("--min_volume", type=float, default=100)
    args = parser.parse_args()

    today = datetime.date.today()

    # 1. 解析数据
    print(f"解析 {args.data} ...")
    S, df = parse_cboe_csv(args.data)
    print(f"  SPY 现价: {S:.2f}")
    print(f"  原始合约数: {len(df)}")

    # 2. 计算到期时间
    df["T"] = df["expiry_date"].apply(lambda d: (d - today).days / 365.0)
    df = df[df["T"] > 1/365].copy()   # 过滤已到期

    # 3. 过滤：只保留有效看涨价格（volume > 0 用 IV 字段判断）
    df = df[
        df["call_iv"].notna() &
        (df["call_iv"] > 0.01) &
        (df["call_iv"] < 5.0) &
        (df["call_last"] > 0.01)
    ].copy()
    print(f"  过滤后合约数: {len(df)}")

    # 4. 加载 PINN
    param_list = [
        ModelParams.from_bsm(),
        ModelParams.from_cev(beta=0.3), ModelParams.from_cev(beta=0.5),
        ModelParams.from_cev(beta=0.7), ModelParams.from_cev(beta=0.9),
        ModelParams.from_heston(),
    ]
    model = UnifiedPINN(param_list)
    model.load(args.ckpt)
    print(f"  已加载模型: {args.ckpt}\n")

    # 5. 逐合约计算
    r = args.r
    results = []
    for _, row in df.iterrows():
        K   = row["strike"]
        T   = row["T"]
        iv  = row["call_iv"]   # CBOE 已提供 IV，直接用

        # BS 解析价格（用 CBOE 提供的 IV）
        bs_price = bs_call_scalar(S, K, T, r, iv)

        # PINN 定价：归一化到 S'=100，K'=100*(K/S)，结果乘以 S/100
        scale = S / 100.0
        K_n   = 100.0 * K / S
        # T 和 iv 直接传入（PINN 训练时 T=1，这里 T 可能不同，是主要误差来源）
        p = ModelParams.from_bsm(K=K_n, T=T, r=r, sigma=iv)
        pinn_price = model.price(p, S=100.0) * scale

        results.append({
            "expiry":       row["expiry_str"],
            "T_years":      round(T, 4),
            "strike":       K,
            "moneyness":    round(K / S, 4),   # K/S：<1 OTM put，>1 OTM call
            "market_call":  row["call_last"],
            "bs_call":      round(bs_price, 4),
            "pinn_call":    round(pinn_price, 4),
            "iv":           round(iv, 4),
            "bs_err":       round(bs_price - row["call_last"], 4),
            "pinn_err":     round(pinn_price - row["call_last"], 4),
        })

    out = pd.DataFrame(results)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False, encoding="utf-8-sig")

    # 6. 按到期日分组汇总
    print(f"{'到期日':<22} {'T(年)':>6} {'合约数':>6} {'BS_MAE':>8} {'PINN_MAE':>10} {'PINN/BS':>8}")
    print("-" * 68)
    for expiry, grp in out.groupby("expiry"):
        T_val    = grp["T_years"].iloc[0]
        n        = len(grp)
        bs_mae   = grp["bs_err"].abs().mean()
        pinn_mae = grp["pinn_err"].abs().mean()
        ratio    = pinn_mae / bs_mae if bs_mae > 0 else float("nan")
        print(f"{expiry:<22} {T_val:>6.3f} {n:>6} {bs_mae:>8.3f} {pinn_mae:>10.3f} {ratio:>8.2f}x")

    print(f"\n全局 BS MAE:   {out['bs_err'].abs().mean():.4f}")
    print(f"全局 PINN MAE: {out['pinn_err'].abs().mean():.4f}")
    print(f"\n结果已保存至 {args.out}")


if __name__ == "__main__":
    main()
