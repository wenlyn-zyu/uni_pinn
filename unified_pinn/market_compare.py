"""
market_compare.py — 从 akshare 下载上证50指数期权数据，
计算隐含波动率，与 PINN 定价结果对比。

用法：
  python market_compare.py --expiry ho2606 --out results/market_compare.csv
"""

import argparse
import sys
import os
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(__file__))
from unified_pinn import ModelParams, UnifiedPINN, _bs_call
import torch


# ---------------------------------------------------------------------------
# BS 隐含波动率（标量版，用于 scipy 求根）
# ---------------------------------------------------------------------------

def bs_call_scalar(S, K, T, r, sigma):
    if sigma <= 0 or T <= 0:
        return max(S - K * np.exp(-r * T), 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def implied_vol(market_price, S, K, T, r, option_type="call"):
    """用 Brent 法从市场价格反解隐含波动率。失败返回 nan。"""
    if option_type == "put":
        # put-call parity 转换为看涨价格
        market_price = market_price + S - K * np.exp(-r * T)
    intrinsic = max(S - K * np.exp(-r * T), 0.0)
    if market_price <= intrinsic + 1e-6:
        return float("nan")
    try:
        iv = brentq(
            lambda sig: bs_call_scalar(S, K, T, r, sig) - market_price,
            1e-4, 5.0, xtol=1e-6, maxiter=200
        )
        return iv
    except (ValueError, RuntimeError):
        return float("nan")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expiry", type=str, default="ho2606",
                        help="合约月份代码，如 ho2606（2026年6月）")
    parser.add_argument("--ckpt",   type=str, default="results/unified_v14.pt")
    parser.add_argument("--r",      type=float, default=0.02,
                        help="无风险利率（A股用1年期LPR，约2%）")
    parser.add_argument("--out",    type=str, default="results/market_compare.csv")
    args = parser.parse_args()

    # 1. 下载期权链
    print(f"下载 {args.expiry} 期权链...")
    import akshare as ak
    df = ak.option_cffex_sz50_spot_sina(symbol=args.expiry)

    # 2. 获取标的现价（上证50指数）
    try:
        idx = ak.stock_zh_index_spot_sina(symbol="sh000016")
        S = float(idx[idx["代码"] == "sh000016"]["最新价"].values[0])
    except Exception:
        # 备用：用 ATM 行权价附近的中间值估算
        strikes = df["行权价"].astype(float).values
        S = float(strikes[len(strikes) // 2])
        print(f"  无法获取指数现价，使用估算值 S={S}")
    print(f"  上证50指数现价: {S:.2f}")

    # 3. 计算到期时间（年）
    # expiry 格式 ho2606 → 2026年6月，取第三个周五
    year  = 2000 + int(args.expiry[2:4])
    month = int(args.expiry[4:6])
    import datetime
    # 找该月第三个周五
    d = datetime.date(year, month, 1)
    fridays = []
    while d.month == month:
        if d.weekday() == 4:  # 周五
            fridays.append(d)
        d += datetime.timedelta(days=1)
    expiry_date = fridays[2] if len(fridays) >= 3 else fridays[-1]
    today = datetime.date.today()
    T = max((expiry_date - today).days / 365.0, 1/365)
    print(f"  到期日: {expiry_date}，剩余 {T*365:.0f} 天（T={T:.4f}年）")

    # 4. 加载 PINN 模型
    param_list = [
        ModelParams.from_bsm(),
        ModelParams.from_cev(beta=0.3), ModelParams.from_cev(beta=0.5),
        ModelParams.from_cev(beta=0.7), ModelParams.from_cev(beta=0.9),
        ModelParams.from_heston(),
    ]
    model = UnifiedPINN(param_list)
    model.load(args.ckpt)
    print(f"  已加载模型: {args.ckpt}")

    # 5. 逐行权价计算隐含波动率和 PINN 价格
    r = args.r
    rows = []
    for _, row in df.iterrows():
        K = float(row["行权价"])
        call_price = float(row["看涨合约-最新价"])
        put_price  = float(row["看跌合约-最新价"])

        # 隐含波动率（从看涨价格反解）
        iv = implied_vol(call_price, S, K, T, r, option_type="call")

        # PINN 定价：归一化到 S'=100，K'=100*(K/S)，结果乘以 S/100 还原
        # 这样利用了期权定价的齐次性：V(S,K) = S * V(1, K/S)
        if not np.isnan(iv):
            scale = S / 100.0
            K_n   = 100.0 * K / S
            p_bsm = ModelParams.from_bsm(K=K_n, T=T, r=r, sigma=iv)
            pinn_call = model.price(p_bsm, S=100.0) * scale
            p_put = ModelParams.from_bsm(K=K_n, T=T, r=r, sigma=iv,
                                          option_type="put")
            pinn_put = model.price(p_put, S=100.0) * scale
        else:
            pinn_call = pinn_put = float("nan")

        # BS 解析价格（用于验证）
        bs_call = bs_call_scalar(S, K, T, r, iv) if not np.isnan(iv) else float("nan")
        bs_put  = bs_call - S + K * np.exp(-r * T) if not np.isnan(iv) else float("nan")

        rows.append({
            "行权价": K,
            "市场看涨价": call_price,
            "市场看跌价": put_price,
            "隐含波动率": round(iv, 4) if not np.isnan(iv) else float("nan"),
            "BS看涨价":   round(bs_call, 4) if not np.isnan(bs_call) else float("nan"),
            "PINN看涨价": round(pinn_call, 4) if not np.isnan(pinn_call) else float("nan"),
            "PINN看涨误差": round(pinn_call - call_price, 4) if not np.isnan(pinn_call) else float("nan"),
            "BS看跌价":   round(bs_put, 4) if not np.isnan(bs_put) else float("nan"),
            "PINN看跌价": round(pinn_put, 4) if not np.isnan(pinn_put) else float("nan"),
        })

    result = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    result.to_csv(args.out, index=False, encoding="utf-8-sig")

    # 6. 打印摘要
    print(f"\n{'行权价':>8} {'市场看涨':>10} {'PINN看涨':>10} {'误差':>8} {'隐含波动率':>10}")
    print("-" * 55)
    valid = result.dropna(subset=["隐含波动率"])
    for _, r_ in valid.iterrows():
        print(f"{r_['行权价']:>8.0f} {r_['市场看涨价']:>10.2f} "
              f"{r_['PINN看涨价']:>10.2f} {r_['PINN看涨误差']:>+8.2f} "
              f"{r_['隐含波动率']:>10.4f}")

    mae = valid["PINN看涨误差"].abs().mean()
    print(f"\nMAE（有效行权价）: {mae:.4f}")
    print(f"结果已保存至 {args.out}")


if __name__ == "__main__":
    main()
