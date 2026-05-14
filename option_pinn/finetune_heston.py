# option_pinn/finetune_heston.py
"""在 SPY 市场数据上对 unified_v2 的 Heston 分支做 fine-tune。

用法:
  python finetune_heston.py [--epochs 3000] [--lr 1e-4]
"""
import os
import sys
import argparse
import re
import datetime
import numpy as np
import pandas as pd
import torch
from scipy.optimize import brentq

sys.path.insert(0, os.path.dirname(__file__))
from ref_solvers import bsm_call

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE   = os.path.dirname(os.path.abspath(__file__))


def _load_spy(moneyness_lo=0.8, moneyness_hi=1.2):
    csv_path = os.path.join(BASE, "data/spy_quotedata.csv")
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
            rows.append({"S": S_spot, "K": K, "tau": tau,
                         "bid": bid, "ask": ask})
        except (ValueError, IndexError):
            continue
    df = pd.DataFrame(rows)
    df["mid"] = (df["bid"] + df["ask"]) / 2.0
    df = df[(df["bid"] > 0) & (df["ask"] > 0)]
    df = df[df["tau"] > 0.05]
    df["moneyness"] = df["S"] / df["K"]
    df = df[(df["moneyness"] >= moneyness_lo) & (df["moneyness"] <= moneyness_hi)]
    return df.reset_index(drop=True)


def _build_param_list():
    from unified_pinn_v2 import ModelParams
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
    return params


def _find_heston_idx(param_list, kappa, theta, xi, rho):
    """在 param_list 中找到最接近目标参数的 Heston 条目索引。"""
    from unified_pinn_v2 import ModelParams
    best_idx, best_dist = 0, float("inf")
    for i, p in enumerate(param_list):
        lam = p.to_lambda_tensor("cpu").squeeze().tolist()
        # lam = [model_type, kappa, theta, xi, rho, sigma/beta]
        if abs(lam[0] - 2.0) > 0.1:  # model_type==2 表示 Heston
            continue
        dist = (lam[1]-kappa)**2 + (lam[2]-theta)**2 + (lam[3]-xi)**2 + (lam[4]-rho)**2
        if dist < best_dist:
            best_dist, best_idx = dist, i
    return best_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--lr",     type=float, default=1e-4)
    args = parser.parse_args()

    from unified_pinn_v2 import UnifiedPINN, ModelParams

    # ── 加载 SPY 数据 ──
    print("加载 SPY 数据...")
    df = _load_spy()
    print(f"有效合约数: {len(df)}")

    # ── 加载 unified_v2 checkpoint ──
    print("加载 unified_v16_gl.pt...")
    param_list = _build_param_list()
    pinn = UnifiedPINN(param_list, hidden=128, depth=6, device=DEVICE)
    pinn.load(os.path.join(BASE, "results/unified_v16_gl.pt"))
    pinn.net.train()

    # ── 构建 Heston 参数（与 eval_all.py 一致）──
    HESTON_FT = dict(kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, v0=0.04)
    heston_idx = _find_heston_idx(param_list, **{k: v for k, v in HESTON_FT.items() if k != "v0"})
    print(f"Heston 参数索引: {heston_idx}")

    # ── 构建 ref_data：用 SPY mid-price 作为数据锚点 ──
    p = param_list[heston_idx]
    S_arr = df["S"].values.astype(np.float32)
    # 用 v0 作为初始方差（固定）
    v_arr = np.full(len(df), HESTON_FT["v0"], dtype=np.float32)
    # t=0（当前时刻），tau 作为 T（到期时间）
    t_arr = np.zeros(len(df), dtype=np.float32)
    V_arr = df["mid"].values.astype(np.float32)

    # 过滤掉 S 超出 S_max 的合约
    mask = S_arr < p.S_max
    S_arr, v_arr, t_arr, V_arr = S_arr[mask], v_arr[mask], t_arr[mask], V_arr[mask]
    print(f"过滤后合约数: {len(S_arr)}")

    ref_data = {heston_idx: (S_arr, v_arr, t_arr, V_arr)}

    # ── 重建 UnifiedPINN 并注入 ref_data ──
    pinn_ft = UnifiedPINN(param_list, hidden=128, depth=6,
                          lr=args.lr, ref_data=ref_data, device=DEVICE)
    pinn_ft.net.load_state_dict(pinn.net.state_dict())

    # ── Fine-tune：降低 PDE 权重，提高数据权重 ──
    print(f"开始 fine-tune，epochs={args.epochs}, lr={args.lr}...")
    history = pinn_ft.train(
        epochs=args.epochs,
        n_per_model=500,
        w_pde=0.1,
        w_bc=1.0,
        w_ic=1.0,
        w_data=1.0,
        log_every=200,
    )

    # ── 保存 ──
    out_path = os.path.join(BASE, "results/unified_v2_ft.pt")
    pinn_ft.save(out_path)
    print(f"Fine-tuned checkpoint → {out_path}")

    # ── 简单验证 ──
    pinn_ft.net.eval()
    p_heston = ModelParams.from_heston(**HESTON_FT)
    sample = df.head(5)
    print("\n验证（前5条合约）:")
    print(f"{'S':>8} {'K':>8} {'tau':>6} {'mid':>8} {'pred':>8}")
    for _, row in sample.iterrows():
        pred = pinn_ft.price(p_heston, row["S"])
        print(f"{row['S']:8.2f} {row['K']:8.2f} {row['tau']:6.3f} "
              f"{row['mid']:8.4f} {pred:8.4f}")


if __name__ == "__main__":
    main()
