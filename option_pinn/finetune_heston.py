# option_pinn/finetune_heston.py
"""在 SPY 市场数据上对 unified_v2 做 fine-tune。

每个到期日先用 L-BFGS-B 校准 Heston 参数，再把该到期日的合约
映射到训练域（moneyness 归一化），加入 ref_data 一起 fine-tune。

用法:
  python finetune_heston.py [--epochs 3000] [--lr 1e-4]
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize, brentq

sys.path.insert(0, os.path.dirname(__file__))
from ref_solvers import heston_call, bsm_call

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE   = os.path.dirname(os.path.abspath(__file__))
K_REF  = 100.0
r_mkt  = 0.05


# ── 数据加载（与 eval_all.py 一致）──────────────────────────────────────────

def _load_spy(moneyness_lo=0.8, moneyness_hi=1.2):
    import re, datetime
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
            rows.append({"S": S_spot, "K": K, "tau": tau, "bid": bid, "ask": ask,
                         "expiry": expiry_date})
        except (ValueError, IndexError):
            continue
    df = pd.DataFrame(rows)
    df["mid"] = (df["bid"] + df["ask"]) / 2.0
    df = df[(df["bid"] > 0) & (df["ask"] > 0)]
    df = df[df["tau"] > 0.05]
    df["moneyness"] = df["S"] / df["K"]
    df = df[(df["moneyness"] >= moneyness_lo) & (df["moneyness"] <= moneyness_hi)]
    return df.reset_index(drop=True)


# ── Per-expiry Heston 校准（与 eval_all.py 一致）────────────────────────────

def _bsm_iv(S, K, T, r, call_price):
    try:
        return brentq(lambda sig: bsm_call(S, K, T, r, sig) - call_price,
                      1e-4, 5.0, xtol=1e-6, maxiter=100)
    except Exception:
        return float("nan")


def _calibrate_heston(calls, S, Ks, T, r, sigma_bsm=0.2):
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
        [1.0, 0.04, 0.2, -0.5, 0.04],
        [2.0, 0.04, 0.3, -0.7, 0.04],
        [3.0, 0.06, 0.4, -0.9, 0.06],
        [0.5, 0.02, 0.1, -0.3, 0.02],
        [5.0, 0.08, 0.5, -0.8, 0.08],
        [1.5, 0.03, 0.25, -0.6, 0.03],
        [4.0, 0.05, 0.35, -0.75, 0.05],
        [0.8, 0.015, 0.15, -0.4, 0.015],
    ]
    best_loss, best_params = np.inf, starts[0]
    for x0 in starts:
        try:
            res = minimize(loss, x0, method="L-BFGS-B", bounds=bounds,
                           options={"maxiter": 200, "ftol": 1e-10})
            if res.fun < best_loss:
                best_loss, best_params = res.fun, res.x.tolist()
        except Exception:
            pass
    return tuple(best_params)


# ── 构建 param_list（与 eval_all.py 一致）───────────────────────────────────

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--lr",     type=float, default=1e-4)
    args = parser.parse_args()

    from unified_pinn_v2 import UnifiedPINN, ModelParams

    # ── 加载 SPY 数据 ──
    print("加载 SPY 数据...")
    df = _load_spy()
    S_spot = float(df["S"].iloc[0])
    scale  = S_spot / K_REF
    print(f"有效合约数: {len(df)}, S_spot={S_spot:.2f}, scale={scale:.4f}")

    # ── 加载 unified_v2 checkpoint ──
    print("加载 unified_v16_gl.pt...")
    param_list = _build_param_list()
    pinn_base = UnifiedPINN(param_list, hidden=128, depth=6, device=DEVICE)
    pinn_base.load(os.path.join(BASE, "results/unified_v16_gl.pt"))

    # ── 按到期日校准 Heston 参数，构建 ref_data ──
    print("\n按到期日校准 Heston 参数...")
    ref_data = {}  # {param_idx: (S_arr, v_arr, t_arr, V_arr)}

    expiry_groups = df.groupby("expiry")
    for exp_date, grp in expiry_groups:
        T_val = float(grp["tau"].iloc[0])
        Ks    = grp["K"].values
        calls = grp["mid"].values

        # BSM IV 中位数（用于 Heston 初始 v0 估计）
        ivs = [_bsm_iv(S_spot, K, T_val, r_mkt, c) for K, c in zip(Ks, calls)]
        ivs = [iv for iv in ivs if not np.isnan(iv) and iv > 0]
        sigma_bsm = float(np.median(ivs)) if ivs else 0.2

        # Heston 校准
        kappa, theta, xi, rho, v0 = _calibrate_heston(
            calls, S_spot, Ks, T_val, r_mkt, sigma_bsm)
        print(f"  {exp_date}  T={T_val:.3f}  kappa={kappa:.2f}  "
              f"theta={theta:.4f}  xi={xi:.3f}  rho={rho:.3f}  v0={v0:.4f}")

        # 找 param_list 中最接近的 Heston 条目
        best_idx, best_dist = 0, float("inf")
        for i, p in enumerate(param_list):
            lam = p.to_lambda_tensor("cpu").squeeze().tolist()
            if abs(lam[0] - 2.0) > 0.1:  # model_type==2 是 Heston
                continue
            dist = ((lam[1]-kappa)**2 + (lam[2]-theta)**2 +
                    (lam[3]-xi)**2 + (lam[4]-rho)**2)
            if dist < best_dist:
                best_dist, best_idx = dist, i

        p_ref = param_list[best_idx]

        # Moneyness 映射：S_scaled = (S/K) * K_REF，V_scaled = mid / K * K_REF
        S_scaled = (S_spot / Ks * K_REF).astype(np.float32)
        V_scaled = (calls / Ks * K_REF).astype(np.float32)
        v_arr    = np.full(len(Ks), v0, dtype=np.float32)
        t_arr    = np.zeros(len(Ks), dtype=np.float32)

        # 过滤超出 S_max 的合约
        mask = (S_scaled < p_ref.S_max) & (S_scaled > 0) & np.isfinite(V_scaled)
        if mask.sum() == 0:
            continue

        S_scaled = S_scaled[mask]
        v_arr    = v_arr[mask]
        t_arr    = t_arr[mask]
        V_scaled = V_scaled[mask]

        # 合并到同一 param_idx 的 ref_data
        if best_idx in ref_data:
            S0, v0_, t0, V0 = ref_data[best_idx]
            ref_data[best_idx] = (
                np.concatenate([S0, S_scaled]),
                np.concatenate([v0_, v_arr]),
                np.concatenate([t0, t_arr]),
                np.concatenate([V0, V_scaled]),
            )
        else:
            ref_data[best_idx] = (S_scaled, v_arr, t_arr, V_scaled)

    total_pts = sum(len(v[0]) for v in ref_data.values())
    print(f"\n共 {len(ref_data)} 个 param 条目，{total_pts} 个训练点")

    # ── Fine-tune ──
    pinn_ft = UnifiedPINN(param_list, hidden=128, depth=6,
                          lr=args.lr, ref_data=ref_data, device=DEVICE)
    pinn_ft.net.load_state_dict(pinn_base.net.state_dict())

    print(f"\n开始 fine-tune，epochs={args.epochs}, lr={args.lr}...")
    pinn_ft.train(
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
    print(f"\nFine-tuned checkpoint → {out_path}")

    # ── 简单验证（前5条合约）──
    pinn_ft.net.eval()
    sample = df.head(5)
    print("\n验证（前5条合约）:")
    print(f"{'S':>8} {'K':>8} {'tau':>6} {'mid':>8} {'pred':>8}")
    for _, row in sample.iterrows():
        T_val = float(row["tau"])
        Ks_s  = np.array([row["K"]])
        calls_s = np.array([row["mid"]])
        ivs_s = [_bsm_iv(S_spot, row["K"], T_val, r_mkt, row["mid"])]
        ivs_s = [iv for iv in ivs_s if not np.isnan(iv) and iv > 0]
        sigma_s = float(np.median(ivs_s)) if ivs_s else 0.2
        kappa_s, theta_s, xi_s, rho_s, v0_s = _calibrate_heston(
            calls_s, S_spot, Ks_s, T_val, r_mkt, sigma_s)
        p = ModelParams.from_heston(kappa=kappa_s, theta=theta_s, xi=xi_s,
                                    rho=rho_s, v0=v0_s,
                                    K=K_REF * row["K"] / S_spot, T=T_val)
        pred_scaled = pinn_ft.price(p, S=K_REF)
        pred = pred_scaled * scale
        print(f"{row['S']:8.2f} {row['K']:8.2f} {row['tau']:6.3f} "
              f"{row['mid']:8.4f} {pred:8.4f}")


if __name__ == "__main__":
    main()
