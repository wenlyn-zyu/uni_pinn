# option_pinn/finetune_heston.py
"""在 SPY 市场数据上对 unified_v2 做 fine-tune（修复版）。

修复了三个根本问题：
1. Heston 条目识别：用 xi>0.01 判断，而非错误的 lam[0]>2.0
2. per-expiry T/r/K：每个到期日构造独立 ModelParams，不用硬编码 T=1/r=0.05
3. 归一化与 eval_all.py 的 _unified_prices_for() 完全一致

用法:
  python finetune_heston.py [--epochs 2000] [--lr 5e-5]
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
r_mkt  = 0.043  # 与 eval_all.py 一致


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
                         "expiry_date": expiry_date})
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


def _to_device(arr, device):
    return torch.tensor(arr, dtype=torch.float32, device=device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr",     type=float, default=5e-5)
    parser.add_argument("--w_pde",  type=float, default=0.05)
    parser.add_argument("--w_data", type=float, default=1.0)
    args = parser.parse_args()

    from unified_pinn_v2 import UnifiedPINN, ModelParams, unified_pde_residual

    # ── 加载 SPY 数据 ──
    print("加载 SPY 数据...")
    df = _load_spy()
    S_spot = float(df["S"].iloc[0])
    scale  = S_spot / K_REF
    print(f"有效合约数: {len(df)}, S_spot={S_spot:.2f}, scale={scale:.4f}")

    # ── 加载 unified_v2 checkpoint ──
    print("加载 unified_v16_gl.pt...")
    param_list = _build_param_list()
    pinn = UnifiedPINN(param_list, hidden=128, depth=6, lr=args.lr, device=DEVICE)
    pinn.load(os.path.join(BASE, "results/unified_v16_gl.pt"))

    # ── 按到期日校准 Heston 参数，构建训练批次 ──
    # 每个到期日：(ModelParams, S_tensor, V_target_tensor)
    print("\n按到期日校准 Heston 参数...")
    expiry_batches = []  # list of (p, S_t, V_t) — 已在训练域归一化

    S_MAX = 300.0

    for exp_date, grp in df.groupby("expiry_date"):
        T_val = float(grp["tau"].iloc[0])
        Ks    = grp["K"].values.astype(float)
        calls = grp["mid"].values.astype(float)

        # BSM IV 中位数
        ivs = [_bsm_iv(S_spot, K, T_val, r_mkt, c) for K, c in zip(Ks, calls)]
        ivs = [iv for iv in ivs if not np.isnan(iv) and iv > 0]
        sigma_bsm = float(np.median(ivs)) if ivs else 0.2

        # Heston 校准
        kappa, theta, xi, rho, v0 = _calibrate_heston(
            calls, S_spot, Ks, T_val, r_mkt, sigma_bsm)
        print(f"  {exp_date}  T={T_val:.3f}  kappa={kappa:.2f}  "
              f"theta={theta:.4f}  xi={xi:.3f}  rho={rho:.3f}  v0={v0:.4f}")

        # 与 eval_all.py _unified_prices_for() 完全一致的归一化：
        #   K_n = K_REF * K_market / S_spot
        #   网络输入 S=K_REF，输出乘 scale 得市场价格
        #   因此训练目标 V_scaled = calls / scale = calls * K_REF / S_spot
        K_ns    = K_REF * Ks / S_spot          # 归一化行权价
        V_scaled = calls / scale               # 归一化目标价格

        # 过滤：K_n 在合理范围内，V_scaled 有限
        mask = (K_ns > 10) & (K_ns < S_MAX * 2) & np.isfinite(V_scaled) & (V_scaled > 0)
        if mask.sum() == 0:
            continue

        K_ns_m    = K_ns[mask]
        V_scaled_m = V_scaled[mask].astype(np.float32)

        # 每个合约构造独立 ModelParams（K=K_n, T=T_val, r=r_mkt）
        for K_n, V_tgt in zip(K_ns_m, V_scaled_m):
            p = ModelParams.from_heston(
                K=float(K_n), T=max(T_val, 0.01), r=r_mkt,
                kappa=kappa, theta=theta, xi=xi, rho=rho, v0=v0,
                S_max=S_MAX
            )
            expiry_batches.append((p, float(K_REF), float(v0), float(V_tgt)))

    print(f"\n共 {len(expiry_batches)} 个训练样本")

    # ── 手动 fine-tune 循环 ──
    optimizer = torch.optim.Adam(pinn.net.parameters(), lr=args.lr)
    pinn.net.train()

    print(f"\n开始 fine-tune，epochs={args.epochs}, lr={args.lr}, "
          f"w_pde={args.w_pde}, w_data={args.w_data}...")

    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()

        # 随机采样一批合约（batch_size=64）
        batch_size = min(64, len(expiry_batches))
        idx = np.random.choice(len(expiry_batches), batch_size, replace=False)
        batch = [expiry_batches[i] for i in idx]

        # ── Data loss ──
        loss_data = torch.tensor(0.0, device=DEVICE)
        for p, S_val, v_val, V_tgt in batch:
            lam = p.to_lambda_tensor(DEVICE)
            S_t = _to_device([[S_val]], DEVICE)
            v_t = _to_device([[v_val]], DEVICE)
            t_t = _to_device([[0.0]],  DEVICE)
            pred = pinn.net(S_t/p.S_max, v_t/p.v_max, t_t/p.T,
                            lam, S_t, t_t, p.K, p.T, p.r)
            rel_err = (pred - V_tgt) / (abs(V_tgt) + p.K * 0.1)
            loss_data = loss_data + rel_err ** 2
        loss_data = loss_data / batch_size

        # ── PDE loss（用第一个样本的参数，随机配点）──
        p0 = batch[0][0]
        n_pde = 256
        S_c = torch.FloatTensor(n_pde, 1).uniform_(1.0, p0.S_max).to(DEVICE)
        v_c = torch.FloatTensor(n_pde, 1).uniform_(1e-4, p0.v_max).to(DEVICE)
        t_c = torch.FloatTensor(n_pde, 1).uniform_(0.0, p0.T * 0.999).to(DEVICE)
        lam_c = p0.to_lambda_tensor(DEVICE).expand(n_pde, -1)
        res = unified_pde_residual(pinn.net, S_c, v_c, t_c, lam_c,
                                   p0.K, p0.T, p0.r, p0.S_max, p0.v_max)
        loss_pde = torch.mean(res ** 2)

        loss = args.w_pde * loss_pde + args.w_data * loss_data
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pinn.net.parameters(), max_norm=0.5)
        optimizer.step()

        if epoch % 200 == 0:
            print(f"  epoch {epoch:4d}  loss={loss.item():.3e}  "
                  f"pde={loss_pde.item():.3e}  data={loss_data.item():.3e}")

    # ── 保存 ──
    out_path = os.path.join(BASE, "results/unified_v2_ft.pt")
    pinn.save(out_path)
    print(f"\nFine-tuned checkpoint → {out_path}")

    # ── 验证：对每个到期日计算 MAE，与 unified_v2 对比 ──
    print("\n=== 验证（per-expiry MAE vs 市场）===")
    pinn.net.eval()

    # 加载原始 unified_v2 用于对比
    pinn_base = UnifiedPINN(param_list, hidden=128, depth=6, device=DEVICE)
    pinn_base.load(os.path.join(BASE, "results/unified_v16_gl.pt"))
    pinn_base.net.eval()

    print(f"{'到期日':12s} {'T':>6} {'n':>4} {'MAE_base':>10} {'MAE_ft':>10} {'改善':>8}")
    total_base, total_ft, total_n = 0.0, 0.0, 0
    for exp_date, grp in df.groupby("expiry_date"):
        T_val = float(grp["tau"].iloc[0])
        Ks    = grp["K"].values.astype(float)
        calls = grp["mid"].values.astype(float)

        ivs = [_bsm_iv(S_spot, K, T_val, r_mkt, c) for K, c in zip(Ks, calls)]
        ivs = [iv for iv in ivs if not np.isnan(iv) and iv > 0]
        sigma_bsm = float(np.median(ivs)) if ivs else 0.2
        kappa, theta, xi, rho, v0 = _calibrate_heston(
            calls, S_spot, Ks, T_val, r_mkt, sigma_bsm)

        preds_base, preds_ft = [], []
        for K in Ks:
            K_n = K_REF * K / S_spot
            p = ModelParams.from_heston(K=K_n, T=max(T_val, 0.01), r=r_mkt,
                                        kappa=kappa, theta=theta,
                                        xi=xi, rho=rho, v0=v0, S_max=S_MAX)
            preds_base.append(pinn_base.price(p, S=K_REF) * scale)
            preds_ft.append(pinn.price(p, S=K_REF) * scale)

        mae_base = float(np.mean(np.abs(np.array(preds_base) - calls)))
        mae_ft   = float(np.mean(np.abs(np.array(preds_ft)   - calls)))
        improve  = (mae_base - mae_ft) / (mae_base + 1e-8) * 100
        print(f"  {str(exp_date):12s} {T_val:6.3f} {len(Ks):4d} "
              f"{mae_base:10.4f} {mae_ft:10.4f} {improve:+7.1f}%")
        total_base += mae_base * len(Ks)
        total_ft   += mae_ft   * len(Ks)
        total_n    += len(Ks)

    overall_base = total_base / total_n
    overall_ft   = total_ft   / total_n
    print(f"\n  {'全局':12s} {'':>6} {total_n:4d} "
          f"{overall_base:10.4f} {overall_ft:10.4f} "
          f"{(overall_base-overall_ft)/overall_base*100:+7.1f}%")


if __name__ == "__main__":
    main()
