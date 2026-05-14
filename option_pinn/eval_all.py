# option_pinn/eval_all.py
"""统一评估脚本：合成数据 + 真实市场数据，输出 CSV 供论文引用。

用法:
  python eval_all.py --mode synthetic   # 合成数据评估
  python eval_all.py --mode market      # 真实市场数据评估
  python eval_all.py --mode all         # 全部（默认）
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from scipy.optimize import brentq

sys.path.insert(0, os.path.dirname(__file__))
from ref_solvers import (bsm_call, bsm_greeks,
                          cev_call,
                          heston_call, heston_greeks_fd)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE   = os.path.dirname(os.path.abspath(__file__))

# ── 合成数据固定参数（与各模型训练参数一致）──────────────────────────────────
K_SYN, T_SYN, r_SYN = 100.0, 1.0, 0.05
S_GRID = np.linspace(50, 250, 50)

BSM_PARAMS    = dict(sigma=0.2)
CEV_PARAMS    = dict(sigma=0.25, beta=0.5)
# 独立 CEV PINN 实际训练参数（从 checkpoint 读取）
CEV_INDEP     = dict(sigma=0.2, beta=0.5)
# 独立 Heston PINN 实际训练参数（从 checkpoint 读取）
HESTON_INDEP  = dict(kappa=1.0, theta=0.08, xi=0.39, rho=-0.93, v0=0.04)
r_HESTON_INDEP = 0.1   # Heston 独立 PINN 训练时用的 r
# Unified v2 训练时覆盖的 Heston 参数集（取代表性一组）
HESTON_UNIFIED = dict(kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, v0=0.04)


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def _mse_relmse(pred, ref):
    pred, ref = np.array(pred, dtype=float), np.array(ref, dtype=float)
    mse    = float(np.mean((pred - ref) ** 2))
    # 只在参考解足够大时计算相对误差（避免深度虚值时分母趋零）
    mask   = np.abs(ref) > 0.01
    if mask.sum() == 0:
        relmse = float("nan")
    else:
        relmse = float(np.mean(((pred[mask] - ref[mask]) / np.abs(ref[mask])) ** 2))
    return mse, relmse


def _mae(pred, ref):
    return float(np.mean(np.abs(np.array(pred) - np.array(ref))))


# ── 模型加载 ──────────────────────────────────────────────────────────────────

def load_indep_bsm():
    sys.path.insert(0, os.path.join(BASE, "independent"))
    from bsm_pinn import GatedPINN, BSM_PINN
    ckpt_path = os.path.join(BASE, "results/indep_bsm.pt")
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    params = ckpt["params"]
    pinn = BSM_PINN(**params, device=DEVICE)
    pinn.net.load_state_dict(ckpt["state_dict"])
    pinn.net.eval()
    return pinn


def load_indep_cev():
    sys.path.insert(0, os.path.join(BASE, "independent"))
    from cev_pinn import GatedPINN, CEV_PINN
    ckpt_path = os.path.join(BASE, "results/indep_cev.pt")
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    params = ckpt["params"]
    pinn = CEV_PINN(**params, device=DEVICE)
    pinn.net.load_state_dict(ckpt["state_dict"])
    pinn.net.eval()
    return pinn


def load_indep_heston():
    sys.path.insert(0, os.path.join(BASE, "independent"))
    from heston_pinn import Heston_PINN
    ckpt_path = os.path.join(BASE, "results/indep_heston.pt")
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    params = ckpt["params"]
    pinn = Heston_PINN(**params, device=DEVICE)
    pinn.aux_net.load_state_dict(ckpt["aux_state"])
    pinn.main_net.load_state_dict(ckpt["main_state"])
    pinn.aux_net.eval()
    pinn.main_net.eval()
    return pinn


def _build_param_list():
    """复现 unified_pinn_v2 训练时的参数列表（BSM×6 + CEV×12 + Heston×36 = 54）。"""
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


def load_unified(ckpt_name="unified_v16_gl.pt"):
    from unified_pinn_v2 import UnifiedPINN
    param_list = _build_param_list()
    pinn = UnifiedPINN(param_list, hidden=128, depth=6, device=DEVICE)
    pinn.load(os.path.join(BASE, f"results/{ckpt_name}"))
    pinn.net.eval()
    return pinn


def load_parametric():
    sys.path.insert(0, os.path.join(BASE, "parametric_pinn"))
    from fully_parametric_pinn import FullyParametricPINN
    pinn = FullyParametricPINN(device=DEVICE)
    pinn.load(os.path.join(BASE, "parametric_pinn/results/fully_param_v1.pt"))
    pinn.net.eval()
    return pinn


# ── Greeks via autograd（unified PINN）────────────────────────────────────────

def unified_greeks_autograd(pinn, p, S_arr):
    """计算 unified PINN 的 Delta、Gamma、Vega（autograd）。"""
    from unified_pinn_v2 import ModelParams
    results = []
    lam = p.to_lambda_tensor(DEVICE)
    for S_val in S_arr:
        S_t = torch.tensor([[float(S_val)]], dtype=torch.float32,
                            device=DEVICE, requires_grad=True)
        v_t = torch.tensor([[float(p.v0)]], dtype=torch.float32,
                            device=DEVICE, requires_grad=True)
        t_t = torch.tensor([[0.0]], dtype=torch.float32, device=DEVICE)
        V = pinn.net(S_t/p.S_max, v_t/p.v_max, t_t/p.T,
                     lam, S_t, t_t, p.K, p.T, p.r)
        dV_dS = torch.autograd.grad(V, S_t, create_graph=True, retain_graph=True)[0]
        d2V_dS2 = torch.autograd.grad(dV_dS, S_t, retain_graph=True)[0]
        dV_dv   = torch.autograd.grad(V, v_t, retain_graph=True)[0]
        results.append({
            "delta": float(dV_dS.item()),
            "gamma": float(d2V_dS2.item()),
            "vega":  float(dV_dv.item()),
        })
    return results


# ── 合成数据评估 ──────────────────────────────────────────────────────────────

def eval_synthetic():
    print("=== 合成数据评估 ===")
    rows_price  = []
    rows_greeks = []

    # 参考解
    ref_bsm    = [bsm_call(S, K_SYN, T_SYN, r_SYN, **BSM_PARAMS)    for S in S_GRID]
    ref_cev    = [cev_call(S, K_SYN, T_SYN, r_SYN, **CEV_PARAMS)     for S in S_GRID]
    # 独立 PINN 各自用训练时的参数生成参考解
    ref_cev_i    = [cev_call(S, K_SYN, T_SYN, r_SYN, **CEV_INDEP)        for S in S_GRID]
    ref_heston_i = [heston_call(S, K_SYN, T_SYN, r_HESTON_INDEP, **HESTON_INDEP) for S in S_GRID]
    ref_heston_u = [heston_call(S, K_SYN, T_SYN, r_SYN, **HESTON_UNIFIED) for S in S_GRID]
    ref_bsm_g    = [bsm_greeks(S, K_SYN, T_SYN, r_SYN, **BSM_PARAMS)      for S in S_GRID]
    ref_heston_g = [heston_greeks_fd(S, K_SYN, T_SYN, r_SYN, **HESTON_UNIFIED) for S in S_GRID]

    def add_price(name, pred_bsm, pred_cev, pred_heston, ref_heston):
        bm, br = _mse_relmse(pred_bsm,    ref_bsm)
        cm, cr = _mse_relmse(pred_cev,    ref_cev)
        hm, hr = _mse_relmse(pred_heston, ref_heston)
        rows_price.append({
            "model": name,
            "bsm_mse": bm, "bsm_relmse": br,
            "cev_mse": cm, "cev_relmse": cr,
            "heston_mse": hm, "heston_relmse": hr,
        })
        print(f"  {name:20s}  BSM RelMSE={br:.4f}  CEV RelMSE={cr:.4f}  "
              f"Heston RelMSE={hr:.4f}")

    def add_greeks(name, model_type, pred_g, ref_g):
        dm = _mae([g["delta"] for g in pred_g], [g["delta"] for g in ref_g])
        gm = _mae([g["gamma"] for g in pred_g], [g["gamma"] for g in ref_g])
        vm = _mae([g["vega"]  for g in pred_g], [g["vega"]  for g in ref_g])
        rows_greeks.append({
            "model": name, "model_type": model_type,
            "delta_mae": dm, "gamma_mae": gm, "vega_mae": vm,
        })

    # ── 独立 PINN ──
    print("加载独立 PINN...")
    bsm_pinn = load_indep_bsm()
    cev_pinn = load_indep_cev()
    heston_pinn = load_indep_heston()

    pred_bsm_i    = [bsm_pinn.price(S) / bsm_pinn.K    for S in S_GRID]
    pred_cev_i    = [cev_pinn.price(S)                  for S in S_GRID]
    pred_heston_i = [heston_pinn.price(S)               for S in S_GRID]
    # 独立 PINN 用各自训练参数的参考解评估（验证拟合能力）
    bm, br = _mse_relmse(pred_bsm_i, ref_bsm)
    cm, cr = _mse_relmse(pred_cev_i, ref_cev_i)
    hm, hr = _mse_relmse(pred_heston_i, ref_heston_i)
    rows_price.append({
        "model": "indep",
        "bsm_mse": bm, "bsm_relmse": br,
        "cev_mse": cm, "cev_relmse": cr,
        "heston_mse": hm, "heston_relmse": hr,
    })
    print(f"  {'indep':20s}  BSM RelMSE={br:.4f}  CEV RelMSE={cr:.4f}  "
          f"Heston RelMSE={hr:.4f}")

    # ── Unified v2 ──
    print("加载 Unified v2...")
    from unified_pinn_v2 import ModelParams
    unified = load_unified("unified_v16_gl.pt")
    p_bsm    = ModelParams.from_bsm(sigma=BSM_PARAMS["sigma"])
    p_cev    = ModelParams.from_cev(sigma=CEV_PARAMS["sigma"], beta=CEV_PARAMS["beta"])
    p_heston = ModelParams.from_heston(**HESTON_UNIFIED)

    pred_bsm_u    = [unified.price(p_bsm,    S) for S in S_GRID]
    pred_cev_u    = [unified.price(p_cev,    S) for S in S_GRID]
    pred_heston_u = [unified.price(p_heston, S) for S in S_GRID]
    add_price("unified_v2", pred_bsm_u, pred_cev_u, pred_heston_u, ref_heston_u)

    greeks_bsm_u    = unified_greeks_autograd(unified, p_bsm,    S_GRID)
    greeks_heston_u = unified_greeks_autograd(unified, p_heston, S_GRID)
    add_greeks("unified_v2", "BSM",    greeks_bsm_u,    ref_bsm_g)
    add_greeks("unified_v2", "Heston", greeks_heston_u, ref_heston_g)

    # ── Unified v2 fine-tuned（如果存在）──
    ft_ckpt = os.path.join(BASE, "results/unified_v2_ft.pt")
    if os.path.exists(ft_ckpt):
        print("加载 Unified v2 fine-tuned...")
        unified_ft = load_unified("unified_v2_ft.pt")
        pred_bsm_ft    = [unified_ft.price(p_bsm,    S) for S in S_GRID]
        pred_cev_ft    = [unified_ft.price(p_cev,    S) for S in S_GRID]
        pred_heston_ft = [unified_ft.price(p_heston, S) for S in S_GRID]
        add_price("unified_v2_ft", pred_bsm_ft, pred_cev_ft, pred_heston_ft, ref_heston_u)
        greeks_bsm_ft    = unified_greeks_autograd(unified_ft, p_bsm,    S_GRID)
        greeks_heston_ft = unified_greeks_autograd(unified_ft, p_heston, S_GRID)
        add_greeks("unified_v2_ft", "BSM",    greeks_bsm_ft,    ref_bsm_g)
        add_greeks("unified_v2_ft", "Heston", greeks_heston_ft, ref_heston_g)

    # ── 参数化 PINN ──
    print("加载参数化 PINN...")
    param_pinn = load_parametric()
    pred_bsm_p = [param_pinn.price(S=S, K=K_SYN, T=T_SYN, r=r_SYN,
                                    sigma=BSM_PARAMS["sigma"], beta=1.0)
                  for S in S_GRID]
    pred_cev_p = [param_pinn.price(S=S, K=K_SYN, T=T_SYN, r=r_SYN,
                                    sigma=CEV_PARAMS["sigma"],
                                    beta=CEV_PARAMS["beta"])
                  for S in S_GRID]
    pred_heston_p = [param_pinn.price(S=S, K=K_SYN, T=T_SYN, r=r_SYN,
                                       kappa=HESTON_UNIFIED["kappa"],
                                       theta=HESTON_UNIFIED["theta"],
                                       xi=HESTON_UNIFIED["xi"],
                                       rho=HESTON_UNIFIED["rho"])
                     for S in S_GRID]
    add_price("parametric", pred_bsm_p, pred_cev_p, pred_heston_p, ref_heston_u)

    # 保存
    out_dir = os.path.join(BASE, "results")
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(rows_price).to_csv(
        os.path.join(out_dir, "eval_synthetic.csv"), index=False)
    pd.DataFrame(rows_greeks).to_csv(
        os.path.join(out_dir, "eval_greeks.csv"), index=False)
    print(f"\n合成数据结果 → {os.path.join(out_dir, 'eval_synthetic.csv')}")
    print(f"Greeks 结果  → {os.path.join(out_dir, 'eval_greeks.csv')}")


import re
import datetime

# ── 真实市场数据评估 ──────────────────────────────────────────────────────────

def _load_spy(csv_path, moneyness_lo=0.7, moneyness_hi=1.3):
    """解析 CBOE 原始格式 CSV（前3行为元数据，第4行为列名，数据从第5行开始）。"""
    with open(csv_path, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    # 从第2行提取当前股价
    m = re.search(r"Last:\s*([\d.]+)", lines[1] if len(lines) > 1 else lines[0])
    S_spot = float(m.group(1)) if m else None
    # 解析数据行（跳过前4行：2行元数据+1空行+1列名行）
    rows = []
    today = datetime.date.today()
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


def _bsm_iv(S, K, tau, r, mid):
    """BSM 隐含波动率（二分法）。"""
    intrinsic = max(S - K * np.exp(-r * tau), 0.0)
    if mid <= intrinsic + 1e-6:
        return 0.2
    try:
        return brentq(lambda sig: bsm_call(S, K, tau, r, sig) - mid,
                      1e-4, 5.0, xtol=1e-6, maxiter=100)
    except Exception:
        return 0.2


def eval_market():
    print("=== 真实市场数据评估 ===")
    spy_csv = os.path.join(BASE, "data/spy_quotedata.csv")
    df = _load_spy(spy_csv)
    print(f"有效合约数: {len(df)}")

    # 所有 PINN 模型用 moneyness 映射到训练域（K_SYN=100）
    K_SYN = 100.0
    df["S_scaled"] = df["S"] / df["K"] * K_SYN   # moneyness * 100
    df["V_scaled"] = df["mid"] / df["K"] * K_SYN  # 价格按比例缩放

    df["iv"] = df.apply(
        lambda row: _bsm_iv(row["S"], row["K"], row["tau"], 0.05, row["mid"]),
        axis=1)

    h = HESTON_UNIFIED
    rows = []

    def score(name, pred_fn):
        preds = []
        for _, row in df.iterrows():
            try:
                p = float(pred_fn(row))
            except Exception:
                p = float("nan")
            preds.append(p)
        preds = np.array(preds)
        ref   = df["mid"].values
        mask  = ~np.isnan(preds)
        mse, relmse = _mse_relmse(preds[mask], ref[mask])
        rows.append({"model": name, "mse": round(mse, 6),
                     "relmse": round(relmse, 6), "n": int(mask.sum())})
        print(f"  {name:20s}  MSE={mse:.4f}  RelMSE={relmse:.4f}  n={mask.sum()}")

    # 解析解基准（直接用原始价格）
    score("bsm_analytical",
          lambda row: bsm_call(row["S"], row["K"], row["tau"], 0.05, row["iv"]))
    score("heston_analytical",
          lambda row: heston_call(row["S"], row["K"], row["tau"], 0.05, **h))

    # 独立 PINN（moneyness 映射 + 反映射）
    bsm_pinn    = load_indep_bsm()
    cev_pinn    = load_indep_cev()
    heston_pinn = load_indep_heston()
    score("indep_bsm",
          lambda row: bsm_pinn.price(row["S_scaled"]) / bsm_pinn.K * row["K"])
    score("indep_cev",
          lambda row: cev_pinn.price(row["S_scaled"]) * row["K"] / K_SYN)
    score("indep_heston",
          lambda row: heston_pinn.price(row["S_scaled"]) * row["K"] / K_SYN)

    # Unified v2（moneyness 映射 + 反映射）
    from unified_pinn_v2 import ModelParams
    unified  = load_unified("unified_v16_gl.pt")
    p_heston = ModelParams.from_heston(**h)
    score("unified_v2",
          lambda row: unified.price(p_heston, row["S_scaled"]) * row["K"] / K_SYN)

    # Unified v2 fine-tuned
    ft_ckpt = os.path.join(BASE, "results/unified_v2_ft.pt")
    if os.path.exists(ft_ckpt):
        unified_ft = load_unified("unified_v2_ft.pt")
        score("unified_v2_ft",
              lambda row: unified_ft.price(p_heston, row["S_scaled"]) * row["K"] / K_SYN)

    # 参数化 PINN（moneyness 映射 + 反映射）
    param_pinn = load_parametric()
    score("parametric",
          lambda row: param_pinn.price(
              S=row["S_scaled"], K=K_SYN, T=row["tau"], r=0.05,
              kappa=h["kappa"], theta=h["theta"], xi=h["xi"], rho=h["rho"]
          ) * row["K"] / K_SYN)

    out_path = os.path.join(BASE, "results/eval_market.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\n市场数据结果 → {out_path}")


# ── 入口 ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["synthetic", "market", "all"],
                        default="all")
    args = parser.parse_args()
    if args.mode in ("synthetic", "all"):
        eval_synthetic()
    if args.mode in ("market", "all"):
        eval_market()
