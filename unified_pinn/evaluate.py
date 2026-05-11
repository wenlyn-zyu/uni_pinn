"""
evaluate.py — 评估统一PINN在三个模型上的精度

对比基准：
  BSM    -> Black-Scholes 解析解
  CEV    -> Crank-Nicolson 有限差分
  Heston -> 特征函数半解析解

用法：
  python evaluate.py [--ckpt results/unified.pt] [--out results/]
"""

import argparse
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from unified_pinn import ModelParams, UnifiedPINN

# 从原始仓库借用参考解
sys.path.insert(0, os.path.expanduser("~/zhuwl2022/ppin/src/models"))
try:
    from bsm_pinn import bs_call_price
    from heston_pinn import heston_call_price
    HAS_REF = True
except ImportError:
    HAS_REF = False
    print("[警告] 无法导入参考解，仅输出PINN预测值")


def cn_cev_price(S_arr, K, T, r, sigma, beta, N_S=200, N_t=500):
    """Crank-Nicolson 有限差分求解 CEV 欧式看涨期权。"""
    S_max = 3 * K
    dS = S_max / N_S
    dt = T / N_t
    S = np.linspace(0, S_max, N_S + 1)

    V = np.maximum(S - K, 0.0)

    alpha = 0.5 * dt * (sigma**2 * S**(2*beta) / dS**2 - r * S / dS)
    beta_c = 1 + dt * (sigma**2 * S**(2*beta) / dS**2 + r)
    gamma = 0.5 * dt * (sigma**2 * S**(2*beta) / dS**2 + r * S / dS)

    for _ in range(N_t):
        rhs = alpha[1:-1] * V[:-2] + (2 - beta_c[1:-1]) * V[1:-1] + gamma[1:-1] * V[2:]
        A = np.diag(beta_c[1:-1]) - np.diag(alpha[2:-1], -1) - np.diag(gamma[1:-2], 1)
        bc_lo = 0.0
        bc_hi = S_max - K * np.exp(-r * dt)
        rhs[0]  += alpha[1] * bc_lo
        rhs[-1] += gamma[-2] * bc_hi
        V[1:-1] = np.linalg.solve(A, rhs)
        V[0]  = bc_lo
        V[-1] = bc_hi

    return np.interp(S_arr, S, V)


def metrics(pred, ref, K=100.):
    err = np.abs(pred - ref)
    # RelMAE 只在参考价格 > 0.5 的点上计算，避免深度 OTM 的 ref≈0 导致除零爆炸
    mask = ref > 0.5
    rel = float(np.mean(err[mask] / ref[mask])) if mask.any() else float("nan")
    return {
        "MAE":    float(np.mean(err)),
        "RMSE":   float(np.sqrt(np.mean(err**2))),
        "MaxErr": float(np.max(err)),
        "RelMAE": rel,
    }


def evaluate_bsm(model, p, S_grid):
    pred = np.array([model.price(p, S) for S in S_grid])
    if HAS_REF:
        ref = np.array([bs_call_price(S, p.K, p.T, p.r, p.sigma) for S in S_grid])
        m = metrics(pred, ref)
        print(f"  BSM  MAE={m['MAE']:.4f}  RMSE={m['RMSE']:.4f}  "
              f"MaxErr={m['MaxErr']:.4f}  RelMAE={m['RelMAE']:.4f}")
        return pred, ref, m
    return pred, None, {}


def evaluate_cev(model, p, S_grid):
    pred = np.array([model.price(p, S) for S in S_grid])
    ref  = cn_cev_price(S_grid, p.K, p.T, p.r, p.sigma, p.beta)
    m = metrics(pred, ref)
    print(f"  CEV  MAE={m['MAE']:.4f}  RMSE={m['RMSE']:.4f}  "
          f"MaxErr={m['MaxErr']:.4f}  RelMAE={m['RelMAE']:.4f}")
    return pred, ref, m


def evaluate_heston(model, p, S_grid):
    pred = np.array([model.price(p, S, v=p.v0) for S in S_grid])
    if HAS_REF:
        ref = np.array([
            heston_call_price(S, p.K, p.T, p.r,
                              p.kappa, p.theta, p.xi, p.rho, p.v0)
            for S in S_grid
        ])
        m = metrics(pred, ref)
        print(f"  Heston MAE={m['MAE']:.4f}  RMSE={m['RMSE']:.4f}  "
              f"MaxErr={m['MaxErr']:.4f}  RelMAE={m['RelMAE']:.4f}")
        return pred, ref, m
    return pred, None, {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="results/unified.pt")
    parser.add_argument("--out",  type=str, default="results/")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    p_bsm    = ModelParams.from_bsm(K=100., T=1., r=0.05, sigma=0.2)
    p_cev    = ModelParams.from_cev(K=100., T=1., r=0.05, sigma=0.2, beta=0.5)
    p_heston = ModelParams.from_heston(K=100., T=1., r=0.05,
                                       kappa=2.0, theta=0.04,
                                       xi=0.3, rho=-0.7, v0=0.04)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = UnifiedPINN([p_bsm, p_cev, p_heston], device=device)
    model.load(args.ckpt)
    print(f"已加载模型: {args.ckpt}\n")

    S_grid = np.linspace(60, 160, 50)

    print("=== 评估结果 ===")
    pred_bsm,    ref_bsm,    m_bsm    = evaluate_bsm(model, p_bsm, S_grid)
    pred_cev,    ref_cev,    m_cev    = evaluate_cev(model, p_cev, S_grid)
    pred_heston, ref_heston, m_heston = evaluate_heston(model, p_heston, S_grid)

    # 绘图
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        titles = ["BSM", "CEV (beta=0.5)", "Heston"]
        preds  = [pred_bsm, pred_cev, pred_heston]
        refs   = [ref_bsm,  ref_cev,  ref_heston]

        for ax, title, pred, ref in zip(axes, titles, preds, refs):
            ax.plot(S_grid, pred, label="UnifiedPINN", lw=2)
            if ref is not None:
                ax.plot(S_grid, ref, "--", label="Reference", lw=1.5)
            ax.set_title(title)
            ax.set_xlabel("S")
            ax.set_ylabel("V")
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path = os.path.join(args.out, "unified_eval.pdf")
        plt.savefig(fig_path)
        print(f"\n图表已保存至 {fig_path}")
    except Exception as e:
        print(f"[绘图跳过] {e}")


if __name__ == "__main__":
    main()
