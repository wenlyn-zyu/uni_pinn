"""
train.py — 训练统一PINN

用法：
  python train.py [--epochs 30000] [--n_per_model 5000] [--out results/unified.pt]
"""

import argparse
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from unified_pinn import ModelParams, UnifiedPINN


def build_ref_data(param_list):
    """
    预计算所有模型的参考解，作为数据驱动锚点。
    只在 t=0 处计算，S 从 60 到 160 取 20 个点。
    """
    import torch
    from unified_pinn import _bs_call
    from evaluate import cn_cev_price
    ref_data = {}
    S_grid = np.linspace(60., 160., 20)

    for idx, p in enumerate(param_list):
        if p.xi == 0. and p.beta == 1.0:
            # BSM：用 BS 解析解（零成本）
            S_t = torch.tensor(S_grid, dtype=torch.float32).reshape(-1, 1)
            tau = torch.full_like(S_t, p.T)
            sig = torch.full_like(S_t, p.sigma)
            V_ref = _bs_call(S_t, p.K, tau, p.r, sig).numpy().flatten()
            v_arr = np.full_like(S_grid, p.v0)
            t_arr = np.zeros_like(S_grid)
            ref_data[idx] = (S_grid, v_arr, t_arr, V_ref)
            print(f"  BSM 参考解: {len(S_grid)} 点, "
                  f"V范围=[{V_ref.min():.2f}, {V_ref.max():.2f}]")
        elif p.xi == 0. and p.beta != 1.0:
            # CEV 模型：用 CN 有限差分
            V_ref = cn_cev_price(S_grid, p.K, p.T, p.r, p.sigma, p.beta)
            v_arr = np.full_like(S_grid, p.v0)
            t_arr = np.zeros_like(S_grid)
            ref_data[idx] = (S_grid, v_arr, t_arr, V_ref)
            print(f"  CEV(beta={p.beta}) 参考解: {len(S_grid)} 点, "
                  f"V范围=[{V_ref.min():.2f}, {V_ref.max():.2f}]")
        elif p.xi > 0.:
            # Heston 模型：尝试用半解析解
            try:
                sys.path.insert(0, os.path.expanduser("~/zhuwl2022/ppin/src/models"))
                from heston_pinn import heston_call_price
                V_ref = np.array([
                    heston_call_price(S, p.K, p.T, p.r,
                                      p.kappa, p.theta, p.xi, p.rho, p.v0)
                    for S in S_grid
                ])
                v_arr = np.full_like(S_grid, p.v0)
                t_arr = np.zeros_like(S_grid)
                ref_data[idx] = (S_grid, v_arr, t_arr, V_ref)
                print(f"  Heston 参考解: {len(S_grid)} 点, "
                      f"V范围=[{V_ref.min():.2f}, {V_ref.max():.2f}]")
            except ImportError:
                print("  Heston 参考解不可用，跳过")
    return ref_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",       type=int,   default=30000)
    parser.add_argument("--n_per_model",  type=int,   default=5000)
    parser.add_argument("--hidden",       type=int,   default=128)
    parser.add_argument("--depth",        type=int,   default=6)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--w_pde",        type=float, default=1.0)
    parser.add_argument("--w_bc",         type=float, default=50.0)
    parser.add_argument("--w_data",       type=float, default=100.0)
    parser.add_argument("--warmstart",    type=str,   default=None,
                        help="从已有 checkpoint 热启动")
    parser.add_argument("--out",          type=str,   default="results/unified.pt")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # 多参数变体训练：BSM + 4个CEV(beta) + Heston
    param_list = [
        ModelParams.from_bsm(K=100., T=1., r=0.05, sigma=0.2),
        ModelParams.from_cev(K=100., T=1., r=0.05, sigma=0.2, beta=0.3),
        ModelParams.from_cev(K=100., T=1., r=0.05, sigma=0.2, beta=0.5),
        ModelParams.from_cev(K=100., T=1., r=0.05, sigma=0.2, beta=0.7),
        ModelParams.from_cev(K=100., T=1., r=0.05, sigma=0.2, beta=0.9),
        ModelParams.from_heston(K=100., T=1., r=0.05,
                                kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, v0=0.04),
    ]

    print("设备检测...")
    import torch
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"  使用设备: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    print("\n预计算参考解（数据驱动锚点）...")
    ref_data = build_ref_data(param_list)
    print(f"  共 {len(ref_data)} 个模型有参考解\n")

    print(f"开始训练 UnifiedPINN（{len(param_list)} 个模型混合）")
    print(f"  epochs={args.epochs}, n_per_model={args.n_per_model}")
    print(f"  hidden={args.hidden}, depth={args.depth}")
    print(f"  w_pde={args.w_pde}, w_bc={args.w_bc}, w_data={args.w_data}\n")

    model = UnifiedPINN(param_list,
                        hidden=args.hidden,
                        depth=args.depth,
                        lr=args.lr,
                        ref_data=ref_data,
                        device=device)

    if args.warmstart:
        model.load(args.warmstart)
        print(f"热启动自: {args.warmstart}")

    history = model.train(
        epochs=args.epochs,
        n_per_model=args.n_per_model,
        w_pde=args.w_pde,
        w_bc=args.w_bc,
        w_data=args.w_data,
    )

    model.save(args.out)
    print(f"\n模型已保存至 {args.out}")

    log_path = args.out.replace(".pt", "_history.json")
    with open(log_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"训练曲线已保存至 {log_path}")


if __name__ == "__main__":
    main()
