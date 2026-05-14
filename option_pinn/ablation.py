"""消融实验脚本：逐一去除统一框架的三个核心组件，评估对BSM和Heston MAE的影响。

三个消融变体：
  1. no_soft_mask  -- 硬切换（xi>0 用Heston系数，xi=0 用BSM/CEV系数）
  2. no_additive   -- 直接输出（网络直接预测期权价格，不加BS基线）
  3. no_anchor     -- 纯PDE约束（w_data=0，去掉数据锚点损失）

用法:
  python ablation.py                          # 训练全部三个变体
  python ablation.py --variant no_soft_mask   # 只训练某一个
  python ablation.py --epochs 10000           # 快速验证用较少步数
  python ablation.py --save results/ablation_results.json
"""

import argparse
import copy
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from unified_pinn_v2 import (
    UnifiedPINN, UnifiedNet, ModelParams,
    unified_pde_residual, _bs_call,
)
from ref_solvers import bsm_call, heston_call
from eval_all import S_GRID, K_SYN, T_SYN, r_SYN, _mae

# ---------------------------------------------------------------------------
# Ablation network variants
# ---------------------------------------------------------------------------

class HardSwitchNet(nn.Module):
    """Ablation: replace soft mask with hard switch (xi>0 -> Heston, else BSM/CEV)."""

    def __init__(self, hidden: int = 128, depth: int = 6):
        super().__init__()
        layers = [nn.Linear(9, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        last = nn.Linear(hidden, 1)
        nn.init.normal_(last.weight, std=0.001)
        nn.init.zeros_(last.bias)
        layers.append(last)
        self.net = nn.Sequential(*layers)

    def forward(self, S_n, v_n, t_n, lam, S_raw, t_raw, K, T, r, return_raw=False):
        sigma = lam[:, 0:1]
        beta  = lam[:, 1:2]
        xi    = lam[:, 4:5]
        # Hard switch: mask=1 if xi>0, mask=0 if xi==0
        mask = (xi.abs() > 1e-6).float()
        v_approx  = torch.clamp(v_n, min=1e-6)
        sigma_eff = (1 - mask) * sigma + mask * torch.sqrt(v_approx)
        tau   = torch.clamp(T - t_raw, min=1e-4)
        V_bs  = _bs_call(S_raw, K, tau, r, sigma_eff)
        x   = torch.cat([S_n, v_n, t_n, lam], dim=1)
        raw = self.net(x)
        V = torch.clamp(V_bs + K * raw, min=0.0)
        return (V, raw) if return_raw else V


class DirectOutputNet(nn.Module):
    """Ablation: direct output (no additive BS baseline, network predicts price directly)."""

    def __init__(self, hidden: int = 128, depth: int = 6):
        super().__init__()
        layers = [nn.Linear(9, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        last = nn.Linear(hidden, 1)
        nn.init.normal_(last.weight, std=0.001)
        nn.init.zeros_(last.bias)
        layers.append(last)
        self.net = nn.Sequential(*layers)

    def forward(self, S_n, v_n, t_n, lam, S_raw, t_raw, K, T, r, return_raw=False):
        x   = torch.cat([S_n, v_n, t_n, lam], dim=1)
        raw = self.net(x)
        # Scale output by K so network learns O(1) corrections
        V = torch.clamp(K * torch.sigmoid(raw) * 2, min=0.0)
        return (V, raw) if return_raw else V


# ---------------------------------------------------------------------------
# Build standard param list (same as llm_router / eval_all)
# ---------------------------------------------------------------------------

def _param_key(p) -> str:
    """Unique string key for a ModelParams instance."""
    return f"s{p.sigma:.3f}_b{p.beta:.2f}_k{p.kappa:.2f}_t{p.theta:.3f}_x{p.xi:.2f}_r{p.rho:.2f}"


def _model_type(p) -> str:
    if p.xi > 1e-6:
        return "heston"
    if abs(p.beta - 1.0) > 1e-6:
        return "cev"
    return "bsm"


def _build_param_list():
    params = []
    for sigma in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]:
        params.append(ModelParams.from_bsm(sigma=sigma))
    for sigma in [0.15, 0.20, 0.25]:
        for beta in [0.3, 0.5, 0.7, 0.9]:
            params.append(ModelParams.from_cev(sigma=sigma, beta=beta))
    for kappa in [1.0, 2.0, 3.0]:
        for theta in [0.02, 0.04, 0.06]:
            for xi in [0.2, 0.3, 0.4]:
                for rho in [-0.7, -0.5]:
                    params.append(ModelParams.from_heston(
                        kappa=kappa, theta=theta, xi=xi, rho=rho, v0=theta))
    return params


def build_ref_data(param_list):
    """Build ref_data dict keyed by integer index (as expected by UnifiedPINN)."""
    from ref_solvers import cev_call as _cev_call
    ref_data = {}
    S_anchor = np.linspace(60, 160, 20)
    v_anchor = np.full_like(S_anchor, 0.04)
    t_anchor = np.full_like(S_anchor, 0.5)
    for idx, p in enumerate(param_list):
        mt = _model_type(p)
        if mt == "bsm":
            vals = [bsm_call(S, p.K, p.T, p.r, sigma=p.sigma) for S in S_anchor]
        elif mt == "cev":
            vals = [_cev_call(S, p.K, p.T, p.r, sigma=p.sigma, beta=p.beta)
                    for S in S_anchor]
        else:
            vals = [heston_call(S, p.K, p.T, p.r,
                                kappa=p.kappa, theta=p.theta,
                                xi=p.xi, rho=p.rho, v0=p.v0)
                    for S in S_anchor]
        ref_data[idx] = (S_anchor, v_anchor, t_anchor, np.array(vals))
    return ref_data


# ---------------------------------------------------------------------------
# Evaluate a trained PINN on BSM and Heston MAE
# ---------------------------------------------------------------------------

def evaluate(pinn: UnifiedPINN) -> dict:
    bsm_ref  = np.array([bsm_call(S, K_SYN, T_SYN, r_SYN, sigma=0.20)
                         for S in S_GRID])
    heston_ref = np.array([heston_call(S, K_SYN, T_SYN, r_SYN,
                                       kappa=2.0, theta=0.04, xi=0.3,
                                       rho=-0.7, v0=0.04)
                           for S in S_GRID])

    p_bsm    = ModelParams.from_bsm(sigma=0.20)
    p_heston = ModelParams.from_heston(kappa=2.0, theta=0.04, xi=0.3,
                                       rho=-0.7, v0=0.04)

    pred_bsm    = np.array([pinn.price(p_bsm,    S=S) for S in S_GRID])
    pred_heston = np.array([pinn.price(p_heston,  S=S) for S in S_GRID])

    return {
        "bsm_mae":    round(float(_mae(pred_bsm,    bsm_ref)),    4),
        "heston_mae": round(float(_mae(pred_heston, heston_ref)), 4),
    }


# ---------------------------------------------------------------------------
# Train one ablation variant
# ---------------------------------------------------------------------------

def train_variant(variant: str, epochs: int, device: torch.device) -> dict:
    print(f"\n{'='*60}")
    print(f"  Ablation variant: {variant}  ({epochs} epochs)")
    print(f"{'='*60}")

    param_list = _build_param_list()
    ref_data = build_ref_data(param_list)

    pinn = UnifiedPINN(param_list, hidden=128, depth=6, device=device,
                       ref_data=ref_data)

    # Swap network for ablation variants
    if variant == "no_soft_mask":
        pinn.net = HardSwitchNet(hidden=128, depth=6).to(device)
        pinn.optimizer = torch.optim.Adam(pinn.net.parameters(), lr=1e-3)
        w_data = 100.0
    elif variant == "no_additive":
        pinn.net = DirectOutputNet(hidden=128, depth=6).to(device)
        pinn.optimizer = torch.optim.Adam(pinn.net.parameters(), lr=1e-3)
        w_data = 100.0
    elif variant == "no_anchor":
        w_data = 0.0   # keep standard net, just zero out anchor loss
    else:
        raise ValueError(f"Unknown variant: {variant}")

    pinn.train(epochs=epochs, w_pde=1.0, w_bc=10.0, w_ic=10.0, w_data=w_data,
               log_every=2000)

    metrics = evaluate(pinn)
    print(f"  BSM MAE={metrics['bsm_mae']:.4f}  Heston MAE={metrics['heston_mae']:.4f}")
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

VARIANTS = ["no_soft_mask", "no_additive", "no_anchor"]


def main():
    parser = argparse.ArgumentParser(description="Ablation study for unified PINN")
    parser.add_argument("--variant", choices=VARIANTS + ["all"], default="all",
                        help="Which ablation variant to run (default: all)")
    parser.add_argument("--epochs", type=int, default=30000,
                        help="Training epochs per variant (default: 30000)")
    parser.add_argument("--save", default="results/ablation_results.json",
                        help="Path to save JSON results")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    variants_to_run = VARIANTS if args.variant == "all" else [args.variant]
    results = {}

    for v in variants_to_run:
        results[v] = train_variant(v, epochs=args.epochs, device=device)

    print(f"\n{'='*60}")
    print("  Ablation Summary")
    print(f"{'='*60}")
    print(f"  {'Variant':<25} {'BSM MAE':>10} {'Heston MAE':>12}")
    print(f"  {'-'*50}")
    for v, m in results.items():
        print(f"  {v:<25} {m['bsm_mae']:>10.4f} {m['heston_mae']:>12.4f}")

    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    with open(args.save, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.save}")


if __name__ == "__main__":
    main()
