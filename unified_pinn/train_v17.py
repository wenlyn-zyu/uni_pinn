"""
train_v17.py -- parametric PINN training (v17)

v17: continuous Heston parameter sampling instead of fixed discrete grid.
  - BSM: sigma in {0.13, 0.15, 0.17, 0.20, 0.25, 0.30}  (6 fixed)
  - CEV: beta in {0.1, 0.3, 0.5, 0.7, 0.9} x sigma in {0.15, 0.20}  (10 fixed)
  - Heston: 32 parameter sets sampled per step from log-uniform distributions
            kappa ~ LogU(0.05, 20), theta ~ LogU(0.005, 0.25),
            xi ~ LogU(0.01, 1.5), rho ~ U(-0.98, -0.01), v0 ~ LogU(0.005, 0.25)

Usage:
  python train_v17.py [--epochs 30000] [--out results/unified_v17.pt]
"""

import argparse
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from unified_pinn_v2 import ModelParams, _bs_call
from unified_pinn_v3 import ParametricPINN


def build_bsm_cev_params():
    params = []
    for sigma in [0.13, 0.15, 0.17, 0.20, 0.25, 0.30]:
        params.append(ModelParams.from_bsm(K=100., T=1., r=0.05, sigma=sigma))
    for beta in [0.1, 0.3, 0.5, 0.7, 0.9]:
        for sigma in [0.15, 0.20]:
            params.append(ModelParams.from_cev(
                K=100., T=1., r=0.05, sigma=sigma, beta=beta))
    return params


def build_bsm_cev_ref_data(param_list):
    """Pre-compute BSM/CEV reference solutions for data anchors."""
    import torch
    from evaluate import cn_cev_price

    ref_data = {}
    S_grid = np.linspace(60., 160., 20)

    for idx, p in enumerate(param_list):
        if p.xi == 0. and p.beta == 1.0:
            S_t   = torch.tensor(S_grid, dtype=torch.float32).reshape(-1, 1)
            tau   = torch.full_like(S_t, p.T)
            sig   = torch.full_like(S_t, p.sigma)
            V_ref = _bs_call(S_t, p.K, tau, p.r, sig).numpy().flatten()
            v_arr = np.full_like(S_grid, p.v0)
            t_arr = np.zeros_like(S_grid)
            ref_data[idx] = (S_grid, v_arr, t_arr, V_ref)
        elif p.xi == 0. and p.beta != 1.0:
            V_ref = cn_cev_price(S_grid, p.K, p.T, p.r, p.sigma, p.beta)
            v_arr = np.full_like(S_grid, p.v0)
            t_arr = np.zeros_like(S_grid)
            ref_data[idx] = (S_grid, v_arr, t_arr, V_ref)

    print(f"  {len(ref_data)}/{len(param_list)} BSM/CEV models have reference solutions")
    return ref_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",           type=int,   default=30000)
    parser.add_argument("--n_per_bc",         type=int,   default=512,
                        help="collocation points per BSM/CEV model per step")
    parser.add_argument("--n_per_heston",     type=int,   default=256,
                        help="collocation points per sampled Heston set per step")
    parser.add_argument("--n_heston_per_step",type=int,   default=32,
                        help="number of Heston parameter sets sampled per step")
    parser.add_argument("--n_anchor",         type=int,   default=10,
                        help="S grid points for dynamic Heston data anchors")
    parser.add_argument("--hidden",           type=int,   default=128)
    parser.add_argument("--depth",            type=int,   default=6)
    parser.add_argument("--lr",               type=float, default=1e-3)
    parser.add_argument("--w_pde",            type=float, default=1.0)
    parser.add_argument("--w_bc",             type=float, default=10.0)
    parser.add_argument("--w_data",           type=float, default=100.0)
    parser.add_argument("--out",              type=str,   default="results/unified_v17.pt")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    import torch
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    bsm_cev_params = build_bsm_cev_params()
    bsm_n = sum(1 for p in bsm_cev_params if p.beta == 1.0)
    cev_n = sum(1 for p in bsm_cev_params if p.beta != 1.0)
    print(f"BSM/CEV fixed variants: {len(bsm_cev_params)} (BSM={bsm_n}, CEV={cev_n})")
    print(f"Heston per step: {args.n_heston_per_step} sampled sets "
          f"x {args.n_per_heston} colloc pts = "
          f"{args.n_heston_per_step * args.n_per_heston} pts/step")

    print("\nPre-computing BSM/CEV reference solutions...")
    ref_data = build_bsm_cev_ref_data(bsm_cev_params)

    print(f"\nBuilding ParametricPINN v17")
    model = ParametricPINN(
        bsm_cev_params=bsm_cev_params,
        n_heston_per_step=args.n_heston_per_step,
        n_anchor_per_heston=args.n_anchor,
        hidden=args.hidden,
        depth=args.depth,
        lr=args.lr,
        device=device,
    )

    # Inject pre-computed BSM/CEV anchors into parent's data cache
    import torch
    S_list, v_list, t_list, V_list, lam_list = [], [], [], [], []
    for idx, (S_arr, v_arr, t_arr, V_arr) in ref_data.items():
        p   = bsm_cev_params[idx]
        n   = len(S_arr)
        lam = p.to_lambda_tensor(device).expand(n, -1)
        S_list.append(torch.tensor(S_arr, dtype=torch.float32, device=device).reshape(-1, 1))
        v_list.append(torch.tensor(v_arr, dtype=torch.float32, device=device).reshape(-1, 1))
        t_list.append(torch.tensor(t_arr, dtype=torch.float32, device=device).reshape(-1, 1))
        V_list.append(torch.tensor(V_arr, dtype=torch.float32, device=device).reshape(-1, 1))
        lam_list.append(lam)
    if S_list:
        model._data_cache = (
            torch.cat(S_list), torch.cat(v_list),
            torch.cat(t_list), torch.cat(V_list),
            torch.cat(lam_list)
        )
        print(f"  Cached {sum(len(S_list[i]) for i in range(len(S_list)))} BSM/CEV anchor points")

    print(f"\nTraining ParametricPINN v17")
    print(f"  epochs={args.epochs}, n_per_bc={args.n_per_bc}, "
          f"n_per_heston={args.n_per_heston}")
    print(f"  hidden={args.hidden}, depth={args.depth}")
    print(f"  w_pde={args.w_pde}, w_bc={args.w_bc}, w_data={args.w_data}\n")

    history = model.train_parametric(
        epochs=args.epochs,
        n_per_bc=args.n_per_bc,
        n_per_heston=args.n_per_heston,
        w_pde=args.w_pde,
        w_bc=args.w_bc,
        w_data=args.w_data,
    )

    model.save(args.out)
    print(f"\nModel saved to {args.out}")

    log_path = args.out.replace(".pt", "_history.json")
    with open(log_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to {log_path}")

    txt_log = args.out.replace(".pt", ".log")
    with open(txt_log, "w") as f:
        for h in history:
            f.write(f"epoch={h['epoch']} loss={h['loss']:.4e} "
                    f"pde={h['pde']:.4e} bc={h['bc']:.4e} "
                    f"data={h['data']:.4e}\n")
    print(f"Text log saved to {txt_log}")


if __name__ == "__main__":
    main()
