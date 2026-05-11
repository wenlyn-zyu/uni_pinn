"""
train.py -- unified PINN training script

v15: expanded parameter ranges to cover real market calibrated values
  - BSM: sigma in {0.13, 0.15, 0.17, 0.20, 0.25, 0.30}
  - CEV: beta in {0.1, 0.3, 0.5, 0.7, 0.9}, sigma in {0.15, 0.20}
  - Heston: kappa in {0.5, 2.0, 5.0, 8.0}, xi in {0.1, 0.3, 0.5},
            rho in {-0.9, -0.7, -0.5}
  Total: 6 + 10 + 36 = 52 parameter variants

Usage:
  python train.py [--epochs 30000] [--n_per_model 512] [--out results/unified_v15.pt]
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
    Pre-compute reference solutions as data-driven anchors.
    Evaluated at t=0, S in [60, 160] with 20 points.
    """
    import torch
    from unified_pinn import _bs_call
    from evaluate import cn_cev_price

    sys.path.insert(0, os.path.expanduser("~/zhuwl2022/ppin/src/models"))
    try:
        from heston_pinn import heston_call_price
        HAS_HESTON = True
    except ImportError:
        HAS_HESTON = False
        print("  [warn] heston_call_price unavailable, skipping Heston anchors")

    ref_data = {}
    S_grid = np.linspace(60., 160., 20)

    for idx, p in enumerate(param_list):
        if p.xi == 0. and p.beta == 1.0:
            # BSM analytical
            S_t = torch.tensor(S_grid, dtype=torch.float32).reshape(-1, 1)
            tau = torch.full_like(S_t, p.T)
            sig = torch.full_like(S_t, p.sigma)
            V_ref = _bs_call(S_t, p.K, tau, p.r, sig).numpy().flatten()
            v_arr = np.full_like(S_grid, p.v0)
            t_arr = np.zeros_like(S_grid)
            ref_data[idx] = (S_grid, v_arr, t_arr, V_ref)

        elif p.xi == 0. and p.beta != 1.0:
            # CEV Crank-Nicolson
            V_ref = cn_cev_price(S_grid, p.K, p.T, p.r, p.sigma, p.beta)
            v_arr = np.full_like(S_grid, p.v0)
            t_arr = np.zeros_like(S_grid)
            ref_data[idx] = (S_grid, v_arr, t_arr, V_ref)

        elif p.xi > 0. and HAS_HESTON:
            # Heston semi-analytical
            try:
                V_ref = np.array([
                    heston_call_price(S, p.K, p.T, p.r,
                                      p.kappa, p.theta, p.xi, p.rho, p.v0)
                    for S in S_grid
                ])
                v_arr = np.full_like(S_grid, p.v0)
                t_arr = np.zeros_like(S_grid)
                ref_data[idx] = (S_grid, v_arr, t_arr, V_ref)
            except Exception as e:
                print(f"  [warn] Heston ref failed idx={idx}: {e}")

    print(f"  {len(ref_data)}/{len(param_list)} models have reference solutions")
    return ref_data


def build_param_list():
    """
    Build training parameter list covering real market calibrated ranges.

    Market calibration observations (SPY 2026-05-11):
      BSM  sigma* in [0.13, 0.17]
      CEV  beta*  in [0.01, 1.0] (hits boundaries), sigma* in [0.13, 0.17]
      Heston kappa in [0.05, 8.0], xi in [0.01, 0.5], rho in [-0.98, 0]
    """
    params = []

    # BSM: 6 sigma variants covering market range and beyond
    for sigma in [0.13, 0.15, 0.17, 0.20, 0.25, 0.30]:
        params.append(ModelParams.from_bsm(K=100., T=1., r=0.05, sigma=sigma))

    # CEV: beta x sigma grid -- beta=0.1 added for near-boundary market cases
    for beta in [0.1, 0.3, 0.5, 0.7, 0.9]:
        for sigma in [0.15, 0.20]:
            params.append(ModelParams.from_cev(
                K=100., T=1., r=0.05, sigma=sigma, beta=beta))

    # Heston: kappa x xi x rho grid
    # kappa up to 8 (market shows kappa=8 for Sep/Dec 2026)
    # xi down to 0.1 (market shows xi near 0.01 lower bound)
    # rho down to -0.9 (market shows rho=-0.98 for several expiries)
    for kappa in [0.5, 2.0, 5.0, 8.0]:
        for xi in [0.1, 0.3, 0.5]:
            for rho in [-0.9, -0.7, -0.5]:
                params.append(ModelParams.from_heston(
                    K=100., T=1., r=0.05,
                    kappa=kappa, theta=0.04, xi=xi, rho=rho, v0=0.04))

    bsm_n = sum(1 for p in params if p.xi == 0 and p.beta == 1.0)
    cev_n = sum(1 for p in params if p.xi == 0 and p.beta != 1.0)
    hes_n = sum(1 for p in params if p.xi > 0)
    print(f"Parameter list: {len(params)} variants  "
          f"(BSM={bsm_n}, CEV={cev_n}, Heston={hes_n})")
    return params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",      type=int,   default=30000)
    parser.add_argument("--n_per_model", type=int,   default=512,
                        help="collocation points per model per step")
    parser.add_argument("--hidden",      type=int,   default=128)
    parser.add_argument("--depth",       type=int,   default=6)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--w_pde",       type=float, default=1.0)
    parser.add_argument("--w_bc",        type=float, default=10.0)
    parser.add_argument("--w_data",      type=float, default=100.0)
    parser.add_argument("--warmstart",   type=str,   default=None,
                        help="warm-start from existing checkpoint")
    parser.add_argument("--out",         type=str,   default="results/unified_v15.pt")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    import torch
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    param_list = build_param_list()

    print("\nPre-computing reference solutions...")
    ref_data = build_ref_data(param_list)

    print(f"\nTraining UnifiedPINN v15")
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
        print(f"Warm-started from: {args.warmstart}")

    history = model.train(
        epochs=args.epochs,
        n_per_model=args.n_per_model,
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
