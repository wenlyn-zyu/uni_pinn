"""
train_parametric.py

Train the fully-parameterized PINN so it can price BSM, CEV, and Heston options
for ANY parameter combination, without re-training.

Usage:
  # 1. Generate reference data (once):
  python generate_ref_data.py --out results/ref_data.pkl

  # 2. Train:
  python train_parametric.py --ref results/ref_data.pkl --epochs 50000 \
                              --out results/fully_param.pt

  # 3. Continue training from checkpoint:
  python train_parametric.py --ref results/ref_data.pkl --ckpt results/fully_param.pt \
                              --epochs 20000 --lr 3e-4 --out results/fully_param_v2.pt
"""

import argparse
import os
import sys
import pickle
import json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from fully_parametric_pinn import FullyParametricPINN


def load_ref_data(path: str, max_heston_configs: int = None):
    """Load pre-generated reference data.

    Args:
        path:              path to .pkl file
        max_heston_configs: cap Heston anchors to limit GPU memory
    """
    print(f"Loading reference data from {path}...")
    with open(path, "rb") as f:
        anchors = pickle.load(f)

    # Separate by model type for optional filtering
    bsm_anchors, cev_anchors, heston_anchors = [], [], []
    for a in anchors:
        lam_arr = a[5]
        xi   = lam_arr[0, 4]
        beta = lam_arr[0, 1]
        if xi == 0 and beta == 1.0:
            bsm_anchors.append(a)
        elif xi == 0:
            cev_anchors.append(a)
        else:
            heston_anchors.append(a)

    total_before = sum(len(a[0]) for a in anchors)
    print(f"  BSM:    {len(bsm_anchors)} configs, {sum(len(a[0]) for a in bsm_anchors)} points")
    print(f"  CEV:    {len(cev_anchors)} configs, {sum(len(a[0]) for a in cev_anchors)} points")
    print(f"  Heston: {len(heston_anchors)} configs, {sum(len(a[0]) for a in heston_anchors)} points")

    if max_heston_configs and len(heston_anchors) > max_heston_configs:
        import random
        random.seed(42)
        heston_anchors = random.sample(heston_anchors, max_heston_configs)
        print(f"  Heston capped to {max_heston_configs} configs "
              f"({sum(len(a[0]) for a in heston_anchors)} points)")

    filtered = bsm_anchors + cev_anchors + heston_anchors
    total_after = sum(len(a[0]) for a in filtered)
    print(f"  Total: {total_after} anchor points from {len(filtered)} configs\n")
    return filtered


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", type=str, default="results/ref_data.pkl",
                        help="Path to pre-generated reference data (.pkl)")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Resume from checkpoint")
    parser.add_argument("--epochs", type=int, default=50000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--out", type=str, default="results/fully_param.pt")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-every", type=int, default=5000,
                        help="Save checkpoint every N epochs (0=disable)")
    parser.add_argument("--max-heston-configs", type=int, default=None,
                        help="Cap number of Heston anchor configs")
    parser.add_argument("--w-data", type=float, default=100.0)
    parser.add_argument("--w-pde", type=float, default=1.0)
    parser.add_argument("--w-bc", type=float, default=10.0)
    parser.add_argument("--w-ic", type=float, default=10.0)
    parser.add_argument("--w-bsm-raw", type=float, default=1.0)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Load reference data
    ref_data = load_ref_data(args.ref, args.max_heston_configs)

    # Initialize model
    print("Initializing FullyParametricPINN...")
    model = FullyParametricPINN(
        hidden=args.hidden,
        depth=args.depth,
        lr=args.lr,
        ref_data=ref_data,
        device=device,
    )

    if args.ckpt:
        model.load(args.ckpt)
        print(f"Resumed from {args.ckpt}")

    n_params = sum(p.numel() for p in model.net.parameters())
    print(f"  Network: {args.hidden} hidden × {args.depth} layers = {n_params:,} params\n")

    # Train
    print(f"Training {args.epochs} epochs, lr={args.lr}...")
    history = model.train(
        epochs=args.epochs,
        w_pde=args.w_pde,
        w_bc=args.w_bc,
        w_ic=args.w_ic,
        w_data=args.w_data,
        w_bsm_raw=args.w_bsm_raw,
        log_every=500,
        save_every=args.save_every,
        save_path=args.out,
    )

    # Save final model
    model.save(args.out)
    print(f"\nModel saved: {args.out}")

    # Save training history
    hist_out = args.out.replace(".pt", "_history.json")
    with open(hist_out, "w") as f:
        json.dump(history, f, indent=2)
    print(f"History saved: {hist_out}")


if __name__ == "__main__":
    main()
