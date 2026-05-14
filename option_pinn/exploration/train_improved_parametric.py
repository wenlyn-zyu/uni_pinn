# option_pinn/exploration/train_improved_parametric.py
"""Train improved parametric PINN v2.

Usage:
  python train_improved_parametric.py --epochs 80000 --out results/imp_param_v2.pt
"""

import argparse
import os
import sys
import json

sys.path.insert(0, os.path.dirname(__file__))
from improved_parametric_pinn import (
    ImprovedParametricPINN, generate_ref_anchors,
)

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=80000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--out", type=str, default="results/imp_param_v2.pt")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--save-every", type=int, default=10000)
    # Data anchor counts
    parser.add_argument("--n-bsm-anchors", type=int, default=10)
    parser.add_argument("--n-cev-anchors", type=int, default=16)
    parser.add_argument("--n-heston-anchors", type=int, default=36)
    parser.add_argument("--n-S-per-config", type=int, default=60)
    # Loss weights
    parser.add_argument("--w-pde", type=float, default=1.0)
    parser.add_argument("--w-bc", type=float, default=10.0)
    parser.add_argument("--w-ic", type=float, default=10.0)
    parser.add_argument("--w-data", type=float, default=500.0)
    parser.add_argument("--w-bsm-raw", type=float, default=5.0)
    args = parser.parse_args()

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Generate reference anchors
    print("Generating reference anchor data...")
    ref_data = generate_ref_anchors(
        n_bsm=args.n_bsm_anchors,
        n_cev=args.n_cev_anchors,
        n_heston=args.n_heston_anchors,
        n_S_per_config=args.n_S_per_config,
    )

    # Initialize model
    print("\nInitializing ImprovedParametricPINN...")
    model = ImprovedParametricPINN(
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
    print(f"  Network: {args.hidden} hidden × {args.depth} layers = {n_params:,} params")

    # Train
    print(f"\nTraining {args.epochs} epochs, lr={args.lr}...")
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

    # Save
    model.save(args.out)
    print(f"\nModel saved: {args.out}")

    hist_out = args.out.replace(".pt", "_history.json")
    with open(hist_out, "w") as f:
        json.dump(history, f, indent=2)
    print(f"History saved: {hist_out}")


if __name__ == "__main__":
    main()
