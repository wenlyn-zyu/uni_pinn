"""
Train independent PINN models (BSM, CEV, Heston) and save checkpoints.

Usage:
  python independent/train_independent.py              # train all
  python independent/train_independent.py --model bsm  # train BSM only
  python independent/train_independent.py --model cev
  python independent/train_independent.py --model heston

Checkpoints are saved to results/ directory.
"""

import os
import sys

# Allow running from project root or from independent/ directory
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
if _root not in sys.path:
    sys.path.insert(0, _root)

from independent.bsm_pinn import BSM_PINN
from independent.cev_pinn import CEV_PINN
from independent.heston_pinn import Heston_PINN

CKPT_DIR = os.path.join(_root, "results")
os.makedirs(CKPT_DIR, exist_ok=True)


def train_bsm():
    print("=" * 60)
    print("Training BSM-PINN (Dhiman & Hu 2023 gated network)")
    print("  sigma=0.20, K=100, T=1, r=0.05")
    print("=" * 60)
    model = BSM_PINN(K=100, T=1.0, r=0.05, sigma=0.2, S_max=300)
    model.train(epochs=30000, log_every=5000, w_pde=1.0, w_ic=10.0, w_bc=10.0)
    model.save(os.path.join(CKPT_DIR, "indep_bsm.pt"))
    print("BSM checkpoint saved to results/indep_bsm.pt\n")
    return model


def train_cev():
    print("=" * 60)
    print("Training CEV-PINN (beta=0.5, Schroder 1989 reference)")
    print("  sigma=0.20, beta=0.5, K=100, T=1, r=0.05")
    print("=" * 60)
    model = CEV_PINN(K=100, T=1.0, r=0.05, sigma=0.20, beta=0.5, S_max=300)
    model.train(epochs=30000, log_every=5000, w_pde=5.0, w_ic=10.0, w_bc=10.0)
    model.save(os.path.join(CKPT_DIR, "indep_cev.pt"))
    print("CEV checkpoint saved to results/indep_cev.pt\n")
    return model


def train_heston():
    print("=" * 60)
    print("Training Heston-PINN (with GL data anchors)")
    print("  kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, v0=0.04")
    print("  K=100, T=1, r=0.05")
    print("=" * 60)
    model = Heston_PINN(
        K=100, T=1.0, r=0.05,
        kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, v0=0.04,
        S_max=300, v_max=1.0,
    )
    model.train(epochs=30000, log_every=2000)
    model.save(os.path.join(CKPT_DIR, "indep_heston.pt"))
    print("Heston checkpoint saved to results/indep_heston.pt\n")
    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train independent PINN models")
    parser.add_argument("--model", choices=["bsm", "cev", "heston", "all"],
                        default="all", help="Which model to train")
    args = parser.parse_args()

    if args.model in ("bsm", "all"):
        train_bsm()
    if args.model in ("cev", "all"):
        train_cev()
    if args.model in ("heston", "all"):
        train_heston()
