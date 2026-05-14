"""
Train independent PINN models (BSM, CEV) and save checkpoints.

Usage:
  python independent/train_independent.py              # train all
  python independent/train_independent.py --model bsm  # train BSM only
  python independent/train_independent.py --model cev

Note: The Heston independent baseline (Hainaut & Casas 2024) is trained
separately via its own script (heston_hainaut.py).

Checkpoints are saved to results/ directory.
"""

import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
if _root not in sys.path:
    sys.path.insert(0, _root)

from independent.bsm_pinn import BSM_PINN
from independent.cev_pinn import CEV_PINN

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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train independent PINN models")
    parser.add_argument("--model", choices=["bsm", "cev", "all"],
                        default="all", help="Which model to train")
    args = parser.parse_args()

    if args.model in ("bsm", "all"):
        train_bsm()
    if args.model in ("cev", "all"):
        train_cev()
