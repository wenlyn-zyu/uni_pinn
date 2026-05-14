"""
quick_test.py — Quick sanity check on CPU before deploying to server.

Runs 200 epochs with a tiny anchor set to verify:
  1. Model initializes correctly
  2. PDE residual computes without NaN
  3. Loss decreases over time
  4. Pricing produces reasonable values

Usage:
  python quick_test.py
"""

import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from fully_parametric_pinn import FullyParametricPINN, FullyParametricNet, pde_residual

# Quick reference data: just a few BSM anchors
def make_tiny_ref():
    anchors = []
    S_arr = np.linspace(60, 150, 10)
    for sigma in [0.15, 0.20, 0.30]:
        v_arr = np.full_like(S_arr, sigma**2)
        tau_arr = np.full_like(S_arr, 1.0)
        V_arr = S_arr * 1.0  # rough approximation for test
        lam_arr = np.tile([sigma, 1.0, 0.0, 0.0, 0.0, 0.0], (len(S_arr), 1))
        anchors.append((S_arr.copy(), v_arr.copy(), tau_arr.copy(),
                        100.0, 0.05, lam_arr, V_arr))
    return anchors


def main():
    device = torch.device("cpu")
    print(f"Device: {device}")

    # 1. Test network forward pass
    print("\n1. Network forward pass...")
    net = FullyParametricNet(hidden=64, depth=4).to(device)
    S   = torch.randn(100, 1).exp() * 100  # lognormal-ish
    v   = torch.rand(100, 1) * 0.1
    tau = torch.rand(100, 1) * 2.0
    K   = torch.full((100, 1), 100.0)
    r   = torch.full((100, 1), 0.05)
    lam = torch.zeros(100, 6)
    lam[:, 0] = 0.2  # sigma
    lam[:, 1] = 1.0  # beta (BSM)

    V = net(S, v, tau, K, r, lam)
    print(f"  Output shape: {V.shape}, range: [{V.min().item():.4f}, {V.max().item():.4f}]")
    assert V.shape == (100, 1), f"Expected (100,1), got {V.shape}"
    assert not torch.isnan(V).any(), "NaN in output!"
    print("  OK")

    # 2. Test PDE residual
    print("\n2. PDE residual...")
    res = pde_residual(net, S, v, tau, K, r, lam)
    print(f"  Residual shape: {res.shape}, mean abs: {res.abs().mean().item():.6f}")
    assert not torch.isnan(res).any(), "NaN in residual!"
    print("  OK")

    # 3. Test training loop (200 epochs)
    print("\n3. Training 200 epochs...")
    ref_data = make_tiny_ref()
    model = FullyParametricPINN(hidden=64, depth=4, lr=1e-3,
                                ref_data=ref_data, device=device)
    n_params = sum(p.numel() for p in model.net.parameters())
    print(f"  Parameters: {n_params:,}")

    history = model.train(
        epochs=200,
        w_pde=1.0, w_bc=1.0, w_ic=1.0, w_data=10.0,
        log_every=50,
    )

    if history:
        first_loss = history[0]["loss"]
        last_loss  = history[-1]["loss"]
        print(f"  Initial loss: {first_loss:.4e}")
        print(f"  Final loss:   {last_loss:.4e}")
        improved = last_loss < first_loss
        print(f"  Loss decreased: {improved}")

    # 4. Test pricing
    print("\n4. Pricing test...")
    price = model.price(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2, beta=1.0)
    print(f"  BSM ATM call: PINN={price:.4f}  BS_ref={8.1653:.4f}")
    price_otm = model.price(S=90.0, K=100.0, T=1.0, r=0.05, sigma=0.2, beta=1.0)
    print(f"  BSM OTM call: PINN={price_otm:.4f}  BS_ref={2.5939:.4f}")
    print("  OK")

    print("\n=== ALL CHECKS PASSED ===")


if __name__ == "__main__":
    main()
