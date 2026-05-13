"""
Train Heston_PINN with unified evaluation parameters (kappa=2.0, xi=0.3, rho=-0.7, r=0.05).
Uses data anchors from GL semi-analytical solution to ensure convergence.
"""
import sys
sys.path.insert(0, "/home/yz2026/zhuwl2022/unified_pinn")
from independent.heston_pinn import Heston_PINN

print("Training Heston_PINN: r=0.05, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7")
m = Heston_PINN(K=100.0, T=1.0, r=0.05,
                kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, v0=0.04,
                S_max=400.0, v_max=1.0)
m.train(epochs=30000, pretrain_epochs=5000, log_every=5000, w_data=100.0)
m.save("/home/yz2026/zhuwl2022/unified_pinn/results/indep_heston_unified.pt")
print("Saved: results/indep_heston_unified.pt")
