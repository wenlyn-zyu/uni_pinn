"""
train_v16_cont.py — Continue training v16 checkpoint for better Heston convergence.

Loads v16_gl checkpoint and trains additional 20000 epochs with:
- Higher Heston data weight (200 vs 100)
- Lower learning rate (3e-4 vs 1e-3) for fine convergence
- Same param list (52 variants — proven to work)

Usage:
  python train_v16_cont.py --ckpt results/unified_v16_gl.pt --out results/unified_v16_cont.pt
"""

import argparse, os, sys, numpy as np, torch
from scipy.stats import norm
from numpy.polynomial.legendre import leggauss

sys.path.insert(0, os.path.dirname(__file__))
from unified_pinn_v2 import ModelParams, UnifiedPINN, cev_schroder_call

_N_GL = 96
_GL_NODES, _GL_WEIGHTS = leggauss(_N_GL)
_GL_PHI_MAX = 100.0
_GL_PHI = (_GL_NODES + 1.0) * 0.5 * _GL_PHI_MAX
_GL_W   = _GL_WEIGHTS * 0.5 * _GL_PHI_MAX


def _heston_cf_batch(phi_arr, S, Ks, T, r, kappa, theta, xi, rho, v0, j):
    i = 1j
    u, b = (0.5, kappa - rho * xi) if j == 1 else (-0.5, kappa)
    a = kappa * theta; x = np.log(S / Ks)
    phi = phi_arr[:, None]; x2d = x[None, :]
    d_sqrt = np.sqrt((rho * xi * i * phi - b)**2 - xi**2 * (2 * u * i * phi - phi**2))
    num = b - rho * xi * i * phi + d_sqrt
    g = num / (b - rho * xi * i * phi - d_sqrt)
    exp_dT = np.exp(d_sqrt * T)
    C = (r * i * phi * T + (a / xi**2) * (num * T - 2.0 * np.log((1.0 - g * exp_dT) / (1.0 - g))))
    D = (num / xi**2) * ((1.0 - exp_dT) / (1.0 - g * exp_dT))
    return np.exp(C + D * v0 + i * phi * x2d)


def heston_price_gl(S, K, T, r, kappa, theta, xi, rho, v0):
    if T < 1e-6: return max(S - K, 0.0)
    Ks = np.array([K]); phi, w = _GL_PHI, _GL_W
    cf1 = _heston_cf_batch(phi, S, Ks, T, r, kappa, theta, xi, rho, v0, 1)
    cf2 = _heston_cf_batch(phi, S, Ks, T, r, kappa, theta, xi, rho, v0, 2)
    phi2d = phi[:, None]
    I1 = (w[:, None] * np.real(cf1 / (1j * phi2d))).sum(axis=0)
    I2 = (w[:, None] * np.real(cf2 / (1j * phi2d))).sum(axis=0)
    return float(max(S * (0.5 + I1[0] / np.pi) - K * np.exp(-r * T) * (0.5 + I2[0] / np.pi), 0.0))


def bsm_price(S, K, T, r, sigma):
    sqt = sigma * np.sqrt(max(T, 1e-10))
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / sqt
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d1 - sqt)


def build_param_list():
    params = []
    for sigma in [0.13, 0.15, 0.17, 0.20, 0.25, 0.30]:
        params.append(ModelParams.from_bsm(K=100., T=1., r=0.05, sigma=sigma))
    for beta in [0.1, 0.3, 0.5, 0.7, 0.9]:
        for sigma in [0.15, 0.20]:
            params.append(ModelParams.from_cev(K=100., T=1., r=0.05, sigma=sigma, beta=beta))
    for kappa in [0.5, 2.0, 5.0, 8.0]:
        for xi in [0.1, 0.3, 0.5]:
            for rho in [-0.9, -0.7, -0.5]:
                params.append(ModelParams.from_heston(
                    K=100., T=1., r=0.05, kappa=kappa, theta=0.04, xi=xi, rho=rho, v0=0.04))
    return params


def build_ref_data(param_list, n_bsm=120, n_cev=200, n_heston_s=100, n_heston_v=8):
    ref_data = {}
    for idx, p in enumerate(param_list):
        if p.beta == 1.0 and p.xi == 0:
            S_arr = np.linspace(max(p.K*0.5, 1.0), p.S_max*0.98, n_bsm)
            ref_data[idx] = (S_arr, np.full_like(S_arr, p.v0), np.zeros_like(S_arr),
                             np.array([bsm_price(s, p.K, p.T, p.r, p.sigma) for s in S_arr]))
        elif p.xi == 0:
            S_arr = np.linspace(max(p.K*0.3, 1.0), p.S_max*0.98, n_cev)
            ref_data[idx] = (S_arr, np.full_like(S_arr, p.v0), np.zeros_like(S_arr),
                             np.array([cev_schroder_call(s, p.K, p.T, p.r, p.sigma, p.beta) for s in S_arr]))
        else:
            S_pos = np.linspace(max(p.K*0.3, 1.0), p.S_max*0.98, n_heston_s)
            v_pts = np.logspace(-3, np.log10(p.v_max*0.95), n_heston_v)
            Sg, vg = np.meshgrid(S_pos, v_pts)
            S_arr = Sg.ravel(); v_arr = vg.ravel()
            t_arr = np.zeros_like(S_arr)
            V_arr = np.array([heston_price_gl(s, p.K, p.T, p.r, p.kappa, p.theta, p.xi, p.rho,
                                              p.v0 if v < 1e-6 else v) for s, v in zip(S_arr, v_arr)])
            t_mid = 0.5 * p.T
            S_mid = np.linspace(max(p.K*0.5, 1.0), p.S_max*0.95, 50)
            v_mid = np.logspace(-3, np.log10(p.v_max*0.95), 4)
            Sg, vg = np.meshgrid(S_mid, v_mid)
            V_mid = np.array([heston_price_gl(s, p.K, p.T-t_mid, p.r, p.kappa, p.theta, p.xi, p.rho,
                                              p.v0 if v < 1e-6 else v) for s, v in zip(Sg.ravel(), vg.ravel())])
            S_arr = np.concatenate([S_arr, Sg.ravel()])
            v_arr = np.concatenate([v_arr, vg.ravel()])
            t_arr = np.concatenate([t_arr, np.full_like(Sg.ravel(), t_mid)])
            V_arr = np.concatenate([V_arr, V_mid])
            ref_data[idx] = (S_arr, v_arr, t_arr, V_arr)
    return ref_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="results/unified_v16_gl.pt")
    parser.add_argument("--epochs", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--out", type=str, default="results/unified_v16_cont.pt")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-every", type=int, default=5000)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    param_list = build_param_list()
    print(f"Total: {len(param_list)} variants (BSM:6 CEV:10 Heston:36)")

    print("Building data anchors...")
    ref_data = build_ref_data(param_list)
    print(f"  Total anchors: {sum(len(arr) for arr,_,_,_ in ref_data.values())}")

    model = UnifiedPINN(param_list, hidden=128, depth=6, lr=args.lr,
                        ref_data=ref_data, device=device)
    model.load(args.ckpt)
    print(f"Loaded {args.ckpt} — continuing training")

    print(f"\nContinued training: {args.epochs} epochs, lr={args.lr}")
    history = model.train(
        epochs=args.epochs, n_per_model=5000,
        w_pde=1.0, w_bc=10.0, w_ic=10.0, w_data=200.0,
        log_every=500,
        save_every=args.save_every,
        save_path=args.out,
    )

    model.save(args.out)
    print(f"\nSaved: {args.out}")

    import json
    with open(args.out.replace(".pt", "_history.json"), "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
