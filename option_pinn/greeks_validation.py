"""
greeks_validation.py
Validate PINN Greeks (Delta, Gamma, Vega) against analytical/FD references.

BSM reference: closed-form Greeks.
Heston reference: central finite differences on GL prices (no closed form).

Usage:
  python greeks_validation.py --ckpt results/unified_v15.pt \
                               --out  results/greeks_validation.csv
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import torch
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(__file__))
from unified_pinn_v2 import ModelParams, UnifiedPINN, UnifiedNet

# Heston GL pricer (copied from spy_backtest_v15.py to keep this script self-contained)
from numpy.polynomial.legendre import leggauss
_N_GL = 96
_GL_NODES, _GL_WEIGHTS = leggauss(_N_GL)
_GL_PHI_MAX = 100.0
_GL_PHI = (_GL_NODES + 1.0) * 0.5 * _GL_PHI_MAX
_GL_W   = _GL_WEIGHTS * 0.5 * _GL_PHI_MAX


def _heston_cf_batch(phi_arr, S, Ks, T, r, kappa, theta, xi, rho, v0, j):
    i = 1j
    u, b = (0.5, kappa - rho * xi) if j == 1 else (-0.5, kappa)
    a = kappa * theta
    x = np.log(S / Ks)
    phi = phi_arr[:, None]; x2d = x[None, :]
    d = np.sqrt((rho * xi * i * phi - b)**2 - xi**2 * (2 * u * i * phi - phi**2))
    num = b - rho * xi * i * phi + d
    g = num / (b - rho * xi * i * phi - d)
    exp_dT = np.exp(d * T)
    C = (r * i * phi * T
         + (a / xi**2) * (num * T - 2.0 * np.log((1.0 - g * exp_dT) / (1.0 - g))))
    D = (num / xi**2) * ((1.0 - exp_dT) / (1.0 - g * exp_dT))
    return np.exp(C + D * v0 + i * phi * x2d)


def heston_price_gl(S, K, T, r, kappa, theta, xi, rho, v0):
    if T < 1e-6:
        return max(S - K, 0.0)
    Ks = np.array([K])
    phi, w = _GL_PHI, _GL_W
    cf1 = _heston_cf_batch(phi, S, Ks, T, r, kappa, theta, xi, rho, v0, 1)
    cf2 = _heston_cf_batch(phi, S, Ks, T, r, kappa, theta, xi, rho, v0, 2)
    phi2d = phi[:, None]
    I1 = (w[:, None] * np.real(cf1 / (1j * phi2d))).sum(axis=0)
    I2 = (w[:, None] * np.real(cf2 / (1j * phi2d))).sum(axis=0)
    price = S * (0.5 + I1[0] / np.pi) - K * np.exp(-r * T) * (0.5 + I2[0] / np.pi)
    return float(max(price, 0.0))


# ---------------------------------------------------------------------------
# Analytical BSM Greeks
# ---------------------------------------------------------------------------

def bsm_greeks(S, K, T, r, sigma):
    """Returns (delta, gamma, vega) for a BSM call."""
    sqt = sigma * np.sqrt(T)
    d1  = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / sqt
    d2  = d1 - sqt
    delta = float(norm.cdf(d1))
    gamma = float(norm.pdf(d1) / (S * sqt))
    vega  = float(S * norm.pdf(d1) * np.sqrt(T))  # per unit sigma
    return delta, gamma, vega


# ---------------------------------------------------------------------------
# PINN Greeks via autograd
# ---------------------------------------------------------------------------

def pinn_greeks(net, p: ModelParams, S: float, v: float, device):
    """
    Compute Delta = dV/dS, Gamma = d²V/dS², Vega = dV/dv via autograd.
    Returns (delta, gamma, vega).
    Note: vega here is dV/dv (raw autograd on variance input).
    For BSM models, use pinn_bsm_vega() instead.
    """
    net.eval()
    S_t   = torch.tensor([[S]], dtype=torch.float32, device=device, requires_grad=True)
    v_t   = torch.tensor([[v]], dtype=torch.float32, device=device, requires_grad=True)
    t_t   = torch.tensor([[0.0]], dtype=torch.float32, device=device)
    lam_t = p.to_lambda_tensor(device)

    V = net(S_t / p.S_max, v_t / p.v_max, t_t / p.T,
            lam_t, S_t, t_t, p.K, p.T, p.r)

    V_S = torch.autograd.grad(V, S_t, create_graph=True, retain_graph=True)[0]
    V_SS = torch.autograd.grad(V_S, S_t, create_graph=False, retain_graph=True)[0]
    V_v  = torch.autograd.grad(V, v_t, create_graph=False)[0]

    delta = float(V_S.item())
    gamma = float(V_SS.item())
    vega  = float(V_v.item())
    return delta, gamma, vega


def pinn_bsm_vega(net, p: ModelParams, S: float, device, h=0.001):
    """
    Compute BSM Vega = dV/dsigma via central finite difference on sigma.
    This is the correct Vega for BSM models where v input is irrelevant.
    """
    from unified_pinn_v2 import ModelParams as MP
    net.eval()
    with torch.no_grad():
        S_t = torch.tensor([[S]], dtype=torch.float32, device=device)
        v_t = torch.tensor([[p.v0]], dtype=torch.float32, device=device)
        t_t = torch.tensor([[0.0]], dtype=torch.float32, device=device)

        p_up = MP(**{**p.__dict__, "sigma": p.sigma + h})
        p_dn = MP(**{**p.__dict__, "sigma": p.sigma - h})

        V_up = net(S_t / p.S_max, v_t / p.v_max, t_t / p.T,
                   p_up.to_lambda_tensor(device), S_t, t_t, p.K, p.T, p.r)
        V_dn = net(S_t / p.S_max, v_t / p.v_max, t_t / p.T,
                   p_dn.to_lambda_tensor(device), S_t, t_t, p.K, p.T, p.r)

    return float((V_up.item() - V_dn.item()) / (2 * h))


# ---------------------------------------------------------------------------
# Heston FD Greeks (central differences)
# ---------------------------------------------------------------------------

def heston_fd_greeks(S, K, T, r, kappa, theta, xi, rho, v0, dS_frac=0.001, dv_frac=0.01):
    """Central FD Delta, Gamma, Vega for Heston call."""
    dS = S * dS_frac
    dv = max(v0 * dv_frac, 1e-5)

    Vp  = heston_price_gl(S + dS, K, T, r, kappa, theta, xi, rho, v0)
    Vm  = heston_price_gl(S - dS, K, T, r, kappa, theta, xi, rho, v0)
    V0  = heston_price_gl(S,      K, T, r, kappa, theta, xi, rho, v0)
    Vvp = heston_price_gl(S,      K, T, r, kappa, theta, xi, rho, v0 + dv)
    Vvm = heston_price_gl(S,      K, T, r, kappa, theta, xi, rho, v0 - dv)

    delta = (Vp - Vm) / (2 * dS)
    gamma = (Vp - 2 * V0 + Vm) / (dS ** 2)
    vega  = (Vvp - Vvm) / (2 * dv)
    return delta, gamma, vega


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="results/unified_v15.pt")
    parser.add_argument("--out",  default="results/greeks_validation.csv")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build model (same param_list as training)
    param_list = []
    for sigma in [0.13, 0.15, 0.17, 0.20, 0.25, 0.30]:
        param_list.append(ModelParams.from_bsm(K=100., T=1., r=0.05, sigma=sigma))
    for beta in [0.1, 0.3, 0.5, 0.7, 0.9]:
        for sigma in [0.15, 0.20]:
            param_list.append(ModelParams.from_cev(K=100., T=1., r=0.05, sigma=sigma, beta=beta))
    for kappa in [0.5, 2.0, 5.0, 8.0]:
        for xi in [0.1, 0.3, 0.5]:
            for rho in [-0.9, -0.7, -0.5]:
                param_list.append(ModelParams.from_heston(
                    K=100., T=1., r=0.05, kappa=kappa, theta=0.04,
                    xi=xi, rho=rho, v0=0.04))

    model = UnifiedPINN(param_list, hidden=128, depth=6, device=device)
    model.load(args.ckpt)
    net = model.net
    print(f"Loaded: {args.ckpt}\n")

    rows = []

    # ------------------------------------------------------------------
    # BSM Greeks validation
    # ------------------------------------------------------------------
    print("=== BSM Greeks ===")
    r = 0.05
    for sigma in [0.13, 0.17, 0.20, 0.25]:
        for moneyness, S in [("ITM", 110.), ("ATM", 100.), ("OTM", 90.)]:
            K, T = 100., 1.0
            v = sigma ** 2
            p = ModelParams.from_bsm(K=K, T=T, r=r, sigma=sigma)

            d_ref, g_ref, vg_ref = bsm_greeks(S, K, T, r, sigma)
            d_pinn, g_pinn, _ = pinn_greeks(net, p, S, v, device)
            vg_pinn_conv = pinn_bsm_vega(net, p, S, device)  # FD Vega on sigma

            rows.append({
                "model": "BSM", "sigma": sigma, "moneyness": moneyness,
                "S": S, "K": K, "T": T,
                "Delta_ref":  round(d_ref,  5), "Delta_PINN":  round(d_pinn,  5),
                "Delta_err%": round(abs(d_pinn - d_ref) / (abs(d_ref) + 1e-8) * 100, 3),
                "Gamma_ref":  round(g_ref,  6), "Gamma_PINN":  round(g_pinn,  6),
                "Gamma_err%": round(abs(g_pinn - g_ref) / (abs(g_ref) + 1e-8) * 100, 3),
                "Vega_ref":   round(vg_ref, 4), "Vega_PINN":   round(vg_pinn_conv, 4),
                "Vega_err%":  round(abs(vg_pinn_conv - vg_ref) / (abs(vg_ref) + 1e-8) * 100, 3),
            })
            print(f"  σ={sigma:.2f} {moneyness:3s}  "
                  f"Δ: ref={d_ref:.4f} pinn={d_pinn:.4f} err={rows[-1]['Delta_err%']:.2f}%  "
                  f"Γ: ref={g_ref:.5f} pinn={g_pinn:.5f} err={rows[-1]['Gamma_err%']:.2f}%  "
                  f"V: ref={vg_ref:.3f} pinn={vg_pinn_conv:.3f} err={rows[-1]['Vega_err%']:.2f}%")

    # ------------------------------------------------------------------
    # Heston Greeks validation
    # ------------------------------------------------------------------
    print("\n=== Heston Greeks ===")
    heston_cases = [
        dict(kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, v0=0.04, label="base"),
        dict(kappa=0.5, theta=0.04, xi=0.1, rho=-0.9, v0=0.04, label="low-vol-of-vol"),
        dict(kappa=5.0, theta=0.04, xi=0.5, rho=-0.5, v0=0.04, label="high-meanrev"),
        dict(kappa=8.0, theta=0.04, xi=0.5, rho=-0.9, v0=0.04, label="high-kappa-rho"),
    ]
    for hc in heston_cases:
        kappa, theta, xi, rho, v0 = hc["kappa"], hc["theta"], hc["xi"], hc["rho"], hc["v0"]
        for moneyness, S in [("ITM", 110.), ("ATM", 100.), ("OTM", 90.)]:
            K, T = 100., 1.0
            p = ModelParams.from_heston(K=K, T=T, r=r,
                                        kappa=kappa, theta=theta, xi=xi, rho=rho, v0=v0)

            d_ref, g_ref, vg_ref = heston_fd_greeks(S, K, T, r, kappa, theta, xi, rho, v0)
            d_pinn, g_pinn, vg_pinn = pinn_greeks(net, p, S, v0, device)

            rows.append({
                "model": f"Heston({hc['label']})", "sigma": xi, "moneyness": moneyness,
                "S": S, "K": K, "T": T,
                "Delta_ref":  round(d_ref,  5), "Delta_PINN":  round(d_pinn,  5),
                "Delta_err%": round(abs(d_pinn - d_ref) / (abs(d_ref) + 1e-8) * 100, 3),
                "Gamma_ref":  round(g_ref,  6), "Gamma_PINN":  round(g_pinn,  6),
                "Gamma_err%": round(abs(g_pinn - g_ref) / (abs(g_ref) + 1e-8) * 100, 3),
                "Vega_ref":   round(vg_ref, 4), "Vega_PINN":   round(vg_pinn, 4),
                "Vega_err%":  round(abs(vg_pinn - vg_ref) / (abs(vg_ref) + 1e-8) * 100, 3),
            })
            print(f"  {hc['label']:20s} {moneyness:3s}  "
                  f"Δ: ref={d_ref:.4f} pinn={d_pinn:.4f} err={rows[-1]['Delta_err%']:.2f}%  "
                  f"Γ: ref={g_ref:.5f} pinn={g_pinn:.5f} err={rows[-1]['Gamma_err%']:.2f}%")

    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else ".", exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"\nSaved to {args.out}")

    # Summary
    bsm_rows = df[df["model"] == "BSM"]
    hes_rows = df[df["model"].str.startswith("Heston")]
    print(f"\nBSM   avg Delta err: {bsm_rows['Delta_err%'].mean():.2f}%  "
          f"Gamma err: {bsm_rows['Gamma_err%'].mean():.2f}%  "
          f"Vega err: {bsm_rows['Vega_err%'].mean():.2f}%")
    print(f"Heston avg Delta err: {hes_rows['Delta_err%'].mean():.2f}%  "
          f"Gamma err: {hes_rows['Gamma_err%'].mean():.2f}%")


if __name__ == "__main__":
    main()
