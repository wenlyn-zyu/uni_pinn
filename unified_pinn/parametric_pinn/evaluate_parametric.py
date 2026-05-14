"""
evaluate_parametric.py

Evaluate the fully-parameterized PINN:
  1. Interpolation test: unseen parameter combinations within training range
  2. Extrapolation test: K, T, r values outside training grid
  3. Greeks validation via autograd
  4. Comparison with analytical/semi-analytical references

Usage:
  python evaluate_parametric.py --ckpt results/fully_param.pt --out results/
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import torch
from scipy.stats import norm
from numpy.polynomial.legendre import leggauss

sys.path.insert(0, os.path.dirname(__file__))
from fully_parametric_pinn import FullyParametricPINN, S_MAX, V_MAX

# ---------------------------------------------------------------------------
# Reference pricers
# ---------------------------------------------------------------------------

def bsm_price(S, K, T, r, sigma):
    if sigma <= 0 or T < 1e-8:
        return max(S - K * np.exp(-r * T), 0.0)
    sqt = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / sqt
    d2 = d1 - sqt
    return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))


def cev_schroder_call(S, K, T, r, sigma, beta):
    from scipy.stats import ncx2
    if abs(beta - 1.0) < 1e-9:
        return bsm_price(S, K, T, r, sigma)
    delta = 1.0 - beta
    nu    = 1.0 / delta
    lam   = (2.0 * r) / (sigma**2 * delta * (np.exp(2.0 * r * delta * T) - 1.0))
    x     = lam * S**(2.0 * delta) * np.exp(2.0 * r * delta * T)
    y     = lam * K**(2.0 * delta)
    d     = 2.0 + nu
    call  = (S * (1.0 - ncx2.cdf(y, df=d,   nc=x))
             - K * np.exp(-r * T) * ncx2.cdf(x, df=d - 2, nc=y))
    return float(max(call, max(S - K * np.exp(-r * T), 0.0)))


# GL Heston
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
    phi = phi_arr[:, None]
    x2d = x[None, :]
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
    try:
        Ks = np.array([K])
        phi, w = _GL_PHI, _GL_W
        cf1 = _heston_cf_batch(phi, S, Ks, T, r, kappa, theta, xi, rho, v0, 1)
        cf2 = _heston_cf_batch(phi, S, Ks, T, r, kappa, theta, xi, rho, v0, 2)
        phi2d = phi[:, None]
        I1 = (w[:, None] * np.real(cf1 / (1j * phi2d))).sum(axis=0)
        I2 = (w[:, None] * np.real(cf2 / (1j * phi2d))).sum(axis=0)
        price = S * (0.5 + I1[0] / np.pi) - K * np.exp(-r * T) * (0.5 + I2[0] / np.pi)
        return float(max(price, 0.0))
    except Exception:
        return np.nan


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def metrics(pred, ref):
    err = np.abs(pred - ref)
    mask = ref > 0.5
    rel = float(np.mean(err[mask] / ref[mask])) if mask.any() else float("nan")
    return {
        "MAE":    float(np.mean(err)),
        "RMSE":   float(np.sqrt(np.mean(err**2))),
        "MaxErr": float(np.max(err)),
        "RelMAE": rel,
    }


# ---------------------------------------------------------------------------
# PINN Greeks via autograd
# ---------------------------------------------------------------------------

def pinn_greeks(model, S, K, T, r, sigma, beta, kappa, theta, xi, rho, v0):
    """Compute Delta, Gamma, Vega via autograd."""
    net = model.net
    net.eval()
    device = model.device

    S_t   = torch.tensor([[S]], dtype=torch.float32, device=device, requires_grad=True)
    v_t   = torch.tensor([[v0]], dtype=torch.float32, device=device, requires_grad=True)
    tau_t = torch.tensor([[T]], dtype=torch.float32, device=device)
    K_t   = torch.tensor([[K]], dtype=torch.float32, device=device)
    r_t   = torch.tensor([[r]], dtype=torch.float32, device=device)
    lam_t = torch.tensor([[sigma, beta, kappa, theta, xi, rho]],
                         dtype=torch.float32, device=device)

    V = net(S_t, v_t, tau_t, K_t, r_t, lam_t)

    V_S  = torch.autograd.grad(V, S_t, create_graph=True, retain_graph=True)[0]
    V_SS = torch.autograd.grad(V_S, S_t, create_graph=False, retain_graph=True)[0]
    V_v  = torch.autograd.grad(V, v_t, create_graph=False)[0]

    return float(V_S.item()), float(V_SS.item()), float(V_v.item())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="results/fully_param.pt")
    parser.add_argument("--out",  type=str, default="results/")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Minimal param_list for model init (not used for pricing, just for constructor)
    from fully_parametric_pinn import FullyParametricPINN as M
    model = M(hidden=256, depth=8, device=device)
    model.load(args.ckpt)
    print(f"Loaded: {args.ckpt}\n")

    rows = []

    # =====================================================================
    # 1. Interpolation test: parameter combos within training grid
    # =====================================================================
    print("=" * 70)
    print("1. INTERPOLATION TEST (parameters within training grid)")
    print("=" * 70)

    # BSM interpolation
    print("\n--- BSM ---")
    K_ref, T_ref, r_ref = 100.0, 1.0, 0.05
    for sigma in [0.10, 0.18, 0.22, 0.35, 0.40]:  # some unseen sigmas
        for S in [80.0, 100.0, 120.0]:
            v0 = sigma**2
            pred = model.price(S, K_ref, T_ref, r_ref, sigma=sigma, beta=1.0)
            ref  = bsm_price(S, K_ref, T_ref, r_ref, sigma)
            err  = abs(pred - ref)
            rel  = err / max(ref, 1e-8) * 100
            print(f"  BSM(K={K_ref},T={T_ref},σ={sigma:.2f}) S={S:.0f}: "
                  f"pred={pred:.4f} ref={ref:.4f} err={err:.4f} rel={rel:.2f}%")
            rows.append({"test": "interp", "model": "BSM", "sigma": sigma,
                         "S": S, "K": K_ref, "T": T_ref, "r": r_ref,
                         "pred": pred, "ref": ref, "abs_err": err, "rel_err%": rel})

    # CEV interpolation
    print("\n--- CEV ---")
    for sigma, beta in [(0.15, 0.3), (0.20, 0.6), (0.18, 0.75)]:
        for S in [80.0, 100.0, 120.0]:
            pred = model.price(S, K_ref, T_ref, r_ref, sigma=sigma, beta=beta)
            ref  = cev_schroder_call(S, K_ref, T_ref, r_ref, sigma, beta)
            err  = abs(pred - ref)
            rel  = err / max(ref, 1e-8) * 100
            print(f"  CEV(σ={sigma:.2f},β={beta:.1f}) S={S:.0f}: "
                  f"pred={pred:.4f} ref={ref:.4f} err={err:.4f} rel={rel:.2f}%")
            rows.append({"test": "interp", "model": "CEV", "sigma": sigma,
                         "S": S, "K": K_ref, "T": T_ref, "r": r_ref,
                         "pred": pred, "ref": ref, "abs_err": err, "rel_err%": rel})

    # Heston interpolation
    print("\n--- Heston ---")
    heston_cases_int = [
        (1.0, 0.04, 0.20, -0.8, "interp1"),
        (3.0, 0.06, 0.40, -0.6, "interp2"),
        (6.0, 0.02, 0.15, -0.4, "interp3"),
    ]
    for kappa, theta, xi, rho, label in heston_cases_int:
        v0 = theta
        for S in [80.0, 100.0, 120.0]:
            pred = model.price(S, K_ref, T_ref, r_ref,
                              kappa=kappa, theta=theta, xi=xi, rho=rho, v0=v0)
            ref  = heston_price_gl(S, K_ref, T_ref, r_ref, kappa, theta, xi, rho, v0)
            err  = abs(pred - ref)
            rel  = err / max(ref, 1e-8) * 100 if not np.isnan(ref) else np.nan
            print(f"  Heston({label}) S={S:.0f}: pred={pred:.4f} ref={ref:.4f} "
                  f"err={err:.4f} rel={rel:.2f}%" if not np.isnan(ref)
                  else f"  Heston({label}) S={S:.0f}: [ref NaN]")
            rows.append({"test": "interp", "model": f"Heston({label})",
                         "sigma": xi, "S": S, "K": K_ref, "T": T_ref, "r": r_ref,
                         "pred": pred, "ref": ref, "abs_err": err, "rel_err%": rel})

    # =====================================================================
    # 2. Extrapolation: K, T, r outside training grid
    # =====================================================================
    print("\n" + "=" * 70)
    print("2. EXTRAPOLATION TEST (K, T, r beyond training grid)")
    print("=" * 70)

    # K extrapolation
    print("\n--- K extrapolation ---")
    for K in [40.0, 250.0]:
        for S in [K * 0.8, K, K * 1.2]:
            S_clip = min(S, 500.0)
            pred = model.price(S_clip, K, T_ref, r_ref, sigma=0.2, beta=1.0)
            ref  = bsm_price(S_clip, K, T_ref, r_ref, 0.2)
            err  = abs(pred - ref)
            rel  = err / max(ref, 1e-8) * 100
            print(f"  BSM(K={K:.0f}) S={S_clip:.0f}: pred={pred:.4f} ref={ref:.4f} "
                  f"err={err:.4f} rel={rel:.2f}%")
            rows.append({"test": "extra_K", "model": "BSM", "sigma": 0.2,
                         "S": S_clip, "K": K, "T": T_ref, "r": r_ref,
                         "pred": pred, "ref": ref, "abs_err": err, "rel_err%": rel})

    # T extrapolation
    print("\n--- T extrapolation ---")
    for T in [0.02, 3.5]:
        for S in [80.0, 100.0, 120.0]:
            pred = model.price(S, K_ref, T, r_ref, sigma=0.2, beta=1.0)
            ref  = bsm_price(S, K_ref, T, r_ref, 0.2)
            err  = abs(pred - ref)
            rel  = err / max(ref, 1e-8) * 100
            print(f"  BSM(T={T:.2f}) S={S:.0f}: pred={pred:.4f} ref={ref:.4f} "
                  f"err={err:.4f} rel={rel:.2f}%")
            rows.append({"test": "extra_T", "model": "BSM", "sigma": 0.2,
                         "S": S, "K": K_ref, "T": T, "r": r_ref,
                         "pred": pred, "ref": ref, "abs_err": err, "rel_err%": rel})

    # r extrapolation
    print("\n--- r extrapolation ---")
    for r in [0.001, 0.15]:
        for S in [80.0, 100.0, 120.0]:
            pred = model.price(S, K_ref, T_ref, r, sigma=0.2, beta=1.0)
            ref  = bsm_price(S, K_ref, T_ref, r, 0.2)
            err  = abs(pred - ref)
            rel  = err / max(ref, 1e-8) * 100
            print(f"  BSM(r={r:.3f}) S={S:.0f}: pred={pred:.4f} ref={ref:.4f} "
                  f"err={err:.4f} rel={rel:.2f}%")
            rows.append({"test": "extra_r", "model": "BSM", "sigma": 0.2,
                         "S": S, "K": K_ref, "T": T_ref, "r": r,
                         "pred": pred, "ref": ref, "abs_err": err, "rel_err%": rel})

    # Cross-model: Heston with unusual K, T
    print("\n--- Heston extrapolation ---")
    for K, T, label in [(70.0, 0.5, "lowK-shortT"), (150.0, 2.0, "highK-longT")]:
        kappa, theta, xi, rho = 2.0, 0.04, 0.3, -0.7
        v0 = theta
        for S in [K * 0.8, K, K * 1.2]:
            S_c = min(S, 500.0)
            pred = model.price(S_c, K, T, r_ref,
                              kappa=kappa, theta=theta, xi=xi, rho=rho, v0=v0)
            ref  = heston_price_gl(S_c, K, T, r_ref, kappa, theta, xi, rho, v0)
            if np.isnan(ref):
                continue
            err = abs(pred - ref)
            rel = err / max(ref, 1e-8) * 100
            print(f"  Heston({label}) S={S_c:.0f}: pred={pred:.4f} ref={ref:.4f} "
                  f"err={err:.4f} rel={rel:.2f}%")
            rows.append({"test": "extra_cross", "model": f"Heston({label})",
                         "sigma": xi, "S": S_c, "K": K, "T": T, "r": r_ref,
                         "pred": pred, "ref": ref, "abs_err": err, "rel_err%": rel})

    # =====================================================================
    # 3. Greeks validation
    # =====================================================================
    print("\n" + "=" * 70)
    print("3. GREEKS VALIDATION")
    print("=" * 70)

    # BSM Greeks (analytical reference)
    print("\n--- BSM Greeks ---")
    for sigma in [0.15, 0.20, 0.30]:
        for S in [90.0, 100.0, 110.0]:
            d_ref, g_ref, vg_ref = bsm_greeks(S, K_ref, T_ref, r_ref, sigma)
            d_pinn, g_pinn, vg_pinn = pinn_greeks(
                model, S, K_ref, T_ref, r_ref,
                sigma=sigma, beta=1.0, kappa=0., theta=0., xi=0., rho=0.,
                v0=sigma**2)
            print(f"  BSM(σ={sigma:.2f}) S={S:.0f}: "
                  f"Δ ref={d_ref:.4f} pinn={d_pinn:.4f} "
                  f"Γ ref={g_ref:.5f} pinn={g_pinn:.5f}")

    # Heston Greeks (FD reference)
    print("\n--- Heston Greeks (FD reference) ---")
    for kappa, theta, xi, rho, label in [
        (2.0, 0.04, 0.3, -0.7, "base"),
        (0.5, 0.04, 0.1, -0.9, "low-vol"),
    ]:
        v0 = theta
        for S in [90.0, 100.0, 110.0]:
            d_fd, g_fd, vg_fd = heston_fd_greeks(
                S, K_ref, T_ref, r_ref, kappa, theta, xi, rho, v0)
            d_pinn, g_pinn, vg_pinn = pinn_greeks(
                model, S, K_ref, T_ref, r_ref,
                sigma=0., beta=1.0, kappa=kappa, theta=theta, xi=xi, rho=rho, v0=v0)
            print(f"  Heston({label}) S={S:.0f}: "
                  f"Δ ref={d_fd:.4f} pinn={d_pinn:.4f} "
                  f"Γ ref={g_fd:.5f} pinn={g_pinn:.5f}")

    # =====================================================================
    # 4. Summary
    # =====================================================================
    df = pd.DataFrame(rows)
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for test_type in ["interp", "extra_K", "extra_T", "extra_r", "extra_cross"]:
        sub = df[df["test"] == test_type]
        if len(sub) == 0:
            continue
        valid = sub[sub["rel_err%"].notna()]
        print(f"  {test_type:15s}: avg RelErr={valid['rel_err%'].mean():.2f}%  "
              f"max RelErr={valid['rel_err%'].max():.2f}%  "
              f"MAE={valid['abs_err'].mean():.4f}  n={len(valid)}")

    out_csv = os.path.join(args.out, "eval_parametric.csv")
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\nResults saved to {out_csv}")


def bsm_greeks(S, K, T, r, sigma):
    sqt = sigma * np.sqrt(T)
    d1  = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / sqt
    d2  = d1 - sqt
    delta = float(norm.cdf(d1))
    gamma = float(norm.pdf(d1) / (S * sqt))
    vega  = float(S * norm.pdf(d1) * np.sqrt(T))
    return delta, gamma, vega


def heston_fd_greeks(S, K, T, r, kappa, theta, xi, rho, v0):
    dS = S * 0.001
    dv = max(v0 * 0.01, 1e-5)
    Vp  = heston_price_gl(S + dS, K, T, r, kappa, theta, xi, rho, v0)
    Vm  = heston_price_gl(S - dS, K, T, r, kappa, theta, xi, rho, v0)
    V0  = heston_price_gl(S,      K, T, r, kappa, theta, xi, rho, v0)
    Vvp = heston_price_gl(S,      K, T, r, kappa, theta, xi, rho, v0 + dv)
    Vvm = heston_price_gl(S,      K, T, r, kappa, theta, xi, rho, v0 - dv)
    delta = (Vp - Vm) / (2 * dS)
    gamma = (Vp - 2 * V0 + Vm) / (dS ** 2)
    vega  = (Vvp - Vvm) / (2 * dv)
    return delta, gamma, vega


if __name__ == "__main__":
    main()
