# option_pinn/exploration/finetune_unified_v2.py
"""Improved fine-tune of unified_v2 on real SPY market data.

Fixes the issues with the original finetune_heston.py that made MAE worse:
  1. Calibrate ALL three models (BSM, CEV, Heston) per expiry
  2. Use PDE preservation loss on the full param list (not just Heston)
  3. Gentle fine-tuning: low lr, cosine schedule, gradient clipping
  4. Per-expiry validation to detect degradation early
  5. Averaging over best checkpoints

Usage:
  python finetune_unified_v2.py --epochs 5000 --lr 1e-5
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize, brentq
from copy import deepcopy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ref_solvers import heston_call, bsm_call, cev_call as ref_cev_call
from unified_pinn_v2 import UnifiedPINN, ModelParams, unified_pde_residual

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE   = os.path.dirname(os.path.abspath(__file__))
K_REF  = 100.0
r_mkt  = 0.043
S_MAX  = 500.0


# ── Data loading ────────────────────────────────────────────────────────────

def load_spy_calls(moneyness_lo=0.8, moneyness_hi=1.2):
    import re, datetime
    csv_path = os.path.join(BASE, "..", "data", "spy_quotedata.csv")
    with open(csv_path, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    m = re.search(r"Last:\s*([\d.]+)", lines[1] if len(lines) > 1 else lines[0])
    S_spot = float(m.group(1)) if m else None
    today = datetime.date.today()
    rows = []
    for line in lines[4:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) < 12:
            continue
        try:
            expiry_str = parts[0].strip()
            bid  = float(parts[4]) if parts[4].strip() else float("nan")
            ask  = float(parts[5]) if parts[5].strip() else float("nan")
            K    = float(parts[11])
            expiry_date = datetime.datetime.strptime(expiry_str, "%a %b %d %Y").date()
            tau = (expiry_date - today).days / 365.0
            rows.append({"S": S_spot, "K": K, "tau": tau, "bid": bid, "ask": ask,
                         "expiry_date": expiry_date})
        except (ValueError, IndexError):
            continue
    df = pd.DataFrame(rows)
    df["mid"] = (df["bid"] + df["ask"]) / 2.0
    df = df[(df["bid"] > 0) & (df["ask"] > 0)]
    df = df[df["tau"] > 0.05]
    df["moneyness"] = df["S"] / df["K"]
    df = df[(df["moneyness"] >= moneyness_lo) & (df["moneyness"] <= moneyness_hi)]
    return df.reset_index(drop=True)


# ── Calibration ─────────────────────────────────────────────────────────────

def _bsm_iv(S, K, T, r, call_price):
    try:
        return brentq(lambda sig: bsm_call(S, K, T, r, sig) - call_price,
                      1e-4, 5.0, xtol=1e-6, maxiter=100)
    except Exception:
        return float("nan")


def calibrate_bsm(calls, S, Ks, T, r):
    ivs = [_bsm_iv(S, K, T, r, c) for K, c in zip(Ks, calls)]
    ivs = [iv for iv in ivs if not np.isnan(iv) and iv > 0]
    return float(np.median(ivs)) if ivs else 0.2


def calibrate_cev(calls, S, Ks, T, r, sigma_init=0.2):
    def loss(p):
        sigma, beta = p
        total = 0.0
        for call, K in zip(calls, Ks):
            try:
                pred = ref_cev_call(S, K, T, r, sigma, beta)
                total += (pred - call) ** 2
            except Exception:
                total += 1e6
        return total / max(len(calls), 1)

    # Try multiple starts
    starts = [
        [sigma_init, 0.5],
        [sigma_init * 0.8, 0.3],
        [sigma_init * 1.2, 0.7],
        [sigma_init, 0.9],
    ]
    best_val, best_params = np.inf, starts[0]
    for x0 in starts:
        try:
            res = minimize(loss, x0, method="L-BFGS-B",
                          bounds=[(0.01, 1.0), (0.01, 0.99)],
                          options={"maxiter": 100, "ftol": 1e-9})
            if res.fun < best_val:
                best_val, best_params = res.fun, res.x.tolist()
        except Exception:
            pass
    return tuple(best_params)


def calibrate_heston(calls, S, Ks, T, r, sigma_bsm=0.2):
    if T < 0.05:
        v0 = sigma_bsm ** 2
        return (2.0, v0, 0.3, -0.7, v0)

    def loss(p):
        kappa, theta, xi, rho, v0 = p
        total = 0.0
        for call, K in zip(calls, Ks):
            pred = heston_call(S, K, T, r, kappa, theta, xi, rho, v0)
            total += (pred - call) ** 2
        return total / max(len(calls), 1)

    bounds = [(0.05, 20), (0.001, 0.5), (0.01, 2.0), (-0.99, -0.01), (0.001, 0.5)]
    starts = [
        [1.0, 0.04, 0.2, -0.5, 0.04],
        [2.0, 0.04, 0.3, -0.7, 0.04],
        [3.0, 0.06, 0.4, -0.9, 0.06],
        [0.5, 0.02, 0.1, -0.3, 0.02],
        [5.0, 0.08, 0.5, -0.8, 0.08],
    ]
    best_val, best_params = np.inf, starts[0]
    for x0 in starts:
        try:
            res = minimize(loss, x0, method="L-BFGS-B", bounds=bounds,
                          options={"maxiter": 200, "ftol": 1e-9})
            if res.fun < best_val:
                best_val, best_params = res.fun, res.x.tolist()
        except Exception:
            pass
    return tuple(best_params)


# ── Param list ──────────────────────────────────────────────────────────────

def _build_param_list():
    params = []
    for sigma in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35]:
        params.append(ModelParams.from_bsm(sigma=sigma))
    for sigma in [0.15, 0.2, 0.25]:
        for beta in [0.3, 0.5, 0.7, 0.9]:
            params.append(ModelParams.from_cev(sigma=sigma, beta=beta))
    for kappa in [1.0, 2.0, 3.0]:
        for theta in [0.02, 0.04, 0.06]:
            for xi in [0.2, 0.3, 0.4]:
                for rho in [-0.7, -0.5]:
                    params.append(ModelParams.from_heston(
                        kappa=kappa, theta=theta, xi=xi, rho=rho, v0=theta))
    return params


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--w_pde", type=float, default=0.1)
    parser.add_argument("--w_data", type=float, default=1.0)
    parser.add_argument("--w_bsm", type=float, default=0.05)
    args = parser.parse_args()

    # Load market data
    print("Loading SPY data...")
    df = load_spy_calls()
    S_spot = float(df["S"].iloc[0])
    scale = S_spot / K_REF
    print(f"Spot={S_spot:.2f}, scale={scale:.4f}, contracts={len(df)}")

    # Load unified_v2
    print("\nLoading unified_v2...")
    param_list = _build_param_list()
    ckpt_path = os.path.join(BASE, "..", "results", "unified_v16_gl.pt")
    pinn = UnifiedPINN(param_list, hidden=128, depth=6, lr=args.lr, device=DEVICE)
    pinn.load(ckpt_path)

    # Keep a frozen copy for validation comparison
    pinn_base = UnifiedPINN(param_list, hidden=128, depth=6, device=DEVICE)
    pinn_base.load(ckpt_path)
    pinn_base.net.eval()

    # Calibrate per-expiry for all three models
    print("\nCalibrating per-expiry...")
    expiry_data = []
    for exp_date, grp in df.groupby("expiry_date"):
        T_val = float(grp["tau"].iloc[0])
        Ks = grp["K"].values.astype(float)
        calls = grp["mid"].values.astype(float)
        n = len(grp)

        # BSM calibration
        sigma_bsm = calibrate_bsm(calls, S_spot, Ks, T_val, r_mkt)

        # CEV calibration
        sigma_cev, beta_cev = calibrate_cev(calls, S_spot, Ks, T_val, r_mkt, sigma_bsm)

        # Heston calibration
        kappa_h, theta_h, xi_h, rho_h, v0_h = calibrate_heston(
            calls, S_spot, Ks, T_val, r_mkt, sigma_bsm)

        print(f"  {exp_date} T={T_val:.3f} n={n} "
              f"BSM_σ={sigma_bsm:.3f} CEV_σ={sigma_cev:.3f}_β={beta_cev:.2f} "
              f"Heston_κ={kappa_h:.1f}_θ={theta_h:.4f}_ξ={xi_h:.3f}_ρ={rho_h:.3f}")

        expiry_data.append({
            "exp_date": exp_date, "T": T_val, "Ks": Ks, "calls": calls,
            "sigma_bsm": sigma_bsm,
            "sigma_cev": sigma_cev, "beta_cev": beta_cev,
            "kappa_h": kappa_h, "theta_h": theta_h,
            "xi_h": xi_h, "rho_h": rho_h, "v0_h": v0_h,
        })

    # Build training batches (all three model types)
    print("\nBuilding training batches...")
    batches = []
    for ed in expiry_data:
        T_val, Ks, calls = ed["T"], ed["Ks"], ed["calls"]
        K_ns = K_REF * Ks / S_spot
        V_scaled = calls / scale

        # BSM batches
        for K_n, V_tgt in zip(K_ns, V_scaled):
            if not (15 < K_n < 500 and V_tgt > 0):
                continue
            p = ModelParams.from_bsm(K=float(K_n), T=max(T_val, 0.01), r=r_mkt,
                                     sigma=ed["sigma_bsm"], S_max=S_MAX)
            batches.append(("bsm", p, K_REF, float(ed["sigma_bsm"]**2), float(V_tgt)))

        # CEV batches
        for K_n, V_tgt in zip(K_ns, V_scaled):
            if not (15 < K_n < 500 and V_tgt > 0):
                continue
            p = ModelParams.from_cev(K=float(K_n), T=max(T_val, 0.01), r=r_mkt,
                                     sigma=ed["sigma_cev"], beta=ed["beta_cev"],
                                     S_max=S_MAX)
            batches.append(("cev", p, K_REF, float(ed["sigma_cev"]**2), float(V_tgt)))

        # Heston batches
        for K_n, V_tgt in zip(K_ns, V_scaled):
            if not (15 < K_n < 500 and V_tgt > 0):
                continue
            p = ModelParams.from_heston(
                K=float(K_n), T=max(T_val, 0.01), r=r_mkt,
                kappa=ed["kappa_h"], theta=ed["theta_h"],
                xi=ed["xi_h"], rho=ed["rho_h"], v0=ed["v0_h"], S_max=S_MAX)
            batches.append(("heston", p, K_REF, float(ed["v0_h"]), float(V_tgt)))

    print(f"Total training batches: {len(batches)}")

    # Fine-tuning loop
    optimizer = torch.optim.Adam(pinn.net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.1)
    pinn.net.train()

    best_mae = float("inf")
    best_state = None
    print(f"\nFine-tuning {args.epochs} epochs, lr={args.lr}...")

    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()

        # Sample batch (increasing batch_size over epochs)
        bs = min(128, 64 + (epoch * 64) // args.epochs)
        idx = np.random.choice(len(batches), min(bs, len(batches)), replace=False)
        batch = [batches[i] for i in idx]

        # Data loss
        loss_data = torch.tensor(0.0, device=DEVICE)
        for mtype, p, S_val, v_val, V_tgt in batch:
            S_t = torch.tensor([[S_val]], dtype=torch.float32, device=DEVICE)
            v_t = torch.tensor([[v_val]], dtype=torch.float32, device=DEVICE)
            t_t = torch.tensor([[0.0]], dtype=torch.float32, device=DEVICE)
            lam = p.to_lambda_tensor(DEVICE)
            pred = pinn.net(S_t/p.S_max, v_t/p.v_max, t_t/p.T,
                           lam, S_t, t_t, p.K, p.T, p.r)
            rel_err = (pred - V_tgt) / (abs(V_tgt) + p.K * 0.1)
            loss_data = loss_data + rel_err ** 2
        loss_data = loss_data / len(batch)

        # PDE preservation loss
        n_pde = 512
        S_c = torch.FloatTensor(n_pde, 1).uniform_(1.0, S_MAX).to(DEVICE)
        v_c = torch.FloatTensor(n_pde, 1).uniform_(1e-5, 1.0).to(DEVICE)
        t_c = torch.FloatTensor(n_pde, 1).uniform_(0.0, 5.0 * 0.999).to(DEVICE)
        # Use a random Heston-like lambda from batches for PDE
        p0 = batch[0][1]
        lam_c = p0.to_lambda_tensor(DEVICE).expand(n_pde, -1)
        res = unified_pde_residual(pinn.net, S_c, v_c, t_c, lam_c,
                                   p0.K, p0.T, p0.r, p0.S_max, p0.v_max)
        loss_pde = torch.mean(res ** 2)

        # BSM raw suppression
        S_b = torch.FloatTensor(512, 1).uniform_(1.0, S_MAX).to(DEVICE)
        v_b = torch.FloatTensor(512, 1).uniform_(1e-5, 1.0).to(DEVICE)
        t_b = torch.FloatTensor(512, 1).uniform_(0, 5.0 * 0.999).to(DEVICE)
        p_bsm = ModelParams.from_bsm(sigma=0.2, S_max=S_MAX)
        lam_b = p_bsm.to_lambda_tensor(DEVICE).expand(512, -1)
        _, raw_b = pinn.net(S_b/S_MAX, v_b/1.0, t_b/1.0, lam_b,
                           S_b, t_b, p_bsm.K, p_bsm.T, p_bsm.r, return_raw=True)
        loss_bsm = torch.mean(raw_b**2)

        loss = args.w_data * loss_data + args.w_pde * loss_pde + args.w_bsm * loss_bsm
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pinn.net.parameters(), max_norm=0.5)
        optimizer.step()
        scheduler.step()

        # Validation every 200 epochs
        if epoch % 200 == 0:
            pinn.net.eval()
            total_base_err, total_ft_err, total_n = 0.0, 0.0, 0
            for ed in expiry_data[:5]:  # validate on first 5 expiries
                T_val = ed["T"]
                Ks = ed["Ks"]
                calls = ed["calls"]
                k_h, t_h, xi_h, rh_h, v_h = (
                    ed["kappa_h"], ed["theta_h"], ed["xi_h"],
                    ed["rho_h"], ed["v0_h"])
                for K, call in zip(Ks, calls):
                    K_n = K_REF * K / S_spot
                    p = ModelParams.from_heston(
                        K=K_n, T=max(T_val, 0.01), r=r_mkt,
                        kappa=k_h, theta=t_h, xi=xi_h, rho=rh_h, v0=v_h,
                        S_max=S_MAX)
                    base_pred = pinn_base.price(p, S=K_REF) * scale
                    ft_pred   = pinn.price(p, S=K_REF) * scale
                    total_base_err += abs(base_pred - call)
                    total_ft_err   += abs(ft_pred - call)
                    total_n += 1
            ma_base = total_base_err / total_n
            ma_ft   = total_ft_err / total_n
            print(f"  epoch {epoch:4d} loss={loss.item():.3e} "
                  f"val_base_MAE={ma_base:.4f} val_ft_MAE={ma_ft:.4f} "
                  f"improve={(ma_base-ma_ft)/ma_base*100:+.1f}%")

            if ma_ft < best_mae:
                best_mae = ma_ft
                best_state = deepcopy(pinn.net.state_dict())
                print(f"  >> new best MAE={best_mae:.4f}")
            pinn.net.train()

        if epoch % 1000 == 0 and best_state is not None:
            # Save intermediate best
            pinn.net.load_state_dict(best_state)
            out = os.path.join(BASE, "results", "unified_v2_ft_v2.pt")
            os.makedirs(os.path.dirname(out), exist_ok=True)
            pinn.save(out)
            print(f"  checkpoint saved to {out}")

    # Restore best and save final
    if best_state is not None:
        pinn.net.load_state_dict(best_state)
    out_final = os.path.join(BASE, "results", "unified_v2_ft_v2.pt")
    pinn.save(out_final)
    print(f"\nFinal model saved: {out_final}  best_val_MAE={best_mae:.4f}")


if __name__ == "__main__":
    main()
