# option_pinn/exploration/fast_finetune.py
"""Fast fine-tune of unified_v2 on SPY data — simplified calibration, print() logging.

Strategy: use BSM implied vol + Heston default params across all expiries,
fine-tune gently to reduce market MAE while preserving PDE constraints.

Usage:
  python fast_finetune.py --epochs 2000 --lr 2e-5
"""

import os, sys, re, datetime, argparse
import numpy as np
import pandas as pd
import torch
from copy import deepcopy
from scipy.optimize import brentq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ref_solvers import bsm_call, heston_call
from unified_pinn_v2 import UnifiedPINN, ModelParams, unified_pde_residual

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE   = os.path.dirname(os.path.abspath(__file__))
K_REF  = 100.0
r_mkt  = 0.043
S_MAX  = 500.0


def load_spy():
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
    df = df[(df["moneyness"] >= 0.8) & (df["moneyness"] <= 1.2)]
    return df.reset_index(drop=True)


def _build_param_list():
    from unified_pinn_v2 import ModelParams
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--w_pde", type=float, default=0.05)
    parser.add_argument("--w_data", type=float, default=1.0)
    args = parser.parse_args()

    # Load data
    print("Loading SPY data...", flush=True)
    df = load_spy()
    S_spot = float(df["S"].iloc[0])
    scale = S_spot / K_REF
    print(f"  S_spot={S_spot:.2f}  scale={scale:.4f}  contracts={len(df)}", flush=True)

    # Load unified_v2
    print("Loading unified_v2...", flush=True)
    param_list = _build_param_list()
    ckpt_path = os.path.join(BASE, "..", "results", "unified_v16_gl.pt")
    pinn = UnifiedPINN(param_list, hidden=128, depth=6, device=DEVICE)
    pinn.load(ckpt_path)

    # Frozen base for comparison
    pinn_base = UnifiedPINN(param_list, hidden=128, depth=6, device=DEVICE)
    pinn_base.load(ckpt_path)
    pinn_base.net.eval()

    # Fast calibration: BSM IV only (skip expensive L-BFGS-B)
    print("Calibrating BSM IV per expiry...", flush=True)
    expiry_data = []
    for exp_date, grp in df.groupby("expiry_date"):
        T_val = float(grp["tau"].iloc[0])
        Ks = grp["K"].values.astype(float)
        calls = grp["mid"].values.astype(float)
        ivs = []
        for K, c in zip(Ks, calls):
            try:
                iv = brentq(lambda sig: bsm_call(S_spot, K, T_val, r_mkt, sig) - c,
                           1e-4, 5.0, xtol=1e-6, maxiter=100)
                ivs.append(iv)
            except Exception:
                pass
        sigma_bsm = float(np.median([iv for iv in ivs if 0.01 < iv < 2.0])) if ivs else 0.18
        print(f"  {str(exp_date):12s} T={T_val:.3f} n={len(Ks)} sigma_bsm={sigma_bsm:.3f}", flush=True)
        expiry_data.append({
            "T": T_val, "Ks": Ks, "calls": calls, "sigma_bsm": sigma_bsm,
        })

    # Build training batches
    print("Building batches...", flush=True)
    batches = []
    for ed in expiry_data:
        T_val, Ks, calls, sig = ed["T"], ed["Ks"], ed["calls"], ed["sigma_bsm"]
        K_ns = K_REF * Ks / S_spot
        V_scaled = calls / scale
        for K_n, V_tgt in zip(K_ns, V_scaled):
            if not (15 < K_n < 500 and V_tgt > 0.01):
                continue
            # Use BSM params (the finetune adapts the full network, not just Heston)
            p = ModelParams.from_bsm(K=float(K_n), T=max(T_val, 0.01), r=r_mkt,
                                     sigma=sig, S_max=S_MAX)
            batches.append((p, K_REF, float(sig**2), float(V_tgt)))
    print(f"  {len(batches)} training samples", flush=True)

    # Training loop
    optimizer = torch.optim.Adam(pinn.net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.1)
    pinn.net.train()

    best_mae = float("inf")
    best_state = None
    print(f"\nTraining {args.epochs} epochs, lr={args.lr}...", flush=True)

    import time
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()

        # Data loss: sample batch
        bs = min(64, len(batches))
        idx = np.random.choice(len(batches), bs, replace=False)
        loss_data = torch.tensor(0.0, device=DEVICE)
        for i in idx:
            p, S_val, v_val, V_tgt = batches[i]
            S_t = torch.tensor([[S_val]], dtype=torch.float32, device=DEVICE)
            v_t = torch.tensor([[v_val]], dtype=torch.float32, device=DEVICE)
            t_t = torch.tensor([[0.0]], dtype=torch.float32, device=DEVICE)
            lam = p.to_lambda_tensor(DEVICE)
            pred = pinn.net(S_t/p.S_max, v_t/p.v_max, t_t/p.T,
                           lam, S_t, t_t, p.K, p.T, p.r)
            rel_err = (pred - V_tgt) / (abs(V_tgt) + p.K * 0.1)
            loss_data = loss_data + rel_err ** 2
        loss_data = loss_data / bs

        # PDE preservation
        n_pde = 256
        p0 = batches[0][0]
        S_c = torch.FloatTensor(n_pde, 1).uniform_(1.0, S_MAX).to(DEVICE)
        v_c = torch.FloatTensor(n_pde, 1).uniform_(1e-5, 1.0).to(DEVICE)
        t_c = torch.FloatTensor(n_pde, 1).uniform_(0.0, 5.0 * 0.999).to(DEVICE)
        lam_c = p0.to_lambda_tensor(DEVICE).expand(n_pde, -1)
        res = unified_pde_residual(pinn.net, S_c, v_c, t_c, lam_c,
                                   p0.K, p0.T, p0.r, p0.S_max, p0.v_max)
        loss_pde = torch.mean(res ** 2)

        loss = args.w_data * loss_data + args.w_pde * loss_pde
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pinn.net.parameters(), max_norm=0.5)
        optimizer.step()
        scheduler.step()

        if epoch % 200 == 0:
            # Validation
            pinn.net.eval()
            total_base, total_ft, total_n = 0.0, 0.0, 0
            for ed in expiry_data[:3]:  # validate on first 3 expiries
                T_val = ed["T"]
                for K, call in zip(ed["Ks"], ed["calls"]):
                    K_n = K_REF * K / S_spot
                    p = ModelParams.from_bsm(K=K_n, T=max(T_val, 0.01), r=r_mkt,
                                            sigma=ed["sigma_bsm"], S_max=S_MAX)
                    total_base += abs(pinn_base.price(p, S=K_REF) * scale - call)
                    total_ft   += abs(pinn.price(p, S=K_REF) * scale - call)
                    total_n += 1
            ma_base = total_base / total_n
            ma_ft   = total_ft / total_n
            imp = (ma_base - ma_ft) / (ma_base + 1e-8) * 100
            print(f"  [{time.time()-t0:.0f}s] epoch {epoch:4d}  loss={loss.item():.3e}  "
                  f"val_base={ma_base:.4f}  val_ft={ma_ft:.4f}  improve={imp:+.1f}%",
                  flush=True)
            if ma_ft < best_mae:
                best_mae = ma_ft
                best_state = deepcopy(pinn.net.state_dict())
                print(f"  >> new best MAE={best_mae:.4f}", flush=True)
            pinn.net.train()

    # Save best
    if best_state is not None:
        pinn.net.load_state_dict(best_state)
    out_path = os.path.join(BASE, "results", "unified_v2_ft_v2.pt")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pinn.save(out_path)
    print(f"\nSaved: {out_path}  best_val_MAE={best_mae:.4f}", flush=True)


if __name__ == "__main__":
    main()
