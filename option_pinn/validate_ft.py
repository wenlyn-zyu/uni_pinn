"""验证 unified_v2_ft vs unified_v2 的市场 MAE，不需要重新训练。"""
import os, sys, warnings
import numpy as np
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

BASE   = os.path.dirname(__file__)
K_REF  = 100.0
r_mkt  = 0.043
S_MAX  = 300.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from finetune_heston import _load_spy, _bsm_iv, _calibrate_heston, _build_param_list


def safe_price(pinn, p, S):
    try:
        v = pinn.price(p, S=S)
        if not np.isfinite(v) or v < 0:
            return np.nan
        return v
    except Exception:
        return np.nan


def main():
    from unified_pinn_v2 import UnifiedPINN, ModelParams

    df     = _load_spy()
    S_spot = float(df["S"].iloc[0])
    scale  = S_spot / K_REF

    param_list = _build_param_list()

    pinn_base = UnifiedPINN(param_list, hidden=128, depth=6, device=DEVICE)
    pinn_base.load(os.path.join(BASE, "results/unified_v16_gl.pt"))
    pinn_base.net.eval()

    pinn_ft = UnifiedPINN(param_list, hidden=128, depth=6, device=DEVICE)
    pinn_ft.load(os.path.join(BASE, "results/unified_v2_ft.pt"))
    pinn_ft.net.eval()

    print(f"{'到期日':12s} {'T':>6} {'n':>4} {'MAE_base':>10} {'MAE_ft':>10} {'改善':>8}")
    total_base, total_ft, total_n = 0.0, 0.0, 0

    for exp_date, grp in df.groupby("expiry_date"):
        T_val = float(grp["tau"].iloc[0])
        Ks    = grp["K"].values.astype(float)
        calls = grp["mid"].values.astype(float)

        ivs = [_bsm_iv(S_spot, K, T_val, r_mkt, c) for K, c in zip(Ks, calls)]
        ivs = [iv for iv in ivs if not np.isnan(iv) and iv > 0]
        sigma_bsm = float(np.median(ivs)) if ivs else 0.2
        kappa, theta, xi, rho, v0 = _calibrate_heston(
            calls, S_spot, Ks, T_val, r_mkt, sigma_bsm)

        preds_base, preds_ft, calls_valid = [], [], []
        for K, c in zip(Ks, calls):
            K_n = K_REF * K / S_spot
            p = ModelParams.from_heston(K=K_n, T=max(T_val, 0.01), r=r_mkt,
                                        kappa=kappa, theta=theta,
                                        xi=xi, rho=rho, v0=v0, S_max=S_MAX)
            pb = safe_price(pinn_base, p, K_REF)
            pf = safe_price(pinn_ft,   p, K_REF)
            if np.isfinite(pb) and np.isfinite(pf):
                preds_base.append(pb * scale)
                preds_ft.append(pf * scale)
                calls_valid.append(c)

        if not preds_base:
            print(f"  {str(exp_date):12s} {T_val:6.3f} {len(Ks):4d}  (skip — all NaN)")
            continue

        calls_arr = np.array(calls_valid)
        mae_base  = float(np.mean(np.abs(np.array(preds_base) - calls_arr)))
        mae_ft    = float(np.mean(np.abs(np.array(preds_ft)   - calls_arr)))
        improve   = (mae_base - mae_ft) / (mae_base + 1e-8) * 100
        print(f"  {str(exp_date):12s} {T_val:6.3f} {len(calls_valid):4d} "
              f"{mae_base:10.4f} {mae_ft:10.4f} {improve:+7.1f}%")
        total_base += mae_base * len(calls_valid)
        total_ft   += mae_ft   * len(calls_valid)
        total_n    += len(calls_valid)

    if total_n > 0:
        ob = total_base / total_n
        of = total_ft   / total_n
        print(f"\n  {'全局':12s} {'':>6} {total_n:4d} "
              f"{ob:10.4f} {of:10.4f} {(ob-of)/ob*100:+7.1f}%")


if __name__ == "__main__":
    main()
