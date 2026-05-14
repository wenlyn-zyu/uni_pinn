"""
Heston 合成数据精度评估，分两部分。
深度虚值 CALL（moneyness S/K < 0.85）在所有评估中均被剔除，
避免 put-call parity 转换误差主导指标。

Part 1 — 各自 in-distribution 精度
  heston_pinn  : HESTON_INDEP (kappa=1.0, xi=0.39, rho=-0.93, r=0.1)，S∈[50,250]
  hainaut_orig : HESTON_UNIFIED (kappa=2.0, xi=0.3, rho=-0.7, r=0.05)，S∈[50,170]
                 theta=0.04 略低于 Hainaut 训练下限 0.062，其余参数均在范围内
  各自与 GL 参考解对比。

Part 2 — 统一测试参数 HESTON_UNIFIED，四模型横向对比
  HESTON_UNIFIED = kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, v0=0.04, r=0.05
  模型：heston_pinn (OOD), hainaut_orig, unified_v2, parametric_pinn
  S 网格 [50, 170]（Hainaut S_RANGE=[20,180] 与 heston_pinn S_max=400 的交集）。

Hainaut 定价 PUT，price() 返回绝对 put 价格，经 put-call parity 转为 CALL。
"""
import sys, os
import numpy as np
import torch

BASE = "/home/yz2026/zhuwl2022/uni_pinn/option_pinn"
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "independent"))
sys.path.insert(0, os.path.join(BASE, "parametric_pinn"))

from ref_solvers import heston_call
from heston_hainaut import HestonHainaut
from heston_pinn import Heston_PINN

K_SYN = 100.0
T_SYN = 1.0
MONEYNESS_LO = 0.85   # 剔除 S/K < 0.85 的深度虚值 CALL

HESTON_INDEP   = dict(kappa=1.0, theta=0.08, xi=0.39, rho=-0.93, v0=0.04)
r_INDEP        = 0.1

HESTON_UNIFIED = dict(kappa=2.0, theta=0.04, xi=0.3,  rho=-0.70, v0=0.04)
r_UNIFIED      = 0.05


def _mask_otm(S_arr, K=K_SYN, lo=MONEYNESS_LO):
    """True where S/K >= lo (keep near-ATM and ITM calls)."""
    return np.array(S_arr) / K >= lo


def mse(pred, ref, mask=None):
    p, r = np.array(pred), np.array(ref)
    if mask is not None:
        p, r = p[mask], r[mask]
    return float(np.mean((p - r) ** 2))


def relmse(pred, ref, mask=None):
    p, r = np.array(pred), np.array(ref)
    if mask is not None:
        p, r = p[mask], r[mask]
    m = np.abs(r) > 0.01
    return float(np.mean(((p[m] - r[m]) / r[m]) ** 2)) if m.sum() else float("nan")


def relmae(pred, ref, mask=None):
    p, r = np.array(pred), np.array(ref)
    if mask is not None:
        p, r = p[mask], r[mask]
    m = np.abs(r) > 0.01
    return float(np.mean(np.abs((p[m] - r[m]) / r[m]))) if m.sum() else float("nan")


def print_row(name, pred, ref, S_arr, note=""):
    mask = _mask_otm(S_arr)
    n_kept = mask.sum()
    print(f"  {name:<22} MSE={mse(pred,ref,mask):10.4e}  "
          f"RelMSE={relmse(pred,ref,mask):8.4f}  "
          f"RelMAE={relmae(pred,ref,mask):8.4f}  "
          f"(n={n_kept})  {note}")


# ── Load models ───────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Loading models...")

ckpt = torch.load(os.path.join(BASE, "results/indep_heston.pt"), map_location=DEVICE)
heston_pinn = Heston_PINN(**ckpt["params"], device=DEVICE)
heston_pinn.aux_net.load_state_dict(ckpt["aux_state"])
heston_pinn.main_net.load_state_dict(ckpt["main_state"])
heston_pinn.aux_net.eval(); heston_pinn.main_net.eval()

hainaut = HestonHainaut(device=DEVICE)
hainaut.load(os.path.join(BASE, "results/hainaut.pt"))

from eval_all import load_unified, load_parametric
unified    = load_unified()
param_pinn = load_parametric()

from unified_pinn_v2 import ModelParams
print("All models loaded.\n")


def hainaut_call(S, params, r):
    """Hainaut put -> call via put-call parity. price() returns absolute put."""
    put = hainaut.price(S=S, V=params["v0"], t=0.0, T=T_SYN, r=r,
                        kappa=params["kappa"], theta=params["theta"],
                        xi=params["xi"],      rho=params["rho"])
    return put + S - K_SYN * np.exp(-r * T_SYN)


# ══════════════════════════════════════════════════════════════════════════════
# Part 1: 各自 in-distribution 精度（剔除深度虚值 CALL）
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 72)
print("Part 1: 各自 in-distribution 精度  (S/K >= 0.85 only)")
print("=" * 72)

# heston_pinn: HESTON_INDEP, S∈[50,250]
S_INDEP = np.linspace(50, 250, 50)
ref_indep  = np.array([heston_call(S, K_SYN, T_SYN, r_INDEP, **HESTON_INDEP) for S in S_INDEP])
pred_indep = np.array([heston_pinn.price(S) for S in S_INDEP])
print(f"\nheston_pinn  HESTON_INDEP: kappa={HESTON_INDEP['kappa']} xi={HESTON_INDEP['xi']} "
      f"rho={HESTON_INDEP['rho']} r={r_INDEP}")
print_row("heston_pinn", pred_indep, ref_indep, S_INDEP, "in-distribution")

# hainaut_orig: HESTON_UNIFIED, S∈[50,170]
S_HAINAUT = np.linspace(50, 170, 50)
ref_hainaut  = np.array([heston_call(S, K_SYN, T_SYN, r_UNIFIED, **HESTON_UNIFIED) for S in S_HAINAUT])
pred_hainaut = np.array([hainaut_call(S, HESTON_UNIFIED, r_UNIFIED) for S in S_HAINAUT])
print(f"\nhainaut_orig HESTON_UNIFIED: kappa={HESTON_UNIFIED['kappa']} xi={HESTON_UNIFIED['xi']} "
      f"rho={HESTON_UNIFIED['rho']} r={r_UNIFIED}  (theta=0.04 略低于训练下限 0.062)")
print_row("hainaut_orig", pred_hainaut, ref_hainaut, S_HAINAUT, "near in-distribution")


# ══════════════════════════════════════════════════════════════════════════════
# Part 2: HESTON_UNIFIED 统一参数，四模型横向对比（剔除深度虚值 CALL）
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("Part 2: HESTON_UNIFIED 统一参数，四模型对比  (S/K >= 0.85 only)")
print(f"  kappa={HESTON_UNIFIED['kappa']} theta={HESTON_UNIFIED['theta']} "
      f"xi={HESTON_UNIFIED['xi']} rho={HESTON_UNIFIED['rho']} "
      f"v0={HESTON_UNIFIED['v0']} r={r_UNIFIED}")
print("=" * 72)

S_U = np.linspace(50, 170, 50)
ref_u = np.array([heston_call(S, K_SYN, T_SYN, r_UNIFIED, **HESTON_UNIFIED) for S in S_U])

pred_pinn_u    = np.array([heston_pinn.price(S) for S in S_U])
pred_hainaut_u = np.array([hainaut_call(S, HESTON_UNIFIED, r_UNIFIED) for S in S_U])

p_u = ModelParams.from_heston(K=K_SYN, T=T_SYN, r=r_UNIFIED, **HESTON_UNIFIED)
pred_unified_u = np.array([unified.price(p_u, S) for S in S_U])

pred_param_u = np.array([
    param_pinn.price(S=S, K=K_SYN, T=T_SYN, r=r_UNIFIED,
                     kappa=HESTON_UNIFIED["kappa"], theta=HESTON_UNIFIED["theta"],
                     xi=HESTON_UNIFIED["xi"],       rho=HESTON_UNIFIED["rho"],
                     v0=HESTON_UNIFIED["v0"])
    for S in S_U])

print()
print_row("heston_pinn",  pred_pinn_u,    ref_u, S_U, "OOD (diff params)")
print_row("hainaut_orig", pred_hainaut_u, ref_u, S_U, "theta略OOD")
print_row("unified_v2",   pred_unified_u, ref_u, S_U, "in-distribution")
print_row("parametric",   pred_param_u,   ref_u, S_U, "in-distribution")

mask_u = _mask_otm(S_U)
print("\nSample prices (S/K>=0.85, ref, heston_pinn, hainaut, unified, parametric):")
for i in np.where(mask_u)[0][[0, 5, 12, 20, 30, -1]]:
    S = S_U[i]
    print(f"  S={S:6.1f}  ref={ref_u[i]:7.4f}  "
          f"pinn={pred_pinn_u[i]:7.4f}  "
          f"hainaut={pred_hainaut_u[i]:7.4f}  "
          f"unified={pred_unified_u[i]:7.4f}  "
          f"param={pred_param_u[i]:7.4f}")
