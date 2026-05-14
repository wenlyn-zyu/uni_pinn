"""
Heston 合成数据精度评估，分两部分：

Part 1 — 各自 in-distribution 精度
  heston_pinn  : HESTON_INDEP (kappa=1.0, xi=0.39, rho=-0.93, r=0.1)
  hainaut_orig : Hainaut 训练范围内参数 (kappa=1.15, xi=0.20, rho=-0.40, r=0.04)
  各自与 GL 参考解对比，S 网格取各自训练范围内。

Part 2 — 统一测试参数 HESTON_UNIFIED，五模型横向对比
  HESTON_UNIFIED = kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, v0=0.04, r=0.05
  模型：heston_pinn, hainaut_orig, unified_v2, parametric_pinn
  注：hainaut theta=0.04 略低于其训练下限 0.062，其余参数均在范围内。
  S 网格 [50, 170]（Hainaut S_RANGE=[20,180] 与 heston_pinn S_max=400 的交集）。
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

HESTON_INDEP   = dict(kappa=1.0,  theta=0.08,  xi=0.39, rho=-0.93, v0=0.04)
r_INDEP        = 0.1

HESTON_HAINAUT = dict(kappa=1.15, theta=0.202, xi=0.20, rho=-0.40, v0=0.04)
r_HAINAUT      = 0.04

HESTON_UNIFIED = dict(kappa=2.0,  theta=0.04,  xi=0.3,  rho=-0.70, v0=0.04)
r_UNIFIED      = 0.05


def mse(pred, ref):
    return float(np.mean((np.array(pred) - np.array(ref)) ** 2))

def relmse(pred, ref):
    ref, pred = np.array(ref), np.array(pred)
    mask = np.abs(ref) > 0.01
    return float(np.mean(((pred[mask] - ref[mask]) / ref[mask]) ** 2)) if mask.sum() else float("nan")

def relmae(pred, ref):
    ref, pred = np.array(ref), np.array(pred)
    mask = np.abs(ref) > 0.01
    return float(np.mean(np.abs((pred[mask] - ref[mask]) / ref[mask]))) if mask.sum() else float("nan")

def print_row(name, pred, ref, note=""):
    print(f"  {name:<22} MSE={mse(pred,ref):10.4e}  RelMSE={relmse(pred,ref):8.4f}  RelMAE={relmae(pred,ref):8.4f}  {note}")


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
unified = load_unified()
param_pinn = load_parametric()

print("All models loaded.\n")


def hainaut_call(S, params, r):
    """Hainaut put -> call via put-call parity."""
    put = hainaut.price(S=S, V=params["v0"], t=0.0, T=T_SYN, r=r,
                        kappa=params["kappa"], theta=params["theta"],
                        xi=params["xi"], rho=params["rho"])
    return put + S - K_SYN * np.exp(-r * T_SYN)


# ══════════════════════════════════════════════════════════════════════════════
# Part 1: 各自 in-distribution 精度
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("Part 1: 各自 in-distribution 精度")
print("=" * 70)

# heston_pinn: S∈[50,250], HESTON_INDEP
S_INDEP = np.linspace(50, 250, 50)
ref_indep  = np.array([heston_call(S, K_SYN, T_SYN, r_INDEP, **HESTON_INDEP) for S in S_INDEP])
pred_indep = np.array([heston_pinn.price(S) for S in S_INDEP])
print(f"\nheston_pinn  (HESTON_INDEP: kappa={HESTON_INDEP['kappa']} xi={HESTON_INDEP['xi']} rho={HESTON_INDEP['rho']} r={r_INDEP})")
print_row("heston_pinn", pred_indep, ref_indep, "in-distribution")

# hainaut_orig: S∈[30,170], Hainaut 训练范围内参数
S_HAINAUT = np.linspace(30, 170, 50)
ref_hainaut  = np.array([heston_call(S, K_SYN, T_SYN, r_HAINAUT, **HESTON_HAINAUT) for S in S_HAINAUT])
pred_hainaut = np.array([hainaut_call(S, HESTON_HAINAUT, r_HAINAUT) for S in S_HAINAUT])
print(f"\nhainaut_orig (kappa={HESTON_HAINAUT['kappa']} xi={HESTON_HAINAUT['xi']} rho={HESTON_HAINAUT['rho']} r={r_HAINAUT})")
print_row("hainaut_orig", pred_hainaut, ref_hainaut, "in-distribution")


# ══════════════════════════════════════════════════════════════════════════════
# Part 2: 统一测试参数 HESTON_UNIFIED，五模型横向对比
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Part 2: HESTON_UNIFIED 统一测试参数，五模型横向对比")
print(f"  kappa={HESTON_UNIFIED['kappa']} theta={HESTON_UNIFIED['theta']} "
      f"xi={HESTON_UNIFIED['xi']} rho={HESTON_UNIFIED['rho']} v0={HESTON_UNIFIED['v0']} r={r_UNIFIED}")
print("=" * 70)

S_UNIFIED = np.linspace(50, 170, 50)
ref_u = np.array([heston_call(S, K_SYN, T_SYN, r_UNIFIED, **HESTON_UNIFIED) for S in S_UNIFIED])

# heston_pinn (OOD: trained on different params)
pred_pinn_u = np.array([heston_pinn.price(S) for S in S_UNIFIED])

# hainaut_orig (theta=0.04 略低于训练下限 0.062，其余 in-distribution)
pred_hainaut_u = np.array([hainaut_call(S, HESTON_UNIFIED, r_UNIFIED) for S in S_UNIFIED])

# unified_v2
from unified_pinn_v2 import ModelParams
p_u = ModelParams.from_heston(K=K_SYN, T=T_SYN, r=r_UNIFIED, **HESTON_UNIFIED)
pred_unified_u = np.array([unified.price(p_u, S) for S in S_UNIFIED])

# parametric_pinn
pred_param_u = np.array([
    param_pinn.price(S=S, K=K_SYN, T=T_SYN, r=r_UNIFIED,
                     kappa=HESTON_UNIFIED["kappa"], theta=HESTON_UNIFIED["theta"],
                     xi=HESTON_UNIFIED["xi"], rho=HESTON_UNIFIED["rho"],
                     v0=HESTON_UNIFIED["v0"])
    for S in S_UNIFIED])

print()
print_row("heston_pinn",   pred_pinn_u,     ref_u, "OOD (diff params)")
print_row("hainaut_orig",  pred_hainaut_u,  ref_u, "theta略OOD")
print_row("unified_v2",    pred_unified_u,  ref_u, "in-distribution")
print_row("parametric",    pred_param_u,    ref_u, "in-distribution")

print("\nSample prices (S, ref, heston_pinn, hainaut, unified, parametric):")
for i in [5, 15, 25, 35, 45]:
    S = S_UNIFIED[i]
    print(f"  S={S:6.1f}  ref={ref_u[i]:7.4f}  "
          f"pinn={pred_pinn_u[i]:7.4f}  "
          f"hainaut={pred_hainaut_u[i]:7.4f}  "
          f"unified={pred_unified_u[i]:7.4f}  "
          f"param={pred_param_u[i]:7.4f}")
