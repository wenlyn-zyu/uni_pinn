# 统一评估框架实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 清理 option_pinn 代码、建立统一评估脚本、fine-tune Heston 分支、更新论文实验结果

**Architecture:** 单入口脚本 `eval_all.py` 加载所有模型，在固定参数合成数据和真实 SPY 市场数据上评估，输出三张 CSV 供论文直接引用。`finetune_heston.py` 在 SPY 数据上对 unified_v2 的 Heston 分支做域适应。

**Tech Stack:** Python 3, PyTorch, NumPy, SciPy (ncx2, GL quadrature), pandas, tqdm

---

## 文件结构

| 操作 | 路径 | 说明 |
|------|------|------|
| 删除 | `option_pinn/spy_backtest_v15.py` | 旧版本 |
| 删除 | `option_pinn/spy_backtest_v15_ft2.py` | 旧版本 |
| 删除 | `option_pinn/train_v15_schr.py` | 训练脚本已完成使命 |
| 删除 | `option_pinn/train_v16_cont.py` | 同上 |
| 删除 | `option_pinn/train_v16_heston_gl.py` | 同上 |
| 删除 | `option_pinn/eval_compare.py` | 被 eval_all.py 取代 |
| 删除 | `option_pinn/evaluate.py` | 被 eval_all.py 取代 |
| 删除 | `option_pinn/parametric_pinn/quick_test.py` | 临时脚本 |
| 删除 | `option_pinn/parametric_pinn/generate_ref_data.py` | 数据已生成 |
| 删除 | `option_pinn/parametric_pinn/generate_ref_fast.py` | 数据已生成 |
| 删除 | 根目录 `spy_backtest_parametric.py` | 重复文件 |
| 新建 | `option_pinn/eval_all.py` | 统一评估入口 |
| 新建 | `option_pinn/finetune_heston.py` | Heston fine-tune |
| 新建 | `option_pinn/results/eval_synthetic.csv` | 合成数据结果（脚本生成） |
| 新建 | `option_pinn/results/eval_market.csv` | 市场数据结果（脚本生成） |
| 新建 | `option_pinn/results/eval_greeks.csv` | Greeks 精度（脚本生成） |

---

## Checkpoint 对应关系（服务器路径）

| 模型 | 路径 |
|------|------|
| BSM 独立 | `results/indep_bsm.pt` |
| CEV 独立 | `results/indep_cev.pt` |
| Heston 独立 | `results/indep_heston.pt` |
| Unified v2 | `results/unified_v16_gl.pt` |
| Unified v2 fine-tuned | `results/unified_v2_ft.pt`（finetune_heston.py 生成） |
| 参数化 PINN | `parametric_pinn/results/fully_param_v1.pt` |

---

## Task 1: 清理冗余文件并同步 SPY 数据到服务器

**Files:**
- 删除: `option_pinn/spy_backtest_v15.py` 等（见上表）
- 同步: `option_pinn/data/spy_quotedata.csv` → 服务器

- [ ] **Step 1: 本地删除冗余文件**

```powershell
cd D:\Willing\Study\Course\final\sym\paper\uni_pinn
Remove-Item option_pinn/spy_backtest_v15.py
Remove-Item option_pinn/spy_backtest_v15_ft2.py
Remove-Item option_pinn/train_v15_schr.py
Remove-Item option_pinn/train_v16_cont.py
Remove-Item option_pinn/train_v16_heston_gl.py
Remove-Item option_pinn/eval_compare.py
Remove-Item option_pinn/evaluate.py
Remove-Item option_pinn/parametric_pinn/quick_test.py
Remove-Item option_pinn/parametric_pinn/generate_ref_data.py
Remove-Item option_pinn/parametric_pinn/generate_ref_fast.py
Remove-Item spy_backtest_parametric.py
```

- [ ] **Step 2: 同步 SPY 数据到服务器**

```bash
ssh idata2 "mkdir -p /home/yz2026/zhuwl2022/uni_pinn/option_pinn/data"
scp "D:\Willing\Study\Course\final\sym\paper\uni_pinn\option_pinn\data\spy_quotedata.csv" idata2:/home/yz2026/zhuwl2022/uni_pinn/option_pinn/data/
```

- [ ] **Step 3: git 提交清理**

```bash
cd D:\Willing\Study\Course\final\sym\paper\uni_pinn
git add -A
git commit -m "refactor: remove redundant scripts, keep core model files"
git push
ssh idata2 "cd /home/yz2026/zhuwl2022/uni_pinn && git pull"
```

---

## Task 2: 新建参考解模块 `option_pinn/ref_solvers.py`

**Files:**
- 新建: `option_pinn/ref_solvers.py`

- [ ] **Step 1: 新建文件**

内容见下方（分三部分：BSM、CEV、Heston GL）。

```python
# option_pinn/ref_solvers.py
import numpy as np
from scipy.stats import norm, ncx2
from numpy.polynomial.legendre import leggauss

def bsm_call(S, K, T, r, sigma):
    eps = 1e-10
    T = max(T, eps); sigma = max(sigma, eps)
    sqt = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / sqt
    d2 = d1 - sqt
    return float(max(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2), 0.0))

def bsm_greeks(S, K, T, r, sigma):
    eps = 1e-10
    T = max(T, eps); sigma = max(sigma, eps)
    sqt = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / sqt
    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S * sqt)
    vega  = S * norm.pdf(d1) * np.sqrt(T)
    return {"delta": delta, "gamma": gamma, "vega": vega}

def cev_call(S, K, T, r, sigma, beta):
    if abs(beta - 1.0) < 1e-9:
        return bsm_call(S, K, T, r, sigma)
    d = 1.0 - beta
    nu  = 1.0 / d
    lam = (2.0 * r) / (sigma**2 * d * (np.exp(2.0 * r * d * T) - 1.0))
    x   = lam * S**(2.0 * d) * np.exp(2.0 * r * d * T)
    y   = lam * K**(2.0 * d)
    call = (S * (1.0 - ncx2.cdf(y, df=2.0 + nu, nc=x))
            - K * np.exp(-r * T) * ncx2.cdf(x, df=nu, nc=y))
    return float(max(call, max(S - K * np.exp(-r * T), 0.0)))

_N_GL = 96
_GL_NODES, _GL_WEIGHTS = leggauss(_N_GL)
_PHI_MAX = 100.0
_GL_PHI = (_GL_NODES + 1.0) * 0.5 * _PHI_MAX
_GL_W   = _GL_WEIGHTS * 0.5 * _PHI_MAX

def _heston_cf(phi, S, K, T, r, kappa, theta, xi, rho, v0, j):
    u = 0.5 if j == 1 else -0.5
    b = (kappa - rho * xi) if j == 1 else kappa
    a = kappa * theta
    x = np.log(S / K)
    d = np.sqrt((rho * xi * 1j * phi - b)**2 - xi**2 * (2*u*1j*phi - phi**2))
    g = (b - rho*xi*1j*phi + d) / (b - rho*xi*1j*phi - d)
    C = (r*1j*phi*T + a/xi**2 * ((b - rho*xi*1j*phi + d)*T
         - 2*np.log((1 - g*np.exp(d*T))/(1 - g))))
    D = ((b - rho*xi*1j*phi + d)/xi**2
         * (1 - np.exp(d*T))/(1 - g*np.exp(d*T)))
    return np.exp(C + D*v0 + 1j*phi*x)

def heston_call(S, K, T, r, kappa, theta, xi, rho, v0):
    phi = _GL_PHI
    cf1 = _heston_cf(phi, S, K, T, r, kappa, theta, xi, rho, v0, j=1)
    cf2 = _heston_cf(phi, S, K, T, r, kappa, theta, xi, rho, v0, j=2)
    P1 = 0.5 + (1.0/np.pi) * np.sum(_GL_W * np.real(cf1 / (1j*phi)))
    P2 = 0.5 + (1.0/np.pi) * np.sum(_GL_W * np.real(cf2 / (1j*phi)))
    return float(max(S*P1 - K*np.exp(-r*T)*P2, 0.0))

def heston_greeks_fd(S, K, T, r, kappa, theta, xi, rho, v0, dS=0.5, dv=1e-3):
    p_up  = heston_call(S+dS, K, T, r, kappa, theta, xi, rho, v0)
    p_dn  = heston_call(S-dS, K, T, r, kappa, theta, xi, rho, v0)
    p_mid = heston_call(S,    K, T, r, kappa, theta, xi, rho, v0)
    p_vup = heston_call(S, K, T, r, kappa, theta, xi, rho, v0+dv)
    p_vdn = heston_call(S, K, T, r, kappa, theta, xi, rho, v0-dv)
    return {
        "delta": (p_up - p_dn) / (2*dS),
        "gamma": (p_up - 2*p_mid + p_dn) / dS**2,
        "vega":  (p_vup - p_vdn) / (2*dv),
    }
```

- [ ] **Step 2: 验证**

```bash
cd /home/yz2026/zhuwl2022/uni_pinn/option_pinn
python -c "
from ref_solvers import bsm_call, cev_call, heston_call
print('BSM:   ', round(bsm_call(100,100,1,0.05,0.2), 4))    # ~10.4506
print('CEV:   ', round(cev_call(100,100,1,0.05,0.25,0.5), 4))
print('Heston:', round(heston_call(100,100,1,0.05,2.0,0.04,0.3,-0.7,0.04), 4))
"
```

期望：三个值均在 8~13 之间，无报错。

- [ ] **Step 3: 提交**

```bash
git add option_pinn/ref_solvers.py
git commit -m "feat: add ref_solvers -- BSM/CEV/Heston analytical reference solutions"
git push && ssh idata2 "cd /home/yz2026/zhuwl2022/uni_pinn && git pull"
```

---

## Task 3: 编写 `eval_all.py` — 合成数据评估部分

**Files:**
- 新建: `option_pinn/eval_all.py`（本 Task 只写合成数据部分）

合成数据固定参数：BSM σ=0.2，CEV σ=0.25/β=0.5，Heston κ=2/θ=0.04/ξ=0.3/ρ=-0.7/v₀=0.04，K=100，T=1，r=0.05，S∈linspace(50,250,50)。

- [ ] **Step 1: 新建 eval_all.py（合成数据部分）**

```python
# option_pinn/eval_all.py
"""统一评估脚本：合成数据 + 真实市场数据"""
import os, sys, argparse
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))
from ref_solvers import bsm_call, bsm_greeks, cev_call, heston_call, heston_greeks_fd
from unified_pinn_v2 import ModelParams, UnifiedPINN, UnifiedNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE   = os.path.dirname(__file__)

# ── 合成数据参数 ──────────────────────────────────────────────────────────────
K, T, r = 100.0, 1.0, 0.05
S_GRID  = np.linspace(50, 250, 50)

BSM_PARAMS    = dict(sigma=0.2)
CEV_PARAMS    = dict(sigma=0.25, beta=0.5)
HESTON_PARAMS = dict(kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, v0=0.04)

def _mse_relmse(pred, ref):
    pred, ref = np.array(pred), np.array(ref)
    mse    = float(np.mean((pred - ref)**2))
    relmse = float(np.mean(((pred - ref) / (np.abs(ref) + 1e-8))**2))
    return mse, relmse

# ── 独立 PINN 加载 ────────────────────────────────────────────────────────────

def load_indep_bsm(ckpt):
    sys.path.insert(0, os.path.join(BASE, "independent"))
    from bsm_pinn import GatedPINN
    net = GatedPINN().to(DEVICE)
    net.load_state_dict(torch.load(ckpt, map_location=DEVICE)["model_state_dict"])
    net.eval()
    return net

def load_indep_cev(ckpt):
    sys.path.insert(0, os.path.join(BASE, "independent"))
    from cev_pinn import GatedPINN
    net = GatedPINN().to(DEVICE)
    net.load_state_dict(torch.load(ckpt, map_location=DEVICE)["model_state_dict"])
    net.eval()
    return net

def load_indep_heston(ckpt):
    sys.path.insert(0, os.path.join(BASE, "independent"))
    from heston_pinn import AuxNet, MainNet
    aux  = AuxNet().to(DEVICE); main = MainNet().to(DEVICE)
    ckpt_data = torch.load(ckpt, map_location=DEVICE)
    aux.load_state_dict(ckpt_data["aux_state_dict"])
    main.load_state_dict(ckpt_data["main_state_dict"])
    aux.eval(); main.eval()
    return aux, main

def load_unified(ckpt):
    from unified_pinn_v2 import ModelParams, UnifiedPINN
    # 构造与训练时相同的 param_list（52个模型）
    param_list = _build_param_list()
    pinn = UnifiedPINN(param_list, hidden=128, depth=6, device=DEVICE)
    pinn.load(ckpt)
    pinn.net.eval()
    return pinn

def load_parametric(ckpt):
    sys.path.insert(0, os.path.join(BASE, "parametric_pinn"))
    from fully_parametric_pinn import FullyParametricPINN
    pinn = FullyParametricPINN(device=DEVICE)
    pinn.load(ckpt)
    pinn.net.eval()
    return pinn

def _build_param_list():
    """复现 unified_pinn_v2 训练时的 52 个模型参数列表。"""
    params = []
    # BSM: 6 个 sigma 值
    for sigma in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35]:
        params.append(ModelParams.from_bsm(sigma=sigma))
    # CEV: 10 个 (sigma, beta) 组合
    for sigma in [0.15, 0.2, 0.25]:
        for beta in [0.3, 0.5, 0.7, 0.9]:
            params.append(ModelParams.from_cev(sigma=sigma, beta=beta))
    # Heston: 36 个组合
    for kappa in [1.0, 2.0, 3.0]:
        for theta in [0.02, 0.04, 0.06]:
            for xi in [0.2, 0.3, 0.4]:
                for rho in [-0.7, -0.5]:
                    params.append(ModelParams.from_heston(
                        kappa=kappa, theta=theta, xi=xi, rho=rho, v0=theta))
    return params

# ── 独立 PINN 推理 ────────────────────────────────────────────────────────────

def predict_indep_bsm(net, S_arr, K=100., T=1., S_max=300.):
    preds = []
    with torch.no_grad():
        for S in S_arr:
            s_t = torch.tensor([[S/S_max]], dtype=torch.float32, device=DEVICE)
            t_t = torch.tensor([[0.0]],    dtype=torch.float32, device=DEVICE)
            out = net(s_t, t_t)
            preds.append(out.item() * K)
    return np.array(preds)

def predict_indep_cev(net, S_arr, K=100., T=1., S_max=300.):
    return predict_indep_bsm(net, S_arr, K, T, S_max)

def predict_indep_heston(aux, main, S_arr, K=100., T=1.,
                          S_max=400., v0=0.04, v_max=1.0):
    preds = []
    with torch.no_grad():
        for S in S_arr:
            s_t = torch.tensor([[S/S_max]],  dtype=torch.float32, device=DEVICE)
            v_t = torch.tensor([[v0/v_max]], dtype=torch.float32, device=DEVICE)
            t_t = torch.tensor([[0.0]],      dtype=torch.float32, device=DEVICE)
            payoff = max(S - K, 0.0)
            tau_n  = 1.0
            aux_out  = aux(torch.cat([s_t, v_t, t_t], dim=1))
            main_out = main(torch.cat([s_t, v_t, t_t], dim=1))
            price = payoff + S/S_max * tau_n * (aux_out + main_out).item() * K
            preds.append(max(price, 0.0))
    return np.array(preds)

# ── Greeks via autograd ───────────────────────────────────────────────────────

def pinn_greeks_unified(pinn, p: ModelParams, S_arr):
    """用 autograd 计算 unified PINN 的 Delta、Gamma、Vega。"""
    results = []
    K, T, r = p.K, p.T, p.r
    S_max, v_max = p.S_max, p.v_max
    lam = p.to_lambda_tensor(DEVICE)
    for S_val in S_arr:
        S_t = torch.tensor([[S_val]], dtype=torch.float32,
                            device=DEVICE, requires_grad=True)
        v_t = torch.tensor([[p.v0]], dtype=torch.float32,
                            device=DEVICE, requires_grad=True)
        t_t = torch.tensor([[0.0]], dtype=torch.float32, device=DEVICE)
        V = pinn.net(S_t/S_max, v_t/v_max, t_t/T, lam, S_t, t_t, K, T, r)
        dV_dS = torch.autograd.grad(V, S_t, create_graph=True)[0]
        d2V_dS2 = torch.autograd.grad(dV_dS, S_t)[0]
        dV_dv   = torch.autograd.grad(V, v_t, retain_graph=True)[0]
        results.append({
            "delta": dV_dS.item(),
            "gamma": d2V_dS2.item(),
            "vega":  dV_dv.item(),
        })
    return results

# ── 合成数据评估主函数 ────────────────────────────────────────────────────────

def eval_synthetic(args):
    rows_price  = []
    rows_greeks = []

    # 参考解
    ref_bsm    = [bsm_call(S, K, T, r, **BSM_PARAMS)    for S in S_GRID]
    ref_cev    = [cev_call(S, K, T, r, **CEV_PARAMS)     for S in S_GRID]
    ref_heston = [heston_call(S, K, T, r, **HESTON_PARAMS) for S in S_GRID]
    ref_bsm_g  = [bsm_greeks(S, K, T, r, **BSM_PARAMS)  for S in S_GRID]
    ref_heston_g = [heston_greeks_fd(S, K, T, r, **HESTON_PARAMS) for S in S_GRID]

    def add_price_row(name, pred_bsm, pred_cev, pred_heston):
        bsm_mse,    bsm_rel    = _mse_relmse(pred_bsm,    ref_bsm)
        cev_mse,    cev_rel    = _mse_relmse(pred_cev,    ref_cev)
        heston_mse, heston_rel = _mse_relmse(pred_heston, ref_heston)
        rows_price.append({
            "model": name,
            "bsm_mse": bsm_mse, "bsm_relmse": bsm_rel,
            "cev_mse": cev_mse, "cev_relmse": cev_rel,
            "heston_mse": heston_mse, "heston_relmse": heston_rel,
        })

    def add_greeks_row(name, model_type, pred_greeks, ref_greeks):
        delta_mae = float(np.mean([abs(p["delta"]-r["delta"])
                                   for p,r in zip(pred_greeks, ref_greeks)]))
        gamma_mae = float(np.mean([abs(p["gamma"]-r["gamma"])
                                   for p,r in zip(pred_greeks, ref_greeks)]))
        vega_mae  = float(np.mean([abs(p["vega"] -r["vega"])
                                   for p,r in zip(pred_greeks, ref_greeks)]))
        rows_greeks.append({
            "model": name, "model_type": model_type,
            "delta_mae": delta_mae, "gamma_mae": gamma_mae, "vega_mae": vega_mae,
        })

    # ── 独立 PINN ──
    bsm_net = load_indep_bsm(os.path.join(BASE, "results/indep_bsm.pt"))
    cev_net = load_indep_cev(os.path.join(BASE, "results/indep_cev.pt"))
    aux, main = load_indep_heston(os.path.join(BASE, "results/indep_heston.pt"))

    pred_bsm_i    = predict_indep_bsm(bsm_net, S_GRID)
    pred_cev_i    = predict_indep_cev(cev_net, S_GRID)
    pred_heston_i = predict_indep_heston(aux, main, S_GRID, **HESTON_PARAMS)
    add_price_row("indep", pred_bsm_i, pred_cev_i, pred_heston_i)

    # ── Unified v2 ──
    unified = load_unified(os.path.join(BASE, "results/unified_v16_gl.pt"))
    p_bsm    = ModelParams.from_bsm(sigma=BSM_PARAMS["sigma"])
    p_cev    = ModelParams.from_cev(sigma=CEV_PARAMS["sigma"], beta=CEV_PARAMS["beta"])
    p_heston = ModelParams.from_heston(**HESTON_PARAMS)

    pred_bsm_u    = [unified.price(p_bsm,    S) for S in S_GRID]
    pred_cev_u    = [unified.price(p_cev,    S) for S in S_GRID]
    pred_heston_u = [unified.price(p_heston, S) for S in S_GRID]
    add_price_row("unified_v2", pred_bsm_u, pred_cev_u, pred_heston_u)

    # Greeks for unified
    greeks_bsm_u    = pinn_greeks_unified(unified, p_bsm,    S_GRID)
    greeks_heston_u = pinn_greeks_unified(unified, p_heston, S_GRID)
    add_greeks_row("unified_v2", "BSM",    greeks_bsm_u,    ref_bsm_g)
    add_greeks_row("unified_v2", "Heston", greeks_heston_u, ref_heston_g)

    # ── 参数化 PINN ──
    param_pinn = load_parametric(
        os.path.join(BASE, "parametric_pinn/results/fully_param_v1.pt"))
    pred_bsm_p    = [param_pinn.price(S=S, K=K, tau=T, r=r,
                                       sigma=BSM_PARAMS["sigma"], beta=1.0,
                                       kappa=0., theta=0., xi=0., rho=0.)
                     for S in S_GRID]
    pred_cev_p    = [param_pinn.price(S=S, K=K, tau=T, r=r,
                                       sigma=CEV_PARAMS["sigma"],
                                       beta=CEV_PARAMS["beta"],
                                       kappa=0., theta=0., xi=0., rho=0.)
                     for S in S_GRID]
    pred_heston_p = [param_pinn.price(S=S, K=K, tau=T, r=r,
                                       sigma=0., beta=1.,
                                       kappa=HESTON_PARAMS["kappa"],
                                       theta=HESTON_PARAMS["theta"],
                                       xi=HESTON_PARAMS["xi"],
                                       rho=HESTON_PARAMS["rho"])
                     for S in S_GRID]
    add_price_row("parametric", pred_bsm_p, pred_cev_p, pred_heston_p)

    # 保存
    out_dir = os.path.join(BASE, "results")
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(rows_price).to_csv(
        os.path.join(out_dir, "eval_synthetic.csv"), index=False)
    pd.DataFrame(rows_greeks).to_csv(
        os.path.join(out_dir, "eval_greeks.csv"), index=False)
    print("合成数据评估完成 →", os.path.join(out_dir, "eval_synthetic.csv"))
    print("Greeks 评估完成  →", os.path.join(out_dir, "eval_greeks.csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["synthetic", "market", "all"],
                        default="all")
    args = parser.parse_args()
    if args.mode in ("synthetic", "all"):
        eval_synthetic(args)
    # market 部分在 Task 4 添加
    if args.mode in ("market", "all"):
        eval_market(args)
```

- [ ] **Step 2: 同步到服务器并运行**

```bash
git add option_pinn/eval_all.py
git commit -m "feat: add eval_all.py -- synthetic data evaluation"
git push
ssh idata2 "cd /home/yz2026/zhuwl2022/uni_pinn && git pull"
ssh idata2 "cd /home/yz2026/zhuwl2022/uni_pinn/option_pinn && python eval_all.py --mode synthetic 2>&1 | tee results/eval_synthetic.log"
```

- [ ] **Step 3: 检查输出**

```bash
ssh idata2 "cat /home/yz2026/zhuwl2022/uni_pinn/option_pinn/results/eval_synthetic.csv"
ssh idata2 "cat /home/yz2026/zhuwl2022/uni_pinn/option_pinn/results/eval_greeks.csv"
```

期望：两个 CSV 均有数据行，无 NaN，unified_v2 的 bsm_relmse < 0.01。

---

## Task 4: fine-tune Heston 分支 — `finetune_heston.py`

**Files:**
- 新建: `option_pinn/finetune_heston.py`

在 SPY 市场数据上对 unified_v2 的 Heston 分支做域适应。加载 `unified_v16_gl.pt`，用市场 mid-price 作为数据锚点，保留 PDE loss，lr 降低 10 倍，保存为 `results/unified_v2_ft.pt`。

- [ ] **Step 1: 新建 finetune_heston.py**

```python
# option_pinn/finetune_heston.py
"""Fine-tune unified_v2 Heston 分支 on SPY market data."""
import os, sys
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))
from unified_pinn_v2 import ModelParams, UnifiedPINN, UnifiedNet, unified_pde_residual
from ref_solvers import bsm_call

BASE   = os.path.dirname(__file__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── 从 SPY 数据中提取 Heston 适用合约 ────────────────────────────────────────

def load_spy_heston_contracts(csv_path, moneyness_lo=0.85, moneyness_hi=1.15,
                               min_tau=0.1, max_tau=2.0):
    """加载 SPY 数据，筛选适合 Heston 定价的合约（ATM 附近，中等到期）。"""
    df = pd.read_csv(csv_path)
    # 列名适配（CBOE 格式）
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns={"underlying_last": "S", "strike": "K",
                             "expiration": "expiry", "bid": "bid", "ask": "ask"})
    df["mid"] = (df["bid"] + df["ask"]) / 2.0
    df = df[(df["bid"] > 0) & (df["ask"] > 0)]
    df = df[df["type"].str.lower() == "call"]

    # 计算 tau（年化）
    import datetime
    df["quote_date"] = pd.to_datetime(df["quote_date"])
    df["expiry"]     = pd.to_datetime(df["expiry"])
    df["tau"] = (df["expiry"] - df["quote_date"]).dt.days / 365.0
    df = df[(df["tau"] >= min_tau) & (df["tau"] <= max_tau)]

    # moneyness 过滤
    df["moneyness"] = df["S"] / df["K"]
    df = df[(df["moneyness"] >= moneyness_lo) & (df["moneyness"] <= moneyness_hi)]

    # 用 BSM 反推隐含波动率（简单估计，用于 Heston 参数初始化）
    df = df.reset_index(drop=True)
    return df[["S", "K", "tau", "mid"]].values  # (N, 4)

# ── Fine-tune ─────────────────────────────────────────────────────────────────

def finetune(ckpt_in, ckpt_out, spy_csv, epochs=3000, lr=1e-4,
             w_pde=0.1, w_data=1.0, batch_size=256):
    # 构造与训练时相同的 param_list
    from eval_all import _build_param_list
    param_list = _build_param_list()

    pinn = UnifiedPINN(param_list, hidden=128, depth=6, lr=lr, device=DEVICE)
    pinn.load(ckpt_in)
    pinn.net.train()

    # 重新设置 optimizer（lr 降低 10 倍）
    optimizer = torch.optim.Adam(pinn.net.parameters(), lr=lr)

    # 加载 SPY 数据
    contracts = load_spy_heston_contracts(spy_csv)
    print(f"Fine-tune 合约数: {len(contracts)}")

    # 使用 Heston 默认参数（κ=2, θ=0.04, ξ=0.3, ρ=-0.7）作为 lambda
    p_heston = ModelParams.from_heston(kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, v0=0.04)
    lam_h = p_heston.to_lambda_tensor(DEVICE)

    K_const, T_const, r_const = 100.0, 1.0, 0.05
    S_max, v_max = 300.0, 1.0

    from tqdm import tqdm
    for epoch in tqdm(range(1, epochs + 1), desc="FineTune"):
        optimizer.zero_grad()

        # 随机采样一批市场合约
        idx = np.random.choice(len(contracts), min(batch_size, len(contracts)),
                               replace=False)
        batch = contracts[idx]
        S_b   = torch.tensor(batch[:, 0:1], dtype=torch.float32, device=DEVICE)
        K_b   = torch.tensor(batch[:, 1:2], dtype=torch.float32, device=DEVICE)
        tau_b = torch.tensor(batch[:, 2:3], dtype=torch.float32, device=DEVICE)
        V_b   = torch.tensor(batch[:, 3:4], dtype=torch.float32, device=DEVICE)
        t_b   = T_const - tau_b  # t = T - tau
        v_b   = torch.full_like(S_b, p_heston.v0)
        lam_b = lam_h.expand(len(idx), -1)

        # Data loss（市场 mid-price 锚点）
        pred = pinn.net(S_b/S_max, v_b/v_max, t_b/T_const,
                        lam_b, S_b, t_b, K_const, T_const, r_const)
        loss_data = torch.mean(((pred - V_b) / (V_b.abs() + K_const * 0.1))**2)

        # PDE loss（防止物理约束退化）
        n_pde = 512
        S_c = torch.FloatTensor(n_pde, 1).uniform_(0.01, S_max).to(DEVICE)
        v_c = torch.FloatTensor(n_pde, 1).uniform_(1e-4, v_max).to(DEVICE)
        t_c = torch.FloatTensor(n_pde, 1).uniform_(0.0, T_const * 0.999).to(DEVICE)
        lam_c = lam_h.expand(n_pde, -1)
        res = unified_pde_residual(pinn.net, S_c, v_c, t_c, lam_c,
                                   K_const, T_const, r_const, S_max, v_max)
        loss_pde = torch.mean(res**2)

        loss = w_pde * loss_pde + w_data * loss_data
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pinn.net.parameters(), max_norm=1.0)
        optimizer.step()

        if epoch % 500 == 0:
            print(f"Epoch {epoch}: loss={loss.item():.3e} "
                  f"pde={loss_pde.item():.3e} data={loss_data.item():.3e}")

    pinn.save(ckpt_out)
    print(f"Fine-tuned model saved → {ckpt_out}")


if __name__ == "__main__":
    finetune(
        ckpt_in  = os.path.join(BASE, "results/unified_v16_gl.pt"),
        ckpt_out = os.path.join(BASE, "results/unified_v2_ft.pt"),
        spy_csv  = os.path.join(BASE, "data/spy_quotedata.csv"),
    )
```

- [ ] **Step 2: 同步并在服务器运行**

```bash
git add option_pinn/finetune_heston.py
git commit -m "feat: add finetune_heston.py -- domain adaptation on SPY market data"
git push
ssh idata2 "cd /home/yz2026/zhuwl2022/uni_pinn && git pull"
ssh idata2 "cd /home/yz2026/zhuwl2022/uni_pinn/option_pinn && python finetune_heston.py 2>&1 | tee results/finetune.log"
```

- [ ] **Step 3: 确认 checkpoint 生成**

```bash
ssh idata2 "ls -lh /home/yz2026/zhuwl2022/uni_pinn/option_pinn/results/unified_v2_ft.pt"
```

期望：文件存在，大小与 `unified_v16_gl.pt` 相近（约 1-5 MB）。

---

## Task 5: 扩展 `eval_all.py` — 真实市场数据评估

**Files:**
- 修改: `option_pinn/eval_all.py`（添加 `eval_market` 函数）

- [ ] **Step 1: 在 eval_all.py 中添加 eval_market 函数**

在 `if __name__ == "__main__":` 之前插入：

```python
# ── 真实市场数据评估 ──────────────────────────────────────────────────────────

def _load_spy(csv_path, moneyness_lo=0.7, moneyness_hi=1.3):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns={"underlying_last": "S", "strike": "K",
                             "expiration": "expiry"})
    df["mid"] = (df["bid"] + df["ask"]) / 2.0
    df = df[(df["bid"] > 0) & (df["ask"] > 0)]
    df = df[df["type"].str.lower() == "call"]
    df["quote_date"] = pd.to_datetime(df["quote_date"])
    df["expiry"]     = pd.to_datetime(df["expiry"])
    df["tau"] = (df["expiry"] - df["quote_date"]).dt.days / 365.0
    df = df[df["tau"] > 0.05]
    df["moneyness"] = df["S"] / df["K"]
    df = df[(df["moneyness"] >= moneyness_lo) & (df["moneyness"] <= moneyness_hi)]
    return df.reset_index(drop=True)

def _calibrate_bsm_iv(row):
    """用二分法反推 BSM 隐含波动率。"""
    from scipy.optimize import brentq
    S, K, tau, mid = row["S"], row["K"], row["tau"], row["mid"]
    r = 0.05
    intrinsic = max(S - K * np.exp(-r * tau), 0.0)
    if mid <= intrinsic + 1e-6:
        return 0.2  # fallback
    try:
        iv = brentq(lambda sig: bsm_call(S, K, tau, r, sig) - mid,
                    1e-4, 5.0, xtol=1e-6, maxiter=100)
        return iv
    except Exception:
        return 0.2

def eval_market(args):
    spy_csv = os.path.join(BASE, "data/spy_quotedata.csv")
    df = _load_spy(spy_csv)
    print(f"市场合约数: {len(df)}")

    # 为每条合约反推 BSM IV，用于 CEV/BSM 定价
    df["iv"] = df.apply(_calibrate_bsm_iv, axis=1)

    # Heston 固定参数（训练时参数）
    h = dict(kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, v0=0.04)

    rows = []

    def score_model(name, pred_fn):
        preds = []
        for _, row in df.iterrows():
            try:
                p = pred_fn(row)
            except Exception:
                p = np.nan
            preds.append(p)
        preds = np.array(preds)
        ref   = df["mid"].values
        mask  = ~np.isnan(preds)
        mse    = float(np.mean((preds[mask] - ref[mask])**2))
        relmse = float(np.mean(((preds[mask] - ref[mask])
                                / (np.abs(ref[mask]) + 1e-8))**2))
        rows.append({"model": name, "mse": mse, "relmse": relmse,
                     "n_contracts": int(mask.sum())})

    # BSM 解析解（基准）
    score_model("bsm_analytical",
                lambda row: bsm_call(row["S"], row["K"], row["tau"], 0.05, row["iv"]))

    # Heston GL 解析解（基准）
    score_model("heston_analytical",
                lambda row: heston_call(row["S"], row["K"], row["tau"], 0.05, **h))

    # 独立 PINN
    bsm_net = load_indep_bsm(os.path.join(BASE, "results/indep_bsm.pt"))
    score_model("indep_bsm",
                lambda row: predict_indep_bsm(bsm_net, [row["S"]])[0])

    cev_net = load_indep_cev(os.path.join(BASE, "results/indep_cev.pt"))
    score_model("indep_cev",
                lambda row: predict_indep_cev(cev_net, [row["S"]])[0])

    aux, main = load_indep_heston(os.path.join(BASE, "results/indep_heston.pt"))
    score_model("indep_heston",
                lambda row: predict_indep_heston(
                    aux, main, [row["S"]], v0=h["v0"])[0])

    # Unified v2
    unified = load_unified(os.path.join(BASE, "results/unified_v16_gl.pt"))
    p_h = ModelParams.from_heston(**h)
    score_model("unified_v2",
                lambda row: unified.price(p_h, row["S"], t=0.0))

    # Unified v2 fine-tuned
    ft_ckpt = os.path.join(BASE, "results/unified_v2_ft.pt")
    if os.path.exists(ft_ckpt):
        unified_ft = load_unified(ft_ckpt)
        score_model("unified_v2_ft",
                    lambda row: unified_ft.price(p_h, row["S"], t=0.0))

    # 参数化 PINN
    param_pinn = load_parametric(
        os.path.join(BASE, "parametric_pinn/results/fully_param_v1.pt"))
    score_model("parametric",
                lambda row: param_pinn.price(
                    S=row["S"], K=row["K"], tau=row["tau"], r=0.05,
                    sigma=0., beta=1.,
                    kappa=h["kappa"], theta=h["theta"],
                    xi=h["xi"], rho=h["rho"]))

    out_path = os.path.join(BASE, "results/eval_market.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print("市场数据评估完成 →", out_path)
```

- [ ] **Step 2: 同步并运行**

```bash
git add option_pinn/eval_all.py
git commit -m "feat: add eval_market to eval_all.py"
git push
ssh idata2 "cd /home/yz2026/zhuwl2022/uni_pinn && git pull"
ssh idata2 "cd /home/yz2026/zhuwl2022/uni_pinn/option_pinn && python eval_all.py --mode market 2>&1 | tee results/eval_market.log"
```

- [ ] **Step 3: 检查输出**

```bash
ssh idata2 "cat /home/yz2026/zhuwl2022/uni_pinn/option_pinn/results/eval_market.csv"
```

期望：每个模型一行，mse 和 relmse 均为正数，无 NaN。

---

## Task 6: 同步结果到本地并提交

- [ ] **Step 1: 拉取服务器结果**

```bash
scp idata2:/home/yz2026/zhuwl2022/uni_pinn/option_pinn/results/eval_synthetic.csv "D:\Willing\Study\Course\final\sym\paper\uni_pinn\option_pinn\results\"
scp idata2:/home/yz2026/zhuwl2022/uni_pinn/option_pinn/results/eval_greeks.csv "D:\Willing\Study\Course\final\sym\paper\uni_pinn\option_pinn\results\"
scp idata2:/home/yz2026/zhuwl2022/uni_pinn/option_pinn/results/eval_market.csv "D:\Willing\Study\Course\final\sym\paper\uni_pinn\option_pinn\results\"
```

- [ ] **Step 2: 提交结果**

```bash
cd D:\Willing\Study\Course\final\sym\paper\uni_pinn
git add option_pinn/results/eval_synthetic.csv option_pinn/results/eval_greeks.csv option_pinn/results/eval_market.csv
git commit -m "results: unified eval -- synthetic/market/greeks for all 5 models"
git push
```

---


