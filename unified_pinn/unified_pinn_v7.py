"""
unified_pinn.py — 参数化统一PINN，一个网络同时覆盖BSM / CEV / Heston

统一PDE算子（2D状态空间 S, v）：

  F[V] = V_t
       + 0.5 * a(S,v;lam) * V_SS
       + b(S,v;lam)       * V_Sv
       + 0.5 * c(v;lam)   * V_vv
       + r*S               * V_S
       + d(v;lam)          * V_v
       - r*V  = 0

系数由参数向量 lam = (sigma, beta, kappa, theta, xi, rho) 控制：

  模型    a(S,v)          b(S,v)          c(v)       d(v)
  BSM     sigma^2*S^2     0               0          0
  CEV     sigma^2*S^2b    0               0          0
  Heston  v*S^2           rho*xi*v*S      xi^2*v     kappa*(theta-v)

网络输入（归一化后）：
  [S/S_max, v/v_max, t/T, sigma, beta, kappa, theta, xi, rho]  共9维

BS归一化输出（解决量级问题的核心）：
  V = V_bs(S, tau; sigma_eff) * (1 + net(x))

  V_bs 是用等效波动率 sigma_eff 计算的 BS 解析解，作为归一化因子。
  net 学习相对修正量（量级约为 0~0.1），初始化时 net≈0，V≈V_bs。

  对 BSM：V_bs 精确，net 趋近于 0
  对 CEV：V_bs 是 beta=1 近似，net 学习弹性指数修正
  对 Heston：V_bs 是确定性波动率近似，net 学习随机波动率修正

  终值条件通过软约束 L_ic 在 t=T 处施加（V_bs(tau=0)=payoff 已近似满足）。

软掩码（Soft Mask）：
  mask = tanh(xi/0.05)^2
  a = (1-mask)*sigma^2*S^(2*beta) + mask*v*S^2
  当 xi->0 时 mask->0，自动退化为 BSM/CEV；xi 大时退化为 Heston。
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# 参数容器
# ---------------------------------------------------------------------------

@dataclass
class ModelParams:
    """期权与模型参数。通过 from_bsm / from_cev / from_heston 构造。"""
    K:     float = 100.0
    T:     float = 1.0
    r:     float = 0.05
    sigma: float = 0.2
    beta:  float = 1.0
    kappa: float = 0.0
    theta: float = 0.0
    xi:    float = 0.0
    rho:   float = 0.0
    v0:    float = 0.04
    S_max: float = 300.0
    v_max: float = 1.0

    @classmethod
    def from_bsm(cls, K=100., T=1., r=0.05, sigma=0.2, S_max=300.):
        return cls(K=K, T=T, r=r, sigma=sigma, beta=1.0,
                   kappa=0., theta=0., xi=0., rho=0.,
                   v0=sigma**2, S_max=S_max)

    @classmethod
    def from_cev(cls, K=100., T=1., r=0.05, sigma=0.2, beta=0.5, S_max=300.):
        return cls(K=K, T=T, r=r, sigma=sigma, beta=beta,
                   kappa=0., theta=0., xi=0., rho=0.,
                   v0=sigma**2, S_max=S_max)

    @classmethod
    def from_heston(cls, K=100., T=1., r=0.05,
                    kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, v0=0.04,
                    S_max=300., v_max=1.0):
        return cls(K=K, T=T, r=r, sigma=0., beta=1.,
                   kappa=kappa, theta=theta, xi=xi, rho=rho,
                   v0=v0, S_max=S_max, v_max=v_max)

    def to_lambda_tensor(self, device):
        """返回 shape=(1,6) 的参数张量 [sigma, beta, kappa, theta, xi, rho]。"""
        return torch.tensor(
            [[self.sigma, self.beta, self.kappa, self.theta, self.xi, self.rho]],
            dtype=torch.float32, device=device
        )


# ---------------------------------------------------------------------------
# BS 解析解（向量化）
# ---------------------------------------------------------------------------

def _bs_call(S, K, tau, r, sigma):
    """
    欧式看涨 BS 解析解，全部为 torch tensor，支持批量计算。
    tau = T - t（剩余时间）。当 tau<=0 时退化为 payoff。
    """
    eps   = 1e-8
    tau   = torch.clamp(tau,   min=eps)
    sigma = torch.clamp(sigma, min=eps)
    sqt   = sigma * torch.sqrt(tau)
    d1    = (torch.log(S / K) + (r + 0.5 * sigma**2) * tau) / sqt
    d2    = d1 - sqt
    from torch.distributions import Normal
    nd    = Normal(0., 1.)
    call  = S * nd.cdf(d1) - K * torch.exp(-r * tau) * nd.cdf(d2)
    return torch.clamp(call, min=0.0)


# ---------------------------------------------------------------------------
# 网络架构
# ---------------------------------------------------------------------------

class UnifiedNet(nn.Module):
    """
    输入维度 = 9：[S_n, v_n, t_n, sigma, beta, kappa, theta, xi, rho]

    BS归一化输出：V = V_bs(S, tau; sigma_eff) * (1 + net(x))

    net 学习相对修正量（量级约 0~0.1），初始化时 net≈0，V≈V_bs。
    梯度信号充足，不会陷入低输出局部最优。
    """

    def __init__(self, hidden: int = 128, depth: int = 6):
        super().__init__()
        layers = [nn.Linear(9, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, S_n, v_n, t_n, lam, S_raw, t_raw, K, T, r):
        """
        S_n, v_n, t_n : (N,1) 归一化状态
        lam           : (N,6) [sigma, beta, kappa, theta, xi, rho]
        S_raw, t_raw  : (N,1) 原始值
        K, T, r       : 标量
        """
        sigma = lam[:, 0:1]
        xi    = lam[:, 4:5]

        # 等效波动率：BSM/CEV 用 sigma，Heston 用 sqrt(v)
        mask      = torch.tanh(xi / 0.05) ** 2
        v_approx  = torch.clamp(v_n, min=1e-6)   # v_n ≈ v（v_max=1 时精确）
        sigma_eff = (1 - mask) * sigma + mask * torch.sqrt(v_approx)

        tau   = torch.clamp(T - t_raw, min=1e-4)
        V_bs  = _bs_call(S_raw, K, tau, r, sigma_eff)

        x   = torch.cat([S_n, v_n, t_n, lam], dim=1)
        raw = self.net(x)
        return V_bs * (1.0 + raw)


# ---------------------------------------------------------------------------
# 统一PDE算子
# ---------------------------------------------------------------------------

def unified_pde_residual(net, S, v, t, lam, K, T, r, S_max, v_max):
    """
    计算统一PDE残差。
    lam shape=(N,6): [sigma, beta, kappa, theta, xi, rho]
    软掩码：xi->0 时交叉项和v方向扩散自然归零，退化为BSM/CEV。
    """
    S.requires_grad_(True)
    v.requires_grad_(True)
    t.requires_grad_(True)

    S_n = S / S_max
    v_n = v / v_max
    t_n = t / T

    V = net(S_n, v_n, t_n, lam, S, t, K, T, r)

    ones = torch.ones_like(V)
    V_t  = torch.autograd.grad(V, t,  grad_outputs=ones, create_graph=True)[0]
    V_S  = torch.autograd.grad(V, S,  grad_outputs=ones, create_graph=True)[0]
    V_v  = torch.autograd.grad(V, v,  grad_outputs=ones, create_graph=True)[0]
    V_SS = torch.autograd.grad(V_S, S, grad_outputs=torch.ones_like(V_S),
                               create_graph=True)[0]
    V_vv = torch.autograd.grad(V_v, v, grad_outputs=torch.ones_like(V_v),
                               create_graph=True)[0]
    V_Sv = torch.autograd.grad(V_S, v, grad_outputs=torch.ones_like(V_S),
                               create_graph=True)[0]

    sigma = lam[:, 0:1]
    beta  = lam[:, 1:2]
    kappa = lam[:, 2:3]
    theta = lam[:, 3:4]
    xi    = lam[:, 4:5]
    rho   = lam[:, 5:6]

    mask = torch.tanh(xi / 0.05) ** 2
    a = (1 - mask) * sigma**2 * S**(2 * beta) + mask * v * S**2
    b = rho * xi * v * S
    c = xi**2 * v
    d = kappa * (theta - v)

    residual = (V_t
                + 0.5 * a * V_SS
                + b * V_Sv
                + 0.5 * c * V_vv
                + r * S * V_S
                + d * V_v
                - r * V)
    # 相对残差：除以 V 量级，避免 ATM 区域（V~10）主导 loss
    return residual / (V.detach().abs() + 1.0)


# ---------------------------------------------------------------------------
# 训练器
# ---------------------------------------------------------------------------

class UnifiedPINN:
    """
    统一PINN训练器。
    训练时从三个模型的参数分布中混合采样配点，
    单个网络同时学习所有模型的解映射。
    """

    def __init__(self,
                 param_list: list,
                 hidden: int = 128,
                 depth: int = 6,
                 device=None):
        self.param_list = param_list
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.net = UnifiedNet(hidden=hidden, depth=depth).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        self.scheduler = None

    def _to(self, x):
        return x.to(self.device)

    def _sample_batch(self, n_per_model: int = 5000):
        """从每个模型均匀采样配点，拼接成混合批次。"""
        S_list, v_list, t_list, lam_list = [], [], [], []
        for p in self.param_list:
            n = n_per_model
            S   = torch.FloatTensor(n, 1).uniform_(0.01, p.S_max)
            v   = torch.FloatTensor(n, 1).uniform_(1e-4, p.v_max)
            t   = torch.FloatTensor(n, 1).uniform_(0.0, p.T * 0.999)
            lam = p.to_lambda_tensor(torch.device("cpu")).expand(n, -1).clone()
            S_list.append(S); v_list.append(v)
            t_list.append(t); lam_list.append(lam)
        return (self._to(torch.cat(S_list)),
                self._to(torch.cat(v_list)),
                self._to(torch.cat(t_list)),
                self._to(torch.cat(lam_list)))

    def _ic_loss(self, n_per_model: int = 500):
        """终值条件软约束：t=T 时 V = payoff(S)，用相对误差避免量级失衡。"""
        loss = torch.tensor(0.0, device=self.device)
        for p in self.param_list:
            n = n_per_model
            r, K, T, S_max, v_max = p.r, p.K, p.T, p.S_max, p.v_max
            lam = p.to_lambda_tensor(self.device).expand(n, -1)
            S_ic = self._to(torch.FloatTensor(n, 1).uniform_(0.01, S_max))
            v_ic = self._to(torch.FloatTensor(n, 1).uniform_(1e-4, v_max))
            t_ic = self._to(torch.full((n, 1), T))
            pred   = self.net(S_ic/S_max, v_ic/v_max, t_ic/T, lam, S_ic, t_ic, K, T, r)
            payoff = torch.clamp(S_ic - K, min=0.)
            # 相对误差：除以 payoff 量级，使 OTM/ATM/ITM 贡献均衡
            rel_err = (pred - payoff) / (payoff + K * 0.1)
            loss = loss + torch.mean(rel_err**2)
        return loss / len(self.param_list)

    def _boundary_loss(self, n_per_model: int = 500):
        """空间边界条件：S=0 时 V=0，S=S_max 时 V=S_max-K*exp(-r*(T-t))。"""
        loss = torch.tensor(0.0, device=self.device)
        for p in self.param_list:
            n = n_per_model
            r, K, T, S_max, v_max = p.r, p.K, p.T, p.S_max, p.v_max
            lam = p.to_lambda_tensor(self.device).expand(n, -1)

            # S=0: V_bs(0)=0 自动满足，仍保留软约束作为正则
            S0  = self._to(torch.zeros(n, 1))
            v0b = self._to(torch.FloatTensor(n, 1).uniform_(1e-4, v_max))
            t0  = self._to(torch.FloatTensor(n, 1).uniform_(0, T))
            pred0 = self.net(S0/S_max, v0b/v_max, t0/T, lam, S0, t0, K, T, r)
            loss  = loss + torch.mean(pred0**2)

            # S=S_max: 相对误差，避免大 S 值主导 loss
            Sm    = self._to(torch.full((n, 1), S_max))
            vmb   = self._to(torch.FloatTensor(n, 1).uniform_(1e-4, v_max))
            tm    = self._to(torch.FloatTensor(n, 1).uniform_(0, T))
            Vm    = S_max - K * torch.exp(-r * (T - tm))
            predm = self.net(Sm/S_max, vmb/v_max, tm/T, lam, Sm, tm, K, T, r)
            rel_err = (predm - Vm) / (Vm + 1.0)
            loss  = loss + torch.mean(rel_err**2)

        return loss / len(self.param_list)

    def train(self, epochs: int = 30000, n_per_model: int = 5000,
              w_pde: float = 1.0, w_bc: float = 5.0, w_ic: float = 5.0,
              log_every: int = 500):
        from tqdm import tqdm
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=1e-5
        )
        p0 = self.param_list[0]
        history = []
        pbar = tqdm(range(1, epochs + 1), desc="UnifiedPINN", dynamic_ncols=True)
        for epoch in pbar:
            self.optimizer.zero_grad()

            S_c, v_c, t_c, lam_c = self._sample_batch(n_per_model)
            res      = unified_pde_residual(
                self.net, S_c, v_c, t_c, lam_c,
                p0.K, p0.T, p0.r, p0.S_max, p0.v_max
            )
            loss_pde = torch.mean(res**2)
            loss_bc  = self._boundary_loss()
            loss_ic  = self._ic_loss()
            loss     = w_pde * loss_pde + w_bc * loss_bc + w_ic * loss_ic

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()

            if epoch % log_every == 0:
                history.append({
                    "epoch": epoch,
                    "loss":  loss.item(),
                    "pde":   loss_pde.item(),
                    "bc":    loss_bc.item(),
                    "ic":    loss_ic.item(),
                })
                pbar.set_postfix(
                    loss=f"{loss.item():.3e}",
                    pde=f"{loss_pde.item():.3e}",
                    ic=f"{loss_ic.item():.3e}",
                )
        return history

    def price(self, p: ModelParams, S: float,
              v: Optional[float] = None, t: float = 0.0) -> float:
        """用统一网络对给定参数集推断期权价格。"""
        if v is None:
            v = p.v0
        self.net.eval()
        with torch.no_grad():
            S_t   = self._to(torch.tensor([[S]], dtype=torch.float32))
            v_t   = self._to(torch.tensor([[v]], dtype=torch.float32))
            t_t   = self._to(torch.tensor([[t]], dtype=torch.float32))
            lam_t = p.to_lambda_tensor(self.device)
            out   = self.net(
                S_t/p.S_max, v_t/p.v_max, t_t/p.T,
                lam_t, S_t, t_t, p.K, p.T, p.r
            )
        return out.item()

    def save(self, path: str):
        torch.save({"state_dict": self.net.state_dict()}, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ckpt["state_dict"])
