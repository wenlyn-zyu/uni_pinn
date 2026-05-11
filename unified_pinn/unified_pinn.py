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

BS归一化输出（加法参数化）：
  V = V_bs(S, tau; sigma_eff) + K * net(x)

  V_bs 是用等效波动率 sigma_eff 计算的 BS 解析解，作为基线。
  net 学习绝对修正量（以 K 为单位，量级约为 0~0.1），初始化时 net≈0，V≈V_bs。

  加法参数化的优势：当 V_bs→0（深度 OTM）时，net 仍可输出非零修正，
  解决了乘法参数化 V=V_bs*(1+net) 在 OTM 区域退化为 0 的问题。

  对 BSM：V_bs 精确，net 趋近于 0
  对 CEV：V_bs 是 beta=1 近似，net 学习弹性指数修正（OTM 区域尤其重要）
  对 Heston：V_bs 是确定性波动率近似，net 学习随机波动率修正

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
    K:           float = 100.0
    T:           float = 1.0
    r:           float = 0.05
    sigma:       float = 0.2
    beta:        float = 1.0
    kappa:       float = 0.0
    theta:       float = 0.0
    xi:          float = 0.0
    rho:         float = 0.0
    v0:          float = 0.04
    S_max:       float = 300.0
    v_max:       float = 1.0
    option_type: str   = "call"   # "call" 或 "put"

    @classmethod
    def from_bsm(cls, K=100., T=1., r=0.05, sigma=0.2, S_max=300., option_type="call"):
        return cls(K=K, T=T, r=r, sigma=sigma, beta=1.0,
                   kappa=0., theta=0., xi=0., rho=0.,
                   v0=sigma**2, S_max=S_max, option_type=option_type)

    @classmethod
    def from_cev(cls, K=100., T=1., r=0.05, sigma=0.2, beta=0.5, S_max=300., option_type="call"):
        return cls(K=K, T=T, r=r, sigma=sigma, beta=beta,
                   kappa=0., theta=0., xi=0., rho=0.,
                   v0=sigma**2, S_max=S_max, option_type=option_type)

    @classmethod
    def from_heston(cls, K=100., T=1., r=0.05,
                    kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, v0=0.04,
                    S_max=300., v_max=1.0, option_type="call"):
        return cls(K=K, T=T, r=r, sigma=0., beta=1.,
                   kappa=kappa, theta=theta, xi=xi, rho=rho,
                   v0=v0, S_max=S_max, v_max=v_max, option_type=option_type)

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
        last = nn.Linear(hidden, 1)
        nn.init.normal_(last.weight, std=0.001)
        nn.init.zeros_(last.bias)
        layers.append(last)
        self.net = nn.Sequential(*layers)

    def forward(self, S_n, v_n, t_n, lam, S_raw, t_raw, K, T, r):
        """
        加法参数化：V = V_bs + K * net(x)
        小初始化（std=0.001）确保训练开始时 K*net ~0.1 << V_bs ~10。
        OTM 区域 V_bs→0 时 net 仍可输出非零修正，解决乘法参数化的退化问题。
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
        # 期权价格非负：clamp 保证 V >= 0，不影响 ATM/ITM 区域的梯度
        return torch.clamp(V_bs + K * raw, min=0.0)


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
                 lr: float = 1e-3,
                 ref_data: dict = None,
                 device=None):
        """
        ref_data: 可选的参考解字典，格式为
          { param_idx: (S_arr, v_arr, t_arr, V_arr) }
          其中 param_idx 是 param_list 中的索引。
          用于在训练时施加数据驱动约束，解决纯 PDE 约束不唯一的问题。
        """
        self.param_list = param_list
        self.ref_data   = ref_data or {}
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.net = UnifiedNet(hidden=hidden, depth=depth).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.scheduler = None

    def _to(self, x):
        return x.to(self.device)

    def _sample_batch(self, n_per_model: int = 5000):
        """从每个模型均匀采样配点，并在 OTM 过渡区 [0.7K, K] 加密 30%。"""
        S_list, v_list, t_list, lam_list = [], [], [], []
        for p in self.param_list:
            n_base = int(n_per_model * 0.7)
            n_otm  = n_per_model - n_base
            S_base = torch.FloatTensor(n_base, 1).uniform_(0.01, p.S_max)
            S_otm  = torch.FloatTensor(n_otm,  1).uniform_(0.7 * p.K, p.K)
            S   = torch.cat([S_base, S_otm])
            v   = torch.FloatTensor(n_per_model, 1).uniform_(1e-4, p.v_max)
            t   = torch.FloatTensor(n_per_model, 1).uniform_(0.0, p.T * 0.999)
            lam = p.to_lambda_tensor(torch.device("cpu")).expand(n_per_model, -1).clone()
            S_list.append(S); v_list.append(v)
            t_list.append(t); lam_list.append(lam)
        return (self._to(torch.cat(S_list)),
                self._to(torch.cat(v_list)),
                self._to(torch.cat(t_list)),
                self._to(torch.cat(lam_list)))

    def _ic_loss(self, n_per_model: int = 500):
        """终值条件软约束：t=T 时 V = payoff(S)，支持看涨/看跌。"""
        loss = torch.tensor(0.0, device=self.device)
        for p in self.param_list:
            n = n_per_model
            r, K, T, S_max, v_max = p.r, p.K, p.T, p.S_max, p.v_max
            lam = p.to_lambda_tensor(self.device).expand(n, -1)
            S_ic = self._to(torch.FloatTensor(n, 1).uniform_(0.01, S_max))
            v_ic = self._to(torch.FloatTensor(n, 1).uniform_(1e-4, v_max))
            t_ic = self._to(torch.full((n, 1), T))
            pred = self.net(S_ic/S_max, v_ic/v_max, t_ic/T, lam, S_ic, t_ic, K, T, r)
            if p.option_type == "put":
                payoff = torch.clamp(K - S_ic, min=0.)
            else:
                payoff = torch.clamp(S_ic - K, min=0.)
            rel_err = (pred - payoff) / (payoff + K * 0.1)
            loss = loss + torch.mean(rel_err**2)
        return loss / len(self.param_list)

    def _data_loss(self):
        """数据驱动约束：用预计算的参考解作为锚点，解决 PDE 解不唯一的问题。"""
        if not self.ref_data:
            return torch.tensor(0.0, device=self.device)
        loss = torch.tensor(0.0, device=self.device)
        count = 0
        for idx, (S_arr, v_arr, t_arr, V_arr) in self.ref_data.items():
            p = self.param_list[idx]
            n = len(S_arr)
            S_t = self._to(torch.tensor(S_arr, dtype=torch.float32).reshape(-1, 1))
            v_t = self._to(torch.tensor(v_arr, dtype=torch.float32).reshape(-1, 1))
            t_t = self._to(torch.tensor(t_arr, dtype=torch.float32).reshape(-1, 1))
            V_t = self._to(torch.tensor(V_arr, dtype=torch.float32).reshape(-1, 1))
            lam = p.to_lambda_tensor(self.device).expand(n, -1)
            pred = self.net(S_t/p.S_max, v_t/p.v_max, t_t/p.T, lam, S_t, t_t, p.K, p.T, p.r)
            # 相对误差
            rel_err = (pred - V_t) / (V_t.abs() + p.K * 0.1)
            loss = loss + torch.mean(rel_err**2)
            count += 1
        return loss / max(count, 1)

    def _boundary_loss(self, n_per_model: int = 500):
        """空间边界条件，支持看涨/看跌：
        看涨：S=0 → V=0；S=S_max → V=S_max - K·e^{-r(T-t)}
        看跌：S=0 → V=K·e^{-r(T-t)}；S=S_max → V=0
        """
        loss = torch.tensor(0.0, device=self.device)
        for p in self.param_list:
            n = n_per_model
            r, K, T, S_max, v_max = p.r, p.K, p.T, p.S_max, p.v_max
            lam = p.to_lambda_tensor(self.device).expand(n, -1)
            tm  = self._to(torch.FloatTensor(n, 1).uniform_(0, T))
            disc = torch.exp(-r * (T - tm))   # 贴现因子

            # S=0 边界
            S0   = self._to(torch.zeros(n, 1))
            v0b  = self._to(torch.FloatTensor(n, 1).uniform_(1e-4, v_max))
            pred0 = self.net(S0/S_max, v0b/v_max, tm/T, lam, S0, tm, K, T, r)
            if p.option_type == "put":
                V0 = K * disc
                loss = loss + torch.mean(((pred0 - V0) / (V0 + 1.0))**2)
            else:
                loss = loss + torch.mean(pred0**2)

            # S=S_max 边界
            Sm   = self._to(torch.full((n, 1), S_max))
            vmb  = self._to(torch.FloatTensor(n, 1).uniform_(1e-4, v_max))
            predm = self.net(Sm/S_max, vmb/v_max, tm/T, lam, Sm, tm, K, T, r)
            if p.option_type == "put":
                loss = loss + torch.mean(predm**2)
            else:
                Vm = S_max - K * disc
                loss = loss + torch.mean(((predm - Vm) / (Vm + 1.0))**2)

        return loss / len(self.param_list)

    def train(self, epochs: int = 30000, n_per_model: int = 5000,
              w_pde: float = 1.0, w_bc: float = 50.0, w_ic: float = 50.0,
              w_data: float = 100.0, log_every: int = 500):
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
            loss_pde  = torch.mean(res**2)
            loss_bc   = self._boundary_loss()
            loss_ic   = self._ic_loss()
            loss_data = self._data_loss()
            loss      = (w_pde * loss_pde + w_bc * loss_bc
                         + w_ic * loss_ic + w_data * loss_data)

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
                    "data":  loss_data.item(),
                })
                pbar.set_postfix(
                    loss=f"{loss.item():.3e}",
                    pde=f"{loss_pde.item():.3e}",
                    ic=f"{loss_ic.item():.3e}",
                )
        return history

    def price(self, p: ModelParams, S: float,
              v: Optional[float] = None, t: float = 0.0) -> float:
        """用统一网络推断期权价格。
        看跌期权用 put-call parity 从看涨价格换算，不需要重新训练。
        """
        if v is None:
            v = p.v0
        # 如果是看跌，先用看涨参数推断，再用 put-call parity 转换
        if p.option_type == "put":
            p_call = ModelParams(**{**p.__dict__, "option_type": "call"})
            call_price = self.price(p_call, S, v, t)
            tau = p.T - t
            put_price = call_price - S + p.K * np.exp(-p.r * tau)
            return max(put_price, 0.0)
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
