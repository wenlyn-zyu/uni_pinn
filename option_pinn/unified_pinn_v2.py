"""
unified_pinn.py -- batched version for speed

Key change: _boundary_loss, _ic_loss, _data_loss now batch ALL models into
a single forward pass instead of 52 sequential calls. This reduces per-epoch
time from ~3.5s to ~0.3s on GPU.

All models share K=100, T=1, r=0.05, S_max=300, v_max=1 so scalar constants
are safe to use in the batched forward pass.
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Parameter container
# ---------------------------------------------------------------------------

@dataclass
class ModelParams:
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
    option_type: str   = "call"

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
        return torch.tensor(
            [[self.sigma, self.beta, self.kappa, self.theta, self.xi, self.rho]],
            dtype=torch.float32, device=device
        )


# ---------------------------------------------------------------------------
# BS analytical solution (vectorized)
# ---------------------------------------------------------------------------

def _bs_call(S, K, tau, r, sigma):
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
# CEV analytical solution (Schroder 1989, non-central chi-squared)
# ---------------------------------------------------------------------------

def cev_schroder_call(S, K, T, r, sigma, beta):
    """CEV European call via non-central chi-squared (Schroder 1989).
    Valid for beta < 1. Falls back to BS when beta ≈ 1.

    Formula:
      C = S · [1 - ncx2.cdf(λ·K^{2δ}; 2+1/δ, λ·S^{2δ}·e^{2rδT})]
          - K·e^{-rT} · ncx2.cdf(λ·S^{2δ}·e^{2rδT}; 1/δ, λ·K^{2δ})
    where δ = 1-β, λ = 2r/(σ²·δ·(e^{2rδT}-1)).
    """
    from scipy.stats import ncx2
    if abs(beta - 1.0) < 1e-9:
        from scipy.stats import norm
        sqt = sigma * np.sqrt(max(T, 1e-10))
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / sqt
        d2 = d1 - sqt
        return float(max(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2), 0.0))
    delta = 1.0 - beta
    nu    = 1.0 / delta
    lam   = (2.0 * r) / (sigma**2 * delta * (np.exp(2.0 * r * delta * T) - 1.0))
    x     = lam * S**(2.0 * delta) * np.exp(2.0 * r * delta * T)
    y     = lam * K**(2.0 * delta)
    d     = 2.0 + nu
    call  = (S * (1.0 - ncx2.cdf(y, df=d,   nc=x))
             - K * np.exp(-r * T) * ncx2.cdf(x, df=d - 2, nc=y))
    return float(max(call, max(S - K * np.exp(-r * T), 0.0)))


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class UnifiedNet(nn.Module):
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

    def forward(self, S_n, v_n, t_n, lam, S_raw, t_raw, K, T, r, return_raw=False):
        sigma = lam[:, 0:1]
        beta  = lam[:, 1:2]
        xi    = lam[:, 4:5]
        mask      = torch.tanh(xi / 0.05) ** 2
        v_approx  = torch.clamp(v_n, min=1e-6)
        sigma_eff = (1 - mask) * sigma + mask * torch.sqrt(v_approx)
        tau   = torch.clamp(T - t_raw, min=1e-4)
        V_bs  = _bs_call(S_raw, K, tau, r, sigma_eff)
        x   = torch.cat([S_n, v_n, t_n, lam], dim=1)
        raw = self.net(x)
        # For BSM models (xi=0, beta=1): force raw->0 to preserve analytical Vega.
        # BS base function is exact for BSM, so no network correction is needed.
        is_bsm = (xi.abs() < 1e-9) & ((beta - 1.0).abs() < 1e-9)
        raw = raw * (1.0 - is_bsm.float() * 0.99)
        V = torch.clamp(V_bs + K * raw, min=0.0)
        return (V, raw) if return_raw else V


# ---------------------------------------------------------------------------
# Unified PDE operator
# ---------------------------------------------------------------------------

def unified_pde_residual(net, S, v, t, lam, K, T, r, S_max, v_max):
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
    return residual / (V.detach().abs() + 1.0)


# ---------------------------------------------------------------------------
# Trainer -- batched version
# ---------------------------------------------------------------------------

class UnifiedPINN:

    def __init__(self,
                 param_list: list,
                 hidden: int = 128,
                 depth: int = 6,
                 lr: float = 1e-3,
                 ref_data: dict = None,
                 device=None):
        self.param_list = param_list
        self.ref_data   = ref_data or {}
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.net = UnifiedNet(hidden=hidden, depth=depth).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.scheduler = None

        # Pre-build batched lambda tensor for all models (shape: M x 6)
        self._lam_all = torch.cat(
            [p.to_lambda_tensor(self.device) for p in param_list], dim=0
        )  # (M, 6)

        # Shared constants (all models use same K, T, r, S_max, v_max)
        p0 = param_list[0]
        self.K     = p0.K
        self.T     = p0.T
        self.r     = p0.r
        self.S_max = p0.S_max
        self.v_max = p0.v_max

        # Pre-cache data anchor tensors on device to avoid per-epoch CPU->GPU copies
        self._data_cache = None
        if ref_data:
            S_list, v_list, t_list, V_list, lam_list = [], [], [], [], []
            for idx, (S_arr, v_arr, t_arr, V_arr) in ref_data.items():
                p   = param_list[idx]
                n   = len(S_arr)
                lam = p.to_lambda_tensor(self.device).expand(n, -1)
                S_list.append(torch.tensor(S_arr, dtype=torch.float32, device=self.device).reshape(-1, 1))
                v_list.append(torch.tensor(v_arr, dtype=torch.float32, device=self.device).reshape(-1, 1))
                t_list.append(torch.tensor(t_arr, dtype=torch.float32, device=self.device).reshape(-1, 1))
                V_list.append(torch.tensor(V_arr, dtype=torch.float32, device=self.device).reshape(-1, 1))
                lam_list.append(lam)
            self._data_cache = (
                torch.cat(S_list), torch.cat(v_list),
                torch.cat(t_list), torch.cat(V_list),
                torch.cat(lam_list)
            )

    def _to(self, x):
        return x.to(self.device)

    def _sample_batch(self, n_per_model: int = 5000):
        """Sample collocation points for all models, batched."""
        M = len(self.param_list)
        n_base = int(n_per_model * 0.7)
        n_otm  = n_per_model - n_base
        K = self.K

        S_base = torch.FloatTensor(M * n_base, 1).uniform_(0.01, self.S_max)
        S_otm  = torch.FloatTensor(M * n_otm,  1).uniform_(0.7 * K, K)
        S = torch.cat([S_base, S_otm])  # (M*n_per_model, 1) -- shuffled below

        v = torch.FloatTensor(M * n_per_model, 1).uniform_(1e-4, self.v_max)
        t = torch.FloatTensor(M * n_per_model, 1).uniform_(0.0, self.T * 0.999)

        # Repeat lambda for each model: each model gets n_per_model rows
        lam = self._lam_all.unsqueeze(1).expand(M, n_per_model, 6).reshape(M * n_per_model, 6)

        # Shuffle S to break the base/otm block structure
        idx = torch.randperm(M * n_per_model)
        S = S[idx]

        return (self._to(S), self._to(v), self._to(t), lam)

    def _ic_loss(self, n_per_model: int = 500):
        """Terminal condition loss -- single batched forward pass."""
        M = len(self.param_list)
        n = n_per_model
        K, T, S_max, v_max, r = self.K, self.T, self.S_max, self.v_max, self.r

        S_ic = self._to(torch.FloatTensor(M * n, 1).uniform_(0.01, S_max))
        v_ic = self._to(torch.FloatTensor(M * n, 1).uniform_(1e-4, v_max))
        t_ic = self._to(torch.full((M * n, 1), T))
        lam  = self._lam_all.unsqueeze(1).expand(M, n, 6).reshape(M * n, 6)

        pred   = self.net(S_ic/S_max, v_ic/v_max, t_ic/T, lam, S_ic, t_ic, K, T, r)
        payoff = torch.clamp(S_ic - K, min=0.)  # all call options
        rel_err = (pred - payoff) / (payoff + K * 0.1)
        return torch.mean(rel_err**2)

    def _data_loss(self):
        """Data anchor loss -- uses pre-cached GPU tensors, single forward pass."""
        if self._data_cache is None:
            return torch.tensor(0.0, device=self.device)
        K, T, S_max, v_max, r = self.K, self.T, self.S_max, self.v_max, self.r
        S_t, v_t, t_t, V_t, lam = self._data_cache
        pred    = self.net(S_t/S_max, v_t/v_max, t_t/T, lam, S_t, t_t, K, T, r)
        rel_err = (pred - V_t) / (V_t.abs() + K * 0.1)
        return torch.mean(rel_err**2)

    def _bsm_raw_loss(self, n_per_model: int = 1000):
        """Penalize non-zero network raw output for BSM models.

        For BSM (xi=0, beta=1), the BS base function is exact, so the network
        correction `raw` should be zero.  Non-zero raw destroys Vega accuracy.
        """
        M = len(self.param_list)
        n = n_per_model
        K, T, S_max, v_max, r = self.K, self.T, self.S_max, self.v_max, self.r

        S_b = self._to(torch.FloatTensor(M * n, 1).uniform_(0.01, S_max))
        v_b = self._to(torch.FloatTensor(M * n, 1).uniform_(1e-4, v_max))
        t_b = self._to(torch.FloatTensor(M * n, 1).uniform_(0.0, T * 0.999))
        lam = self._lam_all.unsqueeze(1).expand(M, n, 6).reshape(M * n, 6)

        _, raw = self.net(S_b/S_max, v_b/v_max, t_b/T, lam, S_b, t_b, K, T, r,
                          return_raw=True)

        # Identify BSM rows: xi==0 and beta==1
        bsm_mask = (lam[:, 4] == 0) & (lam[:, 1] == 1.0)
        if bsm_mask.any():
            return torch.mean(raw[bsm_mask]**2)
        return torch.tensor(0.0, device=self.device)

    def _boundary_loss(self, n_per_model: int = 500):
        """Boundary condition loss -- two batched forward passes (S=0, S=S_max)."""
        M = len(self.param_list)
        n = n_per_model
        K, T, S_max, v_max, r = self.K, self.T, self.S_max, self.v_max, self.r

        lam = self._lam_all.unsqueeze(1).expand(M, n, 6).reshape(M * n, 6)
        tm  = self._to(torch.FloatTensor(M * n, 1).uniform_(0, T))
        disc = torch.exp(-r * (T - tm))

        # S=0 boundary: V=0 for calls
        S0    = self._to(torch.zeros(M * n, 1))
        v0b   = self._to(torch.FloatTensor(M * n, 1).uniform_(1e-4, v_max))
        pred0 = self.net(S0/S_max, v0b/v_max, tm/T, lam, S0, tm, K, T, r)
        loss0 = torch.mean(pred0**2)

        # S=S_max boundary: V = S_max - K*disc for calls
        Sm    = self._to(torch.full((M * n, 1), S_max))
        vmb   = self._to(torch.FloatTensor(M * n, 1).uniform_(1e-4, v_max))
        predm = self.net(Sm/S_max, vmb/v_max, tm/T, lam, Sm, tm, K, T, r)
        Vm    = S_max - K * disc
        lossm = torch.mean(((predm - Vm) / (Vm + 1.0))**2)

        return (loss0 + lossm) * 0.5

    def train(self, epochs: int = 30000, n_per_model: int = 5000,
              w_pde: float = 1.0, w_bc: float = 10.0, w_ic: float = 10.0,
              w_data: float = 100.0, w_bsm_raw: float = 0.0,
              log_every: int = 500,
              save_every: int = 0, save_path: str = None):
        from tqdm import tqdm
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=1e-5
        )
        history = []
        pbar = tqdm(range(1, epochs + 1), desc="UnifiedPINN", dynamic_ncols=True)
        for epoch in pbar:
            self.optimizer.zero_grad()

            S_c, v_c, t_c, lam_c = self._sample_batch(n_per_model)
            res      = unified_pde_residual(
                self.net, S_c, v_c, t_c, lam_c,
                self.K, self.T, self.r, self.S_max, self.v_max
            )
            loss_pde  = torch.mean(res**2)
            loss_bc   = self._boundary_loss()
            loss_ic   = self._ic_loss()
            loss_data = self._data_loss()
            loss_bsm_raw = self._bsm_raw_loss()
            loss      = (w_pde * loss_pde + w_bc * loss_bc
                         + w_ic * loss_ic + w_data * loss_data
                         + w_bsm_raw * loss_bsm_raw)

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
            if save_every > 0 and save_path and epoch % save_every == 0:
                ckpt_path = save_path.replace(".pt", f"_e{epoch}.pt")
                torch.save({"state_dict": self.net.state_dict(),
                             "epoch": epoch, "optimizer": self.optimizer.state_dict()},
                            ckpt_path)
        return history

    def price(self, p: ModelParams, S: float,
              v: Optional[float] = None, t: float = 0.0) -> float:
        if v is None:
            v = p.v0
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
