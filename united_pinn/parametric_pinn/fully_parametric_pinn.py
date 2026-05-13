"""
fully_parametric_pinn.py

Single PINN that covers the ENTIRE parameter space:
  State:   S, v, tau (time-to-maturity)
  Contract: K, r
  Model:   sigma, beta, kappa, theta, xi, rho

All are network inputs. Train once → price any parameter combination.
Supports BSM (xi=0,beta=1), CEV (xi=0,beta≠1), Heston (xi>0).
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Normalization constants
# ---------------------------------------------------------------------------
S_MAX   = 500.0    # max underlying price
V_MAX   = 1.0      # max variance
TAU_MAX = 3.0      # max time-to-maturity (years)
K_REF   = 100.0    # reference strike


# ---------------------------------------------------------------------------
# BS analytical formula (vectorized)
# ---------------------------------------------------------------------------

def bs_call(S, K, tau, r, sigma):
    eps   = 1e-8
    tau   = torch.clamp(tau,   min=eps)
    sigma = torch.clamp(sigma, min=eps)
    sqt   = sigma * torch.sqrt(tau)
    d1    = (torch.log(S / K) + (r + 0.5 * sigma**2) * tau) / sqt
    d2    = d1 - sqt
    nd    = torch.distributions.Normal(0., 1.)
    call  = S * nd.cdf(d1) - K * torch.exp(-r * tau) * nd.cdf(d2)
    return torch.clamp(call, min=0.0)


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class FullyParametricNet(nn.Module):
    """MLP that maps (log_m, v_n, tau_n, K_n, r, sigma, beta, kappa, theta, xi, rho) → price correction.

    Input:  11 normalized / raw features.
    Output: raw correction such that V = V_BS + K * raw.
    """

    def __init__(self, hidden: int = 256, depth: int = 8):
        super().__init__()
        in_dim = 11
        layers = [nn.Linear(in_dim, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        last = nn.Linear(hidden, 1)
        nn.init.normal_(last.weight, std=0.001)
        nn.init.zeros_(last.bias)
        layers.append(last)
        self.mlp = nn.Sequential(*layers)

    def forward(self, S, v, tau, K, r, lam, return_raw=False):
        """
        Args:
            S, v, tau, K, r:  actual (not normalized) tensors, shape (N,1)
            lam:              [sigma, beta, kappa, theta, xi, rho], shape (N,6)
        Returns:
            V:    option price, shape (N,1)
            raw:  network correction (only if return_raw=True)
        """
        sigma = lam[:, 0:1]
        beta  = lam[:, 1:2]
        xi    = lam[:, 4:5]

        # --- smooth model blending ---
        mask = torch.tanh(xi / 0.05) ** 2
        v_safe = torch.clamp(v, min=1e-6)
        sigma_eff = (1.0 - mask) * sigma + mask * torch.sqrt(v_safe)

        # --- BS base price ---
        V_bs = bs_call(S, K, tau, r, sigma_eff)

        # --- normalized network inputs ---
        log_m  = torch.log(torch.clamp(S / K, min=1e-6))
        v_n    = v / V_MAX
        tau_n  = tau / TAU_MAX
        K_n    = K / K_REF

        x = torch.cat([log_m, v_n, tau_n, K_n,
                        r,
                        sigma, beta,
                        lam[:, 2:3], lam[:, 3:4],  # kappa, theta
                        xi,
                        lam[:, 5:6]],  # rho
                       dim=1)

        raw = self.mlp(x)

        # Suppress raw correction for pure BSM (BS base is already exact)
        is_bsm = (xi.abs() < 1e-9) & ((beta - 1.0).abs() < 1e-9)
        raw = raw * (1.0 - is_bsm.float() * 0.99)

        V = torch.clamp(V_bs + K * raw, min=0.0)
        return (V, raw) if return_raw else V


# ---------------------------------------------------------------------------
# Unified PDE residual
# ---------------------------------------------------------------------------

def pde_residual(net, S, v, tau, K, r, lam):
    """Compute normalized PDE residual at collocation points.

    PDE (forward in tau = T-t):
      V_tau = 0.5*a*V_SS + b*V_Sv + 0.5*c*V_vv + r*S*V_S + d*V_v - r*V

    where:
      a = (1-mask)*sigma^2*S^(2*beta) + mask*v*S^2
      b = rho*xi*v*S
      c = xi^2*v
      d = kappa*(theta - v)
      mask = tanh(xi/0.05)^2
    """
    S.requires_grad_(True)
    v.requires_grad_(True)
    tau.requires_grad_(True)

    V = net(S, v, tau, K, r, lam)

    ones = torch.ones_like(V)
    V_tau = torch.autograd.grad(V, tau, grad_outputs=ones, create_graph=True)[0]
    V_S   = torch.autograd.grad(V, S,   grad_outputs=ones, create_graph=True)[0]
    V_v   = torch.autograd.grad(V, v,   grad_outputs=ones, create_graph=True)[0]
    V_SS  = torch.autograd.grad(V_S, S, grad_outputs=torch.ones_like(V_S),
                                 create_graph=True)[0]
    V_vv  = torch.autograd.grad(V_v, v, grad_outputs=torch.ones_like(V_v),
                                 create_graph=True)[0]
    V_Sv  = torch.autograd.grad(V_S, v, grad_outputs=torch.ones_like(V_S),
                                 create_graph=True)[0]

    sigma = lam[:, 0:1]
    beta  = lam[:, 1:2]
    kappa = lam[:, 2:3]
    theta = lam[:, 3:4]
    xi    = lam[:, 4:5]
    rho   = lam[:, 5:6]

    mask = torch.tanh(xi / 0.05) ** 2
    a = (1.0 - mask) * sigma**2 * S**(2.0 * beta) + mask * v * S**2
    b = rho * xi * v * S
    c = xi**2 * v
    d = kappa * (theta - v)

    residual = (V_tau
                - 0.5 * a * V_SS
                - b * V_Sv
                - 0.5 * c * V_vv
                - r * S * V_S
                - d * V_v
                + r * V)
    return residual / (V.detach().abs() + 1.0)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class FullyParametricPINN:
    def __init__(self,
                 hidden: int = 256,
                 depth: int = 8,
                 lr: float = 1e-3,
                 ref_data: dict = None,
                 device=None):
        self.ref_data = ref_data or {}
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.net = FullyParametricNet(hidden=hidden, depth=depth).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.scheduler = None

        # Pre-cache reference data on GPU if provided
        self._data_cache = None
        if ref_data:
            self._cache_ref_data(ref_data)

    def _cache_ref_data(self, ref_data):
        S_list, v_list, tau_list, K_list, r_list, V_list, lam_list = [], [], [], [], [], [], []
        for (S_arr, v_arr, tau_arr, K_val, r_val, lam_arr, V_arr) in ref_data:
            n = len(S_arr)
            S_list.append(torch.tensor(S_arr, dtype=torch.float32, device=self.device).reshape(-1, 1))
            v_list.append(torch.tensor(v_arr, dtype=torch.float32, device=self.device).reshape(-1, 1))
            tau_list.append(torch.tensor(tau_arr, dtype=torch.float32, device=self.device).reshape(-1, 1))
            K_list.append(torch.full((n, 1), K_val, dtype=torch.float32, device=self.device))
            r_list.append(torch.full((n, 1), r_val, dtype=torch.float32, device=self.device))
            lam_list.append(torch.tensor(lam_arr, dtype=torch.float32, device=self.device).expand(n, -1))
            V_list.append(torch.tensor(V_arr, dtype=torch.float32, device=self.device).reshape(-1, 1))
        self._data_cache = (
            torch.cat(S_list), torch.cat(v_list), torch.cat(tau_list),
            torch.cat(K_list), torch.cat(r_list), torch.cat(lam_list),
            torch.cat(V_list),
        )

    def _to(self, x):
        return x.to(self.device)

    # ------------------------------------------------------------------
    # Parameter sampling
    # ------------------------------------------------------------------

    def _sample_params(self, n: int):
        """Sample n random parameter combinations uniformly from the 3 models.

        Returns:
            K:     (n,1)
            r:     (n,1)
            lam:   (n,6)  [sigma, beta, kappa, theta, xi, rho]
        """
        # Split evenly among model types
        n_bsm   = n // 3
        n_cev   = n // 3
        n_hest  = n - n_bsm - n_cev

        # --- BSM: xi=0, beta=1, kappa=0, theta=0, rho=0 ---
        sigma_bsm = torch.FloatTensor(n_bsm, 1).uniform_(0.05, 0.50)
        beta_bsm  = torch.ones(n_bsm, 1)
        kappa_bsm = torch.zeros(n_bsm, 1)
        theta_bsm = torch.zeros(n_bsm, 1)
        xi_bsm    = torch.zeros(n_bsm, 1)
        rho_bsm   = torch.zeros(n_bsm, 1)
        lam_bsm   = torch.cat([sigma_bsm, beta_bsm, kappa_bsm, theta_bsm, xi_bsm, rho_bsm], dim=1)

        # --- CEV: xi=0, kappa=0, theta=0, rho=0 ---
        sigma_cev = torch.FloatTensor(n_cev, 1).uniform_(0.10, 0.30)
        beta_cev  = torch.FloatTensor(n_cev, 1).uniform_(0.10, 0.90)
        kappa_cev = torch.zeros(n_cev, 1)
        theta_cev = torch.zeros(n_cev, 1)
        xi_cev    = torch.zeros(n_cev, 1)
        rho_cev   = torch.zeros(n_cev, 1)
        lam_cev   = torch.cat([sigma_cev, beta_cev, kappa_cev, theta_cev, xi_cev, rho_cev], dim=1)

        # --- Heston: beta=1, sigma=0 ---
        sigma_hest = torch.zeros(n_hest, 1)
        beta_hest  = torch.ones(n_hest, 1)
        kappa_hest = 10.0 ** torch.FloatTensor(n_hest, 1).uniform_(
            np.log10(0.5), np.log10(10.0))
        theta_hest = torch.FloatTensor(n_hest, 1).uniform_(0.01, 0.10)
        xi_hest    = 10.0 ** torch.FloatTensor(n_hest, 1).uniform_(
            np.log10(0.05), np.log10(0.50))
        rho_hest   = torch.FloatTensor(n_hest, 1).uniform_(-0.95, -0.10)
        lam_hest   = torch.cat([sigma_hest, beta_hest, kappa_hest, theta_hest, xi_hest, rho_hest], dim=1)

        lam = torch.cat([lam_bsm, lam_cev, lam_hest], dim=0)

        # Contract parameters
        K = torch.FloatTensor(n, 1).uniform_(50.0, 200.0)
        r = torch.FloatTensor(n, 1).uniform_(0.01, 0.10)

        # Shuffle
        idx = torch.randperm(n)
        return K[idx], r[idx], lam[idx]

    # ------------------------------------------------------------------
    # Collocation point sampling
    # ------------------------------------------------------------------

    def _sample_collocation(self, n_total: int):
        """Sample collocation points for PDE residual.

        For each parameter set, sample (S, v, tau) within appropriate ranges.
        S ∈ [0.01*K, min(3*K, S_MAX)]
        v ∈ [1e-4, V_MAX]
        tau ∈ [1e-4, TAU_MAX]
        """
        K, r, lam = self._sample_params(n_total)

        # S: sample around K (moneyness ~ 0.5 to 2.0 for base, plus OTM tail)
        n_base = int(n_total * 0.7)
        n_otm  = n_total - n_base

        K_np = K.squeeze()
        S_base = torch.FloatTensor(n_base).uniform_(0.5, 1.5) * K_np[:n_base]
        S_otm  = torch.FloatTensor(n_otm).uniform_(0.3, 0.8) * K_np[n_base:]
        S = torch.cat([S_base, S_otm]).reshape(-1, 1)
        S = torch.clamp(S, min=1.0, max=S_MAX)

        v = torch.FloatTensor(n_total, 1).uniform_(1e-4, V_MAX)
        tau = torch.FloatTensor(n_total, 1).uniform_(1e-4, TAU_MAX)

        # Shuffle to mix model types
        idx = torch.randperm(n_total)
        return (self._to(S[idx]), self._to(v[idx]), self._to(tau[idx]),
                self._to(K[idx]), self._to(r[idx]),
                lam[idx].to(self.device))

    # ------------------------------------------------------------------
    # Loss terms
    # ------------------------------------------------------------------

    def _pde_loss(self, n: int = 8000):
        S, v, tau, K, r, lam = self._sample_collocation(n)
        res = pde_residual(self.net, S, v, tau, K, r, lam)
        return torch.mean(res**2)

    def _ic_loss(self, n: int = 1000):
        """Payoff condition at tau=0: V(S,v,0) = max(S-K, 0) for calls."""
        K, r, lam = self._sample_params(n)
        S   = torch.FloatTensor(n, 1).uniform_(1.0, S_MAX)
        v   = torch.FloatTensor(n, 1).uniform_(1e-4, V_MAX)
        tau = torch.zeros(n, 1)

        S, v, tau = self._to(S), self._to(v), self._to(tau)
        K, r = self._to(K), self._to(r)
        lam = lam.to(self.device)

        pred   = self.net(S, v, tau, K, r, lam)
        payoff = torch.clamp(S - K, min=0.)
        rel_err = (pred - payoff) / (payoff + K * 0.1)
        return torch.mean(rel_err**2)

    def _boundary_loss(self, n: int = 500):
        """Boundary conditions at S=0 and S=S_max."""
        K, r, lam = self._sample_params(n)
        tau = torch.FloatTensor(n, 1).uniform_(1e-4, TAU_MAX)
        disc = torch.exp(-r * tau)

        # S=0: V=0 for calls
        S0  = torch.zeros(n, 1)
        v0  = torch.FloatTensor(n, 1).uniform_(1e-4, V_MAX)
        S0, v0, tau_d = self._to(S0), self._to(v0), self._to(tau)
        K_d, r_d = self._to(K), self._to(r)
        lam_d = lam.to(self.device)
        pred0 = self.net(S0, v0, tau_d, K_d, r_d, lam_d)
        loss0 = torch.mean(pred0**2)

        # S=S_max: V = S_max - K*exp(-r*tau) for calls
        Sm  = self._to(torch.full((n, 1), S_MAX))
        vm  = self._to(torch.FloatTensor(n, 1).uniform_(1e-4, V_MAX))
        predm = self.net(Sm, vm, tau_d, K_d, r_d, lam_d)
        Vm = S_MAX - K_d * disc.to(self.device)
        lossm = torch.mean(((predm - Vm) / (Vm + 1.0))**2)

        return (loss0 + lossm) * 0.5

    def _data_loss(self):
        if self._data_cache is None:
            return torch.tensor(0.0, device=self.device)
        S, v, tau, K, r, lam, V_ref = self._data_cache
        pred    = self.net(S, v, tau, K, r, lam)
        rel_err = (pred - V_ref) / (V_ref.abs() + torch.clamp(K, min=1.0) * 0.1)
        return torch.mean(rel_err**2)

    def _bsm_raw_loss(self, n: int = 1000):
        """Penalize non-zero raw for BSM (where BS base is exact)."""
        K, r, lam = self._sample_params(n)
        S   = torch.FloatTensor(n, 1).uniform_(1.0, S_MAX)
        v   = torch.FloatTensor(n, 1).uniform_(1e-4, V_MAX)
        tau = torch.FloatTensor(n, 1).uniform_(1e-4, TAU_MAX)

        S, v, tau = self._to(S), self._to(v), self._to(tau)
        K, r = self._to(K), self._to(r)
        lam = lam.to(self.device)

        _, raw = self.net(S, v, tau, K, r, lam, return_raw=True)
        bsm_mask = (lam[:, 4].abs() < 1e-9) & ((lam[:, 1] - 1.0).abs() < 1e-9)
        if bsm_mask.any():
            return torch.mean(raw[bsm_mask]**2)
        return torch.tensor(0.0, device=self.device)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self, epochs: int = 50000,
              w_pde: float = 1.0, w_bc: float = 10.0, w_ic: float = 10.0,
              w_data: float = 100.0, w_bsm_raw: float = 1.0,
              log_every: int = 500,
              save_every: int = 0, save_path: str = None):
        from tqdm import tqdm

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=1e-5
        )
        history = []
        pbar = tqdm(range(1, epochs + 1), desc="FullyParamPINN", dynamic_ncols=True)

        for epoch in pbar:
            self.optimizer.zero_grad()

            loss_pde      = self._pde_loss(8000)
            loss_bc       = self._boundary_loss(500)
            loss_ic       = self._ic_loss(1000)
            loss_data     = self._data_loss()
            loss_bsm_raw  = self._bsm_raw_loss(1000)
            loss = (w_pde * loss_pde + w_bc * loss_bc
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
                    data=f"{loss_data.item():.3e}",
                )

            if save_every > 0 and save_path and epoch % save_every == 0:
                ckpt_path = save_path.replace(".pt", f"_e{epoch}.pt")
                torch.save({"state_dict": self.net.state_dict(),
                             "epoch": epoch,
                             "optimizer": self.optimizer.state_dict()},
                            ckpt_path)

        return history

    # ------------------------------------------------------------------
    # Pricing
    # ------------------------------------------------------------------

    def price(self, S: float, K: float, T: float, r: float,
              sigma: float = 0.0, beta: float = 1.0,
              kappa: float = 0.0, theta: float = 0.0,
              xi: float = 0.0, rho: float = 0.0,
              v0: Optional[float] = None) -> float:
        """Price a call option for given parameters.

        Args:
            S:     underlying price
            K:     strike
            T:     time to maturity
            r:     risk-free rate
            sigma: BSM σ or CEV σ
            beta:  CEV β (1.0 = BSM)
            kappa, theta, xi, rho: Heston params (xi=0 for BSM/CEV)
            v0:    initial variance (defaults to sigma² for BSM/CEV, theta for Heston)
        """
        if v0 is None:
            v0 = sigma**2 if xi == 0 else theta

        self.net.eval()
        with torch.no_grad():
            S_t   = self._to(torch.tensor([[S]], dtype=torch.float32))
            v_t   = self._to(torch.tensor([[v0]], dtype=torch.float32))
            tau_t = self._to(torch.tensor([[T]], dtype=torch.float32))
            K_t   = self._to(torch.tensor([[K]], dtype=torch.float32))
            r_t   = self._to(torch.tensor([[r]], dtype=torch.float32))
            lam_t = self._to(torch.tensor(
                [[sigma, beta, kappa, theta, xi, rho]], dtype=torch.float32))
            out = self.net(S_t, v_t, tau_t, K_t, r_t, lam_t)
        return out.item()

    def save(self, path: str):
        torch.save({"state_dict": self.net.state_dict()}, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ckpt["state_dict"])
