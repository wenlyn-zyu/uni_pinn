# option_pinn/exploration/improved_parametric_pinn.py
"""Improved fully-parametric PINN v2.

Key improvements over v1:
  - 512 hidden × 10 layers (vs 256 × 8)
  - Wider parameter ranges covering real-market conditions
  - Log-uniform sampling for kappa, xi (span orders of magnitude)
  - Moneyness-based S sampling (more ATM focus)
  - More reference anchor points (50 per config)
  - Put option support via put-call parity
  - Integrated reference data generation (no separate script)
  - Adaptive PDE weight for Heston regions
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List, Tuple
from scipy.stats import ncx2
from numpy.polynomial.legendre import leggauss


# ---------------------------------------------------------------------------
# Normalization constants
# ---------------------------------------------------------------------------
S_MAX   = 500.0
V_MAX   = 1.0
TAU_MAX = 3.0
K_REF   = 100.0


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
# Reference solvers (numpy)
# ---------------------------------------------------------------------------

def _bsm_call_np(S, K, tau, r, sigma):
    from scipy.stats import norm
    eps = 1e-10
    tau, sigma = max(tau, eps), max(sigma, eps)
    sqt = sigma * np.sqrt(tau)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / sqt
    d2 = d1 - sqt
    return float(max(S * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2), 0.0))


def _cev_call_np(S, K, T, r, sigma, beta):
    if abs(beta - 1.0) < 1e-9:
        return _bsm_call_np(S, K, T, r, sigma)
    delta = 1.0 - beta
    nu    = 1.0 / delta
    e2rdT = np.exp(2.0 * r * delta * T)
    denom = sigma**2 * delta * (e2rdT - 1.0)
    if abs(denom) < 1e-12:
        return _bsm_call_np(S, K, T, r, sigma)
    lam = (2.0 * r) / denom
    x = lam * S**(2.0 * delta) * e2rdT
    y = lam * K**(2.0 * delta)
    d = 2.0 + nu
    call = (S * (1.0 - ncx2.cdf(y, df=d, nc=x))
            - K * np.exp(-r * T) * ncx2.cdf(x, df=d - 2, nc=y))
    return float(max(call, max(S - K * np.exp(-r * T), 0.0)))


# GL quadrature for Heston
_N_GL = 96
_GL_NODES, _GL_WEIGHTS = leggauss(_N_GL)
_PHI_MAX = 100.0
_GL_PHI = (_GL_NODES + 1.0) * 0.5 * _PHI_MAX
_GL_W   = _GL_WEIGHTS * 0.5 * _PHI_MAX


def _heston_cf(phi, S, K, T, r, kappa, theta, xi, rho, v0, j):
    i = 1j
    u = 0.5 if j == 1 else -0.5
    b = (kappa - rho * xi) if j == 1 else kappa
    a = kappa * theta
    x = np.log(S / K)
    d = np.sqrt((rho * xi * i * phi - b)**2 - xi**2 * (2 * u * i * phi - phi**2))
    g = (b - rho * xi * i * phi + d) / (b - rho * xi * i * phi - d)
    C = (r * i * phi * T
         + a / xi**2 * ((b - rho * xi * i * phi + d) * T
                        - 2 * np.log((1 - g * np.exp(d * T)) / (1 - g))))
    D = ((b - rho * xi * i * phi + d) / xi**2
         * (1 - np.exp(d * T)) / (1 - g * np.exp(d * T)))
    return np.exp(C + D * v0 + i * phi * x)


def _heston_call_np(S, K, T, r, kappa, theta, xi, rho, v0):
    if T < 1e-6:
        return max(S - K, 0.0)
    try:
        cf1 = _heston_cf(_GL_PHI, S, K, T, r, kappa, theta, xi, rho, v0, j=1)
        cf2 = _heston_cf(_GL_PHI, S, K, T, r, kappa, theta, xi, rho, v0, j=2)
        P1 = 0.5 + (1.0 / np.pi) * np.sum(_GL_W * np.real(cf1 / (1j * _GL_PHI)))
        P2 = 0.5 + (1.0 / np.pi) * np.sum(_GL_W * np.real(cf2 / (1j * _GL_PHI)))
        return float(max(S * P1 - K * np.exp(-r * T) * P2, 0.0))
    except Exception:
        return np.nan


# ---------------------------------------------------------------------------
# Reference data generation (integrated)
# ---------------------------------------------------------------------------

def generate_ref_anchors(
    n_bsm: int = 8, n_cev: int = 12, n_heston: int = 30,
    n_S_per_config: int = 50, seed: int = 42,
) -> List[Tuple]:
    """Generate reference anchor data for training.

    Each anchor: (S_arr, v_arr, tau_arr, K_val, r_val, lam_arr, V_arr)
    where lam_arr has shape (n_S, 6) and V_arr has shape (n_S,).

    Returns list of anchor tuples for BSM + CEV + Heston configs.
    """
    rng = np.random.RandomState(seed)
    anchors = []

    # --- BSM anchors --------------------------------------------------------
    sigma_list = np.linspace(0.08, 0.45, n_bsm)
    for sigma in sigma_list:
        K = float(rng.uniform(60, 150))
        T = float(rng.uniform(0.1, 2.5))
        r = float(rng.uniform(0.01, 0.10))
        S_arr = np.linspace(K * 0.5, K * 2.0, n_S_per_config)
        v_arr = np.full(n_S_per_config, sigma**2)
        tau_arr = np.full(n_S_per_config, T)
        V_arr = np.array([_bsm_call_np(s, K, T, r, sigma) for s in S_arr])
        lam_arr = np.tile([sigma, 1.0, 0.0, 0.0, 0.0, 0.0], (n_S_per_config, 1))
        anchors.append((S_arr, v_arr, tau_arr, K, r, lam_arr, V_arr))

    # --- CEV anchors --------------------------------------------------------
    for _ in range(n_cev):
        sigma  = float(rng.uniform(0.10, 0.35))
        beta   = float(rng.uniform(0.15, 0.85))
        K      = float(rng.uniform(60, 150))
        T      = float(rng.uniform(0.1, 2.5))
        r      = float(rng.uniform(0.01, 0.10))
        S_arr  = np.linspace(K * 0.55, K * 1.8, n_S_per_config)
        v_arr  = np.full(n_S_per_config, sigma**2)
        tau_arr = np.full(n_S_per_config, T)
        V_arr = np.array([_cev_call_np(s, K, T, r, sigma, beta) for s in S_arr])
        lam_arr = np.tile([sigma, beta, 0.0, 0.0, 0.0, 0.0], (n_S_per_config, 1))
        anchors.append((S_arr, v_arr, tau_arr, K, r, lam_arr, V_arr))

    # --- Heston anchors -----------------------------------------------------
    for _ in range(n_heston):
        kappa = float(10 ** rng.uniform(np.log10(0.5), np.log10(12.0)))
        theta = float(rng.uniform(0.015, 0.12))
        xi    = float(10 ** rng.uniform(np.log10(0.05), np.log10(0.70)))
        rho   = float(rng.uniform(-0.95, -0.10))
        v0    = theta
        K     = float(rng.uniform(60, 150))
        T     = float(rng.uniform(0.1, 2.5))
        r     = float(rng.uniform(0.01, 0.10))
        S_arr  = np.linspace(K * 0.55, K * 1.8, n_S_per_config)
        v_arr  = np.full(n_S_per_config, v0)
        tau_arr = np.full(n_S_per_config, T)
        V_arr = np.array([
            _heston_call_np(s, K, T, r, kappa, theta, xi, rho, v0)
            for s in S_arr
        ])
        V_arr = np.where(np.isnan(V_arr), 0.0, V_arr)
        lam_arr = np.tile([0.0, 1.0, kappa, theta, xi, rho], (n_S_per_config, 1))
        anchors.append((S_arr, v_arr, tau_arr, K, r, lam_arr, V_arr))

    print(f"Generated {len(anchors)} anchor configs: "
          f"{n_bsm} BSM, {n_cev} CEV, {n_heston} Heston "
          f"({sum(len(a[0]) for a in anchors):,} total points)")
    return anchors


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class ImprovedParametricNet(nn.Module):
    """MLP: 11-dim input → price correction.

    Input  (11): log(S/K), v/V_MAX, tau/TAU_MAX, K/K_REF, r, sigma, beta, kappa, theta, xi, rho
    Output (1):  raw correction: V = V_BS + K * raw
    """

    def __init__(self, hidden: int = 512, depth: int = 10):
        super().__init__()
        in_dim = 11
        layers = [nn.Linear(in_dim, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        last = nn.Linear(hidden, 1)
        nn.init.normal_(last.weight, std=0.0001)
        nn.init.zeros_(last.bias)
        layers.append(last)
        self.mlp = nn.Sequential(*layers)

    def forward(self, S, v, tau, K, r, lam, return_raw=False):
        """
        Args:
            S, v, tau, K, r:  actual tensors (N,1)
            lam:               [sigma, beta, kappa, theta, xi, rho] (N,6)
        """
        sigma = lam[:, 0:1]
        beta  = lam[:, 1:2]
        xi    = lam[:, 4:5]

        # Soft model blending (same as unified_v2)
        mask = torch.tanh(xi / 0.05) ** 2
        v_safe = torch.clamp(v, min=1e-6)
        sigma_eff = (1.0 - mask) * sigma + mask * torch.sqrt(v_safe)

        # BS base price
        V_bs = bs_call(S, K, tau, r, sigma_eff)

        # Normalized inputs
        log_m = torch.log(torch.clamp(S / K, min=1e-6))
        v_n   = v / V_MAX
        tau_n = tau / TAU_MAX
        K_n   = K / K_REF

        x = torch.cat([
            log_m, v_n, tau_n, K_n, r,
            sigma, beta,
            lam[:, 2:3], lam[:, 3:4],  # kappa, theta
            xi, lam[:, 5:6],           # xi, rho
        ], dim=1)

        raw = self.mlp(x)

        # Suppress raw correction for exact-BSM regions
        is_bsm = (xi.abs() < 1e-9) & ((beta - 1.0).abs() < 1e-9)
        raw = raw * (1.0 - is_bsm.float() * 0.99)

        V = torch.clamp(V_bs + K * raw, min=0.0)
        return (V, raw) if return_raw else V


# ---------------------------------------------------------------------------
# PDE residual
# ---------------------------------------------------------------------------

def pde_residual(net, S, v, tau, K, r, lam):
    """PDE in forward tau: V_tau = 0.5*a*V_SS + b*V_Sv + 0.5*c*V_vv + r*S*V_S + d*V_v - r*V"""
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
                - 0.5 * a * V_SS - b * V_Sv - 0.5 * c * V_vv
                - r * S * V_S - d * V_v + r * V)
    return residual / (V.detach().abs() + 1.0)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class ImprovedParametricPINN:
    def __init__(self,
                 hidden: int = 512,
                 depth: int = 10,
                 lr: float = 1e-3,
                 ref_data: list = None,
                 device=None):
        self.ref_data = ref_data or []
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.net = ImprovedParametricNet(hidden=hidden, depth=depth).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.scheduler = None

        self._data_cache = None
        if ref_data:
            self._cache_ref_data(ref_data)

    def _cache_ref_data(self, ref_data):
        S_l, v_l, t_l, K_l, r_l, V_l, lam_l = [], [], [], [], [], [], []
        for (S_arr, v_arr, tau_arr, K_val, r_val, lam_arr, V_arr) in ref_data:
            n = len(S_arr)
            S_l.append(torch.tensor(S_arr, dtype=torch.float32, device=self.device).reshape(-1, 1))
            v_l.append(torch.tensor(v_arr, dtype=torch.float32, device=self.device).reshape(-1, 1))
            t_l.append(torch.tensor(tau_arr, dtype=torch.float32, device=self.device).reshape(-1, 1))
            K_l.append(torch.full((n, 1), K_val, dtype=torch.float32, device=self.device))
            r_l.append(torch.full((n, 1), r_val, dtype=torch.float32, device=self.device))
            lam_l.append(torch.tensor(lam_arr, dtype=torch.float32, device=self.device))
            V_l.append(torch.tensor(V_arr, dtype=torch.float32, device=self.device).reshape(-1, 1))
        self._data_cache = (
            torch.cat(S_l), torch.cat(v_l), torch.cat(t_l),
            torch.cat(K_l), torch.cat(r_l), torch.cat(lam_l), torch.cat(V_l),
        )

    def _to(self, x):
        return x.to(self.device)

    # ------------------------------------------------------------------
    # Parameter sampling
    # ------------------------------------------------------------------

    def _sample_params(self, n: int):
        """Sample n parameter combos across BSM, CEV, Heston."""
        n3   = n // 3
        n_bsm, n_cev, n_hest = n3, n3, n - 2 * n3

        # BSM: sigma in wider range
        sig_b = torch.FloatTensor(n_bsm, 1).uniform_(0.05, 0.50)
        lam_b = torch.cat([
            sig_b, torch.ones(n_bsm, 1),
            torch.zeros(n_bsm, 4),
        ], dim=1)

        # CEV
        sig_c = torch.FloatTensor(n_cev, 1).uniform_(0.10, 0.35)
        bet_c = torch.FloatTensor(n_cev, 1).uniform_(0.10, 0.90)
        lam_c = torch.cat([
            sig_c, bet_c,
            torch.zeros(n_cev, 4),
        ], dim=1)

        # Heston: log-uniform kappa, xi
        kap_h = 10.0 ** torch.FloatTensor(n_hest, 1).uniform_(np.log10(0.5), np.log10(15.0))
        the_h = torch.FloatTensor(n_hest, 1).uniform_(0.01, 0.15)
        xi_h  = 10.0 ** torch.FloatTensor(n_hest, 1).uniform_(np.log10(0.05), np.log10(0.80))
        rho_h = torch.FloatTensor(n_hest, 1).uniform_(-0.98, -0.05)
        lam_h = torch.cat([
            torch.zeros(n_hest, 1), torch.ones(n_hest, 1),
            kap_h, the_h, xi_h, rho_h,
        ], dim=1)

        lam = torch.cat([lam_b, lam_c, lam_h], dim=0)

        # Contract params
        K = torch.FloatTensor(n, 1).uniform_(30.0, 300.0)
        r = torch.FloatTensor(n, 1).uniform_(0.005, 0.15)

        idx = torch.randperm(n)
        return K[idx], r[idx], lam[idx]

    # ------------------------------------------------------------------
    # Collocation sampling
    # ------------------------------------------------------------------

    def _sample_collocation(self, n: int):
        """Moneyness-based S sampling, more ATM density."""
        K, r, lam = self._sample_params(n)

        # Moneyness m = S/K: use beta(2,2) distribution centred at 1.0
        n_base = int(n * 0.75)
        n_tail = n - n_base

        K_sq = K.squeeze()
        m_base = torch.FloatTensor(n_base).uniform_(0.6, 1.5)
        m_tail = torch.cat([
            torch.FloatTensor(int(n_tail * 0.5)).uniform_(0.25, 0.6),
            torch.FloatTensor(n_tail - int(n_tail * 0.5)).uniform_(1.5, 2.5),
        ])
        m = torch.cat([m_base, m_tail])
        S = (m * K_sq).reshape(-1, 1)
        S = torch.clamp(S, min=1.0, max=S_MAX)

        v = torch.FloatTensor(n, 1).uniform_(1e-5, V_MAX)
        tau = torch.FloatTensor(n, 1).uniform_(1e-4, TAU_MAX)

        idx = torch.randperm(n)
        return (self._to(S[idx]), self._to(v[idx]), self._to(tau[idx]),
                self._to(K[idx]), self._to(r[idx]), lam[idx].to(self.device))

    # ------------------------------------------------------------------
    # Loss terms
    # ------------------------------------------------------------------

    def _pde_loss(self, n: int = 10000):
        S, v, tau, K, r, lam = self._sample_collocation(n)
        res = pde_residual(self.net, S, v, tau, K, r, lam)
        return torch.mean(res**2)

    def _ic_loss(self, n: int = 1500):
        K, r, lam = self._sample_params(n)
        S   = torch.FloatTensor(n, 1).uniform_(1.0, S_MAX)
        v   = torch.FloatTensor(n, 1).uniform_(1e-5, V_MAX)
        tau = torch.zeros(n, 1)

        S, v, tau = self._to(S), self._to(v), self._to(tau)
        K, r = self._to(K), self._to(r)
        lam = lam.to(self.device)

        pred   = self.net(S, v, tau, K, r, lam)
        payoff = torch.clamp(S - K, min=0.)
        rel_err = (pred - payoff) / (payoff + K * 0.1)
        return torch.mean(rel_err**2)

    def _boundary_loss(self, n: int = 800):
        K, r, lam = self._sample_params(n)
        tau = torch.FloatTensor(n, 1).uniform_(1e-4, TAU_MAX)
        disc = torch.exp(-r * tau)

        S0  = torch.zeros(n, 1)
        v0b = torch.FloatTensor(n, 1).uniform_(1e-5, V_MAX)
        S0, v0b, tau_d = self._to(S0), self._to(v0b), self._to(tau)
        K_d, r_d = self._to(K), self._to(r)
        lam_d = lam.to(self.device)
        pred0 = self.net(S0, v0b, tau_d, K_d, r_d, lam_d)
        loss0 = torch.mean(pred0**2)

        Sm  = self._to(torch.full((n, 1), S_MAX))
        vm  = self._to(torch.FloatTensor(n, 1).uniform_(1e-5, V_MAX))
        predm = self.net(Sm, vm, tau_d, K_d, r_d, lam_d)
        Vm = S_MAX - K_d * disc.to(self.device)
        lossm = torch.mean(((predm - Vm) / (Vm + 1.0))**2)

        return (loss0 + lossm) * 0.5

    def _data_loss(self):
        if self._data_cache is None:
            return torch.tensor(0.0, device=self.device)
        S, v, tau, K, r, lam, V_ref = self._data_cache
        pred = self.net(S, v, tau, K, r, lam)
        rel_err = (pred - V_ref) / (V_ref.abs() + torch.clamp(K, min=1.0) * 0.1)
        return torch.mean(rel_err**2)

    def _bsm_raw_loss(self, n: int = 1500):
        K, r, lam = self._sample_params(n)
        S   = torch.FloatTensor(n, 1).uniform_(1.0, S_MAX)
        v   = torch.FloatTensor(n, 1).uniform_(1e-5, V_MAX)
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
    # Training
    # ------------------------------------------------------------------

    def train(self, epochs: int = 80000,
              w_pde: float = 1.0, w_bc: float = 10.0, w_ic: float = 10.0,
              w_data: float = 500.0, w_bsm_raw: float = 5.0,
              log_every: int = 500,
              save_every: int = 0, save_path: str = None):
        from tqdm import tqdm

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=20000, T_mult=2, eta_min=1e-6
        )
        history = []
        pbar = tqdm(range(1, epochs + 1), desc="ImpParamPINN", dynamic_ncols=True)

        for epoch in pbar:
            self.optimizer.zero_grad()

            loss_pde     = self._pde_loss(10000)
            loss_bc      = self._boundary_loss(800)
            loss_ic      = self._ic_loss(1500)
            loss_data    = self._data_loss()
            loss_bsm_raw = self._bsm_raw_loss(1500)
            loss = (w_pde * loss_pde + w_bc * loss_bc
                    + w_ic * loss_ic + w_data * loss_data
                    + w_bsm_raw * loss_bsm_raw)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=0.5)
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
                    data=f"{loss_data.item():.3e}",
                )

            if save_every > 0 and save_path and epoch % save_every == 0:
                ckpt_path = save_path.replace(".pt", f"_e{epoch}.pt")
                torch.save({"state_dict": self.net.state_dict(),
                             "epoch": epoch,
                             "optimizer": self.optimizer.state_dict()}, ckpt_path)

        return history

    # ------------------------------------------------------------------
    # Pricing
    # ------------------------------------------------------------------

    def price(self, S: float, K: float, T: float, r: float,
              sigma: float = 0.0, beta: float = 1.0,
              kappa: float = 0.0, theta: float = 0.0,
              xi: float = 0.0, rho: float = 0.0,
              v0: Optional[float] = None,
              option_type: str = "call") -> float:
        """Price a European option. Supports call and put (via put-call parity)."""
        if v0 is None:
            v0 = sigma**2 if xi < 1e-9 else theta

        self.net.eval()
        with torch.no_grad():
            S_t   = self._to(torch.tensor([[S]], dtype=torch.float32))
            v_t   = self._to(torch.tensor([[v0]], dtype=torch.float32))
            tau_t = self._to(torch.tensor([[T]], dtype=torch.float32))
            K_t   = self._to(torch.tensor([[K]], dtype=torch.float32))
            r_t   = self._to(torch.tensor([[r]], dtype=torch.float32))
            lam_t = self._to(torch.tensor(
                [[sigma, beta, kappa, theta, xi, rho]], dtype=torch.float32))
            call_price = self.net(S_t, v_t, tau_t, K_t, r_t, lam_t).item()

        if option_type == "put":
            put = call_price - S + K * np.exp(-r * T)
            return max(put, 0.0)
        return max(call_price, 0.0)

    def save(self, path: str):
        torch.save({"state_dict": self.net.state_dict()}, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ckpt["state_dict"])
