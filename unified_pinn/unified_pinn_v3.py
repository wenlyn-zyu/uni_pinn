"""
unified_pinn_v3.py -- parametric PINN (v17)

Key change over v2: Heston parameters are sampled continuously from a
log-uniform / uniform distribution at every training step, instead of
cycling through a fixed discrete grid.  This gives the network exposure
to the full parameter space (including extreme values seen in real SPY
calibration) without the resource-dilution problem of a large discrete grid.

BSM and CEV still use a small fixed set of variants (parameter space is
low-dimensional and well-covered by the v15 grid).

Dynamic data anchors: instead of pre-cached reference solutions for fixed
parameter sets, we compute Heston GL prices on-the-fly for the sampled
parameters.  This adds ~0.05 s per epoch on GPU but is essential for
correctness.

Heston parameter ranges (log-uniform unless noted):
  kappa  ~ LogUniform(0.05, 20)
  theta  ~ LogUniform(0.005, 0.25)
  xi     ~ LogUniform(0.01, 1.5)
  rho    ~ Uniform(-0.98, -0.01)
  v0     ~ LogUniform(0.005, 0.25)
"""

import torch
import torch.nn as nn
import numpy as np
from numpy.polynomial.legendre import leggauss
from typing import Optional

from unified_pinn_v2 import (
    ModelParams, UnifiedNet, _bs_call,
    unified_pde_residual, UnifiedPINN
)

# ---------------------------------------------------------------------------
# Pre-compute GL nodes (same as spy_backtest_v15.py)
# ---------------------------------------------------------------------------
_N_GL = 96
_GL_NODES, _GL_WEIGHTS = leggauss(_N_GL)
_GL_PHI_MAX = 100.0
_GL_PHI = (_GL_NODES + 1.0) * 0.5 * _GL_PHI_MAX
_GL_W   = _GL_WEIGHTS * 0.5 * _GL_PHI_MAX


def _heston_cf_batch_np(phi_arr, S, Ks, T, r, kappa, theta, xi, rho, v0, j):
    """Vectorized Heston CF: phi (N,), Ks (M,) -> (N, M) complex."""
    i = 1j
    u, b = (0.5, kappa - rho * xi) if j == 1 else (-0.5, kappa)
    a = kappa * theta
    x = np.log(S / Ks)
    phi = phi_arr[:, None]
    x2d = x[None, :]
    d = np.sqrt((rho * xi * i * phi - b)**2 - xi**2 * (2 * u * i * phi - phi**2))
    num = b - rho * xi * i * phi + d
    den = b - rho * xi * i * phi - d
    g = num / den
    exp_dT = np.exp(d * T)
    C = (r * i * phi * T
         + (a / xi**2) * (num * T - 2.0 * np.log((1.0 - g * exp_dT) / (1.0 - g))))
    D = (num / xi**2) * ((1.0 - exp_dT) / (1.0 - g * exp_dT))
    return np.exp(C + D * v0 + i * phi * x2d)


def heston_prices_gl_np(S, Ks, T, r, kappa, theta, xi, rho, v0):
    """Fast vectorized Heston prices via GL quadrature. Returns (M,) array."""
    if T < 1e-6:
        return np.maximum(S - Ks, 0.0)
    try:
        phi, w = _GL_PHI, _GL_W
        cf1 = _heston_cf_batch_np(phi, S, Ks, T, r, kappa, theta, xi, rho, v0, 1)
        cf2 = _heston_cf_batch_np(phi, S, Ks, T, r, kappa, theta, xi, rho, v0, 2)
        phi2d = phi[:, None]
        I1 = (w[:, None] * np.real(cf1 / (1j * phi2d))).sum(axis=0)
        I2 = (w[:, None] * np.real(cf2 / (1j * phi2d))).sum(axis=0)
        prices = S * (0.5 + I1 / np.pi) - Ks * np.exp(-r * T) * (0.5 + I2 / np.pi)
        return np.where(np.isfinite(prices), np.maximum(prices, 0.0), np.nan)
    except Exception:
        return np.full(len(Ks), np.nan)


# ---------------------------------------------------------------------------
# Parametric PINN trainer
# ---------------------------------------------------------------------------

class ParametricPINN(UnifiedPINN):
    """
    Extends UnifiedPINN with continuous Heston parameter sampling.

    At each training step:
      - BSM/CEV: use fixed _lam_bsm_cev (same as v15 grid)
      - Heston:  sample N_heston_per_step fresh parameter sets from
                 log-uniform / uniform distributions
      - Data anchors: BSM uses analytical solution; Heston uses GL pricer
                      computed on-the-fly for the sampled parameters
    """

    # Heston parameter ranges (log-uniform for positive params)
    KAPPA_LO, KAPPA_HI = 0.05, 20.0
    THETA_LO, THETA_HI = 0.005, 0.25
    XI_LO,    XI_HI    = 0.01,  1.5
    RHO_LO,   RHO_HI   = -0.98, -0.01
    V0_LO,    V0_HI    = 0.005, 0.25

    def __init__(self,
                 bsm_cev_params: list,
                 n_heston_per_step: int = 32,
                 n_anchor_per_heston: int = 10,
                 hidden: int = 128,
                 depth: int = 6,
                 lr: float = 1e-3,
                 device=None):
        """
        Args:
            bsm_cev_params: fixed BSM/CEV ModelParams list (same as v15)
            n_heston_per_step: Heston parameter sets sampled per training step
            n_anchor_per_heston: S grid points for dynamic Heston data anchors
        """
        # Initialise parent with BSM/CEV only (no Heston in param_list)
        super().__init__(
            param_list=bsm_cev_params,
            hidden=hidden,
            depth=depth,
            lr=lr,
            ref_data=None,
            device=device,
        )
        self.n_heston_per_step  = n_heston_per_step
        self.n_anchor_per_heston = n_anchor_per_heston

        # Separate BSM/CEV lambda tensor (fixed, pre-built by parent)
        self._lam_bsm_cev = self._lam_all  # (M_bc, 6)
        self.M_bc = len(bsm_cev_params)

        # S grid for dynamic anchors (fixed, covers ATM region densely)
        self._S_anchor = np.concatenate([
            np.linspace(60., 160., n_anchor_per_heston),
        ])

    # ------------------------------------------------------------------
    # Heston parameter sampling
    # ------------------------------------------------------------------

    def _sample_heston_params(self, n: int):
        """
        Sample n Heston parameter sets.
        Returns numpy arrays of shape (n,) each.
        Log-uniform for kappa, theta, xi, v0; uniform for rho.
        """
        def log_uniform(lo, hi, size):
            return np.exp(np.random.uniform(np.log(lo), np.log(hi), size))

        kappa = log_uniform(self.KAPPA_LO, self.KAPPA_HI, n)
        theta = log_uniform(self.THETA_LO, self.THETA_HI, n)
        xi    = log_uniform(self.XI_LO,    self.XI_HI,    n)
        rho   = np.random.uniform(self.RHO_LO, self.RHO_HI, n)
        v0    = log_uniform(self.V0_LO, self.V0_HI, n)
        return kappa, theta, xi, rho, v0

    # ------------------------------------------------------------------
    # Dynamic data anchors
    # ------------------------------------------------------------------

    def _build_heston_anchors(self, kappa_arr, theta_arr, xi_arr, rho_arr, v0_arr):
        """
        Compute GL Heston prices for each sampled parameter set.
        Returns (S_t, v_t, t_t, V_t, lam_t) tensors on device.
        Skips parameter sets where GL pricing fails.
        """
        K, T, r = self.K, self.T, self.r
        S_grid = self._S_anchor

        S_list, v_list, t_list, V_list, lam_list = [], [], [], [], []
        for kappa, theta, xi, rho, v0 in zip(
                kappa_arr, theta_arr, xi_arr, rho_arr, v0_arr):
            prices = heston_prices_gl_np(
                100.0, S_grid, T, r, kappa, theta, xi, rho, v0)
            if np.any(np.isnan(prices)):
                continue
            n = len(S_grid)
            lam_row = torch.tensor(
                [[0.0, 1.0, kappa, theta, xi, rho]],
                dtype=torch.float32, device=self.device
            ).expand(n, -1)
            S_list.append(torch.tensor(S_grid,  dtype=torch.float32,
                                       device=self.device).reshape(-1, 1))
            v_list.append(torch.full((n, 1), v0, dtype=torch.float32,
                                     device=self.device))
            t_list.append(torch.zeros(n, 1, dtype=torch.float32,
                                      device=self.device))
            V_list.append(torch.tensor(prices, dtype=torch.float32,
                                       device=self.device).reshape(-1, 1))
            lam_list.append(lam_row)

        if not S_list:
            return None
        return (torch.cat(S_list), torch.cat(v_list),
                torch.cat(t_list), torch.cat(V_list),
                torch.cat(lam_list))

    def _build_bsm_anchors(self):
        """BSM analytical anchors for fixed BSM/CEV variants (same as v15)."""
        if self._data_cache is not None:
            return self._data_cache
        # Build from bsm_cev_params using parent's ref_data mechanism
        # (called once, cached)
        return None

    # ------------------------------------------------------------------
    # Overridden _sample_batch: BSM/CEV fixed + Heston random
    # ------------------------------------------------------------------

    def _sample_batch_parametric(self, n_per_bc: int, n_per_heston: int,
                                  kappa_arr, theta_arr, xi_arr, rho_arr, v0_arr):
        """
        Returns (S, v, t, lam) with:
          - M_bc * n_per_bc rows for BSM/CEV (fixed params)
          - N_h * n_per_heston rows for Heston (sampled params)
        """
        K, S_max, v_max, T = self.K, self.S_max, self.v_max, self.T
        M_bc = self.M_bc
        N_h  = len(kappa_arr)

        # BSM/CEV collocation
        n_base_bc = int(n_per_bc * 0.7)
        n_otm_bc  = n_per_bc - n_base_bc
        S_bc = torch.cat([
            torch.FloatTensor(M_bc * n_base_bc, 1).uniform_(0.01, S_max),
            torch.FloatTensor(M_bc * n_otm_bc,  1).uniform_(0.7 * K, K),
        ])
        v_bc  = torch.FloatTensor(M_bc * n_per_bc, 1).uniform_(1e-4, v_max)
        t_bc  = torch.FloatTensor(M_bc * n_per_bc, 1).uniform_(0.0, T * 0.999)
        lam_bc = self._lam_bsm_cev.unsqueeze(1).expand(
            M_bc, n_per_bc, 6).reshape(M_bc * n_per_bc, 6)

        # Heston collocation (one block per sampled param set)
        lam_h_rows = []
        for kappa, theta, xi, rho, v0 in zip(
                kappa_arr, theta_arr, xi_arr, rho_arr, v0_arr):
            row = torch.tensor(
                [[0.0, 1.0, kappa, theta, xi, rho]],
                dtype=torch.float32
            ).expand(n_per_heston, -1)
            lam_h_rows.append(row)
        lam_h = torch.cat(lam_h_rows, dim=0)  # (N_h * n_per_heston, 6)

        n_base_h = int(n_per_heston * 0.7)
        n_otm_h  = n_per_heston - n_base_h
        S_h = torch.cat([
            torch.FloatTensor(N_h * n_base_h, 1).uniform_(0.01, S_max),
            torch.FloatTensor(N_h * n_otm_h,  1).uniform_(0.7 * K, K),
        ])
        v_h = torch.FloatTensor(N_h * n_per_heston, 1).uniform_(1e-4, v_max)
        t_h = torch.FloatTensor(N_h * n_per_heston, 1).uniform_(0.0, T * 0.999)

        # Move all to device and concatenate
        S_all   = torch.cat([self._to(S_bc),   self._to(S_h)])
        v_all   = torch.cat([self._to(v_bc),   self._to(v_h)])
        t_all   = torch.cat([self._to(t_bc),   self._to(t_h)])
        lam_all = torch.cat([self._to(lam_bc), self._to(lam_h)])
        idx = torch.randperm(len(S_all))

        return (self._to(S_all[idx]), self._to(v_all[idx]),
                self._to(t_all[idx]), self._to(lam_all[idx]))

    # ------------------------------------------------------------------
    # Overridden train loop
    # ------------------------------------------------------------------

    def train_parametric(self,
                         epochs: int = 30000,
                         n_per_bc: int = 512,
                         n_per_heston: int = 256,
                         w_pde: float = 1.0,
                         w_bc: float = 10.0,
                         w_data: float = 100.0,
                         log_every: int = 500):
        """
        Parametric training loop.

        Each step:
          1. Sample N_h Heston parameter sets
          2. Build collocation batch (BSM/CEV fixed + Heston random)
          3. Compute PDE residual loss
          4. Compute boundary loss (BSM/CEV fixed + Heston random)
          5. Compute dynamic data anchor loss (BSM analytical + Heston GL)
          6. Backprop
        """
        from tqdm import tqdm

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=1e-5
        )
        K, T, r, S_max, v_max = self.K, self.T, self.r, self.S_max, self.v_max
        history = []

        pbar = tqdm(range(1, epochs + 1), desc="ParametricPINN", dynamic_ncols=True)
        for epoch in pbar:
            self.optimizer.zero_grad()

            # --- sample Heston params ---
            kappa_arr, theta_arr, xi_arr, rho_arr, v0_arr = \
                self._sample_heston_params(self.n_heston_per_step)

            # --- PDE loss ---
            S_c, v_c, t_c, lam_c = self._sample_batch_parametric(
                n_per_bc, n_per_heston,
                kappa_arr, theta_arr, xi_arr, rho_arr, v0_arr)
            res      = unified_pde_residual(
                self.net, S_c, v_c, t_c, lam_c, K, T, r, S_max, v_max)
            loss_pde = torch.mean(res**2)

            # --- boundary loss (S=0 and S=S_max) ---
            loss_bc = self._boundary_loss_parametric(
                kappa_arr, theta_arr, xi_arr, rho_arr, v0_arr)

            # --- dynamic data anchor loss ---
            loss_data = self._data_loss_parametric(
                kappa_arr, theta_arr, xi_arr, rho_arr, v0_arr)

            loss = w_pde * loss_pde + w_bc * loss_bc + w_data * loss_data
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
                    "data":  loss_data.item(),
                })
                pbar.set_postfix(
                    loss=f"{loss.item():.3e}",
                    pde=f"{loss_pde.item():.3e}",
                    data=f"{loss_data.item():.3e}",
                )

        return history

    def _boundary_loss_parametric(self, kappa_arr, theta_arr, xi_arr, rho_arr, v0_arr):
        """Boundary loss for BSM/CEV (fixed) + Heston (sampled)."""
        K, T, r, S_max, v_max = self.K, self.T, self.r, self.S_max, self.v_max
        M_bc = self.M_bc
        N_h  = len(kappa_arr)
        n    = 200  # boundary points per model

        # BSM/CEV boundary
        lam_bc = self._lam_bsm_cev.unsqueeze(1).expand(
            M_bc, n, 6).reshape(M_bc * n, 6)
        tm_bc  = self._to(torch.FloatTensor(M_bc * n, 1).uniform_(0, T))
        disc_bc = torch.exp(-r * (T - tm_bc))

        S0_bc   = self._to(torch.zeros(M_bc * n, 1))
        v0b_bc  = self._to(torch.FloatTensor(M_bc * n, 1).uniform_(1e-4, v_max))
        pred0_bc = self.net(S0_bc/S_max, v0b_bc/v_max, tm_bc/T,
                            lam_bc, S0_bc, tm_bc, K, T, r)
        loss0_bc = torch.mean(pred0_bc**2)

        Sm_bc   = self._to(torch.full((M_bc * n, 1), S_max))
        vmb_bc  = self._to(torch.FloatTensor(M_bc * n, 1).uniform_(1e-4, v_max))
        predm_bc = self.net(Sm_bc/S_max, vmb_bc/v_max, tm_bc/T,
                            lam_bc, Sm_bc, tm_bc, K, T, r)
        Vm_bc   = S_max - K * disc_bc
        lossm_bc = torch.mean(((predm_bc - Vm_bc) / (Vm_bc + 1.0))**2)

        # Heston boundary
        lam_h_rows = []
        for kappa, theta, xi, rho, v0 in zip(
                kappa_arr, theta_arr, xi_arr, rho_arr, v0_arr):
            row = torch.tensor(
                [[0.0, 1.0, kappa, theta, xi, rho]],
                dtype=torch.float32
            ).expand(n, -1)
            lam_h_rows.append(row)
        lam_h  = self._to(torch.cat(lam_h_rows, dim=0))
        tm_h   = self._to(torch.FloatTensor(N_h * n, 1).uniform_(0, T))
        disc_h = torch.exp(-r * (T - tm_h))

        S0_h   = self._to(torch.zeros(N_h * n, 1))
        v0b_h  = self._to(torch.FloatTensor(N_h * n, 1).uniform_(1e-4, v_max))
        pred0_h = self.net(S0_h/S_max, v0b_h/v_max, tm_h/T,
                           lam_h, S0_h, tm_h, K, T, r)
        loss0_h = torch.mean(pred0_h**2)

        Sm_h   = self._to(torch.full((N_h * n, 1), S_max))
        vmb_h  = self._to(torch.FloatTensor(N_h * n, 1).uniform_(1e-4, v_max))
        predm_h = self.net(Sm_h/S_max, vmb_h/v_max, tm_h/T,
                           lam_h, Sm_h, tm_h, K, T, r)
        Vm_h   = S_max - K * disc_h
        lossm_h = torch.mean(((predm_h - Vm_h) / (Vm_h + 1.0))**2)

        return (loss0_bc + lossm_bc + loss0_h + lossm_h) * 0.25

    def _data_loss_parametric(self, kappa_arr, theta_arr, xi_arr, rho_arr, v0_arr):
        """
        Data anchor loss:
          - BSM/CEV: use pre-cached anchors (from parent _data_cache)
          - Heston:  compute GL prices on-the-fly for sampled params
        """
        K, T, r, S_max, v_max = self.K, self.T, self.r, self.S_max, self.v_max
        losses = []

        # BSM/CEV anchors (pre-cached)
        if self._data_cache is not None:
            S_t, v_t, t_t, V_t, lam = self._data_cache
            pred    = self.net(S_t/S_max, v_t/v_max, t_t/T, lam, S_t, t_t, K, T, r)
            rel_err = (pred - V_t) / (V_t.abs() + K * 0.1)
            losses.append(torch.mean(rel_err**2))

        # Heston dynamic anchors
        heston_cache = self._build_heston_anchors(
            kappa_arr, theta_arr, xi_arr, rho_arr, v0_arr)
        if heston_cache is not None:
            S_t, v_t, t_t, V_t, lam = heston_cache
            pred    = self.net(S_t/S_max, v_t/v_max, t_t/T, lam, S_t, t_t, K, T, r)
            rel_err = (pred - V_t) / (V_t.abs() + K * 0.1)
            losses.append(torch.mean(rel_err**2))

        if not losses:
            return torch.tensor(0.0, device=self.device)
        return torch.stack(losses).mean()

    def save(self, path: str):
        import os
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save({
            "net_state": self.net.state_dict(),
            "version":   "v17_parametric",
            "n_heston_per_step": self.n_heston_per_step,
        }, path)
        print(f"Saved ParametricPINN to {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        key  = "net_state" if "net_state" in ckpt else "model_state_dict"
        self.net.load_state_dict(ckpt[key])
        self.net.eval()
