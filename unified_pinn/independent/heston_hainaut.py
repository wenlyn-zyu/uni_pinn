"""
Strict reproduction of Hainaut & Casas (2024) parametric PINN for Heston.

Reference:
  Hainaut, D. & Casas, A. (2024). Option Pricing in the Heston Model with
  Physics Inspired Neural Networks. Annals of Finance, 20, 353-376.
  DOI: 10.1007/s10436-024-00452-7

Key design choices (from paper):
  - PUT option, K=100 fixed
  - Parametric: inputs include (t, S, V, r, kappa, theta, xi, rho, T) — 9 dims
  - Skip connections: each hidden layer receives concatenated [prev_output, x_input]
  - Z-score normalisation of state variables (S, V) and time
  - Soft constraints: terminal and lower boundary via loss terms
  - 4 hidden layers x 256 neurons
  - 4-phase Adam schedule: lr 0.005/0.002/0.001/0.0001 for 500/1000/1000/1000 epochs
  - n_D=20000 interior, n_T=5000 terminal, n_low=5000 lower boundary

Training parameter ranges (Table 1):
  S in [20, 180], V in [0.032, 0.52]
  r in [0.01, 0.07], kappa in [0.062, 0.42]
  theta in [0.5, 2], xi in [0.1, 0.9], rho in [-0.8, 0.8]
  T in [0, 5] years

Default evaluation parameters (Table 6):
  kappa=1.15, r=0.04, theta=0.202, xi=0.20, rho=-0.40
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.integrate import quad


# ---------------------------------------------------------------------------
# Heston semi-analytical put price
# ---------------------------------------------------------------------------
def heston_put_price(S0, K, T, r, kappa, theta, xi, rho, v0):
    """Heston put price via put-call parity from semi-analytical call."""
    call = heston_call_price(S0, K, T, r, kappa, theta, xi, rho, v0)
    return call - S0 + K * np.exp(-r * T)


def heston_call_price(S0, K, T, r, kappa, theta, xi, rho, v0):
    def char_func(phi, j):
        u = 0.5 if j == 1 else -0.5
        b = kappa - rho * xi if j == 1 else kappa
        d = np.sqrt((rho * xi * phi * 1j - b) ** 2
                    - xi ** 2 * (2 * u * phi * 1j - phi ** 2))
        g = (b - rho * xi * phi * 1j + d) / (b - rho * xi * phi * 1j - d)
        C = (r * phi * 1j * T
             + kappa * theta / xi ** 2
             * ((b - rho * xi * phi * 1j + d) * T
                - 2 * np.log((1 - g * np.exp(d * T)) / (1 - g))))
        D = ((b - rho * xi * phi * 1j + d) / xi ** 2
             * (1 - np.exp(d * T)) / (1 - g * np.exp(d * T)))
        return np.exp(C + D * v0 + 1j * phi * np.log(S0))

    def integrand(phi, j):
        return np.real(np.exp(-1j * phi * np.log(K))
                       * char_func(phi, j) / (1j * phi))

    P = np.zeros(2)
    for j in [1, 2]:
        P[j - 1] = 0.5 + (1 / np.pi) * quad(
            integrand, 1e-6, 200, args=(j,), limit=500)[0]
    return S0 * P[0] - K * np.exp(-r * T) * P[1]


# ---------------------------------------------------------------------------
# Skip-connection network (Eq. 4-6 in paper)
# ---------------------------------------------------------------------------
class SkipNet(nn.Module):
    """
    Feed-forward network with skip connections.
    Each hidden layer k receives [h_{k-1}, x_input] concatenated.
    Architecture: input_dim -> 256 -> 256 -> 256 -> 256 -> 1
    """
    def __init__(self, input_dim=9, hidden=256, depth=4):
        super().__init__()
        self.input_dim = input_dim
        self.hidden = hidden

        self.layer0 = nn.Linear(input_dim, hidden)
        self.layers = nn.ModuleList([
            nn.Linear(hidden + input_dim, hidden)
            for _ in range(depth - 1)
        ])
        self.out = nn.Linear(hidden, 1)
        self.act = nn.Tanh()
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        h = self.act(self.layer0(x))
        for layer in self.layers:
            h = self.act(layer(torch.cat([h, x], dim=1)))
        return self.out(h)


# ---------------------------------------------------------------------------
# Hainaut & Casas PINN
# ---------------------------------------------------------------------------
class HestonHainaut:
    """
    Parametric PINN for Heston put option pricing.
    Single network prices across a range of parameters without retraining.

    Input vector (9-dim, z-score normalised):
      (t, S, V, r, kappa, theta, xi, rho, T)
    Output: put option price V(t, S, V | params)
    """

    S_RANGE    = (20.0,   180.0)
    V_RANGE    = (0.032,  0.52)
    R_RANGE    = (0.01,   0.07)
    K_RANGE    = (0.5,    2.0)
    TH_RANGE   = (0.062,  0.42)
    XI_RANGE   = (0.1,    0.9)
    RHO_RANGE  = (-0.8,   0.8)
    T_RANGE    = (0.1,    5.0)

    K_STRIKE   = 100.0

    def __init__(self, device=None):
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("cpu"))
        self.net = SkipNet(input_dim=9, hidden=256, depth=4).to(self.device)
        self.mean_ = None
        self.std_  = None

    def _to(self, t):
        return t.to(self.device)

    def _sample_params(self, n):
        def u(lo, hi): return torch.FloatTensor(n, 1).uniform_(lo, hi)
        S     = u(*self.S_RANGE)
        V     = u(*self.V_RANGE)
        r     = u(*self.R_RANGE)
        kappa = u(*self.K_RANGE)
        theta = u(*self.TH_RANGE)
        xi    = u(*self.XI_RANGE)
        rho   = u(*self.RHO_RANGE)
        T     = u(*self.T_RANGE)
        return S, V, r, kappa, theta, xi, rho, T

    def _compute_stats(self, n=50000):
        S, V, r, kappa, theta, xi, rho, T = self._sample_params(n)
        t = torch.FloatTensor(n, 1).uniform_(0, 1) * T

        raw = torch.cat([t, S, V, r, kappa, theta, xi, rho, T], dim=1)
        self.mean_ = raw.mean(dim=0, keepdim=True).to(self.device)
        self.std_  = raw.std(dim=0, keepdim=True).clamp(min=1e-6).to(self.device)

    def _normalise_input(self, t, S, V, r, kappa, theta, xi, rho, T):
        raw = torch.cat([t, S, V, r, kappa, theta, xi, rho, T], dim=1)
        return (raw - self.mean_) / self.std_

    def _forward(self, t, S, V, r, kappa, theta, xi, rho, T):
        x = self._normalise_input(t, S, V, r, kappa, theta, xi, rho, T)
        return self.net(x)

    def _pde_residual(self, t, S, V, r, kappa, theta, xi, rho, T):
        S.requires_grad_(True)
        V.requires_grad_(True)
        t.requires_grad_(True)

        F = self._forward(t, S, V, r, kappa, theta, xi, rho, T)
        ones = torch.ones_like(F)

        F_t  = torch.autograd.grad(F, t,  grad_outputs=ones, create_graph=True)[0]
        F_S  = torch.autograd.grad(F, S,  grad_outputs=ones, create_graph=True)[0]
        F_V  = torch.autograd.grad(F, V,  grad_outputs=ones, create_graph=True)[0]
        F_SS = torch.autograd.grad(F_S, S, grad_outputs=torch.ones_like(F_S), create_graph=True)[0]
        F_VV = torch.autograd.grad(F_V, V, grad_outputs=torch.ones_like(F_V), create_graph=True)[0]
        F_SV = torch.autograd.grad(F_S, V, grad_outputs=torch.ones_like(F_S), create_graph=True)[0]

        res = (F_t
               - r * F
               + (r - 0.5 * V) * S * F_S
               + kappa * (theta - V) * F_V
               + 0.5 * V * S ** 2 * F_SS
               + rho * xi * V * S * F_SV
               + 0.5 * xi ** 2 * V * F_VV)
        return res / self.K_STRIKE

    def train(self, log_every=500):
        """
        4-phase Adam training (Table 2):
          Phase 1: lr=0.005, 500 epochs
          Phase 2: lr=0.002, 1000 epochs
          Phase 3: lr=0.001, 1000 epochs
          Phase 4: lr=0.0001, 1000 epochs
        n_D=20000, n_T=5000, n_low=5000
        """
        print("Computing z-score statistics...")
        self._compute_stats()

        phases = [
            (0.005,  500),
            (0.002, 1000),
            (0.001, 1000),
            (0.0001, 1000),
        ]
        n_D, n_T, n_low = 20000, 5000, 5000
        epoch = 0

        for phase_idx, (lr, n_epochs) in enumerate(phases):
            opt = torch.optim.Adam(self.net.parameters(), lr=lr)
            print(f"\n[Phase {phase_idx+1}] lr={lr}, {n_epochs} epochs")

            for _ in range(n_epochs):
                epoch += 1
                opt.zero_grad()

                S_r, V_r, r_r, k_r, th_r, xi_r, rho_r, T_r = [
                    self._to(x) for x in self._sample_params(n_D)]
                t_r = self._to(torch.FloatTensor(n_D, 1).uniform_(0, 1) * T_r.cpu()).requires_grad_(True)
                S_r = S_r.requires_grad_(True)
                V_r = V_r.requires_grad_(True)
                loss_D = torch.mean(
                    self._pde_residual(t_r, S_r, V_r, r_r, k_r, th_r, xi_r, rho_r, T_r) ** 2)

                S_T, V_T, r_T, k_T, th_T, xi_T, rho_T, T_T = [
                    self._to(x) for x in self._sample_params(n_T)]
                t_T = T_T.clone()
                F_T = self._forward(t_T, S_T, V_T, r_T, k_T, th_T, xi_T, rho_T, T_T)
                payoff = torch.clamp(self.K_STRIKE - S_T, min=0.0) / self.K_STRIKE
                loss_T = torch.mean((F_T / self.K_STRIKE - payoff) ** 2)

                _, V_l, r_l, k_l, th_l, xi_l, rho_l, T_l = [
                    self._to(x) for x in self._sample_params(n_low)]
                S_l = self._to(torch.zeros(n_low, 1))
                t_l = self._to(torch.FloatTensor(n_low, 1).uniform_(0, 1) * T_l.cpu())
                F_l = self._forward(t_l, S_l, V_l, r_l, k_l, th_l, xi_l, rho_l, T_l)
                tau_l = (T_l - t_l).clamp(min=0.0)
                bc_l  = self.K_STRIKE * torch.exp(-r_l * tau_l) / self.K_STRIKE
                loss_low = torch.mean((F_l / self.K_STRIKE - bc_l) ** 2)

                loss = loss_D + loss_T + loss_low
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()

                if epoch % log_every == 0:
                    print(f"  epoch {epoch:5d} | loss={loss.item():.4e}  "
                          f"D={loss_D.item():.4e}  T={loss_T.item():.4e}  "
                          f"low={loss_low.item():.4e}")

    def price(self, S, V, t, T, r, kappa, theta, xi, rho):
        """Price a put option. t: current time (t=T means 'now')."""
        self.net.eval()
        def _t(v): return self._to(torch.tensor([[float(v)]], dtype=torch.float32))
        with torch.no_grad():
            out = self._forward(_t(t), _t(S), _t(V), _t(r),
                                _t(kappa), _t(theta), _t(xi), _t(rho), _t(T))
        return out.item()

    def save(self, path):
        torch.save({
            "net_state": self.net.state_dict(),
            "mean": self.mean_.cpu(),
            "std":  self.std_.cpu(),
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ckpt["net_state"])
        self.mean_ = ckpt["mean"].to(self.device)
        self.std_  = ckpt["std"].to(self.device)
