"""
Heston-PINN with data anchor mechanism.

Based on the ICPINN trial solution structure (Tan & Zhang 2026), extended with
GL semi-analytical data anchors to enable convergence for arbitrary parameter sets.

The ICPINN trial solution alone fails to converge when kappa*theta is small
(e.g., kappa=2.0, theta=0.04) because the Fichera degenerate boundary condition
at v=0 dominates the loss and the frozen aux_net cannot correct it.
The data anchor mechanism (w_data=100) pins the solution at t=0 using
GL characteristic function prices, resolving this convergence issue.

Architecture:
  AuxNet:  3 hidden layers x 40 neurons, ELU
  MainNet: 8 hidden layers x 40 neurons, Tanh
  Trial solution: U = payoff(S) + S_n*tau_n * (AuxNet + MainNet)

Default parameters (unified framework evaluation set):
  K=100, T=1, r=0.05, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, v0=0.04
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.integrate import quad


# -- Heston semi-analytical price (GL characteristic function) ---------------
def heston_call_price(S0, K, T, r, kappa, theta, xi, rho, v0):
    def char_func(phi, j):
        if j == 1:
            u, b = 0.5, kappa - rho * xi
        else:
            u, b = -0.5, kappa
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


# -- Weight initialisation (glorot_normal as in paper) ----------------------
def _glorot_normal(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


# -- Auxiliary network: 3 hidden layers x 40 neurons, ELU -------------------
class AuxNet(nn.Module):
    def __init__(self, hidden=40):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden), nn.ELU(),
            nn.Linear(hidden, hidden), nn.ELU(),
            nn.Linear(hidden, hidden), nn.ELU(),
            nn.Linear(hidden, 1),
        )
        self.net.apply(_glorot_normal)

    def forward(self, S_n, v_n, tau_n):
        x = torch.cat([S_n, v_n, tau_n], dim=1)
        return self.net(x)


# -- Main network: 8 hidden layers x 40 neurons, Tanh -----------------------
class MainNet(nn.Module):
    def __init__(self, hidden=40, depth=8):
        super().__init__()
        layers = [nn.Linear(3, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)
        self.net.apply(_glorot_normal)

    def forward(self, S_n, v_n, tau_n):
        x = torch.cat([S_n, v_n, tau_n], dim=1)
        return self.net(x)


# -- PINN trainer ------------------------------------------------------------
class Heston_PINN:
    """
    Heston PINN with data anchor mechanism for stable convergence.

    Trial solution (backward time tau = T - t):
        U(S,v,tau) = payoff(S) + S_n*tau_n * (AuxNet + MainNet)
    where payoff(S) = max(S/K - 1, 0) hard-encodes the terminal condition,
    and S_n*tau_n vanishes at tau=0 and S=0.

    Data anchors: GL semi-analytical prices at t=0 (tau=T) with weight w_data=100
    pin the solution to the correct branch, enabling convergence for any params.
    """
    def __init__(self, K=100.0, T=1.0, r=0.05,
                 kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, v0=0.04,
                 S_max=None, v_max=1.0, device=None):
        self.K = K
        self.T = T
        self.r = r
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.rho = rho
        self.v0 = v0
        self.S_max = S_max if S_max is not None else 4.0 * K
        self.v_max = v_max
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.aux_net  = AuxNet().to(self.device)
        self.main_net = MainNet().to(self.device)

    def _to(self, t):
        return t.to(self.device)

    def _normalise(self, S, v, tau):
        return S / self.S_max, v / self.v_max, tau / self.T

    def _trial(self, S, v, tau):
        """u = U/K in normalised units."""
        S_n, v_n, tau_n = self._normalise(S, v, tau)
        payoff = torch.clamp(S / self.K - 1.0, min=0.0)
        A = self.aux_net(S_n, v_n, tau_n)
        N = self.main_net(S_n, v_n, tau_n)
        B = S_n * tau_n
        return payoff + B * (A + N)

    def _pretrain_aux(self, epochs=3000, n=5000, lr=5e-3):
        opt = torch.optim.Adam(self.aux_net.parameters(), lr=lr)
        for ep in range(1, epochs + 1):
            opt.zero_grad()
            S_0   = self._to(torch.zeros(n, 1))
            v_0   = self._to(torch.FloatTensor(n, 1).uniform_(0, self.v_max))
            tau_0 = self._to(torch.FloatTensor(n, 1).uniform_(0, self.T))
            S_n0, v_n0, tau_n0 = self._normalise(S_0, v_0, tau_0)
            loss_0 = torch.mean(self.aux_net(S_n0, v_n0, tau_n0) ** 2)

            S_int   = self._to(torch.FloatTensor(n, 1).uniform_(0, self.S_max))
            v_int   = self._to(torch.FloatTensor(n, 1).uniform_(0, self.v_max))
            tau_int = self._to(torch.FloatTensor(n, 1).uniform_(0, self.T))
            S_ni, v_ni, tau_ni = self._normalise(S_int, v_int, tau_int)
            loss_int = 0.1 * torch.mean(self.aux_net(S_ni, v_ni, tau_ni) ** 2)

            loss = loss_0 + loss_int
            loss.backward()
            opt.step()
            if ep % 500 == 0:
                print(f"  [aux pretrain] ep {ep:5d}  loss={loss.item():.4e}")

    def _pde_residual(self, S, v, tau):
        S.requires_grad_(True)
        v.requires_grad_(True)
        tau.requires_grad_(True)
        u = self._trial(S, v, tau)

        ones = torch.ones_like(u)
        u_tau = torch.autograd.grad(u, tau, grad_outputs=ones, create_graph=True)[0]
        u_S   = torch.autograd.grad(u, S,   grad_outputs=ones, create_graph=True)[0]
        u_v   = torch.autograd.grad(u, v,   grad_outputs=ones, create_graph=True)[0]
        u_SS  = torch.autograd.grad(u_S, S, grad_outputs=torch.ones_like(u_S), create_graph=True)[0]
        u_vv  = torch.autograd.grad(u_v, v, grad_outputs=torch.ones_like(u_v), create_graph=True)[0]
        u_Sv  = torch.autograd.grad(u_S, v, grad_outputs=torch.ones_like(u_S), create_graph=True)[0]

        res = (u_tau
               - 0.5 * v * S ** 2 * u_SS
               - self.rho * self.xi * v * S * u_Sv
               - 0.5 * self.xi ** 2 * v * u_vv
               - self.r * S * u_S
               - self.kappa * (self.theta - v) * u_v
               + self.r * u)
        return res

    def _build_data_anchors(self, n_data=50):
        S_vals = np.linspace(0.2 * self.K, 3.0 * self.K, n_data)
        u_vals = np.array([
            heston_call_price(s, self.K, self.T, self.r,
                              self.kappa, self.theta, self.xi, self.rho, self.v0)
            / self.K
            for s in S_vals
        ])
        S_t = self._to(torch.tensor(S_vals, dtype=torch.float32).unsqueeze(1))
        v_t = self._to(torch.full((n_data, 1), self.v0, dtype=torch.float32))
        u_t = self._to(torch.tensor(u_vals, dtype=torch.float32).unsqueeze(1))
        return S_t, v_t, u_t

    def train(self, epochs=20000, log_every=2000,
              n_r=10000, n_bc=500, lr=1e-3,
              pretrain_epochs=3000, w_data=100.0):
        print("Pre-training auxiliary network...")
        self._pretrain_aux(epochs=pretrain_epochs, lr=lr)
        for p in self.aux_net.parameters():
            p.requires_grad_(False)

        print("Building semi-analytical data anchors at t=0...")
        S_anc, v_anc, u_anc = self._build_data_anchors(n_data=50)
        tau_anc = self._to(torch.full((50, 1), self.T, dtype=torch.float32))

        opt = torch.optim.Adam(self.main_net.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=5000, gamma=0.75)

        for epoch in range(1, epochs + 1):
            opt.zero_grad()

            S_r   = self._to(torch.FloatTensor(n_r, 1).uniform_(0, self.S_max))
            v_r   = self._to(torch.FloatTensor(n_r, 1).uniform_(0, self.v_max))
            tau_r = self._to(torch.FloatTensor(n_r, 1).uniform_(0, self.T))
            loss_pde = torch.mean(self._pde_residual(S_r, v_r, tau_r) ** 2)

            S_sm   = self._to(torch.full((n_bc, 1), self.S_max))
            v_sm   = self._to(torch.FloatTensor(n_bc, 1).uniform_(0, self.v_max))
            tau_sm = self._to(torch.FloatTensor(n_bc, 1).uniform_(0, self.T))
            S_sm   = S_sm.detach().requires_grad_(True)
            u_sm   = self._trial(S_sm, v_sm, tau_sm)
            du_dS  = torch.autograd.grad(u_sm, S_sm,
                                         grad_outputs=torch.ones_like(u_sm),
                                         create_graph=True)[0]
            loss_Smax = torch.mean((du_dS - 1.0 / self.K) ** 2)

            S_vm   = self._to(torch.FloatTensor(n_bc, 1).uniform_(0, self.S_max))
            v_vm   = self._to(torch.full((n_bc, 1), self.v_max))
            tau_vm = self._to(torch.FloatTensor(n_bc, 1).uniform_(0, self.T))
            v_vm   = v_vm.detach().requires_grad_(True)
            u_vm   = self._trial(S_vm, v_vm, tau_vm)
            du_dv  = torch.autograd.grad(u_vm, v_vm,
                                         grad_outputs=torch.ones_like(u_vm),
                                         create_graph=True)[0]
            loss_vmax = torch.mean(du_dv ** 2)

            S_dg   = self._to(torch.FloatTensor(n_bc, 1).uniform_(0, self.S_max))
            v_dg   = self._to(torch.full((n_bc, 1), 1e-6))
            tau_dg = self._to(torch.FloatTensor(n_bc, 1).uniform_(0, self.T))
            S_dg   = S_dg.detach().requires_grad_(True)
            v_dg   = v_dg.detach().requires_grad_(True)
            tau_dg = tau_dg.detach().requires_grad_(True)
            u_dg   = self._trial(S_dg, v_dg, tau_dg)
            ones_d = torch.ones_like(u_dg)
            du_dS_d   = torch.autograd.grad(u_dg, S_dg,   grad_outputs=ones_d, create_graph=True)[0]
            du_dv_d   = torch.autograd.grad(u_dg, v_dg,   grad_outputs=ones_d, create_graph=True)[0]
            du_dtau_d = torch.autograd.grad(u_dg, tau_dg, grad_outputs=ones_d, create_graph=True)[0]
            deg_res = (S_dg * du_dS_d
                       + self.kappa * self.theta * du_dv_d
                       - self.r * u_dg
                       - du_dtau_d)
            loss_deg = torch.mean(deg_res ** 2)

            u_pred_anc = self._trial(S_anc, v_anc, tau_anc)
            loss_data = torch.mean((u_pred_anc - u_anc) ** 2)

            loss = loss_pde + loss_Smax + loss_vmax + loss_deg + w_data * loss_data
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.main_net.parameters(), 1.0)
            opt.step()
            sched.step()

            if epoch % log_every == 0:
                print(f"[Heston-PINN] epoch {epoch:6d} | "
                      f"loss={loss.item():.4e}  "
                      f"pde={loss_pde.item():.4e}  "
                      f"data={loss_data.item():.4e}")

    def price(self, S, v=None, t=0.0):
        """Price at calendar time t (tau = T - t). Returns absolute price."""
        if v is None:
            v = self.v0
        tau = self.T - t
        self.aux_net.eval()
        self.main_net.eval()
        with torch.no_grad():
            S_t   = self._to(torch.tensor([[float(S)]],   dtype=torch.float32))
            v_t   = self._to(torch.tensor([[float(v)]],   dtype=torch.float32))
            tau_t = self._to(torch.tensor([[float(tau)]], dtype=torch.float32))
            u = self._trial(S_t, v_t, tau_t).item()
            return self.K * u

    def save(self, path):
        torch.save({
            "aux_state":  self.aux_net.state_dict(),
            "main_state": self.main_net.state_dict(),
            "params": dict(K=self.K, T=self.T, r=self.r, kappa=self.kappa,
                           theta=self.theta, xi=self.xi, rho=self.rho,
                           v0=self.v0, S_max=self.S_max, v_max=self.v_max),
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.aux_net.load_state_dict(ckpt["aux_state"])
        self.main_net.load_state_dict(ckpt["main_state"])
