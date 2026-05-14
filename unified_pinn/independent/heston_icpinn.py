"""
Strict reproduction of Tan & Zhang (2026) ICPINN for Heston model.

Reference:
  Tan, J. & Zhang, X. (2026). Improved constrained physics-informed neural
  networks (ICPINNs) to solve PDE and its application to option pricing.
  Mathematics and Computers in Simulation, 241, 908-924.

Paper parameters (Section 3.2):
  K=1, r=0.1, kappa=1, theta=0.08, xi=0.39, rho=-0.93
  S_max=4*K=4, v_max=1, T=1
  N_interior=10000, N_bc=500

Time convention (from paper):
  tau in [0, T], tau=0 is EXPIRY, tau=T is "now" (backward time).
  The PDE is: LU = 0.5*v*S^2*U_SS + rho*xi*v*S*U_Sv + 0.5*xi^2*v*U_vv
                   + r*S*U_S + kappa*(theta-v)*U_v - r*U - U_tau = 0

Trial solution (Eq. 3.11):
  Psi(S,v,tau) = phi(S,v,tau) + psi(S,v,tau) * NN(S,v,tau)
  psi(S,v,tau) = tau * v   <-- vanishes at tau=0 (expiry) AND v=0 (degenerate)
  phi: aux network pre-trained to satisfy Dirichlet BCs:
    phi(S,v,0) = max(S-K, 0)   [payoff at tau=0]
    phi(0,v,tau) = 0            [S=0 boundary]

Loss:
  L = L_pde + L_Smax + L_vmax + L_deg
  L_pde  : PDE residual at interior points
  L_Smax : dU/dS = 1 at S=S_max  (Neumann, deep ITM)
  L_vmax : dU/dv = 0 at v=v_max  (Neumann, large vol)
  L_deg  : Fichera BC at v=0: S*dU/dS + kappa*theta*dU/dv - r*U - dU/dtau = 0

K scaling:
  Paper uses K=1. We work internally in K=1 units (S_int=S/K, price_int=price/K)
  so all network quantities are O(1) regardless of K.

Note on convergence:
  This strict reproduction converges well for the paper's parameters
  (r=0.1, kappa=1, theta=0.08, xi=0.39, rho=-0.93).
  For the unified framework parameters (r=0.05, kappa=2.0, theta=0.04, xi=0.3,
  rho=-0.7), use heston_pinn.py which adds data anchors for stable convergence.
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.integrate import quad


# ---------------------------------------------------------------------------
# Heston semi-analytical call price
# ---------------------------------------------------------------------------
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
# Weight initialisation (glorot_normal as in paper)
# ---------------------------------------------------------------------------
def _glorot_normal(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


# ---------------------------------------------------------------------------
# Auxiliary network: 3 hidden layers x 40 neurons, ELU (Table 4)
# ---------------------------------------------------------------------------
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
        return self.net(torch.cat([S_n, v_n, tau_n], dim=1))


# ---------------------------------------------------------------------------
# Main network: 8 hidden layers x 40 neurons, Tanh (Table 3)
# ---------------------------------------------------------------------------
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
        return self.net(torch.cat([S_n, v_n, tau_n], dim=1))


# ---------------------------------------------------------------------------
# ICPINN trainer
# ---------------------------------------------------------------------------
class HestonICPINN:
    """
    Strict reproduction of Tan & Zhang (2026) ICPINN.

    Time convention: tau in [0,T], tau=0 is expiry, tau=T is 'now'.
    Trial solution: U = K * (phi(S_n,v_n,tau_n) + tau_n*v_n * NN(S_n,v_n,tau_n))
      - psi = tau_n * v_n vanishes at tau=0 (terminal) and v=0 (degenerate BC)
      - phi pre-trained to satisfy: phi(S,v,0)=max(S-K,0), phi(0,v,tau)=0
    All internal quantities in K=1 units for O(1) network outputs.
    """

    def __init__(self, K=1.0, T=1.0, r=0.1,
                 kappa=1.0, theta=0.08, xi=0.39, rho=-0.93, v0=0.04,
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
            else torch.device("cpu"))
        self.aux_net  = AuxNet().to(self.device)
        self.main_net = MainNet().to(self.device)

    def _to(self, t):
        return t.to(self.device)

    def _normalise(self, S, v, tau):
        """Scale inputs to [0,1]: S_n=S/S_max, v_n=v/v_max, tau_n=tau/T."""
        return S / self.S_max, v / self.v_max, tau / self.T

    def _trial(self, S, v, tau):
        """
        U = K * (phi(S_n,v_n,tau_n) + tau_n*v_n * NN(S_n,v_n,tau_n))
        psi = tau_n * v_n  in [0,1], vanishes at tau=0 and v=0.
        phi and NN output O(1) values (K=1 units).
        """
        S_n, v_n, tau_n = self._normalise(S, v, tau)
        phi   = self.aux_net(S_n, v_n, tau_n)
        nn_out = self.main_net(S_n, v_n, tau_n)
        psi   = tau_n * v_n
        return self.K * (phi + psi * nn_out)

    # ------------------------------------------------------------------
    # Pre-train aux network on Dirichlet BCs
    # ------------------------------------------------------------------
    def _pretrain_aux(self, epochs=10000, n=5000, lr=1e-3):
        """
        Fit aux_net to:
          phi(S,v,0) = max(S/K - 1, 0)   [payoff at tau=0, in K=1 units]
          phi(0,v,tau) = 0                [S=0 boundary]
        """
        opt   = torch.optim.Adam(self.aux_net.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=2000, gamma=0.5)

        for ep in range(1, epochs + 1):
            opt.zero_grad()

            # Payoff at tau=0
            S_T   = self._to(torch.FloatTensor(n, 1).uniform_(0, self.S_max))
            v_T   = self._to(torch.FloatTensor(n, 1).uniform_(0, self.v_max))
            tau_T = self._to(torch.zeros(n, 1))
            S_n, v_n, tau_n = self._normalise(S_T, v_T, tau_T)
            target = torch.clamp(S_T / self.K - 1.0, min=0.0)
            loss_T = torch.mean((self.aux_net(S_n, v_n, tau_n) - target) ** 2)

            # S=0 boundary
            S_0   = self._to(torch.zeros(n, 1))
            v_0   = self._to(torch.FloatTensor(n, 1).uniform_(0, self.v_max))
            tau_0 = self._to(torch.FloatTensor(n, 1).uniform_(0, self.T))
            S_n0, v_n0, tau_n0 = self._normalise(S_0, v_0, tau_0)
            loss_0 = torch.mean(self.aux_net(S_n0, v_n0, tau_n0) ** 2)

            loss = loss_T + loss_0
            loss.backward()
            opt.step()
            sched.step()
            if ep % 1000 == 0:
                print(f"  [aux pretrain] ep {ep:5d}  loss={loss.item():.4e}")

    # ------------------------------------------------------------------
    # PDE residual (backward time tau, tau=0 is expiry)
    # ------------------------------------------------------------------
    def _pde_residual(self, S, v, tau):
        """
        Heston PDE (Eq. 3.8):
          LU = 0.5*v*S^2*U_SS + rho*xi*v*S*U_Sv + 0.5*xi^2*v*U_vv
               + r*S*U_S + kappa*(theta-v)*U_v - r*U - U_tau = 0
        Normalised by K so residual is O(1).
        """
        S.requires_grad_(True)
        v.requires_grad_(True)
        tau.requires_grad_(True)
        U = self._trial(S, v, tau)

        ones = torch.ones_like(U)
        U_tau = torch.autograd.grad(U, tau, grad_outputs=ones, create_graph=True)[0]
        U_S   = torch.autograd.grad(U, S,   grad_outputs=ones, create_graph=True)[0]
        U_v   = torch.autograd.grad(U, v,   grad_outputs=ones, create_graph=True)[0]
        U_SS  = torch.autograd.grad(U_S, S, grad_outputs=torch.ones_like(U_S), create_graph=True)[0]
        U_vv  = torch.autograd.grad(U_v, v, grad_outputs=torch.ones_like(U_v), create_graph=True)[0]
        U_Sv  = torch.autograd.grad(U_S, v, grad_outputs=torch.ones_like(U_S), create_graph=True)[0]

        res = (0.5 * v * S ** 2 * U_SS
               + self.rho * self.xi * v * S * U_Sv
               + 0.5 * self.xi ** 2 * v * U_vv
               + self.r * S * U_S
               + self.kappa * (self.theta - v) * U_v
               - self.r * U
               - U_tau)
        return res / self.K

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------
    def train(self, epochs=20000, log_every=1000,
              n_r=10000, n_bc=500, lr=1e-3,
              pretrain_epochs=10000):
        """
        Paper protocol:
          1. Pre-train aux_net on Dirichlet BCs, then FREEZE.
          2. Train main_net: Adam + StepLR(step=5000, gamma=0.75).
          3. Loss = L_pde + L_Smax + L_vmax + L_deg (equal weights).
        """
        print("Pre-training auxiliary network...")
        self._pretrain_aux(epochs=pretrain_epochs)
        for p in self.aux_net.parameters():
            p.requires_grad_(False)

        opt   = torch.optim.Adam(self.main_net.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=5000, gamma=0.75)

        for epoch in range(1, epochs + 1):
            opt.zero_grad()

            # --- Interior PDE residual ---
            S_r   = self._to(torch.FloatTensor(n_r, 1).uniform_(0, self.S_max))
            v_r   = self._to(torch.FloatTensor(n_r, 1).uniform_(0, self.v_max))
            tau_r = self._to(torch.FloatTensor(n_r, 1).uniform_(0, self.T))
            loss_pde = torch.mean(self._pde_residual(S_r, v_r, tau_r) ** 2)

            # --- Neumann at S=S_max: dU/dS = 1 ---
            S_sm   = self._to(torch.full((n_bc, 1), self.S_max)).requires_grad_(True)
            v_sm   = self._to(torch.FloatTensor(n_bc, 1).uniform_(0, self.v_max))
            tau_sm = self._to(torch.FloatTensor(n_bc, 1).uniform_(0, self.T))
            U_sm   = self._trial(S_sm, v_sm, tau_sm)
            dU_dS  = torch.autograd.grad(U_sm, S_sm,
                                         grad_outputs=torch.ones_like(U_sm),
                                         create_graph=True)[0]
            loss_Smax = torch.mean((dU_dS - 1.0) ** 2)

            # --- Neumann at v=v_max: dU/dv = 0 ---
            S_vm   = self._to(torch.FloatTensor(n_bc, 1).uniform_(0, self.S_max))
            v_vm   = self._to(torch.full((n_bc, 1), self.v_max)).requires_grad_(True)
            tau_vm = self._to(torch.FloatTensor(n_bc, 1).uniform_(0, self.T))
            U_vm   = self._trial(S_vm, v_vm, tau_vm)
            dU_dv  = torch.autograd.grad(U_vm, v_vm,
                                         grad_outputs=torch.ones_like(U_vm),
                                         create_graph=True)[0]
            loss_vmax = torch.mean(dU_dv ** 2)

            # --- Fichera degenerate BC at v=0 ---
            S_dg   = self._to(torch.FloatTensor(n_bc, 1).uniform_(0, self.S_max)).requires_grad_(True)
            v_dg   = self._to(torch.full((n_bc, 1), 1e-6)).requires_grad_(True)
            tau_dg = self._to(torch.FloatTensor(n_bc, 1).uniform_(0, self.T)).requires_grad_(True)
            U_dg   = self._trial(S_dg, v_dg, tau_dg)
            ones_d = torch.ones_like(U_dg)
            dU_dS_d   = torch.autograd.grad(U_dg, S_dg,   grad_outputs=ones_d, create_graph=True)[0]
            dU_dv_d   = torch.autograd.grad(U_dg, v_dg,   grad_outputs=ones_d, create_graph=True)[0]
            dU_dtau_d = torch.autograd.grad(U_dg, tau_dg, grad_outputs=ones_d, create_graph=True)[0]
            deg_res = (S_dg * dU_dS_d
                       + self.kappa * self.theta * dU_dv_d
                       - self.r * U_dg
                       - dU_dtau_d) / self.K
            loss_deg = torch.mean(deg_res ** 2)

            loss = loss_pde + loss_Smax + loss_vmax + loss_deg
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.main_net.parameters(), 1.0)
            opt.step()
            sched.step()

            if epoch % log_every == 0:
                print(f"[ICPINN] epoch {epoch:6d} | "
                      f"loss={loss.item():.4e}  "
                      f"pde={loss_pde.item():.4e}  "
                      f"Smax={loss_Smax.item():.4e}  "
                      f"vmax={loss_vmax.item():.4e}  "
                      f"deg={loss_deg.item():.4e}")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def price(self, S, v=None, tau=None):
        """
        Price at backward time tau (tau=T means 'now', tau=0 means at expiry).
        Default: tau=T (price today).
        """
        if v is None:
            v = self.v0
        if tau is None:
            tau = self.T
        self.aux_net.eval()
        self.main_net.eval()
        with torch.no_grad():
            S_t   = self._to(torch.tensor([[float(S)]],   dtype=torch.float32))
            v_t   = self._to(torch.tensor([[float(v)]],   dtype=torch.float32))
            tau_t = self._to(torch.tensor([[float(tau)]], dtype=torch.float32))
            return self._trial(S_t, v_t, tau_t).item()

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
