"""
CEV-PINN: Physics-Informed Neural Network for Constant Elasticity of Variance
option pricing.

PDE: dV/dt + 0.5*sigma^2*S^(2*beta)*d2V/dS2 + r*S*dV/dS - r*V = 0
  beta=1  -> reduces to BSM
  beta=0.5 -> square-root process (most stable, used as default)
  beta<1  -> volatility decreases as S increases (leverage effect)

No closed-form solution for general beta; we validate against the Schroder (1989)
non-central chi-square analytical solution for beta=0.5.
"""

import torch
import torch.nn as nn
import numpy as np


# ── analytical reference (Schroder 1989, non-central chi-square) ────────────
def cev_analytical_call(S0, K, T, r, sigma, beta):
    """CEV European call via non-central chi-square (Schroder 1989).
    Valid for beta < 1. beta=0.5 (square-root process) is most stable.
    """
    from scipy.stats import ncx2
    if abs(beta - 1.0) < 1e-9:
        from independent.bsm_pinn import bs_call_price
        return bs_call_price(S0, K, T, r, sigma)
    nu   = 1.0 / (1.0 - beta)
    lam  = 2.0 * r / (sigma**2 * (1.0 - beta) * (np.exp(2*r*(1-beta)*T) - 1))
    x    = lam * S0**(2*(1-beta)) * np.exp(2*r*(1-beta)*T)
    y    = lam * K**(2*(1-beta))
    d    = 2.0 + nu
    call = (S0 * (1.0 - ncx2.cdf(y, df=d,   nc=x))
            - K * np.exp(-r*T) * ncx2.cdf(x, df=d-2, nc=y))
    return float(max(call, max(S0 - K*np.exp(-r*T), 0.0)))


# ── finite-difference reference (Crank-Nicolson) ───────────────────────────
def cev_fd_call(S0, K, T, r, sigma, beta, N_S=400, N_t=400):
    """Crank-Nicolson FD for CEV call price. Used as ground truth."""
    S_max = 3.0 * K
    dS = S_max / N_S
    dt = T / N_t
    S = np.linspace(0, S_max, N_S + 1)

    V = np.maximum(S - K, 0.0)
    alpha = 0.5 * sigma ** 2 * S ** (2 * beta)

    a_diag = -0.5 * dt * (alpha / dS ** 2 - r * S / (2 * dS))
    b_diag = 1.0 + dt * (alpha / dS ** 2 + r)
    c_diag = -0.5 * dt * (alpha / dS ** 2 + r * S / (2 * dS))

    a_rhs = 0.5 * dt * (alpha / dS ** 2 - r * S / (2 * dS))
    b_rhs = 1.0 - dt * (alpha / dS ** 2 + r)
    c_rhs = 0.5 * dt * (alpha / dS ** 2 + r * S / (2 * dS))

    for step in range(N_t):
        tau = (step + 1) * dt

        rhs = np.zeros_like(V)
        rhs[1:-1] = (a_rhs[1:-1] * V[:-2]
                     + b_rhs[1:-1] * V[1:-1]
                     + c_rhs[1:-1] * V[2:])
        rhs[0] = 0.0
        rhs[-1] = S_max - K * np.exp(-r * tau)

        aa = a_diag.copy()
        bb = b_diag.copy()
        cc = c_diag.copy()
        n = len(rhs)

        for i in range(1, n):
            if abs(bb[i - 1]) < 1e-14:
                break
            m = aa[i] / bb[i - 1]
            bb[i] -= m * cc[i - 1]
            rhs[i] -= m * rhs[i - 1]

        V_new = np.zeros(n)
        V_new[-1] = rhs[-1] / bb[-1]
        for i in range(n - 2, -1, -1):
            V_new[i] = (rhs[i] - cc[i] * V_new[i + 1]) / bb[i]
        V = V_new

    idx = int(S0 / dS)
    idx = min(idx, N_S - 1)
    w = (S0 - S[idx]) / dS
    return float((1 - w) * V[idx] + w * V[idx + 1])


# ── shared network (same gated architecture as BSM) ────────────────────────
class GatedPINN(nn.Module):
    def __init__(self, in_dim=2, hidden=64, depth=4):
        super().__init__()
        self.branch_shallow = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        deep_layers = [nn.Linear(in_dim, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            deep_layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        self.branch_deep = nn.Sequential(*deep_layers)
        self.gate = nn.Linear(hidden, 1)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x):
        h_s = self.branch_shallow(x)
        h_d = self.branch_deep(x)
        g = torch.sigmoid(self.gate(h_s))
        return self.out(g * h_s + (1 - g) * h_d)


# ── PINN trainer ────────────────────────────────────────────────────────────
class CEV_PINN:
    OPTION_TYPES = {"european_call", "european_put", "american_call", "american_put"}

    def __init__(self, K=100.0, T=1.0, r=0.05, sigma=0.25, beta=0.5,
                 S_max=300.0, option_type="european_call", device=None):
        assert option_type in self.OPTION_TYPES
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.beta = beta
        self.S_max = S_max
        self.option_type = option_type
        self.is_call = "call" in option_type
        self.is_american = "american" in option_type
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.net = GatedPINN().to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=5000, gamma=0.5
        )

    def _payoff(self, S):
        if self.is_call:
            return torch.clamp(S - self.K, min=0.0)
        else:
            return torch.clamp(self.K - S, min=0.0)

    def _sample_collocation(self, n=10000):
        S = torch.FloatTensor(n, 1).uniform_(0.01, self.S_max)
        t = torch.FloatTensor(n, 1).uniform_(0.0, self.T)
        return S.to(self.device), t.to(self.device)

    def _sample_terminal(self, n=2000):
        S = torch.FloatTensor(n, 1).uniform_(0.01, self.S_max)
        t = torch.full((n, 1), self.T)
        V = self._payoff(S)
        return S.to(self.device), t.to(self.device), V.to(self.device)

    def _sample_boundary(self, n=1000):
        t_lo = torch.FloatTensor(n, 1).uniform_(0.0, self.T)
        t_hi = torch.FloatTensor(n, 1).uniform_(0.0, self.T)
        S_lo = torch.zeros(n, 1)
        S_hi = torch.full((n, 1), self.S_max)
        if self.is_call:
            V_lo = torch.zeros(n, 1)
            if self.is_american:
                V_hi = torch.full((n, 1), self.S_max - self.K)
            else:
                V_hi = self.S_max - self.K * torch.exp(-self.r * (self.T - t_hi))
        else:
            if self.is_american:
                V_lo = torch.full((n, 1), self.K)
            else:
                V_lo = self.K * torch.exp(-self.r * (self.T - t_lo))
            V_hi = torch.zeros(n, 1)
        return (
            S_lo.to(self.device), t_lo.to(self.device), V_lo.to(self.device),
            S_hi.to(self.device), t_hi.to(self.device), V_hi.to(self.device),
        )

    def _pde_residual(self, S, t):
        S.requires_grad_(True)
        t.requires_grad_(True)
        x = torch.cat([S / self.S_max, t / self.T], dim=1)
        V = self.net(x)

        V_t = torch.autograd.grad(V, t, grad_outputs=torch.ones_like(V),
                                  create_graph=True)[0]
        V_S = torch.autograd.grad(V, S, grad_outputs=torch.ones_like(V),
                                  create_graph=True)[0]
        V_SS = torch.autograd.grad(V_S, S, grad_outputs=torch.ones_like(V_S),
                                   create_graph=True)[0]

        diffusion = 0.5 * self.sigma ** 2 * S ** (2 * self.beta)
        residual = V_t + diffusion * V_SS + self.r * S * V_S - self.r * V
        return residual

    def _predict(self, S, t):
        x = torch.cat([S / self.S_max, t / self.T], dim=1)
        return self.net(x)

    def train(self, epochs=20000, log_every=1000,
              w_pde=1.0, w_ic=10.0, w_bc=5.0, w_american=50.0):
        from tqdm import tqdm
        losses = []
        pbar = tqdm(range(1, epochs + 1), desc=f"CEV β={self.beta} {self.option_type}",
                    unit="epoch", dynamic_ncols=True)
        for epoch in pbar:
            self.optimizer.zero_grad()

            S_c, t_c = self._sample_collocation()
            loss_pde = torch.mean(self._pde_residual(S_c, t_c) ** 2)

            S_ic, t_ic, V_ic = self._sample_terminal()
            loss_ic = torch.mean((self._predict(S_ic, t_ic) - V_ic) ** 2)

            S_lo, t_lo, V_lo, S_hi, t_hi, V_hi = self._sample_boundary()
            loss_bc = (torch.mean((self._predict(S_lo, t_lo) - V_lo) ** 2)
                       + torch.mean((self._predict(S_hi, t_hi) - V_hi) ** 2))

            loss = w_pde * loss_pde + w_ic * loss_ic + w_bc * loss_bc

            if self.is_american:
                S_a, t_a = self._sample_collocation(n=5000)
                V_pred = self._predict(S_a, t_a)
                loss_american = torch.mean(
                    torch.clamp(self._payoff(S_a) - V_pred, min=0.0) ** 2
                )
                loss = loss + w_american * loss_american

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            pbar.set_postfix(loss=f"{loss.item():.3e}",
                             pde=f"{loss_pde.item():.3e}",
                             ic=f"{loss_ic.item():.3e}")
            if epoch % log_every == 0:
                losses.append(loss.item())
        return losses

    def price(self, S, t=0.0):
        self.net.eval()
        with torch.no_grad():
            S_t = torch.tensor([[S]], dtype=torch.float32).to(self.device)
            t_t = torch.tensor([[t]], dtype=torch.float32).to(self.device)
            return self._predict(S_t, t_t).item()

    def save(self, path):
        torch.save({
            "state_dict": self.net.state_dict(),
            "params": dict(K=self.K, T=self.T, r=self.r, sigma=self.sigma,
                           beta=self.beta, S_max=self.S_max,
                           option_type=self.option_type),
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ckpt["state_dict"])
