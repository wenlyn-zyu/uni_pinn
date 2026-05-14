"""
BSM-PINN: Physics-Informed Neural Network for Black-Scholes-Merton option pricing.

PDE: dV/dt + 0.5*sigma^2*S^2*d2V/dS2 + r*S*dV/dS - r*V = 0

Supported option types:
  european_call  -- terminal: max(S-K, 0)
  european_put   -- terminal: max(K-S, 0)  [or via put-call parity from call]
  american_call  -- adds early-exercise complementarity penalty
  american_put   -- most common; adds complementarity penalty
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.stats import norm


# ── analytical Black-Scholes price for validation ──────────────────────────
def bs_call_price(S, K, T, r, sigma):
    S, K, T, r, sigma = map(float, [S, K, T, r, sigma])
    if T <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put_price(S, K, T, r, sigma):
    """European put via put-call parity: P = C - S + K*exp(-rT)."""
    return bs_call_price(S, K, T, r, sigma) - S + K * np.exp(-r * T)


# ── network architecture (gated, from Dhiman & Hu 2023) ────────────────────
class GatedPINN(nn.Module):
    """Two-branch gated network: wide shallow branch + deep branch."""

    def __init__(self, hidden=64, depth=4):
        super().__init__()
        # shallow branch: 1 hidden layer, captures simple features
        self.branch_shallow = nn.Sequential(
            nn.Linear(2, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        # deep branch: `depth` hidden layers, captures complex features
        deep_layers = [nn.Linear(2, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            deep_layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        self.branch_deep = nn.Sequential(*deep_layers)

        self.gate = nn.Linear(hidden, 1)          # scalar gate weight
        self.out = nn.Linear(hidden, 1)

    def forward(self, x):
        h_s = self.branch_shallow(x)
        h_d = self.branch_deep(x)
        g = torch.sigmoid(self.gate(h_s))         # gate in [0,1]
        h = g * h_s + (1 - g) * h_d
        return self.out(h)


# ── PINN trainer ────────────────────────────────────────────────────────────
class BSM_PINN:
    OPTION_TYPES = {"european_call", "european_put", "american_call", "american_put"}

    def __init__(self, K=100.0, T=1.0, r=0.05, sigma=0.2,
                 S_max=300.0, option_type="european_call", device=None):
        assert option_type in self.OPTION_TYPES, f"Unknown option_type: {option_type}"
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
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
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=30000, eta_min=1e-5
        )

    def _payoff(self, S):
        """Intrinsic value: used for terminal condition and American penalty."""
        if self.is_call:
            return torch.clamp(S - self.K, min=0.0)
        else:
            return torch.clamp(self.K - S, min=0.0)

    # ── sampling helpers ────────────────────────────────────────────────────
    def _sample_collocation(self, n=10000):
        S = torch.FloatTensor(n, 1).uniform_(0.01, self.S_max)
        t = torch.FloatTensor(n, 1).uniform_(0.0, self.T)
        return S.to(self.device), t.to(self.device)

    def _sample_terminal(self, n=2000):
        S = torch.FloatTensor(n, 1).uniform_(0.01, self.S_max)
        t = torch.full((n, 1), self.T)
        V = self._payoff(S) / self.K   # normalise by K
        return S.to(self.device), t.to(self.device), V.to(self.device)

    def _sample_boundary(self, n=1000):
        t_lo = torch.FloatTensor(n, 1).uniform_(0.0, self.T)
        t_hi = torch.FloatTensor(n, 1).uniform_(0.0, self.T)
        S_lo = torch.zeros(n, 1)
        # sample upper boundary densely near S_max
        S_hi = torch.FloatTensor(n, 1).uniform_(0.8 * self.S_max, self.S_max)

        if self.is_call:
            V_lo = torch.zeros(n, 1)
            if self.is_american:
                V_hi = (S_hi - self.K) / self.K
            else:
                V_hi = (S_hi - self.K * torch.exp(-self.r * (self.T - t_hi))) / self.K
        else:
            if self.is_american:
                V_lo = torch.full((n, 1), 1.0)
            else:
                V_lo = self.K * torch.exp(-self.r * (self.T - t_lo)) / self.K
            V_hi = torch.zeros(n, 1)

        return (
            S_lo.to(self.device), t_lo.to(self.device), V_lo.to(self.device),
            S_hi.to(self.device), t_hi.to(self.device), V_hi.to(self.device),
        )

    # ── PDE residual ────────────────────────────────────────────────────────
    def _pde_residual(self, S, t):
        S.requires_grad_(True)
        t.requires_grad_(True)
        x = torch.cat([S / self.S_max, t / self.T], dim=1)  # normalise inputs
        u = self.net(x)   # u = V/K (normalised price)

        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u),
                                  create_graph=True)[0]
        u_S = torch.autograd.grad(u, S, grad_outputs=torch.ones_like(u),
                                  create_graph=True)[0]
        u_SS = torch.autograd.grad(u_S, S, grad_outputs=torch.ones_like(u_S),
                                   create_graph=True)[0]

        # BSM PDE: u_t + 0.5*sigma^2*S^2*u_SS + r*S*u_S - r*u = 0
        residual = (u_t
                    + 0.5 * self.sigma ** 2 * S ** 2 * u_SS
                    + self.r * S * u_S
                    - self.r * u)
        return residual

    # ── forward pass (normalised inputs) ────────────────────────────────────
    def _predict(self, S, t):
        x = torch.cat([S / self.S_max, t / self.T], dim=1)
        return self.net(x)   # returns u = V/K

    # ── training loop ────────────────────────────────────────────────────────
    def train(self, epochs=20000, log_every=1000,
              w_pde=1.0, w_ic=10.0, w_bc=5.0, w_american=50.0):
        from tqdm import tqdm
        losses = []
        pbar = tqdm(range(1, epochs + 1), desc=f"BSM-{self.option_type}",
                    unit="epoch", dynamic_ncols=True)
        for epoch in pbar:
            self.optimizer.zero_grad()

            S_c, t_c = self._sample_collocation()
            res = self._pde_residual(S_c, t_c)
            loss_pde = torch.mean(res ** 2)

            S_ic, t_ic, V_ic = self._sample_terminal()
            loss_ic = torch.mean((self._predict(S_ic, t_ic) - V_ic) ** 2)

            S_lo, t_lo, V_lo, S_hi, t_hi, V_hi = self._sample_boundary()
            loss_bc = (torch.mean((self._predict(S_lo, t_lo) - V_lo) ** 2)
                       + torch.mean((self._predict(S_hi, t_hi) - V_hi) ** 2))

            loss = w_pde * loss_pde + w_ic * loss_ic + w_bc * loss_bc

            # American early-exercise constraint: V >= payoff everywhere
            if self.is_american:
                S_a, t_a = self._sample_collocation(n=5000)
                u_pred = self._predict(S_a, t_a)
                payoff_a = self._payoff(S_a) / self.K   # normalised
                loss_american = torch.mean(
                    torch.clamp(payoff_a - u_pred, min=0.0) ** 2
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

    # ── inference ────────────────────────────────────────────────────────────
    def price(self, S, t=0.0):
        """Price the option at spot S and current time t."""
        self.net.eval()
        with torch.no_grad():
            S_t = torch.tensor([[S]], dtype=torch.float32).to(self.device)
            t_t = torch.tensor([[t]], dtype=torch.float32).to(self.device)
            return self._predict(S_t, t_t).item() * self.K   # scale back by K

    def save(self, path):
        torch.save({
            "state_dict": self.net.state_dict(),
            "params": dict(K=self.K, T=self.T, r=self.r, sigma=self.sigma,
                           S_max=self.S_max, option_type=self.option_type),
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ckpt["state_dict"])
