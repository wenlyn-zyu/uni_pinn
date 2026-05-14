# option_pinn/ref_solvers.py
"""参考解：BSM 解析解、CEV Schroder 1989、Heston GL 96点半解析解"""
import numpy as np
from scipy.stats import norm, ncx2
from numpy.polynomial.legendre import leggauss


# ── BSM ──────────────────────────────────────────────────────────────────────

def bsm_call(S, K, T, r, sigma):
    """Black-Scholes European call."""
    eps = 1e-10
    T = max(float(T), eps)
    sigma = max(float(sigma), eps)
    sqt = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / sqt
    d2 = d1 - sqt
    return float(max(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2), 0.0))


def bsm_greeks(S, K, T, r, sigma):
    """BSM Delta, Gamma, Vega (analytical)."""
    eps = 1e-10
    T = max(float(T), eps)
    sigma = max(float(sigma), eps)
    sqt = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / sqt
    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S * sqt)
    vega  = S * norm.pdf(d1) * np.sqrt(T)
    return {"delta": float(delta), "gamma": float(gamma), "vega": float(vega)}


# ── CEV (Schroder 1989) ───────────────────────────────────────────────────────

def cev_call(S, K, T, r, sigma, beta):
    """CEV European call via non-central chi-squared (Schroder 1989)."""
    if abs(beta - 1.0) < 1e-9:
        return bsm_call(S, K, T, r, sigma)
    d = 1.0 - beta
    nu  = 1.0 / d
    lam = (2.0 * r) / (sigma**2 * d * (np.exp(2.0 * r * d * T) - 1.0))
    x   = lam * S**(2.0 * d) * np.exp(2.0 * r * d * T)
    y   = lam * K**(2.0 * d)
    call = (S * (1.0 - ncx2.cdf(y, df=2.0 + nu, nc=x))
            - K * np.exp(-r * T) * ncx2.cdf(x, df=nu, nc=y))
    return float(max(call, max(S - K * np.exp(-r * T), 0.0)))


# ── Heston GL ─────────────────────────────────────────────────────────────────

_N_GL = 96
_GL_NODES, _GL_WEIGHTS = leggauss(_N_GL)
_PHI_MAX = 100.0
_GL_PHI = (_GL_NODES + 1.0) * 0.5 * _PHI_MAX
_GL_W   = _GL_WEIGHTS * 0.5 * _PHI_MAX


def _heston_cf(phi, S, K, T, r, kappa, theta, xi, rho, v0, j):
    u = 0.5 if j == 1 else -0.5
    b = (kappa - rho * xi) if j == 1 else kappa
    a = kappa * theta
    x = np.log(S / K)
    d = np.sqrt((rho * xi * 1j * phi - b)**2 - xi**2 * (2*u*1j*phi - phi**2))
    g = (b - rho*xi*1j*phi + d) / (b - rho*xi*1j*phi - d)
    C = (r*1j*phi*T + a/xi**2 * ((b - rho*xi*1j*phi + d)*T
         - 2*np.log((1 - g*np.exp(d*T))/(1 - g))))
    D = ((b - rho*xi*1j*phi + d)/xi**2
         * (1 - np.exp(d*T))/(1 - g*np.exp(d*T)))
    return np.exp(C + D*v0 + 1j*phi*x)


def heston_call(S, K, T, r, kappa, theta, xi, rho, v0):
    """Heston European call via 96-point Gauss-Legendre quadrature."""
    phi = _GL_PHI
    cf1 = _heston_cf(phi, S, K, T, r, kappa, theta, xi, rho, v0, j=1)
    cf2 = _heston_cf(phi, S, K, T, r, kappa, theta, xi, rho, v0, j=2)
    P1 = 0.5 + (1.0/np.pi) * np.sum(_GL_W * np.real(cf1 / (1j*phi)))
    P2 = 0.5 + (1.0/np.pi) * np.sum(_GL_W * np.real(cf2 / (1j*phi)))
    return float(max(S*P1 - K*np.exp(-r*T)*P2, 0.0))


def heston_greeks_fd(S, K, T, r, kappa, theta, xi, rho, v0, dS_frac=0.005):
    """Heston Delta, Gamma, Vega via central finite differences on GL prices."""
    dS = S * dS_frac
    dv = max(v0 * 0.05, 1e-4)
    p_up  = heston_call(S+dS, K, T, r, kappa, theta, xi, rho, v0)
    p_dn  = heston_call(S-dS, K, T, r, kappa, theta, xi, rho, v0)
    p_mid = heston_call(S,    K, T, r, kappa, theta, xi, rho, v0)
    v0_up = min(v0 + dv, 0.99)
    v0_dn = max(v0 - dv, 1e-5)
    p_vup = heston_call(S, K, T, r, kappa, theta, xi, rho, v0_up)
    p_vdn = heston_call(S, K, T, r, kappa, theta, xi, rho, v0_dn)
    return {
        "delta": float((p_up - p_dn) / (2*dS)),
        "gamma": float((p_up - 2*p_mid + p_dn) / dS**2),
        "vega":  float((p_vup - p_vdn) / (v0_up - v0_dn)),
    }
