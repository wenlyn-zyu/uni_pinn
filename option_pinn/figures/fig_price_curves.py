"""Figure: Price curves for BSM, CEV, Heston — Unified PINN vs Reference."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
PINN_DIR = os.path.join(HERE, "..")
OUT  = os.path.join(HERE, "..", "..", "thesis", "Tex_thesis", "Img")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["DejaVu Serif", "Times New Roman", "serif"],
    "font.size":          11,
    "axes.titlesize":     13,
    "axes.labelsize":     11,
    "axes.linewidth":     1.0,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "xtick.direction":    "in",
    "ytick.direction":    "in",
    "xtick.major.size":   4,
    "ytick.major.size":   4,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.minor.size":   2,
    "ytick.minor.size":   2,
    "legend.framealpha":  0.95,
    "legend.edgecolor":   "#bbbbbb",
    "legend.fontsize":    9.5,
    "legend.handlelength": 2.8,
    "text.usetex":        False,
    "figure.dpi":         150,
})

C1   = "#1565C0"   # deep blue  — Unified PINN
C2   = "#C62828"   # deep red   — (unused here, reserved for Independent)
C3   = "#2E7D32"   # deep green
GRAY = "#555555"

DASH_LONG = (0, (8, 4))   # long dashes — Unified PINN


def savefig(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {name}")


def _load_unified():
    import torch
    sys.path.insert(0, PINN_DIR)
    from unified_pinn_v2 import UnifiedPINN, ModelParams
    device = torch.device("cpu")
    p0   = ModelParams.from_bsm(sigma=0.20)
    pinn = UnifiedPINN([p0], hidden=128, depth=6, device=device)
    pinn.load(os.path.join(PINN_DIR, "results", "unified_v16_gl.pt"))
    pinn.net.eval()
    return pinn


def _unified_price(pinn, S_arr, p):
    return np.array([pinn.price(p, float(s)) for s in S_arr])


def fig_price_curves():
    try:
        sys.path.insert(0, PINN_DIR)
        from ref_solvers import bsm_call, cev_call, heston_call
        from unified_pinn_v2 import ModelParams
        pinn = _load_unified()
    except Exception as e:
        print(f"  skipping price_curves.pdf: {e}")
        return

    S = np.linspace(50, 250, 60)
    K, T, r = 100., 1., 0.05

    p_bsm    = ModelParams.from_bsm(sigma=0.20)
    p_cev    = ModelParams.from_cev(sigma=0.20, beta=0.5)
    p_heston = ModelParams.from_heston(kappa=2.0, theta=0.04, xi=0.3,
                                       rho=-0.7, v0=0.04)

    ref_bsm    = np.array([bsm_call(s, K, T, r, 0.20) for s in S])
    ref_cev    = np.array([cev_call(s, K, T, r, 0.20, 0.5) for s in S])
    ref_heston = np.array([heston_call(s, K, T, r, 2.0, 0.04, 0.3, -0.7, 0.04)
                           for s in S])
    pred_bsm    = _unified_price(pinn, S, p_bsm)
    pred_cev    = _unified_price(pinn, S, p_cev)
    pred_heston = _unified_price(pinn, S, p_heston)

    configs = [
        ("BSM",                    ref_bsm,    pred_bsm),
        (r"CEV ($\beta\!=\!0.5$)", ref_cev,    pred_cev),
        ("Heston",                 ref_heston, pred_heston),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    for ax, (title, ref, pred) in zip(axes, configs):
        ax.plot(S, ref,  color="black", lw=2.8, linestyle="-",
                label="Reference", zorder=4)
        ax.plot(S, pred, color=C1,     lw=2.2, linestyle=DASH_LONG,
                label="Unified PINN", zorder=5)
        ax.axvline(K, color=GRAY, lw=0.7, linestyle=":", alpha=0.6)
        ax.set_title(title, pad=6)
        ax.set_xlabel(r"$S$")
        ax.set_ylabel(r"$V$")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.15, linewidth=0.5)
        ax.set_xlim(50, 250)

    fig.tight_layout(pad=0.8)
    savefig(fig, "price_curves.pdf")


if __name__ == "__main__":
    fig_price_curves()
