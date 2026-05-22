"""Figure: Comparison of Independent PINN vs Unified PINN vs Reference."""
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
C2   = "#C62828"   # deep red   — Independent PINN
GRAY = "#555555"

DASH_LONG    = (0, (8, 4))          # Unified PINN
DASH_LONG_C2 = (0, (8, 4))          # Independent PINN — same rhythm, distinct color


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


def _load_indep(name):
    import torch
    path = os.path.join(PINN_DIR, "results", f"{name}.pt")
    ckpt = torch.load(path, map_location="cpu")
    sys.path.insert(0, os.path.join(PINN_DIR, "independent"))
    if name == "indep_bsm":
        from bsm_pinn import BSM_PINN
        m = BSM_PINN(**ckpt["params"], device="cpu")
    else:
        from cev_pinn import CEV_PINN
        m = CEV_PINN(**ckpt["params"], device="cpu")
    m.net.load_state_dict(ckpt["state_dict"])
    m.net.eval()
    return m


def _unified_price(pinn, S_arr, p):
    return np.array([pinn.price(p, float(s)) for s in S_arr])


def fig_eval_compare():
    try:
        sys.path.insert(0, PINN_DIR)
        from ref_solvers import bsm_call, cev_call, heston_call
        from unified_pinn_v2 import ModelParams
        pinn = _load_unified()
    except Exception as e:
        print(f"  skipping eval_compare.pdf: {e}")
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
    uni_bsm    = _unified_price(pinn, S, p_bsm)
    uni_cev    = _unified_price(pinn, S, p_cev)
    uni_heston = _unified_price(pinn, S, p_heston)

    ind_bsm = ind_cev = None
    try:
        ib  = _load_indep("indep_bsm")
        raw = np.array([ib.price(float(s)) for s in S])
        if raw.max() < 300 and raw.min() >= 0:
            ind_bsm = raw
        else:
            print(f"  indep BSM prices out of range (max={raw.max():.1f}), skipping")
    except Exception as e:
        print(f"  indep BSM unavailable: {e}")
    try:
        ic  = _load_indep("indep_cev")
        raw = np.array([ic.price(float(s)) for s in S])
        if raw.max() < 300 and raw.min() >= 0:
            ind_cev = raw
        else:
            print(f"  indep CEV prices out of range (max={raw.max():.1f}), skipping")
    except Exception as e:
        print(f"  indep CEV unavailable: {e}")

    titles = ["BSM", r"CEV ($\beta\!=\!0.5$)", "Heston"]
    refs   = [ref_bsm,  ref_cev,  ref_heston]
    unis   = [uni_bsm,  uni_cev,  uni_heston]
    inds   = [ind_bsm,  ind_cev,  None]

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    for ax, title, ref, uni, ind in zip(axes, titles, refs, unis, inds):
        ax.plot(S, ref, color="black", lw=2.8, linestyle="-",
                label="Reference", zorder=4)
        ax.plot(S, uni, color=C1,     lw=2.2, linestyle=DASH_LONG,
                label="Unified PINN", zorder=5)
        if ind is not None:
            ax.plot(S, ind, color=C2, lw=2.2, linestyle=DASH_LONG_C2,
                    label="Independent PINN", zorder=3)
        ax.axvline(K, color=GRAY, lw=0.7, linestyle=":", alpha=0.6)
        ax.set_title(title, pad=6)
        ax.set_xlabel(r"$S$")
        ax.set_ylabel(r"$V$")
        ax.legend(loc="upper left", fontsize=8.5)
        ax.grid(True, alpha=0.15, linewidth=0.5)
        ax.set_xlim(50, 250)

    fig.tight_layout(pad=0.8)
    savefig(fig, "eval_compare.pdf")


if __name__ == "__main__":
    fig_eval_compare()
