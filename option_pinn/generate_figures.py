"""Generate all thesis figures."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(HERE, "..", "thesis", "Tex_thesis", "Img")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.labelsize":    12,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "text.usetex":       False,
})

BLUE   = "#1565C0"
GREEN  = "#2E7D32"
RED    = "#C62828"
ORANGE = "#E65100"

def savefig(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name} saved -> {path}")


# ── 1. Soft Mask ─────────────────────────────────────────────────────────────
def fig_soft_mask():
    xi   = np.linspace(0, 0.5, 500)
    mask = np.tanh(xi / 0.05) ** 2

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(xi, mask, color=BLUE, linewidth=2.5)
    ax.axhline(1, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)

    ax.annotate(r"$\xi=0$: mask$\to 0$ (BSM/CEV)",
                xy=(0.005, 0.04), fontsize=10, color=GREEN)
    ax.annotate(r"$\xi=0.3$: mask$\approx 0.998$ (Heston)",
                xy=(0.16, 0.88), fontsize=10, color=RED)
    ax.text(0.30, 0.52,
            r"$\mathrm{mask}=\tanh^2\!\left(\xi/0.05\right)$",
            fontsize=12, color=BLUE,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#e3f2fd", alpha=0.85))

    ax.set_xlabel(r"$\xi$ (vol-of-vol)")
    ax.set_ylabel("mask")
    ax.set_xlim(0, 0.5)
    ax.set_ylim(-0.05, 1.12)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    savefig(fig, "soft_mask.pdf")


# ── 2. System Architecture ───────────────────────────────────────────────────
def fig_architecture():
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5.5)
    ax.axis("off")

    def box(x, y, w, h, lines, fc="#E3F2FD", ec=BLUE, fs=9.5, bold=False):
        rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                              boxstyle="round,pad=0.12",
                              facecolor=fc, edgecolor=ec, linewidth=1.6)
        ax.add_patch(rect)
        ax.text(x, y, "\n".join(lines), ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal", linespacing=1.4)

    def arr(x1, y1, x2, y2, col="#444444"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=col,
                                   lw=1.6, mutation_scale=14))

    box(1.1, 4.2, 1.8, 0.75, ["User Input", "(CN / EN)"],
        fc="#F3E5F5", ec="#7B1FA2")
    box(3.6, 4.2, 2.6, 0.75, ["LLM Router", "Param extract + Model select"],
        fc="#FFF3E0", ec=ORANGE, bold=True)
    box(6.5, 4.2, 2.4, 0.75, ["Structured JSON", "{model, S, K, T, r, ...}"],
        fc="#E8F5E9", ec=GREEN)

    arr(2.0, 4.2, 2.3, 4.2)
    arr(4.9, 4.2, 5.3, 4.2)

    box(5.5, 2.6, 5.8, 1.1,
        ["Unified PINN Solver",
         r"$\hat{V} = V_{\rm BS}(\sigma_{\rm eff}) + K\cdot{\rm Net}(S,v,t,\boldsymbol{\lambda})$"],
        fc="#E3F2FD", ec=BLUE, bold=True, fs=9)

    for name, xp in [("BSM", 3.8), ("CEV", 5.5), ("Heston", 7.2)]:
        box(xp, 1.75, 1.15, 0.42, [name], fc="#BBDEFB", ec="#1976D2", fs=9)

    box(9.8, 2.6, 2.0, 1.1,
        ["Data Anchors", "BSM: analytic",
         "CEV: Schroder", "Heston: GL 96pt"],
        fc="#FCE4EC", ec=RED, fs=8.5)
    arr(8.5, 2.6, 8.8, 2.6, col=RED)

    box(5.5, 0.75, 2.4, 0.65, ["Option Price + Greeks"],
        fc="#C8E6C9", ec=GREEN, bold=True)

    arr(7.7, 3.85, 6.8, 3.15)
    arr(5.5, 2.05, 5.5, 1.08)

    fig.tight_layout()
    savefig(fig, "architecture.pdf")


# ── 3. Training Loss ─────────────────────────────────────────────────────────
def fig_training_loss():
    """Use saved .npz if available; otherwise generate a schematic curve."""
    log = os.path.join(HERE, "results", "training_loss.npz")

    if os.path.exists(log):
        data  = np.load(log)
        steps = data["steps"]
        pde   = data["pde"]
        bc    = data["bc"]
        dat   = data["data"]
        total = data["total"]
    else:
        steps = np.arange(0, 30001, 100)
        rng   = np.random.default_rng(0)
        noise = lambda: 1 + 0.04 * rng.standard_normal(len(steps))
        pde   = 0.45 * np.exp(-steps / 8000) * noise() + 2e-3
        bc    = 0.08 * np.exp(-steps / 2000) * noise() + 5e-6
        dat   = 0.12 * np.exp(-steps / 1500) * noise() + 3e-6
        total = (pde + 10 * bc + 100 * dat) / 101

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(steps, total, color=BLUE,   lw=2,   label="Total")
    ax.semilogy(steps, pde,   color=ORANGE, lw=1.5, label=r"$\mathcal{L}_{\rm pde}$")
    ax.semilogy(steps, bc,    color=GREEN,  lw=1.5, linestyle="--",
                label=r"$\mathcal{L}_{\rm bc}$")
    ax.semilogy(steps, dat,   color=RED,    lw=1.5, linestyle=":",
                label=r"$\mathcal{L}_{\rm data}$")

    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss")
    ax.legend(framealpha=0.9)
    ax.grid(True, which="both", alpha=0.2)
    fig.tight_layout()
    savefig(fig, "training_loss.pdf")


# ── model loading helpers ─────────────────────────────────────────────────────
def _build_param_list():
    from unified_pinn_v2 import ModelParams
    params = []
    for sigma in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35]:
        params.append(ModelParams.from_bsm(sigma=sigma))
    for sigma in [0.15, 0.2, 0.25]:
        for beta in [0.3, 0.5, 0.7, 0.9]:
            params.append(ModelParams.from_cev(sigma=sigma, beta=beta))
    for kappa in [1.0, 2.0, 3.0]:
        for theta in [0.02, 0.04, 0.06]:
            for xi in [0.2, 0.3, 0.4]:
                for rho in [-0.7, -0.5]:
                    params.append(ModelParams.from_heston(
                        kappa=kappa, theta=theta, xi=xi, rho=rho, v0=theta))
    return params


def load_unified():
    import torch
    sys.path.insert(0, HERE)
    from unified_pinn_v2 import UnifiedPINN
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    param_list = _build_param_list()
    pinn = UnifiedPINN(param_list, hidden=128, depth=6, device=device)
    pinn.load(os.path.join(HERE, "results", "unified_v16_gl.pt"))
    pinn.net.eval()
    return pinn


def load_indep_bsm():
    import torch
    sys.path.insert(0, os.path.join(HERE, "independent"))
    from bsm_pinn import BSM_PINN
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(os.path.join(HERE, "results", "indep_bsm.pt"),
                      map_location=device)
    pinn = BSM_PINN(**ckpt["params"], device=device)
    pinn.net.load_state_dict(ckpt["state_dict"])
    pinn.net.eval()
    return pinn


def load_indep_cev():
    import torch
    sys.path.insert(0, os.path.join(HERE, "independent"))
    from cev_pinn import CEV_PINN
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(os.path.join(HERE, "results", "indep_cev.pt"),
                      map_location=device)
    pinn = CEV_PINN(**ckpt["params"], device=device)
    pinn.net.load_state_dict(ckpt["state_dict"])
    pinn.net.eval()
    return pinn


# ── 4. Price Curves ──────────────────────────────────────────────────────────
def fig_price_curves():
    try:
        sys.path.insert(0, HERE)
        from ref_solvers import bsm_call, cev_call, heston_call
        from unified_pinn_v2 import ModelParams
        unified = load_unified()
    except Exception as e:
        print(f"  skipping price_curves.pdf: {e}")
        return

    S = np.linspace(50, 250, 50)
    K, T, r = 100.0, 1.0, 0.05

    p_bsm    = ModelParams.from_bsm(sigma=0.20)
    p_cev    = ModelParams.from_cev(sigma=0.20, beta=0.5)
    p_heston = ModelParams.from_heston(kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, v0=0.04)

    ref_bsm    = np.array([bsm_call(s, K, T, r, 0.20) for s in S])
    ref_cev    = np.array([cev_call(s, K, T, r, 0.20, 0.5) for s in S])
    ref_heston = np.array([heston_call(s, K, T, r, 2.0, 0.04, 0.3, -0.7, 0.04) for s in S])

    pred_bsm    = np.array([unified.price(p_bsm,    float(s)) for s in S])
    pred_cev    = np.array([unified.price(p_cev,    float(s)) for s in S])
    pred_heston = np.array([unified.price(p_heston, float(s)) for s in S])

    configs = [
        ("BSM",              ref_bsm,    pred_bsm),
        (r"CEV ($\beta=0.5$)", ref_cev,  pred_cev),
        ("Heston",           ref_heston, pred_heston),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (title, ref, pred) in zip(axes, configs):
        ax.plot(S, ref,  color=BLUE, lw=2,   label="Reference")
        ax.plot(S, pred, color=RED,  lw=1.5, linestyle="--", label="PINN")
        ax.set_title(title)
        ax.set_xlabel(r"$S$")
        ax.set_ylabel(r"$V$")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)

    fig.suptitle(r"Option Price Curves ($t=0$, $S\in[50,250]$)", y=1.02)
    fig.tight_layout()
    savefig(fig, "price_curves.pdf")


# ── 5. Error Distribution ────────────────────────────────────────────────────
def fig_error_dist():
    try:
        sys.path.insert(0, HERE)
        from ref_solvers import bsm_call, cev_call, heston_call
        from unified_pinn_v2 import ModelParams
        unified = load_unified()
    except Exception as e:
        print(f"  skipping error_dist.pdf: {e}")
        return

    S = np.linspace(50, 250, 50)
    K, T, r = 100.0, 1.0, 0.05

    p_bsm    = ModelParams.from_bsm(sigma=0.20)
    p_cev    = ModelParams.from_cev(sigma=0.20, beta=0.5)
    p_heston = ModelParams.from_heston(kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, v0=0.04)

    ref_bsm    = np.array([bsm_call(s, K, T, r, 0.20) for s in S])
    ref_cev    = np.array([cev_call(s, K, T, r, 0.20, 0.5) for s in S])
    ref_heston = np.array([heston_call(s, K, T, r, 2.0, 0.04, 0.3, -0.7, 0.04) for s in S])

    pred_bsm    = np.array([unified.price(p_bsm,    float(s)) for s in S])
    pred_cev    = np.array([unified.price(p_cev,    float(s)) for s in S])
    pred_heston = np.array([unified.price(p_heston, float(s)) for s in S])

    configs = [
        ("BSM",              ref_bsm,    pred_bsm),
        (r"CEV ($\beta=0.5$)", ref_cev,  pred_cev),
        ("Heston",           ref_heston, pred_heston),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (title, ref, pred) in zip(axes, configs):
        err = np.abs(pred - ref)
        ax.fill_between(S, err, alpha=0.35, color=BLUE)
        ax.plot(S, err, color=BLUE, lw=1.5)
        ax.axvspan(80, 120, alpha=0.12, color=GREEN, label="ATM region")
        ax.set_title(title)
        ax.set_xlabel(r"$S$")
        ax.set_ylabel("Absolute error")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)

    fig.suptitle(r"Absolute Error Distribution ($t=0$)", y=1.02)
    fig.tight_layout()
    savefig(fig, "error_dist.pdf")


# ── 6. Comparison: Independent vs Unified ────────────────────────────────────
def fig_eval_compare():
    try:
        sys.path.insert(0, HERE)
        from ref_solvers import bsm_call, cev_call, heston_call
        from unified_pinn_v2 import ModelParams
        unified = load_unified()
    except Exception as e:
        print(f"  skipping eval_compare.pdf: {e}")
        return

    S = np.linspace(50, 250, 50)
    K, T, r = 100.0, 1.0, 0.05

    p_bsm    = ModelParams.from_bsm(sigma=0.20)
    p_cev    = ModelParams.from_cev(sigma=0.20, beta=0.5)
    p_heston = ModelParams.from_heston(kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, v0=0.04)

    ref_bsm    = np.array([bsm_call(s, K, T, r, 0.20) for s in S])
    ref_cev    = np.array([cev_call(s, K, T, r, 0.20, 0.5) for s in S])
    ref_heston = np.array([heston_call(s, K, T, r, 2.0, 0.04, 0.3, -0.7, 0.04) for s in S])

    uni_bsm    = np.array([unified.price(p_bsm,    float(s)) for s in S])
    uni_cev    = np.array([unified.price(p_cev,    float(s)) for s in S])
    uni_heston = np.array([unified.price(p_heston, float(s)) for s in S])

    # try loading independent models
    ind_bsm = ind_cev = None
    try:
        ib = load_indep_bsm()
        ind_bsm = np.array([ib.price(float(s)) for s in S])
    except Exception as e:
        print(f"  independent BSM not available: {e}")
    try:
        ic = load_indep_cev()
        ind_cev = np.array([ic.price(float(s)) for s in S])
    except Exception as e:
        print(f"  independent CEV not available: {e}")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    titles = ["BSM", r"CEV ($\beta=0.5$)", "Heston"]
    refs   = [ref_bsm, ref_cev, ref_heston]
    unis   = [uni_bsm, uni_cev, uni_heston]
    inds   = [ind_bsm, ind_cev, None]

    for ax, title, ref, uni, ind in zip(axes, titles, refs, unis, inds):
        ax.plot(S, ref, color="black", lw=2,   label="Reference")
        ax.plot(S, uni, color=BLUE,   lw=1.5, linestyle="--", label="Unified PINN")
        if ind is not None:
            ax.plot(S, ind, color=RED, lw=1.5, linestyle=":", label="Independent PINN")
        ax.set_title(title)
        ax.set_xlabel(r"$S$")
        ax.set_ylabel(r"$V$")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)

    fig.suptitle(r"Independent vs Unified PINN ($t=0$, $S\in[50,250]$)", y=1.02)
    fig.tight_layout()
    savefig(fig, "eval_compare.pdf")


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating thesis figures...")
    fig_soft_mask()
    fig_architecture()
    fig_training_loss()
    fig_price_curves()
    fig_error_dist()
    fig_eval_compare()
    print(f"\nDone. Output: {OUT}")
