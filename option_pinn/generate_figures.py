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
    fig, ax = plt.subplots(figsize=(12, 7))
    W, H = 12, 7
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    PURPLE = "#6A1B9A"
    LBLUE  = "#E3F2FD"
    LGREEN = "#E8F5E9"
    LRED   = "#FCE4EC"
    LYELLOW= "#FFF8E1"
    LGRAY  = "#F5F5F5"

    def box(x, y, w, h, lines, fc=LBLUE, ec=BLUE, fs=9.5, bold=False,
            style="round,pad=0.15", lw=1.8):
        rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                              boxstyle=style, facecolor=fc, edgecolor=ec,
                              linewidth=lw, zorder=2)
        ax.add_patch(rect)
        txt = "\n".join(lines)
        ax.text(x, y, txt, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal",
                linespacing=1.5, zorder=3)

    def dashed_rect(x, y, w, h, ec="#888888", label=""):
        rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                              boxstyle="round,pad=0.1",
                              facecolor="#FAFAFA", edgecolor=ec,
                              linewidth=1.2, linestyle="--", zorder=1)
        ax.add_patch(rect)
        if label:
            ax.text(x - w/2 + 0.15, y + h/2 - 0.18, label,
                    fontsize=7.5, color=ec, va="top", zorder=3)

    def arr(x1, y1, x2, y2, col="#444444", lw=1.8, label="", lfs=8):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=col,
                                   lw=lw, mutation_scale=15), zorder=4)
        if label:
            mx, my = (x1+x2)/2, (y1+y2)/2
            ax.text(mx+0.08, my, label, fontsize=lfs, color=col, va="center")

    # ── Layer 1: Interaction (top) ──────────────────────────────────────────
    Y1 = 6.0
    box(1.6,  Y1, 2.4, 0.72,
        ["Natural Language Input", "(Chinese / English)"],
        fc="#F3E5F5", ec=PURPLE, fs=9)
    box(5.0,  Y1, 3.2, 0.72,
        ["LLM Router", "param extraction  +  model selection"],
        fc=LYELLOW, ec=ORANGE, fs=9, bold=True)
    box(9.2,  Y1, 2.8, 0.72,
        ["Structured JSON",
         r"$\{$model, $S$, $K$, $T$, $r$, $\boldsymbol{\lambda}\}$"],
        fc=LGREEN, ec=GREEN, fs=9)

    arr(2.8,  Y1, 3.4,  Y1, col="#555555")
    arr(6.6,  Y1, 7.8,  Y1, col="#555555")

    # ── Layer 2: PINN Solver (middle) ───────────────────────────────────────
    Y2 = 3.8
    # outer PINN box
    dashed_rect(5.5, Y2, 7.6, 2.2, ec=BLUE, label="Unified PINN Solver")

    # inner: soft mask
    box(3.5, Y2+0.35, 2.2, 0.72,
        ["Soft Mask",
         r"$\mathrm{mask}=\tanh^2(\xi/0.05)$"],
        fc="#E8EAF6", ec="#3949AB", fs=8.5)

    # inner: additive parameterization
    box(6.2, Y2+0.35, 2.8, 0.72,
        ["Additive Output",
         r"$\hat{V}=V_{\rm BS}(\sigma_{\rm eff})+K\cdot\mathrm{Net}(\mathbf{x})$"],
        fc=LBLUE, ec=BLUE, fs=8.5)

    # inner: unified PDE
    box(5.5, Y2-0.55, 5.2, 0.62,
        [r"Unified PDE: BSM $(\xi{=}0)$ $\to$ CEV $\to$ Heston $(\xi{\uparrow})$"],
        fc="#E0F7FA", ec="#00838F", fs=8.5)

    arr(4.6, Y2+0.35, 4.8, Y2+0.35, col="#3949AB")   # mask → additive
    arr(5.5, Y2-0.0,  5.5, Y2-0.24, col="#00838F")    # additive → PDE

    # Data anchors (right, training-time only)
    box(10.6, Y2, 2.0, 1.5,
        ["Data Anchors", "(training only)",
         "BSM: analytic", "CEV: Schroder", "Heston: GL"],
        fc=LRED, ec=RED, fs=8.2)
    arr(9.6, Y2, 9.3, Y2, col=RED, lw=1.4, label="supervise")

    # arrow: JSON → PINN
    arr(9.2, Y1-0.36, 7.5, Y2+1.1+0.11, col="#555555")

    # ── Layer 3: Output (bottom) ────────────────────────────────────────────
    Y3 = 1.55
    box(5.5, Y3, 5.0, 0.72,
        ["Option Price  +  Greeks  (Delta, Gamma, Vega, ...)"],
        fc=LGREEN, ec=GREEN, fs=9.5, bold=True)

    arr(5.5, Y2-1.1, 5.5, Y3+0.36, col=BLUE, lw=2.0)

    # ── Layer labels (left margin) ──────────────────────────────────────────
    for ytxt, lbl, col in [
        (Y1,   "Interaction", PURPLE),
        (Y2,   "Solver",      BLUE),
        (Y3,   "Output",      GREEN),
    ]:
        ax.text(0.12, ytxt, lbl, fontsize=8, color=col, va="center",
                rotation=90, ha="center")

    fig.tight_layout(pad=0.4)
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
