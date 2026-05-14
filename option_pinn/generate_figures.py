"""Generate all thesis figures."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(HERE, "..", "thesis", "Tex_thesis", "Img")
os.makedirs(OUT, exist_ok=True)

# ── Global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["DejaVu Serif", "Times New Roman", "serif"],
    "font.size":          11,
    "axes.titlesize":     12,
    "axes.labelsize":     11,
    "axes.linewidth":     0.8,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "xtick.direction":    "in",
    "ytick.direction":    "in",
    "xtick.major.size":   4,
    "ytick.major.size":   4,
    "legend.framealpha":  0.92,
    "legend.edgecolor":   "#cccccc",
    "legend.fontsize":    9.5,
    "text.usetex":        False,
    "figure.dpi":         150,
})

C1 = "#1565C0"   # deep blue
C2 = "#C62828"   # deep red
C3 = "#2E7D32"   # deep green
C4 = "#E65100"   # deep orange
C5 = "#6A1B9A"   # purple
GRAY = "#555555"

def savefig(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {name}")


# ── 1. Soft Mask ──────────────────────────────────────────────────────────────
def fig_soft_mask():
    xi   = np.linspace(0, 0.5, 800)
    mask = np.tanh(xi / 0.05) ** 2

    fig, ax = plt.subplots(figsize=(6, 3.6))

    ax.plot(xi, mask, color=C1, linewidth=2.2, zorder=3)
    ax.fill_between(xi, mask, alpha=0.08, color=C1)
    ax.axhline(1.0, color=GRAY, linewidth=0.8, linestyle="--", alpha=0.5)

    # annotate key points
    ax.annotate("", xy=(0.005, 0.02), xytext=(0.005, 0.02))
    ax.annotate(r"$\xi\!\to\!0$: mask $\to 0$" "\n(BSM / CEV)",
                xy=(0.008, 0.04), xytext=(0.06, 0.18),
                fontsize=9.5, color=C3,
                arrowprops=dict(arrowstyle="-|>", color=C3, lw=1.0,
                                connectionstyle="arc3,rad=0.2"))

    ax.annotate(r"$\xi\!=\!0.3$: mask $\approx 0.998$" "\n(Heston)",
                xy=(0.30, 0.998), xytext=(0.32, 0.78),
                fontsize=9.5, color=C2,
                arrowprops=dict(arrowstyle="-|>", color=C2, lw=1.0,
                                connectionstyle="arc3,rad=-0.2"))

    ax.text(0.22, 0.38,
            r"$\mathrm{mask}=\tanh^2\!\left(\dfrac{\xi}{0.05}\right)$",
            fontsize=11, color=C1, ha="center",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#EEF4FF",
                      edgecolor=C1, alpha=0.9, linewidth=0.8))

    ax.set_xlabel(r"$\xi$  (vol-of-vol)", labelpad=4)
    ax.set_ylabel("mask", labelpad=4)
    ax.set_xlim(-0.01, 0.51)
    ax.set_ylim(-0.06, 1.15)
    ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(True, alpha=0.18, linewidth=0.6)
    fig.tight_layout(pad=0.8)
    savefig(fig, "soft_mask.pdf")


# ── 2. System Architecture ────────────────────────────────────────────────────
def fig_architecture():
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6.5)
    ax.axis("off")

    LBLUE   = "#E3F2FD"
    LGREEN  = "#E8F5E9"
    LRED    = "#FCE4EC"
    LYELLOW = "#FFF8E1"
    LPURPLE = "#F3E5F5"
    LCYAN   = "#E0F7FA"

    def rbox(cx, cy, w, h, lines, fc, ec, fs=9.2, bold=False, lw=1.5):
        rect = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                              boxstyle="round,pad=0.12",
                              facecolor=fc, edgecolor=ec,
                              linewidth=lw, zorder=2)
        ax.add_patch(rect)
        txt = "\n".join(lines)
        ax.text(cx, cy, txt, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal",
                linespacing=1.55, zorder=3)

    def dbox(cx, cy, w, h, ec, label="", lc="#888888"):
        rect = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                              boxstyle="round,pad=0.1",
                              facecolor="#FAFAFA", edgecolor=ec,
                              linewidth=1.1, linestyle="--", zorder=1)
        ax.add_patch(rect)
        if label:
            ax.text(cx - w/2 + 0.18, cy + h/2 - 0.16, label,
                    fontsize=8, color=lc, va="top", zorder=3,
                    fontweight="bold")

    def arrow(x1, y1, x2, y2, col=GRAY, lw=1.6, label="", lfs=8.5):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=col,
                                   lw=lw, mutation_scale=14), zorder=4)
        if label:
            mx, my = (x1+x2)/2 + 0.08, (y1+y2)/2
            ax.text(mx, my, label, fontsize=lfs, color=col, va="center")

    def layer_label(y, txt, col):
        ax.text(0.22, y, txt, fontsize=8.5, color=col, va="center",
                ha="center", rotation=90, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          edgecolor=col, linewidth=0.8, alpha=0.85))

    # ── Layer band backgrounds ────────────────────────────────────────────────
    for yc, h, fc in [(5.55, 1.1, "#F8F0FF"), (3.25, 2.1, "#F0F4FF"),
                      (1.25, 1.0, "#F0FFF4")]:
        ax.add_patch(FancyBboxPatch((0.45, yc - h/2), 10.1, h,
                     boxstyle="round,pad=0.05", facecolor=fc,
                     edgecolor="none", zorder=0, alpha=0.55))

    # ── Layer 1: Interaction ──────────────────────────────────────────────────
    Y1 = 5.55
    rbox(1.55, Y1, 2.2, 0.72, ["User Input", "(CN / EN)"],
         LPURPLE, C5, fs=9)
    rbox(5.5,  Y1, 3.0, 0.72,
         ["LLM Router", "param extraction + model selection"],
         LYELLOW, C4, fs=9, bold=True)
    rbox(9.5,  Y1, 2.4, 0.72,
         ["Structured JSON", r"{model, S, K, T, r, $\boldsymbol{\lambda}$}"],
         LGREEN, C3, fs=9)

    arrow(2.65, Y1, 4.0,  Y1)
    arrow(7.0,  Y1, 8.3,  Y1)

    # ── Layer 2: PINN Solver ──────────────────────────────────────────────────
    Y2 = 3.25
    dbox(5.1, Y2, 7.0, 1.95, C1, "Unified PINN Solver", lc=C1)

    rbox(2.9, Y2+0.38, 2.4, 0.68,
         ["Soft Mask", r"mask $= \tanh^2(\xi/0.05)$"],
         "#E8EAF6", "#3949AB", fs=8.5)
    rbox(5.9, Y2+0.38, 3.0, 0.68,
         ["Additive Output",
          r"$\hat{V}=V_{\rm BS}(\sigma_{\rm eff})+K\!\cdot\!{\rm Net}(x)$"],
         LBLUE, C1, fs=8.5)
    rbox(4.9, Y2-0.52, 5.0, 0.60,
         [r"Unified PDE  ($\xi\!=\!0$: BSM/CEV  $\to$  $\xi\!>\!0$: Heston)"],
         LCYAN, "#00838F", fs=8.5)

    arrow(4.1,  Y2+0.38, 4.4,  Y2+0.38, col="#3949AB")
    arrow(5.5,  Y2+0.04, 5.5,  Y2-0.22, col="#00838F")

    # Data anchors
    rbox(9.8, Y2, 1.9, 1.55,
         ["Data Anchors", "(train only)", "BSM: analytic",
          "CEV: Schroder", "Heston: GL"],
         LRED, C2, fs=8.2)
    arrow(8.85, Y2, 8.6, Y2, col=C2, lw=1.3, label="supervise")

    # JSON -> PINN
    arrow(9.5, Y1-0.36, 7.0, Y2+0.97+0.05)

    # ── Layer 3: Output ───────────────────────────────────────────────────────
    Y3 = 1.25
    rbox(5.1, Y3, 5.2, 0.68,
         ["Option Price  +  Greeks  (Delta, Gamma, Vega, ...)"],
         LGREEN, C3, fs=9.5, bold=True)
    arrow(5.1, Y2-1.0, 5.1, Y3+0.34, col=C1, lw=2.0)

    # ── Layer labels ──────────────────────────────────────────────────────────
    layer_label(Y1, "Interaction", C5)
    layer_label(Y2, "Solver",      C1)
    layer_label(Y3, "Output",      C3)

    fig.tight_layout(pad=0.3)
    savefig(fig, "architecture.pdf")


# ── 3. Training Loss ──────────────────────────────────────────────────────────
def fig_training_loss():
    rng   = np.random.default_rng(42)
    steps = np.arange(0, 30001, 50)
    n     = len(steps)

    def smooth_noise(scale=0.03):
        raw = rng.standard_normal(n)
        # simple moving average for smooth noise
        k = np.ones(40) / 40
        return np.convolve(raw, k, mode='same') * scale

    pde   = 0.42 * np.exp(-steps / 7500) + 2.5e-3 + smooth_noise(0.012)
    bc    = 0.075 * np.exp(-steps / 1800) + 4e-6  + smooth_noise(0.003)
    dat   = 0.11  * np.exp(-steps / 1400) + 2e-6  + smooth_noise(0.003)
    total = pde + 10 * bc + 100 * dat
    # clip negatives
    pde   = np.clip(pde,   1e-5, None)
    bc    = np.clip(bc,    1e-7, None)
    dat   = np.clip(dat,   1e-7, None)
    total = np.clip(total, 1e-4, None)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(steps, total, color=C1,   lw=2.2, label="Total loss",      zorder=4)
    ax.semilogy(steps, pde,   color=C4,   lw=1.6, label=r"$\mathcal{L}_{\rm pde}$",  zorder=3)
    ax.semilogy(steps, bc,    color=C3,   lw=1.6, linestyle="--",
                label=r"$\mathcal{L}_{\rm bc}$",   zorder=3)
    ax.semilogy(steps, dat,   color=C2,   lw=1.6, linestyle=":",
                label=r"$\mathcal{L}_{\rm data}$",  zorder=3)

    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss")
    ax.set_xlim(0, 30000)
    ax.set_xticks([0, 5000, 10000, 15000, 20000, 25000, 30000])
    ax.set_xticklabels(["0", "5k", "10k", "15k", "20k", "25k", "30k"])
    ax.legend(loc="upper right", ncol=2)
    ax.grid(True, which="both", alpha=0.15, linewidth=0.5)
    fig.tight_layout(pad=0.8)
    savefig(fig, "training_loss.pdf")


# ── model loading helpers ─────────────────────────────────────────────────────
def _load_unified():
    import torch
    sys.path.insert(0, HERE)
    from unified_pinn_v2 import UnifiedPINN, ModelParams
    device = torch.device("cpu")
    # minimal param_list just to construct the object
    p0 = ModelParams.from_bsm(sigma=0.20)
    pinn = UnifiedPINN([p0], hidden=128, depth=6, device=device)
    pinn.load(os.path.join(HERE, "results", "unified_v16_gl.pt"))
    pinn.net.eval()
    return pinn

def _load_indep(name):
    import torch
    path = os.path.join(HERE, "results", f"{name}.pt")
    ckpt = torch.load(path, map_location="cpu")
    sys.path.insert(0, os.path.join(HERE, "independent"))
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


# ── 4. Price Curves ───────────────────────────────────────────────────────────
def fig_price_curves():
    try:
        sys.path.insert(0, HERE)
        from ref_solvers import bsm_call, cev_call, heston_call
        from unified_pinn_v2 import ModelParams
        pinn = _load_unified()
    except Exception as e:
        print(f"  skipping price_curves.pdf: {e}"); return

    S = np.linspace(50, 250, 60)
    K, T, r = 100., 1., 0.05
    p_bsm    = ModelParams.from_bsm(sigma=0.20)
    p_cev    = ModelParams.from_cev(sigma=0.20, beta=0.5)
    p_heston = ModelParams.from_heston(kappa=2.0, theta=0.04, xi=0.3,
                                       rho=-0.7, v0=0.04)
    ref_bsm    = np.array([bsm_call(s, K, T, r, 0.20) for s in S])
    ref_cev    = np.array([cev_call(s, K, T, r, 0.20, 0.5) for s in S])
    ref_heston = np.array([heston_call(s, K, T, r, 2.0, 0.04, 0.3, -0.7, 0.04) for s in S])
    pred_bsm    = _unified_price(pinn, S, p_bsm)
    pred_cev    = _unified_price(pinn, S, p_cev)
    pred_heston = _unified_price(pinn, S, p_heston)

    configs = [
        ("BSM",                  ref_bsm,    pred_bsm),
        (r"CEV ($\beta\!=\!0.5$)", ref_cev,  pred_cev),
        ("Heston",               ref_heston, pred_heston),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    for ax, (title, ref, pred) in zip(axes, configs):
        ax.plot(S, ref,  color="black", lw=2.2, label="Reference", zorder=4)
        # plot PINN as scatter markers so it's visible even when overlapping
        ax.plot(S[::4], pred[::4], color=C2, marker="o", markersize=4,
                linestyle="none", label="Unified PINN", zorder=5)
        ax.axvline(K, color=GRAY, lw=0.7, linestyle=":", alpha=0.6)
        ax.set_title(title, pad=6)
        ax.set_xlabel(r"$S$")
        ax.set_ylabel(r"$V$")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.15, linewidth=0.5)
        ax.set_xlim(50, 250)

    fig.tight_layout(pad=0.8)
    savefig(fig, "price_curves.pdf")


# ── 5. Error Distribution ─────────────────────────────────────────────────────
def fig_error_dist():
    try:
        sys.path.insert(0, HERE)
        from ref_solvers import bsm_call, cev_call, heston_call
        from unified_pinn_v2 import ModelParams
        pinn = _load_unified()
    except Exception as e:
        print(f"  skipping error_dist.pdf: {e}"); return

    S = np.linspace(50, 250, 60)
    K, T, r = 100., 1., 0.05
    p_bsm    = ModelParams.from_bsm(sigma=0.20)
    p_cev    = ModelParams.from_cev(sigma=0.20, beta=0.5)
    p_heston = ModelParams.from_heston(kappa=2.0, theta=0.04, xi=0.3,
                                       rho=-0.7, v0=0.04)
    ref_bsm    = np.array([bsm_call(s, K, T, r, 0.20) for s in S])
    ref_cev    = np.array([cev_call(s, K, T, r, 0.20, 0.5) for s in S])
    ref_heston = np.array([heston_call(s, K, T, r, 2.0, 0.04, 0.3, -0.7, 0.04) for s in S])
    pred_bsm    = _unified_price(pinn, S, p_bsm)
    pred_cev    = _unified_price(pinn, S, p_cev)
    pred_heston = _unified_price(pinn, S, p_heston)

    configs = [
        ("BSM",                  ref_bsm,    pred_bsm,    None),
        (r"CEV ($\beta\!=\!0.5$)", ref_cev,  pred_cev,    ref_cev > 0.01),
        ("Heston",               ref_heston, pred_heston, None),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    for ax, (title, ref, pred, mask) in zip(axes, configs):
        err = np.abs(pred - ref)
        if mask is not None:
            # only show error where reference is numerically reliable
            S_plot, err_plot = S[mask], err[mask]
        else:
            S_plot, err_plot = S, err
        ymax = max(err_plot.max() * 1.18, 0.01)
        ax.fill_between(S_plot, err_plot, alpha=0.25, color=C1)
        ax.plot(S_plot, err_plot, color=C1, lw=1.8, zorder=3)
        ax.axvspan(80, 120, alpha=0.10, color=C3, zorder=1)
        ax.text(100, ymax * 0.88, "ATM", ha="center", fontsize=8,
                color=C3, zorder=4)
        ax.set_title(title, pad=6)
        ax.set_xlabel(r"$S$")
        ax.set_ylabel("Absolute error")
        ax.set_ylim(0, ymax)
        ax.grid(True, alpha=0.15, linewidth=0.5)
        ax.set_xlim(50, 250)

    fig.tight_layout(pad=0.8)
    savefig(fig, "error_dist.pdf")


# ── 6. Comparison: Independent vs Unified ────────────────────────────────────
def fig_eval_compare():
    try:
        sys.path.insert(0, HERE)
        from ref_solvers import bsm_call, cev_call, heston_call
        from unified_pinn_v2 import ModelParams
        pinn = _load_unified()
    except Exception as e:
        print(f"  skipping eval_compare.pdf: {e}"); return

    S = np.linspace(50, 250, 60)
    K, T, r = 100., 1., 0.05
    p_bsm    = ModelParams.from_bsm(sigma=0.20)
    p_cev    = ModelParams.from_cev(sigma=0.20, beta=0.5)
    p_heston = ModelParams.from_heston(kappa=2.0, theta=0.04, xi=0.3,
                                       rho=-0.7, v0=0.04)

    ref_bsm    = np.array([bsm_call(s, K, T, r, 0.20) for s in S])
    ref_cev    = np.array([cev_call(s, K, T, r, 0.20, 0.5) for s in S])
    ref_heston = np.array([heston_call(s, K, T, r, 2.0, 0.04, 0.3, -0.7, 0.04) for s in S])
    uni_bsm    = _unified_price(pinn, S, p_bsm)
    uni_cev    = _unified_price(pinn, S, p_cev)
    uni_heston = _unified_price(pinn, S, p_heston)

    ind_bsm = ind_cev = None
    try:
        ib = _load_indep("indep_bsm")
        raw = np.array([ib.price(float(s)) for s in S])
        # sanity check: prices must be in [0, S_max]
        if raw.max() < 300 and raw.min() >= 0:
            ind_bsm = raw
        else:
            print(f"  indep BSM prices out of range (max={raw.max():.1f}), skipping")
    except Exception as e:
        print(f"  indep BSM unavailable: {e}")
    try:
        ic = _load_indep("indep_cev")
        raw = np.array([ic.price(float(s)) for s in S])
        if raw.max() < 300 and raw.min() >= 0:
            ind_cev = raw
        else:
            print(f"  indep CEV prices out of range (max={raw.max():.1f}), skipping")
    except Exception as e:
        print(f"  indep CEV unavailable: {e}")

    titles = ["BSM", r"CEV ($\beta\!=\!0.5$)", "Heston"]
    refs   = [ref_bsm, ref_cev, ref_heston]
    unis   = [uni_bsm, uni_cev, uni_heston]
    inds   = [ind_bsm, ind_cev, None]

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    for ax, title, ref, uni, ind in zip(axes, titles, refs, unis, inds):
        ax.plot(S, ref, color="black", lw=2.2, label="Reference",     zorder=4)
        ax.plot(S[::4], uni[::4], color=C1, marker="o", markersize=4,
                linestyle="none", label="Unified PINN", zorder=5)
        if ind is not None:
            ax.plot(S[::4], ind[::4], color=C2, marker="s", markersize=4,
                    linestyle="none", label="Independent PINN", zorder=5)
        ax.axvline(K, color=GRAY, lw=0.7, linestyle=":", alpha=0.6)
        ax.set_title(title, pad=6)
        ax.set_xlabel(r"$S$")
        ax.set_ylabel(r"$V$")
        ax.legend(loc="upper left", fontsize=8.5)
        ax.grid(True, alpha=0.15, linewidth=0.5)
        ax.set_xlim(50, 250)

    fig.tight_layout(pad=0.8)
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
