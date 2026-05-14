"""Generate thesis figures: soft mask, system architecture."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                   "thesis", "Tex_thesis", "Img")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "text.usetex": False,
})

# ── Figure 1: Soft Mask Visualization ──────────────────────────────────────

def fig_soft_mask():
    xi = np.linspace(0, 0.5, 500)
    mask = np.tanh(xi / 0.05) ** 2

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xi, mask, 'b-', linewidth=2.5)
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.axhline(1, color='gray', linewidth=0.5, linestyle='--')

    # Annotate regimes
    ax.annotate(r'BSM/CEV regime: $\xi=0$, mask$\to 0$',
                xy=(0.02, 0.05), fontsize=10, color='#228B22',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#e8f5e9', alpha=0.8))
    ax.annotate(r'Heston regime: $\xi=0.3$, mask$\approx 0.998$',
                xy=(0.18, 0.92), fontsize=10, color='#C62828',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffebee', alpha=0.8))
    ax.annotate(r'$\mathrm{mask}=\tanh^2(\xi/0.05)$',
                xy=(0.28, 0.55), fontsize=12, color='#1565C0',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#e3f2fd', alpha=0.8))

    ax.set_xlabel(r'$\xi$ (vol-of-vol)', fontsize=13)
    ax.set_ylabel(r'mask', fontsize=13)
    ax.set_title('Soft Mask: Continuous Interpolation Between BSM/CEV and Heston', fontsize=14)
    ax.set_xlim(0, 0.5)
    ax.set_ylim(-0.05, 1.1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "soft_mask.pdf"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  soft_mask.pdf saved")


# ── Figure 2: System Architecture ──────────────────────────────────────────

def fig_architecture():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')

    def box(x, y, w, h, text, color='#E3F2FD', edge='#1565C0', fontsize=10, bold=False):
        rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                              boxstyle="round,pad=0.15", facecolor=color,
                              edgecolor=edge, linewidth=1.5)
        ax.add_patch(rect)
        weight = 'bold' if bold else 'normal'
        ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
                weight=weight, wrap=True)

    def arrow(x1, y1, x2, y2, color='#555555'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.8))

    # User
    box(1.0, 5.0, 2.0, 0.7, "User\nNatural Language", color='#F3E5F5', edge='#7B1FA2')

    # LLM Router
    box(3.2, 5.0, 2.4, 0.7, "LLM Router (DeepSeek)\nParam Extraction + Model Selection",
        color='#FFF3E0', edge='#E65100', bold=True)

    # JSON
    box(5.8, 5.0, 2.2, 0.7, 'Structured JSON\n{model, S, K, T, r, ...}',
        color='#E8F5E9', edge='#2E7D32')

    # Unified PINN
    box(5.0, 3.2, 3.8, 1.0,
        'Unified PINN Solver\n'
        r'$V = V_{\rm BS} + K \cdot {\rm Net}(S,v,t,\sigma,\beta,\kappa,\theta,\xi,\rho)$',
        color='#E3F2FD', edge='#1565C0', bold=True, fontsize=9)

    # Three models inside
    for i, (name, x_pos) in enumerate([('BSM', 3.4), ('CEV', 5.0), ('Heston', 6.6)]):
        box(x_pos, 2.15, 1.0, 0.45, name, color='#BBDEFB', edge='#1976D2', fontsize=9)

    # Output
    box(5.0, 1.0, 2.0, 0.7, "Option Price + Greeks", color='#C8E6C9', edge='#388E3C', bold=True)

    # Arrows
    arrow(2.0, 5.0, 2.0, 5.0)  # user -> LLM
    arrow(4.4, 5.0, 4.7, 5.0)  # LLM -> JSON
    arrow(5.8, 4.65, 5.8, 3.7)  # JSON -> PINN (vertical)
    arrow(5.0, 2.7, 5.0, 1.35)  # PINN -> Output

    # Data anchors (side)
    box(8.5, 3.2, 2.0, 1.0,
        "Data Anchors\nBSM: analytical\nCEV: Schroder 1989\nHeston: GL 96-pt",
        color='#FCE4EC', edge='#C62828', fontsize=8)
    arrow(7.0, 3.2, 7.5, 3.2, color='#C62828')

    ax.set_title('Unified Option Pricing Framework: LLM + PINN', fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "architecture.pdf"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  architecture.pdf saved")


if __name__ == "__main__":
    print("Generating thesis figures...")
    fig_soft_mask()
    fig_architecture()
    print("Done. Output:", OUT)
