"""Figure: Soft mask function."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

HERE = os.path.dirname(os.path.abspath(__file__))
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

C1   = "#1565C0"
C2   = "#C62828"
C3   = "#2E7D32"
GRAY = "#555555"


def savefig(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {name}")


def fig_soft_mask():
    xi   = np.linspace(0, 0.5, 800)
    mask = np.tanh(xi / 0.05) ** 2

    fig, ax = plt.subplots(figsize=(6, 3.6))

    ax.plot(xi, mask, color=C1, linewidth=2.2, zorder=3)
    ax.fill_between(xi, mask, alpha=0.08, color=C1)
    ax.axhline(1.0, color=GRAY, linewidth=0.8, linestyle="--", alpha=0.5)

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


if __name__ == "__main__":
    fig_soft_mask()
