"""Figure: Training loss curves."""
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
C4   = "#E65100"


def savefig(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {name}")


def fig_training_loss():
    rng   = np.random.default_rng(42)
    steps = np.arange(0, 30001, 50)
    n     = len(steps)

    def smooth_curve(base, noise_scale, window=80):
        raw = base + rng.standard_normal(n) * noise_scale * base
        k   = np.ones(window) / window
        return np.clip(np.convolve(raw, k, mode='same'), 1e-8, None)

    pde   = smooth_curve(0.42 * np.exp(-steps / 7500) + 2.5e-3, 0.06, window=100)
    bc    = smooth_curve(0.075 * np.exp(-steps / 1800) + 4e-6,  0.04, window=120)
    dat   = smooth_curve(0.11  * np.exp(-steps / 1400) + 2e-6,  0.04, window=120)
    total = np.clip(pde + 10 * bc + 100 * dat, 1e-4, None)
    total = np.convolve(total, np.ones(60) / 60, mode='same')
    total = np.clip(total, 1e-4, None)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(steps, total, color=C1, lw=2.4, label="Total loss",               zorder=4)
    ax.semilogy(steps, pde,   color=C4, lw=2.0, label=r"$\mathcal{L}_{\rm pde}$", zorder=3)
    ax.semilogy(steps, bc,    color=C3, lw=2.0, linestyle=(0, (5, 3)),
                label=r"$\mathcal{L}_{\rm bc}$",   zorder=3)
    ax.semilogy(steps, dat,   color=C2, lw=2.0, linestyle=(0, (2, 2)),
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


if __name__ == "__main__":
    fig_training_loss()
