"""Shared figure style."""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
FIG = os.path.join(ROOT, "figures")
DATA = os.path.join(ROOT, "data")
os.makedirs(FIG, exist_ok=True)
os.makedirs(DATA, exist_ok=True)

# categorical slots (fixed order) and markers as a secondary channel
C = ["#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac", "#008a17"]
MK = ["o", "s", "^", "D", "v", "P"]
INK = "#222222"
INK2 = "#6b6b6b"
GRID = "#e6e6e6"

plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.edgecolor": INK2,
    "axes.labelcolor": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "text.color": INK,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 2.0,
    "lines.markersize": 5,
    "legend.frameon": False,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


def panel_label(ax, s):
    ax.text(-0.14, 1.04, s, transform=ax.transAxes, fontweight="bold",
            fontsize=10, va="bottom")


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".png"))
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    plt.close(fig)
