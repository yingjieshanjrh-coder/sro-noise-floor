"""Fig. 3: noise floor of first-shell Warren-Cowley parameters and the
supercell size needed to resolve a given degree of short-range order."""
import csv
import os

import numpy as np

from _style import C, DATA, INK2, MK, panel_label, plt, save
from sronoise import WCNull, build_supercell, min_sites, sigma_asymptotic

# (a) minimum N, leading order, equiatomic K components, fcc first shell
alphas = np.logspace(-2, np.log10(0.5), 80)
Ks = [2, 3, 4, 5, 6]
z = 12

# (b) exact noise floor, equiatomic quinary on fcc, first shell
rows = []
for r in [2, 3, 4, 5, 6, 7, 8]:
    cell = build_supercell("fcc", r, 1)
    N = cell.n_sites
    counts = [N // 5] * 5
    for i in range(N - sum(counts)):
        counts[i] += 1
    null = WCNull(cell, counts, [0])
    sd = np.sqrt(np.diag(null.cov("unbiased")))
    labs = null.labels()
    like = np.mean([s for s, (_, a, b) in zip(sd, labs) if a == b])
    unlike = np.mean([s for s, (_, a, b) in zip(sd, labs) if a < b])
    rows.append(dict(N=N, sigma_like=like, sigma_unlike=unlike))
    print(N, round(like, 4), round(unlike, 4))
with open(os.path.join(DATA, "noise_floor_fcc_quinary.csv"), "w",
          newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
ax = axes[0]
for i, K in enumerate(Ks):
    c = 1.0 / K
    ax.loglog(alphas, [min_sites(a, c, c, z, 2.0, True) for a in alphas],
              "-", color=C[i], label=f"K = {K}")
for n in (108, 256, 500, 2048):
    ax.axhline(n, color=INK2, lw=0.6, ls=":", zorder=0)
    ax.text(0.0105, n * 1.08, f"{n}", color=INK2, fontsize=7)
ax.set_xlabel(r"true $|\alpha|$ to be resolved (first shell)")
ax.set_ylabel("sites needed for a 2σ signal")
ax.set_ylim(10, 3e6)
ax.legend(loc="upper right", fontsize=7, title="like pairs, equiatomic",
          title_fontsize=7)
panel_label(ax, "a")

ax = axes[1]
N = np.array([d["N"] for d in rows])
ax.loglog(N, [d["sigma_like"] for d in rows], "-o", color=C[0], ms=4,
          label="like pairs (exact)")
ax.loglog(N, [d["sigma_unlike"] for d in rows], "-s", color=C[1], ms=4,
          label="unlike pairs (exact)")
Nf = np.logspace(np.log10(20), np.log10(3000), 50)
ax.loglog(Nf, [sigma_asymptotic(.2, .2, n * 6, True) for n in Nf], ":",
          color=C[0], lw=1.2)
ax.loglog(Nf, [sigma_asymptotic(.2, .2, n * 6, False) for n in Nf], ":",
          color=C[1], lw=1.2)
for lv in (0.05, 0.1):
    ax.axhline(lv, color=INK2, lw=0.8, zorder=0)
    ax.text(2200, lv * 1.07, f"{lv:g}", color=INK2, fontsize=7)
ax.set_xlabel("sites in supercell, $N$")
ax.set_ylabel(r"null s.d. of $\alpha$, one snapshot")
ax.legend(loc="lower left", fontsize=7)
ax.set_title("equiatomic quinary, fcc, first shell", fontsize=8,
             color=INK2)
panel_label(ax, "b")
fig.tight_layout()
save(fig, "fig3_detectability")
