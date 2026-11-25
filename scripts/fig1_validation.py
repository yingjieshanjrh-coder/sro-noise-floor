"""Fig. 1: exact null moments versus permutation sampling, and the
approach of the exact standard deviation to the leading-order formula."""
import csv
import os

import numpy as np

from _style import C, DATA, MK, INK2, panel_label, plt, save
from sronoise import (WCNull, build_supercell, random_configs,
                      sigma_asymptotic, warren_cowley)

rng = np.random.default_rng(20261125)
N_SAMP = 20000

cases = [
    ("fcc", 2, [8, 8, 8, 8]),
    ("fcc", 3, [22, 22, 22, 21, 21]),
    ("bcc", 3, [27, 27]),
    ("bcc", 4, [40, 44, 44]),
    ("hcp", (3, 2, 2), [12, 12, 12, 12]),
    ("hcp", (4, 3, 2), [48, 24, 24]),
    ("diamond", 2, [60, 4]),
    ("diamond", 2, [32, 24, 8]),
]

rows = []
for structure, reps, counts in cases:
    cell = build_supercell(structure, reps, 3)
    shells = [s for s in range(3) if not cell.has_self_bonds(s)]
    null = WCNull(cell, counts, shells)
    X = random_configs(counts, N_SAMP, rng)
    for est in ("standard", "unbiased"):
        A = warren_cowley(cell, X, shells, len(counts), est)
        A = A.reshape(N_SAMP, -1)
        mu, var = null.mean(est), np.diag(null.cov(est))
        for (s, a, b), m, v, em, ev in zip(null.labels(), mu, var,
                                           A.mean(0), A.var(0, ddof=1)):
            if a > b or not np.isfinite(v):
                continue
            rows.append(dict(structure=structure, N=cell.n_sites,
                             K=len(counts), shell=s + 1, a=a, b=b,
                             estimator=est, exact_mean=m, exact_var=v,
                             sample_mean=em, sample_var=ev,
                             n_samples=N_SAMP))
    print(structure, cell.n_sites, "done")

with open(os.path.join(DATA, "validation.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

# convergence of exact sigma to the leading-order expression
conv = []
for structure, sizes in [("fcc", [2, 3, 4, 5, 6, 8]),
                         ("bcc", [2, 3, 4, 5, 6, 8, 10]),
                         ("diamond", [2, 3, 4, 5, 6])]:
    for r in sizes:
        cell = build_supercell(structure, r, 1)
        N = cell.n_sites
        counts = [N // 5] * 5
        counts[-1] += N - sum(counts)
        null = WCNull(cell, counts, [0])
        B = N * cell.coordination(0) / 2
        c = np.array(counts) / N
        sd_ll = float(null.exact_var(0, 0, 0, "unbiased")) ** 0.5
        sd_ul = float(null.exact_var(0, 0, 1, "unbiased")) ** 0.5
        conv.append((structure, N,
                     sd_ll / sigma_asymptotic(c[0], c[0], B, True),
                     sd_ul / sigma_asymptotic(c[0], c[1], B, False)))

fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
ax = axes[0]
lat = sorted({r["structure"] for r in rows})
for i, L in enumerate(["fcc", "bcc", "hcp", "diamond"]):
    sel = [r for r in rows if r["structure"] == L]
    ev = np.array([r["exact_var"] for r in sel])
    sv = np.array([r["sample_var"] for r in sel])
    ax.loglog(ev, sv, MK[i], color=C[i], ms=4, mfc="none", mew=1.0,
              label=L, alpha=0.9)
lim = [1e-3, 3]
ax.plot(lim, lim, "-", color=INK2, lw=1.0, zorder=0)
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_xlabel(r"exact null variance of $\alpha$")
ax.set_ylabel(r"sampled variance ($2\times10^4$ configs)")
ax.legend(loc="upper left")
panel_label(ax, "a")

ax = axes[1]
for i, L in enumerate(["fcc", "bcc", "diamond"]):
    d = np.array([(n, x, y) for (s, n, x, y) in conv if s == L])
    ax.semilogx(d[:, 0], d[:, 1], "-" + MK[i], color=C[i], ms=4,
                label=f"{L}, like pair")
    ax.semilogx(d[:, 0], d[:, 2], "--" + MK[i], color=C[i], ms=4,
                mfc="white", label=f"{L}, unlike pair")
ax.axhline(1.0, color=INK2, lw=1.0, zorder=0)
ax.set_xlabel("sites in supercell, $N$")
ax.set_ylabel(r"$\sigma_{\rm exact}/\sigma_{\rm leading}$  (1st shell)")
ax.legend(ncol=1, fontsize=7, loc="lower right")
panel_label(ax, "b")
fig.tight_layout()
save(fig, "fig1_validation")

zs = []
for r in rows:
    se = np.sqrt(r["exact_var"] / N_SAMP)
    zs.append((r["sample_mean"] - r["exact_mean"]) / se)
zs = np.array(zs)
rel = np.array([r["sample_var"] / r["exact_var"] - 1 for r in rows])
print(f"{len(rows)} entries; mean z={zs.mean():+.3f} sd z={zs.std():.3f} "
      f"max|z|={np.abs(zs).max():.2f}; var ratio-1: mean {rel.mean():+.4f} "
      f"sd {rel.std():.4f} (expected sd {np.sqrt(2 / N_SAMP):.4f})")
