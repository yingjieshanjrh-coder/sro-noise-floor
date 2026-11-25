"""Fig. 2: small-cell bias of the standard estimator and the breakdown of
normal-theory tests for dilute solutes, with GeSn (diamond lattice) as the
example."""
import csv
import os

import numpy as np

from _style import C, DATA, INK2, MK, panel_label, plt, save
from sronoise import WCNull, build_supercell, standard_bias

rng = np.random.default_rng(7)
N_SAMP = 200000

# ---------------------------------------------------------------- (a) bias
Ns = np.unique(np.logspace(np.log10(16), np.log10(4000), 60).astype(int))
concs = [0.05, 0.10, 0.20, 0.50]

# ------------------------------------------- (b,c) GeSn null distributions
def snsn_counts(cell, n_sn, n_samp, shell=0, batch=20000):
    """Number of Sn-Sn bonds (unordered) in random arrangements."""
    N = cell.n_sites
    nb = cell.neighbours[shell]
    out = np.empty(n_samp, dtype=np.int64)
    done = 0
    while done < n_samp:
        m = min(batch, n_samp - done)
        occ = np.zeros((m, N), dtype=bool)
        idx = np.argsort(rng.random((m, N)), axis=1)[:, :n_sn]
        np.put_along_axis(occ, idx, True, axis=1)
        out[done:done + m] = (occ[:, :, None] & occ[:, nb]).sum((1, 2)) // 2
        done += m
    return out


REDO = not os.path.exists(os.path.join(DATA, "gesn_null.csv"))
rows = []
reps_list = [2, 3, 4, 5]
xs = [0.03, 0.06, 0.08, 0.10, 0.125, 0.15, 0.20, 0.25]
for r in (reps_list if REDO else []):
    cell = build_supercell("diamond", r, 1)
    N = cell.n_sites
    for x in xs:
        n_sn = int(round(x * N))
        if n_sn < 2:
            continue
        null = WCNull(cell, [N - n_sn, n_sn], [0])
        # unordered Sn-Sn bond count = M_SnSn / 2
        mean_b = float(null._mean[(0, 1, 1)]) / 2
        var_b = float(null._cov[((0, 1, 1), (0, 1, 1))]) / 4
        cnt = snsn_counts(cell, n_sn, N_SAMP)
        zz = (cnt - mean_b) / np.sqrt(var_b)
        p0 = np.mean(cnt == 0)
        rows.append(dict(N=N, n_sn=n_sn, x=n_sn / N, mean_bonds=mean_b,
                         sd_bonds=np.sqrt(var_b),
                         sample_mean=cnt.mean(), sample_sd=cnt.std(),
                         typeI_z_two_sided=np.mean(np.abs(zz) > 1.959964),
                         typeI_z_avoid=np.mean(zz < -1.644854),
                         p_zero=p0,
                         alpha_std_bias=float(standard_bias(N, [N - n_sn,
                                                                n_sn], 1, 1)),
                         alpha_unb_sd=float(null.exact_var(0, 1, 1,
                                                           "unbiased")) ** .5))
        print(N, n_sn, f"<bonds>={mean_b:.2f}", f"P0={p0:.4f}",
              f"typeI={rows[-1]['typeI_z_avoid']:.3f}")

if REDO:
    with open(os.path.join(DATA, "gesn_null.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
else:
    with open(os.path.join(DATA, "gesn_null.csv")) as f:
        rows = [{k: (int(v) if k in ("N", "n_sn") else float(v))
                 for k, v in r.items()} for r in csv.DictReader(f)]

# ------------------------------------------------------------------ figure
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
ax = axes[0]
for i, c in enumerate(concs):
    b = (1 - c) / (c * (Ns - 1))
    ax.loglog(Ns, b, "-", color=C[i], label=f"$c_a$ = {c:g}")
for n in (64, 128, 256, 1000):
    ax.axvline(n, color=INK2, lw=0.6, ls=":", zorder=0)
ax.axhline(0.05, color=INK2, lw=0.8, zorder=0)
ax.text(17, 0.056, "0.05", color=INK2, fontsize=7)
ax.set_xlabel("sites in supercell, $N$")
ax.set_ylabel(r"null mean of standard $\alpha_{aa}$")
ax.legend(loc="upper right", fontsize=7)
panel_label(ax, "a")

ax = axes[1]
for i, r in enumerate(reps_list):
    sel = [d for d in rows if d["N"] == 8 * r ** 3]
    ax.semilogx([d["mean_bonds"] for d in sel],
                [d["typeI_z_avoid"] for d in sel], "-" + MK[i], color=C[i],
                ms=4, label=f"N = {8 * r ** 3}")
ax.axhline(0.05, color=INK2, lw=1.0, zorder=0)
ax.set_xlabel("expected Sn–Sn bonds (random)")
ax.set_ylabel("actual false-positive rate\n(nominal 5%, one-sided z)")
handles, labs = ax.get_legend_handles_labels()
panel_label(ax, "b")

ax = axes[2]
for i, r in enumerate(reps_list):
    sel = [d for d in rows if d["N"] == 8 * r ** 3]
    sel = [d for d in sel if d["p_zero"] > 0]       # below sampling floor
    p = np.array([d["p_zero"] for d in sel])
    ax.semilogy([100 * d["x"] for d in sel], p, "-" + MK[i], color=C[i],
                ms=4, label=f"N = {8 * r ** 3}")
ax.axhline(0.05, color=INK2, lw=1.0, zorder=0)
ax.set_xlabel("Sn content (at.%)")
ax.set_ylabel("smallest attainable p-value\n(zero Sn–Sn bonds)")
ax.set_xlim(0, 27)
panel_label(ax, "c")
fig.tight_layout(w_pad=1.2)
fig.legend(handles, labs, loc="lower center", ncol=4, fontsize=7,
           bbox_to_anchor=(0.68, 0.97))
save(fig, "fig2_dilute_gesn")
