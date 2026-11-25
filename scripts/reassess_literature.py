"""Reassessment of published small-cell Warren-Cowley values against the
exact random-cell null.

Sources (values transcribed from the publications):
  T15  A. Tamm et al., Acta Mater. 99, 307 (2015), doi:10.1016/j.actamat.2015.08.015
       DFT-MC, 108-atom fcc cell, 400 (ternary) / 600 (quaternary) swap
       trials in total (4-6 per atom), SRO averaged over the last steps.
       Tables 1 and 3, first shell.
  D18  J. Ding et al., PNAS 115, 8919 (2018), doi:10.1073/pnas.1808660115
       DFT-MC at 500 K, 18 independent 180-atom CrCoNi cells.  Reported as
       dDelta = N0 - N with N0 = z c_j = 4 for the first shell, so
       alpha = dDelta / 4.
  C20  B. Cao et al., ACS Appl. Mater. Interfaces 12, 57245 (2020);
       arXiv:2010.02991
       DFT-MC, 64-atom Ge0.75Sn0.25 (16 Sn), Sn-Sn first-shell coordination
       number 0.33 +- 0.02 at 300 K, compared with 1 for a random alloy.
"""
import csv
import os

import numpy as np
from scipy import stats

from _style import C, DATA, INK2, MK, panel_label, plt, save
from sronoise import WCNull, build_supercell

rows = []


def null_for(structure, reps, counts):
    cell = build_supercell(structure, reps, 1)
    return cell, WCNull(cell, counts, [0])


# --------------------------------------------------------------- Tamm 2015
T15 = {
    "NiCrCo": (["Ni", "Cr", "Co"], [36, 36, 36], {
        ("Ni", "Ni"): (0.03, 0.01, 0.04), ("Ni", "Cr"): (-0.15, -0.14, -0.08),
        ("Ni", "Co"): (0.18, 0.10, 0.12), ("Cr", "Cr"): (0.31, 0.26, 0.12),
        ("Cr", "Co"): (-0.17, -0.12, -0.04), ("Co", "Co"): (-0.01, 0.02, -0.08),
    }),
    "NiCrCoFe": (["Ni", "Cr", "Co", "Fe"], [27, 27, 27, 27], {
        ("Ni", "Ni"): (0.20, 0.18, 0.04), ("Ni", "Cr"): (-0.15, -0.12, -0.13),
        ("Ni", "Co"): (0.10, 0.05, 0.14), ("Ni", "Fe"): (-0.15, -0.11, -0.05),
        ("Cr", "Cr"): (0.44, 0.21, 0.39), ("Cr", "Co"): (-0.24, -0.08, -0.19),
        ("Cr", "Fe"): (-0.06, -0.01, -0.08), ("Co", "Co"): (0.28, 0.13, 0.02),
        ("Co", "Fe"): (-0.14, -0.10, 0.01), ("Fe", "Fe"): (0.35, 0.22, 0.11),
    }),
}
TEMPS = (500, 800, 1200)
global_tests = []
resid = {"like": [], "unlike": []}
for alloy, (names, counts, table) in T15.items():
    cell, null = null_for("fcc", 3, counts)
    K = len(names)
    mu = null.mean("standard").reshape(K, K)
    cov = null.cov("standard")
    sd = np.sqrt(np.diag(cov)).reshape(K, K)
    for it, T in enumerate(TEMPS):
        A = np.zeros((K, K))
        for (p, q), vals in table.items():
            a, b = names.index(p), names.index(q)
            A[a, b] = A[b, a] = vals[it]
            z = (vals[it] - mu[a, b]) / sd[a, b]
            rows.append(dict(source="Tamm 2015", system=alloy, N=108,
                             T=T, pair=f"{p}-{q}", alpha=vals[it],
                             null_mean=mu[a, b], null_sd=sd[a, b], z=z,
                             p_two_sided=2 * stats.norm.sf(abs(z)),
                             n_eff_for_2sigma=(2 * sd[a, b] / abs(vals[it] -
                                               mu[a, b])) ** 2))
        # global chi2 on entries a <= b (pseudo-inverse of exact covariance)
        labs = null.labels()
        keep = np.array([a <= b for (_, a, b) in labs])
        d = (A.reshape(-1) - mu.reshape(-1))[keep]
        Ck = cov[np.ix_(keep, keep)]
        w, V = np.linalg.eigh(Ck)
        good = w > w.max() * 1e-9
        chi2 = float(np.sum((V[:, good].T @ d) ** 2 / w[good]))
        dof = int(good.sum())
        global_tests.append((alloy, T, chi2, dof, stats.chi2.sf(chi2, dof)))
    # scatter about a smooth temperature dependence alpha = u + v / T
    X = np.column_stack([np.ones(3), 1.0 / np.array(TEMPS)])
    for (p, q), vals in table.items():
        a, b = names.index(p), names.index(q)
        coef, res, *_ = np.linalg.lstsq(X, np.array(vals), rcond=None)
        rss = float(np.sum((np.array(vals) - X @ coef) ** 2))
        kind = "like" if p == q else "unlike"
        resid[kind].append(rss / sd[a, b] ** 2)   # in units of null var

print("Tamm 2015, global chi2 per temperature (single snapshot null):")
for g in global_tests:
    print("  %-9s %4d K  chi2=%6.2f  dof=%d  p=%.3g" % g)
ratio = np.mean(resid["like"] + resid["unlike"])
print(f"  scatter about alpha=u+v/T: mean RSS / null var = {ratio:.2f} "
      f"(1 dof per pair) -> implied n_eff ~ {1 / ratio:.1f}")

# --------------------------------------------------------------- Ding 2018
cell, null = null_for("fcc", (3, 3, 5), [60, 60, 60])
mu = null.mean("standard").reshape(3, 3)
sd = np.sqrt(np.diag(null.cov("standard"))).reshape(3, 3)
for (p, q, dd) in [("Cr", "Cr", 1.58), ("Cr", "Co", -0.98),
                   ("Cr", "Ni", -0.61)]:
    a = 0
    b = {"Cr": 0, "Co": 1, "Ni": 2}[q]
    al = dd / 4.0
    for n_eff in (1, 18):
        z = (al - mu[a, b]) / (sd[a, b] / np.sqrt(n_eff))
        rows.append(dict(source="Ding 2018", system="CrCoNi", N=180, T=500,
                         pair=f"{p}-{q}" + ("" if n_eff == 1 else
                                            " (18 cells)"),
                         alpha=al, null_mean=mu[a, b],
                         null_sd=sd[a, b] / np.sqrt(n_eff), z=z,
                         p_two_sided=2 * stats.norm.sf(abs(z)),
                         n_eff_for_2sigma=(2 * sd[a, b] / abs(al - mu[a, b]))
                         ** 2))

# ---------------------------------------------------------------- Cao 2020
cell, null = null_for("diamond", 2, [48, 16])
cn_rand_cell = 4 * 15 / 63
al_std = 1 - 0.33 / 1.0
al_unb = 1 - 0.33 / cn_rand_cell
sd_u = float(null.exact_var(0, 1, 1, "unbiased")) ** 0.5
rows.append(dict(source="Cao 2020", system="Ge0.75Sn0.25", N=64, T=300,
                 pair="Sn-Sn", alpha=al_unb, null_mean=0.0, null_sd=sd_u,
                 z=al_unb / sd_u, p_two_sided=2 * stats.norm.sf(al_unb / sd_u),
                 n_eff_for_2sigma=(2 * sd_u / al_unb) ** 2))
print(f"\nCao 2020: random Sn-Sn CN in a 64-atom cell with 16 Sn = "
      f"{cn_rand_cell:.3f} (not 1); alpha = {al_std:.3f} (vs infinite) / "
      f"{al_unb:.3f} (vs cell); single-snapshot null sd = {sd_u:.3f}")
for x, nsn in [(0.125, 8), (0.1875, 12), (0.25, 16)]:
    print(f"  x={x}: random CN, infinite = {4 * x:.3f}, 64-atom cell = "
          f"{4 * (nsn - 1) / 63:.3f}  ({100 * (1 - (nsn - 1) / 63 / x):.1f}% "
          f"lower)")

keys = list(rows[0])
with open(os.path.join(DATA, "literature_reassessment.csv"), "w",
          newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader()
    w.writerows(rows)
with open(os.path.join(DATA, "literature_global_tests.csv"), "w",
          newline="") as f:
    w = csv.writer(f)
    w.writerow(["system", "T", "chi2", "dof", "p"])
    w.writerows(global_tests)

t15 = [r for r in rows if r["source"] == "Tamm 2015"]
sig = sum(abs(r["z"]) > 2 for r in t15)
print(f"\nTamm 2015: {sig}/{len(t15)} values exceed 2 sigma of the "
      f"single-snapshot null; median |z| = "
      f"{np.median([abs(r['z']) for r in t15]):.2f}")
for r in rows:
    if r["source"] != "Tamm 2015":
        print(f"  {r['source']:9s} {r['pair']:18s} alpha={r['alpha']:+.3f} "
              f"sd={r['null_sd']:.3f} z={r['z']:+.2f}")

# ------------------------------------------------------------------ figure
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9),
                         gridspec_kw={"width_ratios": [1.15, 1]})
ax = axes[0]
names, counts, table = T15["NiCrCoFe"]
cell, null = null_for("fcc", 3, counts)
K = 4
mu = null.mean("standard").reshape(K, K)
sd = np.sqrt(np.diag(null.cov("standard"))).reshape(K, K)
ax.axhspan(mu[0, 0] - 2 * sd[0, 0], mu[0, 0] + 2 * sd[0, 0], color="#eeeeee",
           zorder=0, label="random cell, ±2σ (like)")
ax.axhline(mu[0, 0], color=INK2, lw=0.8, zorder=0)
for i, sp in enumerate(names):
    ax.plot(TEMPS, table[(sp, sp)], "-" + MK[i], color=C[i], ms=5,
            label=f"{sp}–{sp}")
ax.set_xlabel("temperature (K)")
ax.set_ylabel(r"reported first-shell $\alpha$")
ax.set_xticks(TEMPS)
ax.set_title("NiCrCoFe, 108 atoms (Tamm 2015)", fontsize=8, color=INK2)
ax.legend(fontsize=7, ncol=2, loc="lower left")
ax.set_ylim(-0.6, 0.55)
panel_label(ax, "a")

ax = axes[1]
groups = [("NiCrCo 108", [r for r in t15 if r["system"] == "NiCrCo"], C[0]),
          ("NiCrCoFe 108", [r for r in t15 if r["system"] == "NiCrCoFe"],
           C[1])]
rng = np.random.default_rng(0)
for k, (lab, rr, col) in enumerate(groups):
    zs = np.array([r["z"] for r in rr])
    ax.plot(zs, k + rng.uniform(-0.15, 0.15, len(zs)), MK[k], color=col,
            mfc="none", mew=1.0, ms=5)
d18 = [r for r in rows if r["source"] == "Ding 2018"]
ax.plot([r["z"] for r in d18 if "18" not in r["pair"]], [2] * 3, MK[2],
        color=C[2], mfc="none", mew=1.0, ms=5)
ax.plot([r["z"] for r in d18 if "18" in r["pair"]], [3] * 3, MK[2],
        color=C[2], ms=5)
c20 = [r for r in rows if r["source"] == "Cao 2020"][0]
ax.plot([c20["z"]], [4], MK[3], color=C[3], ms=5)
ax.axvspan(-2, 2, color="#eeeeee", zorder=0)
ax.set_yticks(range(5))
ax.set_yticklabels(["Tamm: NiCrCo, 108", "Tamm: NiCrCoFe, 108",
                    "Ding: CrCoNi, 180 (1 cell)", "Ding: CrCoNi, 18×180",
                    "Cao: GeSn, 64 (1 snapshot)"], fontsize=7)
ax.set_xscale("symlog", linthresh=2)
ax.set_xlim(-35, 40)
ax.set_xticks([-20, -5, -2, 0, 2, 5, 20])
ax.set_xticklabels(["−20", "−5", "−2", "0", "2", "5", "20"])
ax.set_xlabel("z against random-cell null (symlog)")
ax.set_ylim(-0.6, 4.6)
ax.invert_yaxis()
panel_label(ax, "b")
fig.tight_layout()
save(fig, "fig5_literature")
