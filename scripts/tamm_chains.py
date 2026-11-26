"""Monte Carlo trajectories of Tamm et al. (DFT-MC, NiCrCo and NiCrCoFe, 108
atoms) analysed against the exact random-cell null.

Data: A. Tamm, "Short-range order structures of Ni-based FCC ternary and
quaternary alloys", DATADOI (2025), doi:10.23673/RE-518 (CC BY 4.0).  Download
the dataset and pass the path of 2015-MC-DFT_structures.tar.gz:

    python3 tamm_chains.py /path/to/2015-MC-DFT_structures.tar.gz

According to the dataset README these are mostly preliminary runs, not the
production runs of the 2015 paper, and the POSCAR files are the trial
structures of the Metropolis chain.  The accepted chain is reconstructed from
the site differences between consecutive trials: a trial that is followed by
a structure one exchange away from it was accepted.

Output: data/tamm_chains.csv, data/tamm_between_runs.csv,
figures/fig6_tamm_chains.{png,pdf}.
"""
import csv
import glob
import os
import re
import sys
import tarfile
import tempfile

import numpy as np
from scipy import stats

from _style import C, DATA, INK2, MK, panel_label, plt, save
from sronoise import WCNull, from_positions, warren_cowley


def read_poscar(path):
    L = open(path).read().split("\n")
    scale = float(L[1])
    lat = np.array([[float(x) for x in L[i].split()] for i in (2, 3, 4)]) * scale
    counts = [int(x) for x in L[6].split()]
    n = sum(counts)
    pos = np.array([[float(x) for x in L[8 + i].split()[:3]] for i in range(n)])
    if L[7].strip().lower().startswith("c"):
        pos = pos @ np.linalg.inv(lat)
    return lat, pos % 1.0, np.repeat(np.arange(len(counts)), counts)


def grid_keys(pos):
    # 108-site fcc cell = 3x3x3 conventional cubes: sites on a 1/6 grid
    g = np.round(pos * 6).astype(int) % 6
    assert np.abs(pos * 6 - np.round(pos * 6)).max() < 0.5
    return [tuple(k) for k in g]


def ham(a, b):
    return int((a != b).sum())


def accepted_chain(t):
    s, cur = [t[0]], t[0]
    n_acc = 0
    for k in range(1, len(t)):
        if ham(t[k], cur) and k + 1 < len(t):
            if ham(t[k + 1], t[k]) in (0, 2) and ham(t[k + 1], cur) != 2:
                cur = t[k]
                n_acc += 1
        s.append(cur)
    return np.array(s), n_acc


def tau_int(x):
    x = x - x.mean(0)
    v = (x * x).mean(0)
    v[v == 0] = np.inf
    t = np.full(x.shape[1], 0.5)
    alive = np.ones(x.shape[1], bool)
    for k in range(1, len(x) // 4):
        r = (x[:-k] * x[k:]).mean(0) / v
        alive &= r > 0.05
        if not alive.any():
            break
        t += np.where(alive, r, 0)
    return float(np.mean(t))


def global_test(alpha, null, scale=1.0):
    labs = null.labels()
    keep = np.array([a <= b for (_, a, b) in labs])
    d = alpha.reshape(-1)[keep] - null.mean("standard")[keep]
    w, V = np.linalg.eigh(null.cov("standard")[np.ix_(keep, keep)] / scale)
    g = w > w.max() * 1e-9
    q = float(np.sum((V[:, g].T @ d) ** 2 / w[g]))
    return q, int(g.sum()), float(stats.chi2.sf(q, int(g.sum())))


def main(tarpath):
    work = tempfile.mkdtemp()
    with tarfile.open(tarpath) as tf:
        tf.extractall(work, filter="data")
    root = glob.glob(os.path.join(work, "*", "MC-DFT"))[0]
    dirs = sorted(d for d in glob.glob(root + "/**/simulation_*", recursive=True)
                  if os.path.isdir(d + "/structures"))
    lat, pos, _ = read_poscar(sorted(glob.glob(dirs[0] + "/structures/POSCAR_*"))[0])
    keys = sorted(set(grid_keys(pos)))
    index = {k: i for i, k in enumerate(keys)}
    cell = from_positions(np.array(keys) / 6.0, lat, n_shells=1, cartesian=False)
    assert cell.n_sites == 108 and cell.coordination(0) == 12

    def config(path):
        lat, pos, spec = read_poscar(path)
        cfg = np.empty(108, dtype=np.int32)
        for k, s in zip(grid_keys(pos), spec):
            cfg[index[k]] = s
        return cfg

    rows, chains, nulls = [], {}, {}
    for d in dirs:
        alloy = "NiCrCo" if "/NiCrCo/" in d else "NiCrCoFe"
        T = int(re.search(r"/(\d{4})K/", d).group(1))
        sim = int(d[-1])
        t = np.array([config(f) for f in sorted(glob.glob(d + "/structures/POSCAR_*"))])
        K = int(t.max()) + 1
        if K not in nulls:
            nulls[K] = WCNull(cell, list(np.bincount(t[0], minlength=K)), [0])
        null = nulls[K]
        s, n_acc = accepted_chain(t)
        al = warren_cowley(cell, s, [0], K, "standard").reshape(len(s), K, K)
        iu = np.triu_indices(K)
        half = al[len(al) // 2:]
        tau = tau_int(half[:, iu[0], iu[1]])
        neff = max(1.0, len(half) / (2 * tau))
        p_chain = [global_test(a, null)[2] for a in al]
        chains[(alloy, T, sim)] = dict(alpha=al, p=np.array(p_chain), n_acc=n_acc, neff=neff)
        rows.append(dict(alloy=alloy, T=T, sim=sim, trials=len(t), accepted=n_acc,
                         accepted_per_atom=round(n_acc / 108, 2),
                         p_start=p_chain[0], p_final=p_chain[-1],
                         n_snapshots_second_half=len(half), tau=round(tau, 1),
                         n_eff=round(neff, 1),
                         p_second_half_neff=global_test(half.mean(0), null, neff)[2]))
        print(rows[-1], flush=True)

    with open(os.path.join(DATA, "tamm_chains.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # independent seeds (simulations 1 and 2) against the predicted scatter
    brows, ratios = [], []
    for alloy, K in (("NiCrCo", 3), ("NiCrCoFe", 4)):
        sd = np.sqrt(np.diag(nulls[K].cov("standard"))).reshape(K, K)
        iu = np.triu_indices(K)
        for T in (500, 800, 1200):
            a, b = chains[(alloy, T, 1)], chains[(alloy, T, 2)]
            ma = a["alpha"][len(a["alpha"]) // 2:].mean(0)[iu]
            mb = b["alpha"][len(b["alpha"]) // 2:].mean(0)[iu]
            pred = sd[iu] ** 2 * (1 / a["neff"] + 1 / b["neff"])
            ratios += list((ma - mb) ** 2 / pred)
            brows.append(dict(alloy=alloy, T=T, rms_difference=float(np.sqrt(np.mean((ma - mb) ** 2))),
                              rms_predicted=float(np.sqrt(np.mean(pred)))))
    with open(os.path.join(DATA, "tamm_between_runs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(brows[0]))
        w.writeheader()
        w.writerows(brows)
    print("between-run variance ratio (observed/predicted): %.2f over %d values"
          % (np.mean(ratios), len(ratios)))
    neffs = [r["n_eff"] for r in rows]
    print("n_eff range %.1f-%.1f, median %.1f" % (min(neffs), max(neffs), np.median(neffs)))

    # figure: Cr-Cr along the accepted chain and single-configuration p values
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    K = 3
    sd = np.sqrt(np.diag(nulls[K].cov("standard"))).reshape(K, K)
    mu = nulls[K].mean("standard").reshape(K, K)
    ax = axes[0]
    ax.axhspan(mu[1, 1] - 2 * sd[1, 1], mu[1, 1] + 2 * sd[1, 1], color="#ececec", zorder=0,
               label="random cell, $\\pm2\\sigma$")
    for i, T in enumerate((500, 800, 1200)):
        ch = chains[("NiCrCo", T, 1)]
        x = np.arange(len(ch["alpha"])) / 108.0
        ax.plot(x, ch["alpha"][:, 1, 1], color=C[i], lw=1.4, label=f"{T} K")
    ax.set_xlabel("Monte Carlo trials per atom")
    ax.set_ylabel("Cr–Cr $\\alpha$ (first shell)")
    panel_label(ax, "a")
    ax = axes[1]
    for i, T in enumerate((500, 800, 1200)):
        ch = chains[("NiCrCo", T, 1)]
        x = np.arange(len(ch["p"])) / 108.0
        ax.semilogy(x, np.clip(ch["p"], 1e-8, 1), color=C[i], lw=1.2, label=f"{T} K")
    ax.axhline(0.05, color=INK2, lw=0.8, ls="--")
    ax.text(1.01, 0.05, "0.05", color=INK2, fontsize=7, va="center", transform=ax.get_yaxis_transform())
    ax.set_xlabel("Monte Carlo trials per atom")
    ax.set_ylabel("$p$, single configuration")
    ax.set_ylim(1e-8, 2)
    panel_label(ax, "b")
    fig.tight_layout()
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=7, bbox_to_anchor=(0.5, 0.98))
    save(fig, "fig6_tamm_chains")


if __name__ == "__main__":
    main(sys.argv[1])
