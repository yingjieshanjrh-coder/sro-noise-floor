"""Fig. 4: how often a single small-cell snapshot detects short-range order
that is really there.  Equiatomic quinary fcc lattice gas with fixed
first-shell pair interactions, sampled by swap Monte Carlo."""
import csv
import os
from multiprocessing import Pool

import numpy as np

from _style import C, DATA, INK2, MK, panel_label, plt, save
from sronoise import WCNull, build_supercell, significance_test, warren_cowley
from sronoise.mc import SwapMC

K = 5
J = 0.025                                   # eV
W = np.random.default_rng(1).normal(0, 1, (K, K))
W = (W + W.T) / 2
np.fill_diagonal(W, 0.0)
V = (J * W)[None]                           # first shell only
TEMPS = [6000, 4000, 3000, 2000, 1500, 1200, 1000]
SIZES = [2, 3, 4, 5, 6, 8]                  # fcc reps -> N = 4 r^3
N_RUNS = 300
EQ_SWEEPS = 400


def equiatomic(N):
    cnt = [N // K] * K
    for i in range(N - sum(cnt)):
        cnt[i] += 1
    return np.repeat(np.arange(K), cnt).astype(np.int32)


def truth(T):
    cell = build_supercell("fcc", 16, 1)
    mc = SwapMC(cell, V, [0])
    rng = np.random.default_rng(int(T))
    cfg = np.ascontiguousarray(rng.permutation(equiatomic(cell.n_sites)))
    mc.run(cfg, T, 2000, int(T) * 7 + 1)
    acc = []
    for k in range(200):
        mc.run(cfg, T, 20, int(T) * 1000 + k)
        acc.append(warren_cowley(cell, cfg, [0], K, "unbiased")[0])
    acc = np.array(acc)
    return T, acc.mean(0), acc.std(0) / np.sqrt(len(acc))


def power_job(args):
    reps, T = args
    cell = build_supercell("fcc", reps, 1)
    base = equiatomic(cell.n_sites)
    counts = np.bincount(base, minlength=K)
    null = WCNull(cell, counts, [0])
    mc = SwapMC(cell, V, [0])
    rng = np.random.default_rng(reps * 100003 + (int(T) if T else 0))
    out = []
    for run in range(N_RUNS):
        cfg = np.ascontiguousarray(rng.permutation(base))
        if T is not None:
            mc.run(cfg, T, EQ_SWEEPS, int(rng.integers(2 ** 62)))
        r = significance_test(cell, cfg, [0], "unbiased", null=null)
        al = r.alpha.reshape(K, K)
        pmin = r.p_two_sided.reshape(K, K)[np.triu_indices(K)].min()
        out.append((r.p_global, pmin, al))
    return reps, T, cell.n_sites, out


def compute():
    with Pool(2) as pool:
        tru = {T: (m, e) for T, m, e in pool.map(truth, TEMPS)}
        print("truth done")
        jobs = [(r, T) for r in SIZES for T in TEMPS + [None]]
        res = pool.map(power_job, jobs)

    iu = np.triu_indices(K)
    rows = []
    for reps, T, N, out in res:
        pg = np.array([o[0] for o in out])
        pm = np.array([o[1] for o in out])
        al = np.array([o[2][iu] for o in out])
        row = dict(N=N, T=T if T is not None else "random",
                   power_global=np.mean(pg < 0.05),
                   power_bonferroni=np.mean(pm < 0.05 / len(iu[0])))
        if T is not None:
            t = tru[T][0][iu]
            big = np.abs(t) > 0.02
            row["true_max_abs_alpha"] = float(np.abs(t).max())
            row["sign_agreement"] = float(np.mean(np.sign(al[:, big])
                                                  == np.sign(t[big])))
            row["rmse_alpha"] = float(np.sqrt(np.mean((al - t) ** 2)))
        rows.append(row)
        print(row)

    keys = ["N", "T", "true_max_abs_alpha", "power_global",
            "power_bonferroni", "sign_agreement", "rmse_alpha"]
    with open(os.path.join(DATA, "power_fcc_quinary.csv"), "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    with open(os.path.join(DATA, "truth_fcc_quinary.csv"), "w",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["T", "a", "b", "alpha_true", "stderr"])
        for T in TEMPS:
            m, e = tru[T]
            for a, b in zip(*iu):
                w.writerow([T, a, b, m[a, b], e[a, b]])



def plot():
    rows = []
    with open(os.path.join(DATA, "power_fcc_quinary.csv")) as f:
        for r in csv.DictReader(f):
            d = {"N": int(r["N"]), "T": r["T"] if r["T"] == "random"
                 else float(r["T"]),
                 "power_global": float(r["power_global"])}
            for k in ("true_max_abs_alpha", "sign_agreement"):
                d[k] = float(r[k]) if r[k] not in ("", "nan") else np.nan
            rows.append(d)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    for i, reps in enumerate(SIZES):
        N = 4 * reps ** 3
        sel = sorted([r for r in rows if r["N"] == N and r["T"] != "random"],
                     key=lambda r: r["true_max_abs_alpha"])
        x = [r["true_max_abs_alpha"] for r in sel]
        axes[0].plot(x, [r["power_global"] for r in sel], "-" + MK[i],
                     color=C[i], ms=4, label=f"N = {N}")
        axes[1].plot(x, [r["sign_agreement"] for r in sel], "-" + MK[i],
                     color=C[i], ms=4, label=f"N = {N}")
    axes[0].axhline(0.05, color=INK2, lw=0.8, zorder=0)
    axes[0].set_ylabel("detection rate, one snapshot\n(global test, 5% level)")
    panel_label(axes[0], "a")
    axes[1].axhline(0.5, color=INK2, lw=0.8, zorder=0)
    axes[1].set_ylabel("fraction of pairs with\ncorrect sign of α")
    axes[1].set_ylim(0.45, 1.02)
    panel_label(axes[1], "b")
    for ax in axes:
        ax.set_xlabel(r"largest true $|\alpha|$ (first shell)")
        ax.set_xlim(0.015, 0.175)
    fig.tight_layout()
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, fontsize=7,
               bbox_to_anchor=(0.5, 0.98))
    save(fig, "fig4_power")


if __name__ == "__main__":
    import sys
    if "--plot-only" not in sys.argv:
        compute()
    plot()
