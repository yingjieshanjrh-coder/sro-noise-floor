"""False-positive rate of the global test when Warren-Cowley parameters are
averaged over correlated snapshots of one random-swap chain.

A 108-site equiatomic quinary fcc cell is evolved by random swaps of unlike
atoms (infinite temperature, so the stationary state is the random null).
Snapshots are taken every m swaps and n of them are averaged.  The averaged
parameters are tested with the null covariance divided either by n (naive)
or by n_eff = max(1, n / (2 tau)), with tau the integrated autocorrelation time of
the parameters measured on an independent long chain."""
import csv, os
import numpy as np
from scipy import stats
from sronoise import build_supercell, WCNull, warren_cowley

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
K = 5
cell = build_supercell("fcc", 3, 1)
N = cell.n_sites
counts = [22, 22, 22, 21, 21]
base = np.repeat(np.arange(K), counts)
null = WCNull(cell, counts, [0])
labs = null.labels()
keep = np.array([a <= b for (_, a, b) in labs])
C = null.cov("unbiased")[np.ix_(keep, keep)]
w, V = np.linalg.eigh(C)
good = w > w.max() * 1e-9
W, Vg, dof = w[good], V[:, good], int(good.sum())


def swap(cfg, rng, m):
    for _ in range(m):
        while True:
            i, j = rng.integers(N, size=2)
            if cfg[i] != cfg[j]:
                cfg[i], cfg[j] = cfg[j], cfg[i]
                break


def alpha(cfg):
    return warren_cowley(cell, cfg, [0], K, "unbiased").reshape(-1)[keep]


def tau_int(x, cmax=400):
    x = x - x.mean(0)
    v = (x * x).mean(0)
    t = np.full(x.shape[1], 0.5)
    for k in range(1, cmax):
        r = (x[:-k] * x[k:]).mean(0) / v
        if np.all(r < 0.02):
            break
        t += np.where(r > 0, r, 0)
    return float(np.mean(t))


def run(m, n, reps, rng, tau):
    naive = corr = 0
    for _ in range(reps):
        cfg = rng.permutation(base)
        acc = []
        for k in range(n):
            swap(cfg, rng, m)
            acc.append(alpha(cfg))
        d = np.mean(acc, 0)
        q = np.sum((Vg.T @ d) ** 2 / W)
        naive += stats.chi2.sf(q * n, dof) < 0.05
        neff = max(1.0, min(n, n / (2 * tau)))
        corr += stats.chi2.sf(q * neff, dof) < 0.05
    return naive / reps, corr / reps


if __name__ == "__main__":
    rng = np.random.default_rng(11)
    rows = []
    for m in (2, 5, 20, 100):
        cfg = rng.permutation(base)
        chain = []
        for k in range(20000):
            swap(cfg, rng, m)
            chain.append(alpha(cfg))
        tau = tau_int(np.array(chain))
        for n in (10, 50):
            fn, fc = run(m, n, 1000, rng, tau)
            rows.append(dict(swaps_between=m, swaps_per_atom_total=round(m * n / N, 2),
                             n_snapshots=n, tau=round(tau, 2),
                             n_eff=round(max(1.0, min(n, n / (2 * tau))), 2),
                             false_pos_naive=fn, false_pos_neff=fc))
            print(rows[-1], flush=True)
    with open(os.path.join(DATA, "snapshot_averaging.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader(); wr.writerows(rows)
