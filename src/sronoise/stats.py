"""Significance tests and closed-form noise-floor estimates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats as _st

from .counting import random_configs, warren_cowley
from .moments import WCNull


# ---------------------------------------------------------------------------
# Leading-order closed forms (large N, fixed composition)
# ---------------------------------------------------------------------------

def sigma_asymptotic(c_a: float, c_b: float, n_bonds: float,
                     like: bool | None = None) -> float:
    """Leading-order null standard deviation of the Warren-Cowley parameter.

    With B = N z / 2 bonds in the shell,

        Var(alpha_aa) = (1 - c_a)^2 / (B c_a^2)
        Var(alpha_ab) = [(1 - c_a)(1 - c_b) + c_a c_b] / (2 B c_a c_b)

    Corrections are O(1/N^2).  Note that the lattice enters only through
    the number of bonds.
    """
    if like is None:
        like = c_a == c_b
    if like:
        return float(np.sqrt((1 - c_a) ** 2 / (n_bonds * c_a ** 2)))
    v = ((1 - c_a) * (1 - c_b) + c_a * c_b) / (2 * n_bonds * c_a * c_b)
    return float(np.sqrt(v))


def bias_standard(N: int, c_a: float, like: bool) -> float:
    """Exact null mean of the standard estimator for a zero-diagonal graph."""
    if not like:
        return -1.0 / (N - 1)
    return (1 - c_a) / (c_a * (N - 1)) if c_a > 0 else np.inf


def min_sites(alpha: float, c_a: float, c_b: float, z: int,
              k_sigma: float = 2.0, like: bool | None = None) -> float:
    """Smallest N for which |alpha| exceeds k_sigma null standard deviations
    (leading order, unbiased estimator)."""
    s1 = sigma_asymptotic(c_a, c_b, 1.0, like)        # sigma at B = 1
    B = (k_sigma * s1 / abs(alpha)) ** 2
    return 2.0 * B / z


# ---------------------------------------------------------------------------
# Tests for a single configuration (or an average over configurations)
# ---------------------------------------------------------------------------

@dataclass
class SignificanceResult:
    labels: list
    alpha: np.ndarray
    null_mean: np.ndarray
    null_std: np.ndarray
    z: np.ndarray
    p_two_sided: np.ndarray
    chi2: float
    dof: int
    p_global: float

    def table(self, species=None):
        rows = []
        for (s, a, b), al, m, sd, zz, p in zip(self.labels, self.alpha,
                                               self.null_mean, self.null_std,
                                               self.z, self.p_two_sided):
            if a > b:
                continue
            na = species[a] if species else str(a)
            nb = species[b] if species else str(b)
            rows.append(f"shell {s + 1}  {na}-{nb}  alpha={al:+.4f}  "
                        f"null={m:+.4f}+-{sd:.4f}  z={zz:+.2f}  p={p:.3g}")
        rows.append(f"global chi2={self.chi2:.2f}  dof={self.dof}  "
                    f"p={self.p_global:.3g}")
        return "\n".join(rows)


def _rank_tol(C):
    w = np.linalg.eigvalsh(C)
    return w.max() * 1e-9


def significance_test(cell, config, shells=None, estimator="unbiased",
                       n_average: int = 1, null: WCNull | None = None):
    """Test a configuration against the random-solution null.

    Parameters
    ----------
    cell, config : supercell and (N,) labels, or (n, N) labels whose
        Warren-Cowley parameters are averaged (n independent snapshots).
    n_average : number of statistically independent configurations that
        were averaged to obtain alpha (the null variance is divided by it).
        Leave at 1 when `config` is 2-D: it is set automatically.
    """
    config = np.asarray(config)
    K = int(config.max()) + 1
    if shells is None:
        shells = list(range(cell.n_shells))
    first = config if config.ndim == 1 else config[0]
    counts = np.bincount(first, minlength=K)
    if null is None:
        null = WCNull(cell, counts, shells)
    al = warren_cowley(cell, config, shells, K, estimator)
    if config.ndim == 2:
        n_average = config.shape[0]
        al = al.mean(axis=0)
    al = al.reshape(-1)
    mu = null.mean(estimator)
    C = null.cov(estimator) / n_average
    sd = np.sqrt(np.diag(C))
    z = (al - mu) / sd
    p = 2 * _st.norm.sf(np.abs(z))

    # global test on the independent entries (a <= b), pseudo-inverse
    labels = null.labels()
    keep = np.array([a <= b for (_, a, b) in labels]) & np.isfinite(mu)
    d = (al - mu)[keep]
    Ck = C[np.ix_(keep, keep)]
    w, V = np.linalg.eigh(Ck)
    good = w > _rank_tol(Ck)
    proj = V[:, good].T @ d
    chi2 = float(np.sum(proj ** 2 / w[good]))
    dof = int(good.sum())
    return SignificanceResult(labels, al, mu, sd, z, p, chi2, dof,
                      float(_st.chi2.sf(chi2, dof)))


def permutation_pvalues(cell, config, shells=None, estimator="unbiased",
                        n_perm: int = 20000, rng=None, batch: int = 2000):
    """Two-sided permutation p-values for each Warren-Cowley parameter.

    Exact up to Monte Carlo error; recommended when the expected number of
    a-b bonds is small (dilute species, small cells), where the normal
    approximation fails.  The statistic is |alpha - E0[alpha]|.
    """
    config = np.asarray(config)
    K = int(config.max()) + 1
    if shells is None:
        shells = list(range(cell.n_shells))
    counts = np.bincount(config, minlength=K)
    null = WCNull(cell, counts, shells)
    mu = null.mean(estimator).reshape(len(shells), K, K)
    obs = np.abs(warren_cowley(cell, config, shells, K, estimator) - mu)
    rng = np.random.default_rng(rng)
    ge = np.zeros_like(obs)
    done = 0
    while done < n_perm:
        m = min(batch, n_perm - done)
        X = random_configs(counts, m, rng)
        A = np.abs(warren_cowley(cell, X, shells, K, estimator) - mu)
        ge += (A >= obs - 1e-12).sum(axis=0)
        done += m
    return (ge + 1) / (n_perm + 1)
