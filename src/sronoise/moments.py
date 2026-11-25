"""Exact moments of pair counts and Warren-Cowley parameters.

Null model
----------
A supercell of N sites holds n_a atoms of species a (sum_a n_a = N).  Under
the null hypothesis every arrangement of these atoms is equally likely,
i.e. the species labels are a uniformly random permutation.  This is the
ideal random solid solution *at the composition actually used*, which is
the reference against which a measured Warren-Cowley parameter is
compared.

Pair counts
-----------
For shell s with (symmetric, zero-diagonal, possibly weighted) adjacency
A^s, the ordered pair count is

    M^s_ab = sum_ij A^s_ij x_i^a x_j^b ,      x_i^a = [site i holds a].

Its first two moments depend on the lattice only through three graph
invariants (see `Supercell.graph_invariants`):

    S_s   = sum_ij A^s_ij
    R_st  = sum_i d^s_i d^t_i
    P_st  = sum_ij A^s_ij A^t_ij

and on the composition only through falling factorials of the n_a.
All arithmetic is carried out in rational numbers, so the results are
exact.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import product

import numpy as np


def _ff(n: int, m: int) -> int:
    """Falling factorial n (n-1) ... (n-m+1)."""
    out = 1
    for k in range(m):
        out *= n - k
    return out


class _Joint:
    """E[x_{i1}^{a1} ... x_{im}^{am}] for distinct sites (multivariate
    hypergeometric)."""

    def __init__(self, counts):
        self.n = [int(c) for c in counts]
        self.N = sum(self.n)
        self._memo = {}

    def __call__(self, *labels) -> Fraction:
        key = tuple(sorted(labels))
        if key not in self._memo:
            m = len(labels)
            num = 1
            for a in set(labels):
                num *= _ff(self.n[a], labels.count(a))
            self._memo[key] = Fraction(num, _ff(self.N, m))
        return self._memo[key]


def pair_count_moments(counts, S, R, P):
    """Exact mean and covariance of all ordered pair counts M^s_ab.

    Parameters
    ----------
    counts : sequence of int, length K
    S, R, P : graph invariants for n_shells shells

    Returns
    -------
    mean : dict (s, a, b) -> Fraction
    cov  : dict ((s, a, b), (t, c, d)) -> Fraction
    """
    K = len(counts)
    F = _Joint(counts)
    ns = len(S)
    mean = {}
    for s, a, b in product(range(ns), range(K), range(K)):
        mean[(s, a, b)] = int(S[s]) * F(a, b)

    cov = {}
    keys = list(mean)
    for i, (s, a, b) in enumerate(keys):
        for (t, c, d) in keys[i:]:
            Sst = int(S[s]) * int(S[t])
            r, p = int(R[s, t]), int(P[s, t])
            e = (Sst - 4 * r + 2 * p) * F(a, b, c, d)
            three = 0
            if a == c:
                three += F(a, b, d)
            if a == d:
                three += F(a, b, c)
            if b == c:
                three += F(a, b, d)
            if b == d:
                three += F(a, b, c)
            e += (r - p) * three
            two = (int(a == c and b == d) + int(a == d and b == c))
            if two:
                e += p * two * F(a, b)
            v = e - mean[(s, a, b)] * mean[(t, c, d)]
            cov[((s, a, b), (t, c, d))] = v
            cov[((t, c, d), (s, a, b))] = v
    return mean, cov


class WCNull:
    """Exact null distribution moments of Warren-Cowley parameters.

    Two estimators are provided.

    * ``'standard'``
      alpha_ab = 1 - P(b|a) / c_b  with P(b|a) = M_ab / (z n_a) and
      c_b = n_b / N.  This is the textbook definition used by most codes.
    * ``'unbiased'``  alpha'_ab = 1 - M_ab (N-1) / (z n_a (n_b - delta_ab)),
      i.e. the reference probability is the one for drawing a neighbour
      *without replacement*.  Its null mean is exactly zero.
    """

    def __init__(self, cell, counts, shells=None):
        counts = [int(c) for c in counts]
        if sum(counts) != cell.n_sites:
            raise ValueError("counts do not sum to the number of sites")
        if min(counts) < 1:
            raise ValueError("every species must be present")
        if shells is None:
            shells = list(range(cell.n_shells))
        for s in shells:
            if cell.has_self_bonds(s):
                raise ValueError(f"shell {s} contains self bonds; the cell is "
                                 "too small for this shell")
        S, R, P = cell.graph_invariants()
        sh = np.asarray(shells)
        self.S, self.R, self.P = S[sh], R[np.ix_(sh, sh)], P[np.ix_(sh, sh)]
        self.shells = list(shells)
        self.counts = counts
        self.N = cell.n_sites
        self.K = len(counts)
        self._mean, self._cov = pair_count_moments(counts, self.S, self.R,
                                                   self.P)
        self._labels = [(s, a, b) for s in range(len(shells))
                        for a in range(self.K) for b in range(self.K)]

    # --- linear map M -> alpha ------------------------------------------------
    def _coef(self, s, a, b, estimator):
        N, n = self.N, self.counts
        S = int(self.S[s])
        if estimator == "standard":
            # P(b|a)/c_b = M_ab N^2 / (S n_a n_b)   (z = S/N)
            return Fraction(N * N, S * n[a] * n[b])
        if estimator == "unbiased":
            denom = S * n[a] * (n[b] - (a == b))
            if denom == 0:
                return None
            return Fraction(N * (N - 1), denom)
        raise ValueError(estimator)

    def labels(self):
        """List of (shell_index, a, b) in the order used by mean()/cov()."""
        return [(self.shells[s], a, b) for (s, a, b) in self._labels]

    def mean(self, estimator="standard"):
        out = []
        for (s, a, b) in self._labels:
            k = self._coef(s, a, b, estimator)
            out.append(np.nan if k is None else
                       float(1 - k * self._mean[(s, a, b)]))
        return np.array(out)

    def cov(self, estimator="standard"):
        L = self._labels
        n = len(L)
        out = np.full((n, n), np.nan)
        coef = [self._coef(*l, estimator) for l in L]
        for i in range(n):
            if coef[i] is None:
                continue
            for j in range(i, n):
                if coef[j] is None:
                    continue
                v = float(coef[i] * coef[j] * self._cov[(L[i], L[j])])
                out[i, j] = out[j, i] = v
        return out

    def exact_mean(self, s, a, b, estimator="standard") -> Fraction:
        k = self._coef(s, a, b, estimator)
        return 1 - k * self._mean[(s, a, b)]

    def exact_var(self, s, a, b, estimator="standard") -> Fraction:
        k = self._coef(s, a, b, estimator)
        return k * k * self._cov[((s, a, b), (s, a, b))]


def standard_bias(N: int, counts, a: int, b: int) -> Fraction:
    """Closed form null mean of the standard estimator (zero-diagonal graph).

    E[alpha_ab] = -1/(N-1)                       for a != b
    E[alpha_aa] = (N - n_a) / (n_a (N - 1))      for a == b

    Independent of the lattice and the shell.
    """
    if a != b:
        return Fraction(-1, N - 1)
    na = int(counts[a])
    return Fraction(N - na, na * (N - 1))
