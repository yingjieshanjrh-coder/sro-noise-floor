"""Periodic supercells and coordination-shell neighbour tables.

Neighbour tables are built by translating each site by the exact shell
vectors of the infinite crystal and wrapping back into the box.  In small
cells the same pair of sites can therefore be joined by more than one
lattice vector; such repeated bonds are kept, which is what a periodic
DFT or Monte Carlo code sees when it counts neighbours.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Conventional cells: (cell lengths in units of a, fractional basis).
_SQ3 = np.sqrt(3.0)
_IDEAL_COA = np.sqrt(8.0 / 3.0)

_STRUCTURES = {
    "sc": ((1.0, 1.0, 1.0), [(0, 0, 0)]),
    "bcc": ((1.0, 1.0, 1.0), [(0, 0, 0), (0.5, 0.5, 0.5)]),
    "fcc": ((1.0, 1.0, 1.0),
            [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]),
    "diamond": ((1.0, 1.0, 1.0),
                [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0),
                 (0.25, 0.25, 0.25), (0.25, 0.75, 0.75),
                 (0.75, 0.25, 0.75), (0.75, 0.75, 0.25)]),
    # orthohexagonal setting, a x sqrt(3)a x c
    "hcp": ((1.0, _SQ3, _IDEAL_COA),
            [(0, 0, 0), (0.5, 0.5, 0), (0.5, 1 / 6, 0.5), (0, 2 / 3, 0.5)]),
}

# All basis fractions above are multiples of 1/12.
_GRID = 12


@dataclass
class Supercell:
    """A periodic supercell with neighbour lists for several shells."""

    structure: str
    reps: tuple[int, int, int]
    positions: np.ndarray            # (N, 3) Cartesian, units of a
    box: np.ndarray                  # (3,) orthorhombic box lengths
    shell_radii: list[float]
    neighbours: list[np.ndarray]     # per shell, (N, z_s) site indices
    _cache: dict = field(default_factory=dict, repr=False)

    @property
    def n_sites(self) -> int:
        return len(self.positions)

    @property
    def n_shells(self) -> int:
        return len(self.neighbours)

    def coordination(self, shell: int) -> int:
        return self.neighbours[shell].shape[1]

    def has_self_bonds(self, shell: int) -> bool:
        nb = self.neighbours[shell]
        return bool(np.any(nb == np.arange(self.n_sites)[:, None]))

    def graph_invariants(self):
        """Return (S, R, P) used by the exact moment formulas.

        S[s]    = sum_ij A^s_ij              (= N z_s)
        R[s,t]  = sum_i d^s_i d^t_i          (= N z_s z_t)
        P[s,t]  = sum_ij A^s_ij A^t_ij       (bond multiplicities shared
                                              between shells s and t)
        """
        if "inv" in self._cache:
            return self._cache["inv"]
        n = self.n_sites
        ns = self.n_shells
        mats = []
        for nb in self.neighbours:
            rows = np.repeat(np.arange(n), nb.shape[1])
            key = rows * n + nb.ravel()
            uniq, cnt = np.unique(key, return_counts=True)
            mats.append(dict(zip(uniq.tolist(), cnt.tolist())))
        deg = np.array([[nb.shape[1]] * n for nb in self.neighbours])
        S = np.array([d.sum() for d in deg], dtype=np.int64)
        R = np.array([[int(np.dot(deg[s], deg[t])) for t in range(ns)]
                      for s in range(ns)], dtype=np.int64)
        P = np.zeros((ns, ns), dtype=np.int64)
        for s in range(ns):
            for t in range(s, ns):
                ms, mt = mats[s], mats[t]
                if len(mt) < len(ms):
                    ms, mt = mt, ms
                val = sum(c * mt.get(k, 0) for k, c in ms.items())
                P[s, t] = P[t, s] = val
        self._cache["inv"] = (S, R, P)
        return S, R, P


def _shell_vectors(structure: str, n_shells: int, tol: float = 1e-6):
    """Exact neighbour vectors for each basis site, grouped into shells."""
    cell, basis = _STRUCTURES[structure]
    cell = np.asarray(cell)
    basis = np.asarray(basis, dtype=float)
    m = 3 + n_shells
    rng = np.arange(-m, m + 1)
    shifts = np.array(np.meshgrid(rng, rng, rng, indexing="ij")).reshape(3, -1).T
    cluster = (shifts[:, None, :] + basis[None, :, :]).reshape(-1, 3) * cell
    out = []
    radii = None
    for b in basis:
        d = cluster - b * cell
        r = np.linalg.norm(d, axis=1)
        keep = r > tol
        d, r = d[keep], r[keep]
        uniq = []
        for x in np.sort(r):
            if not uniq or x - uniq[-1] > tol:
                uniq.append(x)
            if len(uniq) > n_shells:
                break
        uniq = uniq[:n_shells]
        if radii is None:
            radii = uniq
        elif not np.allclose(radii, uniq, atol=1e-6):
            raise RuntimeError("basis sites have different shell radii")
        out.append([d[np.abs(r - rad) < tol] for rad in uniq])
    return radii, out


def build_supercell(structure: str, reps, n_shells: int = 3) -> Supercell:
    """Build an orthorhombic supercell of `reps` conventional cells.

    Parameters
    ----------
    structure : one of 'sc', 'bcc', 'fcc', 'hcp', 'diamond'
    reps : int or (nx, ny, nz)
    n_shells : number of coordination shells to tabulate
    """
    if structure not in _STRUCTURES:
        raise ValueError(f"unknown structure {structure!r}")
    if np.isscalar(reps):
        reps = (int(reps),) * 3
    reps = tuple(int(r) for r in reps)
    cell, basis = _STRUCTURES[structure]
    cell = np.asarray(cell)
    basis = np.asarray(basis, dtype=float)
    nb_ = len(basis)

    idx = np.array(np.meshgrid(*[np.arange(r) for r in reps],
                               indexing="ij")).reshape(3, -1).T
    frac = (idx[:, None, :] + basis[None, :, :]).reshape(-1, 3)   # conv. units
    btype = np.tile(np.arange(nb_), len(idx))
    positions = frac * cell
    box = np.asarray(reps) * cell
    n = len(frac)

    reps_arr = np.asarray(reps)

    def key_of(fr):
        g = np.rint(fr * _GRID).astype(np.int64)
        g %= reps_arr * _GRID
        return (g[..., 0] * (reps_arr[1] * _GRID) + g[..., 1]) \
            * (reps_arr[2] * _GRID) + g[..., 2]

    keys = key_of(frac)
    order = np.argsort(keys)
    sorted_keys = keys[order]
    if np.any(np.diff(sorted_keys) == 0):
        raise RuntimeError("duplicate sites in supercell")

    radii, vecs = _shell_vectors(structure, n_shells)
    neighbours = []
    for s in range(n_shells):
        z = len(vecs[0][s])
        nb = np.empty((n, z), dtype=np.int64)
        for b in range(nb_):
            sel = np.where(btype == b)[0]
            dv = vecs[b][s] / cell                     # conv. fractional
            if len(dv) != z:
                raise RuntimeError("inequivalent coordination numbers")
            tgt = frac[sel][:, None, :] + dv[None, :, :]
            k = key_of(tgt)
            pos = np.searchsorted(sorted_keys, k)
            pos = np.clip(pos, 0, n - 1)
            if not np.all(sorted_keys[pos] == k):
                raise RuntimeError("neighbour lookup failed")
            nb[sel] = order[pos]
        neighbours.append(nb)
    return Supercell(structure, reps, positions, box, list(radii), neighbours)
