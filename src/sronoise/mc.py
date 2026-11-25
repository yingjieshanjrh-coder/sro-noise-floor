"""Swap Monte Carlo on a Supercell (used for power analysis).

The C kernel is compiled with the system C compiler on first use and
cached next to this module (or in $SRONOISE_CACHE).
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess

import numpy as np

KB_EV = 8.617333262e-5

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "_kawasaki.c")
_lib = None


def _load():
    global _lib
    if _lib is not None:
        return _lib
    with open(_SRC, "rb") as f:
        tag = hashlib.sha1(f.read()).hexdigest()[:10]
    cache = os.environ.get("SRONOISE_CACHE", _HERE)
    so = os.path.join(cache, f"_kawasaki_{tag}.so")
    if not os.path.exists(so):
        cc = os.environ.get("CC", "cc")
        subprocess.check_call([cc, "-O3", "-shared", "-fPIC", _SRC, "-o",
                               so, "-lm"])
    lib = ctypes.CDLL(so)
    ip = np.ctypeslib.ndpointer(np.int32, flags="C_CONTIGUOUS")
    dp = np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS")
    lib.swap_mc.restype = ctypes.c_double
    lib.swap_mc.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ip, ip,
                            ip, dp, ctypes.c_double, ip, ctypes.c_long,
                            ctypes.c_uint64]
    _lib = lib
    return lib


class SwapMC:
    """Canonical swap Monte Carlo with pair interactions V[s, a, b] (eV)."""

    def __init__(self, cell, V, shells=None):
        V = np.asarray(V, dtype=np.float64)
        if shells is None:
            shells = list(range(V.shape[0]))
        if V.shape[0] != len(shells):
            raise ValueError("one interaction matrix per shell")
        if not np.allclose(V, np.swapaxes(V, 1, 2)):
            raise ValueError("interaction matrices must be symmetric")
        self.cell = cell
        self.K = V.shape[1]
        self.V = np.ascontiguousarray(V)
        tabs = [np.ascontiguousarray(cell.neighbours[s], dtype=np.int32)
                for s in shells]
        self.z = np.array([t.shape[1] for t in tabs], dtype=np.int32)
        self.off = np.concatenate([[0], np.cumsum([t.size for t in tabs])[:-1]
                                   ]).astype(np.int32)
        self.nbr = np.ascontiguousarray(np.concatenate([t.ravel()
                                                        for t in tabs]))
        self.lib = _load()

    def run(self, config, T, n_sweeps, seed):
        """Advance `config` (modified in place, int32) by n_sweeps at T (K)."""
        if config.dtype != np.int32 or not config.flags.c_contiguous:
            raise TypeError("config must be a contiguous int32 array")
        if config.shape != (self.cell.n_sites,):
            raise ValueError("config length does not match the cell")
        if config.min() < 0 or config.max() >= self.K:
            raise ValueError("species labels out of range")
        beta = 1.0 / (KB_EV * T)
        return self.lib.swap_mc(self.cell.n_sites, self.K, len(self.z),
                                self.z, self.off, self.nbr, self.V, beta,
                                config, int(n_sweeps), int(seed) & (2**64 - 1))

    def energy(self, config):
        e = 0.0
        for si in range(len(self.z)):
            nb = self.nbr[self.off[si]:self.off[si] + self.cell.n_sites
                          * self.z[si]].reshape(self.cell.n_sites, -1)
            e += self.V[si][config[:, None], config[nb]].sum()
        return 0.5 * e
