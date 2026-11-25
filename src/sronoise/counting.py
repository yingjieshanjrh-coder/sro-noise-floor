"""Pair counting and Warren-Cowley estimators for configurations."""

from __future__ import annotations

import numpy as np


def pair_counts(cell, config, shells=None, n_species=None):
    """Ordered pair counts M[..., s, a, b].

    Parameters
    ----------
    cell : Supercell
    config : (N,) or (n_samples, N) integer species labels 0..K-1
    shells : shell indices (default: all)
    n_species : K (default: max label + 1)
    """
    config = np.asarray(config)
    single = config.ndim == 1
    if single:
        config = config[None, :]
    if shells is None:
        shells = list(range(cell.n_shells))
    K = int(config.max()) + 1 if n_species is None else int(n_species)
    ns = config.shape[0]
    out = np.zeros((ns, len(shells), K, K), dtype=np.int64)
    offs = (np.arange(ns) * K * K)[:, None, None]
    for si, s in enumerate(shells):
        nb = cell.neighbours[s]
        a = config[:, :, None]
        b = config[:, nb]
        idx = (offs + a * K + b).ravel()
        out[:, si] = np.bincount(idx, minlength=ns * K * K).reshape(ns, K, K)
    return out[0] if single else out


def warren_cowley(cell, config, shells=None, n_species=None,
                  estimator="standard"):
    """Warren-Cowley parameters alpha[..., s, a, b].

    'standard' : alpha_ab = 1 - P(b|a) / c_b
    'unbiased' : alpha_ab = 1 - P(b|a) / ((n_b - delta_ab) / (N - 1))
    """
    config = np.asarray(config)
    if shells is None:
        shells = list(range(cell.n_shells))
    M = pair_counts(cell, config, shells, n_species)
    K = M.shape[-1]
    N = cell.n_sites
    lab = config if config.ndim == 1 else config[0]
    n = np.bincount(lab, minlength=K).astype(float)
    z = np.array([cell.coordination(s) for s in shells], dtype=float)
    p_cond = M / (z[:, None, None] * n[:, None])      # P(b|a)
    if estimator == "standard":
        ref = n / N
        ref = np.broadcast_to(ref[None, :], (K, K))
    elif estimator == "unbiased":
        ref = (n[None, :] - np.eye(K)) / (N - 1)
    else:
        raise ValueError(estimator)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 1.0 - p_cond / ref


def random_configs(counts, n_samples, rng=None):
    """Uniformly random arrangements with fixed species counts."""
    rng = np.random.default_rng(rng)
    base = np.repeat(np.arange(len(counts)), counts)
    return rng.permuted(np.tile(base, (n_samples, 1)), axis=1)
