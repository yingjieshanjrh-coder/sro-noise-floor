import itertools
from fractions import Fraction

import numpy as np
import pytest

from sronoise import (WCNull, build_supercell, pair_count_moments,
                      sigma_asymptotic, standard_bias, significance_test,
                      warren_cowley)


def all_arrangements(counts):
    """Every distinct arrangement of a multiset of species labels."""
    N = sum(counts)
    out = []

    def rec(pos_left, k, cfg):
        if k == len(counts):
            out.append(cfg.copy())
            return
        for chosen in itertools.combinations(pos_left, counts[k]):
            for p in chosen:
                cfg[p] = k
            rest = [p for p in pos_left if p not in chosen]
            rec(rest, k + 1, cfg)

    rec(list(range(N)), 0, np.zeros(N, dtype=np.int64))
    return np.array(out)


@pytest.mark.parametrize("structure,radii,z", [
    ("sc", [1.0, np.sqrt(2), np.sqrt(3)], [6, 12, 8]),
    ("bcc", [np.sqrt(3) / 2, 1.0, np.sqrt(2)], [8, 6, 12]),
    ("fcc", [np.sqrt(0.5), 1.0, np.sqrt(1.5)], [12, 6, 24]),
    ("hcp", [1.0, np.sqrt(2), np.sqrt(8 / 3)], [12, 6, 2]),
    ("diamond", [np.sqrt(3) / 4, np.sqrt(0.5), np.sqrt(11) / 4],
     [4, 12, 12]),
])
def test_shells(structure, radii, z):
    cell = build_supercell(structure, 3, 3)
    assert np.allclose(cell.shell_radii, radii)
    assert [cell.coordination(s) for s in range(3)] == z
    for s in range(3):
        nb = cell.neighbours[s]
        # symmetric multigraph: every bond is seen from both ends
        fwd = np.sort(np.arange(cell.n_sites).repeat(z[s]) * cell.n_sites
                      + nb.ravel())
        rev = np.sort(nb.ravel() * cell.n_sites
                      + np.arange(cell.n_sites).repeat(z[s]))
        assert np.array_equal(fwd, rev)


@pytest.mark.parametrize("structure,reps,counts", [
    ("diamond", 1, [3, 3, 2]),
    ("bcc", 2, [8, 8]),
    ("bcc", 2, [10, 3, 3]),
    ("hcp", (2, 1, 2), [6, 6, 4]),
])
def test_exact_against_enumeration(structure, reps, counts):
    cell = build_supercell(structure, reps, 3)
    shells = [s for s in range(3) if not cell.has_self_bonds(s)]
    X = all_arrangements(counts)
    null = WCNull(cell, counts, shells)
    for est in ("standard", "unbiased"):
        s1, s2 = 0.0, 0.0
        for start in range(0, len(X), 20000):
            A = warren_cowley(cell, X[start:start + 20000], shells,
                              len(counts), est).reshape(-1, len(null.labels()))
            s1 = s1 + A.sum(0)
            s2 = s2 + A.T @ A
        mean = s1 / len(X)
        C = s2 / len(X) - np.outer(mean, mean)
        assert np.allclose(mean, null.mean(est), atol=1e-12)
        assert np.allclose(C, null.cov(est), atol=1e-10)


def test_unbiased_mean_is_zero():
    cell = build_supercell("fcc", 3, 2)
    null = WCNull(cell, [50, 30, 20, 8])
    assert np.allclose(null.mean("unbiased"), 0.0, atol=1e-14)


def test_standard_bias_closed_form():
    cell = build_supercell("diamond", 2, 3)
    counts = [60, 4]
    null = WCNull(cell, counts)
    for s in range(3):
        for a in range(2):
            for b in range(2):
                assert null.exact_mean(s, a, b) == standard_bias(64, counts,
                                                                 a, b)
    # 4 Sn in a 64-atom cell: random Sn-Sn alpha is +60/252
    assert standard_bias(64, counts, 1, 1) == Fraction(60, 252)


@pytest.mark.parametrize("comp", [(0.5, 0.5), (0.2, 0.3, 0.5),
                                  (0.1, 0.2, 0.3, 0.4)])
@pytest.mark.parametrize("z", [4, 12])
def test_asymptotic_variance(comp, z):
    N = 200000
    counts = [int(round(c * N)) for c in comp]
    counts[-1] = N - sum(counts[:-1])
    S = [N * z]
    R = np.array([[N * z * z]])
    P = np.array([[N * z]])
    _, cov = pair_count_moments(counts, S, R, P)
    K = len(comp)
    for a in range(K):
        for b in range(a, K):
            k = Fraction(N * N, N * z * counts[a] * counts[b])
            v = float(k * k * cov[((0, a, b), (0, a, b))])
            ref = sigma_asymptotic(comp[a], comp[b], N * z / 2,
                                   like=(a == b)) ** 2
            assert v == pytest.approx(ref, rel=2e-4)


def test_global_dof_and_calibration():
    cell = build_supercell("bcc", 4, 2)
    counts = [43, 43, 42]
    null = WCNull(cell, counts)
    rng = np.random.default_rng(3)
    base = np.repeat(np.arange(3), counts)
    pvals = []
    for _ in range(300):
        cfg = rng.permutation(base)
        r = significance_test(cell, cfg, null=null)
        pvals.append(r.p_global)
    assert r.dof == 2 * 3  # two shells x K(K-1)/2
    # p-values of the global test are approximately uniform under the null
    assert 0.02 < np.mean(np.array(pvals) < 0.05) < 0.09


@pytest.mark.parametrize("structure,reps", [("fcc", 3), ("bcc", 4),
                                            ("hcp", (3, 2, 2)),
                                            ("diamond", 2)])
def test_from_positions_matches_builder(structure, reps):
    from sronoise import from_positions
    ref = build_supercell(structure, reps, 3)
    lat = np.diag(ref.box) * 3.6                     # Angstrom-like scale
    rng = np.random.default_rng(0)
    perm = rng.permutation(ref.n_sites)              # shuffled site order
    pos = ref.positions[perm] * 3.6
    pos += rng.normal(0, 0.01, pos.shape)            # small noise
    cell = from_positions(pos, lat, 3, tol=0.08)
    inv = np.argsort(perm)
    for s in range(3):
        a = np.sort(inv[ref.neighbours[s][perm]], axis=1)
        b = np.sort(cell.neighbours[s], axis=1)
        assert np.array_equal(a, b)
    assert np.array_equal(cell.graph_invariants()[2],
                          ref.graph_invariants()[2])


def test_poscar_cli(tmp_path, capsys):
    from sronoise.__main__ import main
    cell = build_supercell("fcc", 3, 1)
    rng = np.random.default_rng(1)
    lab = rng.permutation(np.repeat([0, 1, 2], 36))
    order = np.argsort(lab, kind="stable")
    a = 3.6
    with open(tmp_path / "POSCAR", "w") as f:
        f.write("test\n1.0\n")
        for v in np.diag(cell.box) * a:
            f.write(" ".join(f"{x:.6f}" for x in v) + "\n")
        f.write("Ni Co Cr\n36 36 36\nCartesian\n")
        for p in cell.positions[order] * a:
            f.write(" ".join(f"{x:.6f}" for x in p) + "\n")
    main([str(tmp_path / "POSCAR"), "--shells", "2"])
    out = capsys.readouterr().out
    assert "108 sites" in out and "Ni-Co" in out and "global chi2" in out
