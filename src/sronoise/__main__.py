"""Command line: test a POSCAR for significant short-range order.

    python -m sronoise POSCAR [--shells 3] [--tol 0.1]
                              [--estimator unbiased] [--perm 0]
                              [--species Ni,Co,Cr]

Only the species listed with --species (default: all) are treated as
occupying the disordered sublattice; for alloys this is every species.
"""

import argparse

import numpy as np

from .lattice import from_positions, read_poscar
from .stats import permutation_pvalues, significance_test


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m sronoise",
                                 description=__doc__.split("\n\n")[0])
    ap.add_argument("poscar")
    ap.add_argument("--shells", type=int, default=3)
    ap.add_argument("--tol", type=float, default=0.1,
                    help="shell grouping tolerance in Angstrom")
    ap.add_argument("--estimator", default="unbiased",
                    choices=["unbiased", "standard"])
    ap.add_argument("--perm", type=int, default=0,
                    help="also compute permutation p-values with this many "
                         "random arrangements")
    ap.add_argument("--species", default=None,
                    help="comma-separated species on the mixed sublattice")
    args = ap.parse_args(argv)

    lat, pos, lab, names = read_poscar(args.poscar)
    if args.species:
        keep_names = args.species.split(",")
        keep = np.isin(np.array(names)[lab], keep_names)
        pos, lab = pos[keep], lab[keep]
        remap = {names.index(s): k for k, s in enumerate(keep_names)}
        lab = np.array([remap[x] for x in lab])
        names = keep_names
    cell = from_positions(pos, lat, args.shells, args.tol)
    print(f"{cell.n_sites} sites; shells at "
          + ", ".join(f"{r:.3f}" for r in cell.shell_radii)
          + " ; z = " + ", ".join(str(cell.coordination(s))
                                  for s in range(cell.n_shells)))
    shells = [s for s in range(cell.n_shells) if not cell.has_self_bonds(s)]
    res = significance_test(cell, lab, shells, args.estimator)
    print(res.table(names))
    if args.perm:
        p = permutation_pvalues(cell, lab, shells, args.estimator, args.perm)
        print("\npermutation p-values (two-sided):")
        for si, s in enumerate(shells):
            for a in range(len(names)):
                for b in range(a, len(names)):
                    print(f"shell {s + 1}  {names[a]}-{names[b]}  "
                          f"p={p[si, a, b]:.4g}")


if __name__ == "__main__":
    main()
