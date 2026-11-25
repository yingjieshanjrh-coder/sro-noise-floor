"""Write example POSCARs: a random and a Monte Carlo ordered 108-atom
CrMnFeCoNi cell (ideal fcc positions, a = 3.6 A)."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from sronoise import build_supercell  # noqa: E402
from sronoise.mc import SwapMC  # noqa: E402

NAMES = ["Cr", "Mn", "Fe", "Co", "Ni"]
A = 3.6


def write(cell, lab, path, comment):
    order = np.argsort(lab, kind="stable")
    counts = np.bincount(lab, minlength=len(NAMES))
    with open(path, "w") as f:
        f.write(comment + "\n1.0\n")
        for v in np.diag(cell.box) * A:
            f.write("  " + "  ".join(f"{x:12.6f}" for x in v) + "\n")
        f.write("  " + "  ".join(NAMES) + "\n")
        f.write("  " + "  ".join(str(c) for c in counts) + "\n")
        f.write("Cartesian\n")
        for p in cell.positions[order] * A:
            f.write("  " + "  ".join(f"{x:12.6f}" for x in p) + "\n")


cell = build_supercell("fcc", 3, 1)
rng = np.random.default_rng(2026)
base = np.repeat(np.arange(5), [22, 22, 22, 21, 21]).astype(np.int32)
rand = rng.permutation(base)
write(cell, rand, os.path.join(HERE, "POSCAR_random"),
      "CrMnFeCoNi 108 atoms, random arrangement")

# strong Cr-Cr repulsion and Cr-Ni attraction, quenched at 600 K
V = np.zeros((1, 5, 5))
V[0, 0, 0] = 0.06
V[0, 0, 4] = V[0, 4, 0] = -0.04
mc = SwapMC(cell, V, [0])
cfg = np.ascontiguousarray(rand.copy())
mc.run(cfg, 600.0, 2000, 11)
write(cell, cfg, os.path.join(HERE, "POSCAR_ordered"),
      "CrMnFeCoNi 108 atoms, swap MC at 600 K with Cr-Cr repulsion")
