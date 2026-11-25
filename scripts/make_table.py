"""Reference noise floor for common supercell setups (first shell)."""
import csv
import os

import numpy as np

from _style import DATA
from sronoise import WCNull, build_supercell, min_sites, standard_bias

setups = [
    # label, structure, reps list, composition fractions, pair (a, b)
    ("CrMnFeCoNi (fcc, equiatomic)", "fcc", [3, 4, 5, 6, 8],
     [.2] * 5),
    ("CrCoNi (fcc, equiatomic)", "fcc", [3, 4, 5, 6, 8], [1 / 3] * 3),
    ("NbMoTaW (bcc, equiatomic)", "bcc", [4, 5, 6, 8, 10], [.25] * 4),
    ("Ge0.9375Sn0.0625 (diamond)", "diamond", [2, 3, 4, 5], [.9375, .0625]),
    ("Ge0.875Sn0.125 (diamond)", "diamond", [2, 3, 4, 5], [.875, .125]),
]

rows = []
for label, st, reps_list, comp in setups:
    for reps in reps_list:
        cell = build_supercell(st, reps, 1)
        N = cell.n_sites
        counts = [int(round(c * N)) for c in comp]
        counts[-1] = N - sum(counts[:-1])
        null = WCNull(cell, counts, [0])
        a = len(comp) - 1        # minority/last species: worst like pair
        sd_like = float(null.exact_var(0, a, a, "unbiased")) ** 0.5
        sd_unl = float(null.exact_var(0, 0, a, "unbiased")) ** 0.5
        bias = float(standard_bias(N, counts, a, a))
        rows.append(dict(system=label, N=N, z=cell.coordination(0),
                         counts="/".join(map(str, counts)),
                         sigma_like=round(sd_like, 4),
                         sigma_unlike=round(sd_unl, 4),
                         bias_standard_like=round(bias, 4)))
    c = comp[-1]
    rows[-1]["N_for_0.05_at_2sigma_like"] = int(round(min_sites(
        0.05, c, c, cell.coordination(0), 2.0, True)))

keys = ["system", "N", "z", "counts", "sigma_like", "sigma_unlike",
        "bias_standard_like", "N_for_0.05_at_2sigma_like"]
with open(os.path.join(DATA, "reference_table.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k, "") for k in keys})

print("| system | N | σ(α) like | σ(α) unlike | standard-estimator bias (like) |")
print("|---|---:|---:|---:|---:|")
for r in rows:
    print(f"| {r['system']} | {r['N']} | {r['sigma_like']:.3f} | "
          f"{r['sigma_unlike']:.3f} | {r['bias_standard_like']:+.3f} |")
print()
for r in rows:
    if "N_for_0.05_at_2sigma_like" in r:
        print(r["system"], r["N_for_0.05_at_2sigma_like"])
