#!/usr/bin/env python3
"""Simulate the GWAS used for the GENESISasian parity fixtures (eas_sim_m{2,3}.json).

    python make_eas_sim.py EAS_BUNDLE_DIR OUT.tsv.gz

GENESISasian ships no East Asian GWAS (its ``heightGWAS`` is the European one of
GENESIS), so the parity run uses summary statistics simulated on the converted
``eas_1kg_hm3`` reference: joint effects from a three-component model, marginal effects
through the reference's pairwise LD, LD-correlated noise from null replicate 0, and 90 %
of the reference SNPs kept as markers with per-SNP sample sizes of 135,000-150,000.
The output (columns ``snp``, ``z``, ``n``) is deterministic given the bundle; the test
regenerates it and checks its MD5 against the fixture.
"""
import sys

import numpy as np
import pandas as pd

import genesis_gwas as gg

TRUTH = np.array([0.004, 0.10, 3.0e-4, 3.0e-5, 1.0e-6])  # pic, p1, sigmasq1, sigmasq2, a
SEED = 20261123


def simulate(bundle):
    ref = gg.load_reference(bundle, download=False)
    rng = np.random.default_rng(SEED)
    n = np.round(150_000 * rng.uniform(0.9, 1.0, ref.M))
    rows = np.sort(rng.choice(ref.M, size=int(0.9 * ref.M), replace=False))
    sim = gg.simulate_ld_sumstats(ref, TRUTH, n, seed=SEED, replicate=0, rows=rows)
    return ref, pd.DataFrame(
        {"snp": np.asarray(sim.snp).astype(str), "z": sim.z, "n": sim.n.astype(np.int64)}
    )


def write(table, path):
    table.to_csv(path, sep="\t", index=False, float_format="%.10g",
                 compression={"method": "gzip", "mtime": 0} if str(path).endswith(".gz") else None)


if __name__ == "__main__":
    ref, table = simulate(sys.argv[1])
    write(table, sys.argv[2])
    h2 = ref.M * TRUTH[0] * (TRUTH[1] * TRUTH[2] + (1 - TRUTH[1]) * TRUTH[3])
    print(f"M={ref.M:,} rows={len(table):,} sSNPs={ref.M * TRUTH[0]:,.0f} h2={h2:.4f} "
          f"mean chi2={np.mean(table.z**2):.4f} max chi2={np.max(table.z**2):.1f}")
