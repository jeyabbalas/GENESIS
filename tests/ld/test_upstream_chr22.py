"""Reproduce the packaged chromosome-22 reference from the 1000 Genomes genotypes.

Skipped unless the environment variable ``GENESIS_1000G_PLINK`` points at the directory
holding the LDSC ``1000G_EUR_Phase3_plink`` files (``1000G.EUR.QC.22.{bed,bim,fam}``); see
``tools/ld_validation`` for the genome-wide version of this check.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

import genesis_gwas as gg
from genesis_gwas.ld import BedReader, build_reference, compute_ld

PLINK_DIR = os.environ.get("GENESIS_1000G_PLINK", "")
pytestmark = pytest.mark.skipif(
    not (PLINK_DIR and Path(PLINK_DIR, "1000G.EUR.QC.22.bed").exists()),
    reason="set GENESIS_1000G_PLINK to the LDSC 1000G_EUR_Phase3_plink directory",
)

# The two configurations shipped in the packaged mini bundle.
CONFIGS = [(1000.0, 0.1), (500.0, 0.2)]


def test_compute_ld_reproduces_the_original_tables():
    refs = [gg.load_reference("mini_chr22", window_kb=w, r2_min=t) for w, t in CONFIGS]
    base = refs[0]
    reader, bim, fam = BedReader.from_prefix(Path(PLINK_DIR) / "1000G.EUR.QC.22")
    assert len(fam) == 489
    lookup = {s: i for i, s in enumerate(bim["snp"].tolist())}
    idx = np.array([lookup[s] for s in base.rsid.astype("U").tolist()])
    np.testing.assert_array_equal(bim["pos"].to_numpy()[idx], base.pos)
    geno = reader.read(idx)
    assert not (geno < 0).any()
    tables = compute_ld(geno, base.pos, configs=CONFIGS, round_r2=6)
    for ref, table in zip(refs, tables, strict=True):
        np.testing.assert_array_equal(table.nstar, ref.nstar)
        np.testing.assert_array_equal(table.indptr, ref.neighborhoods.indptr)
        np.testing.assert_array_equal(table.indices, ref.neighborhoods.indices)
        np.testing.assert_allclose(table.ldscore, ref.ldscore, rtol=0, atol=1e-11)
    # pairwise r2 is stored as uint16 in the bundle: agreement to half a quantisation step
    assert np.max(np.abs(tables[0].r2 - base.pair_r2())) <= 0.5 / 65535 + 1e-7
    # without rounding the tagging sets are the same on this chromosome and LD scores move by ~1e-5
    (plain,) = compute_ld(geno, base.pos, configs=[(1000.0, 0.1)])
    np.testing.assert_array_equal(plain.nstar, base.nstar)
    assert 1e-7 < np.max(np.abs(plain.ldscore - base.ldscore)) < 1e-4


def test_build_reference_reproduces_the_packaged_bundle(tmp_path):
    base = gg.load_reference("mini_chr22")
    out = build_reference(
        str(Path(PLINK_DIR) / "1000G.EUR.QC.{chrom}"),
        tmp_path / "rebuilt",
        name="rebuilt_chr22",
        snps=base.rsid.astype("U").tolist(),
        configs=CONFIGS,
        round_r2=6,
        chromosomes=[22],
        population="EUR",
        pair_r2_for=[(1000.0, 0.1)],
    )
    for w, t in CONFIGS:
        ref = gg.load_reference("mini_chr22", window_kb=w, r2_min=t)
        mine = gg.load_reference(out, window_kb=w, r2_min=t)
        assert mine.n_ref == 489
        np.testing.assert_array_equal(mine.rsid, ref.rsid)
        np.testing.assert_array_equal(mine.pos, ref.pos)
        np.testing.assert_array_equal(mine.nstar, ref.nstar)
        np.testing.assert_array_equal(mine.neighborhoods.indices, ref.neighborhoods.indices)
        np.testing.assert_allclose(mine.ldscore, ref.ldscore, rtol=0, atol=1e-11)
    mine = gg.load_reference(out)
    np.testing.assert_array_equal(mine.pair_r2(), base.pair_r2())
