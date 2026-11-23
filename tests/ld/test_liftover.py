"""Chain-file liftover and references that carry positions in both genome builds."""

from __future__ import annotations

import gzip
import warnings

import numpy as np
import pandas as pd
import pytest

import genesis_gwas as gg
from genesis_gwas.exceptions import DataError
from genesis_gwas.ld import build_reference
from genesis_gwas.ld.liftover import ChainLiftover, lift_positions

# chr1: [0,1000) -> +500 ; gap ; [1100,2000) -> starts at 1700 (i.e. +600); bases 1000-1099 deleted.
# chr1: [5000,5100) -> reverse strand of a 10,000-base chr1.
# chr2: [0,1000) -> chr9 (another chromosome).
# chr6: 40 Mb shifted by +1000.
CHAIN = """\
chain 1000 chr1 249250621 + 0 2000 chr1 248956422 + 500 2600 1
1000 100 200
900

chain 900 chr1 249250621 + 5000 5100 chr1 10000 - 100 200 2
100

chain 800 chr2 243199373 + 0 1000 chr9 138394717 + 0 1000 3
1000

chain 700 chr6 171115067 + 0 40000000 chr6 170805979 + 1000 40001000 4
40000000

"""


@pytest.fixture()
def chain(tmp_path):
    path = tmp_path / "aToB.over.chain.gz"
    with gzip.open(path, "wt") as fh:
        fh.write(CHAIN)
    return path


def test_point_liftover(chain):
    lo = ChainLiftover(chain)
    name, pos, status = lo.lift(
        "chr1", np.array([1, 1000, 1001, 1100, 1101, 2000, 2001, 5001, 5100])
    )
    assert list(status) == [0, 0, 1, 1, 0, 0, 1, 0, 0]
    # 1-based: base 1 -> 501; base 1000 -> 1500; base 1101 (0-based 1100) -> 0-based 1700 -> 1701
    assert list(pos[[0, 1, 4, 5]]) == [501, 1500, 1701, 2600]
    # reverse strand: 0-based target 5000 -> reverse coord 100 -> forward 0-based 10000-1-100 = 9899
    assert list(pos[[7, 8]]) == [9900, 9801]
    assert all(n == "chr1" for n in name[status == 0])
    assert list(lo.lift("chr5", np.array([10]))[2]) == [1]


def test_lift_positions_flags_other_chromosomes(chain):
    out, stats = lift_positions(chain, np.array([1, 1, 2, 2]), np.array([10, 1050, 10, 5000]))
    assert list(out) == [510, -1, -1, -1]
    assert stats == {"mapped": 1, "deleted": 2, "duplicated": 0, "other_chromosome": 1}


def test_plain_text_chain_and_missing_file(tmp_path):
    path = tmp_path / "plain.chain"
    path.write_text(CHAIN)
    assert lift_positions(path, np.array([1]), np.array([10]))[0][0] == 510
    with pytest.raises(DataError, match="not found"):
        ChainLiftover(tmp_path / "absent.chain")


@pytest.mark.parametrize("build", ["GRCh37", "GRCh38"])
def test_reference_with_both_builds(tmp_path, chain, simulate_genotypes, write_fileset, build):
    rng = np.random.default_rng(3)
    m, n = 60, 80
    geno = simulate_genotypes(rng, m, n)
    chrom = np.array(["6"] * m, dtype=object)
    pos = np.sort(rng.choice(np.arange(20_000_000, 39_000_000), m, replace=False)).astype(np.int64)
    ids = [f"rs{i + 1}" for i in range(m)]
    prefix = write_fileset(tmp_path / "panel", geno, chrom, pos, ids)
    root = build_reference(
        prefix, tmp_path / "ref", name="toy", genome_build=build, liftover_chain=chain, min_maf=0.0
    )
    ref = gg.load_reference(root)
    other = ref.pos_grch38 if build == "GRCh37" else ref.pos_grch37
    assert other is not None
    np.testing.assert_array_equal(other, ref.pos + 1000)
    assert (ref.pos_grch37 if build == "GRCh37" else ref.pos_grch38) is None
    assert ref.manifest["source"]["liftover"]["mapped"] == ref.M

    # summary statistics in the *other* build, without usable identifiers
    table = pd.DataFrame(
        {
            "SNP": [f"chr6:{p}" for p in other],
            "CHR": 6,
            "POS": other,
            "Z": rng.standard_normal(ref.M),
            "N": 50_000,
        }
    )
    path = tmp_path / "ss.tsv"
    table.to_csv(path, sep="\t", index=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = gg.prepare(path, reference=ref, qc=gg.QCConfig(max_chisq=None, n_min_fraction=None))
    other_name = "GRCh38" if build == "GRCh37" else "GRCh37"
    assert data.meta["matching"]["build"] == other_name
    # the positional MHC filter is applied in GRCh37 coordinates whichever build is native
    p37 = ref.pos if build == "GRCh37" else ref.pos_grch37
    in_mhc = int(((p37 >= 26_000_000) & (p37 <= 34_000_000)).sum())
    assert 0 < in_mhc < ref.M
    assert ref.M - in_mhc == data.K


def test_wrong_chain_is_rejected(tmp_path, chain, simulate_genotypes, write_fileset):
    rng = np.random.default_rng(4)
    geno = simulate_genotypes(rng, 30, 60)
    pos = np.arange(1, 31, dtype=np.int64) * 10_000
    prefix = write_fileset(
        tmp_path / "p", geno, np.array(["3"] * 30, dtype=object), pos, [f"rs{i}" for i in range(30)]
    )
    with pytest.raises(DataError, match="could be lifted"):
        build_reference(prefix, tmp_path / "r", name="x", liftover_chain=chain, min_maf=0.0)
