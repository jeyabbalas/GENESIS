"""VCF -> PLINK conversion (``genesis_gwas.ld.vcf``)."""

from __future__ import annotations

import gzip

import numpy as np
import pytest

from genesis_gwas.exceptions import DataError, InputError
from genesis_gwas.ld import BedReader, vcf_to_plink

SAMPLES = ["S1", "S2", "S3", "S4", "S5"]
HEADER = (
    "##fileformat=VCFv4.1\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
    + "\t".join(SAMPLES)
)
# (chrom, pos, id, ref, alt, format, genotypes) and the expected ALT dosages (-1 missing)
RECORDS = [
    ("1", 100, "rs1", "A", "G", "GT", "0|0 0|1 1|0 1|1 0|0", [0, 1, 1, 2, 0]),
    ("1", 200, "rs2", "C", "T", "GT", "1|1 .|. 0|1 0|0 1|1", [2, -1, 1, 0, 2]),
    ("1", 300, "rs3", "AT", "A", "GT", "0|0 0|1 1|0 1|1 0|0", None),  # indel
    ("1", 400, "rs4", "A", "C,G", "GT", "0|0 0|1 1|2 1|1 0|0", None),  # multi-allelic
    ("1", 500, "rs5;rs55", "G", "A", "GT:DP", "0/1:9 1/1:8 ./.:0 0/0:7 0/1:3", [1, 2, -1, 0, 1]),
    ("1", 600, "rs1", "A", "G", "GT", "1|1 1|1 1|1 1|1 1|1", None),  # duplicated identifier
    ("chr2", 50, "rs6", "t", "c", "GT", "0|1 0|1 0 1|1 0|0", [1, 1, -1, 2, 0]),  # haploid call
    ("X", 10, "rs7", "A", "G", "GT", "0|0 0|1 1|0 1|1 0|0", None),  # not autosomal
    ("2", 70, "rs8", "A", "G", "GT", "0|0 0|1 1|0 1|1 0|0", [0, 1, 1, 2, 0]),
    ("2", 90, ".", "A", "G", "GT", "0|0 0|0 1|0 1|1 0|0", [0, 0, 1, 2, 0]),
]


def _write_vcf(path, records=RECORDS, gz=False):
    lines = [HEADER]
    for c, pos, ident, ref, alt, fmt, gts, _ in records:
        gt = "\t".join(gts.split())
        lines.append(f"{c}\t{pos}\t{ident}\t{ref}\t{alt}\t.\tPASS\t.\t{fmt}\t{gt}")
    text = ("\n".join(lines) + "\n").encode()
    if gz:
        with gzip.open(path, "wb") as fh:
            fh.write(text)
    else:
        path.write_bytes(text)
    return path


def _read(prefix):
    reader, bim, fam = BedReader.from_prefix(prefix)
    return reader.read(count_a1=True), bim, fam


@pytest.mark.parametrize("gz", [False, True])
def test_all_biallelic_snps_are_converted(tmp_path, gz):
    vcf = _write_vcf(tmp_path / ("a.vcf.gz" if gz else "a.vcf"), gz=gz)
    (res,) = vcf_to_plink(vcf, tmp_path / "out", processes=1)
    geno, bim, fam = _read(tmp_path / "out")
    expected = [r for r in RECORDS if r[7] is not None]
    assert list(fam["iid"]) == SAMPLES
    assert list(bim["snp"]) == ["rs1", "rs2", "rs5", "rs6", "rs8", "2:90:A:G"]
    assert list(bim["chrom"].astype(str)) == ["1", "1", "1", "2", "2", "2"]
    assert list(bim["pos"]) == [100, 200, 500, 50, 70, 90]
    np.testing.assert_array_equal(geno, np.array([r[7] for r in expected], dtype=np.int8))
    # A1 is ALT, A2 is REF (upper-cased)
    assert list(bim["a1"]) == ["G", "T", "A", "C", "G", "G"]
    assert list(bim["a2"]) == ["A", "C", "G", "T", "A", "A"]
    assert res["records"] == len(RECORDS)
    assert res["written"] == 6
    assert res["not_biallelic_snp"] == 2
    assert res["duplicate_id"] == 1
    assert res["not_autosomal"] == 1


def test_snp_list_and_sample_selection(tmp_path):
    vcf = _write_vcf(tmp_path / "a.vcf")
    snps = tmp_path / "snps.txt"
    snps.write_text("rs2\nrs55\nrs8\nrs_absent\n")
    keep = tmp_path / "keep.txt"
    keep.write_text("S5\nS2\nS4\n")
    (res,) = vcf_to_plink(vcf, tmp_path / "sub", snps=snps, keep=keep, processes=1)
    geno, bim, fam = _read(tmp_path / "sub")
    assert list(fam["iid"]) == ["S2", "S4", "S5"]  # VCF column order
    assert list(bim["snp"]) == ["rs2", "rs55", "rs8"]  # the listed alias is the one written
    np.testing.assert_array_equal(geno, np.array([[-1, 0, 2], [2, 0, 1], [1, 2, 0]], dtype=np.int8))
    assert res["not_listed"] == len(RECORDS) - 3


def test_per_chromosome_template_and_build(tmp_path):
    by_chrom = {"1": [r for r in RECORDS if r[0] == "1"], "2": [r for r in RECORDS if "2" in r[0]]}
    for c, recs in by_chrom.items():
        _write_vcf(tmp_path / f"p.chr{c}.vcf.gz", recs, gz=True)
    res = vcf_to_plink(
        tmp_path / "p.chr{chrom}.vcf.gz", tmp_path / "bed" / "p.{chrom}", chromosomes=[1, 2]
    )
    assert [r["written"] for r in res] == [3, 3]
    g1, bim1, _ = _read(tmp_path / "bed" / "p.1")
    assert list(bim1["snp"]) == ["rs1", "rs2", "rs5"]
    assert g1.shape == (3, 5)


def test_errors(tmp_path):
    vcf = _write_vcf(tmp_path / "a.vcf")
    with pytest.raises(DataError, match="not found"):
        vcf_to_plink(tmp_path / "missing.vcf", tmp_path / "o")
    with pytest.raises(InputError, match="chrom"):
        vcf_to_plink(tmp_path / "x.{chrom}.vcf", tmp_path / "o")
    with pytest.raises(DataError, match="No SNP was written"):
        vcf_to_plink(vcf, tmp_path / "o", snps=["rs_none"], processes=1)
    with pytest.raises(DataError, match="none of the samples"):
        vcf_to_plink(vcf, tmp_path / "o", keep=["nobody"], processes=1)
    bad = tmp_path / "bad.vcf"
    bad.write_text("1\t100\trs1\tA\tG\n")
    with pytest.raises(DataError, match="not a VCF"):
        vcf_to_plink(bad, tmp_path / "o", processes=1)


def test_cli_from_vcf(tmp_path):
    from genesis_gwas.cli.main import main

    vcf = _write_vcf(tmp_path / "a.vcf")
    rc = main(["ld", "from-vcf", "--vcf", str(vcf), "--out", str(tmp_path / "cli"), "--quiet"])
    assert rc == 0
    geno, _, _ = _read(tmp_path / "cli")
    assert geno.shape == (6, 5)


def test_selection_by_position(tmp_path):
    """Panels without rsIDs: choose SNPs by position and name them from the list."""
    records = [
        ("chr1", 100, ".", "A", "G", "GT", "0|0 0|1 1|0 1|1 0|0", [0, 1, 1, 2, 0]),
        ("chr1", 200, "1:200:C:CT", "C", "CT", "GT", "0|0 0|1 1|0 1|1 0|0", None),  # indel
        ("chr1", 200, "1:200:C:T", "C", "T", "GT", "1|1 0|0 0|1 0|0 1|1", [2, 0, 1, 0, 2]),
        ("chr1", 200, "1:200:C:G", "C", "G", "GT", "0|0 0|0 0|1 0|0 0|0", None),  # same position
        ("chr1", 300, ".", "G", "A", "GT", "0|1 1|1 0|0 0|0 0|1", None),  # not listed
        ("chr2", 50, ".", "T", "C", "GT", "0|1 0|1 0|0 1|1 0|0", [1, 1, 0, 2, 0]),
    ]
    vcf = _write_vcf(tmp_path / "p.vcf", records)
    pos = tmp_path / "positions.tsv"
    pos.write_text("CHR\tPOS\tSNP\n1\t100\trs100\nchr1\t200\trs200\n2\t50\n7\tNA\trs_unlifted\n")
    (res,) = vcf_to_plink(vcf, tmp_path / "out", positions=pos, processes=1)
    geno, bim, _ = _read(tmp_path / "out")
    assert list(bim["snp"]) == ["rs100", "rs200", "2:50:T:C"]
    assert list(bim["pos"]) == [100, 200, 50]
    np.testing.assert_array_equal(geno, np.array([r[7] for r in records if r[7]], dtype=np.int8))
    assert res["written"] == 3 and res["not_listed"] == 1
    assert res["not_biallelic_snp"] == 1 and res["duplicate_position"] == 1
    # the same through tuples, and the two kinds of list exclude each other
    (res2,) = vcf_to_plink(
        vcf, tmp_path / "o2", positions=[(1, 100, "a"), ("chr2", 50)], processes=1
    )
    assert res2["written"] == 2
    with pytest.raises(InputError, match="not both"):
        vcf_to_plink(vcf, tmp_path / "o3", snps=["rs1"], positions=pos)
    empty = tmp_path / "empty.tsv"
    empty.write_text("CHR POS\nX 5\n")
    with pytest.raises(DataError, match="no usable autosomal position"):
        vcf_to_plink(vcf, tmp_path / "o4", positions=empty)
