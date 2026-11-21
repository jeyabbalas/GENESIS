"""PLINK .bed/.bim/.fam reader."""

from __future__ import annotations

import numpy as np
import pytest

from genesis_gwas.exceptions import DataError, InputError
from genesis_gwas.ld.plink import BedReader, read_bim, read_fam, resolve_fileset


def test_decode_hand_written_bytes(tmp_path):
    # 5 samples, 2 variants -> 2 bytes per variant.
    # Variant 0 genotypes (A1 dosage): 2, missing, 1, 0, 2
    #   codes (sample order): 00, 01, 10, 11 | 00  -> byte0 = 0b11_10_01_00 = 0xE4, byte1 = 0x00
    # Variant 1 genotypes: 0, 0, 1, missing, 1
    #   codes: 11, 11, 10, 01 | 10 -> byte0 = 0b01_10_11_11 = 0x6F, byte1 = 0b10 = 0x02
    bed = tmp_path / "t.bed"
    bed.write_bytes(bytes([0x6C, 0x1B, 0x01, 0xE4, 0x00, 0x6F, 0x02]))
    reader = BedReader(bed, n_samples=5, n_variants=2)
    assert reader.bytes_per_variant == 2
    g = reader.read()
    assert g.dtype == np.int8 and g.shape == (2, 5)
    np.testing.assert_array_equal(g, [[2, -1, 1, 0, 2], [0, 0, 1, -1, 1]])
    # counting the other allele flips 0 <-> 2 and leaves het / missing alone
    np.testing.assert_array_equal(reader.read(count_a1=False), [[0, -1, 1, 2, 0], [2, 2, 1, -1, 1]])


def test_padding_bits_are_ignored(tmp_path):
    # 3 samples in one byte; the unused top bit pair is set to garbage (0b11).
    bed = tmp_path / "p.bed"
    bed.write_bytes(bytes([0x6C, 0x1B, 0x01, 0b11_10_00_11]))
    g = BedReader(bed, n_samples=3, n_variants=1).read()
    np.testing.assert_array_equal(g, [[0, 2, 1]])


@pytest.mark.parametrize("n_samples", [1, 3, 4, 5, 7, 8, 9, 13])
def test_round_trip_against_independent_encoder(tmp_path, encode_bed, n_samples):
    rng = np.random.default_rng(n_samples)
    geno = rng.choice(
        np.array([-1, 0, 1, 2], dtype=np.int8), size=(37, n_samples), p=[0.1, 0.3, 0.3, 0.3]
    )
    bed = tmp_path / "r.bed"
    bed.write_bytes(encode_bed(geno))
    reader = BedReader(bed, n_samples=n_samples, n_variants=37)
    np.testing.assert_array_equal(reader.read(), geno)
    # tiny chunk size exercises the chunk loop
    np.testing.assert_array_equal(reader.read(chunk=5), geno)


def test_variant_and_sample_selection(tmp_path, encode_bed):
    rng = np.random.default_rng(3)
    geno = rng.integers(0, 3, size=(20, 11)).astype(np.int8)
    bed = tmp_path / "s.bed"
    bed.write_bytes(encode_bed(geno))
    reader = BedReader(bed, 11, 20)
    v = np.array([19, 0, 7, 7, 3])  # arbitrary order, repeats allowed
    s = np.array([10, 2, 5])
    np.testing.assert_array_equal(reader.read(v), geno[v])
    np.testing.assert_array_equal(reader.read(v, samples=s), geno[np.ix_(v, s)])
    assert reader.read(np.array([], dtype=int)).shape == (0, 11)
    with pytest.raises(InputError):
        reader.read([20])
    with pytest.raises(InputError):
        reader.read([0], samples=[11])


def test_rejects_bad_files(tmp_path):
    bad_magic = tmp_path / "a.bed"
    bad_magic.write_bytes(bytes([0x00, 0x1B, 0x01, 0x00]))
    with pytest.raises(DataError, match="magic"):
        BedReader(bad_magic, 4, 1)
    sample_major = tmp_path / "b.bed"
    sample_major.write_bytes(bytes([0x6C, 0x1B, 0x00, 0x00]))
    with pytest.raises(DataError, match="sample-major"):
        BedReader(sample_major, 4, 1)
    wrong_size = tmp_path / "c.bed"
    wrong_size.write_bytes(bytes([0x6C, 0x1B, 0x01, 0x00, 0x00]))
    with pytest.raises(DataError, match="does not match"):
        BedReader(wrong_size, 4, 1)
    with pytest.raises(DataError, match="not found"):
        BedReader(tmp_path / "missing.bed", 4, 1)


def test_bim_fam_and_prefix(tmp_path, write_fileset):
    geno = np.array([[0, 1, 2], [2, 2, -1]], dtype=np.int8)
    prefix = write_fileset(
        tmp_path / "x" / "toy",
        geno,
        np.array([1, 22]),
        np.array([123, 456789]),
        ids=["rs10", "chr22:456789:A:G"],
        iids=["A", "B", "C"],
    )
    bim = read_bim(str(prefix) + ".bim")
    assert list(bim.columns) == ["chrom", "snp", "cm", "pos", "a1", "a2"]
    assert bim["snp"].tolist() == ["rs10", "chr22:456789:A:G"]
    assert bim["chrom"].tolist() == ["1", "22"] and bim["pos"].tolist() == [123, 456789]
    fam = read_fam(str(prefix) + ".fam")
    assert fam["iid"].tolist() == ["A", "B", "C"]
    for spec in (prefix, str(prefix) + ".bed", str(prefix) + ".bim"):
        bed, bim_p, fam_p = resolve_fileset(spec)
        assert bed.suffix == ".bed" and bim_p.exists() and fam_p.exists()
    reader, bim2, fam2 = BedReader.from_prefix(prefix)
    np.testing.assert_array_equal(reader.read(), geno)
    assert len(bim2) == 2 and len(fam2) == 3
    (tmp_path / "x" / "toy.fam").unlink()
    with pytest.raises(DataError, match="incomplete"):
        resolve_fileset(prefix)
