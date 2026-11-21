"""Reading summary statistics in the many shapes they come in."""

from __future__ import annotations

import gzip

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from genesis_gwas.data.sumstats import SumStats, read_sumstats
from genesis_gwas.exceptions import DataError, InputError


def _write(tmp_path, name, text, compress=False):
    p = tmp_path / name
    if compress:
        with gzip.open(p, "wt") as fh:
            fh.write(text)
    else:
        p.write_text(text)
    return p


def test_plain_snp_z_n(tmp_path):
    p = _write(tmp_path, "a.tsv", "SNP\tZ\tN\nrs1\t1.5\t1000\nrs2\t-2.25\t1200\n")
    ss = read_sumstats(p)
    assert list(ss.table.columns) == ["SNP", "Z", "N"] and len(ss) == 2
    np.testing.assert_allclose(ss.table["Z"], [1.5, -2.25])
    assert ss.info["z_from"] == "Z" and ss.info["n_from"] == "N" and ss.source.endswith("a.tsv")


@pytest.mark.parametrize("sep,name", [(",", "a.csv"), (" ", "a.txt"), ("   ", "b.txt"), ("\t", "a.tsv.gz")])
def test_delimiters_and_compression(tmp_path, sep, name):
    rows = ["rsid beta se n", "rs1 0.02 0.01 5000", "rs2 -0.03 0.01 5000"]
    text = "\n".join(sep.join(r.split()) for r in rows) + "\n"
    ss = read_sumstats(_write(tmp_path, name, text, compress=name.endswith(".gz")))
    np.testing.assert_allclose(ss.table["Z"], [2.0, -3.0])
    assert ss.info["z_from"] == "beta / se"


def test_z_from_odds_ratio_and_from_p(tmp_path):
    p = _write(tmp_path, "or.tsv", "MarkerName\tOR\tSE\tP\tN\nrs1\t1.2\t0.05\t0.001\t100\nrs2\t0.8\t0.05\t0.5\t100\n")
    ss = read_sumstats(p)
    np.testing.assert_allclose(ss.table["Z"], np.log([1.2, 0.8]) / 0.05)
    # P + sign of effect, without a standard error
    p = _write(tmp_path, "p.tsv", "snp\tbeta\tpval\tn\nrs1\t0.3\t1e-8\t100\nrs2\t-0.1\t0.05\t100\nrs3\t0.2\t1e-300\t100\n")
    ss = read_sumstats(p)
    expect = np.array([1, -1, 1]) * norm.isf(np.array([1e-8, 0.05, 1e-300]) / 2)
    np.testing.assert_allclose(ss.table["Z"], expect, rtol=1e-10)
    assert np.isfinite(ss.table["Z"]).all() and ss.table["Z"].iloc[2] > 37


def test_sample_size_options(tmp_path):
    base = "SNP\tZ\tNcase\tNcontrol\tNtot\nrs1\t1\t1000\t3000\t4000\nrs2\t2\t500\t500\t1000\n"
    p = _write(tmp_path, "cc.tsv", base)
    ss = read_sumstats(p, n_cases="Ncase", n_controls="Ncontrol")
    np.testing.assert_allclose(ss.table["N"], [750.0, 250.0])  # ncase*ncontrol/(ncase+ncontrol)
    assert "effective" in ss.info["n_from"]
    np.testing.assert_allclose(read_sumstats(p, n_cases=1000, n_controls=3000).table["N"], [750.0, 750.0])
    np.testing.assert_allclose(read_sumstats(p, n=1234).table["N"], [1234.0, 1234.0])
    np.testing.assert_allclose(read_sumstats(p, n="Ntot", case_fraction=0.25).table["N"], [750.0, 187.5])
    # METAL/PGC-style effective N is four times the GENESIS convention
    np.testing.assert_allclose(read_sumstats(p, n="Ntot", n_scale=0.25).table["N"], [1000.0, 250.0])
    # explicit choices beat auto-detected columns; conflicting choices are refused
    np.testing.assert_allclose(read_sumstats(p, n="Ntot").table["N"], [4000.0, 1000.0])
    with pytest.raises(InputError, match="not both"):
        read_sumstats(p, n=100, n_cases=10, n_controls=10)
    with pytest.raises(InputError):
        read_sumstats(p, n="Ntot", case_fraction=1.5)
    with pytest.raises(InputError):
        read_sumstats(p, n="Ntot", n_scale=0.0)


def test_n_from_standard_error_and_allele_frequency(tmp_path):
    f, n_eff = np.array([0.1, 0.4]), np.array([20000.0, 50000.0])
    se = 1.0 / np.sqrt(2 * f * (1 - f) * n_eff)
    df = pd.DataFrame({"SNP": ["rs1", "rs2"], "BETA": [0.01, 0.02], "SE": se, "EAF": f})
    ss = read_sumstats(df, n_from_se=True)
    np.testing.assert_allclose(ss.table["N"], n_eff, rtol=1e-12)
    with pytest.raises(DataError):
        read_sumstats(df.drop(columns="EAF"), n_from_se=True)


def test_column_overrides_and_case_insensitivity(tmp_path):
    p = _write(tmp_path, "odd.tsv", "variant_name\tmy_z\tsamples\tZ\nrs1\t1.0\t10\t99\nrs2\t2.0\t20\t99\n")
    ss = read_sumstats(p, snp="variant_name", z="MY_Z", n="samples")
    np.testing.assert_allclose(ss.table["Z"], [1.0, 2.0])
    np.testing.assert_allclose(ss.table["N"], [10.0, 20.0])
    with pytest.raises(InputError, match="not in the file"):
        read_sumstats(p, snp="nope")


def test_missing_values_survive_as_nan(tmp_path):
    p = _write(tmp_path, "na.tsv", "SNP\tZ\tN\nrs1\tNA\t100\nrs2\t1.0\t.\nrs3\tabc\t100\nrs4\t2.0\t100\n")
    ss = read_sumstats(p)
    assert ss.table["Z"].isna().tolist() == [True, False, True, False]
    assert ss.table["N"].isna().tolist() == [False, True, False, False]


def test_positions_alleles_and_extras(tmp_path):
    p = _write(
        tmp_path, "pos.tsv",
        "CHR\tBP\tA1\tA2\tZ\tN\tINFO\tFRQ\nchr1\t100\ta\tg\t1\t10\t0.9\t0.2\n22\t200\tC\tT\t2\t10\t0.5\t0.6\n",
    )
    ss = read_sumstats(p)
    assert ss.table["SNP"].tolist() == ["1:100", "22:200"]  # synthesised identifier
    assert ss.table["A1"].tolist() == ["A", "C"] and ss.table["POS"].tolist() == [100, 200]
    assert {"INFO", "EAF"} <= set(ss.table.columns)


def test_helpful_errors(tmp_path):
    with pytest.raises(DataError, match="not found"):
        read_sumstats(tmp_path / "missing.tsv")
    with pytest.raises(DataError, match="empty"):
        read_sumstats(_write(tmp_path, "empty.tsv", ""))
    with pytest.raises(DataError, match="Cannot derive Z"):
        read_sumstats(_write(tmp_path, "noz.tsv", "SNP\tN\nrs1\t10\n"))
    with pytest.raises(DataError, match="sample sizes"):
        read_sumstats(_write(tmp_path, "non.tsv", "SNP\tZ\nrs1\t1\n"))
    with pytest.raises(DataError, match="identifier"):
        read_sumstats(_write(tmp_path, "noid.tsv", "Z\tN\n1\t10\n"))


def test_from_arrays():
    ss = SumStats.from_arrays(["rs1", "rs2"], [1.0, 2.0], 500, chr=[1, 2])
    assert ss.table["N"].tolist() == [500.0, 500.0] and "CHR" in ss.table.columns
    with pytest.raises(InputError):
        SumStats.from_arrays(["rs1"], [1.0, 2.0], 500)
