"""Quality control and merging with the LD reference."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import genesis_gwas as gg
from genesis_gwas.data.prepare import QCConfig, prepare
from genesis_gwas.data.sumstats import SumStats
from genesis_gwas.exceptions import DataError, DataWarning, InputError


@pytest.fixture(scope="module")
def ref():
    return gg.load_reference("mini_chr22")


@pytest.fixture(scope="module")
def table(ref):
    """Clean synthetic summary statistics for 3,000 reference SNPs, shuffled."""
    rng = np.random.default_rng(4)
    rows = np.sort(rng.choice(ref.M, size=3000, replace=False))
    df = pd.DataFrame(
        {
            "SNP": ref.subset_ids(rows),
            "CHR": ref.chrom[rows],
            "POS": ref.pos[rows],
            "POS38": ref.pos_grch38[rows],
            "Z": rng.normal(size=rows.size),
            "N": rng.integers(90_000, 110_000, rows.size).astype(float),
            "ROW": rows,
        }
    )
    return df.sample(frac=1.0, random_state=1).reset_index(drop=True)


def _steps(data):
    return {s["step"]: s["removed"] for s in data.meta["qc"]}


def test_merge_without_qc_matches_the_original_preprocessing(ref, table):
    data = prepare(table[["SNP", "Z", "N"]], ref, qc=False)
    assert data.K == 3000 and data.M == ref.M
    # output is in reference (genomic) order regardless of input order
    assert np.all(np.diff(data.ref_index) > 0)
    back = table.set_index("ROW").loc[data.ref_index]
    np.testing.assert_allclose(data.betahat, back["Z"] / np.sqrt(back["N"]))  # betahat = Z / sqrt(N)
    np.testing.assert_allclose(data.varbetahat, 1.0 / back["N"])              # varbetahat = 1 / N
    np.testing.assert_array_equal(data.nstar, ref.nstar[data.ref_index])
    np.testing.assert_array_equal(data.ldscore, ref.ldscore[data.ref_index])
    np.testing.assert_array_equal(data.snp, back["SNP"].to_numpy())
    assert data.neighborhoods.n_ref == ref.M
    assert data.meta["reference"]["config"] == "w1000kb_r0.1" and data.meta["n_final"] == 3000


def test_recommended_filters_in_the_original_order(ref, table):
    df = table[["SNP", "Z", "N"]].copy()
    df.loc[0, "Z"] = np.nan                       # unusable rows
    df.loc[1, "N"] = -5
    df.loc[2, "N"] = np.inf
    df = pd.concat([df, df.iloc[[10, 11]]])       # duplicated identifiers
    df.loc[20:39, "N"] = 30_000                   # poorly covered SNPs
    df.loc[50, "Z"] = 9.5                         # chi2 = 90.25 > 80
    df.loc[51, "Z"] = -8.9                        # chi2 = 79.21 is kept
    extra = pd.DataFrame({"SNP": ["rs_absent_1", "rs_absent_2"], "Z": [0.1, 0.2], "N": [1e5, 1e5]})
    df = pd.concat([df, extra], ignore_index=True)

    data = prepare(df, ref)                       # qc=None -> recommended filters
    steps = _steps(data)
    assert steps["missing / non-finite Z or N, or N <= 0"] == 3
    assert steps["duplicated SNP identifier (first kept)"] == 2
    n_step = next(k for k in steps if k.startswith("N < 0.67"))
    assert steps[n_step] == 20
    assert next(v for k, v in steps.items() if k.startswith("not in the LD reference")) == 2
    assert steps["chi-square > 80"] == 1
    kept = set(data.snp.tolist())
    assert df.loc[51, "SNP"] in kept and df.loc[50, "SNP"] not in kept
    hm3_removed = steps["outside the HapMap3-without-MHC list"]
    assert data.K == 3000 - 3 - 20 - 1 - hm3_removed
    assert np.all(ref.hm3_no_mhc[data.ref_index])
    # the sample-size yardstick is the 90th percentile of N among usable input rows
    usable = df.drop(index=[0, 1, 2]).drop_duplicates("SNP")
    assert f"({0.67 * np.quantile(usable['N'], 0.9):,.0f})" in n_step
    assert data.meta["qc_config"]["max_chisq"] == 80.0


def test_qc_switches(ref, table):
    df = table[["SNP", "Z", "N"]].copy()
    df.loc[5, "Z"] = 12.0
    keep_big = prepare(df, ref, qc=QCConfig(max_chisq=None, hm3_no_mhc=False, n_min_fraction=None))
    assert keep_big.K == 3000
    custom = prepare(df, ref, qc=QCConfig(max_chisq=100.0, hm3_no_mhc=False, n_min_fraction=None))
    assert custom.K == 2999
    df["INFO"] = 0.95
    df.loc[:9, "INFO"] = 0.3
    df["EAF"] = 0.3
    df.loc[10:14, "EAF"] = 0.995
    filtered = prepare(df, ref, qc=QCConfig(max_chisq=None, hm3_no_mhc=False, n_min_fraction=None, min_info=0.8, min_maf=0.01))
    assert filtered.K == 3000 - 10 - 5
    assert QCConfig.none().hm3_no_mhc is False and QCConfig.none().max_chisq is None


def test_matching_by_position_and_build_detection(ref, table):
    by_pos = table.rename(columns={"SNP": "ID"})[["CHR", "POS", "Z", "N"]]
    data = prepare(by_pos, ref, qc=False)
    assert data.K == 3000 and data.meta["matching"]["match_on"] == "position"
    assert data.meta["matching"]["build"] == "GRCh37"

    hg38 = table[table["POS38"] > 0][["CHR", "POS38", "Z", "N"]].rename(columns={"POS38": "POS"})
    data38 = prepare(hg38, ref, qc=False)        # build auto-detected from the match counts
    assert data38.meta["matching"]["build"] == "GRCh38" and data38.K == len(hg38)
    forced = prepare(hg38, ref, qc=False, build="hg38", match_on="position")
    np.testing.assert_array_equal(forced.ref_index, data38.ref_index)

    # identifiers that are not rsIDs fall back to positions automatically
    odd = table[["CHR", "POS", "Z", "N"]].copy()
    odd["SNP"] = "chr" + odd["CHR"].astype(str) + ":" + odd["POS"].astype(str) + ":A:G"
    assert prepare(odd, ref, qc=False).meta["matching"]["match_on"] == "position"
    with pytest.raises(DataError, match="chromosome and position"):
        prepare(table[["SNP", "Z", "N"]], ref, qc=False, match_on="position")
    with pytest.raises(InputError):
        prepare(table[["SNP", "Z", "N"]], ref, match_on="alleles")


def test_several_inputs_mapping_to_one_reference_snp(ref, table):
    df = table[["CHR", "POS", "Z", "N"]].head(100).copy()
    df["SNP"] = [f"1:{i}:A:G" for i in range(100)]
    dup = df.iloc[[0, 1]].copy()                  # e.g. two alleles at one position
    dup["SNP"] = ["1:0:A:T", "1:1:A:T"]
    with pytest.warns(DataWarning):
        data = prepare(pd.concat([df, dup], ignore_index=True), ref, qc=False)
    assert data.K == 100
    assert _steps(data)["several input SNPs map to the same reference SNP (first kept)"] == 2
    # without distinct identifiers the same rows are caught one step earlier
    with pytest.warns(DataWarning):
        again = prepare(pd.concat([df, dup], ignore_index=True).drop(columns="SNP"), ref, qc=False)
    assert again.K == 100 and _steps(again)["duplicated SNP identifier (first kept)"] == 2


def test_nothing_left_is_an_error_and_low_overlap_warns(ref, table):
    with pytest.raises(DataError, match="No SNPs left"):
        prepare(pd.DataFrame({"SNP": ["a", "b"], "Z": [1.0, 2.0], "N": [10.0, 10.0]}), ref, qc=False)
    with pytest.warns(DataWarning, match="reference SNPs"):
        prepare(table[["SNP", "Z", "N"]].head(500), ref, qc=False)


def test_accepts_paths_sumstats_objects_and_reference_names(ref, table, tmp_path):
    path = tmp_path / "s.tsv"
    table[["SNP", "Z", "N"]].to_csv(path, sep="\t", index=False)
    a = prepare(str(path), "mini_chr22", qc=False)
    b = prepare(SumStats.from_arrays(table["SNP"], table["Z"], table["N"]), ref, qc=False)
    np.testing.assert_array_equal(a.ref_index, b.ref_index)
    np.testing.assert_allclose(a.betahat, b.betahat)
    c = prepare(str(path), ref, qc=False, n=50_000)   # reader options are forwarded
    assert np.all(c.n == 50_000)
    assert prepare(str(path), ref, qc=False, M=1_000_000).M == 1_000_000
    with pytest.raises(InputError):
        prepare(SumStats.from_arrays(["rs1"], [1.0], 10), ref, n=5)


def test_packaged_example_end_to_end(ref):
    data = prepare(ref.example_path(), ref)
    assert 12_000 < data.K <= 13_230
    assert abs(float(np.mean(data.z**2)) - 1.0) < 2.0 and data.z.max() ** 2 <= 80
