"""Downstream analyses compared with outputs of the ORIGINAL R package.

The fixtures under ``tests/fixtures/upstream`` were produced by running the genuine
upstream code (see ``tools/upstream_oracle``). Each test is skipped if its fixture file
is absent. Tolerances are as tight as the arithmetic of the original allows; every
looser one is justified where it is set.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import pytest
from tests.helpers import load_fixture

import genesis_gwas as gg
from genesis_gwas.analysis.density import effect_density, n_in_interval
from genesis_gwas.analysis.heritability import h2_transfer
from genesis_gwas.analysis.posterior import posterior_marginal
from genesis_gwas.analysis.prediction import polyrisk_predict
from genesis_gwas.analysis.projection import project
from genesis_gwas.analysis.simulate import future_gc, qq_simulation
from genesis_gwas.exceptions import InputError

EPS = float(np.finfo(float).eps)


@pytest.fixture(scope="module")
def rfuncs() -> dict:
    return load_fixture("rfuncs.json")


@pytest.fixture(scope="module")
def stochastic_mini() -> dict:
    return load_fixture("mini_chr22_stochastic.json")


def ok_cases(cases: list[dict]) -> list[dict]:
    good = [c for c in cases if c.get("status", "ok") == "ok"]
    assert good, "fixture contains no successful upstream runs"
    return good


# ------------------------------------------------------------------ deterministic ports
def test_h2transfer(rfuncs):
    for case in rfuncs["h2transfer"]:
        res = h2_transfer(
            case["h2.log"],
            case["se.h2.log"],
            case["population.prevalence"],
            case["sample.prevalence"],
        )
        got = [
            res["h2_observed"],
            res["se_h2_observed"],
            res["h2_liability"],
            res["se_h2_liability"],
        ]
        want = [
            case["h2.observed"],
            case["se.h2.observed"],
            case["h2.liability"],
            case["se.h2.liability"],
        ]
        np.testing.assert_allclose(got, want, rtol=1e-12)


def test_dmixssnp(rfuncs):
    assert len(rfuncs["dmixssnp"]) >= 4
    for case in rfuncs["dmixssnp"]:
        got = effect_density(case["x"], case["est"])
        np.testing.assert_allclose(got, case["density"], rtol=1e-12)
        np.testing.assert_allclose(got, case["density_vectorised_call"], rtol=1e-12)


def test_numinterval(rfuncs):
    assert len(rfuncs["numInterval"]) >= 50
    for case in rfuncs["numInterval"]:
        got = float(n_in_interval(case["lower"], case["upper"], case["est"], M=case["M"]))
        # upstream evaluates 2 M {G(upper) - G(lower)} with G close to 1, so its own result
        # carries an absolute rounding error of a few M * eps; ours is the accurate one
        assert got == pytest.approx(case["value"], rel=1e-12, abs=8 * case["M"] * EPS)


def test_numinterval_negative_lower_is_rejected(rfuncs):
    """Upstream silently counts the null point mass (millions of "SNPs"); we refuse."""
    for case in rfuncs["numInterval_negative_lower"]:
        assert case["value"] > case["M"]  # the upstream answer exceeds the number of SNPs
        with pytest.raises(InputError):
            n_in_interval(case["lower"], case["upper"], case["est"], M=case["M"])


def test_projection_point_estimates(rfuncs):
    cases = ok_cases(rfuncs["projection"] + rfuncs["projection_other_M"])
    assert len(cases) >= 40
    for case in cases:
        res = project(case["est"], case["n"], alpha=case["gwas.significance"], M=case["M"])
        want = case["value"]
        assert res.heritability == pytest.approx(want["heritability"], rel=1e-13)
        # upstream integrates numerically with rel.tol = 1e-12 on (-Inf, Inf); the closed
        # form is exact, so the difference is upstream's quadrature error (observed < 2e-11)
        assert res.n_discoveries[0] == pytest.approx(want["Numdiscoveries"], rel=1e-9)
        assert res.gv_percentage[0] == pytest.approx(want["GVpercentage"], rel=1e-9)
        assert res.pheno_variance[0] == pytest.approx(want["pheno.variance"], rel=1e-9)


def test_polyriskpredict(rfuncs):
    cases = ok_cases(rfuncs["polyriskpredict"])
    assert len(cases) >= 20
    assert {c["type"] for c in cases} == {"optimum", "GWAS"}
    for case in cases:
        res = polyrisk_predict(
            case["N"],
            ps=case["Ps"],
            sig2s=case["Sig2s"],
            M=case["M"],
            M1=case["M1"],
            threshold="optimum" if case["type"] == "optimum" else "gwas",
            alpha_gwas=case["alp.GWAS"],
            k_fold=case["k.fold"],
            legacy=True,  # the original's s = sqrt(mu) in PPI/PCI
        )
        want = case["value"]
        ppi = np.array(list(want["PPI"].values()), dtype=float)
        pci = np.array(list(want["PCI"].values()), dtype=float)
        # AUC: upstream's cmp5 uses 1 - pnorm(c), which keeps ~9 digits at c = 5.45
        assert res.auc == pytest.approx(want["AUC"], rel=1e-9), case["label"]
        # alpha: the optimum is located by Brent's method with tol 1e-5 on a flat maximum;
        # porting R's routine reproduces its iterates to ~1e-7 relative in alpha
        assert res.alpha == pytest.approx(want["alpha"], rel=1e-6), case["label"]
        # PPI/PCI inherit both effects; values below 1e-12 are rounding noise in upstream
        np.testing.assert_allclose(res.ppi, ppi, rtol=1e-6, atol=1e-12, err_msg=case["label"])
        np.testing.assert_allclose(res.pci, pci, rtol=1e-6, atol=1e-12, err_msg=case["label"])


def test_postmarginal_synthetic(rfuncs, make_fit):
    block = rfuncs["postmarginal"]
    d = block["data"]
    data = gg.ModelData(
        betahat=d["betahat"],
        varbetahat=d["varbetahat"],
        ldscore=d["ldscore"],
        nstar=d["Nstar"],
        n=d["N"],
        M=1070777,
        snp=np.array(d["SNP"]),
    )
    assert len(block["cases"]) >= 8
    for case in block["cases"]:
        fit = make_fit(case["est"], c0=int(case["c0"]))
        out = posterior_marginal(data, fit, include_inflation=False, device="cpu", threads=1)
        # same arithmetic in log space; upstream multiplies raw densities (observed 1e-13)
        np.testing.assert_allclose(out["p_null"], case["prob_beta0"], rtol=1e-10, atol=1e-300)
        np.testing.assert_allclose(
            out["post_mean"], case["posteriormean_marginal"], rtol=1e-10, atol=1e-18
        )


# ------------------------------------------------------------- genuine fits on chr22 data
@pytest.mark.parametrize("name", ["mini_chr22_m2.json", "mini_chr22_m3.json"])
def test_mini_chr22_downstream_of_upstream_fits(name, mini_data, make_fit):
    fx = load_fixture(name)
    assert len(fx["fits"]) >= 1
    for f in fx["fits"]:
        est, M = np.array(f["est"]), int(f["M"])
        proj = project(est, 253288, M=M)
        want = f["projection_n253288"]
        assert proj.heritability == pytest.approx(want["heritability"], rel=1e-13)
        assert proj.n_discoveries[0] == pytest.approx(want["Numdiscoveries"], rel=1e-9)
        assert proj.gv_percentage[0] == pytest.approx(want["GVpercentage"], rel=1e-9)
        assert proj.pheno_variance[0] == pytest.approx(want["pheno.variance"], rel=1e-9)
        got = float(n_in_interval(0.005, np.inf, est, M=M))
        assert got == pytest.approx(f["numInterval_0.005_Inf"], rel=1e-12, abs=8 * M * EPS)

        pm = f["postmarginal_first200"]
        fit = make_fit(est, c0=int(pm["c0_used"]), M=M)
        out = posterior_marginal(mini_data, fit, include_inflation=False, threads=2)
        first = out.set_index("snp").loc[pm["snp"]]
        np.testing.assert_allclose(first["p_null"], pm["prob_beta0"], rtol=1e-10)
        np.testing.assert_allclose(
            first["post_mean"], pm["posteriormean_marginal"], rtol=1e-10, atol=1e-18
        )
        sums = f["postmarginal_sums"]
        assert out["p_null"].sum() == pytest.approx(sums["sum_prob_beta0"], rel=1e-11)
        assert out["post_mean"].abs().sum() == pytest.approx(
            sums["sum_abs_posteriormean"], rel=1e-11
        )


@pytest.mark.parametrize("name", ["height_full_m2.json", "height_full_m3.json"])
def test_full_height_projection_and_interval(name):
    f = load_fixture(name)
    est = np.array(f["est"])
    proj = project(est, 253288, M=int(f["M"]))
    want = f["projection_n253288"]
    assert proj.n_discoveries[0] == pytest.approx(want["Numdiscoveries"], rel=1e-9)
    assert proj.gv_percentage[0] == pytest.approx(want["GVpercentage"], rel=1e-9)
    assert proj.heritability == pytest.approx(want["heritability"], rel=1e-13)
    got = float(n_in_interval(0.005, np.inf, est, M=int(f["M"])))
    assert got == pytest.approx(f["numInterval_0.005_Inf"], rel=1e-12, abs=8 * f["M"] * EPS)


# ------------------------------------------------- stochastic: statistical agreement only
@pytest.mark.parametrize("key", ["qqplotdata_simu_m2", "qqplotdata_simu_m3"])
def test_qq_simulation_agrees_with_upstream_statistically(
    stochastic_mini, key, mini_ref, mini_data, make_fit
):
    """R's random numbers cannot be reproduced, so only distributional summaries are compared.

    Upstream ran 100 simulations with its genome-wide null errors; the packaged mini
    reference carries the first 10 of those replicates.
    """
    q = stochastic_mini[key]
    if q.get("status") != "ok":
        pytest.skip(f"upstream qqplotdata.simu failed: {q.get('error_message')}")
    fit = make_fit(q["est"], c0=int(q["c0"]))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mine = qq_simulation(
            mini_data, fit, mini_ref, n_sim=10, coverage=q["CI.coverage"], seed=123
        )
    rows = np.array(q["QQdata_rows1"], dtype=int) - 1
    up = {k: np.array(v, dtype=float) for k, v in q["QQdata"].items()}

    # the observed quantiles and lambda are deterministic: identical
    np.testing.assert_allclose(mine.observed[rows], up["log_obs_pvalues"], rtol=1e-10, atol=1e-12)
    assert mine.lambda_obs == pytest.approx(q["observedlambda"], rel=1e-12)

    # expected curve: Monte-Carlo error of two independent simulations (100 vs 10
    # replicates of ~13k markers); bulk = all but the 2% most significant markers
    bulk = rows < int(0.98 * mini_data.K)
    diff = np.abs(mine.expected[rows] - up["mean_log_exp_pvalues"])
    assert np.max(diff[bulk]) < 0.04
    assert np.median(diff[bulk] / np.maximum(up["mean_log_exp_pvalues"][bulk], 1e-3)) < 0.03
    assert np.median(diff[~bulk] / up["mean_log_exp_pvalues"][~bulk]) < 0.05
    assert mine.lambda_mean == pytest.approx(q["meanEXPlambda"], abs=0.06)

    # envelope: upstream adds ONE N(0, a) draw to all markers of a replicate (a shared
    # shift), which inflates the replicate-to-replicate spread; with the term drawn per
    # marker the band is narrower and lies inside upstream's lambda range
    assert (
        q["lowEXPlambda"] - 0.02 < mine.lambda_lower < mine.lambda_upper < q["highEXPlambda"] + 0.02
    )
    width_up = np.mean((up["upper"] - up["lower"])[bulk])
    width_mine = np.mean((mine.upper - mine.lower)[rows][bulk])
    assert 0.25 * width_up < width_mine < width_up


def test_projection_ci_agrees_with_upstream_statistically(stochastic_mini):
    block = stochastic_mini["projection_CI_m2"]
    if block.get("status") != "ok":
        pytest.skip("upstream projection(CI=TRUE) failed for the chr22 M2 fit")
    m2 = load_fixture("mini_chr22_m2.json")["fits"][0]
    np.testing.assert_allclose(block["est"], m2["est"], rtol=1e-15)
    res = project(
        block["est"], block["n"], covariance=np.array(m2["var_est"]), ci=True,
        coverage=block["CI.coverage"], n_sim=20000, seed=1, M=int(m2["M"]),
    )  # fmt: skip
    want = block["value"]
    nd, gv, pv = (
        want[k]
        for k in (
            "Numdiscoveries_est_low_high",
            "GVpercentage_est_low_high",
            "pheno.variance_est_low_high",
        )
    )
    assert res.n_discoveries[0] == pytest.approx(nd[0], rel=1e-9)
    assert res.gv_percentage[0] == pytest.approx(gv[0], rel=1e-9)
    # interval limits: upstream used 1000 draws of a strongly skewed distribution (its own
    # Monte-Carlo error at the upper limit is ~5-10%) and R's random numbers
    assert res.n_discoveries_lower[0] == pytest.approx(nd[1], rel=0.2)
    assert res.n_discoveries_upper[0] == pytest.approx(nd[2], rel=0.2)
    assert res.gv_percentage_lower[0] == pytest.approx(gv[1], rel=0.2)
    assert res.gv_percentage_upper[0] == pytest.approx(gv[2], rel=0.2)
    assert res.pheno_variance_lower[0] == pytest.approx(pv[1], rel=0.2)
    assert res.pheno_variance_upper[0] == pytest.approx(pv[2], rel=0.2)


def test_projection_ci_succeeds_where_upstream_fails(stochastic_mini):
    """The chr22 three-component fit is degenerate; upstream's mvrnorm() stops with
    "'Sigma' is not positive definite". The eigenvalue-clipped sampler here still works."""
    block = stochastic_mini["projection_CI_m3"]
    m3 = load_fixture("mini_chr22_m3.json")["fits"][0]
    cov = np.array(m3["var_est"], dtype=float)
    if not np.all(np.isfinite(cov)):
        pytest.skip("upstream covariance matrix is not finite for this fit")
    res = project(
        block["est"], block["n"], covariance=cov, ci=True, n_sim=500, seed=3, M=int(m3["M"])
    )
    assert np.isfinite(res.n_discoveries_lower[0]) and np.isfinite(res.gv_percentage_upper[0])
    assert res.n_discoveries_lower[0] <= res.n_discoveries_upper[0]
    if block.get("status") != "ok":
        assert "positive definite" in block.get("error_message", "")


def test_projection_ci_height_examples(rfuncs):
    sto = rfuncs["stochastic"]
    for key in ("projection_CI_m2_height", "projection_CI_m3_height"):
        block = sto[key]
        if block.get("status") != "ok":
            continue
        v = np.array(block["v"], dtype=float)
        # MASS::mvrnorm reads only the lower triangle (eigen(symmetric = TRUE)); mirror it
        v = np.tril(v) + np.tril(v, -1).T
        res = project(
            block["est"],
            block["n"],
            covariance=v,
            ci=True,
            coverage=block["CI.coverage"],
            n_sim=20000,
            seed=5,
        )
        want = block["value"]
        nd, gv = want["Numdiscoveries_est_low_high"], want["GVpercentage_est_low_high"]
        assert res.n_discoveries[0] == pytest.approx(nd[0], rel=1e-9)
        # well-determined fits: limits of two independent Monte-Carlo runs agree closely
        assert res.n_discoveries_lower[0] == pytest.approx(nd[1], rel=0.06), key
        assert res.n_discoveries_upper[0] == pytest.approx(nd[2], rel=0.06), key
        assert res.gv_percentage_lower[0] == pytest.approx(gv[1], rel=0.06), key
        assert res.gv_percentage_upper[0] == pytest.approx(gv[2], rel=0.06), key


def test_projection_ci_with_negative_a_is_finite(rfuncs):
    """Upstream emits 'NaNs produced' here (log of a negative `a`); no NaN is involved here."""
    block = rfuncs["stochastic"]["projection_CI_negative_a"]
    res = project(
        block["est"], block["n"], covariance=np.array(block["v"], dtype=float), ci=True, n_sim=300
    )
    assert np.all(np.isfinite([res.n_discoveries_lower[0], res.n_discoveries_upper[0]]))
    assert res.n_discoveries[0] == pytest.approx(
        block["value"]["Numdiscoveries_est_low_high"][0], rel=1e-9
    )


# --------------------------------------------------- needs the full reference bundle
def _full_reference() -> gg.LDReference:
    if not os.environ.get("GENESIS_DATA_DIR"):
        pytest.skip("set GENESIS_DATA_DIR to a directory holding the eur_1kg_hm3 bundle")
    try:
        ref = gg.load_reference("eur_1kg_hm3", download=False)
        ref.pair_r2()
        ref.null_z(0)
    except Exception as exc:  # incomplete local bundle
        pytest.skip(f"full reference bundle not usable offline: {exc}")
    return ref


@pytest.mark.refdata
@pytest.mark.slow
def test_future_gc_against_upstream_seeded_runs(rfuncs):
    runs = rfuncs["stochastic"].get("futuregc")
    if not runs:
        pytest.skip("fixture has no futuregc runs")
    ref = _full_reference()
    for run in runs:
        mine = future_gc(run["est"], run["n"], ref, n_sim=10, seed=1)
        # single-replicate lambda varies by ~0.02 between random streams; upstream's value
        # is one (or two) such draws
        assert mine == pytest.approx(run["lambdaGC"], abs=0.08), run


@pytest.mark.refdata
@pytest.mark.slow
@pytest.mark.parametrize("name", ["height_full_m2.json", "height_full_m3.json"])
def test_full_height_postmarginal(name, make_fit):
    f = load_fixture(name)
    ref = _full_reference()
    example = Path(ref.root) / "examples" / "height_giant2010.sumstats.tsv.gz"
    if not example.exists():
        pytest.skip("height example not present in the local bundle")
    data = gg.prepare(example, ref, qc=False)
    assert int(f["K"]) == data.K
    pm = f["postmarginal_first200"]
    fit = make_fit(f["est"], c0=int(pm["c0_used"]), M=int(f["M"]))
    out = posterior_marginal(data, fit, include_inflation=False)
    first = out.set_index("snp").loc[pm["snp"]]
    np.testing.assert_allclose(first["p_null"], pm["prob_beta0"], rtol=1e-10)
    np.testing.assert_allclose(
        first["post_mean"], pm["posteriormean_marginal"], rtol=1e-10, atol=1e-18
    )
    sums = f["postmarginal_sums"]
    assert out["p_null"].sum() == pytest.approx(sums["sum_prob_beta0"], rel=1e-10)
    assert out["post_mean"].abs().sum() == pytest.approx(sums["sum_abs_posteriormean"], rel=1e-10)
