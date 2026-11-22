"""End-to-end fitting on the packaged chromosome-22 data."""

from __future__ import annotations

import json
import warnings

import numpy as np
import pytest

import genesis_gwas as gg
from genesis_gwas.core.derived import canonicalize_m3, derived_quantities
from genesis_gwas.core.spec import ModelSpec
from genesis_gwas.exceptions import ConvergenceWarning, DataWarning, InputError


@pytest.fixture(scope="module")
def ref():
    return gg.load_reference("mini_chr22")


@pytest.fixture(scope="module")
def data(ref):
    return gg.prepare(ref.example_path(), ref, qc=False)


# One chromosome cannot pin down pi_c, so on these data the truncation rule would raise c0
# to its ceiling. Most tests below are about other things and fix the truncation level at
# the original's 10; the rule itself is tested separately.
FIXED = gg.EMConfig(c0=10, c0_max=10)


@pytest.fixture(scope="module")
def m2(data):
    return gg.fit(data, 2, threads=2, em=FIXED)


@pytest.fixture(scope="module")
def m2_legacy(data):
    return gg.fit(data, 2, threads=2, legacy=True)


@pytest.fixture(scope="module")
def m2_em_only(data):
    """Default policy but without Newton polishing: the point where the EM stops."""
    return gg.fit(data, 2, threads=2, polish=False, em=FIXED)


@pytest.fixture(scope="module")
def m2_slow(data):
    """EM to the original's stopping rule, then Newton (no acceleration)."""
    return gg.fit(data, 2, threads=2, accelerate=False, em=FIXED)


def test_two_component_fit_is_sane(m2, data):
    assert m2.model == "M2" and m2.converged and m2.status == "converged"
    assert m2.n_snps == data.K == 13230 and m2.M == 15633
    assert m2.param_names == ("pic", "sigmasq", "a")
    assert 0 < m2.pic < 1 and m2.estimates[1] > 0
    assert np.all(m2.se > 0) and np.all(np.isfinite(m2.covariance))
    np.testing.assert_allclose(m2.covariance, m2.covariance.T, rtol=1e-9)
    assert m2.heritability == pytest.approx(m2.M * m2.pic * m2.estimates[1])
    assert m2.n_ssnp == pytest.approx(m2.M * m2.pic)
    assert np.isfinite(m2.bic) and np.isfinite(m2.aic) and m2.ds > 0
    assert m2.diagnostics["information_pd"]
    assert m2.diagnostics["truncation"]["c0"] == m2.c0
    assert max(abs(v) for v in m2.diagnostics["newton_distance_se"]) < 0.25
    assert len(m2.history) == m2.n_iter
    assert m2.settings["sandwich_double_count"] is False
    assert m2.provenance["version"] == gg.__version__
    text = m2.summary()
    assert "Number of susceptibility SNPs" in text and "M2" in text


def test_legacy_shares_the_em_path_but_has_wider_standard_errors(m2_em_only, m2_legacy):
    """Same EM path, so identical estimates; the original's double-counted 'meat' inflates SEs."""
    np.testing.assert_allclose(m2_legacy.estimates, m2_em_only.estimates, rtol=1e-9)
    assert m2_legacy.loglik == pytest.approx(m2_em_only.loglik, rel=1e-12)
    assert m2_legacy.n_iter == m2_em_only.n_iter
    assert np.all(m2_legacy.se > m2_em_only.se)
    assert m2_legacy.ds > m2_em_only.ds and m2_legacy.bic > m2_em_only.bic
    assert (
        m2_legacy.settings["sandwich_double_count"] is True
        and m2_legacy.settings["polish"] is False
    )
    assert m2_legacy.diagnostics["polish"] is None
    # J_legacy - J_paper = sum_k S_k S_k' is positive semi-definite
    diff = m2_legacy.J - m2_em_only.J
    assert np.all(np.linalg.eigvalsh(diff) > -1e-6 * np.abs(diff).max())


def test_polishing_reaches_the_stationary_point(m2_slow, m2_em_only):
    """EM to the original's stopping rule, then Newton (``accelerate=False``)."""
    slow = m2_slow
    pol = slow.diagnostics["polish"]
    assert pol["status"] == "converged" and 1 <= pol["n_steps"] <= 10 and pol["fixed"] == []
    assert pol["accelerated"] is False and slow.diagnostics["em_status"] == "converged"
    np.testing.assert_allclose(pol["em_estimates"], m2_em_only.estimates, rtol=1e-12)
    assert slow.n_iter == m2_em_only.n_iter
    assert pol["loglik_gain"] >= 0 and slow.loglik >= m2_em_only.loglik
    # the EM stopped a small but non-zero fraction of an SE short; the polished fit is at the optimum
    before = np.max(np.abs(m2_em_only.diagnostics["newton_distance_se"]))
    after = np.max(np.abs(slow.diagnostics["newton_distance_se"]))
    assert 1e-4 < before < 0.25 and after < 1e-5
    assert np.max(np.abs(slow.estimates - m2_em_only.estimates) / slow.se) < 0.25
    assert "Newton step" in slow.summary()


def test_accelerated_default_gives_the_same_estimate_with_far_fewer_iterations(data, m2, m2_slow):
    """Default: a 50-iteration EM warm-up, then Newton — same stationary point."""
    slow = m2_slow
    assert m2.settings["accelerate"] is True and m2.n_iter == 50 < slow.n_iter
    assert m2.converged and m2.status == "converged" and m2.diagnostics["em_status"] == "newton"
    assert m2.diagnostics["polish"]["status"] == "converged"
    assert np.max(np.abs(m2.estimates - slow.estimates) / slow.se) < 1e-4
    assert m2.loglik == pytest.approx(slow.loglik, rel=1e-8)
    np.testing.assert_allclose(m2.se, slow.se, rtol=1e-4)
    assert [h["iter"] for h in m2.history] == list(range(50))
    # a longer warm-up changes nothing but the iteration count
    other = gg.fit(data, 2, threads=2, newton_after=120, variance=False, em=FIXED)
    assert other.n_iter == 120 and np.max(np.abs(other.estimates - m2.estimates) / m2.se) < 1e-4


def test_newton_attempt_that_declines_falls_back_to_more_em(data):
    """With a one-iteration warm-up the first Newton attempts may decline; the fit still converges."""
    res = gg.fit(data, 2, threads=2, newton_after=1, starting=[0.3, 1e-6, 0.0], em=FIXED)
    ref_fit = gg.fit(
        data, 2, threads=2, accelerate=False, starting=[0.3, 1e-6, 0.0], variance=False, em=FIXED
    )
    assert res.converged
    assert np.max(np.abs(res.estimates - ref_fit.estimates) / res.se) < 1e-3
    assert [h["iter"] for h in res.history] == list(range(res.n_iter))


def test_both_sandwich_conventions_are_reported(m2_em_only, m2_legacy):
    """Each fit carries the standard errors of the other convention in its diagnostics."""
    alt = m2_em_only.diagnostics["sandwich_alt"]
    assert alt["convention"] == "original_double_count"
    np.testing.assert_allclose(alt["se"], m2_legacy.se, rtol=1e-7)
    assert alt["ds"] == pytest.approx(m2_legacy.ds, rel=1e-8)
    assert alt["bic"] == pytest.approx(m2_legacy.bic, rel=1e-10)
    back = m2_legacy.diagnostics["sandwich_alt"]
    assert back["convention"] == "paper"
    np.testing.assert_allclose(back["se"], m2_em_only.se, rtol=1e-7)
    assert back["ds"] == pytest.approx(m2_em_only.ds, rel=1e-8)


def test_double_count_switch_alone_reproduces_legacy_covariance(data, m2_legacy):
    res = gg.fit(data, 2, threads=2, sandwich_double_count=True, polish=False, em=FIXED)
    np.testing.assert_allclose(res.covariance, m2_legacy.covariance, rtol=1e-7)
    # ... except the sibling-risk SE, which legacy computes with the original's formula
    risk = res.derived["sibling_risk"]
    assert risk.se == pytest.approx(0.5 * risk.estimate * res.derived["heritability"].se)
    leg = m2_legacy.derived["sibling_risk"]
    assert leg.se == pytest.approx(
        np.sqrt(leg.estimate * m2_legacy.derived["heritability"].se ** 2 / 2)
    )


def test_json_round_trip(m2, tmp_path):
    path = tmp_path / "sub" / "fit.json"
    text = m2.to_json(path)
    assert path.exists() and json.loads(text)["model"] == "M2"
    back = gg.FitResult.from_json(path)
    np.testing.assert_allclose(back.estimates, m2.estimates, rtol=0, atol=0)
    np.testing.assert_allclose(back.covariance, m2.covariance)
    assert back.loglik == m2.loglik and back.c0 == m2.c0 and back.status == m2.status
    assert back.derived["heritability"].estimate == m2.derived["heritability"].estimate
    assert len(back.history) == len(m2.history)
    np.testing.assert_allclose(back.history[-1]["theta"], m2.history[-1]["theta"])
    assert back.summary() == m2.summary()
    slim = json.loads(m2.to_json(include_history=False))
    assert "history" not in slim
    with pytest.raises(InputError):
        gg.FitResult.from_dict({**m2.to_dict(), "schema": 999})


def test_three_component_from_two_component_and_model_selection(data, m2):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m3 = gg.fit(data, 3, starting=m2, threads=2, em=FIXED)
    assert m3.model == "M3" and len(m3.estimates) == 5
    np.testing.assert_allclose(
        m3.starting,
        [m2.estimates[0], 1 / 9, 5 * m2.estimates[1], 0.5 * m2.estimates[1], m2.estimates[2]],
    )
    assert m3.estimates[2] >= m3.estimates[3]  # cluster 1 is the larger-variance cluster
    assert m3.loglik >= m2.loglik - 1e-6 * abs(m2.loglik)  # nested model fits at least as well
    cmp = gg.select_model([m2, m3])
    assert cmp.best in (m2, m3) and cmp.criterion == "bic"
    if m3.diagnostics["degenerate"]:
        # one chromosome cannot separate two clusters: M3 collapses onto M2 and is not eligible
        assert cmp.best is m2 and any("not identified" in w for w in m3.warnings)
        assert "(not identified)" in cmp.summary()
    assert sum(r["selected"] for r in cmp.table) == 1
    assert next(r for r in cmp.table if r["selected"])["delta"] == 0
    assert "Model comparison" in cmp.summary()
    with pytest.raises(InputError):
        gg.select_model([m2, m3], criterion="dic")
    with pytest.raises(InputError):
        gg.select_model([])


def test_three_component_fit_that_collapses_is_stopped_early_and_flagged(data, m2):
    """One chromosome cannot support two clusters: no thousands of EM iterations, no garbage BIC."""
    with pytest.warns(gg.exceptions.NumericalWarning, match="not identified"):
        m3 = gg.fit(data, 3, threads=2)  # default settings, internal M2 pre-fit
    assert m3.diagnostics["degenerate"] and m3.n_iter < 500
    assert m3.status in ("not_identified", "converged")
    assert np.isnan(m3.bic) and np.isnan(m3.aic) and np.isnan(m3.ds)
    assert m3.c0 == 10  # a model that is not identified is not refined at higher truncation levels
    text = m3.summary()
    assert (
        "not available (model not identified)" in text and "nan" not in text.split("Parameter")[0]
    )
    cmp = gg.select_model([m2, m3])
    assert cmp.best is m2 and "n/a" in cmp.summary() and "(not identified)" in cmp.summary()
    with pytest.raises(InputError):
        gg.select_model([m3, gg.fit(data, 2, threads=2, variance=False, em=FIXED)])


def test_default_three_component_start_runs_an_internal_prefit(data):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m3 = gg.fit(data, 3, threads=2, em=gg.EMConfig(max_iter=30, c0_max=10), variance=False)
    pre = m3.diagnostics["prefit_m2"]
    assert len(pre) == 1 and pre[0]["status"] in ("converged", "max_iter", "newton")
    np.testing.assert_allclose(m3.starting[0], pre[0]["estimates"][0])
    assert m3.starting[1] == pytest.approx(1 / 9)
    assert np.all(np.isnan(m3.se)) and m3.covariance is None and np.isnan(m3.bic)


def test_multistart_keeps_the_best_loglik(data):
    res = gg.fit(data, 2, starting_pic=[0.02, 0.05], threads=2, em=FIXED)
    ms = res.diagnostics["multistart"]
    assert len(ms) == 2 and sum(m["selected"] for m in ms) == 1
    assert res.loglik == pytest.approx(max(m["loglik"] for m in ms))
    # with Newton's method both starts reach the same optimum
    np.testing.assert_allclose(ms[0]["estimates"], ms[1]["estimates"], rtol=1e-4)
    explicit = gg.fit(
        data,
        2,
        starting=[[0.01, 1e-5, 1e-6], [0.02, 2e-5, 0.0]],
        threads=2,
        variance=False,
        em=FIXED,
    )
    assert len(explicit.diagnostics["multistart"]) == 2


def test_no_stratification(data):
    res = gg.fit(data, 2, stratification=False, threads=2, em=FIXED)
    assert res.a == 0.0 and res.se[2] == 0.0
    assert np.all(res.covariance[2] == 0)
    assert "a fixed at 0" in res.summary()


def test_prevalences_give_liability_scale(data, m2):
    res = gg.fit(data, 2, threads=2, population_prevalence=0.01, sample_prevalence=0.4, em=FIXED)
    liab = res.diagnostics["liability"]
    assert liab["h2_observed"] == pytest.approx(res.heritability * 0.4 * 0.6)
    assert liab["h2_liability"] > 0 and "liability scale" in res.summary()
    with pytest.raises(InputError):
        gg.fit(data, 2, population_prevalence=0.01)


def test_missing_neighbourhoods_warn(data):
    bare = gg.ModelData(
        betahat=data.betahat,
        varbetahat=data.varbetahat,
        ldscore=data.ldscore,
        nstar=data.nstar,
        n=data.n,
        M=data.M,
    )
    with pytest.warns(DataWarning, match="independent"):
        res = gg.fit(bare, 2, threads=2, em=FIXED)
    assert any("independent" in w for w in res.warnings)


def test_iteration_limit_warns(data):
    with pytest.warns(ConvergenceWarning):
        res = gg.fit(data, 2, threads=2, em=gg.EMConfig(max_iter=3, c0_max=10), polish=False)
    assert not res.converged and res.status == "max_iter" and res.n_iter == 3
    # with Newton's method the same three EM iterations are enough to converge
    ok = gg.fit(data, 2, threads=2, em=gg.EMConfig(max_iter=3, c0_max=10))
    assert ok.converged and ok.status == "converged"
    assert ok.diagnostics["em_status"] in ("max_iter", "newton")


def test_truncation_rule_is_path_independent(data):
    """c0 is raised when the truncated prior mass at the solution is too large, whatever the path."""
    with pytest.warns(gg.exceptions.NumericalWarning, match="Truncation still matters"):
        a = gg.fit(data, 2, threads=2)
    with pytest.warns(gg.exceptions.NumericalWarning):
        b = gg.fit(data, 2, threads=2, starting=[0.3, 1e-6, 0.0], newton_after=1, variance=False)
    path = a.diagnostics["c0_path"]
    assert [p["c0"] for p in path] == sorted(p["c0"] for p in path) and path[0]["c0"] == 10
    assert a.c0 == b.c0 == path[-1]["c0"] == 20  # the ceiling: one chromosome cannot identify pi_c
    assert all(p["mean_tail"] > 1e-6 for p in path)
    assert np.max(np.abs(a.estimates - b.estimates) / a.se) < 1e-3
    assert a.pic > path[0]["estimates"][0]  # truncation had been holding pi_c down
    assert len(a.history) == a.n_iter and {h["c0"] for h in a.history} == {p["c0"] for p in path}
    # heritability, unlike pi_c, barely depends on the truncation level
    fixed = gg.fit(data, 2, threads=2, em=FIXED)
    assert fixed.c0 == 10 and len(fixed.diagnostics["c0_path"]) == 1
    assert fixed.heritability == pytest.approx(a.heritability, rel=0.1)
    # a looser tolerance stops earlier
    loose = gg.fit(data, 2, threads=2, em=gg.EMConfig(tail_tol=1e-3), variance=False)
    assert loose.c0 == 10


def test_progress_callback_and_bad_inputs(data):
    seen = []
    with pytest.warns(ConvergenceWarning):
        gg.fit(
            data,
            2,
            threads=2,
            em=gg.EMConfig(max_iter=5, c0_max=10),
            variance=False,
            polish=False,
            progress=seen.append,
        )
    assert [r["iter"] for r in seen] == list(range(5)) and seen[0]["stage"] == "M2"
    with pytest.raises(InputError):
        gg.fit(data, 4)
    with pytest.raises(InputError):
        gg.fit(data, 2, starting=[0.1, 0.2])
    with pytest.raises(InputError):
        gg.fit(data, 2, starting_pic=[])


def test_derived_quantities_and_canonical_labels():
    spec = ModelSpec(3)
    theta = np.array(
        [0.01, 0.2, 1e-5, 4e-5, 1e-6]
    )  # sigmasq1 < sigmasq2: labels are the wrong way round
    cov = np.diag([1e-6, 1e-4, 1e-12, 4e-12, 1e-14])
    cov[1, 2] = cov[2, 1] = 3e-9
    new, (cov2,), swapped = canonicalize_m3(theta, cov)
    assert swapped
    np.testing.assert_allclose(new, [0.01, 0.8, 4e-5, 1e-5, 1e-6])
    assert cov2[2, 2] == cov[3, 3] and cov2[3, 3] == cov[2, 2] and cov2[1, 3] == -cov[1, 2]
    M = 1000
    before = derived_quantities(spec, theta, cov, M)
    after = derived_quantities(spec, new, cov2, M)
    # relabelling must not change label-free quantities ...
    for key in ("n_ssnp", "heritability", "sibling_risk"):
        assert after[key].estimate == pytest.approx(before[key].estimate)
        assert after[key].se == pytest.approx(before[key].se)
    # ... and swaps the cluster-specific ones
    assert after["n_ssnp_cluster1"].estimate == pytest.approx(before["n_ssnp_cluster2"].estimate)
    assert after["heritability_cluster1"].se == pytest.approx(before["heritability_cluster2"].se)
    same, _, swapped = canonicalize_m3(new, cov2)
    assert not swapped and np.array_equal(same, new)
    # numerical check of one delta-method gradient
    eps = 1e-7
    h = lambda th: derived_quantities(spec, th, None, M)["heritability"].estimate  # noqa: E731
    grad = np.array(
        [
            (h(new + eps * np.eye(5)[i] * new[i]) - h(new - eps * np.eye(5)[i] * new[i]))
            / (2 * eps * new[i])
            for i in range(5)
        ]
    )
    assert after["heritability"].se == pytest.approx(float(np.sqrt(grad @ cov2 @ grad)), rel=1e-6)
