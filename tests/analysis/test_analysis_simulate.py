"""Simulators: composite-model draws, LD-reference GWAS, QQ envelopes, future lambda."""

from __future__ import annotations

import itertools
import warnings

import numpy as np
import pytest
from scipy.stats import binom, multinomial, norm

import genesis_gwas as gg
from genesis_gwas.analysis.density import resolve_model
from genesis_gwas.analysis.simulate import (
    QQData,
    _ConfigSampler,
    _ld_marginal,
    _neg_log10_p,
    _thin_index,
    future_gc,
    lambda_gc,
    qq_simulation,
    simulate_ld_sumstats,
    simulate_sumstats,
)
from genesis_gwas.exceptions import DataWarning, InputError, ReferenceDataError

THETA2 = np.array([0.01, 8e-5, 1e-6])
THETA3 = np.array([0.02, 0.2, 3e-4, 3e-5, 5e-7])


def panel(K: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Realistic N* and LD-score arrays."""
    rng = np.random.default_rng(seed)
    nstar = np.clip(rng.gamma(1.6, 29.0, K).astype(np.int64), 1, 700)
    ld = np.maximum(1.0, nstar * rng.uniform(0.2, 0.55, K))
    return nstar, ld


# ------------------------------------------------------------------------- small helpers
def test_lambda_gc_and_neg_log10_p():
    rng = np.random.default_rng(1)
    z = rng.standard_normal(200001)
    assert lambda_gc(z) == pytest.approx(1.0, abs=0.02)
    assert lambda_gc(2 * z) == pytest.approx(4 * lambda_gc(z), rel=1e-12)
    zz = np.array([0.0, 1.0, -1.959963984540054, 5.0, 40.0, -200.0])
    out = _neg_log10_p(zz)
    np.testing.assert_allclose(
        out[:4], -np.log10(2 * norm.sf(np.abs(zz[:4]))), rtol=1e-12, atol=1e-15
    )
    assert out[0] == 0.0 and not np.signbit(out[0])
    assert out[2] == pytest.approx(-np.log10(0.05), rel=1e-12)

    # far beyond double-precision underflow of the p-value itself: compare with the
    # tail expansion p ~ 2 phi(z) / z * (1 - 1/z^2)
    def tail(zabs: float) -> float:
        log_p = (
            np.log(2)
            - 0.5 * zabs**2
            - 0.5 * np.log(2 * np.pi)
            - np.log(zabs)
            + np.log1p(-1 / zabs**2)
        )
        return -log_p / np.log(10)

    assert norm.sf(40.0) == 0.0  # the p-value itself underflows
    assert np.isfinite(out[4]) and out[4] == pytest.approx(tail(40.0), rel=1e-7)
    assert np.isfinite(out[5]) and out[5] == pytest.approx(tail(200.0), rel=1e-9)


def test_thin_index_keeps_tail_exactly():
    idx = _thin_index(100000, 1000)
    assert idx.size <= 1000 and np.all(np.diff(idx) > 0)
    np.testing.assert_array_equal(idx[-500:], np.arange(99500, 100000))
    assert idx[0] == 0
    np.testing.assert_array_equal(_thin_index(50, 1000), np.arange(50))
    with pytest.raises(InputError):
        _thin_index(100, 1)


# ------------------------------------------------------------------ latent-count sampler
def test_untruncated_counts_have_binomial_moments():
    nstar = np.full(300000, 50)
    cnt = _ConfigSampler(nstar, resolve_model(THETA3), None).draw_counts(np.random.default_rng(3))
    assert cnt.shape == (300000, 2)
    pi1, pi2 = THETA3[0] * THETA3[1], THETA3[0] * (1 - THETA3[1])
    se1 = np.sqrt(50 * pi1 * (1 - pi1) / nstar.size)
    se2 = np.sqrt(50 * pi2 * (1 - pi2) / nstar.size)
    assert cnt[:, 0].mean() == pytest.approx(50 * pi1, abs=5 * se1)
    assert cnt[:, 1].mean() == pytest.approx(50 * pi2, abs=5 * se2)
    # multinomial covariance: -N pi1 pi2
    assert np.cov(cnt[:, 0], cnt[:, 1])[0, 1] == pytest.approx(-50 * pi1 * pi2, abs=2e-3)


def test_truncated_m2_sampler_matches_conditioned_binomial():
    model = resolve_model([0.08, 1e-4, 0.0])
    nstar = np.full(400000, 60)
    c0 = 4
    cnt = _ConfigSampler(nstar, model, c0).draw_counts(np.random.default_rng(4))[:, 0]
    assert cnt.max() <= c0
    pmf = binom.pmf(np.arange(c0 + 1), 60, 0.08)
    assert 1 - pmf.sum() > 0.2  # the truncation really bites here
    pmf /= pmf.sum()
    emp = np.bincount(cnt, minlength=c0 + 1) / cnt.size
    np.testing.assert_allclose(emp, pmf, atol=4 * np.sqrt(0.25 / cnt.size))


def test_truncated_m3_sampler_matches_conditioned_multinomial():
    model = resolve_model([0.05, 0.3, 1e-4, 1e-5, 0.0])
    c0 = 4
    for n_tag in (30, 3):  # 3 < c0 exercises the impossible configurations (j1 + j2 > N*)
        nstar = np.full(300000, n_tag)
        cnt = _ConfigSampler(nstar, model, c0).draw_counts(np.random.default_rng(n_tag))
        assert cnt.max() <= c0 and np.all(cnt.sum(axis=1) <= n_tag)
        pmf = {}
        for j1, j2 in itertools.product(range(c0 + 1), repeat=2):
            if j1 + j2 <= n_tag:
                pmf[(j1, j2)] = multinomial.pmf(
                    [j1, j2, n_tag - j1 - j2], n_tag, [0.015, 0.035, 0.95]
                )
        total = sum(pmf.values())
        for (j1, j2), p in pmf.items():
            emp = np.mean((cnt[:, 0] == j1) & (cnt[:, 1] == j2))
            assert emp == pytest.approx(p / total, abs=4 * np.sqrt(0.25 / nstar.size)), (
                n_tag,
                j1,
                j2,
            )


def test_sampler_handles_mixed_nstar_and_multiplier():
    model = resolve_model(THETA3)
    nstar = np.array([1, 2, 5, 40, 700] * 2000)
    sampler = _ConfigSampler(nstar, model, 10)
    rng_a, rng_b = np.random.default_rng(9), np.random.default_rng(9)
    cnt = sampler.draw_counts(rng_a)
    mult = sampler.draw(rng_b)
    np.testing.assert_allclose(mult, cnt @ model.variances, rtol=1e-15)
    assert np.all(cnt.sum(axis=1) <= nstar)
    with pytest.raises(InputError):
        _ConfigSampler(nstar, model, 0)


# --------------------------------------------------------------------- simulate_sumstats
def test_simulate_sumstats_layout_and_reproducibility():
    nstar, ld = panel(5000)
    a = simulate_sumstats(
        nstar, ld, 1e5, THETA2, seed=10, M=123456, snp=[f"s{i}" for i in range(5000)]
    )
    b = simulate_sumstats(nstar, ld, 1e5, THETA2, seed=10, M=123456)
    c = simulate_sumstats(nstar, ld, 1e5, THETA2, seed=11, M=123456)
    assert isinstance(a, gg.ModelData) and a.K == 5000 and a.M == 123456
    np.testing.assert_array_equal(a.betahat, b.betahat)
    assert not np.array_equal(a.betahat, c.betahat)
    np.testing.assert_array_equal(a.nstar, nstar)
    np.testing.assert_allclose(a.varbetahat, 1e-5)
    assert a.snp[3] == "s3" and b.snp is None and a.neighborhoods is None
    assert a.meta["simulated"]["scheme"] == "composite-model"
    assert simulate_sumstats(nstar, ld, 1e5, THETA2).M == 1070777


@pytest.mark.parametrize("theta", [THETA2, THETA3])
def test_simulate_sumstats_second_moment(theta):
    """E[z^2] = 1 + n a + n pic (sum_h p_h sigma_h^2) ld, marker by marker."""
    nstar, ld = panel(400000, seed=1)
    n = np.random.default_rng(2).uniform(5e4, 2e5, nstar.size)
    data = simulate_sumstats(nstar, ld, n, theta, seed=3)
    model = resolve_model(theta)
    expected = 1 + n * model.a + n * model.pic * float(model.weights @ model.variances) * ld
    z2 = data.z**2
    ratio = z2.mean() / expected.mean()
    assert ratio == pytest.approx(1.0, abs=0.02)
    # heteroscedasticity follows the LD score: the regression slope on ld is recovered
    slope = np.polyfit(n * ld, z2, 1)[0]
    assert slope == pytest.approx(model.pic * float(model.weights @ model.variances), rel=0.06)


def test_simulate_sumstats_truncation_reduces_variance_when_it_bites():
    nstar = np.full(200000, 400)
    ld = np.full(200000, 150.0)
    theta = [0.05, 1e-4, 0.0]  # 20 susceptibility SNPs tagged on average
    free = simulate_sumstats(nstar, ld, 1e5, theta, seed=1)
    cut = simulate_sumstats(nstar, ld, 1e5, theta, seed=1, c0=10)
    assert np.mean(cut.z**2) < 0.7 * np.mean(free.z**2)


def test_simulate_sumstats_validation():
    nstar, ld = panel(100)
    with pytest.raises(InputError):
        simulate_sumstats(nstar, ld[:50], 1e5, THETA2)
    with pytest.raises(InputError):
        simulate_sumstats(nstar, ld, -1.0, THETA2)
    with pytest.raises(InputError):
        simulate_sumstats(nstar, ld, 1e5, [0.01, 8e-5, -1.0])  # a + 1/n <= 0
    with pytest.raises(InputError):
        simulate_sumstats([], [], 1e5, THETA2)
    with pytest.raises(InputError):
        simulate_sumstats(nstar, ld, 1e5, [0.01, 8e-5])


def test_fit_recovers_generating_parameters():
    """End to end: data simulated from the composite model are fitted back to the truth."""
    nstar, ld = panel(60000)
    data = simulate_sumstats(nstar, ld, 120000, THETA2, M=1_000_000, seed=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = gg.fit(data, 2, threads=4, variance=False)
    assert fit.estimates[0] == pytest.approx(THETA2[0], rel=0.15)
    assert fit.heritability == pytest.approx(1e6 * THETA2[0] * THETA2[1], rel=0.06)
    assert fit.estimates[2] == pytest.approx(THETA2[2], rel=0.4)


# ------------------------------------------------------------------ LD-reference simulator
def test_ld_marginal_matches_python_loop(mini_ref):
    rng = np.random.default_rng(8)
    beta = np.where(rng.random(mini_ref.M) < 0.05, rng.standard_normal(mini_ref.M), 0.0)
    fast = _ld_marginal(mini_ref, beta, block=1000)
    nb, r2 = mini_ref.neighborhoods, mini_ref.pair_r2().astype(np.float64)
    rows = np.r_[0:150, mini_ref.M - 150 : mini_ref.M]
    slow = np.array(
        [
            np.sum(
                beta[nb.indices[nb.indptr[k] : nb.indptr[k + 1]]]
                * np.sqrt(r2[nb.indptr[k] : nb.indptr[k + 1]])
            )
            for k in rows
        ]
    )
    np.testing.assert_allclose(fast[rows], slow, rtol=1e-12, atol=1e-15)
    np.testing.assert_array_equal(fast, _ld_marginal(mini_ref, beta))  # block size is irrelevant
    with pytest.raises(InputError):
        _ld_marginal(mini_ref, beta[:-1])


def test_ld_simulation_without_signal_returns_the_null_replicate(mini_ref):
    """With (almost) no susceptibility SNPs and a = 0 the Z scores are the stored null replicate."""
    for rep in (0, 7):
        data = simulate_ld_sumstats(mini_ref, [1e-12, 1e-4, 0.0], 1e5, seed=1, replicate=rep)
        assert data.meta["simulated"]["n_causal"] == 0
        np.testing.assert_allclose(
            data.z, mini_ref.null_z(rep).astype(np.float64), rtol=1e-12, atol=1e-12
        )


def test_ld_simulation_layout(mini_ref):
    data = simulate_ld_sumstats(mini_ref, THETA2, 130000, seed=4)
    assert data.K == mini_ref.M and data.M == mini_ref.M
    np.testing.assert_array_equal(data.ref_index, np.arange(mini_ref.M))
    np.testing.assert_array_equal(data.nstar, mini_ref.nstar)
    np.testing.assert_array_equal(data.ldscore, mini_ref.ldscore)
    assert data.snp[0] == mini_ref.subset_ids(np.array([0]))[0]
    assert data.neighborhoods is mini_ref.neighborhoods
    sim = data.meta["simulated"]
    assert (
        sim["scheme"] == "ld-reference"
        and sim["reference"] == "mini_chr22"
        and sim["replicate"] == 0
    )
    assert sim["n_causal"] == pytest.approx(
        mini_ref.M * THETA2[0], abs=5 * np.sqrt(mini_ref.M * THETA2[0])
    )
    again = simulate_ld_sumstats(mini_ref, THETA2, 130000, seed=4)
    np.testing.assert_array_equal(data.betahat, again.betahat)
    other = simulate_ld_sumstats(mini_ref, THETA2, 130000, seed=5)
    assert not np.array_equal(data.betahat, other.betahat)
    assert lambda_gc(data.z) > 1.3  # polygenic inflation at this sample size


def test_ld_simulation_rows_sample_sizes_and_independent_errors(mini_ref):
    rows = np.arange(100, mini_ref.M, 3)
    n = np.linspace(8e4, 1.6e5, mini_ref.M)
    full = simulate_ld_sumstats(mini_ref, THETA3, n, seed=6, replicate=2)
    part = simulate_ld_sumstats(mini_ref, THETA3, n, seed=6, replicate=2, rows=rows)
    np.testing.assert_array_equal(part.ref_index, rows)
    np.testing.assert_array_equal(part.betahat, full.betahat[rows])
    np.testing.assert_array_equal(part.n, n[rows])
    mask = np.zeros(mini_ref.M, dtype=bool)
    mask[rows] = True
    np.testing.assert_array_equal(
        simulate_ld_sumstats(mini_ref, THETA3, n, seed=6, replicate=2, rows=mask).z, part.z
    )
    indep = simulate_ld_sumstats(mini_ref, THETA3, n, seed=6, replicate=None)
    assert indep.meta["simulated"]["replicate"] is None
    assert not np.array_equal(indep.z, full.z)
    # negative `a` cannot be simulated; it is treated as zero (as in the original)
    neg = simulate_ld_sumstats(mini_ref, [0.01, 8e-5, -5e-7], 1e5, seed=6)
    zero = simulate_ld_sumstats(mini_ref, [0.01, 8e-5, 0.0], 1e5, seed=6)
    np.testing.assert_array_equal(neg.z, zero.z)


def test_ld_simulation_validation(mini_ref):
    with pytest.raises(InputError):
        simulate_ld_sumstats(mini_ref, THETA2, np.ones(10), seed=1)
    with pytest.raises(InputError):
        simulate_ld_sumstats(mini_ref, THETA2, 0.0, seed=1)
    with pytest.raises(InputError):
        simulate_ld_sumstats(mini_ref, THETA2, 1e5, seed=1, rows=[0, 0, 1])
    with pytest.raises(InputError):
        simulate_ld_sumstats(mini_ref, THETA2, 1e5, seed=1, rows=[mini_ref.M])
    with pytest.raises(InputError):
        simulate_ld_sumstats(mini_ref, THETA2, 1e5, seed=1, rows=np.zeros(5, dtype=bool))
    with pytest.raises(ReferenceDataError):
        simulate_ld_sumstats(mini_ref, THETA2, 1e5, seed=1, replicate=10)
    other = gg.load_reference("mini_chr22", window_kb=500, r2_min=0.2)  # no pairwise r2 shipped
    with pytest.raises(ReferenceDataError):
        simulate_ld_sumstats(other, THETA2, 1e5, seed=1)


def test_ld_simulated_data_can_be_fitted(mini_ref):
    """The simulated object is a complete input for fit(), sandwich included."""
    data = simulate_ld_sumstats(mini_ref, [0.03, 1.2e-5, 5e-7], 130000, seed=21)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = gg.fit(data, 2, threads=2)
    assert np.all(np.isfinite(fit.se)) and fit.ds > 3  # LD makes d_s exceed the parameter count
    assert 0.2 < fit.heritability / (mini_ref.M * 0.03 * 1.2e-5) < 2.5


# ----------------------------------------------------------------------------- QQ data
def test_qq_envelope_contains_data_simulated_from_the_model():
    nstar, ld = panel(20000)
    data = simulate_sumstats(nstar, ld, 120000, THETA2, M=1_000_000, seed=11, c0=10)
    with pytest.warns(DataWarning, match="independent"):
        qq = qq_simulation(data, THETA2, None, n_sim=60, coverage=0.98, seed=5)
    assert isinstance(qq, QQData) and qq.n_sim == 60 and not qq.ld_errors and qq.n_snps == 20000
    assert qq.observed.shape == qq.expected.shape == qq.lower.shape == qq.upper.shape == (20000,)
    for arr in (qq.observed, qq.expected, qq.lower, qq.upper):
        assert np.all(np.diff(arr) >= 0)
    assert np.all(qq.lower <= qq.expected + 1e-12) and np.all(qq.expected <= qq.upper + 1e-12)
    inside = np.mean((qq.observed >= qq.lower) & (qq.observed <= qq.upper))
    assert inside > 0.9
    assert qq.lambda_lower <= qq.lambda_obs <= qq.lambda_upper
    assert qq.lambda_lower < qq.lambda_mean < qq.lambda_upper
    assert qq.lambda_obs == pytest.approx(lambda_gc(data.z), rel=1e-14)
    # a clearly wrong model does not reproduce the observed quantiles
    with pytest.warns(DataWarning):
        wrong = qq_simulation(data, [0.0005, 5e-5, 0.0], None, n_sim=30, coverage=0.98, seed=5)
    assert np.mean((wrong.observed >= wrong.lower) & (wrong.observed <= wrong.upper)) < 0.2
    assert wrong.lambda_upper < wrong.lambda_obs


def test_qq_expected_curve_is_null_when_there_is_no_signal():
    rng = np.random.default_rng(3)
    K = 30000
    nstar, ld = panel(K)
    data = gg.ModelData.from_z(
        z=rng.standard_normal(K), n=np.full(K, 1e5), ldscore=ld, nstar=nstar, M=10**6
    )
    with pytest.warns(DataWarning):
        qq = qq_simulation(data, [1e-9, 1e-6, 0.0], None, n_sim=40, seed=1)
    # expected order statistics of -log10 U for uniform U
    ranks = np.arange(K, 0, -1)
    uniform = -np.log10((ranks - 0.5) / K)
    mid = slice(K // 10, K - 20)
    np.testing.assert_allclose(qq.expected[mid], uniform[mid], rtol=0.03, atol=0.003)
    assert qq.lambda_mean == pytest.approx(1.0, abs=0.02)


def test_qq_inflation_term_adds_variance_not_a_shift():
    """nu_k is drawn per marker: lambda rises by n * a in every replicate (upstream adds one
    shared draw to all markers, which makes lambda jump around between replicates)."""
    rng = np.random.default_rng(4)
    K = 40000
    nstar, ld = panel(K)
    n = 1e5
    data = gg.ModelData.from_z(
        z=rng.standard_normal(K), n=np.full(K, n), ldscore=ld, nstar=nstar, M=10**6
    )
    a = 3e-6
    with pytest.warns(DataWarning):
        qq = qq_simulation(data, [1e-9, 1e-6, a], None, n_sim=30, coverage=0.9, seed=2)
    assert qq.lambda_mean == pytest.approx(1 + n * a, rel=0.02)
    assert qq.lambda_upper - qq.lambda_lower < 0.08  # a shared shift would spread this over ~1
    with pytest.warns(DataWarning):
        neg = qq_simulation(data, [1e-9, 1e-6, -a], None, n_sim=10, seed=2)
    assert neg.lambda_mean == pytest.approx(1.0, abs=0.03)  # negative a is treated as zero


def test_qq_with_reference_uses_ld_correlated_nulls(mini_ref, mini_data, mini_fit):
    with pytest.warns(DataWarning, match="only 10 null replicates"):
        capped = qq_simulation(mini_data, mini_fit, mini_ref, n_sim=100)
    assert capped.n_sim == 10 and capped.ld_errors and capped.coverage == 0.8
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        qq = qq_simulation(mini_data, mini_fit, mini_ref, n_sim=10)
        again = qq_simulation(mini_data, mini_fit, mini_ref, n_sim=10)
        fewer = qq_simulation(mini_data, mini_fit, mini_ref, n_sim=4)
    np.testing.assert_array_equal(qq.expected, again.expected)
    np.testing.assert_array_equal(qq.expected, capped.expected)
    assert fewer.n_sim == 4 and not np.array_equal(fewer.expected, qq.expected)
    assert qq.lambda_obs == pytest.approx(lambda_gc(mini_data.z), rel=1e-14)
    # the fitted model reproduces the bulk of the observed distribution (a one-chromosome
    # two-component fit is not expected to match the extreme tail)
    bulk = slice(0, int(0.95 * mini_data.K))
    assert np.max(np.abs(qq.expected[bulk] - qq.observed[bulk])) < 0.08
    assert abs(qq.lambda_mean - qq.lambda_obs) < 0.2
    with pytest.warns(DataWarning):
        indep = qq_simulation(mini_data, mini_fit, None, n_sim=10)
    assert not indep.ld_errors
    frame = qq.to_frame()
    assert (
        list(frame.columns) == ["observed", "expected", "lower", "upper"]
        and len(frame) == mini_data.K
    )


def test_qq_thinning_and_options(mini_ref, mini_data, mini_fit):
    full = qq_simulation(mini_data, mini_fit, mini_ref, n_sim=5, seed=9)
    small = qq_simulation(mini_data, mini_fit, mini_ref, n_sim=5, seed=9, max_points=400)
    thin = full.thin(400)
    assert small.observed.size == thin.observed.size <= 400
    np.testing.assert_array_equal(small.observed, thin.observed)
    np.testing.assert_allclose(small.expected, thin.expected, rtol=1e-12)
    np.testing.assert_allclose(small.upper, thin.upper, rtol=1e-12)
    np.testing.assert_array_equal(thin.observed[-200:], full.observed[-200:])
    assert thin.n_snps == full.n_snps == mini_data.K and thin.lambda_obs == full.lambda_obs
    assert full.thin(10**7).observed.size == mini_data.K
    other_c0 = qq_simulation(mini_data, mini_fit, mini_ref, n_sim=5, seed=9, c0=2)
    assert not np.array_equal(other_c0.expected, full.expected)
    for bad in ({"coverage": 0.0}, {"coverage": 1.0}, {"n_sim": 0}):
        with pytest.raises(InputError):
            qq_simulation(mini_data, mini_fit, mini_ref, **bad)


def test_qq_rejects_mismatched_reference(mini_ref, mini_fit):
    nstar, ld = panel(50)
    data = gg.ModelData.from_z(
        z=np.zeros(50), n=np.full(50, 1e5), ldscore=ld, nstar=nstar, M=10**6,
        ref_index=np.arange(mini_ref.M, mini_ref.M + 50),
    )  # fmt: skip
    with pytest.raises(InputError):
        qq_simulation(data, mini_fit, mini_ref, n_sim=2)


# --------------------------------------------------------------------------- future GC
def test_future_gc_grows_with_sample_size(mini_ref, mini_fit):
    grid = np.array([2e4, 1e5, 5e5, 2e6])
    lam = future_gc(mini_fit, grid, mini_ref, n_sim=5, seed=1)
    assert lam.shape == (4,) and np.all(np.isfinite(lam))
    assert np.all(np.diff(lam) > 0) and lam[0] > 0.9
    one = future_gc(mini_fit, 1e5, mini_ref, n_sim=5, seed=1)
    assert isinstance(one, float) and one == pytest.approx(lam[1], rel=1e-14)
    assert future_gc(mini_fit, 1e5, mini_ref, n_sim=5, seed=1) == one
    assert future_gc(mini_fit, 1e5, mini_ref, n_sim=5, seed=2) != one
    assert future_gc(mini_fit, grid.reshape(2, 2), mini_ref, n_sim=2).shape == (2, 2)


def test_future_gc_matches_explicit_simulation(mini_ref):
    """One replicate equals lambda of simulate_ld_sumstats with the same seed and replicate."""
    theta = [0.03, 1.2e-5, 5e-7]
    lam = future_gc(theta, 130000, mini_ref, n_sim=1, seed=17)
    data = simulate_ld_sumstats(mini_ref, theta, 130000, seed=17, replicate=0)
    assert lam == pytest.approx(lambda_gc(data.z), rel=1e-12)


def test_future_gc_without_signal_is_the_null_lambda(mini_ref):
    lam = future_gc([1e-12, 1e-5, 0.0], 1e6, mini_ref, n_sim=3)
    expected = np.mean([lambda_gc(mini_ref.null_z(i)) for i in range(3)])
    assert lam == pytest.approx(expected, rel=1e-6)


def test_future_gc_options_and_validation(mini_ref, mini_fit):
    with pytest.warns(DataWarning, match="only 10 null replicates"):
        capped = future_gc(mini_fit, 1e5, mini_ref, n_sim=50)
    assert capped == pytest.approx(future_gc(mini_fit, 1e5, mini_ref, n_sim=10), rel=1e-14)
    assert future_gc(mini_fit, 1e5, mini_ref, M=mini_ref.M) > 1
    with pytest.raises(InputError):
        future_gc(mini_fit, 1e5, mini_ref, M=1070777)
    with pytest.raises(InputError):
        future_gc(mini_fit, [1e5, -1.0], mini_ref)
    with pytest.raises(InputError):
        future_gc(mini_fit, 1e5, mini_ref, n_sim=0)
