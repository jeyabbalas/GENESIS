"""Polygenic-prediction projections (port of upstream polyriskpredict)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import optimize
from scipy.stats import norm

from genesis_gwas.analysis.prediction import (
    PolygenicPrediction,
    _brent_fmin,
    polyrisk_mu,
    polyrisk_predict,
)
from genesis_gwas.exceptions import GenesisError, InputError


# --------------------------------------------------------------------------------------
# Literal transcription of the helper functions inside upstream polyriskpredict().
# --------------------------------------------------------------------------------------
def r_comp1(N, c, s):
    sd = math.sqrt((1 + N * s) / N)
    tail = 1 - norm.cdf(c / math.sqrt(N), 0, sd) + norm.cdf(-c / math.sqrt(N), 0, sd)
    return s * tail - N * s**2 / (1 + N * s) * (
        -c / math.sqrt(N) * norm.pdf(-c / math.sqrt(N), 0, sd) * 2
    )


def r_comp2(N, c, s):
    sd = math.sqrt((1 + N * s) / N)
    return 1 / N * (1 - norm.cdf(c / math.sqrt(N), 0, sd) + norm.cdf(-c / math.sqrt(N), 0, sd))


def r_comp3(N, c, s):
    return norm.pdf(-c, 0, math.sqrt(1 + N * s)) + norm.pdf(-c, 0, math.sqrt(1 + N * s))


def r_comp4(N, c, s):
    return s * math.sqrt(N) * c / (1 + N * s) * norm.pdf(c, 0, math.sqrt(1 + N * s)) * 2


def r_mu(N, c, Ps, S, M, M1):
    cmp1 = M1 * (Ps[0] * r_comp1(N, c, S[0]) + Ps[1] * r_comp1(N, c, S[1]))
    cmp2 = M1 * (Ps[0] * r_comp2(N, c, S[0]) + Ps[1] * r_comp2(N, c, S[1]))
    cmp3 = M1 * c / N * (Ps[0] * r_comp3(N, c, S[0]) + Ps[1] * r_comp3(N, c, S[1]))
    cmp4 = M1 / math.sqrt(N) * (Ps[0] * r_comp4(N, c, S[0]) + Ps[1] * r_comp4(N, c, S[1]))
    cmp5 = (
        (M - M1) * (1 - norm.cdf(c)) * 2 / N * (1 + c * 2 * norm.pdf(c) / ((1 - norm.cdf(c)) * 2))
    )
    return (cmp1 + cmp4) / math.sqrt(cmp1 + cmp2 + cmp3 + cmp4 + cmp5)


CASES = [
    (253288.0, [0.5, 0.5], [5.733005e-05, 5.733005e-05], 1070777.0, 5254.0853712690005),
    (253288.0, [0.09476025, 0.90523975], [1.458650e-04, 2.227118e-05], 1070777.0, 9529.71),
    (1e5, [0.1, 0.9], [8e-4, 8e-5], 200000.0, 3000.0),
    (1e4, [0.3, 0.7], [1.5e-4, 2e-5], 1070777.0, 9000.0),
]


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("c", [1e-8, 0.01, 0.5, 1.0, 2.0, 3.3, 5.0])
def test_mu_matches_literal_r(case, c):
    N, ps, s2, M, M1 = case
    # at c <= 5 upstream's `1 - pnorm(c)` still carries ~10 significant digits
    assert float(polyrisk_mu(c, N, ps, s2, M, M1)) == pytest.approx(
        r_mu(N, c, ps, s2, M, M1), rel=1e-9
    )


def test_mu_vectorised_over_threshold():
    N, ps, s2, M, M1 = CASES[1]
    grid = np.array([0.1, 1.0, 3.0, 5.0])
    vec = polyrisk_mu(grid, N, ps, s2, M, M1)
    assert vec.shape == (4,)
    for c, v in zip(grid, vec, strict=True):
        assert v == pytest.approx(float(polyrisk_mu(float(c), N, ps, s2, M, M1)), rel=1e-15)


def test_mu_is_covariance_over_sd_by_monte_carlo():
    """mu(c) = cov(S_hat, S) / sd(S_hat) for a thresholded score with independent SNPs."""
    rng = np.random.default_rng(12)
    N, M, M1 = 50000.0, 400000, 4000
    ps, s2 = np.array([0.2, 0.8]), np.array([6e-4, 6e-5])
    c = 3.0
    reps, cov, var = 60, 0.0, 0.0
    for _ in range(reps):
        beta = np.zeros(M)
        comp = rng.choice(2, size=M1, p=ps)
        beta[:M1] = rng.standard_normal(M1) * np.sqrt(s2[comp])
        bhat = beta + rng.standard_normal(M) / math.sqrt(N)
        keep = np.abs(bhat) * math.sqrt(N) > c
        cov += float(bhat[keep] @ beta[keep])
        var += float(bhat[keep] @ bhat[keep])
    mc = (cov / reps) / math.sqrt(var / reps)
    assert float(polyrisk_mu(c, N, ps, s2, M, M1)) == pytest.approx(mc, rel=0.01)


# ------------------------------------------------------------------------ optimiser
def test_brent_port_finds_minimum_like_scipy():
    def f(x: float) -> float:
        return (x - 1.2345) ** 2 + 0.1 * math.sin(5 * x)

    x_r = _brent_fmin(f, -1.0, 4.0, 1e-8)
    ref = optimize.minimize_scalar(
        f, bounds=(-1.0, 4.0), method="bounded", options={"xatol": 1e-10}
    )
    assert x_r == pytest.approx(ref.x, abs=2e-7)
    # boundary minimum: the routine never evaluates outside [a, b]
    seen: list[float] = []

    def g(x: float) -> float:
        seen.append(x)
        return x

    x_b = _brent_fmin(g, 2.0, 3.0, 1e-6)
    assert min(seen) >= 2.0 and max(seen) <= 3.0
    assert x_b == pytest.approx(2.0, abs=1e-5)


@pytest.mark.parametrize("case", CASES)
def test_optimum_agrees_with_independent_search(case):
    N, ps, s2, M, M1 = case
    res = polyrisk_predict(N, ps=ps, sig2s=s2, M=M, M1=M1)
    assert isinstance(res, PolygenicPrediction) and res.threshold == "optimum"
    grid = np.exp(np.linspace(math.log(1e-8), math.log(7.0), 20001))
    mus = polyrisk_mu(grid, N, ps, s2, M, M1)
    i = int(np.argmax(mus))
    ref = optimize.minimize_scalar(
        lambda c: -float(polyrisk_mu(c, N, ps, s2, M, M1)),
        bounds=(grid[max(i - 1, 0)], grid[min(i + 1, grid.size - 1)]),
        method="bounded",
        options={"xatol": 1e-12},
    )
    assert res.mu == pytest.approx(-ref.fun, rel=1e-9)
    assert res.mu >= float(mus.max()) * (1 - 1e-9)
    c_opt = norm.isf(res.alpha / 2)
    assert c_opt == pytest.approx(
        ref.x, abs=5e-5
    )  # the optimiser tolerance of the original is 1e-5
    assert res.auc == pytest.approx(norm.cdf(res.mu / math.sqrt(2)), rel=1e-14)


def test_gwas_threshold_uses_given_alpha():
    N, ps, s2, M, M1 = CASES[0]
    res = polyrisk_predict(N, ps=ps, sig2s=s2, M=M, M1=M1, threshold="gwas", alpha_gwas=1e-5)
    assert res.alpha == 1e-5 and res.threshold == "gwas"
    assert res.mu == pytest.approx(
        float(polyrisk_mu(norm.isf(0.5e-5), N, ps, s2, M, M1)), rel=1e-12
    )
    best = polyrisk_predict(N, ps=ps, sig2s=s2, M=M, M1=M1)
    assert best.mu >= res.mu and best.auc >= res.auc


# -------------------------------------------------------------------- risk stratification
def test_ppi_pci_formulas_and_risk_sd_conventions():
    N, ps, s2, M, M1 = CASES[2]
    k = np.array([2.0, 3.0, 5.0])
    up = polyrisk_predict(N, ps=ps, sig2s=s2, M=M, M1=M1, k_fold=k, legacy=True)
    de = polyrisk_predict(N, ps=ps, sig2s=s2, M=M, M1=M1, k_fold=k)
    assert up.legacy is True and de.legacy is False
    assert up.mu == de.mu and up.auc == de.auc and up.alpha == de.alpha
    for res, s in ((up, math.sqrt(up.mu)), (de, de.mu)):
        xi = np.log(k) / s
        # upstream: PPI = 1 - pnorm(xi + s/2), PCI = 1 - pnorm(xi - s/2) with s = sqrt(mu)
        np.testing.assert_allclose(res.ppi, 1 - norm.cdf(xi + s / 2), rtol=1e-9, atol=1e-15)
        np.testing.assert_allclose(res.pci, 1 - norm.cdf(xi - s / 2), rtol=1e-9, atol=1e-15)
        assert np.all(np.diff(res.ppi) < 0) and np.all(res.pci > res.ppi)
    assert up.mu < 1 and np.all(de.ppi < up.ppi)  # sqrt(mu) > mu when mu < 1


def test_log_normal_risk_model_identities():
    """Under log-risk ~ N(0, s^2): AUC = Phi(s / sqrt 2); cases have log-risk ~ N(s^2, s^2)."""
    rng = np.random.default_rng(5)
    s = 0.6
    pop = rng.standard_normal(400000) * s
    cases = pop + s * s
    auc_mc = np.mean(cases[:200000] > pop[200000:])
    assert norm.cdf(s / math.sqrt(2)) == pytest.approx(auc_mc, abs=4e-3)
    k = 3.0
    cut = math.log(k) + s * s / 2  # risk >= k times the population average exp(s^2 / 2)
    assert norm.sf(math.log(k) / s + s / 2) == pytest.approx(np.mean(pop > cut), rel=0.05)
    assert norm.sf(math.log(k) / s - s / 2) == pytest.approx(np.mean(cases > cut), rel=0.03)


# --------------------------------------------------------------------- fitted models
def test_fit_defaults_m2_equals_explicit_upstream_call(fit2):
    res = polyrisk_predict(253288, fit2)
    sig = fit2.estimates[1]
    explicit = polyrisk_predict(
        253288, ps=[0.5, 0.5], sig2s=[sig, sig], M=1070777, M1=1070777 * fit2.estimates[0]
    )
    assert res.mu == pytest.approx(explicit.mu, rel=1e-13)
    assert res.alpha == pytest.approx(explicit.alpha, rel=1e-9)
    np.testing.assert_allclose(res.pci, explicit.pci, rtol=1e-9)
    np.testing.assert_array_equal(res.k_fold, [3.0, 4.0, 5.0])


def test_fit_defaults_m3_and_raw_vector(fit3, est3):
    a = polyrisk_predict(1e5, fit3, threshold="gwas")
    b = polyrisk_predict(1e5, est3, threshold="gwas")
    c = polyrisk_predict(
        1e5,
        ps=[est3[1], 1 - est3[1]],
        sig2s=est3[2:4],
        M=1070777,
        M1=1070777 * est3[0],
        threshold="gwas",
    )
    assert a.mu == b.mu == pytest.approx(c.mu, rel=1e-13)


def test_M_override_keeps_fitted_number_of_ssnps(fit3):
    full = polyrisk_predict(1e5, fit3)
    fewer = polyrisk_predict(1e5, fit3, M=200000)
    explicit = polyrisk_predict(1e5, fit3, M=200000, M1=fit3.M * fit3.pic)
    assert fewer.mu == explicit.mu
    assert fewer.auc > full.auc  # fewer null SNPs contribute noise


def test_array_n_matches_scalar_calls_and_auc_grows(fit3):
    grid = np.array([1e4, 5e4, 2e5, 1e6])
    vec = polyrisk_predict(grid, fit3)
    assert vec.auc.shape == vec.alpha.shape == vec.mu.shape == (4,)
    assert vec.ppi.shape == vec.pci.shape == (4, 3)
    for i, n in enumerate(grid):
        one = polyrisk_predict(float(n), fit3)
        assert isinstance(one.auc, float) and one.ppi.shape == (3,)
        assert one.mu == vec.mu[i] and one.alpha == vec.alpha[i]
    assert np.all(np.diff(vec.auc) > 0) and np.all((vec.auc > 0.5) & (vec.auc < 1))
    gws = polyrisk_predict(grid, fit3, threshold="gwas")
    assert np.all(vec.auc >= gws.auc)
    payload = vec.to_dict()
    assert payload["threshold"] == "optimum" and len(payload["auc"]) == 4


def test_no_power_limit_includes_every_snp():
    """With a tiny study the best score uses all SNPs: the optimum sits at the grid edge."""
    res = polyrisk_predict(50.0, ps=[1.0], sig2s=[1e-6], M=100000, M1=1000)
    assert res.alpha == pytest.approx(1.0, abs=1e-7)
    assert 0.5 < res.auc < 0.5001


@pytest.mark.parametrize(
    "kwargs",
    [
        {"threshold": "best"},
        {"alpha_gwas": 0.0},
        {"k_fold": []},
        {"k_fold": [-1.0]},
        {"ps": [0.5, 0.6], "sig2s": [1e-4, 1e-5], "M1": 10.0},
        {"ps": [0.5, 0.5], "sig2s": [1e-4], "M1": 10.0},
        {"ps": [0.5, 0.5], "sig2s": [1e-4, -1e-5], "M1": 10.0},
        {"M1": 0.0},
        {"M": 100.0, "M1": 500.0},
    ],
)
def test_input_validation(fit2, kwargs):
    with pytest.raises(InputError):
        polyrisk_predict(1e5, fit2, **kwargs)


def test_requires_distribution_without_fit():
    with pytest.raises(InputError):
        polyrisk_predict(1e5, ps=[1.0], sig2s=[1e-4])
    with pytest.raises(InputError):
        polyrisk_predict(1e5)
    with pytest.raises(InputError):
        polyrisk_predict([1e5, -1.0], ps=[1.0], sig2s=[1e-4], M1=100.0)


def test_score_without_predictive_value_is_handled():
    """All effect variances zero: mu = 0, AUC = 0.5, nobody deviates from average risk."""
    res = polyrisk_predict(1e5, ps=[1.0], sig2s=[0.0], M=1000, M1=10, k_fold=[0.5, 1.0, 2.0])
    assert res.mu == 0.0 and res.auc == 0.5
    np.testing.assert_array_equal(res.ppi, [1.0, 1.0, 0.0])
    np.testing.assert_array_equal(res.pci, [1.0, 1.0, 0.0])


def test_extreme_sample_size_does_not_overflow():
    """n * sigma^2 overflowing to infinity is the perfect-power limit, not an error."""
    huge = polyrisk_predict(1e306, ps=[1.0], sig2s=[1e-2], M=1000, M1=10, threshold="gwas")
    big = polyrisk_predict(1e30, ps=[1.0], sig2s=[1e-2], M=1000, M1=10, threshold="gwas")
    assert math.isfinite(huge.mu) and huge.mu == pytest.approx(big.mu, rel=1e-9)
    assert huge.mu == pytest.approx(math.sqrt(10 * 1e-2), rel=1e-6)  # sd of the true genetic value


def test_brent_port_rejects_non_finite_objective():
    calls = {"n": 0}

    def f(x: float) -> float:
        calls["n"] += 1
        return x * x if calls["n"] < 3 else float("nan")

    with pytest.raises(GenesisError):
        _brent_fmin(f, -1.0, 2.0, 1e-8)
