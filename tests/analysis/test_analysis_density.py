"""Effect-size density, distribution function and interval counts."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import integrate
from scipy.stats import norm

from genesis_gwas.analysis.density import (
    DEFAULT_M,
    EffectModel,
    effect_cdf,
    effect_density,
    effect_mixture,
    n_in_interval,
    resolve_model,
)
from genesis_gwas.exceptions import InputError


# --------------------------------------------------------------------------------------
# Literal transcriptions of the upstream R functions (the independent reference here).
# --------------------------------------------------------------------------------------
def r_dmixssnp(x: float, est: np.ndarray) -> float:
    if len(est) == 5:
        p0, s1, s2 = est[1], np.sqrt(est[2]), np.sqrt(est[3])
        return p0 * norm.pdf(x / s1) / s1 + (1 - p0) * norm.pdf(x / s2) / s2
    s1 = np.sqrt(est[1])
    return norm.pdf(x / s1) / s1


def r_numinterval(lower: float, upper: float, est: np.ndarray, M: float = 1070777) -> float:
    pic = est[0]
    if len(est) == 3:
        sig = np.sqrt(est[1])

        def cdf(x: float) -> float:
            return pic * norm.cdf(x / sig) + (1 - pic if x >= 0 else 0.0)
    else:
        p0, s1, s2 = est[1], np.sqrt(est[2]), np.sqrt(est[3])

        def cdf(x: float) -> float:
            base = pic * (p0 * norm.cdf(x / s1) + (1 - p0) * norm.cdf(x / s2))
            return base + (1 - pic if x >= 0 else 0.0)

    return M * 2 * (cdf(upper) - cdf(lower))


# --------------------------------------------------------------------------- resolve
def test_resolve_model_from_vectors(est2, est3):
    m2 = resolve_model(est2)
    assert isinstance(m2, EffectModel)
    assert (m2.n_components, m2.M) == (2, DEFAULT_M)
    np.testing.assert_array_equal(m2.weights, [1.0])
    np.testing.assert_array_equal(m2.variances, [est2[1]])
    assert m2.pic == est2[0] and m2.a == est2[2]

    m3 = resolve_model(list(est3), M=500000)
    assert (m3.n_components, m3.M) == (3, 500000)
    np.testing.assert_allclose(m3.weights, [est3[1], 1 - est3[1]], rtol=0, atol=0)
    np.testing.assert_array_equal(m3.variances, est3[2:4])
    assert m3.heritability == pytest.approx(
        500000 * est3[0] * (est3[1] * est3[2] + (1 - est3[1]) * est3[3]), rel=1e-14
    )
    assert m3.n_ssnp == pytest.approx(500000 * est3[0], rel=1e-15)


def test_resolve_model_from_fitresult_uses_fit_M(make_fit, est3):
    fit = make_fit(est3, M=123456)
    assert resolve_model(fit).M == 123456
    assert resolve_model(fit, M=10).M == 10
    again = resolve_model(resolve_model(fit))
    assert again.M == 123456 and again.n_components == 3
    assert resolve_model(resolve_model(fit), M=77).M == 77


@pytest.mark.parametrize(
    "bad",
    [
        [0.01, 5e-5],  # wrong length
        [0.01, 5e-5, 1e-6, 2.0],  # wrong length
        [0.0, 5e-5, 1e-6],  # pic = 0
        [1.5, 5e-5, 1e-6],  # pic > 1
        [0.01, 0.0, 1e-6],  # zero variance
        [0.01, -1e-5, 1e-6],  # negative variance
        [0.01, 1.2, 1e-4, 1e-5, 0.0],  # p1 > 1
        [0.01, 0.2, 1e-4, -1e-5, 0.0],  # negative variance
        [0.01, np.nan, 1e-6],  # non-finite
        "not numbers",
    ],
)
def test_resolve_model_rejects_bad_vectors(bad):
    with pytest.raises(InputError):
        resolve_model(bad)


@pytest.mark.parametrize("bad_M", [0, -5, 2.5, "many"])
def test_resolve_model_rejects_bad_M(est2, bad_M):
    with pytest.raises(InputError):
        resolve_model(est2, M=bad_M)


def test_effect_mixture(est2, est3):
    w, v = effect_mixture(est2)
    assert w.tolist() == [1.0] and v.tolist() == [est2[1]]
    w, v = effect_mixture(est3)
    assert w.sum() == pytest.approx(1.0, abs=1e-15)
    np.testing.assert_array_equal(v, est3[2:4])
    w[0] = 99.0  # returned arrays are copies
    assert effect_mixture(est3)[0][0] == est3[1]


# --------------------------------------------------------------------------- density
@pytest.mark.parametrize("which", ["est2", "est3"])
def test_density_matches_literal_r(which, request):
    est = request.getfixturevalue(which)
    x = np.linspace(-0.03, 0.03, 61)
    expected = np.array([r_dmixssnp(t, est) for t in x])
    np.testing.assert_allclose(effect_density(x, est), expected, rtol=1e-13)


def test_density_shapes_and_symmetry(est3):
    x = np.linspace(-0.02, 0.02, 12).reshape(3, 4)
    d = effect_density(x, est3)
    assert d.shape == (3, 4)
    np.testing.assert_allclose(d, effect_density(-x, est3), rtol=1e-15)
    assert np.ndim(effect_density(0.0, est3)) == 0
    assert effect_density(np.inf, est3) == 0.0


def test_density_is_derivative_of_cdf(est3):
    x = np.linspace(-0.02, 0.02, 9)
    h = 1e-6
    num = (effect_cdf(x + h, est3) - effect_cdf(x - h, est3)) / (2 * h)
    np.testing.assert_allclose(num, effect_density(x, est3), rtol=1e-6)


def test_cdf_limits_and_monotonicity(est2, est3):
    for est in (est2, est3):
        np.testing.assert_allclose(
            effect_cdf([-np.inf, 0.0, np.inf], est), [0.0, 0.5, 1.0], atol=1e-15
        )
        grid = np.linspace(-0.05, 0.05, 400)
        assert np.all(np.diff(effect_cdf(grid, est)) >= 0)


variance = st.floats(min_value=1e-7, max_value=1e-2)
proportion = st.floats(min_value=1e-4, max_value=0.5)


@settings(max_examples=40, deadline=None)
@given(pic=proportion, p1=st.floats(0.01, 0.99), s1=variance, s2=variance)
def test_density_integrates_to_one(pic, p1, s1, s2):
    theta = [pic, p1, s1, s2, 0.0]
    top = 12.0 * np.sqrt(max(s1, s2))
    breaks = sorted({0.0, *(k * np.sqrt(s) for s in (s1, s2) for k in (-6, -1, 1, 6))})
    total, _ = integrate.quad(
        lambda x: float(effect_density(x, theta)), -top, top, points=breaks, limit=400
    )
    assert total == pytest.approx(1.0, abs=1e-8)


# ------------------------------------------------------------------------- intervals
INTERVALS = [
    (0.005, np.inf),
    (0.0, 0.001),
    (0.001, 0.005),
    (0.01, 0.02),
    (0.0, np.inf),
    (0.0005, 0.03),
]


@pytest.mark.parametrize("which", ["est2", "est3"])
@pytest.mark.parametrize("interval", INTERVALS)
def test_n_in_interval_matches_literal_r(which, interval, request):
    est = request.getfixturevalue(which)
    lo, hi = interval
    for M in (1070777, 500000):
        mine = float(n_in_interval(lo, hi, est, M=M))
        ref = r_numinterval(lo, hi, est, M)
        # upstream differences two distribution functions close to 1, losing ~M * eps
        assert mine == pytest.approx(ref, rel=1e-11, abs=8 * M * np.finfo(float).eps)


def test_n_in_interval_total_is_number_of_ssnps(make_fit, est3):
    assert float(n_in_interval(0.0, np.inf, est3)) == pytest.approx(DEFAULT_M * est3[0], rel=1e-14)
    fit = make_fit(est3, M=200000)
    assert float(n_in_interval(0.0, np.inf, fit)) == pytest.approx(200000 * est3[0], rel=1e-14)
    assert float(n_in_interval(0.0, np.inf, fit, M=100)) == pytest.approx(100 * est3[0], rel=1e-14)


def test_n_in_interval_broadcasts(est2):
    lo = np.array([0.0, 0.001, 0.005])
    out = n_in_interval(lo, np.inf, est2)
    assert out.shape == (3,)
    assert np.all(np.diff(out) < 0)
    grid = n_in_interval(lo[:, None], np.array([0.01, 0.02])[None, :], est2)
    assert grid.shape == (3, 2)


def test_n_in_interval_accurate_in_far_tail(est2):
    """25 SD out: the upstream difference of CDFs returns 0; the tail form does not."""
    sd = np.sqrt(est2[1])
    val = float(n_in_interval(25 * sd, np.inf, est2))
    expected = 2 * DEFAULT_M * est2[0] * norm.sf(25.0)
    assert expected > 0
    assert val == pytest.approx(expected, rel=1e-10)
    assert r_numinterval(25 * sd, np.inf, est2) == 0.0


@pytest.mark.parametrize(
    "lower, upper",
    [(-0.005, 0.005), (0.01, 0.01), (0.02, 0.01), (np.nan, 0.1), (0.0, np.nan), (np.inf, np.inf)],
)
def test_n_in_interval_rejects_invalid_intervals(est2, lower, upper):
    with pytest.raises(InputError):
        n_in_interval(lower, upper, est2)


@settings(max_examples=40, deadline=None)
@given(
    pic=proportion,
    p1=st.floats(0.01, 0.99),
    s1=variance,
    s2=variance,
    cuts=st.lists(st.floats(1e-5, 0.2), min_size=3, max_size=3, unique=True),
)
def test_n_in_interval_is_additive(pic, p1, s1, s2, cuts):
    theta = [pic, p1, s1, s2, 0.0]
    lo, mid, hi = sorted(cuts)
    whole = float(n_in_interval(lo, hi, theta))
    parts = float(n_in_interval(lo, mid, theta)) + float(n_in_interval(mid, hi, theta))
    assert whole >= 0
    assert parts == pytest.approx(whole, rel=1e-9, abs=1e-9)
    assert whole <= DEFAULT_M * pic * (1 + 1e-12)
