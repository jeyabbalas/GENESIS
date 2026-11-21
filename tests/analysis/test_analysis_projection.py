"""Future-GWAS projections: closed forms vs quadrature, limits, confidence intervals."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import integrate
from scipy.stats import norm

from genesis_gwas.analysis.projection import Projection, power_integrals, project
from genesis_gwas.exceptions import InputError


# --------------------------------------------------------------------------------------
# Independent reference: the integrals of upstream projection(), by adaptive quadrature.
# The integrand is evaluated in units of the component SD so quad sees an O(1) scale.
# --------------------------------------------------------------------------------------
def quad_power_integrals(n: float, s2: float, alpha: float) -> tuple[float, float]:
    c = abs(norm.ppf(alpha / 2))
    sd = np.sqrt(s2)

    def power(u: float) -> float:
        x = sd * u
        return norm.sf(c - np.sqrt(n) * x) + norm.cdf(-c - np.sqrt(n) * x)

    opts = {"epsabs": 0.0, "epsrel": 1e-13, "limit": 500, "points": [-8, -3, -1, 0, 1, 3, 8]}
    e_pow = integrate.quad(lambda u: power(u) * norm.pdf(u), -40, 40, **opts)[0]
    e_b2 = s2 * integrate.quad(lambda u: u * u * power(u) * norm.pdf(u), -40, 40, **opts)[0]
    return e_pow, e_b2


def r_projection(
    est: np.ndarray, n: float, alpha: float = 5e-8, M: float = 1070777
) -> dict[str, float]:
    """Upstream pp(): number of discoveries, GV percentage, variance explained, heritability."""
    pic = est[0]
    if len(est) == 3:
        comps = [(1.0, est[1])]
    else:
        comps = [(est[1], est[2]), (1 - est[1], est[3])]
    herit = pic * M * sum(p * s2 for p, s2 in comps)
    parts = [(p, *quad_power_integrals(n, s2, alpha)) for p, s2 in comps]
    nd = M * pic * sum(p * e for p, e, _ in parts)
    pv = M * pic * sum(p * e for p, _, e in parts)
    return {"nd": nd, "gv": 100 * pv / herit, "pv": pv, "herit": herit}


@pytest.mark.parametrize("n", [5e3, 5e4, 253288, 2e6, 5e7])
@pytest.mark.parametrize("s2", [1e-6, 5.7e-5, 1e-3])
@pytest.mark.parametrize("alpha", [5e-8, 1e-5, 0.05])
def test_power_integrals_match_quadrature(n, s2, alpha):
    e_pow, e_b2 = power_integrals(n, s2, alpha)
    q_pow, q_b2 = quad_power_integrals(n, s2, alpha)
    assert float(e_pow) == pytest.approx(q_pow, rel=1e-10, abs=1e-300)
    assert float(e_b2) == pytest.approx(q_b2, rel=1e-10, abs=1e-300)


def test_power_integrals_limits():
    alpha = 5e-8
    # no power at all: every SNP is "significant" with probability alpha
    e_pow, e_b2 = power_integrals(1e-12, 1e-4, alpha)
    assert float(e_pow) == pytest.approx(alpha, rel=1e-6)
    assert float(e_b2) == pytest.approx(alpha * 1e-4, rel=1e-6)
    # infinite sample size: everything is found
    e_pow, e_b2 = power_integrals(1e18, 1e-4, alpha)
    assert float(e_pow) == pytest.approx(1.0, abs=1e-5)
    assert float(e_b2) == pytest.approx(1e-4, rel=1e-5)
    # zero variance component: power alpha, explains nothing
    e_pow, e_b2 = power_integrals(1e5, 0.0, alpha)
    assert float(e_pow) == pytest.approx(alpha, rel=1e-12)
    assert float(e_b2) == 0.0


def test_power_integrals_broadcast_and_validation():
    e_pow, e_b2 = power_integrals(
        np.array([1e4, 1e5, 1e6])[:, None], np.array([1e-5, 1e-4])[None, :]
    )
    assert e_pow.shape == e_b2.shape == (3, 2)
    assert np.all(np.diff(e_pow, axis=0) > 0) and np.all(np.diff(e_pow, axis=1) > 0)
    for bad in (
        {"n": -1.0, "variance": 1e-4},
        {"n": 1e4, "variance": -1e-4},
        {"n": np.inf, "variance": 1e-4},
    ):
        with pytest.raises(InputError):
            power_integrals(**bad)
    with pytest.raises(InputError):
        power_integrals(1e4, 1e-4, alpha=0.0)


@pytest.mark.parametrize("which", ["est2", "est3"])
@pytest.mark.parametrize("n", [1e4, 133000, 253288, 1e6])
@pytest.mark.parametrize("alpha", [5e-8, 1e-5])
def test_project_matches_upstream_formulas(which, n, alpha, request):
    est = request.getfixturevalue(which)
    res = project(est, n, alpha=alpha)
    ref = r_projection(est, n, alpha)
    assert isinstance(res, Projection)
    assert res.heritability == pytest.approx(ref["herit"], rel=1e-13)
    assert res.n_discoveries[0] == pytest.approx(ref["nd"], rel=1e-10)
    assert res.gv_percentage[0] == pytest.approx(ref["gv"], rel=1e-10)
    assert res.pheno_variance[0] == pytest.approx(ref["pv"], rel=1e-10)


def test_project_reproduces_upstream_height_example():
    """Values computed by the original R package for its bundled height GWAS fit (n = 253,288).

    From tests/fixtures/upstream/height_full_m2.json (`est`, `projection_n253288`); copied
    here so the check does not depend on the fixture file.
    """
    est = [0.004906835485350529, 5.732970955273252e-05, 1.8614813677495352e-06]
    res = project(est, 253288)
    assert res.heritability == pytest.approx(0.3012175508131953, rel=1e-13)
    # upstream integrates numerically (rel.tol 1e-12); the closed form agrees to ~1e-14
    assert res.n_discoveries[0] == pytest.approx(874.5504093866564, rel=1e-11)
    assert res.gv_percentage[0] == pytest.approx(56.30046920500545, rel=1e-11)
    assert res.pheno_variance[0] == pytest.approx(0.16958689443565464, rel=1e-11)


def test_project_vectorised_over_n_and_frame(fit3):
    grid = np.array([1e4, 1e5, 1e6])
    res = project(fit3, grid)
    assert res.n.shape == res.n_discoveries.shape == res.gv_percentage.shape == (3,)
    for i, n in enumerate(grid):
        one = project(fit3, float(n))
        assert one.n_discoveries[0] == pytest.approx(res.n_discoveries[i], rel=1e-14)
    np.testing.assert_allclose(
        res.pheno_variance, res.gv_percentage * res.heritability / 100, rtol=1e-13
    )
    frame = res.to_frame()
    assert isinstance(frame, pd.DataFrame)
    assert list(frame.columns) == [
        "n",
        "heritability",
        "n_discoveries",
        "gv_percentage",
        "pheno_variance",
    ]
    assert len(frame) == 3 and not res.has_ci
    assert res.M == fit3.M and res.alpha == 5e-8


def test_project_M_scales_counts_not_percentages(est3):
    a = project(est3, 253288)
    b = project(est3, 253288, M=200000)
    scale = 200000 / 1070777
    assert b.n_discoveries[0] == pytest.approx(a.n_discoveries[0] * scale, rel=1e-13)
    assert b.heritability == pytest.approx(a.heritability * scale, rel=1e-13)
    assert b.gv_percentage[0] == pytest.approx(a.gv_percentage[0], rel=1e-13)


def test_project_does_not_depend_on_a(est2):
    shifted = est2.copy()
    shifted[2] = -3e-7  # negative inflation estimates are legitimate
    a, b = project(est2, [1e5, 1e6]), project(shifted, [1e5, 1e6])
    np.testing.assert_array_equal(a.n_discoveries, b.n_discoveries)
    np.testing.assert_array_equal(a.gv_percentage, b.gv_percentage)


@pytest.mark.parametrize("bad_n", [[], 0.0, -10.0, [1e4, np.nan]])
def test_project_rejects_bad_sample_sizes(est2, bad_n):
    with pytest.raises(InputError):
        project(est2, bad_n)


@settings(max_examples=40, deadline=None)
@given(
    pic=st.floats(1e-4, 0.3),
    p1=st.floats(0.01, 0.99),
    s1=st.floats(1e-7, 1e-2),
    s2=st.floats(1e-7, 1e-2),
    alpha=st.sampled_from([5e-8, 1e-5, 1e-2]),
)
def test_projection_is_monotone_and_bounded(pic, p1, s1, s2, alpha):
    theta = [pic, p1, s1, s2, 0.0]
    grid = np.logspace(2, 9, 30)
    res = project(theta, grid, alpha=alpha, M=100000)
    slack = 1e-9
    assert np.all(np.diff(res.n_discoveries) >= -slack * res.n_discoveries[-1])
    assert np.all(np.diff(res.gv_percentage) >= -slack * 100)
    assert np.all(res.n_discoveries >= 0) and np.all(
        res.n_discoveries <= 100000 * pic * (1 + 1e-12)
    )
    assert np.all(res.gv_percentage >= 0) and np.all(res.gv_percentage <= 100 * (1 + 1e-12))
    assert np.all(res.pheno_variance <= res.heritability * (1 + 1e-12))


# ------------------------------------------------------------------- confidence intervals
def test_ci_brackets_point_estimate_and_is_reproducible(fit2):
    grid = [5e4, 253288, 1e6]
    a = project(fit2, grid, ci=True, n_sim=2000, seed=7)
    b = project(fit2, grid, ci=True, n_sim=2000, seed=7)
    c = project(fit2, grid, ci=True, n_sim=2000, seed=8)
    assert a.has_ci and a.coverage == 0.95 and a.n_sim == 2000
    for name in ("n_discoveries", "gv_percentage", "pheno_variance"):
        est, lo, hi = getattr(a, name), getattr(a, name + "_lower"), getattr(a, name + "_upper")
        assert np.all(lo < est) and np.all(est < hi), name
        np.testing.assert_array_equal(lo, getattr(b, name + "_lower"))
        np.testing.assert_array_equal(hi, getattr(b, name + "_upper"))
    assert not np.array_equal(a.gv_percentage_lower, c.gv_percentage_lower)
    np.testing.assert_allclose(a.gv_percentage_lower, c.gv_percentage_lower, rtol=0.05)
    np.testing.assert_allclose(
        a.pheno_variance_upper, a.gv_percentage_upper * a.heritability / 100, rtol=1e-13
    )
    frame = a.to_frame()
    assert {"n_discoveries_lower", "gv_percentage_upper", "pheno_variance_lower"} <= set(
        frame.columns
    )


def test_ci_width_follows_coverage_and_uncertainty(fit2, make_fit, est2):
    wide = project(fit2, 253288, ci=True, coverage=0.95, n_sim=4000, seed=1)
    narrow = project(fit2, 253288, ci=True, coverage=0.5, n_sim=4000, seed=1)
    assert (narrow.gv_percentage_upper - narrow.gv_percentage_lower)[0] < (
        wide.gv_percentage_upper - wide.gv_percentage_lower
    )[0]
    tight_fit = make_fit(est2, covariance=fit2.covariance * 1e-4)
    tight = project(tight_fit, 253288, ci=True, n_sim=4000, seed=1)
    assert (tight.gv_percentage_upper - tight.gv_percentage_lower)[0] < 0.05 * (
        wide.gv_percentage_upper - wide.gv_percentage_lower
    )[0]
    # with (almost) no parameter uncertainty the count interval is the binomial one
    nd = tight.n_discoveries[0]
    assert tight.n_discoveries_lower[0] == pytest.approx(nd - 1.96 * np.sqrt(nd), rel=0.06)
    assert tight.n_discoveries_upper[0] == pytest.approx(nd + 1.96 * np.sqrt(nd), rel=0.06)


def test_ci_delta_method_agrees_with_linearisation(fit2):
    """For a well-determined fit the GV interval is close to a first-order delta interval."""
    n = 253288.0
    res = project(fit2, n, ci=True, n_sim=20000, seed=3)
    eps = 1e-6
    grad = np.zeros(3)
    for j in range(2):
        up, dn = fit2.estimates.copy(), fit2.estimates.copy()
        up[j] *= 1 + eps
        dn[j] *= 1 - eps
        grad[j] = (project(up, n).gv_percentage[0] - project(dn, n).gv_percentage[0]) / (
            2 * eps * fit2.estimates[j]
        )
    se = float(np.sqrt(grad @ fit2.covariance @ grad))
    half_width = 0.5 * (res.gv_percentage_upper[0] - res.gv_percentage_lower[0])
    assert half_width == pytest.approx(1.96 * se, rel=0.08)


def test_ci_works_with_negative_a_and_three_components(fit3, make_fit, est2, cov2):
    """The original fails or produces NaN draws here (log of a negative `a`, p1 > 1)."""
    neg = est2.copy()
    neg[2] = -5e-7
    with pytest.raises(InputError):
        project(make_fit(neg), 1e5, ci=True)  # a fit without covariance attached
    res = project(neg, [1e5, 1e6], covariance=cov2, ci=True, n_sim=500, seed=11)
    assert np.all(np.isfinite(res.gv_percentage_lower)) and np.all(
        np.isfinite(res.n_discoveries_upper)
    )
    res3 = project(fit3, [1e5, 1e6], ci=True, n_sim=1500, seed=11)
    for name in ("n_discoveries", "gv_percentage"):
        lo, hi = getattr(res3, name + "_lower"), getattr(res3, name + "_upper")
        assert (
            np.all(np.isfinite(lo))
            and np.all(lo < getattr(res3, name))
            and np.all(getattr(res3, name) < hi)
        )
    assert np.all(res3.gv_percentage_upper <= 100.0)


def test_ci_input_validation(fit2, est2):
    with pytest.raises(InputError):
        project(est2, 1e5, ci=True)  # raw vector without covariance
    with pytest.raises(InputError):
        project(fit2, 1e5, ci=True, coverage=1.0)
    with pytest.raises(InputError):
        project(fit2, 1e5, ci=True, n_sim=1)
    with pytest.raises(InputError):
        project(est2, 1e5, ci=True, covariance=np.eye(5))
    bad = np.full((3, 3), np.nan)
    with pytest.raises(InputError):
        project(est2, 1e5, ci=True, covariance=bad)


def test_power_integrals_do_not_overflow_for_absurd_sample_sizes():
    e_pow, e_b2 = power_integrals(1e306, 1e-2)
    assert float(e_pow) == pytest.approx(1.0, abs=1e-12)
    assert float(e_b2) == pytest.approx(1e-2, rel=1e-12)


def test_ci_needs_interior_p1(est3, cov3_raw):
    edge = est3.copy()
    edge[1] = 1.0
    sym = 0.5 * (cov3_raw + cov3_raw.T)
    assert project(edge, 1e5).n_discoveries[0] > 0  # point estimates are fine at the boundary
    with pytest.raises(InputError):
        project(edge, 1e5, covariance=sym, ci=True)


def test_ci_tolerates_asymmetric_and_indefinite_covariance(est3, cov3_raw):
    """Upstream's three-component covariance is asymmetric (and need not be PSD)."""
    res = project(est3, [1e5, 1e6], covariance=cov3_raw, ci=True, n_sim=400, seed=2)
    assert np.all(np.isfinite(res.gv_percentage_lower)) and np.all(
        res.gv_percentage_lower < res.gv_percentage_upper
    )
    indefinite = -np.eye(5) * 1e-12
    flat = project(est3, 1e5, covariance=indefinite, ci=True, n_sim=50, seed=2)
    # no positive variance direction: draws collapse on the estimate (GV interval has no width)
    assert flat.gv_percentage_lower[0] == pytest.approx(flat.gv_percentage[0], rel=1e-9)
    assert flat.gv_percentage_upper[0] == pytest.approx(flat.gv_percentage[0], rel=1e-9)
