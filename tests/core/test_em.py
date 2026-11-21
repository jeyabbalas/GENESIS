"""Behaviour of the EM driver: parity with the original control flow and the safeguards."""

from __future__ import annotations

import numpy as np
import pytest
import tests.upstream_port as up

from genesis_gwas.core.em import EMConfig, newton_direction, newton_polish, run_em
from genesis_gwas.core.engine import Engine
from genesis_gwas.core.likelihood import CompositeLikelihood
from genesis_gwas.core.spec import ModelSpec
from genesis_gwas.exceptions import GenesisError, InputError

TOL2 = dict(eps1=1e-6, eps2=1e-8, eps3=1e-9, eps=1e-6)
TOL3 = dict(eps1=1e-6, eps2=1e-5, eps3=1e-8, eps4=1e-8, eps5=1e-9, eps=1e-6)
LO, HI = (1e-7, 1e-7), (0.5, 0.5)


def _legacy(data, plan, spec, start, **kw):
    with Engine(data, plan) as eng:
        lik = CompositeLikelihood(spec, eng, kw.get("c0", 10))
        return run_em(lik, start, EMConfig(legacy=True, **kw))


def _port2(d, start, n_iter, c0=10, strat=True):
    return up.EM_func(
        start,
        d.betahat,
        d.varbetahat,
        d.ldscore,
        d.nstar,
        d.M,
        c0,
        Meps=n_iter,
        steps=2,
        stratification=strat,
        **TOL2,
    )


def _port3(d, start, n_iter, c0=10, strat=True):
    return up.EM_func3(
        start,
        LO,
        HI,
        d.betahat,
        d.varbetahat,
        d.ldscore,
        d.nstar,
        d.M,
        c0,
        Meps=n_iter,
        steps=2,
        stratification=strat,
        **TOL3,
    )


# ------------------------------------------------------------------ legacy parity
@pytest.mark.parametrize(
    "start,c0,strat",
    [
        (np.array([0.01, 6e-5, 8e-7]), 10, True),  # regular
        (np.array([0.05, 2e-5, 8e-7]), 3, True),  # truncation bites: c0 gets raised
        (np.array([0.01, 6e-5, 8e-7]), 10, False),  # a forced to 0 (joint-Newton quirk)
        (np.array([0.4, 0.5, 1e-6]), 10, True),  # absurd start: clamp/reset path
    ],
    ids=["regular", "c0=3", "nostrat", "bad-start"],
)
def test_legacy_m2_follows_port_iteration_by_iteration(small_data, plan1, start, c0, strat):
    spec = ModelSpec(2, stratification=strat)
    for n_iter in (1, 2, 3, 5, 8, 13, 25):
        ref = _port2(small_data, start, n_iter, c0, strat)
        res = _legacy(small_data, plan1, spec, start, c0=c0, max_iter=n_iter)
        np.testing.assert_allclose(
            res.theta, ref[2:5], rtol=1e-7, atol=1e-18, err_msg=f"iter {n_iter}"
        )
        assert res.c0 == int(ref[6])
        assert res.loglik == pytest.approx(ref[1], rel=1e-9)


@pytest.mark.parametrize(
    "start,c0,strat",
    [
        (np.array([0.01, 1 / 9, 3e-4, 3e-5, 8e-7]), 10, True),
        (np.array([0.04, 0.2, 1e-4, 2e-5, 8e-7]), 3, True),
        (np.array([0.01, 1 / 9, 3e-4, 3e-5, 8e-7]), 10, False),
        (np.array([0.01, 0.49, 3e-4, 3e-5, 8e-7]), 10, True),  # p1 near its upper bound
    ],
    ids=["regular", "c0=3", "nostrat", "p1-bound"],
)
def test_legacy_m3_follows_port_iteration_by_iteration(small_data, plan1, start, c0, strat):
    spec = ModelSpec(3, stratification=strat)
    for n_iter in (1, 2, 3, 6, 12):
        ref = _port3(small_data, start, n_iter, c0, strat)
        res = _legacy(small_data, plan1, spec, start, c0=c0, max_iter=n_iter)
        np.testing.assert_allclose(
            res.theta, ref[2:7], rtol=1e-6, atol=1e-18, err_msg=f"iter {n_iter}"
        )
        assert res.c0 == int(ref[8])
        assert res.loglik == pytest.approx(ref[1], rel=1e-9)


def test_legacy_converged_fit_matches_port(small_data, plan1):
    start = np.array([0.01, 6e-5, 8e-7])
    ref = _port2(small_data, start, 3000)
    res = _legacy(small_data, plan1, ModelSpec(2), start)
    assert res.status in ("converged", "legacy_break")
    assert res.n_iter == int(ref[0]) + 1 or res.status == "legacy_break"
    np.testing.assert_allclose(res.theta, ref[2:5], rtol=1e-6)
    assert res.loglik == pytest.approx(ref[1], rel=1e-9)


# ------------------------------------------------------------ default policy
def test_default_equals_legacy_when_no_safeguard_triggers(medium_data):
    from genesis_gwas.compute import resolve_plan

    plan = resolve_plan("cpu", threads=2)
    start = np.array([0.01, 1e-4, 1e-6])
    with Engine(medium_data, plan) as eng:
        lik = CompositeLikelihood(ModelSpec(2), eng, 10)
        default = run_em(lik, start, EMConfig())
        legacy = run_em(lik, start, EMConfig(legacy=True))
    assert default.status == legacy.status == "converged"
    assert default.n_iter == legacy.n_iter
    assert default.n_backtracks == 0
    np.testing.assert_allclose(default.theta, legacy.theta, rtol=1e-9)
    assert default.loglik == pytest.approx(legacy.loglik, rel=1e-12)


def test_em_objective_is_monotone(medium_data):
    from genesis_gwas.compute import resolve_plan

    with Engine(medium_data, resolve_plan("cpu", threads=2)) as eng:
        lik = CompositeLikelihood(ModelSpec(2), eng, 10)
        res = run_em(lik, np.array([0.01, 1e-4, 1e-6]), EMConfig(max_iter=150))
    llu = np.array([h["loglik_unnorm"] for h in res.history])
    assert np.all(np.diff(llu) >= -1e-9 * np.abs(llu[:-1]))


@pytest.fixture(scope="module")
def m3_run(medium_data):
    """One converged three-component EM run shared by several tests (it is the slow part)."""
    from genesis_gwas.compute import resolve_plan

    with Engine(medium_data, resolve_plan("cpu", threads=4)) as eng:
        lik = CompositeLikelihood(ModelSpec(3), eng, 10)
        em = run_em(lik, np.array([0.01, 1 / 9, 5e-4, 5e-5, 2e-6]), EMConfig())
        pol = newton_polish(lik, em.theta)
        si = lik.score_info(pol.theta)
    return em, pol, si


def test_recovers_three_component_truth(m3_run):
    res = m3_run[0]
    assert res.status == "converged"
    llu = np.array([h["loglik_unnorm"] for h in res.history])
    assert np.all(np.diff(llu) >= -1e-9 * np.abs(llu[:-1]))  # monotone ascent throughout
    pic, p1, s1, s2, _a = res.theta
    # truth: pic=0.02, p1=0.15, sigmasq=(4e-4, 4e-5), a=1e-6 (20k independent markers)
    assert pic == pytest.approx(0.02, rel=0.25)
    assert p1 == pytest.approx(0.15, rel=0.5)
    assert s1 == pytest.approx(4e-4, rel=0.4)
    assert s2 == pytest.approx(4e-5, rel=0.4)
    assert pic * (p1 * s1 + (1 - p1) * s2) == pytest.approx(
        0.02 * (0.15 * 4e-4 + 0.85 * 4e-5), rel=0.15
    )


def test_safeguards_survive_a_start_that_breaks_the_original(small_data, plan1):
    """sigmasq = 0.5 sends the original through its reset branch; the default must still ascend."""
    start = np.array([0.4, 0.5, 1e-6])
    with Engine(small_data, plan1) as eng:
        lik = CompositeLikelihood(ModelSpec(2), eng, 10)
        res = run_em(lik, start, EMConfig(max_iter=400))
    assert np.all(np.isfinite(res.theta))
    assert res.theta[1] >= 0 and 0 < res.theta[0] < 1
    llu = np.array([h["loglik_unnorm"] for h in res.history])
    assert np.all(np.diff(llu) >= -1e-9 * np.abs(llu[:-1]))


def test_no_stratification_fixes_a_and_reaches_the_constrained_optimum(medium_data):
    from genesis_gwas.compute import resolve_plan

    spec = ModelSpec(2, stratification=False)
    with Engine(medium_data, resolve_plan("cpu", threads=2)) as eng:
        lik = CompositeLikelihood(spec, eng, 10)
        res = run_em(
            lik,
            np.array([0.01, 1e-4, 3e-6]),
            EMConfig(tol_pic=1e-9, tol_sigmasq=1e-12, tol_loglik=1e-12),
        )
        si = lik.score_info(res.theta)
    assert res.theta[2] == 0.0
    total = si.scores.sum(axis=0)
    step = np.linalg.solve(si.information[:2, :2], total[:2])
    # At the constrained optimum the free scores vanish (the score for `a` need not).
    assert abs(step[0]) < 1e-3 * res.theta[0]
    assert abs(step[1]) < 1e-3 * res.theta[1]


def test_m3_proportions_respect_bounds(medium_data):
    from genesis_gwas.compute import resolve_plan

    cfg = EMConfig(p1_bounds=(1e-7, 0.05), max_iter=60)
    with Engine(medium_data, resolve_plan("cpu", threads=2)) as eng:
        lik = CompositeLikelihood(ModelSpec(3), eng, 10)
        res = run_em(lik, np.array([0.01, 0.04, 5e-4, 5e-5, 1e-6]), cfg)
    p1 = np.array([h["theta"][1] for h in res.history])
    assert np.all(p1 <= 0.05 + 1e-15) and np.all(p1 >= 1e-7)
    assert p1[-1] == pytest.approx(
        0.05
    )  # truth (0.15) lies above the bound: stays pinned, no oscillation
    assert np.all(np.diff(p1[-10:]) == 0)


def test_c0_is_raised_only_up_to_c0_max(small_data, plan1):
    with Engine(small_data, plan1) as eng:
        lik = CompositeLikelihood(ModelSpec(2), eng, 3)
        res = run_em(lik, np.array([0.05, 2e-5, 8e-7]), EMConfig(c0=3, c0_max=5, max_iter=300))
    assert 3 <= res.c0 <= 5
    c0s = [h["c0"] for h in res.history]
    assert c0s == sorted(c0s)


def test_infeasible_start_is_rejected(small_data, plan1):
    with Engine(small_data, plan1) as eng:
        lik = CompositeLikelihood(ModelSpec(2), eng, 10)
        with pytest.raises(InputError):
            run_em(lik, np.array([0.01, 5e-5, -1.0]))
        with pytest.raises(InputError):
            run_em(lik, np.array([1.5, 5e-5, 1e-6]))
        with pytest.raises(InputError):
            run_em(lik, np.array([0.01, 5e-5]))


def test_legacy_reports_breakdown_instead_of_nan(small_data, plan1):
    with Engine(small_data, plan1) as eng:
        lik = CompositeLikelihood(ModelSpec(2), eng, 10)
        with pytest.raises(GenesisError):
            run_em(lik, np.array([0.01, 5e-5, -1.0]), EMConfig(legacy=True))


# ------------------------------------------------------------- Newton direction
def test_newton_direction_plain_and_modified():
    g = np.array([1.0, -2.0])
    H = np.array([[-2.0, 0.5], [0.5, -1.0]])
    d, modified = newton_direction(g, H)
    assert not modified
    np.testing.assert_allclose(d, -np.linalg.solve(H, g))
    assert g @ d > 0  # ascent

    H_bad = np.array([[1.0, 0.0], [0.0, -1.0]])  # indefinite
    d, modified = newton_direction(g, H_bad)
    assert modified and g @ d > 0

    d, modified = newton_direction(g, np.full((2, 2), np.nan))
    assert modified and np.all(d == 0)


def test_emconfig_validation():
    with pytest.raises(InputError):
        EMConfig(c0=0)
    with pytest.raises(InputError):
        EMConfig(c0=10, c0_max=5)
    with pytest.raises(InputError):
        EMConfig(tol_pic=0.0)
    with pytest.raises(InputError):
        EMConfig(p1_bounds=(0.5, 0.2))
    loose = EMConfig().loosened()
    assert loose.tol_sigmasq == 1e-7 and loose.tol_a == 1e-7 and loose.tol_loglik == 1e-5


# ------------------------------------------------------------------ polishing
def test_newton_polish_finds_the_root_of_the_score_equations(medium_data):
    from genesis_gwas.compute import resolve_plan

    with Engine(medium_data, resolve_plan("cpu", threads=2)) as eng:
        lik = CompositeLikelihood(ModelSpec(2), eng, 10)
        em = run_em(lik, np.array([0.01, 1e-4, 1e-6]), EMConfig())
        pol = newton_polish(lik, em.theta)
        si = lik.score_info(pol.theta)
        # a much longer EM run must approach the same point
        tight = run_em(
            lik,
            em.theta,
            EMConfig(
                tol_pic=1e-10, tol_sigmasq=1e-13, tol_a=1e-14, tol_loglik=1e-13, max_iter=4000
            ),
        )
    assert pol.status == "converged" and pol.n_steps >= 1 and pol.fixed == ()
    assert pol.loglik_unnorm >= em.loglik_unnorm
    step = np.linalg.solve(si.information, si.scores.sum(axis=0))
    naive_se = np.sqrt(np.diag(np.linalg.inv(si.information)))
    assert np.max(np.abs(step) / naive_se) < 1e-5
    assert np.max(np.abs(tight.theta - pol.theta) / naive_se) < 0.02
    assert np.max(np.abs(tight.theta - pol.theta)) < np.max(np.abs(em.theta - pol.theta))


def test_newton_polish_three_component(m3_run):
    em, pol, si = m3_run
    assert pol.status == "converged" and pol.fixed == ()
    assert pol.loglik_unnorm >= em.loglik_unnorm
    step = np.linalg.solve(si.information, si.scores.sum(axis=0))
    naive_se = np.sqrt(np.diag(np.linalg.inv(si.information)))
    assert np.max(np.abs(step) / naive_se) < 1e-5
    # the EM had stopped short of this point
    assert np.max(np.abs(em.theta - pol.theta) / naive_se) > 1e-3


def test_newton_polish_respects_bounds_and_fixed_parameters(medium_data):
    from genesis_gwas.compute import resolve_plan

    cfg = EMConfig(
        p1_bounds=(1e-7, 0.05), max_iter=80
    )  # need not converge: Newton finishes the job
    with Engine(medium_data, resolve_plan("cpu", threads=4)) as eng:
        lik = CompositeLikelihood(ModelSpec(3), eng, 10)
        em = run_em(lik, np.array([0.01, 0.04, 5e-4, 5e-5, 1e-6]), cfg)
        pol = newton_polish(lik, em.theta, cfg)
        lik0 = CompositeLikelihood(ModelSpec(2, stratification=False), eng, 10)
        em0 = run_em(lik0, np.array([0.01, 1e-4, 0.0]), EMConfig())
        pol0 = newton_polish(lik0, em0.theta)
    assert (
        pol.theta[1] == pytest.approx(0.05) and "p1" in pol.fixed
    )  # truth 0.15 is outside the box
    assert pol.status == "converged" and pol.loglik_unnorm >= em.loglik_unnorm
    assert pol0.theta[2] == 0.0 and pol0.fixed == ("a",) and pol0.status == "converged"


def test_newton_polish_declines_when_information_is_not_positive_definite(small_data, plan1):
    with Engine(small_data, plan1) as eng:
        lik = CompositeLikelihood(ModelSpec(3), eng, 10)
        # equal variance components: the likelihood is flat in p1, so the information is singular
        theta = np.array([0.012, 0.2, 6e-5, 6e-5, 1.2e-6])
        pol = newton_polish(lik, theta)
    assert pol.status == "not_positive_definite" and pol.n_steps == 0
    np.testing.assert_array_equal(pol.theta, theta)
