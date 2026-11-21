"""The optimised kernels must agree with a literal port of the original C++ code."""

from __future__ import annotations

import numpy as np
import pytest
import tests.upstream_port as up

from genesis_gwas.core.engine import Engine
from genesis_gwas.core.likelihood import CompositeLikelihood
from genesis_gwas.core.spec import ModelSpec

M2_PARS = [
    np.array([0.01, 5e-5, 1e-6]),
    np.array([0.002, 2e-4, 0.0]),
    np.array([0.05, 1e-5, -2e-6]),
    np.array([0.3, 1e-6, 5e-6]),
]
M3_PARS = [
    np.array([0.01, 1 / 9, 2.5e-4, 2.5e-5, 1e-6]),
    np.array([0.02, 0.3, 1e-4, 1e-5, 0.0]),
    np.array([0.005, 0.05, 1e-3, 5e-5, 2e-6]),
]
RTOL = 2e-9


def _args(d):
    return d.betahat, d.varbetahat, d.ldscore


@pytest.mark.parametrize("c0", [10, 3, 20])
@pytest.mark.parametrize("par", M2_PARS, ids=lambda p: f"pic={p[0]}")
def test_m2_matches_port(small_data, plan1, par, c0):
    d = small_data
    b, s2, ld = _args(d)
    with Engine(d, plan1) as eng:
        lik = CompositeLikelihood(ModelSpec(2), eng, c0)
        res = lik.em_pass(par)
        w = lik.weights(par)
        si = lik.score_info(par)

    w_ref, llk_ref = up.weight_loglikelihood(par, b, s2, ld, c0, d.nstar)
    assert res.loglik == pytest.approx(llk_ref, rel=1e-12, abs=1e-8)
    np.testing.assert_allclose(w, w_ref, rtol=RTOL, atol=1e-300)
    np.testing.assert_allclose(w.sum(axis=1), 1.0, rtol=1e-13)

    # closed-form proportion update
    pi_new = res.counts.sum() / d.nstar.sum()
    assert pi_new == pytest.approx(up.update_pi1(w_ref, d.nstar), rel=RTOL)

    # one Newton step for (sigmasq, a)
    step = np.array([par[1], par[2]]) - np.linalg.solve(res.hess, res.grad)
    ref = up.onestep_varcomponent(np.array([par[1], par[2]]), w_ref, b, s2, ld, d.nstar)
    np.testing.assert_allclose(step, ref, rtol=1e-7)

    np.testing.assert_allclose(si.scores, up.SS(par, b, s2, ld, c0, d.nstar), rtol=1e-8, atol=1e-6)
    info_ref = up.I(par, b, s2, ld, c0, d.nstar)
    np.testing.assert_allclose(si.information, info_ref, rtol=1e-8)


@pytest.mark.parametrize("c0", [10, 3, 5])
@pytest.mark.parametrize("par", M3_PARS, ids=lambda p: f"pic={p[0]}")
def test_m3_matches_port(small_data, plan1, par, c0):
    d = small_data
    b, s2, ld = _args(d)
    with Engine(d, plan1) as eng:
        lik = CompositeLikelihood(ModelSpec(3), eng, c0)
        res = lik.em_pass(par)
        w = lik.weights(par)
        si = lik.score_info(par)

    w_ref, llk_ref = up.weight_loglikelihood3(par, b, s2, ld, c0, d.nstar)
    assert res.loglik == pytest.approx(llk_ref, rel=1e-12, abs=1e-8)
    np.testing.assert_allclose(w, w_ref, rtol=RTOL, atol=1e-300)

    p_ref = up.update_p3(w_ref, d.nstar)
    assert res.counts.sum() / d.nstar.sum() == pytest.approx(p_ref[0], rel=RTOL)
    assert res.counts[0] / res.counts.sum() == pytest.approx(p_ref[1], rel=RTOL)

    cur = par[2:5]
    step = cur - np.linalg.solve(res.hess, res.grad)
    ref = up.onestep_varcomponent3(cur, w_ref, b, s2, ld, d.nstar)
    np.testing.assert_allclose(step, ref, rtol=1e-6)

    np.testing.assert_allclose(si.scores, up.SS3(par, b, s2, ld, c0, d.nstar), rtol=1e-8, atol=1e-6)
    # Correct (symmetric) information equals the port with the typo repaired ...
    np.testing.assert_allclose(
        si.information, up.I3(par, b, s2, ld, c0, d.nstar, reproduce_typo=False), rtol=1e-8
    )
    # ... and adding E[r r'] at (sigmasq1, pic) reproduces the original's asymmetric matrix.
    legacy = si.information.copy()
    legacy[2, 0] += si.e_rr[2, 0]
    np.testing.assert_allclose(
        legacy, up.I3(par, b, s2, ld, c0, d.nstar, reproduce_typo=True), rtol=1e-8
    )


def test_second_newton_step_matches_port(small_data, plan1):
    """Derivatives of Q evaluated away from the weight parameters (second Newton step)."""
    d = small_data
    b, s2, ld = _args(d)
    par = M3_PARS[0]
    with Engine(d, plan1) as eng:
        lik = CompositeLikelihood(ModelSpec(3), eng, 10)
        first = lik.em_pass(par)
        cur = par[2:5] - np.linalg.solve(first.hess, first.grad)
        second = lik.em_pass(par, eval_sig=cur[:2], eval_a=cur[2], want_e=False)
    w_ref, _ = up.weight_loglikelihood3(par, b, s2, ld, 10, d.nstar)
    ref1 = up.onestep_varcomponent3(par[2:5], w_ref, b, s2, ld, d.nstar)
    np.testing.assert_allclose(cur, ref1, rtol=1e-6)
    ref2 = up.onestep_varcomponent3(ref1, w_ref, b, s2, ld, d.nstar)
    np.testing.assert_allclose(cur - np.linalg.solve(second.hess, second.grad), ref2, rtol=1e-6)


def test_unnormalised_and_renormalised_loglik(small_data, plan1):
    """loglik = loglik_unnorm - sum_k log(truncated prior mass)."""
    d = small_data
    par = np.array([0.05, 1e-5, 1e-6])
    with Engine(d, plan1) as eng:
        lik = CompositeLikelihood(ModelSpec(2), eng, 3)
        ll, llu = lik.loglik(par)
        trunc = lik.truncation(par)
    assert ll - llu == pytest.approx(trunc.log_mass_deficit, rel=1e-10)
    assert trunc.log_mass_deficit > 0
    assert 0 < trunc.mean_tail < trunc.max_tail <= 1
    # brute force for one marker
    k = int(np.argmax(d.nstar))
    mass = sum(up.cdbinom(j, int(d.nstar[k]), par[0]) for j in range(4))
    assert trunc.max_tail == pytest.approx(1 - mass, rel=1e-9)
