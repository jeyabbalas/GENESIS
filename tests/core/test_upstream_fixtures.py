"""Parity with outputs of the genuine upstream R/C++ implementation.

The fixtures in ``tests/fixtures/upstream`` were produced by running the original
package (see ``tools/upstream_oracle``). These tests check three layers:

1. the literal Python port used as an oracle elsewhere reproduces upstream,
2. the optimised library kernels reproduce upstream's function-level outputs,
3. ``legacy=True`` reproduces upstream's EM trajectories and the complete ``genesis()``
   post-processing (sandwich covariance, model-selection criteria, derived quantities).
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import genesis_gwas as gg
import tests.upstream_port as up
from genesis_gwas.compute import resolve_plan
from genesis_gwas.core.data import ModelData
from genesis_gwas.core.em import EMConfig, run_em
from genesis_gwas.core.engine import Engine
from genesis_gwas.core.likelihood import CompositeLikelihood
from genesis_gwas.core.spec import ModelSpec
from genesis_gwas.core.variance import Neighborhoods
from tests.helpers import load_fixture

# Scenarios in which upstream reads an uninitialised vector (a start outside the
# proportion bounds of EM_func3) and returns NaN; there is nothing to reproduce.
UNINITIALISED = {"e2_p1_0.6_above_upper_bound", "e3_pic_0.6_above_upper_bound"}

# Three-component start with sigmasq = 0.5: the original's unsafeguarded Newton step lands
# on negative variance components and is then evaluated there, where terms like 1/v^3
# of both signs and enormous magnitude cancel. The result is rounding noise amplified by
# the subsequent resets (the original itself returns a = 0.29 and a heritability SE of 28
# for it on the small data set). Parity is only meaningful for the first iterations.
CHAOTIC = {"d1_bad_start_pic0.4_sig0.5"}


@pytest.fixture(
    scope="module",
    params=["synthetic_small.json", "synthetic_medium.json"],
    ids=["K=300", "K=3000"],
)
def fx(request):
    return load_fixture(request.param)


@pytest.fixture(scope="module")
def data(fx):
    ds = fx["dataset"]
    nb = Neighborhoods(
        indptr=np.array(ds["tagging_gwas_rows0"]["indptr"]),
        indices=np.array(ds["tagging_gwas_rows0"]["indices"]),
    )
    return ModelData(
        betahat=ds["betahat"],
        varbetahat=ds["varbetahat"],
        ldscore=ds["ldscore"],
        nstar=ds["Nstar"],
        n=ds["N"],
        M=ds["M"],
        snp=np.array(ds["SNP"]),
        neighborhoods=nb,
    )


@pytest.fixture(scope="module")
def engine(data):
    with Engine(data, resolve_plan("cpu", threads=1)) as eng:
        yield eng


def _arr(x):
    return np.array(x, dtype=float)


# ------------------------------------------------------------- function level
def test_function_level_two_component(fx, data, engine):
    b, s2, ld, ns = data.betahat, data.varbetahat, data.ldscore, data.nstar
    for case in fx["m2"]["function_cases"]:
        par, c0 = _arr(case["par"]), case["c0"]
        lik = CompositeLikelihood(ModelSpec(2), engine, c0)
        res = lik.em_pass(par)
        w = lik.weights(par)
        si = lik.score_info(par)
        tag = f"par={case['par']} c0={c0}"

        assert res.loglik == pytest.approx(case["loglikelihood"], rel=1e-11, abs=1e-9), tag
        w_ref = _arr(case["weight"])  # some cases store only the leading rows
        np.testing.assert_allclose(w[: len(w_ref)], w_ref, rtol=1e-9, atol=1e-300, err_msg=tag)
        assert res.counts.sum() / ns.sum() == pytest.approx(case["update_pi1"], rel=1e-10), tag
        cur = par[1:3]
        step1 = cur - np.linalg.solve(res.hess, res.grad)
        np.testing.assert_allclose(step1, case["onestep_varcomponent"], rtol=1e-7, err_msg=tag)
        again = lik.em_pass(
            par, eval_sig=step1[:1], eval_a=step1[1], want_e=False, allow_infeasible=True
        )
        step2 = step1 - np.linalg.solve(again.hess, again.grad)
        np.testing.assert_allclose(step2, case["onestep_varcomponent_2nd"], rtol=1e-6, err_msg=tag)
        ss_ref = _arr(case["SS"])
        np.testing.assert_allclose(
            si.scores[: len(ss_ref)], ss_ref, rtol=1e-8, atol=1e-7, err_msg=tag
        )
        np.testing.assert_allclose(
            si.scores.sum(axis=0), case["S"], rtol=1e-8, atol=1e-6, err_msg=tag
        )
        np.testing.assert_allclose(si.information, _arr(case["I"]), rtol=1e-8, err_msg=tag)

        if data.K > 500:
            continue  # the pure-Python port is only exercised on the small data set
        # the literal port agrees with upstream too (it is the oracle of other tests)
        w_port, ll_port = up.weight_loglikelihood(par, b, s2, ld, c0, ns)
        assert ll_port == pytest.approx(case["loglikelihood"], rel=1e-11, abs=1e-9)
        np.testing.assert_allclose(up.I(par, b, s2, ld, c0, ns), _arr(case["I"]), rtol=1e-9)
        np.testing.assert_allclose(
            up.onestep_varcomponent(cur, w_port, b, s2, ld, ns),
            case["onestep_varcomponent"],
            rtol=1e-9,
        )


def test_function_level_three_component(fx, data, engine):
    b, s2, ld, ns = data.betahat, data.varbetahat, data.ldscore, data.nstar
    K = data.K
    for case in fx["m3"]["function_cases"]:
        par, c0 = _arr(case["par"]), case["c0"]
        lik = CompositeLikelihood(ModelSpec(3), engine, c0)
        res = lik.em_pass(par)
        w = lik.weights(par)
        si = lik.score_info(par)
        tag = f"par={case['par']} c0={c0}"

        assert res.loglik == pytest.approx(case["loglikelihood3"], rel=1e-11, abs=1e-9), tag
        w_ref = _arr(case["weight"])
        np.testing.assert_allclose(w[: len(w_ref)], w_ref, rtol=1e-9, atol=1e-300, err_msg=tag)
        np.testing.assert_allclose(
            [res.counts.sum() / ns.sum(), res.counts[0] / res.counts.sum()],
            case["update_p3"],
            rtol=1e-10,
            err_msg=tag,
        )
        cur = par[2:5]
        step1 = cur - np.linalg.solve(res.hess, res.grad)
        np.testing.assert_allclose(step1, case["onestep_varcomponent3"], rtol=1e-6, err_msg=tag)
        again = lik.em_pass(
            par, eval_sig=step1[:2], eval_a=step1[2], want_e=False, allow_infeasible=True
        )
        step2 = step1 - np.linalg.solve(again.hess, again.grad)
        np.testing.assert_allclose(step2, case["onestep_varcomponent3_2nd"], rtol=1e-5, err_msg=tag)
        ss_ref = _arr(case["SS3"])
        np.testing.assert_allclose(
            si.scores[: len(ss_ref)], ss_ref, rtol=1e-8, atol=1e-7, err_msg=tag
        )
        np.testing.assert_allclose(
            si.scores.sum(axis=0) / K, case["S3"], rtol=1e-8, atol=1e-8, err_msg=tag
        )

        upstream_info = _arr(case["I3"])  # asymmetric: the (sigmasq1, pic) entry lacks E[r r']
        assert abs(upstream_info[2, 0] - upstream_info[0, 2]) > 0
        emulated = si.information.copy()
        emulated[2, 0] += si.e_rr[2, 0]
        np.testing.assert_allclose(emulated, upstream_info, rtol=1e-8, err_msg=tag)
        fixed = upstream_info.copy()
        fixed[2, 0] = fixed[0, 2]
        np.testing.assert_allclose(si.information, fixed, rtol=1e-8, err_msg=tag)
        if data.K <= 500:
            np.testing.assert_allclose(up.I3(par, b, s2, ld, c0, ns), upstream_info, rtol=1e-9)


# ------------------------------------------------------------------- EM traces
def _scenarios(fx, model):
    return [s for s in fx[model]["em_scenarios"] if s["label"] not in UNINITIALISED]


def _config(sc, **extra):
    t = sc["tolerances"]
    kw = dict(
        c0=sc["c0_start"],
        max_iter=int(t["maxEM"]),
        newton_steps=int(t["steps"]),
        tol_pic=t["pic"],
        tol_p1=t["p1"],
        tol_sigmasq=t["sigsq1"],
        tol_a=t["a"],
        tol_loglik=t["llk"],
        legacy=True,
    )
    if sc.get("lower_pi"):
        kw["pic_bounds"] = (sc["lower_pi"][0], sc["upper_pi"][0])
        kw["p1_bounds"] = (sc["lower_pi"][1], sc["upper_pi"][1])
    kw.update(extra)
    return EMConfig(**kw)


@pytest.mark.parametrize("model", ["m2", "m3"])
def test_legacy_em_reproduces_upstream_trajectories(fx, data, engine, model):
    """Parameters, truncation level and log-likelihood after every one of the first 60 iterations."""
    P = 3 if model == "m2" else 5
    checked = 0
    for sc in _scenarios(fx, model):
        spec = ModelSpec(sc["modelcomponents"], stratification=sc["stratification"])
        lik = CompositeLikelihood(spec, engine, sc["c0_start"])
        trace = sc["trace"]
        n_run = len(trace) + 1
        res = run_em(lik, _arr(sc["em_start"]), _config(sc, max_iter=n_run))
        chaotic = sc["label"] in CHAOTIC
        for entry in trace:
            m = entry["Meps"]
            want = _arr(entry["result"])
            if int(want[0]) != m - 1 or m > len(res.history) - 1:
                break  # upstream stopped earlier (converged or early break): covered below
            if chaotic and m > 3:
                break
            rec, nxt = res.history[m - 1], res.history[m]
            np.testing.assert_allclose(
                rec["theta"],
                want[2 : 2 + P],
                rtol=1e-7,
                atol=1e-18,
                err_msg=f"{sc['label']} Meps={m}",
            )
            assert rec["c0"] == int(want[2 + P + 1]), f"{sc['label']} Meps={m}: c0"
            if math.isfinite(want[1]) and nxt["c0"] == rec["c0"]:
                assert nxt["loglik"] == pytest.approx(want[1], rel=1e-8, abs=1e-7), (
                    f"{sc['label']} Meps={m}"
                )
            checked += 1
    assert checked > 200


@pytest.mark.parametrize("model", ["m2", "m3"])
def test_legacy_em_reproduces_upstream_stopping_point(fx, data, engine, model):
    P = 3 if model == "m2" else 5
    for sc in _scenarios(fx, model):
        if sc["label"] in CHAOTIC:
            continue
        spec = ModelSpec(sc["modelcomponents"], stratification=sc["stratification"])
        lik = CompositeLikelihood(spec, engine, sc["c0_start"])
        res = run_em(lik, _arr(sc["em_start"]), _config(sc))
        want = _arr(sc["converged"]["result"])
        label = sc["label"]
        assert res.n_iter - 1 == int(want[0]) or res.status == "legacy_break", label
        np.testing.assert_allclose(res.theta, want[2 : 2 + P], rtol=1e-7, atol=1e-18, err_msg=label)
        assert res.c0 == int(want[2 + P + 1]), label
        assert res.loglik == pytest.approx(want[1], rel=1e-8), label


# ------------------------------------------------------ complete genesis() output
@pytest.mark.parametrize("model", ["m2", "m3"])
def test_legacy_fit_reproduces_upstream_genesis_output(fx, data, model):
    n_checked = 0
    for sc in _scenarios(fx, model):
        if sc["label"] in CHAOTIC:
            continue
        g = sc["genesis"]
        label = sc["label"]
        kwargs = dict(
            stratification=sc["stratification"],
            em=_config(sc),
            legacy=True,
            threads=1,
            bic_gamma=g["BIC_gamma"],
        )
        if isinstance(sc["starting_argument"], list):
            kwargs["starting"] = _arr(sc["starting_argument"])
        else:  # "NA" upstream: LD-score-regression start with the given startingpic
            kwargs["starting_pic"] = sc["startingpic"]
        d = ModelData(
            betahat=data.betahat,
            varbetahat=data.varbetahat,
            ldscore=data.ldscore,
            nstar=data.nstar,
            n=data.n,
            M=sc["M"],
            neighborhoods=data.neighborhoods,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = gg.fit(d, sc["modelcomponents"], **kwargs)

        np.testing.assert_allclose(res.starting, g["starting"], rtol=1e-9, err_msg=label)
        np.testing.assert_allclose(res.estimates, g["est"], rtol=1e-7, atol=1e-18, err_msg=label)
        assert res.c0 == g["c0_final"], label
        assert res.loglik == pytest.approx(g["llk"], rel=1e-10), label
        assert res.mean_n == pytest.approx(g["mean_n"], rel=1e-12), label

        # Information, "meat", covariance (asymmetric for the three-component model, as
        # upstream), standard errors, effective number of parameters, criteria.
        np.testing.assert_allclose(res.information, _arr(g["m_I"]), rtol=1e-6, err_msg=label)
        np.testing.assert_allclose(
            res.J, _arr(g["J"]), rtol=1e-6, atol=1e-9 * np.abs(_arr(g["J"])).max(), err_msg=label
        )
        np.testing.assert_allclose(res.covariance, _arr(g["var_est"]), rtol=1e-5, err_msg=label)
        ok = np.isfinite(_arr(g["sd_est"]))  # upstream reports NaN for negative variance estimates
        np.testing.assert_allclose(res.se[ok], _arr(g["sd_est"])[ok], rtol=1e-6, err_msg=label)
        assert np.all(np.isnan(res.se[~ok])), label
        assert res.ds == pytest.approx(g["ds"], rel=1e-7), label
        assert res.bic == pytest.approx(g["bic"], rel=1e-8), label
        assert res.aic == pytest.approx(g["aic"], rel=1e-8), label
        pairs = [
            ("n_ssnp", "causalnum", "sd_causalnum"),
            ("heritability", "heritability", "sd_heritability"),
            ("sibling_risk", "risk", "sd_risk"),
        ]
        if model == "m3":
            pairs += [
                ("n_ssnp_cluster1", "largenum", "sd_largenum"),
                ("heritability_cluster1", "heritlarge", "sd_heritlarge"),
                ("heritability_cluster2", "heritsmall", "sd_heritsmall"),
            ]
        for key, est, sd in pairs:
            assert res.derived[key].estimate == pytest.approx(g[est], rel=1e-7), (label, key)
            if isinstance(g[sd], float) and math.isfinite(g[sd]):
                assert res.derived[key].se == pytest.approx(g[sd], rel=1e-6), (label, key)
        n_checked += 1
    assert n_checked >= 9
