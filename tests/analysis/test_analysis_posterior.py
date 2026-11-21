"""Posterior null probabilities and shrinkage estimates (port of upstream postmarginal)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.special import comb

import genesis_gwas as gg
from genesis_gwas.analysis.posterior import posterior_marginal
from genesis_gwas.exceptions import InputError


def r_postmarginal(data: gg.ModelData, est: np.ndarray, c0: int = 10, a: float = 0.0):
    """Line-by-line transcription of upstream postmarginal().

    ``a`` is not part of the upstream code (it uses ``varbetahat`` alone); passing the
    fitted value gives the model-consistent variant for comparison.
    """
    betahat, ldscore = data.betahat, data.ldscore
    varbetahat = data.varbetahat + a
    Nstar = data.nstar.astype(float)
    pic = est[0]
    q0 = (
        (1 - pic) ** Nstar
        / np.sqrt(2 * np.pi * varbetahat)
        * np.exp(-0.5 * betahat**2 / varbetahat)
    )
    beta_exp = np.zeros_like(betahat)
    qsum = q0.copy()
    if len(est) == 3:
        for i in range(1, c0 + 1):
            qtemp = np.zeros_like(betahat)
            delta = i * est[1] * ldscore / Nstar
            ind = Nstar >= i
            qtemp[ind] = (
                comb(Nstar[ind], i)
                * (1 - pic) ** (Nstar[ind] - i)
                * pic**i
                / np.sqrt(2 * np.pi * (delta[ind] + varbetahat[ind]))
                * np.exp(-0.5 * betahat[ind] ** 2 / (delta[ind] + varbetahat[ind]))
            )
            qsum += qtemp
            beta_exp += qtemp * (1 / varbetahat) * betahat / (1 / delta + 1 / varbetahat)
    else:
        p1 = est[1]
        for i in range(c0 + 1):
            for j in range(c0 + 1):
                if i == 0 and j == 0:
                    continue
                qtemp = np.zeros_like(betahat)
                delta = (i * est[2] + j * est[3]) * ldscore / Nstar
                ind = Nstar >= (i + j)
                qtemp[ind] = (
                    comb(Nstar[ind], i)
                    * comb(Nstar[ind] - i, j)
                    * (1 - pic) ** (Nstar[ind] - i - j)
                    * (p1 * pic) ** i
                    * ((1 - p1) * pic) ** j
                    / np.sqrt(2 * np.pi * (delta[ind] + varbetahat[ind]))
                    * np.exp(-0.5 * betahat[ind] ** 2 / (delta[ind] + varbetahat[ind]))
                )
                qsum += qtemp
                beta_exp += qtemp * (1 / varbetahat) * betahat / (1 / delta + 1 / varbetahat)
    return q0 / qsum, beta_exp / qsum


@pytest.fixture(scope="module")
def toy() -> gg.ModelData:
    """80 markers with awkward cases: N* below c0, z = 0, large |z|."""
    rng = np.random.default_rng(7701)
    K = 80
    nstar = np.concatenate(
        [[1, 1, 2, 3, 5, 9, 10, 11, 12, 20, 313, 710], rng.integers(2, 200, K - 12)]
    )
    ld = np.maximum(1.0, nstar * rng.uniform(0.15, 0.7, K))
    ld[nstar == 1] = 1.0
    n = np.round(rng.uniform(5e4, 1.5e5, K))
    z = np.concatenate(
        [rng.standard_normal(K - 6), [np.sqrt(79.9), -np.sqrt(79.9), 0.0, 5.0, -5.0, 3.0]]
    )
    snp = np.array([f"rs{i + 1}" for i in range(K)])
    return gg.ModelData.from_z(z=z, n=n, ldscore=ld, nstar=nstar, M=1070777, snp=snp)


THETAS = {
    "m2": [4.906797e-03, 5.733005e-05, 1.861491e-06],
    "m2_negative_a": [0.02, 1.2e-5, -8e-7],
    "m3": [8.899809e-03, 9.476025e-02, 1.458650e-04, 2.227118e-05, 1.567643e-06],
    "m3_zero_a": [0.015, 0.3, 4e-4, 3e-5, 0.0],
}


@pytest.mark.parametrize("name", list(THETAS))
@pytest.mark.parametrize("c0", [10, 5])
def test_matches_upstream_transcription_without_inflation(toy, make_fit, name, c0):
    theta = np.array(THETAS[name])
    fit = make_fit(theta, c0=c0)
    out = posterior_marginal(toy, fit, include_inflation=False, device="cpu", threads=1)
    p0, mean = r_postmarginal(toy, theta, c0=c0)
    np.testing.assert_allclose(out["p_null"], p0, rtol=1e-10, atol=1e-300)
    np.testing.assert_allclose(out["post_mean"], mean, rtol=1e-10, atol=1e-18)


@pytest.mark.parametrize("name", ["m2", "m3"])
def test_default_includes_inflation(toy, make_fit, name):
    theta = np.array(THETAS[name])
    fit = make_fit(theta)
    out = posterior_marginal(toy, fit, device="cpu", threads=1)
    p0, mean = r_postmarginal(toy, theta, a=theta[-1])
    np.testing.assert_allclose(out["p_null"], p0, rtol=1e-10)
    np.testing.assert_allclose(out["post_mean"], mean, rtol=1e-10, atol=1e-18)
    legacy = posterior_marginal(toy, fit, include_inflation=False, device="cpu", threads=1)
    assert not np.allclose(out["p_null"], legacy["p_null"], rtol=1e-6)
    # a > 0 means more noise: for markers with real evidence of association the
    # model-consistent version is less confident (larger lfdr) and shrinks more
    strong = np.abs(toy.z) >= 2
    assert strong.sum() >= 5
    assert np.all(out["p_null"][strong] > legacy["p_null"][strong])
    assert np.all(np.abs(out["post_mean"][strong]) < np.abs(legacy["post_mean"][strong]))


def test_inflation_flag_is_irrelevant_when_a_is_zero(toy, make_fit):
    fit = make_fit(THETAS["m3_zero_a"])
    a = posterior_marginal(toy, fit, include_inflation=True, device="cpu", threads=1)
    b = posterior_marginal(toy, fit, include_inflation=False, device="cpu", threads=1)
    pd.testing.assert_frame_equal(a, b)


def test_frame_layout_and_basic_properties(toy, make_fit):
    fit = make_fit(THETAS["m3"])
    out = posterior_marginal(toy, fit, device="cpu", threads=1)
    assert list(out.columns) == ["snp", "z", "p_null", "post_mean", "post_sd", "expected_count"]
    assert len(out) == toy.K and out["snp"].tolist() == toy.snp.tolist()
    np.testing.assert_allclose(out["z"], toy.z, rtol=1e-15)
    assert out["p_null"].between(0, 1).all()
    assert (out["post_sd"] >= 0).all() and (out["expected_count"] >= 0).all()
    nonzero = toy.betahat != 0
    assert (np.sign(out["post_mean"][nonzero]) == np.sign(toy.betahat[nonzero])).all()
    assert (np.abs(out["post_mean"]) <= np.abs(toy.betahat) + 1e-18).all()
    zero = out[np.asarray(toy.betahat) == 0.0]
    assert (zero["post_mean"] == 0.0).all()
    # expected number of tagged susceptibility SNPs is at least Pr(not null)
    assert (out["expected_count"] >= 1 - out["p_null"] - 1e-12).all()


def test_stronger_association_is_less_likely_null(make_fit):
    z = np.linspace(0.0, 8.0, 33)
    K = z.size
    data = gg.ModelData.from_z(
        z=z, n=np.full(K, 1e5), ldscore=np.full(K, 15.0), nstar=np.full(K, 40), M=1070777
    )
    out = posterior_marginal(data, make_fit(THETAS["m2"]), device="cpu", threads=1)
    assert "snp" not in out.columns
    assert np.all(np.diff(out["p_null"]) < 0)
    assert np.all(np.diff(out["post_mean"]) > 0)
    shrink = out["post_mean"].to_numpy()[1:] / data.betahat[1:]
    assert np.all(np.diff(shrink) > 0) and shrink[-1] < 1


def test_c0_defaults_to_fit_and_can_be_overridden(toy, make_fit):
    theta = np.array(THETAS["m2"])
    at5 = posterior_marginal(toy, make_fit(theta, c0=5), device="cpu", threads=1)
    forced = posterior_marginal(toy, make_fit(theta, c0=10), c0=5, device="cpu", threads=1)
    pd.testing.assert_frame_equal(at5, forced)
    at2 = posterior_marginal(toy, make_fit(theta), c0=2, device="cpu", threads=1)
    assert not np.allclose(at2["expected_count"], at5["expected_count"], rtol=1e-8)
    with pytest.raises(InputError):
        posterior_marginal(toy, make_fit(theta), c0=0)
    with pytest.raises(InputError):
        posterior_marginal(toy, theta)  # a raw vector carries no truncation level


def test_results_do_not_depend_on_threads_or_backend(toy, make_fit):
    fit = make_fit(THETAS["m3"])
    a = posterior_marginal(toy, fit, device="cpu", threads=1)
    b = posterior_marginal(toy, fit, device="cpu", threads=3)
    for col in ("p_null", "post_mean", "post_sd", "expected_count"):
        np.testing.assert_allclose(a[col], b[col], rtol=1e-12, atol=1e-300)


def test_real_data_summary(mini_data, mini_fit):
    out = posterior_marginal(mini_data, mini_fit, threads=2)
    assert len(out) == mini_data.K
    assert out["snp"].iloc[0].startswith("rs")
    top = out.loc[out["z"].abs().idxmax()]
    assert top["p_null"] < 0.01
    weak = out.loc[out["z"].abs().idxmin()]
    assert weak["p_null"] > top["p_null"]
    assert abs(top["post_mean"]) < abs(mini_data.betahat[out["z"].abs().idxmax()])
