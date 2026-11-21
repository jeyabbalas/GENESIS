"""Sandwich estimator, neighbourhood sums and information criteria."""

from __future__ import annotations

import math

import numpy as np
import pytest

import tests.upstream_port as up
from genesis_gwas.core.variance import (
    Neighborhoods,
    delta_method,
    information_criteria,
    neighborhood_sums,
    sandwich,
    scaled_inverse,
)
from genesis_gwas.exceptions import InputError, NumericalWarning


def _random_symmetric_neighbors(n: int, rng: np.random.Generator, width: int = 6) -> list[list[int]]:
    """Self plus random nearby partners; symmetric like real tagging neighbourhoods."""
    sets = [{i} for i in range(n)]
    for i in range(n):
        for j in rng.integers(max(0, i - width), min(n, i + width + 1), size=rng.integers(0, 4)):
            sets[i].add(int(j))
            sets[int(j)].add(i)
    return [sorted(s) for s in sets]


def test_neighborhood_sums_match_brute_force():
    rng = np.random.default_rng(0)
    n_ref, P = 300, 5
    lists = _random_symmetric_neighbors(n_ref, rng)
    nb = Neighborhoods.from_lists(lists)

    # all reference SNPs present
    S = rng.normal(size=(n_ref, P))
    brute = np.array([S[l].sum(axis=0) for l in lists])
    np.testing.assert_allclose(neighborhood_sums(S, nb), brute, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(neighborhood_sums(S, nb, block=7), brute, rtol=1e-12, atol=1e-12)

    # only a subset of reference SNPs is in the GWAS; absent neighbours contribute nothing
    ref_index = np.sort(rng.choice(n_ref, size=180, replace=False))
    pos = {int(r): k for k, r in enumerate(ref_index)}
    Sg = rng.normal(size=(ref_index.size, P))
    brute = np.array(
        [sum((Sg[pos[m]] for m in lists[int(r)] if m in pos), np.zeros(P)) for r in ref_index]
    )
    np.testing.assert_allclose(neighborhood_sums(Sg, nb, ref_index), brute, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(neighborhood_sums(Sg, nb, ref_index, block=11), brute, rtol=1e-12, atol=1e-12)


def test_neighborhood_sums_handles_empty_rows_and_validates():
    nb = Neighborhoods.from_lists([[0, 1], [], [2], [1, 3]])
    S = np.arange(8.0).reshape(4, 2)
    out = neighborhood_sums(S, nb)
    np.testing.assert_allclose(out, [[2.0, 4.0], [0.0, 0.0], [4.0, 5.0], [8.0, 10.0]])
    with pytest.raises(InputError):
        neighborhood_sums(S[:3], nb)  # ref_index required when sizes differ
    with pytest.raises(InputError):
        neighborhood_sums(S[:2], nb, np.array([0, 0]))  # duplicated reference rows
    with pytest.raises(InputError):
        neighborhood_sums(S[:2], nb, np.array([0, 9]))  # outside the panel
    with pytest.raises(InputError):
        Neighborhoods(indptr=np.array([0, 3]), indices=np.array([0]))


def test_sandwich_conventions_and_port():
    rng = np.random.default_rng(1)
    K, P = 120, 3
    S = rng.normal(size=(K, P)) * np.array([1.0, 1e3, 1e5])
    A = rng.normal(size=(P, P))
    info = (A @ A.T + P * np.eye(P)) * np.outer([1.0, 1e3, 1e5], [1.0, 1e3, 1e5])
    lists = _random_symmetric_neighbors(K, rng)
    nb = Neighborhoods.from_lists(lists)

    paper = sandwich(S, info, nb)
    legacy = sandwich(S, info, nb, self_double_count=True)
    ref = up.sandwich_upstream(S, info, lists)

    # legacy == literal port of genesis.R (which adds the marker's own score twice)
    np.testing.assert_allclose(legacy.J, ref["J"], rtol=1e-10)
    np.testing.assert_allclose(legacy.covariance, ref["var_est"], rtol=1e-8)
    assert legacy.ds == pytest.approx(ref["ds"], rel=1e-9)
    # paper formula: J_legacy = J_paper + S'S
    np.testing.assert_allclose(legacy.J - paper.J, S.T @ S, rtol=1e-9)
    # symmetric neighbourhoods give a symmetric J and covariance
    np.testing.assert_allclose(paper.J, paper.J.T, rtol=1e-9)
    np.testing.assert_allclose(paper.covariance, paper.covariance.T, rtol=1e-9)
    assert paper.information_pd and np.all(paper.se > 0)

    # independence: J = S'S; "double count" then doubles it
    indep = sandwich(S, info, None)
    np.testing.assert_allclose(indep.J, S.T @ S, rtol=1e-12)
    np.testing.assert_allclose(sandwich(S, info, Neighborhoods.identity(K)).J, S.T @ S, rtol=1e-12)


def test_sandwich_free_mask_zeroes_fixed_parameters():
    rng = np.random.default_rng(2)
    S = rng.normal(size=(50, 3))
    info = np.diag([4.0, 9.0, 16.0])
    out = sandwich(S, info, None, free=np.array([True, True, False]))
    assert np.all(out.covariance[2] == 0) and np.all(out.covariance[:, 2] == 0)
    sub = sandwich(S[:, :2], info[:2, :2], None)
    np.testing.assert_allclose(out.covariance[:2, :2], sub.covariance)
    assert out.ds == pytest.approx(sub.ds)


def test_sandwich_warns_when_information_is_not_positive_definite():
    S = np.random.default_rng(3).normal(size=(40, 2))
    with pytest.warns(NumericalWarning, match="not positive definite"):
        res = sandwich(S, np.array([[1.0, 2.0], [2.0, 1.0]]), None)
    assert not res.information_pd


def test_scaled_inverse_handles_extreme_scaling():
    d = np.array([1e-9, 1.0, 1e12])
    core = np.array([[2.0, 0.3, 0.1], [0.3, 1.5, 0.2], [0.1, 0.2, 1.0]])
    A = core * np.outer(d, d)
    inv, cond = scaled_inverse(A)
    np.testing.assert_allclose(inv, np.linalg.inv(core) / np.outer(d, d), rtol=1e-12)
    assert cond < 10  # the scaled problem is perfectly well conditioned
    assert np.linalg.cond(A) > 1e30  # ... although the raw matrix looks hopeless
    inv, cond = scaled_inverse(np.zeros((2, 2)))
    assert np.all(np.isnan(inv)) and math.isinf(cond)


def test_information_criteria_formula():
    llk, ds, n, gamma = 1234.5, 7.25, 98765.4, 0.5
    ic = information_criteria(llk, ds, n, gamma)
    assert ic["aic"] == pytest.approx(-2 * llk + 2 * ds)
    # literal transcription of genesis.R
    assert ic["bic"] == pytest.approx(-2 * llk + ds * math.log(n) + 2 * gamma * math.log(5**ds))
    with pytest.raises(InputError):
        information_criteria(llk, ds, n, 1.5)


def test_delta_method():
    cov = np.array([[4.0, 1.0], [1.0, 9.0]])
    assert delta_method(np.array([1.0, 0.0]), cov) == pytest.approx(2.0)
    assert delta_method(np.array([1.0, 1.0]), cov) == pytest.approx(math.sqrt(15.0))
    assert math.isnan(delta_method(np.array([1.0, -1.0]), np.array([[1.0, 2.0], [2.0, 1.0]])))
