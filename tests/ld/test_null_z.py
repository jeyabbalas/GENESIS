"""null_z: LD-correlated null association statistics of the reference panel."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from genesis_gwas.exceptions import InputError
from genesis_gwas.ld.build import null_z, standardize_genotypes


@pytest.fixture(scope="module")
def panel(simulate_genotypes):
    rng = np.random.default_rng(99)
    geno = simulate_genotypes(rng, 40, 300)
    geno[::2] = 2 - geno[::2]  # count the other allele at every second SNP -> negative r
    return geno


def test_shape_dtype_and_determinism(panel):
    a = null_z(panel, 7, seed=1)
    assert a.shape == (7, 40) and a.dtype == np.float32
    np.testing.assert_array_equal(a, null_z(panel, 7, seed=1))
    assert not np.array_equal(a, null_z(panel, 7, seed=2))
    # chunking is an implementation detail
    np.testing.assert_array_equal(a, null_z(panel, 7, seed=1, chunk=3))


def test_default_is_the_regression_t_statistic(panel):
    rng = np.random.default_rng(0)
    n = panel.shape[1]
    Y = rng.standard_normal((4, n))
    t = null_z(panel, phenotypes=Y)
    # independent computation: ordinary least squares of y on each SNP, one at a time
    for rep in range(Y.shape[0]):
        for k in (0, 7, 23, 39):
            res = stats.linregress(panel[k].astype(float), Y[rep])
            np.testing.assert_allclose(t[rep, k], res.slope / res.stderr, rtol=2e-5)
    # studentised: neither the location nor the scale of the phenotype matters
    np.testing.assert_allclose(null_z(panel, phenotypes=5.0 * Y + 3.0), t, rtol=1e-5, atol=1e-5)


def test_projection_is_exactly_the_projected_phenotype(panel):
    rng = np.random.default_rng(0)
    Y = rng.standard_normal((5, panel.shape[1]))
    u = null_z(panel, phenotypes=Y, statistic="projection")
    Z = standardize_genotypes(panel)
    np.testing.assert_allclose(u, (Z @ Y.T).T, rtol=1e-6, atol=1e-6)
    # genotypes are centred, so the phenotype's mean is irrelevant, but its scale is not
    np.testing.assert_allclose(
        null_z(panel, phenotypes=Y + 3.0, statistic="projection"), u, atol=1e-4
    )
    np.testing.assert_allclose(
        null_z(panel, phenotypes=2.0 * Y, statistic="projection"), 2.0 * u, rtol=1e-5, atol=1e-5
    )
    # equivalently sqrt(n) times the covariance with variance-standardised genotypes
    n = panel.shape[1]
    X = (panel - panel.mean(1, keepdims=True)) / panel.std(1, keepdims=True)
    np.testing.assert_allclose(u, (X @ Y.T).T / np.sqrt(n), rtol=1e-5, atol=1e-5)


def test_normal_scores_are_a_monotone_map_of_t_with_normal_margins(panel):
    n = panel.shape[1]
    t = null_z(panel, 3000, seed=5).astype(np.float64)
    z = null_z(panel, 3000, seed=5, statistic="normal").astype(np.float64)
    np.testing.assert_allclose(z, stats.norm.isf(stats.t.sf(t, n - 2)), atol=2e-5)
    assert np.all(np.sign(z) == np.sign(t)) and np.all(np.abs(z) <= np.abs(t) + 1e-6)
    assert abs(z.var() - 1.0) < 0.02
    assert abs(t.var() - (n - 2) / (n - 4)) < 0.02


@pytest.mark.parametrize("statistic", ["t", "normal", "projection"])
def test_correlation_structure_is_the_sample_ld_matrix(panel, statistic):
    u = null_z(panel, 40000, seed=3, statistic=statistic).astype(np.float64)
    R = np.corrcoef(panel.astype(float))
    emp = (u.T @ u) / u.shape[0]
    assert np.max(np.abs(emp - R)) < 0.05
    assert np.max(np.abs(u.mean(axis=0))) < 0.03
    np.testing.assert_allclose(np.diag(emp), 1.0, atol=0.05)
    # the correlation carries the SIGN of r, not just its magnitude
    off = ~np.eye(R.shape[0], dtype=bool)
    strong = off & (np.abs(R) > 0.3)
    assert strong.any() and (R[strong] < 0).any()
    assert np.all(np.sign(emp[strong]) == np.sign(R[strong]))


def test_studentising_removes_the_shared_scale_of_a_replicate():
    # Many independent SNPs: the mean chi-square of a replicate is then (essentially) the
    # sample variance of its phenotype for the raw projection, and one for the t statistic.
    rng = np.random.default_rng(8)
    n, m = 60, 4000
    geno = rng.integers(0, 3, size=(m, n)).astype(np.int8)
    Y = rng.standard_normal((400, n))
    raw = (null_z(geno, phenotypes=Y, statistic="projection").astype(float) ** 2).mean(axis=1)
    stu = (null_z(geno, phenotypes=Y, statistic="normal").astype(float) ** 2).mean(axis=1)
    np.testing.assert_allclose(raw, Y.var(axis=1, ddof=1), rtol=0.1)
    assert raw.std() > 4 * stu.std()
    assert abs(stu.mean() - 1.0) < 0.01


def test_same_phenotypes_across_chromosomes_give_one_genome_wide_gwas(panel):
    rng = np.random.default_rng(4)
    Y = rng.standard_normal((3, panel.shape[1]))
    whole = null_z(panel, phenotypes=Y)
    parts = np.concatenate(
        [null_z(panel[:15], phenotypes=Y), null_z(panel[15:], phenotypes=Y)], axis=1
    )
    np.testing.assert_array_equal(whole, parts)


def test_missing_genotypes(panel):
    g = panel.copy()
    g[0, :10] = -1
    u = null_z(g, 4, seed=0)
    assert np.all(np.isfinite(u))


def test_validation(panel):
    with pytest.raises(InputError):
        null_z(panel)
    with pytest.raises(InputError):
        null_z(panel, 0)
    with pytest.raises(InputError):
        null_z(panel, phenotypes=np.zeros((2, 5)))
    with pytest.raises(InputError):
        null_z(panel[0], 2)
    with pytest.raises(InputError):
        null_z(panel, 2, statistic="wald")
    with pytest.raises(InputError):
        null_z(panel, phenotypes=np.ones((1, panel.shape[1])))
    with pytest.raises(InputError):
        null_z(panel[:, :4], 2)
