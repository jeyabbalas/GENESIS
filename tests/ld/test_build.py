"""compute_ld: N*, tagging sets and LD scores against an independent brute-force implementation."""

from __future__ import annotations

import numpy as np
import pytest

from genesis_gwas.exceptions import BackendError, DataError, InputError
from genesis_gwas.ld.build import (
    LDTable,
    _round_sig,
    allele_frequencies,
    compute_ld,
    standardize_genotypes,
)


def _rows(table: LDTable) -> list[list[int]]:
    return [
        table.indices[table.indptr[k] : table.indptr[k + 1]].tolist() for k in range(table.n_snps)
    ]


# --------------------------------------------------------------------------- definitions
@pytest.mark.parametrize(
    ("window_kb", "r2_min"), [(1000.0, 0.1), (500.0, 0.2), (2000.0, 0.05), (250.0, 0.5)]
)
def test_matches_brute_force(toy_panel, brute_force_ld, window_kb, r2_min):
    geno, chrom, pos = toy_panel
    (table,) = compute_ld(geno, pos, chrom, configs=[(window_kb, r2_min)])
    sets, nstar, ld, r2 = brute_force_ld(geno, pos, chrom, window_kb, r2_min)
    assert (
        table.window_kb == window_kb and table.r2_min == r2_min and table.n_samples == geno.shape[1]
    )
    np.testing.assert_array_equal(table.nstar, nstar)
    assert _rows(table) == sets
    np.testing.assert_allclose(table.ldscore, ld, rtol=0, atol=1e-10)
    # stored r2 are the brute-force values, aligned with indices
    rows = np.repeat(np.arange(table.n_snps), table.nstar)
    np.testing.assert_allclose(
        table.r2, np.where(rows == table.indices, 1.0, r2[rows, table.indices]), atol=1e-12
    )
    assert table.nstar.dtype == np.int32 and table.indices.dtype == np.int32
    assert table.indptr.dtype == np.int64 and table.ldscore.dtype == np.float64
    assert table.nnz == int(nstar.sum())


def test_structure_invariants(toy_panel):
    geno, chrom, pos = toy_panel
    (t,) = compute_ld(geno, pos, chrom, configs=[(1000.0, 0.1)])
    m = t.n_snps
    np.testing.assert_array_equal(np.diff(t.indptr), t.nstar)
    pairs = set()
    for k, row in enumerate(_rows(t)):
        assert row == sorted(row), "rows must be ascending"
        assert row.count(k) == 1, "a SNP tags itself exactly once"
        assert len(set(row)) == len(row)
        assert all(chrom[j] == chrom[k] for j in row), "tagging never crosses chromosomes"
        assert all(abs(int(pos[j]) - int(pos[k])) <= 1_000_000 for j in row)
        pairs.update((k, j) for j in row)
    assert all((j, k) in pairs for (k, j) in pairs), "the tagging relation is symmetric"
    # r2 is symmetric and equals one on the diagonal
    dense = np.zeros((m, m))
    rows = np.repeat(np.arange(m), t.nstar)
    dense[rows, t.indices] = t.r2
    np.testing.assert_allclose(dense, dense.T, atol=1e-13)
    assert np.all(np.diag(dense) == 1.0)
    assert np.all(t.nstar >= 1) and np.all(t.ldscore >= 1.0 - 1e-12)


def test_bias_correction_formula(toy_panel):
    geno, chrom, pos = toy_panel
    n = geno.shape[1]
    (t,) = compute_ld(geno, pos, chrom, configs=[(1000.0, 0.1)])
    rows = np.repeat(np.arange(t.n_snps), t.nstar)
    raw = np.bincount(rows, weights=t.r2, minlength=t.n_snps)
    np.testing.assert_allclose(t.ldscore, raw - (t.nstar - raw) / (n - 2), atol=1e-11)
    # a singleton has LD score exactly 1 (r2 = 1 with itself, no correction)
    single = t.nstar == 1
    if single.any():
        np.testing.assert_allclose(t.ldscore[single], 1.0, atol=1e-15)


def test_window_is_inclusive():
    # Independent genotypes and a vanishing threshold: the tagging set is "everything in the window".
    rng = np.random.default_rng(5)
    geno = rng.integers(0, 3, size=(6, 200)).astype(np.int8)
    pos = np.array([1_000, 501_000, 501_001, 1_001_000, 1_001_001, 3_000_000])
    (t,) = compute_ld(geno, pos, configs=[(500.0, 1e-300)])
    rows = _rows(t)
    assert rows[0] == [0, 1], (
        "exactly 500,000 bp apart is inside a 500 kb window; 500,001 bp is not"
    )
    assert rows[1] == [0, 1, 2, 3], "1 -> 3 is exactly 500,000 bp; 1 -> 4 is 500,001 bp"
    assert rows[5] == [5]
    (t2,) = compute_ld(geno, pos, configs=[(499.999, 1e-300)])
    assert _rows(t2)[0] == [0]


def test_threshold_is_inclusive(toy_panel):
    geno, chrom, pos = toy_panel
    sel = chrom == 1
    g, p = geno[sel], pos[sel]
    (base,) = compute_ld(g, p, configs=[(1000.0, 0.05)])
    rows = np.repeat(np.arange(base.n_snps), base.nstar)
    off = np.flatnonzero((rows != base.indices) & (base.r2 > 0.2) & (base.r2 < 0.8))
    k, j, v = int(rows[off[0]]), int(base.indices[off[0]]), float(base.r2[off[0]])
    (at,) = compute_ld(g, p, configs=[(1000.0, v)])
    assert j in _rows(at)[k], "r2 exactly equal to r2_min is tagged (>=)"
    (above,) = compute_ld(g, p, configs=[(1000.0, float(np.nextafter(v, 2.0)))])
    assert j not in _rows(above)[k]


def test_multiple_configurations_in_one_pass(toy_panel):
    geno, chrom, pos = toy_panel
    configs = [(w, t) for w in (500.0, 1000.0, 2000.0) for t in (0.05, 0.1, 0.2)]
    together = compute_ld(geno, pos, chrom, configs=configs, keep_r2=[i % 2 == 0 for i in range(9)])
    assert [(t.window_kb, t.r2_min) for t in together] == configs
    for i, (cfg, joint) in enumerate(zip(configs, together, strict=True)):
        (alone,) = compute_ld(geno, pos, chrom, configs=[cfg])
        np.testing.assert_array_equal(joint.nstar, alone.nstar)
        np.testing.assert_array_equal(joint.indices, alone.indices)
        np.testing.assert_array_equal(joint.indptr, alone.indptr)
        np.testing.assert_allclose(joint.ldscore, alone.ldscore, atol=1e-11)
        assert (joint.r2 is not None) == (i % 2 == 0)
    # nesting: a wider window / lower threshold can only add neighbours
    by = dict(zip(configs, together, strict=True))
    assert np.all(by[(2000.0, 0.05)].nstar >= by[(1000.0, 0.05)].nstar)
    assert np.all(by[(1000.0, 0.05)].nstar >= by[(1000.0, 0.1)].nstar)
    assert np.all(by[(1000.0, 0.1)].nstar >= by[(1000.0, 0.2)].nstar)


@pytest.mark.parametrize("block_size", [1, 7, 64])
def test_block_size_does_not_change_results(toy_panel, block_size):
    geno, chrom, pos = toy_panel
    (ref,) = compute_ld(geno, pos, chrom, configs=[(1000.0, 0.1)])
    (blk,) = compute_ld(geno, pos, chrom, configs=[(1000.0, 0.1)], block_size=block_size)
    np.testing.assert_array_equal(blk.indices, ref.indices)
    np.testing.assert_array_equal(blk.nstar, ref.nstar)
    np.testing.assert_allclose(blk.ldscore, ref.ldscore, atol=1e-11)


def test_missing_genotypes_are_mean_imputed(toy_panel, brute_force_ld):
    geno, chrom, pos = toy_panel
    rng = np.random.default_rng(11)
    g = geno.copy()
    holes = rng.random(g.shape) < 0.03
    g[holes] = -1
    (t,) = compute_ld(g, pos, chrom, configs=[(1000.0, 0.1)])
    sets, nstar, ld, _ = brute_force_ld(g, pos, chrom, 1000.0, 0.1)
    np.testing.assert_array_equal(t.nstar, nstar)
    assert _rows(t) == sets
    np.testing.assert_allclose(t.ldscore, ld, atol=1e-10)
    # floating input with NaN is equivalent to integer input with -1
    gf = g.astype(np.float64)
    gf[holes] = np.nan
    (tf,) = compute_ld(gf, pos, chrom, configs=[(1000.0, 0.1)])
    np.testing.assert_array_equal(tf.indices, t.indices)
    np.testing.assert_allclose(tf.ldscore, t.ldscore, atol=1e-12)


def test_chromosomes_are_independent(toy_panel):
    geno, chrom, pos = toy_panel
    (joint,) = compute_ld(geno, pos, chrom, configs=[(1000.0, 0.1)])
    m1 = int((chrom == 1).sum())
    (c1,) = compute_ld(geno[:m1], pos[:m1], configs=[(1000.0, 0.1)])
    (c2,) = compute_ld(geno[m1:], pos[m1:], configs=[(1000.0, 0.1)])
    np.testing.assert_array_equal(joint.nstar, np.concatenate([c1.nstar, c2.nstar]))
    np.testing.assert_array_equal(joint.indices, np.concatenate([c1.indices, c2.indices + m1]))
    np.testing.assert_allclose(joint.ldscore, np.concatenate([c1.ldscore, c2.ldscore]), atol=1e-12)
    cat = LDTable.concatenate([c1, c2])
    np.testing.assert_array_equal(cat.indptr, joint.indptr)
    np.testing.assert_array_equal(cat.indices, joint.indices)
    np.testing.assert_allclose(cat.r2, joint.r2, atol=1e-13)


# ------------------------------------------------------------------------------ rounding
def test_round_sig_matches_printf_g():
    rng = np.random.default_rng(1)
    x = np.concatenate(
        [rng.uniform(1e-4, 1.0, 20000), [1.0, 0.1, 0.05, 0.09999996, 0.123456789, 0.0512345678]]
    )
    for digits in (6, 3):
        expected = np.array([float(f"{v:.{digits}g}") for v in x])
        np.testing.assert_array_equal(_round_sig(x, digits), expected)
    assert _round_sig(np.array([0.09999996]), 6)[0] == 0.1
    assert (
        _round_sig(np.array([0.0512345678]), 6)[0] == 0.0512346
    )  # seven decimals, six significant
    assert _round_sig(np.array([0.0]), 6)[0] == 0.0


def test_round_r2_emulates_a_printed_plink_table(toy_panel, brute_force_ld):
    geno, chrom, pos = toy_panel
    n = geno.shape[1]
    digits = 2  # coarse on purpose, so that rounding visibly moves pairs across thresholds
    low, high = 0.05, 0.2
    t_low, t_high = compute_ld(
        geno, pos, chrom, configs=[(1000.0, low), (1000.0, high)], round_r2=digits
    )
    _, _, _, r2 = brute_force_ld(geno, pos, chrom, 1000.0, low)
    m = geno.shape[0]
    expected = {low: [], high: []}
    ld = {low: np.zeros(m), high: np.zeros(m)}
    rounded_up = 0
    for k in range(m):
        for j in range(m):
            if chrom[j] != chrom[k] or abs(int(pos[j]) - int(pos[k])) > 1_000_000:
                continue
            exact = 1.0 if j == k else r2[k, j]
            printed = 1.0 if j == k else float(f"{exact:.{digits}g}")
            for t in (low, high):
                if j == k or printed >= t:  # the threshold acts on the printed value
                    expected[t].append((k, j))
                    ld[t][k] += printed - (1 - printed) / (n - 2)
                    rounded_up += int(j != k and exact < t)
    assert rounded_up > 0, "the panel must contain pairs that only pass after rounding"
    for table, t in ((t_low, low), (t_high, high)):
        rows = np.repeat(np.arange(m), table.nstar)
        assert list(zip(rows.tolist(), table.indices.tolist(), strict=True)) == expected[t]
        np.testing.assert_allclose(table.ldscore, ld[t], atol=1e-10)
        assert all(float(f"{v:.{digits}g}") == v for v in table.r2.tolist())
    # the result for one configuration does not depend on which others were requested
    (alone,) = compute_ld(geno, pos, chrom, configs=[(1000.0, high)], round_r2=digits)
    np.testing.assert_array_equal(alone.indices, t_high.indices)
    np.testing.assert_allclose(alone.ldscore, t_high.ldscore, atol=1e-12)
    # and it differs from the unrounded computation
    (plain_high,) = compute_ld(geno, pos, chrom, configs=[(1000.0, high)])
    assert plain_high.nnz < t_high.nnz


def test_default_is_unrounded(toy_panel):
    geno, chrom, pos = toy_panel
    (plain,) = compute_ld(geno, pos, chrom, configs=[(1000.0, 0.1)])
    (rounded,) = compute_ld(geno, pos, chrom, configs=[(1000.0, 0.1)], round_r2=6)
    assert np.max(np.abs(plain.r2 - rounded.r2)) <= 5e-7
    assert np.any(plain.r2 != rounded.r2)
    np.testing.assert_allclose(plain.ldscore, rounded.ldscore, atol=1e-3)


# ------------------------------------------------------------------------------ backends
def test_torch_backend_on_cpu_matches_numpy(toy_panel):
    pytest.importorskip("torch")
    geno, chrom, pos = toy_panel
    configs = [(1000.0, 0.1), (500.0, 0.2)]
    a = compute_ld(geno, pos, chrom, configs=configs)
    b = compute_ld(geno, pos, chrom, configs=configs, device="cpu", backend="torch", block_size=50)
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(x.nstar, y.nstar)
        np.testing.assert_array_equal(x.indices, y.indices)
        np.testing.assert_allclose(x.ldscore, y.ldscore, atol=1e-10)
        np.testing.assert_allclose(x.r2, y.r2, atol=1e-12)


def test_float32_is_close(toy_panel):
    geno, chrom, pos = toy_panel
    (a,) = compute_ld(geno, pos, chrom, configs=[(1000.0, 0.1)])
    (b,) = compute_ld(geno, pos, chrom, configs=[(1000.0, 0.1)], dtype="float32")
    assert np.mean(a.nstar == b.nstar) > 0.98
    np.testing.assert_allclose(a.ldscore, b.ldscore, atol=0.5)
    with pytest.raises(InputError, match="float32"):
        compute_ld(geno, pos, chrom, configs=[(1000.0, 0.1)], dtype="float32", round_r2=6)


def test_backend_errors(toy_panel):
    geno, chrom, pos = toy_panel
    with pytest.raises(BackendError):
        compute_ld(geno, pos, chrom, device="tpu")
    with pytest.raises(BackendError):
        compute_ld(geno, pos, chrom, device="cuda", backend="numpy")
    with pytest.raises(BackendError):
        compute_ld(geno, pos, chrom, backend="jax")


# ---------------------------------------------------------------------------- validation
def test_input_validation(toy_panel):
    geno, chrom, pos = toy_panel
    with pytest.raises(InputError, match="sorted"):
        compute_ld(geno, pos[::-1].copy(), chrom)
    shuffled = chrom.copy()
    shuffled[0], shuffled[-1] = shuffled[-1], shuffled[0]
    with pytest.raises(InputError, match="contiguous"):
        compute_ld(geno, pos, shuffled)
    with pytest.raises(InputError):
        compute_ld(geno, pos[:-1], chrom)
    with pytest.raises(InputError):
        compute_ld(geno, pos, chrom[:-1])
    with pytest.raises(InputError):
        compute_ld(geno[:, :2], pos, chrom)
    with pytest.raises(InputError):
        compute_ld(geno[0], pos[:1])
    for bad in (
        [],
        [(0.0, 0.1)],
        [(1000.0, 0.0)],
        [(1000.0, 1.5)],
        [(1000.0, 0.1), (1000.0, 0.1)],
        [(1000.0,)],
    ):
        with pytest.raises(InputError):
            compute_ld(geno, pos, chrom, configs=bad)
    with pytest.raises(InputError):
        compute_ld(geno, pos, chrom, round_r2=0)
    with pytest.raises(InputError):
        compute_ld(geno, pos, chrom, keep_r2=[True, False])
    with pytest.raises(InputError):
        compute_ld(geno, pos, chrom, block_size=0)
    with pytest.raises(InputError):
        compute_ld(geno, pos, chrom, dtype="float16")


def test_monomorphic_snps_are_rejected(toy_panel):
    geno, chrom, pos = toy_panel
    g = geno.copy()
    g[5] = 1
    with pytest.raises(DataError, match="monomorphic"):
        compute_ld(g, pos, chrom)
    g = geno.copy()
    g[7] = -1
    with pytest.raises(DataError, match="monomorphic"):
        compute_ld(g, pos, chrom)


def test_standardize_and_frequencies():
    g = np.array([[0, 1, 2, -1], [2, 2, 0, 0], [1, 1, 1, -1]], dtype=np.int8)
    freq, call = allele_frequencies(g)
    np.testing.assert_allclose(freq, [0.5, 0.5, 0.5])
    np.testing.assert_allclose(call, [0.75, 1.0, 0.75])
    z = standardize_genotypes(g[:2])
    np.testing.assert_allclose((z * z).sum(axis=1), 1.0)
    np.testing.assert_allclose(z.sum(axis=1), 0.0, atol=1e-15)
    assert z[0, 3] == 0.0, "a missing genotype is the mean, i.e. zero after centring"
    with pytest.raises(DataError):
        standardize_genotypes(g)
    all_missing = np.full((1, 4), -1, dtype=np.int8)
    f, c = allele_frequencies(all_missing)
    assert np.isnan(f[0]) and c[0] == 0.0
