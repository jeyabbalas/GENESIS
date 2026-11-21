"""build_reference end to end: PLINK fileset -> bundle -> load_reference -> prepare/fit."""

from __future__ import annotations

import json
import warnings

import numpy as np
import pytest

import genesis_gwas as gg
from genesis_gwas.exceptions import DataError, InputError
from genesis_gwas.ld import build_reference, compute_ld


@pytest.fixture(scope="module")
def messy_panel(simulate_genotypes):
    """Genotypes plus the bookkeeping problems a real fileset has, in shuffled file order."""
    rng = np.random.default_rng(7)
    n = 100
    m1, m2 = 70, 50
    geno = simulate_genotypes(rng, m1 + m2, n)
    chrom = np.array(["1"] * m1 + ["chr2"] * m2, dtype=object)
    pos = np.concatenate(
        [
            np.sort(rng.choice(np.arange(1, 2_500_000), m1, replace=False)),
            np.sort(rng.choice(np.arange(1, 1_500_000), m2, replace=False)),
        ]
    ).astype(np.int64)
    ids = [f"rs{i + 1}" for i in range(m1 + m2)]

    extra_geno, extra_chrom, extra_pos, extra_ids = [], [], [], []

    def add(g, c, p, s):
        extra_geno.append(np.asarray(g, dtype=np.int8))
        extra_chrom.append(c)
        extra_pos.append(p)
        extra_ids.append(s)

    add(np.full(n, 2), "1", 3_000_001, "rs_mono")  # monomorphic
    rare = np.zeros(n, dtype=np.int8)
    rare[:3] = 1
    add(rare, "1", 3_000_002, "rs_rare")  # MAF = 3/200 = 0.015
    add(geno[10], "1", 3_000_003, ids[10])  # duplicated identifier
    add(rng.integers(0, 3, n), "1", int(pos[20]), "rs_same_position")  # duplicated position
    add(rng.integers(0, 3, n), "X", 5_000, "rs_chrX")  # not an autosome
    add(rng.integers(0, 3, n), "2", 2_000_000, "rs_extra_chr2")  # fine, on chr2
    all_missing = np.full(n, -1, dtype=np.int8)
    add(all_missing, "2", 2_000_001, "rs_all_missing")

    G = np.vstack([geno, np.vstack(extra_geno)])
    C = np.concatenate([chrom, np.array(extra_chrom, dtype=object)])
    P = np.concatenate([pos, np.array(extra_pos, dtype=np.int64)])
    I = ids + extra_ids
    order = rng.permutation(len(I))
    # keep the original copy of the duplicated id ahead of its duplicate in file order
    a, b = int(np.flatnonzero(order == 10)[0]), int(np.flatnonzero(order == m1 + m2 + 2)[0])
    if a > b:
        order[a], order[b] = order[b], order[a]
    return G[order], C[order], P[order], [I[i] for i in order]


def _expected(messy_panel, min_maf=0.05):
    """Independent re-statement of the documented filtering and ordering rules."""
    G, C, P, I = messy_panel
    num = {"1": 1, "chr2": 2, "2": 2}
    rows = []
    seen = set()
    for k in range(len(I)):
        if C[k] not in num or I[k] in seen:
            continue
        seen.add(I[k])
        rows.append((num[C[k]], int(P[k]), I[k], k))
    rows.sort()
    kept, last = [], None
    for c, p, s, k in rows:
        if (c, p) == last:
            continue
        last = (c, p)
        g = G[k]
        ok = g >= 0
        if not ok.any():
            continue
        f = g[ok].sum() / (2 * ok.sum())
        if g[ok].min() == g[ok].max() or min(f, 1 - f) < min_maf:
            continue
        kept.append((c, p, s, k))
    return kept


def test_build_load_and_definitions(tmp_path, messy_panel, write_fileset, brute_force_ld):
    G, C, P, I = messy_panel
    prefix = write_fileset(tmp_path / "panel", G, C, P, ids=I)
    configs = [(1000.0, 0.1), (500.0, 0.2)]
    out = build_reference(
        prefix,
        tmp_path / "bundle",
        name="toy_ref",
        configs=configs,
        n_null=12,
        seed=5,
        population="TOY",
        genome_build="GRCh38",
        panel="unit-test panel",
    )
    kept = _expected(messy_panel)
    exp_ids = [s for _, _, s, _ in kept]
    assert "rs_mono" not in exp_ids and "rs_rare" not in exp_ids and "rs_chrX" not in exp_ids
    assert "rs_all_missing" not in exp_ids and "rs_extra_chr2" in exp_ids
    assert exp_ids.count("rs11") == 1

    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["name"] == "toy_ref" and manifest["population"] == "TOY"
    assert manifest["genome_build"] == "GRCh38" and manifest["n_ref"] == 100
    assert manifest["n_snps"] == len(kept)
    assert [c["id"] for c in manifest["ld_configs"]] == ["w1000kb_r0.1", "w500kb_r0.2"]
    assert manifest["ld_configs"][0]["pairld"] and manifest["ld_configs"][1]["pairld"] is None
    assert manifest["nullz"]["n_replicates"] == 12
    counts = manifest["source"]["filter_counts"]
    assert counts["duplicate_id"] == 1 and counts["duplicate_position"] == 1
    assert counts["monomorphic_or_all_missing"] == 2 and counts["below_min_maf"] == 1

    geno = G[[k for *_, k in kept]]
    chrom = np.array([c for c, *_ in kept])
    pos = np.array([p for _, p, *_ in kept])
    for window_kb, r2_min in configs:
        ref = gg.load_reference(out, window_kb=window_kb, r2_min=r2_min)
        assert len(kept) == ref.M and ref.n_ref == 100 and ref.genome_build == "GRCh38"
        assert ref.rsid.astype("U").tolist() == exp_ids, "canonical (chrom, pos, id) order"
        np.testing.assert_array_equal(ref.chrom, chrom)
        np.testing.assert_array_equal(ref.pos, pos)
        sets, nstar, ld, r2 = brute_force_ld(geno, pos, chrom, window_kb, r2_min)
        np.testing.assert_array_equal(ref.nstar, nstar)
        np.testing.assert_allclose(ref.ldscore, ld, atol=1e-10)
        nb = ref.neighborhoods
        got = [nb.indices[nb.indptr[k] : nb.indptr[k + 1]].tolist() for k in range(ref.M)]
        assert got == sets
        if window_kb == 1000.0:
            rows = np.repeat(np.arange(ref.M), ref.nstar)
            want = np.where(rows == nb.indices, 1.0, r2[rows, nb.indices])
            assert np.max(np.abs(ref.pair_r2() - want)) <= 0.5 / 65535 + 1e-7
        else:
            with pytest.raises(gg.GenesisError):
                ref.pair_r2()
    ref = gg.load_reference(out)
    assert ref.config.id == "w1000kb_r0.1" and ref.n_null_replicates == 12
    z0 = ref.null_z(0)
    assert z0.shape == (len(kept),) and np.all(np.isfinite(z0))
    assert np.array_equal(ref.lookup(["rs_extra_chr2", "nope"]) >= 0, [True, False])
    assert int(ref.lookup_position([2], [2_000_000])[0]) == exp_ids.index("rs_extra_chr2")


def test_per_chromosome_template_snp_list_and_sample_subset(tmp_path, messy_panel, write_fileset):
    G, C, P, I = messy_panel
    is1 = np.array([c == "1" for c in C])
    is2 = np.array([c in ("chr2", "2") for c in C])
    iids = [f"ID{i:03d}" for i in range(G.shape[1])]
    write_fileset(
        tmp_path / "split.1",
        G[is1],
        C[is1],
        P[is1],
        ids=[s for s, k in zip(I, is1, strict=True) if k],
        iids=iids,
    )
    write_fileset(
        tmp_path / "split.2",
        G[is2],
        C[is2],
        P[is2],
        ids=[s for s, k in zip(I, is2, strict=True) if k],
        iids=iids,
    )
    whole = write_fileset(tmp_path / "whole", G, C, P, ids=I, iids=iids)

    a = gg.load_reference(build_reference(whole, tmp_path / "a", name="a", pair_r2_for=None))
    b = gg.load_reference(
        build_reference(
            str(tmp_path / "split.{chrom}"),
            tmp_path / "b",
            name="b",
            chromosomes=[1, 2],
            pair_r2_for="all",
        )
    )
    np.testing.assert_array_equal(a.rsid, b.rsid)
    np.testing.assert_array_equal(a.nstar, b.nstar)
    np.testing.assert_allclose(a.ldscore, b.ldscore, atol=1e-12)
    np.testing.assert_array_equal(a.neighborhoods.indices, b.neighborhoods.indices)
    assert a.config.pairld is None and b.config.pairld is not None
    assert a.n_null_replicates == 0

    # only chromosome 2
    only2 = gg.load_reference(build_reference(whole, tmp_path / "c", name="c", chromosomes=[2]))
    assert set(only2.chrom.tolist()) == {2}

    # SNP list (as a file) and sample subset (as a PLINK-style keep file and as a sequence)
    wanted = a.rsid.astype("U")[::2].tolist()
    snp_file = tmp_path / "snps.txt"
    snp_file.write_text("\n".join([*wanted, "rs_not_there"]) + "\n")
    keep_ids = iids[10:90]
    keep_file = tmp_path / "keep.txt"
    keep_file.write_text("".join(f"{s} {s}\n" for s in keep_ids))
    d = gg.load_reference(
        build_reference(whole, tmp_path / "d", name="d", snps=snp_file, keep=keep_file, min_maf=0.0)
    )
    e = gg.load_reference(
        build_reference(whole, tmp_path / "e", name="e", snps=wanted, keep=keep_ids, min_maf=0.0)
    )
    assert d.n_ref == 80 and set(d.rsid.astype("U").tolist()) <= set(wanted)
    np.testing.assert_array_equal(d.rsid, e.rsid)
    np.testing.assert_array_equal(d.nstar, e.nstar)
    # the LD of the subset is that of the subset's genotypes
    cols = np.arange(10, 90)
    pick = [I.index(s) for s in d.rsid.astype("U").tolist()]
    (t,) = compute_ld(G[np.ix_(pick, cols)], d.pos, d.chrom)
    np.testing.assert_array_equal(d.nstar, t.nstar)
    np.testing.assert_allclose(d.ldscore, t.ldscore, atol=1e-12)


def test_round_r2_is_recorded_and_applied(tmp_path, messy_panel, write_fileset):
    G, C, P, I = messy_panel
    prefix = write_fileset(tmp_path / "panel", G, C, P, ids=I)
    out = build_reference(prefix, tmp_path / "r", name="r", round_r2=6)
    assert json.loads((out / "manifest.json").read_text())["source"]["round_r2"] == 6
    plain = gg.load_reference(build_reference(prefix, tmp_path / "p", name="p"))
    rounded = gg.load_reference(out)
    np.testing.assert_allclose(plain.ldscore, rounded.ldscore, atol=1e-3)
    assert np.any(plain.ldscore != rounded.ldscore)


def test_errors(tmp_path, messy_panel, write_fileset):
    G, C, P, I = messy_panel
    prefix = write_fileset(tmp_path / "panel", G, C, P, ids=I)
    with pytest.raises(DataError, match="incomplete"):
        build_reference(tmp_path / "nope", tmp_path / "o", name="x")
    with pytest.raises(DataError, match="No SNPs"):
        build_reference(prefix, tmp_path / "o", name="x", snps=["rs_does_not_exist"])
    with pytest.raises(DataError, match="keep"):
        build_reference(prefix, tmp_path / "o", name="x", keep=["S1", "S2", "S3", "ghost"])
    with pytest.raises(InputError):
        build_reference(prefix, tmp_path / "o", name="x", min_maf=0.6)
    with pytest.raises(InputError):
        build_reference(prefix, tmp_path / "o", name="x", chromosomes=[0, 23])
    with pytest.raises(InputError):
        build_reference(prefix, tmp_path / "o", name="x", pair_r2_for=[(123.0, 0.1)])
    with pytest.raises(InputError):
        build_reference(prefix, tmp_path / "o", name="x", pair_r2_for="some")
    with pytest.raises(InputError):
        build_reference(prefix, tmp_path / "o", name="x", n_null=-1)
    with pytest.raises(DataError, match="snps"):
        build_reference(prefix, tmp_path / "o", name="x", snps=tmp_path / "missing.txt")


def test_built_reference_works_in_prepare_and_fit(tmp_path, simulate_genotypes, write_fileset):
    rng = np.random.default_rng(2026)
    m, n_ref = 500, 150
    geno = simulate_genotypes(rng, m, n_ref, block=10, rho=0.8)
    chrom = np.repeat([1, 2], m // 2)
    pos = np.concatenate([np.arange(m // 2) * 40_000 + 1, np.arange(m // 2) * 40_000 + 1])
    prefix = write_fileset(tmp_path / "panel", geno, chrom, pos)
    ref = gg.load_reference(
        build_reference(prefix, tmp_path / "bundle", name="sim", n_null=3, min_maf=0.0)
    )
    assert m == ref.M

    # summary statistics simulated on this very panel: marginal = R joint, noise ~ N(0, R)
    from genesis_gwas.ld import null_z, standardize_genotypes

    order = [int(s[2:]) - 1 for s in ref.rsid.astype("U").tolist()]
    Z = standardize_genotypes(geno[order])
    R = Z @ Z.T
    same = ref.chrom[:, None] == ref.chrom[None, :]
    R = np.where(same, R, 0.0)
    n_gwas = 50_000.0
    causal = rng.random(m) < 0.1
    joint = np.where(causal, rng.normal(0.0, np.sqrt(2e-4), m), 0.0)
    z = np.sqrt(n_gwas) * (R @ joint) + null_z(geno[order], 1, seed=8)[0]
    ss = gg.SumStats.from_arrays(ref.rsid.astype("U"), z, n_gwas)
    data = gg.prepare(ss, ref, qc=False)
    assert m == data.K and m == data.M
    np.testing.assert_array_equal(data.nstar, ref.nstar)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = gg.fit(data, 2, device="cpu", threads=1)
    assert res.n_snps == m and np.all(np.isfinite(res.estimates)) and np.isfinite(res.loglik)
    assert res.settings["has_neighborhoods"] is True
    assert 0.0 < res.pic < 1.0 and res.heritability > 0


def test_torch_backend_and_null_statistic_options(tmp_path, messy_panel, write_fileset):
    pytest.importorskip("torch")
    G, C, P, I = messy_panel
    prefix = write_fileset(tmp_path / "panel", G, C, P, ids=I)
    a = gg.load_reference(
        build_reference(
            prefix, tmp_path / "np", name="np", n_null=4, seed=3, null_statistic="normal"
        )
    )
    b = gg.load_reference(
        build_reference(
            prefix,
            tmp_path / "th",
            name="th",
            n_null=4,
            seed=3,
            null_statistic="normal",
            backend="torch",
            device="cpu",
            block_size=17,
            threads=2,
        )
    )
    np.testing.assert_array_equal(a.nstar, b.nstar)
    np.testing.assert_array_equal(a.neighborhoods.indices, b.neighborhoods.indices)
    np.testing.assert_allclose(a.ldscore, b.ldscore, atol=1e-10)
    np.testing.assert_array_equal(a.null_z(3), b.null_z(3))
    manifest = json.loads((a.root / "manifest.json").read_text())
    assert manifest["source"]["null_statistic"] == "normal"
    with pytest.raises(InputError):
        build_reference(prefix, tmp_path / "bad", name="bad", n_null=2, null_statistic="wald")
