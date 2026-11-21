"""Reference bundles: loading, lookups, writing and retrieval."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

import genesis_gwas as gg
from genesis_gwas.data import download
from genesis_gwas.data.reference import config_id, default_data_dir, load_reference, save_reference
from genesis_gwas.exceptions import DownloadError, InputError, ReferenceDataError


@pytest.fixture(scope="module")
def ref():
    return load_reference("mini_chr22")


def test_packaged_bundle_loads_and_is_consistent(ref):
    assert ref.name == "mini_chr22" and ref.M == 15633 and ref.n_ref == 489
    assert ref.genome_build == "GRCh37" and ref.population == "EUR"
    assert ref.config.id == "w1000kb_r0.1" and {c.id for c in ref.configs} == {"w1000kb_r0.1", "w500kb_r0.2"}
    nb = ref.neighborhoods
    assert nb.n_ref == ref.M
    np.testing.assert_array_equal(np.diff(nb.indptr), ref.nstar)  # N* = neighbourhood size
    # every SNP tags itself; neighbourhoods are symmetric
    rows = np.repeat(np.arange(ref.M), ref.nstar)
    assert np.all(np.bincount(rows[nb.indices == rows], minlength=ref.M) == 1)
    adjacency = sp.csr_matrix((np.ones(nb.indices.size, dtype=np.int8), nb.indices, nb.indptr), shape=(ref.M, ref.M))
    assert (adjacency != adjacency.T).nnz == 0
    # LD score = sum of bias-corrected r^2 over the tagged SNPs
    r2 = ref.pair_r2().astype(np.float64)
    adj = r2 - (1 - r2) / (ref.n_ref - 2)
    np.testing.assert_allclose(np.add.reduceat(adj, nb.indptr[:-1]), ref.ldscore, rtol=2e-4)
    assert "mini_chr22" in ref.describe()


def test_ld_configuration_selection():
    alt = load_reference("mini_chr22", window_kb=500, r2_min=0.2)
    assert alt.config.id == "w500kb_r0.2"
    base = load_reference("mini_chr22")
    assert np.all(alt.nstar <= base.nstar)  # narrower window and stricter threshold tag fewer SNPs
    with pytest.raises(InputError, match="no LD configuration"):
        load_reference("mini_chr22", window_kb=2000, r2_min=0.05)
    with pytest.raises(ReferenceDataError):
        alt.pair_r2()
    assert config_id(1000, 0.1) == "w1000kb_r0.1" and config_id(500.0, 0.05) == "w500kb_r0.05"


def test_lookup_by_identifier_and_position(ref):
    ids = ref.subset_ids(np.array([0, 5, ref.M - 1]))
    np.testing.assert_array_equal(ref.lookup(list(ids) + ["rs_not_there", ""]), [0, 5, ref.M - 1, -1, -1])
    np.testing.assert_array_equal(ref.lookup(np.array(ids, dtype="S")), [0, 5, ref.M - 1])
    rows = np.array([3, 100, 9999])
    np.testing.assert_array_equal(ref.lookup_position(ref.chrom[rows], ref.pos[rows]), rows)
    np.testing.assert_array_equal(ref.lookup_position(["chr22", "22", "CHR22"], ref.pos[rows], "hg19"), rows)
    ok = ref.pos_grch38[rows] > 0
    got = ref.lookup_position(ref.chrom[rows], ref.pos_grch38[rows], "GRCh38")
    np.testing.assert_array_equal(got[ok], rows[ok])
    np.testing.assert_array_equal(ref.lookup_position([21, 22, "X"], [ref.pos[3], 1, ref.pos[3]]), [-1, -1, -1])
    with pytest.raises(InputError):
        ref.lookup_position([22], [1], "hg17")


def test_null_replicates_and_examples(ref):
    assert ref.n_null_replicates == 10
    z0 = ref.null_z(0)
    assert z0.shape == (ref.M,) and z0.dtype == np.float32 and abs(float(z0.std()) - 1.0) < 0.05
    assert not np.array_equal(z0, ref.null_z(9))
    with pytest.raises(ReferenceDataError):
        ref.null_z(10)
    assert ref.example_path().name == "height_chr22.sumstats.tsv.gz"
    assert ref.example_path("height_chr22.sumstats.tsv.gz").exists()
    with pytest.raises(ReferenceDataError):
        ref.example_path("nope")


def test_save_and_reload_round_trip(tmp_path, ref):
    rows = np.arange(400)
    nb = ref.neighborhoods
    lists = [np.array([j for j in nb.indices[nb.indptr[i] : nb.indptr[i + 1]] if j < 400]) for i in rows]
    indptr = np.concatenate([[0], np.cumsum([len(x) for x in lists])])
    indices = np.concatenate(lists)
    r2 = np.random.default_rng(0).uniform(0.1, 1, indices.size)
    out = save_reference(
        tmp_path / "custom",
        name="custom_panel",
        rsid=ref.subset_ids(rows),
        chrom=ref.chrom[rows],
        pos=ref.pos[rows],
        configs=[{"window_kb": 250, "r2_min": 0.3, "nstar": np.diff(indptr), "ldscore": ref.ldscore[rows],
                  "indptr": indptr, "indices": indices, "r2": r2}],
        population="TEST",
        n_ref=123,
        null_z=np.random.default_rng(1).normal(size=(12, 400)),
    )
    back = load_reference(out)
    assert back.name == "custom_panel" and back.M == 400 and back.n_ref == 123 and back.population == "TEST"
    assert back.config.id == "w250kb_r0.3" and back.hm3_no_mhc is None and back.pos_grch38 is None
    np.testing.assert_array_equal(back.neighborhoods.indices, indices)
    np.testing.assert_allclose(back.pair_r2(), r2, atol=1e-5)
    assert back.n_null_replicates == 12 and back.null_z(11).shape == (400,)
    assert all(v == "ok" for v in download.verify_bundle(out).values())
    # corruption is detected
    victim = Path(out) / "ld" / "w250kb_r0.3.npz"
    victim.write_bytes(victim.read_bytes()[:-10] + b"0123456789")
    assert download.verify_bundle(out)["ld/w250kb_r0.3.npz"] == "corrupt"

    with pytest.raises(InputError, match="sorted"):
        save_reference(tmp_path / "bad", name="x", rsid=["a", "b"], chrom=[2, 1], pos=[1, 1], configs=[])
    with pytest.raises(InputError, match="unique"):
        save_reference(tmp_path / "bad", name="x", rsid=["a", "a"], chrom=[1, 1], pos=[1, 2], configs=[])


def test_unknown_reference_and_data_dir(monkeypatch, tmp_path):
    with pytest.raises(ReferenceDataError, match="neither a bundle directory"):
        load_reference("no_such_reference")
    monkeypatch.setenv("GENESIS_DATA_DIR", str(tmp_path / "store"))
    assert default_data_dir() == tmp_path / "store"
    monkeypatch.delenv("GENESIS_DATA_DIR")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert default_data_dir() == tmp_path / "xdg" / "genesis-gwas"
    with pytest.raises(ReferenceDataError, match="download=False"):
        load_reference("eur_1kg_hm3", data_dir=tmp_path / "empty", download=False)


# ---------------------------------------------------------------- downloads
@pytest.fixture
def fake_remote(tmp_path, monkeypatch):
    """Serve a small bundle from a local directory through file:// URLs."""
    src = Path(str(load_reference("mini_chr22").root))
    manifest = json.loads((src / "manifest.json").read_text())
    manifest["name"] = "eur_1kg_hm3"
    remote = tmp_path / "remote"
    remote.mkdir()
    for rel in manifest["files"]:
        shutil.copy(src / rel, remote / download.asset_name("eur_1kg_hm3", manifest["version"], rel))
    monkeypatch.setattr(download, "load_packaged_manifest", lambda name: manifest)
    monkeypatch.setenv("GENESIS_DATA_URL", remote.as_uri())
    return remote, manifest


def test_fetch_is_lazy_verified_and_atomic(fake_remote, tmp_path):
    remote, manifest = fake_remote
    store = tmp_path / "store"
    root = download.fetch_bundle("eur_1kg_hm3", "default", data_dir=store)
    got = sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())
    assert got == ["ld/w1000kb_r0.1.npz", "manifest.json", "snps.npz"]  # nothing else was fetched
    assert not list(root.rglob("*.part")) and not list(root.rglob("*.lock"))

    ref = load_reference("eur_1kg_hm3", data_dir=store)
    assert ref.M == 15633
    assert ref.pair_r2().size == ref.neighborhoods.indices.size  # fetched on demand
    assert (root / "pairld" / "w1000kb_r0.1.npz").exists()
    download.fetch_bundle("eur_1kg_hm3", ["nullz", "w500kb_r0.2"], data_dir=store)
    assert (root / "nullz" / "nullz_000-009.npy").exists() and (root / "ld" / "w500kb_r0.2.npz").exists()
    status = download.verify_bundle(root)
    assert status["snps.npz"] == "ok" and status["examples/height_chr22.sumstats.tsv.gz"] == "missing"
    with pytest.raises(ReferenceDataError, match="Unknown component"):
        download.fetch_bundle("eur_1kg_hm3", "everything", data_dir=store)


def test_corrupt_download_is_rejected(fake_remote, tmp_path):
    remote, manifest = fake_remote
    asset = remote / download.asset_name("eur_1kg_hm3", manifest["version"], "snps.npz")
    data = bytearray(asset.read_bytes())
    data[100] ^= 0xFF  # same size, different content
    asset.write_bytes(bytes(data))
    with pytest.raises(DownloadError, match="Checksum mismatch"):
        download.fetch_bundle("eur_1kg_hm3", "default", data_dir=tmp_path / "store")
    root = tmp_path / "store" / "eur_1kg_hm3" / manifest["version"]
    assert not (root / "snps.npz").exists() and not list(root.rglob("*.part"))


def test_interrupted_download_resumes(fake_remote, tmp_path):
    remote, manifest = fake_remote
    root = tmp_path / "store" / "eur_1kg_hm3" / manifest["version"]
    root.mkdir(parents=True)
    asset = remote / download.asset_name("eur_1kg_hm3", manifest["version"], "snps.npz")
    (root / "snps.npz.part").write_bytes(asset.read_bytes()[:5000])  # a partial earlier transfer
    path = download.ensure_file("eur_1kg_hm3", "snps.npz", root)
    assert download.sha256_file(path) == manifest["files"]["snps.npz"]["sha256"]


def test_stale_lock_is_broken(fake_remote, tmp_path):
    import os
    import time

    remote, manifest = fake_remote
    root = tmp_path / "store" / "eur_1kg_hm3" / manifest["version"]
    root.mkdir(parents=True)
    lock = root / "snps.npz.lock"
    lock.write_text("12345\n")
    old = time.time() - 3600
    os.utime(lock, (old, old))
    assert download.ensure_file("eur_1kg_hm3", "snps.npz", root).exists()


def test_top_level_exports():
    assert gg.load_reference is load_reference and callable(gg.fetch_bundle)
