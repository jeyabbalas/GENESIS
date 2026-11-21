"""Shared helpers for the LD-builder tests (exposed as fixtures; no cross-module imports)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def _encode_bed(geno: np.ndarray) -> bytes:
    """Independent, deliberately naive PLINK .bed encoder.

    ``geno`` is variant-major (n_variants, n_samples) with A1 dosages 0/1/2 and -1 for
    missing. Written genotype by genotype so that it shares no code with the reader.
    """
    code_of = {2: 0b00, -1: 0b01, 1: 0b10, 0: 0b11}  # A1 dosage -> 2-bit code
    out = bytearray([0x6C, 0x1B, 0x01])
    for row in geno:
        byte = 0
        for i, g in enumerate(row.tolist()):
            byte |= code_of[int(g)] << (2 * (i % 4))
            if i % 4 == 3:
                out.append(byte)
                byte = 0
        if len(row) % 4:
            out.append(byte)
    return bytes(out)


def _write_fileset(
    prefix: Path,
    geno: np.ndarray,
    chrom: np.ndarray,
    pos: np.ndarray,
    ids: list[str] | None = None,
    iids: list[str] | None = None,
) -> Path:
    m, n = geno.shape
    ids = ids or [f"rs{i + 1}" for i in range(m)]
    iids = iids or [f"S{i + 1}" for i in range(n)]
    prefix.parent.mkdir(parents=True, exist_ok=True)
    Path(str(prefix) + ".bed").write_bytes(_encode_bed(geno))
    with open(str(prefix) + ".bim", "w") as fh:
        for c, s, p in zip(chrom.tolist(), ids, pos.tolist(), strict=True):
            fh.write(f"{c}\t{s}\t0\t{p}\tA\tG\n")
    with open(str(prefix) + ".fam", "w") as fh:
        for s in iids:
            fh.write(f"{s} {s} 0 0 0 -9\n")
    return prefix


def _simulate_genotypes(
    rng: np.random.Generator, m: int, n: int, block: int = 12, rho: float = 0.85
) -> np.ndarray:
    """Haplotype-copying simulation with block LD: variant-major int8 dosages in {0,1,2}."""
    freq = rng.uniform(0.1, 0.9, m)
    haps = np.empty((2 * n, m), dtype=np.int8)
    for h in range(2 * n):
        prev = None
        for j in range(m):
            if j % block == 0 or prev is None or rng.random() > rho:
                prev = int(rng.random() < freq[j])
            elif rng.random() < 0.08:
                prev = 1 - prev
            haps[h, j] = prev
    geno = (haps[:n] + haps[n:]).T.astype(np.int8)  # (m, n)
    # make sure nothing is monomorphic
    for j in range(m):
        if geno[j].min() == geno[j].max():
            geno[j, 0] = (geno[j, 0] + 1) % 3
    return np.ascontiguousarray(geno)


def _brute_force_ld(
    geno: np.ndarray, pos: np.ndarray, chrom: np.ndarray, window_kb: float, r2_min: float
):
    """O(m^2) reference implementation straight from the definitions (missing mean-imputed)."""
    m, n = geno.shape
    x = geno.astype(float)
    x[geno < 0] = np.nan
    mu = np.nanmean(x, axis=1)
    x = np.where(np.isnan(x), mu[:, None], x)
    r = np.corrcoef(x)
    r2 = r * r
    sets, nstar, ld = [], np.zeros(m, dtype=int), np.zeros(m)
    for k in range(m):
        members = []
        for j in range(m):
            if chrom[j] != chrom[k] or abs(int(pos[j]) - int(pos[k])) > window_kb * 1000:
                continue
            v = 1.0 if j == k else r2[k, j]
            if j == k or v >= r2_min:
                members.append(j)
                ld[k] += v - (1.0 - v) / (n - 2)
        sets.append(members)
        nstar[k] = len(members)
    return sets, nstar, ld, r2


@pytest.fixture(scope="session")
def encode_bed():
    return _encode_bed


@pytest.fixture(scope="session")
def write_fileset():
    return _write_fileset


@pytest.fixture(scope="session")
def simulate_genotypes():
    return _simulate_genotypes


@pytest.fixture(scope="session")
def brute_force_ld():
    return _brute_force_ld


@pytest.fixture(scope="session")
def toy_panel():
    """A small two-chromosome panel with block LD: (geno, chrom, pos)."""
    rng = np.random.default_rng(20261121)
    m1, m2, n = 150, 90, 120
    geno = _simulate_genotypes(rng, m1 + m2, n)
    chrom = np.concatenate([np.full(m1, 1), np.full(m2, 2)])
    pos = np.concatenate(
        [np.sort(rng.integers(1, 3_000_000, m1)), np.sort(rng.integers(1, 2_000_000, m2))]
    ).astype(np.int64)
    # force a few exact-distance and duplicate-free positions for boundary tests
    pos[:4] = [1000, 501_000, 1_001_000, 1_001_001]
    pos[:m1] = np.sort(pos[:m1])
    assert np.unique(pos[:m1]).size == m1 and np.unique(pos[m1:]).size == m2
    return geno, chrom, pos
