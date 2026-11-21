r"""Build an LD reference bundle from the genotypes of a reference panel.

GENESIS describes each GWAS marker :math:`k` by two numbers computed in a reference
panel of :math:`n` individuals genotyped at :math:`M` SNPs, and by the list of SNPs
behind them (Zhang et al. 2018, Methods, "Calculation of LD-score and number of tagged
SNPs"):

tagging set
    :math:`\mathcal N_k = \{\, m : \text{same chromosome},\ |pos_m - pos_k| \le W,\
    r^2_{km} \ge r^2_{\min} \,\}`. The window :math:`W` is **inclusive** and the marker
    itself is always a member (:math:`r^2_{kk}=1`).
number of tagged SNPs
    :math:`N^*_k = |\mathcal N_k|`.
LD score
    :math:`\ell_k = \sum_{m\in\mathcal N_k}\bigl[r^2_{km} - (1-r^2_{km})/(n-2)\bigr]`,
    i.e. the sum over the tagging set of the bias-corrected squared correlations used
    by LD-score regression (Bulik-Sullivan et al. 2015).

Here :math:`r_{km}` is the Pearson correlation of the allele dosages of SNPs :math:`k`
and :math:`m` across the :math:`n` reference samples (missing genotypes are imputed
with the SNP's mean dosage).

These definitions were established by re-deriving the tables shipped with the original
R package from the 1000 Genomes genotypes they were computed from; see
``tools/ld_validation`` for the script and the measured agreement.

Reproducing the original tables exactly
---------------------------------------
The original tables were evidently derived from a printed PLINK ``--r2`` table, in which
:math:`r^2` has six significant digits: thresholds were applied to, and LD scores summed
from, the printed values. ``round_r2=6`` emulates that (see :func:`compute_ld`) and
reproduces all nine original tables. For new reference panels leave ``round_r2=None``:
rounding is an artefact of that workflow, not part of the method.

Algorithm
---------
Per chromosome, genotypes are standardised to unit-norm rows :math:`\tilde x_k`, so that
:math:`r_{km}=\tilde x_k^\top\tilde x_m`. Blocks of consecutive SNPs are multiplied with
the SNPs inside their window (one BLAS ``gemm`` — or one GPU matrix product — per
block), thresholded, and written out as sorted CSR rows. The dense working memory is
bounded by ``block_size x window width``; beyond that only one chromosome's genotypes
(one byte each) and the sparse result are held. Several (window, threshold)
configurations are produced from a *single* pass: the banded :math:`r^2` is computed
once for the widest window and lowest threshold and filtered down. The whole genome for
the 1.07 million HapMap3 SNPs of 489 samples takes well under a minute on a laptop.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from genesis_gwas._version import __version__
from genesis_gwas.exceptions import BackendError, DataError, InputError
from genesis_gwas.ld.plink import BedReader, read_bim, read_fam, resolve_fileset

__all__ = [
    "LDTable",
    "allele_frequencies",
    "build_reference",
    "compute_ld",
    "null_z",
    "standardize_genotypes",
]

log = logging.getLogger("genesis_gwas")

Config = tuple[float, float]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LDTable:
    r"""LD quantities of one (window, threshold) configuration.

    Attributes
    ----------
    window_kb
        Window half-width in kb (inclusive).
    r2_min
        Tagging threshold on :math:`r^2`.
    nstar
        ``(m,)`` ``int32`` — number of tagged SNPs :math:`N^*_k` (itself included).
    ldscore
        ``(m,)`` ``float64`` — bias-corrected LD score :math:`\ell_k`.
    indptr, indices
        Tagging sets in compressed-sparse-row form: the SNPs tagged by SNP ``k`` are
        ``indices[indptr[k]:indptr[k + 1]]``, in ascending order, ``k`` included.
        Indices refer to the rows of the genotype matrix passed to :func:`compute_ld`.
    r2
        ``(nnz,)`` ``float64`` squared correlations aligned with ``indices`` (after
        rounding, if ``round_r2`` was requested), or ``None`` if not kept.
    n_samples
        Number of reference samples :math:`n` used in the bias correction.
    """

    window_kb: float
    r2_min: float
    nstar: np.ndarray
    ldscore: np.ndarray
    indptr: np.ndarray
    indices: np.ndarray
    r2: np.ndarray | None
    n_samples: int

    @property
    def n_snps(self) -> int:
        """Number of SNPs (rows)."""
        return int(self.nstar.size)

    @property
    def nnz(self) -> int:
        """Total number of (ordered) tagging pairs, self pairs included."""
        return int(self.indices.size)

    @staticmethod
    def concatenate(tables: Sequence[LDTable]) -> LDTable:
        """Stack per-chromosome tables into one, shifting indices by the row offsets."""
        if not tables:
            raise InputError("No tables to concatenate")
        first = tables[0]
        for t in tables:
            if (t.window_kb, t.r2_min, t.n_samples) != (
                first.window_kb,
                first.r2_min,
                first.n_samples,
            ):
                raise InputError("Cannot concatenate LD tables of different configurations")
        offsets = np.cumsum([0] + [t.n_snps for t in tables])
        nnz_off = np.cumsum([0] + [t.nnz for t in tables])
        if int(offsets[-1]) > np.iinfo(np.int32).max:
            raise DataError("Too many SNPs for 32-bit neighbour indices")
        indptr = np.concatenate(
            [np.zeros(1, dtype=np.int64)]
            + [t.indptr[1:].astype(np.int64) + nnz_off[i] for i, t in enumerate(tables)]
        )
        # Shift in place into one preallocated int32 array (no int64 temporaries).
        indices = np.empty(int(nnz_off[-1]), dtype=np.int32)
        for i, t in enumerate(tables):
            np.add(t.indices, np.int32(offsets[i]), out=indices[nnz_off[i] : nnz_off[i + 1]])
        keep_r2 = all(t.r2 is not None for t in tables)
        return LDTable(
            window_kb=first.window_kb,
            r2_min=first.r2_min,
            nstar=np.concatenate([t.nstar for t in tables]).astype(np.int32),
            ldscore=np.concatenate([t.ldscore for t in tables]),
            indptr=indptr,
            indices=indices,
            r2=np.concatenate([t.r2 for t in tables if t.r2 is not None]) if keep_r2 else None,
            n_samples=first.n_samples,
        )


# ---------------------------------------------------------------------------
# Genotype summaries
# ---------------------------------------------------------------------------
def _valid_mask(G: np.ndarray) -> np.ndarray:
    if G.dtype.kind == "f":
        return np.isfinite(G)
    return G >= 0


def allele_frequencies(genotypes: np.ndarray, chunk: int = 8192) -> tuple[np.ndarray, np.ndarray]:
    """Allele frequency and call rate of each SNP.

    Parameters
    ----------
    genotypes
        ``(m, n)`` variant-major dosages in ``{0, 1, 2}``; missing values are ``-1``
        (integer arrays) or ``NaN`` (floating arrays).
    chunk
        SNPs processed at a time.

    Returns
    -------
    (freq, call_rate)
        ``freq[k]`` is the frequency of the counted allele among non-missing samples
        (``NaN`` if every genotype is missing); ``call_rate[k]`` the fraction of
        non-missing samples.
    """
    G = np.asarray(genotypes)
    if G.ndim != 2:
        raise InputError("genotypes must be a 2-D (n_snps, n_samples) array")
    m, n = G.shape
    freq = np.empty(m)
    call = np.empty(m)
    for i0 in range(0, m, chunk):
        g = G[i0 : i0 + chunk]
        valid = _valid_mask(g)
        cnt = valid.sum(axis=1)
        tot = np.where(valid, g, 0).sum(axis=1, dtype=np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            freq[i0 : i0 + chunk] = tot / (2.0 * cnt)
        call[i0 : i0 + chunk] = cnt / n
    return freq, call


def _moments(G: np.ndarray, chunk: int = 8192) -> tuple[np.ndarray, np.ndarray]:
    """Per-SNP mean and centred sum of squares (missing values mean-imputed)."""
    m = G.shape[0]
    mean = np.empty(m)
    ss = np.empty(m)
    for i0 in range(0, m, chunk):
        g = G[i0 : i0 + chunk]
        valid = _valid_mask(g)
        x = np.where(valid, g, 0).astype(np.float64)
        cnt = valid.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            mu = x.sum(axis=1) / cnt
        mean[i0 : i0 + chunk] = mu
        x -= mu[:, None]
        x[~valid] = 0.0
        ss[i0 : i0 + chunk] = np.einsum("ij,ij->i", x, x)
    return mean, ss


def standardize_genotypes(
    genotypes: np.ndarray,
    mean: np.ndarray | None = None,
    ss: np.ndarray | None = None,
    dtype: Any = np.float64,
) -> np.ndarray:
    r"""Centre each SNP and scale it to unit Euclidean norm.

    With :math:`\tilde x_k = (x_k - \bar x_k)/\lVert x_k - \bar x_k\rVert` the Pearson
    correlation of two SNPs is the plain dot product
    :math:`r_{km} = \tilde x_k^\top \tilde x_m`. Missing genotypes are replaced by the
    SNP's mean dosage, i.e. they become zero after centring.

    Parameters
    ----------
    genotypes
        ``(m, n)`` variant-major dosages (``-1``/``NaN`` = missing).
    mean, ss
        Optional pre-computed per-SNP means and centred sums of squares.
    dtype
        Floating-point type of the result.

    Raises
    ------
    DataError
        If a SNP is monomorphic (zero variance) or entirely missing.
    """
    G = np.asarray(genotypes)
    if G.ndim != 2:
        raise InputError("genotypes must be a 2-D (n_snps, n_samples) array")
    if mean is None or ss is None:
        mean, ss = _moments(G)
    if not np.all(np.isfinite(mean)) or np.any(~(ss > 0)):
        bad = int(np.sum(~(ss > 0) | ~np.isfinite(mean)))
        raise DataError(
            f"{bad} SNP(s) are monomorphic or entirely missing in the reference samples; "
            "remove them before computing LD (build_reference does this automatically)."
        )
    valid = _valid_mask(G)
    Z = np.where(valid, G, 0).astype(dtype)
    Z -= mean[:, None].astype(dtype)
    Z[~valid] = 0.0
    Z *= (1.0 / np.sqrt(ss))[:, None].astype(dtype)
    return Z


# ---------------------------------------------------------------------------
# Banded r^2
# ---------------------------------------------------------------------------
def _round_sig(x: np.ndarray, digits: int) -> np.ndarray:
    """Round positive values to ``digits`` significant decimal digits (like C's ``%g``)."""
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros_like(x)
    pos = x > 0
    if np.any(pos):
        xp = x[pos]
        expo = np.floor(np.log10(xp))
        # guard against log10 landing a hair on the wrong side of a power of ten
        expo = np.where(xp >= 10.0 ** (expo + 1), expo + 1, expo)
        expo = np.where(xp < 10.0**expo, expo - 1, expo)
        dec = (digits - 1 - expo).astype(np.int64)
        scale = np.power(10.0, dec.astype(np.float64))
        out[pos] = np.rint(xp * scale) / scale
    return out


def _resolve_backend(device: str, backend: str | None) -> tuple[str, Any]:
    """Return ``(backend_name, torch_module_or_None)`` for a device/backend request."""
    dev = str(device)
    on_gpu = dev.startswith("cuda") or dev == "mps"
    if dev != "cpu" and not on_gpu:
        raise BackendError(f"Unrecognised device {device!r} (expected 'cpu', 'cuda' or 'cuda:N').")
    name = backend or ("torch" if on_gpu else "numpy")
    if name not in ("numpy", "torch"):
        raise BackendError(f"Unknown backend {backend!r}; expected 'numpy' or 'torch'.")
    if name == "numpy":
        if on_gpu:
            raise BackendError("The numpy backend cannot use GPU devices.")
        return "numpy", None
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise BackendError(
            "GPU LD computation needs PyTorch (`pip install 'genesis-gwas[gpu]'`)."
        ) from exc
    if dev.startswith("cuda") and not torch.cuda.is_available():
        raise BackendError("device='cuda' was requested but no CUDA device is available.")
    return "torch", torch


def _block_pairs_numpy(
    Zb: np.ndarray, Zw: np.ndarray, pos_b: np.ndarray, pos_w: np.ndarray, max_bp: int, floor: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rsq = Zb @ Zw.T
    np.multiply(rsq, rsq, out=rsq)
    keep = rsq >= floor
    keep &= np.abs(pos_w[None, :] - pos_b[:, None]) <= max_bp
    rows, cols = np.nonzero(keep)
    return rows, cols, rsq[rows, cols].astype(np.float64)


def _block_pairs_torch(
    torch: Any,
    device: str,
    Zb: np.ndarray,
    Zw: np.ndarray,
    pos_b: np.ndarray,
    pos_w: np.ndarray,
    max_bp: int,
    floor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with torch.inference_mode():
        tb = torch.from_numpy(np.ascontiguousarray(Zb)).to(device)
        tw = torch.from_numpy(np.ascontiguousarray(Zw)).to(device)
        rsq = tb @ tw.T
        rsq *= rsq
        pb = torch.from_numpy(np.ascontiguousarray(pos_b)).to(device)
        pw = torch.from_numpy(np.ascontiguousarray(pos_w)).to(device)
        keep = (rsq >= floor) & ((pw[None, :] - pb[:, None]).abs() <= max_bp)
        rows, cols = torch.nonzero(keep, as_tuple=True)
        vals = rsq[rows, cols].to(torch.float64)
        return rows.cpu().numpy(), cols.cpu().numpy(), vals.cpu().numpy()


def _validate_configs(configs: Sequence[Config]) -> list[Config]:
    out: list[Config] = []
    try:
        items = [(float(w), float(t)) for w, t in configs]
    except (TypeError, ValueError) as exc:
        raise InputError("configs must be a sequence of (window_kb, r2_min) pairs") from exc
    if not items:
        raise InputError("At least one (window_kb, r2_min) configuration is required")
    for w, t in items:
        if not w > 0:
            raise InputError(f"window_kb must be > 0 (got {w})")
        if not 0.0 < t <= 1.0:
            raise InputError(f"r2_min must lie in (0, 1] (got {t})")
        if (w, t) in out:
            raise InputError(f"Duplicate LD configuration (window_kb={w:g}, r2_min={t:g})")
        out.append((w, t))
    return out


def _compute_chromosome(
    G: np.ndarray,
    pos: np.ndarray,
    configs: list[Config],
    keep_r2: list[bool],
    round_r2: int | None,
    block_size: int,
    backend: str,
    torch: Any,
    device: str,
    dtype: Any,
) -> list[LDTable]:
    """LD tables of one chromosome (rows sorted by position)."""
    m, n = G.shape
    mean, ss = _moments(G)
    if not np.all(np.isfinite(mean)) or np.any(~(ss > 0)):
        bad = int(np.sum(~(ss > 0) | ~np.isfinite(mean)))
        raise DataError(
            f"{bad} SNP(s) are monomorphic or entirely missing in the reference samples; "
            "remove them before computing LD (build_reference does this automatically)."
        )
    max_bp = round(max(w for w, _ in configs) * 1000.0)
    min_t = min(t for _, t in configs)
    # When r^2 is rounded, a value just below a threshold may round up onto it. Rounding
    # to d significant digits moves a value by at most half a unit of its d-th digit,
    # i.e. by a relative 0.5 * 10^-(d-1): collect candidates down to that bound and
    # decide after rounding.
    if round_r2 is None:
        floor = min_t
    else:
        floor = min_t / (1.0 + 0.5 * 10.0 ** (-(round_r2 - 1))) * (1.0 - 1e-12)
    floor = max(floor - 4.0 * np.finfo(np.float64).eps, 0.0)
    pos64 = pos.astype(np.int64)

    rows_l: list[np.ndarray] = []
    cols_l: list[np.ndarray] = []
    vals_l: list[np.ndarray] = []
    for i0 in range(0, m, block_size):
        i1 = min(m, i0 + block_size)
        lo = int(np.searchsorted(pos64, pos64[i0] - max_bp, side="left"))
        hi = int(np.searchsorted(pos64, pos64[i1 - 1] + max_bp, side="right"))
        Zw = standardize_genotypes(G[lo:hi], mean[lo:hi], ss[lo:hi], dtype)
        Zb = Zw[i0 - lo : i1 - lo]
        if backend == "numpy":
            r, c, v = _block_pairs_numpy(Zb, Zw, pos64[i0:i1], pos64[lo:hi], max_bp, floor)
        else:
            r, c, v = _block_pairs_torch(
                torch, device, Zb, Zw, pos64[i0:i1], pos64[lo:hi], max_bp, floor
            )
        rows_l.append((r + i0).astype(np.int32))
        cols_l.append((c + lo).astype(np.int32))
        vals_l.append(v)
    rows = np.concatenate(rows_l) if rows_l else np.empty(0, dtype=np.int32)
    cols = np.concatenate(cols_l) if cols_l else np.empty(0, dtype=np.int32)
    r2_exact = np.concatenate(vals_l) if vals_l else np.empty(0)
    del rows_l, cols_l, vals_l
    is_self = rows == cols
    np.minimum(r2_exact, 1.0, out=r2_exact)  # duplicates can exceed one by an ulp
    r2_exact[is_self] = 1.0
    # A SNP always tags itself, whatever floating-point noise did to its diagonal entry.
    missing_self = np.setdiff1d(np.arange(m, dtype=np.int32), rows[is_self], assume_unique=False)
    if missing_self.size:  # pragma: no cover - cannot happen for r2 floor <= 1
        raise DataError("Internal error: self pairs missing from the banded r2")
    dist = np.abs(pos64[cols] - pos64[rows]).astype(np.int32)  # <= widest window

    if round_r2 is None:
        r2_used = r2_exact
    else:
        r2_used = _round_sig(r2_exact, round_r2)
        r2_used[is_self] = 1.0

    tables: list[LDTable] = []
    for (w_kb, t), want_r2 in zip(configs, keep_r2, strict=True):
        # With round_r2 the threshold acts on the rounded ("printed") value, so a pair
        # whose exact r^2 is a hair below r2_min but prints as r2_min is tagged.
        sel = (dist <= round(w_kb * 1000.0)) & (r2_used >= t)
        sel |= is_self
        rs = rows[sel]
        x = r2_used[sel]
        nstar = np.bincount(rs, minlength=m).astype(np.int32)
        ldscore = np.bincount(rs, weights=x - (1.0 - x) / (n - 2.0), minlength=m)
        indptr = np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(nstar, dtype=np.int64)])
        tables.append(
            LDTable(
                window_kb=w_kb,
                r2_min=t,
                nstar=nstar,
                ldscore=ldscore,
                indptr=indptr,
                indices=cols[sel],
                r2=x.copy() if want_r2 else None,
                n_samples=n,
            )
        )
    return tables


def compute_ld(
    genotypes: np.ndarray,
    pos: Any,
    chrom: Any = None,
    *,
    configs: Sequence[Config] = ((1000.0, 0.1),),
    round_r2: int | None = None,
    keep_r2: bool | Sequence[bool] = True,
    block_size: int = 1024,
    device: str = "cpu",
    backend: str | None = None,
    dtype: str = "float64",
) -> list[LDTable]:
    r"""Compute :math:`N^*`, LD scores and tagging sets from a genotype matrix.

    Parameters
    ----------
    genotypes
        ``(m, n)`` **variant-major** allele dosages (one row per SNP) in ``{0, 1, 2}``;
        missing values are ``-1`` for integer arrays or ``NaN`` for floating arrays.
        :meth:`genesis_gwas.ld.plink.BedReader.read` returns exactly this layout.
    pos
        ``(m,)`` base-pair positions. Rows must be sorted by position within each
        chromosome.
    chrom
        ``(m,)`` chromosome labels; rows of one chromosome must be contiguous. ``None``
        means all SNPs are on one chromosome. Tagging never crosses chromosomes.
    configs
        ``(window_kb, r2_min)`` pairs. All configurations are produced from a single
        pass over the genotypes.
    round_r2
        ``None`` (default, recommended for new panels) uses exact :math:`r^2`.

        An integer ``d`` emulates filtering a printed PLINK ``--r2`` table, which is how
        the tables of the original R package were made: every :math:`r^2` is first
        rounded to ``d`` **significant** digits (PLINK prints six, so 0.123456 but
        0.0512346), and the rounded value is what is compared with ``r2_min`` and what
        enters the LD score. ``round_r2=6`` reproduces the original tables (see
        ``tools/ld_validation``). Its only effects are that pairs whose exact
        :math:`r^2` lies within half a unit of the last printed digit below ``r2_min``
        are tagged, and that LD scores move by about :math:`10^{-5}`.
    keep_r2
        Whether to return the pairwise :math:`r^2` of each configuration (a single
        ``bool`` or one per configuration).
    block_size
        SNPs per block. Working memory is about
        ``block_size x (SNPs within the widest window) x 8`` bytes.
    device
        ``"cpu"`` (NumPy/BLAS), or ``"cuda"`` / ``"cuda:N"`` to do the matrix products
        on a GPU with PyTorch (always in the precision given by ``dtype``).
    backend
        Force ``"numpy"`` or ``"torch"``; ``backend="torch"`` with ``device="cpu"`` runs
        the PyTorch code path on the CPU (useful to test an installation without a GPU).
    dtype
        ``"float64"`` (default) or ``"float32"``. Single precision is faster but makes
        threshold decisions for pairs within about :math:`10^{-6}` of ``r2_min``
        arbitrary; never combine it with ``round_r2``.

    Returns
    -------
    list of LDTable
        One table per configuration, in the order of ``configs``. Indices refer to the
        rows of ``genotypes``.

    Raises
    ------
    DataError
        If a SNP is monomorphic or entirely missing.
    InputError
        If the inputs are inconsistent or not sorted.
    """
    G = np.asarray(genotypes)
    if G.ndim != 2:
        raise InputError("genotypes must be a 2-D (n_snps, n_samples) array")
    m, n = G.shape
    if n < 3:
        raise InputError("At least three reference samples are needed (bias correction uses n - 2)")
    p = np.asarray(pos)
    if p.shape != (m,):
        raise InputError("pos must have one entry per SNP (row of genotypes)")
    p = p.astype(np.int64)
    cfgs = _validate_configs(configs)
    if isinstance(keep_r2, (bool, np.bool_)):
        keeps = [bool(keep_r2)] * len(cfgs)
    else:
        keeps = [bool(k) for k in keep_r2]
        if len(keeps) != len(cfgs):
            raise InputError("keep_r2 must be a bool or have one entry per configuration")
    if round_r2 is not None and int(round_r2) < 1:
        raise InputError("round_r2 must be a positive number of significant digits, or None")
    if dtype not in ("float64", "float32"):
        raise InputError("dtype must be 'float64' or 'float32'")
    if dtype == "float32" and round_r2 is not None:
        raise InputError(
            "round_r2 requires dtype='float64': float32 correlations are not accurate enough "
            "to be rounded to a fixed number of digits reproducibly."
        )
    if block_size < 1:
        raise InputError("block_size must be >= 1")
    name, torch = _resolve_backend(device, backend)
    np_dtype = np.float64 if dtype == "float64" else np.float32

    if chrom is None:
        bounds = [(0, m)]
    else:
        c = np.asarray(chrom)
        if c.shape != (m,):
            raise InputError("chrom must have one entry per SNP (row of genotypes)")
        change = np.flatnonzero(c[1:] != c[:-1]) + 1
        starts = np.concatenate([[0], change])
        stops = np.concatenate([change, [m]])
        labels = c[starts]
        if np.unique(labels).size != labels.size:
            raise InputError("Rows of the same chromosome must be contiguous")
        bounds = [(int(a), int(b)) for a, b in zip(starts, stops, strict=True)]
    per_chrom: list[list[LDTable]] = []
    for a, b in bounds:
        if b - a == 0:
            continue
        if np.any(np.diff(p[a:b]) < 0):
            raise InputError("SNPs must be sorted by position within each chromosome")
        per_chrom.append(
            _compute_chromosome(
                G[a:b],
                p[a:b],
                cfgs,
                keeps,
                None if round_r2 is None else int(round_r2),
                int(block_size),
                name,
                torch,
                str(device),
                np_dtype,
            )
        )
    if not per_chrom:
        raise InputError("No SNPs")
    if len(per_chrom) == 1:
        return per_chrom[0]
    return [LDTable.concatenate([tabs[i] for tabs in per_chrom]) for i in range(len(cfgs))]


# ---------------------------------------------------------------------------
# Null Z-scores
# ---------------------------------------------------------------------------
def null_z(
    genotypes: np.ndarray,
    n_replicates: int | None = None,
    seed: int | None = 0,
    *,
    phenotypes: np.ndarray | None = None,
    statistic: str = "t",
    chunk: int = 8192,
) -> np.ndarray:
    r"""LD-correlated null association statistics of the reference panel.

    For a phenotype :math:`y \sim N(0, I_n)` that is unrelated to every SNP, a GWAS of
    the :math:`n` reference samples yields test statistics that are (marginally)
    null-distributed and whose correlation between SNPs :math:`k` and :math:`m` is the
    LD correlation :math:`\hat r_{km}` — the joint null distribution of real GWAS
    Z-scores (Zhang et al. 2018, "Simulation studies"). GENESIS adds such replicates,
    scaled by :math:`1/\sqrt{n_\text{GWAS}}`, to simulated marginal effects so that
    QQ-plot envelopes and genomic-control projections have realistic LD-induced
    variability.

    With :math:`\tilde x_k` the centred, unit-norm genotype of SNP :math:`k` and
    :math:`r_k = \tilde x_k^\top (y-\bar y)/\lVert y-\bar y\rVert` its sample correlation
    with the phenotype, ``statistic`` selects

    ``"t"`` (default)
        :math:`t_k = r_k\sqrt{n-2}/\sqrt{1-r_k^2}`, the *t* statistic of the simple
        regression of :math:`y` on SNP :math:`k`, i.e. what GWAS software reports. This
        is what the null replicates shipped with the original package are: their
        variance, spread across replicates and fourth moment match this statistic and
        not the raw projection (see ``tools/ld_validation``). Marginally
        :math:`t_{n-2}`, so its variance is :math:`(n-2)/(n-4)` (1.004 for 489 samples).
    ``"normal"``
        The same statistic mapped to exact standard-normal margins,
        :math:`z_k = \Phi^{-1}\{F_{t_{n-2}}(t_k)\}`. Prefer this for small reference
        panels (a few hundred samples or fewer), where the heavier tails and inflated
        variance of :math:`t_{n-2}` would otherwise leak into simulations of large GWAS.
    ``"projection"``
        :math:`u_k = \tilde x_k^\top y`, which is exactly :math:`N(0, \hat R)` over
        replicates. It is *not* studentised: all statistics of a replicate share the
        factor :math:`\lVert y-\bar y\rVert/\sqrt{n-1}`, whose standard deviation is
        :math:`\approx 1/\sqrt{2(n-1)}` (3% for 489 samples), so the genome-wide mean
        :math:`\chi^2` of a replicate fluctuates about six times more than for the
        other two choices. Provided for completeness; not recommended for simulations.

    Parameters
    ----------
    genotypes
        ``(m, n)`` variant-major dosages (``-1``/``NaN`` = missing; mean-imputed).
    n_replicates
        Number of independent null phenotypes to draw (ignored if ``phenotypes`` is given).
    seed
        Seed of the NumPy generator used to draw the phenotypes.
    phenotypes
        Optional ``(n_replicates, n)`` phenotype matrix. Pass the *same* matrix for
        every chromosome so that each replicate is one genome-wide null GWAS.
    statistic
        ``"t"``, ``"normal"`` or ``"projection"`` (see above).
    chunk
        SNPs processed at a time.

    Returns
    -------
    numpy.ndarray
        ``float32`` array of shape ``(n_replicates, m)``.
    """
    G = np.asarray(genotypes)
    if G.ndim != 2:
        raise InputError("genotypes must be a 2-D (n_snps, n_samples) array")
    m, n = G.shape
    if statistic not in ("t", "normal", "projection"):
        raise InputError("statistic must be 't', 'normal' or 'projection'")
    if n < 5:
        raise InputError("At least five reference samples are needed for null statistics")
    if phenotypes is None:
        if n_replicates is None or n_replicates < 1:
            raise InputError("n_replicates must be >= 1 when phenotypes are not supplied")
        Y = np.random.default_rng(seed).standard_normal((int(n_replicates), n))
    else:
        Y = np.asarray(phenotypes, dtype=np.float64)
        if Y.ndim != 2 or Y.shape[1] != n:
            raise InputError("phenotypes must have shape (n_replicates, n_samples)")
    if statistic != "projection":
        Yc = Y - Y.mean(axis=1, keepdims=True)
        norm = np.linalg.norm(Yc, axis=1, keepdims=True)
        if np.any(norm == 0):
            raise InputError("A null phenotype is constant; cannot compute association statistics")
        Y = Yc / norm
    out = np.empty((Y.shape[0], m), dtype=np.float32)
    for i0 in range(0, m, chunk):
        Z = standardize_genotypes(G[i0 : i0 + chunk])
        u = (Z @ Y.T).T
        if statistic != "projection":
            np.clip(u, -1.0 + 1e-15, 1.0 - 1e-15, out=u)
            u *= np.sqrt(n - 2.0) / np.sqrt(1.0 - u * u)
            if statistic == "normal":
                from scipy.stats import norm as _norm
                from scipy.stats import t as _t

                u = np.sign(u) * _norm.isf(_t.sf(np.abs(u), n - 2))
        out[:, i0 : i0 + chunk] = u
    return out


# ---------------------------------------------------------------------------
# Bundle construction
# ---------------------------------------------------------------------------
def _read_id_list(spec: Any, what: str) -> np.ndarray | None:
    """Read identifiers from a file (first whitespace-separated column) or a sequence."""
    if spec is None:
        return None
    if isinstance(spec, (str, os.PathLike)):
        path = Path(os.fspath(spec)).expanduser()
        if not path.exists():
            raise DataError(f"{what} file not found: {path}")
        ids = []
        opener: Any
        if path.suffix == ".gz":
            import gzip

            opener = gzip.open
        else:
            opener = open
        with opener(path, "rt") as fh:
            for line in fh:
                parts = line.split()
                if parts:
                    ids.append(parts)
        if not ids:
            raise DataError(f"{what} file {path} is empty")
        if what == "keep" and all(len(p) >= 2 for p in ids):
            return np.array([p[1] for p in ids], dtype=str)  # PLINK --keep: FID IID
        return np.array([p[0] for p in ids], dtype=str)
    arr = np.asarray(list(spec), dtype=str)
    if arr.size == 0:
        raise InputError(f"{what} list is empty")
    return arr


def _autosome(chrom: np.ndarray) -> np.ndarray:
    """Map chromosome labels to 1..22 (0 for anything else)."""
    txt = np.char.lower(np.asarray(chrom, dtype=str))
    txt = np.char.replace(txt, "chr", "")
    out = np.zeros(txt.shape, dtype=np.int64)
    for i in range(1, 23):
        out[txt == str(i)] = i
    return out


def _set_blas_threads(threads: int | None) -> Any:
    """Best-effort cap on BLAS threads; returns a context-manager-like limiter or ``None``."""
    if threads is None:
        return None
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        log.info(
            "threadpoolctl is not installed, so the BLAS thread count cannot be changed from "
            "Python; set OMP_NUM_THREADS / OPENBLAS_NUM_THREADS / VECLIB_MAXIMUM_THREADS before "
            "starting Python to control it."
        )
        return None
    return threadpool_limits(limits=int(threads))


def build_reference(
    bfile: str | os.PathLike[str],
    out_dir: str | os.PathLike[str],
    *,
    name: str,
    snps: Any = None,
    keep: Any = None,
    configs: Sequence[Config] = ((1000.0, 0.1),),
    min_maf: float = 0.05,
    n_null: int = 0,
    null_statistic: str = "t",
    seed: int = 0,
    population: str = "unknown",
    genome_build: str = "GRCh37",
    pair_r2_for: Any = "first",
    round_r2: int | None = None,
    chromosomes: Iterable[int] = range(1, 23),
    version: str = "1.0",
    panel: str = "",
    block_size: int = 1024,
    device: str = "cpu",
    backend: str | None = None,
    threads: int | None = None,
) -> Path:
    r"""Build a GENESIS LD reference bundle from PLINK genotypes.

    The reference panel defines the :math:`M` SNPs to which the proportion of
    susceptibility SNPs :math:`\pi_c` refers, so restrict it deliberately: the original
    analyses use HapMap3 SNPs with minor-allele frequency of at least 5% in an
    ancestry-matched panel (about 1.07 million SNPs for 1000 Genomes Europeans).

    Parameters
    ----------
    bfile
        PLINK fileset prefix, or a template containing ``{chrom}`` for per-chromosome
        filesets (e.g. ``"/data/1000G.EUR.QC.{chrom}"``).
    out_dir
        Directory to write the bundle to.
    name
        Name of the bundle (recorded in the manifest and in every result).
    snps
        SNP identifiers to keep: a path to a text file (first column; ``.gz`` allowed)
        or a sequence. ``None`` keeps every autosomal SNP of the fileset.
    keep
        Samples to use: a path to a file of sample ids (one ``IID`` per line, or
        ``FID IID`` as for ``plink --keep``) or a sequence of ``IID`` values. ``None``
        uses all samples.
    configs
        ``(window_kb, r2_min)`` LD configurations to compute (default: the paper's
        primary choice, 1 Mb and :math:`r^2 \ge 0.1`).
    min_maf
        Keep SNPs whose minor-allele frequency *in the selected samples* is at least
        this value (0.05 as in the paper). Monomorphic SNPs are always removed.
    n_null
        Number of LD-correlated null replicates to store (see :func:`null_z`); 0 for
        none. They are needed for QQ-plot envelopes and genomic-control projections.
        One hundred replicates cost about 0.2 GB per million SNPs.
    null_statistic
        Kind of null statistic: ``"t"`` (default; the regression *t* statistics of a
        null GWAS in the reference samples, as in the original package), ``"normal"``
        (the same with exact standard-normal margins; preferable for panels of a few
        hundred samples or fewer) or ``"projection"``.
    seed
        Seed for the null replicates.
    population, genome_build, version, panel
        Metadata recorded in the manifest. ``genome_build`` must describe the positions
        in the ``.bim`` file(s).
    pair_r2_for
        Which configurations also store pairwise :math:`r^2` (needed only to simulate
        marginal effects, e.g. for genomic-control projections): ``"first"`` (default),
        ``"all"``, ``None``, or a sequence of ``(window_kb, r2_min)`` pairs.
    round_r2
        See :func:`compute_ld`. Leave at ``None`` for new panels.
    chromosomes
        Autosomes to include (default 1–22).
    block_size, device, backend
        Passed to :func:`compute_ld`.
    threads
        Cap on BLAS threads (needs the optional ``threadpoolctl`` package; otherwise the
        BLAS library's environment variables apply).

    Returns
    -------
    pathlib.Path
        The bundle directory; pass it to :func:`genesis_gwas.load_reference`.

    Notes
    -----
    SNPs are written in canonical order (chromosome, position, identifier). Variants
    with a duplicated identifier or a duplicated position are dropped (first occurrence
    kept) and counted in the log, because identifiers and positions must each identify a
    reference SNP uniquely.
    """
    from genesis_gwas.data.reference import config_id, save_reference

    t_start = time.perf_counter()
    cfgs = _validate_configs(configs)
    if not 0.0 <= min_maf < 0.5:
        raise InputError("min_maf must lie in [0, 0.5)")
    if n_null < 0:
        raise InputError("n_null must be >= 0")
    if null_statistic not in ("t", "normal", "projection"):
        raise InputError("null_statistic must be 't', 'normal' or 'projection'")
    chrom_list = sorted({int(c) for c in chromosomes})
    if not chrom_list or chrom_list[0] < 1 or chrom_list[-1] > 22:
        raise InputError("chromosomes must be autosomes in 1..22")
    if pair_r2_for is None:
        keep_r2 = [False] * len(cfgs)
    elif isinstance(pair_r2_for, str):
        if pair_r2_for == "first":
            keep_r2 = [i == 0 for i in range(len(cfgs))]
        elif pair_r2_for == "all":
            keep_r2 = [True] * len(cfgs)
        else:
            raise InputError("pair_r2_for must be 'first', 'all', None or a list of configurations")
    else:
        wanted = _validate_configs(pair_r2_for)
        unknown = [c for c in wanted if c not in cfgs]
        if unknown:
            raise InputError(f"pair_r2_for names configurations that are not in configs: {unknown}")
        keep_r2 = [c in wanted for c in cfgs]

    snp_filter = _read_id_list(snps, "snps")
    snp_set = None if snp_filter is None else set(snp_filter.tolist())
    keep_ids = _read_id_list(keep, "keep")

    template = os.fspath(bfile)
    per_chrom_files = "{chrom}" in template
    filesets: list[tuple[int | None, str]] = (
        [(c, template.format(chrom=c)) for c in chrom_list]
        if per_chrom_files
        else [(None, template)]
    )

    limiter = _set_blas_threads(threads)
    if threads is not None and backend == "torch" and str(device) == "cpu":
        import torch

        torch.set_num_threads(int(threads))
    stats = {
        "variants_seen": 0,
        "not_autosomal_or_not_requested": 0,
        "not_in_snp_list": 0,
        "duplicate_id": 0,
        "duplicate_position": 0,
        "monomorphic_or_all_missing": 0,
        "below_min_maf": 0,
    }
    sample_ids: np.ndarray | None = None
    Y: np.ndarray | None = None
    seen_ids: set[str] = set()
    out_rsid: list[np.ndarray] = []
    out_chrom: list[np.ndarray] = []
    out_pos: list[np.ndarray] = []
    per_config: list[list[LDTable]] = [[] for _ in cfgs]
    out_null: list[np.ndarray] = []
    try:
        for file_chrom, prefix in filesets:
            bed_path, bim_path, fam_path = resolve_fileset(prefix)
            bim = read_bim(bim_path)
            fam = read_fam(fam_path)
            iid = fam["iid"].to_numpy(dtype=str)
            if keep_ids is None:
                sample_idx: np.ndarray | None = None
                ids_here = iid
            else:
                lookup = {s: i for i, s in enumerate(iid.tolist())}
                missing = [s for s in keep_ids.tolist() if s not in lookup]
                if missing:
                    raise DataError(
                        f"{len(missing)} sample id(s) in `keep` are not in {fam_path} "
                        f"(e.g. {missing[:3]})"
                    )
                sample_idx = np.array([lookup[s] for s in keep_ids.tolist()], dtype=np.int64)
                ids_here = iid[sample_idx]
            if sample_ids is None:
                sample_ids = ids_here
                if sample_ids.size < 3:
                    raise DataError("At least three reference samples are required")
                if np.unique(sample_ids).size != sample_ids.size:
                    raise DataError("Sample identifiers (IID) are not unique")
            elif not np.array_equal(sample_ids, ids_here):
                raise DataError(
                    f"{fam_path} lists different samples (or a different order) than the "
                    "previous chromosome's .fam file"
                )
            n = int(sample_ids.size)
            if n_null and Y is None:
                Y = np.random.default_rng(seed).standard_normal((int(n_null), n))

            reader = BedReader(bed_path, len(fam), len(bim))
            stats["variants_seen"] += len(bim)
            auto = _autosome(bim["chrom"].to_numpy())
            all_ids = bim["snp"].to_numpy(dtype=str)
            all_pos = bim["pos"].to_numpy(dtype=np.int64)
            for c in chrom_list if file_chrom is None else [file_chrom]:
                on_chrom = auto == c
                cand = np.flatnonzero(on_chrom)
                if snp_set is not None and cand.size:
                    hit = np.fromiter(
                        (s in snp_set for s in all_ids[cand].tolist()), dtype=bool, count=cand.size
                    )
                    stats["not_in_snp_list"] += int((~hit).sum())
                    cand = cand[hit]
                if cand.size == 0:
                    continue
                # duplicated identifiers (within this chromosome or seen on an earlier one)
                ids_c = all_ids[cand]
                first = np.zeros(cand.size, dtype=bool)
                first[np.unique(ids_c, return_index=True)[1]] = True
                if seen_ids:
                    first &= np.fromiter(
                        (s not in seen_ids for s in ids_c.tolist()), dtype=bool, count=cand.size
                    )
                stats["duplicate_id"] += int((~first).sum())
                cand = cand[first]
                # canonical order and duplicated positions
                order = np.lexsort((all_ids[cand], all_pos[cand]))
                cand = cand[order]
                pos_c = all_pos[cand]
                uniq = np.ones(cand.size, dtype=bool)
                uniq[1:] = pos_c[1:] != pos_c[:-1]
                stats["duplicate_position"] += int((~uniq).sum())
                cand = cand[uniq]
                if cand.size == 0:
                    continue

                G = reader.read(cand, samples=sample_idx)
                freq, _call = allele_frequencies(G)
                _mean, ss = _moments(G)
                poly = np.isfinite(freq) & (ss > 0)
                stats["monomorphic_or_all_missing"] += int((~poly).sum())
                with np.errstate(invalid="ignore"):
                    maf_ok = np.minimum(freq, 1.0 - freq) >= min_maf
                low = poly & ~maf_ok
                stats["below_min_maf"] += int(low.sum())
                good = poly & maf_ok
                if not np.all(good):
                    G = np.ascontiguousarray(G[good])
                    cand = cand[good]
                if cand.size == 0:
                    continue
                t0 = time.perf_counter()
                tables = compute_ld(
                    G,
                    all_pos[cand],
                    configs=cfgs,
                    round_r2=round_r2,
                    keep_r2=keep_r2,
                    block_size=block_size,
                    device=device,
                    backend=backend,
                )
                if Y is not None:
                    out_null.append(null_z(G, phenotypes=Y, statistic=null_statistic))
                seen_ids.update(all_ids[cand].tolist())
                out_rsid.append(all_ids[cand])
                out_chrom.append(np.full(cand.size, c, dtype=np.uint8))
                out_pos.append(all_pos[cand])
                for per, table in zip(per_config, tables, strict=True):
                    per.append(table)
                log.info(
                    "chr%-2d %9s SNPs, %s tagging pairs (%s), %.1f s",
                    c,
                    f"{cand.size:,}",
                    f"{tables[0].nnz:,}",
                    config_id(*cfgs[0]),
                    time.perf_counter() - t0,
                )
            reader.close()
            stats["not_autosomal_or_not_requested"] += int(
                (~np.isin(auto, chrom_list if file_chrom is None else [file_chrom])).sum()
            )
    finally:
        if limiter is not None:
            limiter.restore_original_limits()

    if not out_rsid or sample_ids is None:
        raise DataError(
            "No SNPs left to build a reference from. Filter counts: "
            + ", ".join(f"{k}={v:,}" for k, v in stats.items())
        )
    for key, val in stats.items():
        if val:
            log.info("build_reference: %-32s %12s", key, f"{val:,}")
    rsid = np.concatenate(out_rsid)
    chrom_a = np.concatenate(out_chrom)
    pos_a = np.concatenate(out_pos)
    # Merge one configuration at a time and release its per-chromosome pieces straight
    # away, so that peak memory is one copy of the tables plus one configuration.
    merged: list[LDTable] = []
    for per in per_config:
        merged.append(LDTable.concatenate(per))
        per.clear()
    cfg_dicts = [
        {
            "window_kb": t.window_kb,
            "r2_min": t.r2_min,
            "nstar": t.nstar,
            "ldscore": t.ldscore,
            "indptr": t.indptr,
            "indices": t.indices,
            "r2": t.r2,
        }
        for t in merged
    ]
    null_all = np.concatenate(out_null, axis=1) if out_null else None
    n = int(sample_ids.size)
    root = save_reference(
        out_dir,
        name=name,
        rsid=rsid,
        chrom=chrom_a,
        pos=pos_a,
        configs=cfg_dicts,
        version=version,
        population=population,
        panel=panel or f"{n} samples, {rsid.size:,} SNPs with MAF >= {min_maf:g}",
        genome_build=genome_build,
        n_ref=n,
        null_z=null_all,
        source={
            "built_with": f"genesis-gwas {__version__}",
            "bfile": template,
            "n_samples": n,
            "min_maf": float(min_maf),
            "round_r2": round_r2,
            "seed": int(seed),
            "null_statistic": null_statistic if n_null else None,
            "filter_counts": stats,
        },
    )
    log.info(
        "Wrote reference bundle %s: %s SNPs, %d samples, %d configuration(s), %.1f s",
        root,
        f"{rsid.size:,}",
        n,
        len(cfgs),
        time.perf_counter() - t_start,
    )
    return root
