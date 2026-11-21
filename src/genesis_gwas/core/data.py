"""The per-SNP inputs of the composite likelihood."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from genesis_gwas.exceptions import InputError

if TYPE_CHECKING:
    from genesis_gwas.core.variance import Neighborhoods

__all__ = ["ModelData"]


def _as_float(x: Any, name: str) -> np.ndarray:
    try:
        arr = np.ascontiguousarray(np.asarray(x, dtype=np.float64).ravel())
    except (TypeError, ValueError) as exc:
        raise InputError(f"{name} must be numeric") from exc
    return arr


@dataclass(frozen=True)
class ModelData:
    r"""Aligned per-SNP arrays for ``K`` GWAS markers.

    This is the complete input of the statistical model; everything upstream of it
    (file parsing, quality control, merging with the LD reference) lives in
    :mod:`genesis_gwas.data`.

    Parameters
    ----------
    betahat
        Standardised marginal effect estimates, :math:`\hat\beta_k = z_k/\sqrt{n_k}`.
    varbetahat
        Their sampling variances, :math:`s_k^2 = 1/n_k`.
    ldscore
        LD scores :math:`\ell_k` (sum of bias-corrected :math:`r^2` over the tagged SNPs).
    nstar
        Number of reference SNPs tagged by marker ``k``, :math:`N^*_k \ge 1`.
    n
        Per-SNP (effective) sample sizes; used for the BIC and for simulations.
    M
        Number of SNPs in the reference panel to which :math:`\pi_c` refers
        (1,070,777 for the bundled HapMap3 / 1000 Genomes EUR reference).
    snp
        Optional marker identifiers.
    ref_index
        Optional row index of each marker in the LD reference (needed to look up
        tagging neighbourhoods for the sandwich variance estimator).
    neighborhoods
        Optional tagging neighbourhoods of the LD reference
        (:class:`~genesis_gwas.core.variance.Neighborhoods`); together with
        ``ref_index`` they define which GWAS markers are correlated.
    meta
        Free-form provenance (QC counts, reference name, ...).
    """

    betahat: np.ndarray
    varbetahat: np.ndarray
    ldscore: np.ndarray
    nstar: np.ndarray
    n: np.ndarray
    M: int
    snp: np.ndarray | None = None
    ref_index: np.ndarray | None = None
    neighborhoods: Neighborhoods | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        b = _as_float(self.betahat, "betahat")
        s2 = _as_float(self.varbetahat, "varbetahat")
        ld = _as_float(self.ldscore, "ldscore")
        n = _as_float(self.n, "n")
        ns_raw = np.asarray(self.nstar).ravel()
        K = b.size
        for name, arr in (("varbetahat", s2), ("ldscore", ld), ("n", n), ("nstar", ns_raw)):
            if arr.size != K:
                raise InputError(f"{name} has length {arr.size}, expected {K} (length of betahat)")
        if K == 0:
            raise InputError("No SNPs: the model needs at least one marker.")
        if not np.all(np.isfinite(b)):
            raise InputError("betahat contains non-finite values")
        if not (np.all(np.isfinite(s2)) and np.all(s2 > 0)):
            raise InputError("varbetahat must be finite and strictly positive")
        if not (np.all(np.isfinite(ld)) and np.all(ld > 0)):
            raise InputError("ldscore must be finite and strictly positive")
        if not (np.all(np.isfinite(n)) and np.all(n > 0)):
            raise InputError("n must be finite and strictly positive")
        ns_f = np.asarray(ns_raw, dtype=np.float64)
        if not np.all(np.isfinite(ns_f)) or np.any(ns_f < 1) or np.any(ns_f != np.round(ns_f)):
            raise InputError("nstar must contain integers >= 1")
        if int(self.M) < 1:
            raise InputError("M must be a positive integer")
        object.__setattr__(self, "betahat", b)
        object.__setattr__(self, "varbetahat", s2)
        object.__setattr__(self, "ldscore", ld)
        object.__setattr__(self, "n", n)
        object.__setattr__(self, "nstar", np.ascontiguousarray(ns_f.astype(np.int64)))
        object.__setattr__(self, "M", int(self.M))
        if self.snp is not None:
            snp = np.asarray(self.snp)
            if snp.size != K:
                raise InputError(f"snp has length {snp.size}, expected {K}")
            object.__setattr__(self, "snp", snp)
        if self.ref_index is not None:
            ri = np.asarray(self.ref_index, dtype=np.int64).ravel()
            if ri.size != K:
                raise InputError(f"ref_index has length {ri.size}, expected {K}")
            object.__setattr__(self, "ref_index", ri)

    @property
    def K(self) -> int:
        """Number of GWAS markers."""
        return int(self.betahat.size)

    @property
    def z(self) -> np.ndarray:
        r"""Z statistics :math:`\hat\beta_k / s_k`."""
        return self.betahat / np.sqrt(self.varbetahat)

    @classmethod
    def from_z(
        cls,
        z: Any,
        n: Any,
        ldscore: Any,
        nstar: Any,
        M: int,
        **kwargs: Any,
    ) -> ModelData:
        r"""Build from Z statistics and sample sizes (:math:`\hat\beta = z/\sqrt n`, :math:`s^2 = 1/n`)."""
        zf = _as_float(z, "z")
        nf = _as_float(n, "n")
        if nf.size == 1 and zf.size != 1:
            nf = np.full(zf.size, float(nf[0]))
        if nf.size != zf.size:
            raise InputError(f"n has length {nf.size}, expected {zf.size} (length of z)")
        if not np.all(nf > 0):
            raise InputError("n must be strictly positive")
        return cls(
            betahat=zf / np.sqrt(nf),
            varbetahat=1.0 / nf,
            ldscore=ldscore,
            nstar=nstar,
            n=nf,
            M=M,
            **kwargs,
        )

    def subset(self, index: Any) -> ModelData:
        """Return a new :class:`ModelData` restricted to ``index`` (mask or integer index)."""
        idx = np.asarray(index)
        return ModelData(
            betahat=self.betahat[idx],
            varbetahat=self.varbetahat[idx],
            ldscore=self.ldscore[idx],
            nstar=self.nstar[idx],
            n=self.n[idx],
            M=self.M,
            snp=None if self.snp is None else self.snp[idx],
            ref_index=None if self.ref_index is None else self.ref_index[idx],
            neighborhoods=self.neighborhoods,
            meta=dict(self.meta),
        )
