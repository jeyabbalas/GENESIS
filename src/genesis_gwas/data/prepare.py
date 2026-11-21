r"""Quality control of summary statistics and merging with an LD reference.

The filters are those recommended by Zhang et al. (2018) and implemented by the
original ``preprocessing(filter=TRUE)``, applied in the same order:

1. **Sample size** — drop SNPs with :math:`N < 0.67 \times` the 90th percentile of
   :math:`N` (poorly covered SNPs in a meta-analysis).
2. **HapMap3 without the MHC** — keep reference SNPs on the HapMap3-no-MHC list (the
   extended LD of the major histocompatibility complex violates the model).
3. **Extreme effects** — drop SNPs with :math:`\chi^2 = z^2 > 80`, which the mixture of
   normals cannot accommodate and which otherwise dominate the variance estimates.

In addition, rows with missing or non-finite ``Z``/``N`` and duplicated SNP identifiers
are always removed (the original crashes or silently misbehaves on those).
"""

from __future__ import annotations

import logging
import os
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd

from genesis_gwas.core.data import ModelData
from genesis_gwas.data.reference import LDReference, load_reference
from genesis_gwas.data.sumstats import SumStats, read_sumstats
from genesis_gwas.exceptions import DataError, DataWarning, InputError

if TYPE_CHECKING:
    from genesis_gwas.core.variance import Neighborhoods

__all__ = ["QCConfig", "prepare"]

log = logging.getLogger("genesis_gwas")


@dataclass(frozen=True)
class QCConfig:
    """Quality-control settings for :func:`prepare`.

    Parameters
    ----------
    n_min_fraction
        Keep SNPs with ``N >= n_min_fraction * quantile(N, n_quantile)``; ``None``
        disables the filter.
    n_quantile
        Quantile of the sample-size distribution used as the yardstick (0.9).
    max_chisq
        Drop SNPs with ``Z**2`` above this value; ``None`` disables the filter.
    hm3_no_mhc
        Restrict to the reference's HapMap3-without-MHC list when the reference
        provides one.
    exclude_mhc
        If the reference has no such list, drop chromosome 6, 26–34 Mb (GRCh37).
    min_info, min_maf
        Optional thresholds on imputation quality and minor-allele frequency; applied
        only when the summary statistics carry ``INFO`` / ``EAF`` columns.
    """

    n_min_fraction: float | None = 0.67
    n_quantile: float = 0.9
    max_chisq: float | None = 80.0
    hm3_no_mhc: bool = True
    exclude_mhc: bool = True
    min_info: float | None = None
    min_maf: float | None = None

    @classmethod
    def none(cls) -> QCConfig:
        """No filtering beyond removing unusable rows (the original ``filter=FALSE``)."""
        return cls(n_min_fraction=None, max_chisq=None, hm3_no_mhc=False, exclude_mhc=False)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable representation."""
        from dataclasses import asdict

        return asdict(self)


_MHC = (6, 26_000_000, 34_000_000)


def _match(
    table: pd.DataFrame, ref: LDReference, match_on: str, build: str | None
) -> tuple[np.ndarray, dict[str, Any]]:
    """Reference row of each summary-statistics row (-1 = not in the reference)."""
    info: dict[str, Any] = {}
    has_pos = "CHR" in table.columns and "POS" in table.columns
    by_id = None
    if match_on in ("auto", "rsid"):
        by_id = ref.lookup(table["SNP"].to_numpy(dtype=str))
        info["matched_by_id"] = int((by_id >= 0).sum())
        if (
            match_on == "rsid"
            or not has_pos
            or info["matched_by_id"] >= 0.5 * min(len(table), ref.M)
        ):
            info["match_on"] = "rsid"
            return by_id, info
    if not has_pos:
        raise DataError(
            "Matching by position needs chromosome and position columns in the summary statistics."
        )
    chrom = table["CHR"].to_numpy()
    pos = np.nan_to_num(table["POS"].to_numpy(dtype=np.float64), nan=-1.0).astype(np.int64)
    candidates = [build] if build is not None else [ref.genome_build, "GRCh38", "GRCh37"]
    best: tuple[int, str, np.ndarray] | None = None
    tried: dict[str, int] = {}
    for b in dict.fromkeys(candidates):
        try:
            rows = ref.lookup_position(chrom, pos, b)
        except Exception:
            continue
        hits = int((rows >= 0).sum())
        tried[str(b)] = hits
        if best is None or hits > best[0]:
            best = (hits, str(b), rows)
    if best is None:
        raise DataError("The reference has no positions in a usable genome build.")
    info["matched_by_position"] = tried
    if by_id is not None and info.get("matched_by_id", 0) >= best[0]:
        info["match_on"] = "rsid"
        return by_id, info
    info["match_on"] = "position"
    info["build"] = best[1]
    return best[2], info


def prepare(
    sumstats: SumStats | pd.DataFrame | str | os.PathLike[str],
    reference: LDReference | str | None = None,
    *,
    qc: QCConfig | bool | None = None,
    match_on: str = "auto",
    build: str | None = None,
    M: int | None = None,
    **read_kwargs: Any,
) -> ModelData:
    r"""Filter summary statistics, merge them with an LD reference and build the model inputs.

    Parameters
    ----------
    sumstats
        A :class:`~genesis_gwas.data.sumstats.SumStats`, a DataFrame, or a file path
        (then ``read_kwargs`` are forwarded to :func:`~genesis_gwas.data.sumstats.read_sumstats`).
    reference
        An :class:`~genesis_gwas.data.reference.LDReference` or anything accepted by
        :func:`~genesis_gwas.data.reference.load_reference` (default: the bundled
        European reference, 1 Mb window, :math:`r^2\ge0.1`).
    qc
        :class:`QCConfig`; ``True``/``None`` for the recommended filters; ``False`` for
        none (equivalent to the original's default ``filter=FALSE``).
    match_on
        ``"rsid"``, ``"position"`` or ``"auto"`` (identifiers first; positions if fewer
        than half of the SNPs match by identifier and positions are available).
    build
        Genome build of the positions in the summary statistics (``"GRCh37"`` or
        ``"GRCh38"``); by default the build giving more matches is used.
    M
        Number of SNPs in the reference panel that :math:`\pi_c` refers to (default: all
        SNPs of the reference).

    Returns
    -------
    ModelData
        Markers are ordered as in the reference (by chromosome and position).
        ``.meta["qc"]`` lists every filtering step with the number of SNPs removed.
    """
    if match_on not in ("auto", "rsid", "position"):
        raise InputError("match_on must be 'auto', 'rsid' or 'position'")
    if isinstance(sumstats, SumStats):
        if read_kwargs:
            raise InputError(
                f"Unexpected arguments for an in-memory SumStats: {sorted(read_kwargs)}"
            )
        ss = sumstats
    else:
        ss = read_sumstats(sumstats, **read_kwargs)
    ref = (
        reference
        if isinstance(reference, LDReference)
        else load_reference(reference if reference is not None else "eur_1kg_hm3")
    )
    if qc is None or qc is True:
        cfg = QCConfig()
    elif qc is False:
        cfg = QCConfig.none()
    else:
        cfg = qc

    table = ss.table
    steps: list[dict[str, Any]] = []
    n_start = len(table)

    def apply(keep: np.ndarray, label: str) -> None:
        nonlocal table
        removed = int((~keep).sum())
        steps.append({"step": label, "removed": removed, "remaining": int(keep.sum())})
        if removed:
            log.info(
                "QC: %-58s removed %10s, %10s remain", label, f"{removed:,}", f"{int(keep.sum()):,}"
            )
            table = table.loc[keep]

    zv = table["Z"].to_numpy(dtype=np.float64)
    nv = table["N"].to_numpy(dtype=np.float64)
    apply(np.isfinite(zv) & np.isfinite(nv) & (nv > 0), "missing / non-finite Z or N, or N <= 0")
    apply(
        ~table["SNP"].duplicated(keep="first").to_numpy(), "duplicated SNP identifier (first kept)"
    )

    if cfg.n_min_fraction is not None and len(table):
        nv = table["N"].to_numpy(dtype=np.float64)
        threshold = cfg.n_min_fraction * float(np.quantile(nv, cfg.n_quantile))
        apply(
            nv >= threshold,
            f"N < {cfg.n_min_fraction:g} x {cfg.n_quantile:g}-quantile of N ({threshold:,.0f})",
        )
    if cfg.min_info is not None and "INFO" in table.columns:
        apply(table["INFO"].to_numpy(dtype=np.float64) >= cfg.min_info, f"INFO < {cfg.min_info:g}")
    if cfg.min_maf is not None and "EAF" in table.columns:
        f = table["EAF"].to_numpy(dtype=np.float64)
        apply(np.minimum(f, 1.0 - f) >= cfg.min_maf, f"MAF < {cfg.min_maf:g}")

    rows, match_info = _match(table, ref, match_on, build)
    apply(rows >= 0, f"not in the LD reference ({ref.name}, matched on {match_info['match_on']})")
    rows = rows[rows >= 0]

    if len(rows):
        first = np.zeros(rows.size, dtype=bool)
        first[np.unique(rows, return_index=True)[1]] = True
        apply(first, "several input SNPs map to the same reference SNP (first kept)")
        rows = rows[first]

    if cfg.hm3_no_mhc and ref.hm3_no_mhc is not None:
        keep = ref.hm3_no_mhc[rows]
        apply(keep, "outside the HapMap3-without-MHC list")
        rows = rows[keep]
    elif cfg.exclude_mhc:
        if ref.genome_build.lower() in ("grch37", "hg19"):
            keep = ~(
                (ref.chrom[rows] == _MHC[0])
                & (ref.pos[rows] >= _MHC[1])
                & (ref.pos[rows] <= _MHC[2])
            )
            apply(keep, "MHC region (chr6:26-34 Mb)")
            rows = rows[keep]
        else:
            warnings.warn(
                f"MHC exclusion skipped: reference build {ref.genome_build} is not GRCh37 and the "
                "reference has no HapMap3-without-MHC list.",
                DataWarning,
                stacklevel=2,
            )

    if cfg.max_chisq is not None and len(table):
        zv = table["Z"].to_numpy(dtype=np.float64)
        keep = zv * zv <= cfg.max_chisq
        apply(keep, f"chi-square > {cfg.max_chisq:g}")
        rows = rows[keep]

    if len(table) == 0:
        raise DataError(
            f"No SNPs left after QC and merging with reference {ref.name!r} "
            f"(started with {n_start:,}). Check the SNP identifiers / genome build; "
            f"matching details: {match_info}."
        )
    frac = len(table) / ref.M
    if frac < 0.5:
        warnings.warn(
            f"Only {len(table):,} of the {ref.M:,} reference SNPs ({frac:.0%}) are present in the "
            "summary statistics after QC. Estimates are valid but less precise, and standard "
            "errors rely on fewer neighbouring markers; check identifiers and genome build if "
            "this is unexpected.",
            DataWarning,
            stacklevel=2,
        )

    order = np.argsort(rows, kind="stable")
    rows = rows[order]
    zv = table["Z"].to_numpy(dtype=np.float64)[order]
    nv = table["N"].to_numpy(dtype=np.float64)[order]
    meta = {
        "source": ss.source,
        "sumstats": ss.info,
        "reference": {
            "name": ref.name,
            "version": ref.version,
            "config": ref.config.id,
            "population": ref.population,
            "genome_build": ref.genome_build,
        },
        "matching": match_info,
        "qc_config": cfg.to_dict(),
        "qc": steps,
        "n_input": n_start,
        "n_final": len(rows),
    }
    log.info("Prepared %s SNPs for analysis (of %s read)", f"{len(rows):,}", f"{n_start:,}")
    return ModelData.from_z(
        z=zv,
        n=nv,
        ldscore=ref.ldscore[rows],
        nstar=ref.nstar[rows],
        M=int(M) if M is not None else ref.M,
        snp=ref.subset_ids(rows),
        ref_index=rows,
        neighborhoods=cast("Neighborhoods", _LazyNeighborhoods(ref)),
        meta=meta,
    )


class _LazyNeighborhoods:
    """Defers loading the (large) tagging structure until the variance step needs it."""

    def __init__(self, ref: LDReference) -> None:
        self._ref = ref

    def __getattr__(self, name: str) -> Any:
        return getattr(self._ref.neighborhoods, name)
