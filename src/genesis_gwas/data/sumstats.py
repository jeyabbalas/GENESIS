r"""Reading GWAS summary statistics.

GENESIS needs exactly three things per SNP: an identifier, a Z statistic and a sample
size. Real files spell these in dozens of ways, so :func:`read_sumstats`

* detects the delimiter and (optionally compressed) format,
* recognises common column names case-insensitively (overridable one by one),
* derives ``Z`` from whatever is available — ``Z``; ``BETA/SE``; ``log(OR)/SE``; or a
  signed effect together with a P-value,
* derives ``N`` from a column, from constants, or — for case–control studies — the
  *effective* sample size.

.. important::

   **Binary traits.** GENESIS standardises effects as :math:`\hat\beta = z/\sqrt{n}`, so
   for a case–control GWAS ``n`` must be the effective sample size

   .. math:: n_\text{eff} = \frac{n_\text{case}\, n_\text{control}}{n_\text{case}+n_\text{control}},

   which makes :math:`\hat\beta` a log odds ratio per standard deviation of genotype
   and puts heritability on the log-odds-ratio scale. This is the convention of the
   original software. It is **one quarter** of the "effective N" reported by METAL or
   the PGC (:math:`4/(1/n_\text{case}+1/n_\text{control})`): if your file carries that
   quantity, divide it by four (``n_scale=0.25``).
"""

from __future__ import annotations

import gzip
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import ndtri

from genesis_gwas.exceptions import DataError, InputError

__all__ = ["COLUMN_ALIASES", "SumStats", "read_sumstats"]

log = logging.getLogger("genesis_gwas")

#: Recognised spellings (lower-cased) of each logical column.
COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "SNP": (
        "snp",
        "rsid",
        "rs",
        "rs_id",
        "rsids",
        "markername",
        "marker",
        "snpid",
        "id",
        "variant_id",
        "variant",
        "snp_id",
        "rs_number",
        "hm_rsid",
    ),
    "Z": ("z", "zscore", "z_score", "z-score", "zstat", "z_stat", "tstat", "t_stat", "stat"),
    "BETA": (
        "beta",
        "b",
        "effect",
        "effects",
        "log_or",
        "logor",
        "log_odds",
        "est",
        "estimate",
        "effect_size",
        "hm_beta",
    ),
    "SE": (
        "se",
        "stderr",
        "std_err",
        "standard_error",
        "se_beta",
        "sebeta",
        "stderrlogor",
        "log_or_se",
    ),
    "OR": ("or", "odds_ratio", "oddsratio", "hm_odds_ratio"),
    "P": (
        "p",
        "pval",
        "p_val",
        "pvalue",
        "p_value",
        "p-value",
        "p.value",
        "gc_pvalue",
        "pval_nominal",
    ),
    "N": (
        "n",
        "neff",
        "n_eff",
        "n_effective",
        "samplesize",
        "sample_size",
        "n_total",
        "ntotal",
        "weight",
        "n_complete_samples",
        "totalsamplesize",
        "obs_ct",
    ),
    "NCASE": ("n_cases", "ncase", "ncases", "n_case", "n_cas", "nca", "cases", "n_cases_total"),
    "NCONTROL": (
        "n_controls",
        "ncontrol",
        "ncontrols",
        "n_control",
        "n_con",
        "nco",
        "controls",
        "n_controls_total",
    ),
    "CHR": ("chr", "chrom", "chromosome", "#chrom", "#chr", "hm_chrom", "chr_name"),
    "POS": (
        "pos",
        "bp",
        "position",
        "base_pair_location",
        "base_pair",
        "hm_pos",
        "chr_position",
        "genpos",
        "pos_b37",
        "pos_b38",
    ),
    "A1": ("a1", "effect_allele", "allele1", "ea", "alt", "tested_allele", "hm_effect_allele"),
    "A2": (
        "a2",
        "other_allele",
        "allele2",
        "nea",
        "ref",
        "non_effect_allele",
        "hm_other_allele",
        "allele0",
    ),
    "EAF": (
        "eaf",
        "frq",
        "freq",
        "af",
        "maf",
        "effect_allele_frequency",
        "a1freq",
        "freq1",
        "frq_u",
        "af_alt",
        "a1_freq",
    ),
    "INFO": ("info", "imputation_info", "impinfo", "r2", "rsq", "info_score"),
}


@dataclass
class SumStats:
    """Standardised summary statistics.

    Attributes
    ----------
    table
        DataFrame with at least ``SNP`` (str), ``Z`` (float) and ``N`` (float), plus any
        of ``CHR``, ``POS``, ``A1``, ``A2``, ``EAF``, ``INFO`` that the file provided.
    info
        How the columns were obtained (source columns, derivations, row counts).
    source
        File path or description of the origin.
    """

    table: pd.DataFrame
    info: dict[str, Any] = field(default_factory=dict)
    source: str = "<memory>"

    def __len__(self) -> int:
        return len(self.table)

    @classmethod
    def from_arrays(cls, snp: Any, z: Any, n: Any, **extra: Any) -> SumStats:
        """Build from in-memory arrays (``n`` may be a scalar)."""
        snp_a = np.asarray(snp).astype(str)
        z_a = np.asarray(z, dtype=np.float64)
        n_a = np.broadcast_to(np.asarray(n, dtype=np.float64), z_a.shape).copy()
        if snp_a.shape != z_a.shape:
            raise InputError("snp and z must have the same length")
        df = pd.DataFrame({"SNP": snp_a, "Z": z_a, "N": n_a})
        for key, val in extra.items():
            df[key.upper()] = np.asarray(val)
        return cls(table=df, info={"z_from": "Z", "n_from": "N", "rows_read": len(df)})


def _open_text(path: Path) -> Any:
    """Open a possibly compressed text file (same formats as :func:`pandas.read_csv` infers)."""
    suffix = path.suffix.lower()
    if suffix in (".gz", ".bgz"):
        return gzip.open(path, "rt")
    if suffix == ".bz2":
        import bz2

        return bz2.open(path, "rt")
    if suffix == ".xz":
        import lzma

        return lzma.open(path, "rt")
    if suffix == ".zst":
        try:
            import zstandard
        except ImportError as exc:
            raise DataError(
                f"{path} is zstd-compressed; install the 'zstandard' package or recompress it "
                "with gzip."
            ) from exc
        return zstandard.open(path, "rt")
    return path.open("rt")


def _sniff_delimiter(path: Path) -> str:
    try:
        with _open_text(path) as fh:
            header = fh.readline()
    except (OSError, UnicodeDecodeError) as exc:
        raise DataError(f"Cannot read {path}: {exc}") from exc
    if not header.strip():
        raise DataError(f"{path} is empty")
    if "\t" in header:
        return "\t"
    if "," in header and " " not in header.strip():
        return ","
    return r"\s+"


def _resolve_columns(available: list[str], overrides: dict[str, str | None]) -> dict[str, str]:
    lower = {c.lower(): c for c in available}
    found: dict[str, str] = {}
    for logical, aliases in COLUMN_ALIASES.items():
        if logical in overrides:
            want = overrides[logical]
            if want is None:
                continue  # explicitly disabled
            if want not in available:
                if want.lower() in lower:
                    want = lower[want.lower()]
                else:
                    raise InputError(
                        f"Column {want!r} (given for {logical}) is not in the file. "
                        f"Columns present: {available}"
                    )
            found[logical] = want
            continue
        for alias in aliases:
            if alias in lower:
                found[logical] = lower[alias]
                break
    return found


def read_sumstats(
    path: str | os.PathLike[str] | pd.DataFrame,
    *,
    snp: str | None = None,
    z: str | None = None,
    n: str | float | None = None,
    beta: str | None = None,
    se: str | None = None,
    odds_ratio: str | None = None,
    p: str | None = None,
    n_cases: str | float | None = None,
    n_controls: str | float | None = None,
    case_fraction: float | None = None,
    n_scale: float = 1.0,
    n_from_se: bool = False,
    chrom: str | None = None,
    pos: str | None = None,
    eaf: str | None = None,
    info: str | None = None,
    sep: str | None = None,
    nrows: int | None = None,
) -> SumStats:
    r"""Read GWAS summary statistics into the standard ``SNP``/``Z``/``N`` form.

    Parameters
    ----------
    path
        Delimited text file (optionally ``.gz``/``.bz2``/``.xz``/``.zst``) or an
        in-memory :class:`pandas.DataFrame`.
    snp, z, beta, se, odds_ratio, p, chrom, pos, eaf, info
        Column names, when automatic detection (see :data:`COLUMN_ALIASES`) does not find
        the right one.
    n
        Column name **or** a number (the same sample size for every SNP).
    n_cases, n_controls
        Column names or numbers. When both are available the effective sample size
        :math:`n_\text{case} n_\text{control}/(n_\text{case}+n_\text{control})` is used.
    case_fraction
        For a binary trait whose file reports the *total* sample size: the fraction of
        cases :math:`\phi`; ``N`` becomes :math:`N\phi(1-\phi)`.
    n_scale
        Multiply the sample size by this factor (e.g. ``0.25`` to convert a METAL/PGC
        style effective N to the GENESIS convention).
    n_from_se
        Derive a per-SNP effective sample size from the standard error and the allele
        frequency, :math:`n_k = 1/\{2 f_k(1-f_k)\,\mathrm{SE}_k^2\}`. Appropriate for
        log-odds effects from case–control meta-analyses in which per-SNP sample sizes
        are not reported; requires ``SE`` and ``EAF``.
    sep
        Field delimiter; detected from the header line by default.
    nrows
        Read only the first ``nrows`` rows (debugging).

    Returns
    -------
    SumStats

    Raises
    ------
    DataError
        If ``Z`` or ``N`` cannot be derived from the columns present.
    """
    overrides: dict[str, str | None] = {}
    for key, val in (
        ("SNP", snp),
        ("Z", z),
        ("BETA", beta),
        ("SE", se),
        ("OR", odds_ratio),
        ("P", p),
        ("CHR", chrom),
        ("POS", pos),
        ("EAF", eaf),
        ("INFO", info),
    ):
        if val is not None:
            overrides[key] = val
    n_const: float | None = None
    if isinstance(n, str):
        overrides["N"] = n
    elif n is not None:
        n_const = float(n)
    const_cc: dict[str, float] = {}
    cc_options: tuple[tuple[str, str | float | None], ...] = (
        ("NCASE", n_cases),
        ("NCONTROL", n_controls),
    )
    for key, cc_val in cc_options:
        if isinstance(cc_val, str):
            overrides[key] = cc_val
        elif cc_val is not None:
            const_cc[key] = float(cc_val)

    if isinstance(path, pd.DataFrame):
        source = "<DataFrame>"
        available = [str(c) for c in path.columns]
        cols = _resolve_columns(available, overrides)
        raw = path[list(dict.fromkeys(cols.values()))].copy()
    else:
        fpath = Path(os.fspath(path)).expanduser()
        if not fpath.exists():
            raise DataError(f"Summary-statistics file not found: {fpath}")
        source = str(fpath)
        delim = sep if sep is not None else _sniff_delimiter(fpath)
        try:
            header = pd.read_csv(
                fpath, sep=delim, nrows=0, engine="c" if delim != r"\s+" else "python"
            )
        except Exception as exc:
            raise DataError(f"Cannot parse the header of {fpath}: {exc}") from exc
        available = [str(c) for c in header.columns]
        cols = _resolve_columns(available, overrides)
        use = list(dict.fromkeys(cols.values()))
        if not use:
            raise DataError(f"No recognisable columns in {fpath}. Columns present: {available}")
        try:
            raw = pd.read_csv(
                fpath,
                sep=delim,
                usecols=use,
                nrows=nrows,
                dtype={cols["SNP"]: str} if "SNP" in cols else None,
                na_values=["NA", "NaN", "nan", ".", "", "NULL", "null", "-"],
                keep_default_na=True,
                low_memory=False,
            )
        except Exception as exc:
            raise DataError(f"Cannot parse {fpath}: {exc}") from exc

    def num(logical: str) -> np.ndarray:
        return pd.to_numeric(raw[cols[logical]], errors="coerce").to_numpy(dtype=np.float64)

    details: dict[str, Any] = {"rows_read": len(raw), "columns": dict(cols)}
    out = pd.DataFrame(index=raw.index)
    if "SNP" in cols:
        out["SNP"] = raw[cols["SNP"]].astype(str)
    if "CHR" in cols:
        out["CHR"] = raw[cols["CHR"]]
    if "POS" in cols:
        out["POS"] = pd.to_numeric(raw[cols["POS"]], errors="coerce")
    if "SNP" not in cols and not ("CHR" in cols and "POS" in cols):
        raise DataError(
            "Need a SNP identifier column (or chromosome and position columns). "
            f"Columns present: {available}. Use snp=... to name it."
        )
    if "SNP" not in cols:
        chrom_txt = out["CHR"].astype(str).str.replace("chr", "", case=False, regex=False)
        out["SNP"] = chrom_txt + ":" + out["POS"].astype("Int64").astype(str)

    # ------------------------------------------------------------------ Z
    with np.errstate(divide="ignore", invalid="ignore"):
        if "Z" in cols:
            zval = num("Z")
            details["z_from"] = cols["Z"]
        elif "BETA" in cols and "SE" in cols:
            zval = num("BETA") / num("SE")
            details["z_from"] = f"{cols['BETA']} / {cols['SE']}"
        elif "OR" in cols and "SE" in cols:
            zval = np.log(num("OR")) / num("SE")
            details["z_from"] = f"log({cols['OR']}) / {cols['SE']}"
        elif "P" in cols and ("BETA" in cols or "OR" in cols):
            sign = np.sign(num("BETA")) if "BETA" in cols else np.sign(np.log(num("OR")))
            pv = num("P")
            zval = sign * (-ndtri(pv / 2.0))
            details["z_from"] = (
                f"sign({cols.get('BETA', cols.get('OR'))}) * |Phi^-1({cols['P']}/2)|"
            )
        else:
            raise DataError(
                "Cannot derive Z statistics: need a Z column, BETA and SE, OR and SE, or a "
                f"P-value with a signed effect. Columns recognised: {cols}; present: {available}."
            )
    out["Z"] = zval

    # ------------------------------------------------------------------ N
    # Precedence: what the caller asked for explicitly, then case/control counts found
    # in the file (the effective sample size is the right quantity for a binary trait,
    # whereas a bare "N" column next to them is usually the *total* sample size), then
    # an N column.
    explicit_n = n is not None
    explicit_cc = n_cases is not None or n_controls is not None
    if explicit_n and explicit_cc:
        raise InputError("Give either n or n_cases/n_controls, not both.")
    have_cc = ("NCASE" in cols or "NCASE" in const_cc) and (
        "NCONTROL" in cols or "NCONTROL" in const_cc
    )
    if explicit_cc and not have_cc:
        raise DataError(
            "Both case and control counts are needed (columns or numbers); "
            f"columns recognised: {cols}; present: {available}."
        )
    with np.errstate(divide="ignore", invalid="ignore"):
        if n_from_se:
            if "SE" not in cols or "EAF" not in cols:
                raise DataError("n_from_se=True needs SE and allele-frequency (EAF) columns.")
            f = num("EAF")
            nval = 1.0 / (2.0 * f * (1.0 - f) * num("SE") ** 2)
            details["n_from"] = f"1 / (2 f (1 - f) SE^2) with f = {cols['EAF']}, SE = {cols['SE']}"
        elif explicit_n and n_const is not None:
            nval = np.full(len(raw), n_const)
            details["n_from"] = f"constant {n_const:g}"
        elif explicit_n:
            nval = num("N")
            details["n_from"] = cols["N"]
        elif have_cc:
            nca = np.full(len(raw), const_cc["NCASE"]) if "NCASE" in const_cc else num("NCASE")
            nco = (
                np.full(len(raw), const_cc["NCONTROL"])
                if "NCONTROL" in const_cc
                else num("NCONTROL")
            )
            nval = nca * nco / (nca + nco)
            details["n_from"] = "effective N = n_cases * n_controls / (n_cases + n_controls)"
            if "N" in cols and not explicit_cc:
                log.warning(
                    "Found case/control count columns (%s, %s) as well as %r: using the effective "
                    "sample size n_cases*n_controls/(n_cases+n_controls). Pass n=%r to use that "
                    "column as is.",
                    cols.get("NCASE"),
                    cols.get("NCONTROL"),
                    cols["N"],
                    cols["N"],
                )
        elif "N" in cols:
            nval = num("N")
            details["n_from"] = cols["N"]
        else:
            raise DataError(
                "Cannot determine sample sizes: need an N column, n=<number>, case/control "
                f"counts, or n_from_se=True. Columns recognised: {cols}; present: {available}."
            )
        if case_fraction is not None:
            if not 0.0 < case_fraction < 1.0:
                raise InputError("case_fraction must lie in (0, 1)")
            nval = nval * case_fraction * (1.0 - case_fraction)
            details["n_from"] += f" x {case_fraction:g} x {1 - case_fraction:g} (case fraction)"
        if n_scale != 1.0:
            if not n_scale > 0:
                raise InputError("n_scale must be > 0")
            nval = nval * n_scale
            details["n_from"] += f" x {n_scale:g}"
    out["N"] = nval

    for logical in ("EAF", "INFO"):
        if logical in cols:
            out[logical] = num(logical)
    for logical in ("A1", "A2"):
        if logical in cols:
            out[logical] = raw[cols[logical]].astype(str).str.upper()
    out = out.reset_index(drop=True)
    log.info(
        "Read %s rows from %s (Z from %s; N from %s)",
        f"{len(out):,}",
        source,
        details["z_from"],
        details["n_from"],
    )
    return SumStats(table=out, info=details, source=source)
