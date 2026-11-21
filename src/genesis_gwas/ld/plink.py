r"""Minimal, dependency-free reader for PLINK 1 binary filesets (``.bed``/``.bim``/``.fam``).

Only what the LD builder needs is implemented: the variant and sample tables, and fast
decoding of selected variants into allele dosages.

The ``.bed`` format
-------------------
A ``.bed`` file starts with three bytes: the magic number ``0x6c 0x1b`` and a mode byte
(``0x01`` = variant-major, the only layout PLINK has written for many years). In
variant-major mode each variant occupies :math:`\lceil n/4\rceil` bytes for :math:`n`
samples. Every byte holds four genotypes, **lowest-order bit pair first**, so sample
:math:`i` of a variant sits in byte :math:`\lfloor i/4\rfloor` at bits
:math:`2(i \bmod 4)` and :math:`2(i \bmod 4)+1`. The two-bit code means

====  =====================================================
code  genotype
====  =====================================================
00    homozygous for the first allele of the ``.bim`` (A1)
01    missing
10    heterozygous
11    homozygous for the second allele of the ``.bim`` (A2)
====  =====================================================

(the code is the numeric value ``(byte >> 2j) & 3``, i.e. "01" is the value 1). Padding
bits at the end of a variant's last byte are ignored.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from genesis_gwas.exceptions import DataError, InputError

__all__ = ["BedReader", "read_bim", "read_fam", "resolve_fileset"]

_MAGIC = b"\x6c\x1b"
_MODE_VARIANT_MAJOR = 0x01
MISSING = -1


def _build_lookup(count_a1: bool) -> np.ndarray:
    """256 x 4 table mapping a ``.bed`` byte to the four genotypes it encodes."""
    code = np.array([2, MISSING, 1, 0] if count_a1 else [0, MISSING, 1, 2], dtype=np.int8)
    byte = np.arange(256, dtype=np.uint16)
    table = np.empty((256, 4), dtype=np.int8)
    for j in range(4):
        table[:, j] = code[(byte >> (2 * j)) & 3]
    return table


_LOOKUP = {True: _build_lookup(True), False: _build_lookup(False)}


def resolve_fileset(prefix: str | os.PathLike[str]) -> tuple[Path, Path, Path]:
    """Return the ``(.bed, .bim, .fam)`` paths of a PLINK fileset prefix.

    ``prefix`` may be given with or without one of the three extensions.

    Raises
    ------
    DataError
        If any of the three files is missing.
    """
    p = Path(os.fspath(prefix)).expanduser()
    if p.suffix in (".bed", ".bim", ".fam"):
        p = p.with_suffix("")
    paths = tuple(Path(str(p) + ext) for ext in (".bed", ".bim", ".fam"))
    missing = [str(q) for q in paths if not q.exists()]
    if missing:
        raise DataError(f"PLINK fileset {p} is incomplete; missing: {', '.join(missing)}")
    return paths  # type: ignore[return-value]


def read_bim(path: str | os.PathLike[str]) -> pd.DataFrame:
    """Read a PLINK ``.bim`` variant table.

    Returns
    -------
    pandas.DataFrame
        Columns ``chrom`` (str), ``snp`` (str), ``cm`` (float), ``pos`` (int64),
        ``a1`` (str), ``a2`` (str); one row per variant, in file order (which is the
        order of the variants in the ``.bed`` file).
    """
    p = Path(os.fspath(path)).expanduser()
    try:
        df = pd.read_csv(
            p,
            sep=r"\s+",
            header=None,
            names=["chrom", "snp", "cm", "pos", "a1", "a2"],
            dtype={
                "chrom": str,
                "snp": str,
                "cm": np.float64,
                "pos": np.int64,
                "a1": str,
                "a2": str,
            },
            na_filter=False,
            engine="c",
        )
    except Exception as exc:
        raise DataError(f"Cannot parse PLINK .bim file {p}: {exc}") from exc
    if df.shape[1] != 6:
        raise DataError(f"{p} does not have the six columns of a PLINK .bim file")
    return df


def read_fam(path: str | os.PathLike[str]) -> pd.DataFrame:
    """Read a PLINK ``.fam`` sample table.

    Returns
    -------
    pandas.DataFrame
        Columns ``fid``, ``iid``, ``pid``, ``mid``, ``sex``, ``pheno`` (all str); one
        row per sample, in file order (the order of the samples in the ``.bed`` file).
    """
    p = Path(os.fspath(path)).expanduser()
    try:
        df = pd.read_csv(
            p,
            sep=r"\s+",
            header=None,
            names=["fid", "iid", "pid", "mid", "sex", "pheno"],
            dtype=str,
            na_filter=False,
            engine="c",
        )
    except Exception as exc:
        raise DataError(f"Cannot parse PLINK .fam file {p}: {exc}") from exc
    return df


class BedReader:
    """Random-access decoder for a variant-major PLINK ``.bed`` file.

    Parameters
    ----------
    path
        The ``.bed`` file.
    n_samples, n_variants
        Number of rows of the ``.fam`` and ``.bim`` files. The file size is validated
        against them, which catches a ``.bed`` paired with the wrong ``.bim``/``.fam``.

    Notes
    -----
    The file is memory-mapped; nothing is read until :meth:`read` is called, and each
    call touches only the bytes of the requested variants.
    """

    def __init__(self, path: str | os.PathLike[str], n_samples: int, n_variants: int) -> None:
        self.path = Path(os.fspath(path)).expanduser()
        self.n_samples = int(n_samples)
        self.n_variants = int(n_variants)
        if self.n_samples < 1 or self.n_variants < 0:
            raise InputError("n_samples must be >= 1 and n_variants >= 0")
        if not self.path.exists():
            raise DataError(f"PLINK .bed file not found: {self.path}")
        with self.path.open("rb") as fh:
            head = fh.read(3)
        if len(head) < 3 or head[:2] != _MAGIC:
            raise DataError(
                f"{self.path} is not a PLINK 1 .bed file (bad magic number; "
                "PLINK 2 .pgen files must be converted with `plink2 --make-bed`)."
            )
        if head[2] != _MODE_VARIANT_MAJOR:
            raise DataError(
                f"{self.path} is a sample-major .bed file, which is not supported; "
                "rewrite it with `plink --bfile X --make-bed --out Y`."
            )
        self.bytes_per_variant = (self.n_samples + 3) // 4
        expected = 3 + self.bytes_per_variant * self.n_variants
        actual = self.path.stat().st_size
        if actual != expected:
            raise DataError(
                f"{self.path} has {actual:,} bytes but {self.n_variants:,} variants x "
                f"{self.n_samples:,} samples need {expected:,}; the .bed does not match its "
                ".bim/.fam files."
            )
        self._mm: np.ndarray | None = None

    @classmethod
    def from_prefix(
        cls, prefix: str | os.PathLike[str]
    ) -> tuple[BedReader, pd.DataFrame, pd.DataFrame]:
        """Open a fileset by prefix; returns ``(reader, bim, fam)``."""
        bed, bim_path, fam_path = resolve_fileset(prefix)
        bim = read_bim(bim_path)
        fam = read_fam(fam_path)
        return cls(bed, len(fam), len(bim)), bim, fam

    def _data(self) -> np.ndarray:
        if self._mm is None:
            if self.n_variants == 0:
                self._mm = np.empty((0, self.bytes_per_variant), dtype=np.uint8)
            else:
                self._mm = np.memmap(
                    self.path,
                    dtype=np.uint8,
                    mode="r",
                    offset=3,
                    shape=(self.n_variants, self.bytes_per_variant),
                )
        return self._mm

    def read(
        self,
        variants: Any = None,
        *,
        samples: Any = None,
        count_a1: bool = True,
        chunk: int = 8192,
    ) -> np.ndarray:
        """Decode genotypes as allele dosages.

        Parameters
        ----------
        variants
            Integer indices of the variants to read (rows of the ``.bim``), in the
            order wanted in the output; ``None`` reads all of them.
        samples
            Integer indices of the samples to keep (rows of the ``.fam``), or ``None``
            for all.
        count_a1
            If ``True`` (default) the dosage counts copies of the **first** ``.bim``
            allele (A1, usually the minor/effect allele): code ``00`` → 2, ``10`` → 1,
            ``11`` → 0. If ``False`` it counts the second allele (A2). The two choices
            differ only by ``2 - x`` and give identical LD.
        chunk
            Number of variants decoded at a time (bounds temporary memory).

        Returns
        -------
        numpy.ndarray
            ``int8`` array of shape ``(n_selected_variants, n_selected_samples)``;
            one row per variant ("variant-major"). Missing genotypes are ``-1``.
        """
        if variants is None:
            idx = np.arange(self.n_variants, dtype=np.int64)
        else:
            idx = np.asarray(variants, dtype=np.int64).ravel()
            if idx.size and (int(np.min(idx)) < 0 or int(np.max(idx)) >= self.n_variants):
                raise InputError(
                    f"variant indices must lie in [0, {self.n_variants}); "
                    f"got range [{int(np.min(idx))}, {int(np.max(idx))}]"
                )
        if samples is None:
            cols: np.ndarray | None = None
            n_out = self.n_samples
        else:
            cols = np.asarray(samples, dtype=np.int64).ravel()
            if cols.size and (int(np.min(cols)) < 0 or int(np.max(cols)) >= self.n_samples):
                raise InputError(f"sample indices must lie in [0, {self.n_samples})")
            n_out = cols.size
        lut = _LOOKUP[bool(count_a1)]
        data = self._data()
        out = np.empty((idx.size, n_out), dtype=np.int8)
        step = max(1, int(chunk))
        for start in range(0, idx.size, step):
            sel = idx[start : start + step]
            raw = np.asarray(data[sel])  # (c, bytes_per_variant) copy of the needed bytes
            geno = lut[raw].reshape(sel.size, -1)[:, : self.n_samples]
            out[start : start + sel.size] = geno if cols is None else geno[:, cols]
        return out

    def close(self) -> None:
        """Release the memory map."""
        self._mm = None

    def __enter__(self) -> BedReader:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
