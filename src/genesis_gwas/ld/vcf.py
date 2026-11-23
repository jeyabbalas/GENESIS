r"""Convert VCF genotypes to PLINK 1 binary filesets, without external tools.

Public reference panels (the 1000 Genomes Project, HGDP, …) are distributed as VCF files
with tens of millions of variants and all populations together, while a GENESIS reference
needs about a million SNPs in the samples of one population. :func:`vcf_to_plink` streams
a VCF once, keeps the biallelic SNPs on a list and the samples on a list, and writes the
``.bed``/``.bim``/``.fam`` fileset that :func:`genesis_gwas.ld.build_reference` reads.
Memory use is a few megabytes regardless of the size of the VCF.

Only hard-called diploid genotypes (the ``GT`` field) are read. What is kept and what is
set aside:

* kept: records with one single-base ``REF`` and one single-base ``ALT`` allele on
  chromosomes 1–22;
* set aside, and counted in the returned summary: indels and structural variants,
  multi-allelic records, non-autosomal records, records whose identifier (or position)
  is not on the list given, and second and later records with an identifier (or
  position) that was already written;
* a genotype with a missing or non-reference/alternate allele, or a haploid genotype, is
  written as missing.

In the ``.bim`` the first allele (A1) is the VCF ``ALT`` allele and the second (A2) the
``REF`` allele, the convention of ``plink2 --vcf``.
"""

from __future__ import annotations

import gzip
import logging
import os
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import IO, Any, cast

import numpy as np

from genesis_gwas.exceptions import DataError, InputError

__all__ = ["vcf_to_plink"]

log = logging.getLogger("genesis_gwas")

_BED_HEADER = b"\x6c\x1b\x01"
_BASES = frozenset((b"A", b"C", b"G", b"T"))
# ASCII '0' = 48, '1' = 49, '.' = 46
_ZERO, _ONE = 48, 49


def _open_vcf(path: Path) -> IO[bytes]:
    with path.open("rb") as fh:
        magic = fh.read(2)
    if magic == b"\x1f\x8b":
        # also reads bgzip (concatenated gzip members)
        return cast("IO[bytes]", gzip.open(path, "rb"))
    return path.open("rb")


def _autosome(label: bytes) -> int:
    txt = label[3:] if label[:3].lower() == b"chr" else label
    if txt.isdigit():
        c = int(txt)
        if 1 <= c <= 22:
            return c
    return 0


def _codes_fixed(buf: bytes, n: int) -> np.ndarray | None:
    """PLINK codes for ``n`` genotypes laid out as ``a|b<TAB>`` (4 bytes each); else ``None``."""
    if len(buf) != 4 * n - 1:
        return None
    raw = np.frombuffer(buf + b"\t", dtype=np.uint8).reshape(n, 4)
    sep = raw[:, 1]
    if not (np.all((sep == 124) | (sep == 47)) and np.all(raw[:, 3] == 9)):
        return None
    return _codes_from_alleles(raw[:, 0], raw[:, 2])


def _codes_from_alleles(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Two-bit PLINK codes (A1 = ALT): 0 hom ALT, 2 het, 3 hom REF, 1 missing."""
    ok = ((a == _ZERO) | (a == _ONE)) & ((b == _ZERO) | (b == _ONE))
    alt = (a == _ONE).astype(np.uint8) + (b == _ONE).astype(np.uint8)
    code = np.array([3, 2, 0], dtype=np.uint8)[alt]
    code[~ok] = 1
    return code


def _codes_general(fields: Sequence[bytes], gt_index: int) -> np.ndarray:
    """PLINK codes from arbitrary sample fields (slow path)."""
    n = len(fields)
    a = np.full(n, 46, dtype=np.uint8)
    b = np.full(n, 46, dtype=np.uint8)
    for i, f in enumerate(fields):
        gt = f.split(b":")[gt_index] if gt_index or b":" in f else f
        if len(gt) == 3 and gt[1] in (124, 47):
            a[i] = gt[0]
            b[i] = gt[2]
    return _codes_from_alleles(a, b)


def _pack(codes: np.ndarray) -> bytes:
    """Pack two-bit codes into ``.bed`` bytes (lowest-order pair first)."""
    n = codes.size
    pad = (-n) % 4
    if pad:
        codes = np.concatenate([codes, np.zeros(pad, dtype=np.uint8)])
    q = codes.reshape(-1, 4)
    return (q[:, 0] | (q[:, 1] << 2) | (q[:, 2] << 4) | (q[:, 3] << 6)).astype(np.uint8).tobytes()


def _convert_one(
    vcf: str,
    out_prefix: str,
    snps: frozenset[bytes] | None,
    keep: tuple[str, ...] | None,
    chromosomes: tuple[int, ...],
    positions: dict[tuple[int, int], bytes] | None = None,
) -> dict[str, Any]:
    """Convert one VCF file; returns a summary dictionary."""
    path = Path(vcf)
    if not path.exists():
        raise DataError(f"VCF file not found: {path}")
    out = Path(out_prefix)
    out.parent.mkdir(parents=True, exist_ok=True)
    counts = {
        "records": 0,
        "written": 0,
        "not_listed": 0,
        "not_autosomal": 0,
        "not_biallelic_snp": 0,
        "duplicate_id": 0,
        "duplicate_position": 0,
        "no_gt": 0,
    }
    wanted_chrom = frozenset(chromosomes)
    cols: np.ndarray | None = None
    samples: list[str] = []
    n_all = 0
    seen: set[bytes] = set()
    seen_pos: set[tuple[int, int]] = set()
    tmp_bed = out.with_name(out.name + ".bed.tmp")
    tmp_bim = out.with_name(out.name + ".bim.tmp")
    with _open_vcf(path) as fh, tmp_bed.open("wb") as bed, tmp_bim.open("wb") as bim:
        bed.write(_BED_HEADER)
        for line in fh:
            if line[:1] == b"#":
                if line[:6] == b"#CHROM":
                    header = line.rstrip(b"\r\n").split(b"\t")
                    if len(header) < 10:
                        raise DataError(f"{path}: the VCF has no sample columns")
                    all_samples = [s.decode() for s in header[9:]]
                    n_all = len(all_samples)
                    if keep is None:
                        samples = all_samples
                    else:
                        want = set(keep)
                        idx = [i for i, s in enumerate(all_samples) if s in want]
                        if not idx:
                            raise DataError(f"{path}: none of the samples to keep is in the VCF")
                        missing = len(want) - len(idx)
                        if missing:
                            log.warning(
                                "%s: %d of the %d samples to keep are not in the VCF",
                                path.name,
                                missing,
                                len(want),
                            )
                        cols = np.array(idx, dtype=np.int64)
                        samples = [all_samples[i] for i in idx]
                continue
            if not n_all:
                raise DataError(f"{path}: data before the #CHROM header line; not a VCF file?")
            counts["records"] += 1
            head = line.split(b"\t", 9)
            if len(head) < 10:
                continue
            ident = head[2]
            chrom = _autosome(head[0])
            where: tuple[int, int] | None = None
            if positions is not None:
                try:
                    where = (chrom, int(head[1]))
                except ValueError:
                    where = None
                if where is None or where not in positions:
                    counts["not_listed"] += 1
                    continue
                ident = positions[where] or b"."
            elif snps is not None:
                if ident in snps:
                    pass
                elif b";" in ident:
                    hit = [x for x in ident.split(b";") if x in snps]
                    if not hit:
                        counts["not_listed"] += 1
                        continue
                    ident = hit[0]
                else:
                    counts["not_listed"] += 1
                    continue
            elif b";" in ident:
                ident = ident.split(b";", 1)[0]
            if chrom not in wanted_chrom:
                counts["not_autosomal"] += 1
                continue
            ref, alt = head[3].upper(), head[4].upper()
            if ref not in _BASES or alt not in _BASES:
                counts["not_biallelic_snp"] += 1
                continue
            if where is not None and where in seen_pos:
                counts["duplicate_position"] += 1
                continue
            if ident == b"." or not ident:
                ident = b"%d:%s:%s:%s" % (chrom, head[1], ref, alt)
            if ident in seen:
                counts["duplicate_id"] += 1
                continue
            fmt = head[8].split(b":")
            if b"GT" not in fmt:
                counts["no_gt"] += 1
                continue
            body = head[9].rstrip(b"\r\n")
            codes = _codes_fixed(body, n_all) if len(fmt) == 1 else None
            if codes is None:
                fields = body.split(b"\t")
                if len(fields) != n_all:
                    raise DataError(
                        f"{path}: record {ident.decode(errors='replace')} has {len(fields)} "
                        f"sample fields, expected {n_all}"
                    )
                codes = _codes_general(fields, fmt.index(b"GT"))
            if cols is not None:
                codes = codes[cols]
            bed.write(_pack(codes))
            bim.write(b"%d\t%s\t0\t%s\t%s\t%s\n" % (chrom, ident, head[1], alt, ref))
            seen.add(ident)
            if where is not None:
                seen_pos.add(where)
            counts["written"] += 1
    if not n_all:
        raise DataError(f"{path}: no #CHROM header line found; not a VCF file?")
    with out.with_name(out.name + ".fam").open("w") as fam:
        for s in samples:
            fam.write(f"{s}\t{s}\t0\t0\t0\t-9\n")
    os.replace(tmp_bed, out.with_name(out.name + ".bed"))
    os.replace(tmp_bim, out.with_name(out.name + ".bim"))
    return {"vcf": str(path), "out": str(out), "n_samples": len(samples), **counts}


def _read_ids(spec: Any, what: str) -> list[str] | None:
    if spec is None:
        return None
    if isinstance(spec, (str, os.PathLike)):
        from genesis_gwas.ld.build import _read_id_list

        arr = _read_id_list(spec, what)
        return None if arr is None else [str(x) for x in arr]
    out = [str(x) for x in spec]
    if not out:
        raise InputError(f"{what} list is empty")
    return out


def _read_positions(spec: Any) -> dict[tuple[int, int], bytes]:
    """Read ``CHR POS [ID]`` lines (or an iterable of such tuples) into a lookup table."""
    rows: list[Sequence[Any]]
    if isinstance(spec, (str, os.PathLike)):
        path = Path(os.fspath(spec)).expanduser()
        if not path.exists():
            raise DataError(f"positions file not found: {path}")
        opener: Any = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt") as fh:
            rows = [ln.split() for ln in fh if ln.strip() and not ln.startswith("#")]
    else:
        rows = [tuple(r) for r in spec]
    table: dict[tuple[int, int], bytes] = {}
    skipped = 0
    for r in rows:
        if len(r) < 2:
            raise InputError("positions need at least two columns: chromosome and position")
        chrom = _autosome(str(r[0]).encode())
        try:
            pos = int(r[1])
        except (TypeError, ValueError):
            skipped += 1  # a header line, or a position that could not be lifted ("NA")
            continue
        if chrom == 0 or pos < 1:
            skipped += 1
            continue
        table.setdefault((chrom, pos), str(r[2]).encode() if len(r) > 2 else b"")
    if not table:
        raise DataError("The positions list contains no usable autosomal position.")
    if skipped:
        log.info("positions list: %s line(s) without a usable autosomal position ignored", skipped)
    return table


def vcf_to_plink(
    vcf: str | os.PathLike[str],
    out: str | os.PathLike[str],
    *,
    snps: Any = None,
    positions: Any = None,
    keep: Any = None,
    chromosomes: Iterable[int] = range(1, 23),
    processes: int | None = None,
) -> list[dict[str, Any]]:
    r"""Extract biallelic SNPs from VCF files into PLINK ``.bed``/``.bim``/``.fam`` filesets.

    Parameters
    ----------
    vcf
        A VCF file (plain, gzip or bgzip), or a template containing ``{chrom}`` for one
        file per chromosome, e.g. ``"ALL.chr{chrom}.phase3.genotypes.vcf.gz"``.
    out
        Output fileset prefix. With a ``{chrom}`` template in ``vcf`` it must contain
        ``{chrom}`` too; pass the same template to
        :func:`~genesis_gwas.ld.build_reference`.
    snps
        Identifiers of the SNPs to keep (matched against the ``ID`` column; a record whose
        ``ID`` lists several identifiers separated by ``;`` matches if any is listed, and
        is written under that one): a
        path to a text file (first column; ``.gz`` allowed) or a sequence. ``None`` keeps
        every biallelic autosomal SNP, which for a sequencing panel means very large
        files — supply a list such as HapMap3.
    positions
        Select SNPs by position instead of by identifier, for panels whose ``ID`` column
        has no rsIDs: a path to a text file with the columns chromosome, position (in
        the genome build of the VCF) and, optionally, the identifier to write for that
        SNP; or an iterable of such tuples. Lines whose position is not a positive
        integer are ignored. Without the third column a SNP is named
        ``chrom:pos:ref:alt``. Only the first biallelic SNP at a position is written.
        Cannot be combined with ``snps``.
    keep
        Samples to keep: a path to a file of sample identifiers (one per line) or a
        sequence. ``None`` keeps all samples; the population can then still be selected
        with ``keep`` when the reference is built.
    chromosomes
        Autosomes to convert (default 1–22). With a ``{chrom}`` template, chromosomes
        whose file does not exist are an error.
    processes
        Number of chromosome files converted in parallel (default: one per available
        CPU, at most one per file).

    Returns
    -------
    list of dict
        One summary per VCF file: output prefix, number of samples, and how many records
        were read, written and set aside for each reason.
    """
    vcf_txt, out_txt = os.fspath(vcf), os.fspath(out)
    chroms = tuple(int(c) for c in chromosomes)
    if not chroms or any(c < 1 or c > 22 for c in chroms):
        raise InputError("chromosomes must be autosomes, 1 to 22")
    if snps is not None and positions is not None:
        raise InputError("Give either snps (identifiers) or positions, not both.")
    pos_table = None if positions is None else _read_positions(positions)
    snp_list = _read_ids(snps, "snps")
    snp_set = None if snp_list is None else frozenset(s.encode() for s in snp_list)
    keep_list = _read_ids(keep, "keep")
    keep_ids = None if keep_list is None else tuple(keep_list)

    if "{chrom}" in vcf_txt:
        if "{chrom}" not in out_txt:
            raise InputError("with a {chrom} template for the VCF, --out needs {chrom} as well")
        jobs: list[tuple[Any, ...]] = [
            (
                vcf_txt.format(chrom=c),
                out_txt.format(chrom=c),
                snp_set,
                keep_ids,
                (c,),
                # each worker only needs the positions of its own chromosome
                None if pos_table is None else {k: v for k, v in pos_table.items() if k[0] == c},
            )
            for c in chroms
        ]
        absent = [j[0] for j in jobs if not Path(j[0]).exists()]
        if absent:
            raise DataError(
                f"VCF file not found: {absent[0]}"
                + (f" (and {len(absent) - 1} more)" if len(absent) > 1 else "")
            )
    else:
        if "{chrom}" in out_txt:
            raise InputError("--out contains {chrom} but the VCF path does not")
        jobs = [(vcf_txt, out_txt, snp_set, keep_ids, chroms, pos_table)]

    from genesis_gwas.compute import available_cpus

    workers = min(len(jobs), processes if processes else available_cpus())
    if workers <= 1:
        results = [_convert_one(*job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_convert_one, *zip(*jobs, strict=True)))
    for r in results:
        log.info(
            "%s: %s of %s records written for %d samples -> %s",
            Path(r["vcf"]).name,
            f"{r['written']:,}",
            f"{r['records']:,}",
            r["n_samples"],
            r["out"],
        )
    total = sum(r["written"] for r in results)
    if total == 0:
        raise DataError(
            "No SNP was written. Check that the identifiers of the SNP list are those of the "
            "VCF ID column (or, with a positions list, that its positions are in the genome "
            "build of the VCF)."
        )
    return results
