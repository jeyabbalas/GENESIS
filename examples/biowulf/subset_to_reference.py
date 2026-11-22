#!/usr/bin/env python
"""Shrink a genome-wide summary-statistics file to the SNPs of an LD reference.

GENESIS only uses the SNPs that are in the LD reference (about 1.07 million HapMap3 SNPs
for the bundled European panel). ``genesis fit`` does that filtering itself, but it first
loads the whole file, and a file with ten million rows needs about 7 GB of memory for
that (roughly 0.7 GB per million rows). Filtering once, in chunks, keeps every later job
small (about 2-3 GB) and fast, which matters when the same file is fitted many times in
a swarm.

All columns of the kept rows are written unchanged, so the same ``genesis fit`` column
options work on the output. Rows are matched by SNP identifier, or by chromosome and
position when ``--chr`` and ``--pos`` are given (then also give ``--build``).

Usage (inside an sinteractive session or a batch job, environment activated)::

    python subset_to_reference.py big.tsv.gz small.tsv.gz
    python subset_to_reference.py big.tsv.gz small.tsv.gz --reference /data/$USER/proj/ref/AFR
    python subset_to_reference.py big.tsv.gz small.tsv.gz --chr CHR --pos BP --build GRCh38
"""

from __future__ import annotations

import argparse
import gzip
import sys
from pathlib import Path

import pandas as pd

import genesis_gwas as gg
from genesis_gwas.data.sumstats import COLUMN_ALIASES


def sniff(path: Path) -> str:
    """Delimiter of the header line: tab, comma, or runs of white space."""
    opener = gzip.open if path.suffix in (".gz", ".bgz") else open
    with opener(path, "rt") as fh:
        header = fh.readline()
    if "\t" in header:
        return "\t"
    if "," in header and " " not in header.strip():
        return ","
    return r"\s+"


def find_column(columns: list[str], logical: str, given: str | None) -> str:
    """Return the name of a logical column (``snp``, ``chr``, ``pos``), given or auto-detected."""
    if given is not None:
        if given not in columns:
            sys.exit(f"Column {given!r} not found; columns are: {columns}")
        return given
    lower = {c.lower(): c for c in columns}
    for alias in COLUMN_ALIASES[logical]:
        if alias in lower:
            return lower[alias]
    sys.exit(f"Cannot find the {logical} column automatically; columns are: {columns}")


def main(argv: list[str] | None = None) -> int:
    """Filter the file chunk by chunk and report what was kept."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "sumstats", type=Path, help="input summary statistics (text, optionally .gz)"
    )
    parser.add_argument(
        "out", type=Path, help="output file (tab-separated; gzip if it ends in .gz)"
    )
    parser.add_argument(
        "--reference",
        default="eur_1kg_hm3",
        help="reference name or bundle directory [eur_1kg_hm3]",
    )
    parser.add_argument("--snp", help="SNP identifier column [detected]")
    parser.add_argument(
        "--chr", dest="chrom", help="chromosome column (match by position instead of identifier)"
    )
    parser.add_argument("--pos", help="position column (match by position instead of identifier)")
    parser.add_argument(
        "--build",
        help="genome build of the positions: GRCh37 or GRCh38 [the reference's own build]",
    )
    parser.add_argument("--chunksize", type=int, default=500_000, help="rows per chunk [500000]")
    args = parser.parse_args(argv)
    if (args.chrom is None) != (args.pos is None):
        parser.error("give both --chr and --pos, or neither")

    ref = gg.load_reference(args.reference)
    sep = sniff(args.sumstats)
    engine = "python" if sep == r"\s+" else "c"
    header = pd.read_csv(args.sumstats, sep=sep, nrows=0, engine=engine)
    columns = [str(c) for c in header.columns]
    by_position = args.chrom is not None
    if by_position:
        for col in (args.chrom, args.pos):
            if col not in columns:
                sys.exit(f"Column {col!r} not found; columns are: {columns}")
    else:
        snp_col = find_column(columns, "SNP", args.snp)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_name(args.out.name + ".part")
    n_in = n_out = 0
    first = True
    reader = pd.read_csv(
        args.sumstats,
        sep=sep,
        dtype=str,
        keep_default_na=False,
        chunksize=args.chunksize,
        engine=engine,
    )
    for chunk in reader:
        n_in += len(chunk)
        if by_position:
            pos = (
                pd.to_numeric(chunk[args.pos], errors="coerce")
                .fillna(-1)
                .astype("int64")
                .to_numpy()
            )
            rows = ref.lookup_position(chunk[args.chrom].to_numpy(dtype=str), pos, args.build)
        else:
            rows = ref.lookup(chunk[snp_col].to_numpy(dtype=str))
        keep = chunk[rows >= 0]
        n_out += len(keep)
        keep.to_csv(
            tmp,
            sep="\t",
            index=False,
            header=first,
            mode="w" if first else "a",
            compression="gzip" if args.out.suffix == ".gz" else None,
        )
        first = False
    tmp.replace(args.out)
    how = (
        f"position ({args.build or ref.genome_build})" if by_position else f"identifier ({snp_col})"
    )
    print(
        f"{args.sumstats}: kept {n_out:,} of {n_in:,} rows that match {ref.name} (M = {ref.M:,}) by {how}"
    )
    print(f"wrote {args.out}")
    if n_out < 0.5 * ref.M:
        print(
            "WARNING: fewer than half of the reference SNPs were found. Check the identifier "
            "column, or match by position with --chr/--pos/--build.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
