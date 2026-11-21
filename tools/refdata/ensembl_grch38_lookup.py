#!/usr/bin/env python3
"""Optional: look up GRCh38 positions by rsID for SNPs the chain liftover could not place.

    python -I ensembl_grch38_lookup.py --bundle BUNDLE_DIR --chain hg19ToHg38.over.chain.gz --out TABLE.tsv

The bundle stores ``pos_grch38 = -1`` when UCSC liftOver has no same-chromosome image of
the GRCh37 position (the base falls in a chain gap, or the chain sends it to another
sequence). dbSNP/Ensembl often still place those rsIDs on GRCh38. This script reports why
each such SNP failed and writes Ensembl's primary-assembly location, when there is exactly
one on the same chromosome, to a side table. It does NOT modify the bundle: mixing a
position liftover with rsID-based coordinates is a policy decision for the caller.

Only public rsIDs are sent to rest.ensembl.org.
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # works under ``python -I`` too

import numpy as np  # noqa: E402

import refdata_common as rc  # noqa: E402
from check_liftover_ensembl import primary_mappings, query  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", required=True, type=Path)
    ap.add_argument("--chain", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    with np.load(args.bundle / "snps.npz", allow_pickle=False) as z:
        rsid, chrom, pos, pos38 = z["rsid"], z["chrom"], z["pos"], z["pos_grch38"]
    todo = np.flatnonzero(pos38 == -1)
    lo = rc.ChainLiftover(args.chain)
    reason = {}
    targets = collections.Counter()
    for c in np.unique(chrom[todo]):
        sel = todo[chrom[todo] == c]
        q_name, q_pos, status = lo.lift(f"chr{int(c)}", pos[sel])
        for i, name, st in zip(sel, q_name, status):
            if st == 1:
                reason[int(i)] = "chain_gap"
            elif st == 2:
                reason[int(i)] = "duplicated"
            else:
                reason[int(i)] = f"maps_to_{name}"
                targets[name if "_" not in name else "alt/random/unplaced contig"] += 1
    counts = collections.Counter("maps_elsewhere" if r.startswith("maps_to_") else r for r in reason.values())
    print(f"SNPs with pos_grch38 == -1: {todo.size}; reasons: {dict(counts)}")
    print(f"  targets of the cross-sequence mappings: {dict(targets)}")

    ids = [rsid[i].decode() for i in todo]
    table = primary_mappings(query("https://rest.ensembl.org", ids), "GRCh38")
    n_one = n_multi = n_none = n_other = 0
    with open(args.out, "w") as fh:
        fh.write("rsid\tchrom\tpos_grch37\tliftover_failure\tpos_grch38_ensembl\n")
        for i in todo:
            name = rsid[i].decode()
            locs = table.get(name, set())
            same = sorted(p for c, p in locs if c == str(int(chrom[i])))
            if len(same) == 1:
                n_one += 1
                val = str(same[0])
            else:
                val = "NA"
                if len(same) > 1:
                    n_multi += 1
                elif locs:
                    n_other += 1
                else:
                    n_none += 1
            fh.write(f"{name}\t{int(chrom[i])}\t{int(pos[i])}\t{reason[int(i)]}\t{val}\n")
    print(f"Ensembl GRCh38: unique same-chromosome location for {n_one}; several for {n_multi}; "
          f"only on another chromosome for {n_other}; none for {n_none}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
