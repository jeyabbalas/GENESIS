#!/usr/bin/env python
"""Is the reference SNP set "HapMap3 SNPs with MAF >= 0.05 in the panel"?

The paper describes the reference panel as the ~1.07 million HapMap3 SNPs with minor-allele
frequency of at least 5% in the 489 European samples of 1000 Genomes Phase 3. This script
checks that statement against the shipped tables: among the SNPs of upstream's
``w_hm3.noMHC.snplist`` that are present in the genotype files, those with MAF >= 0.05 should
be exactly the reference SNPs carrying the ``hm3_no_mhc`` flag.

    python check_snp_selection.py --plink "/path/1000G.EUR.QC.{chrom}" \
        --bundle /path/eur_1kg_hm3 --hm3-list /path/raw/hm3/w_hm3.noMHC.snplist.tsv
"""

from __future__ import annotations

import argparse
import json

import numpy as np

import genesis_gwas as gg
from genesis_gwas.ld import BedReader, allele_frequencies


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--plink", required=True)
    ap.add_argument("--bundle", required=True)
    ap.add_argument(
        "--hm3-list",
        required=True,
        help="text file whose first column is the SNP id (header allowed)",
    )
    ap.add_argument("--min-maf", type=float, default=0.05)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ref = gg.load_reference(args.bundle)
    with open(args.hm3_list) as fh:
        hm3 = {line.split()[0] for line in fh if line.strip()}
    ref_ids = ref.rsid.astype("U")
    flagged = set(ref_ids[ref.hm3_no_mhc].tolist())
    unflagged = set(ref_ids[~ref.hm3_no_mhc].tolist())
    res = {
        "hm3_list_size": len(hm3),
        "in_genotypes": 0,
        "pass_maf": 0,
        "pass_maf_and_in_reference": 0,
        "pass_maf_not_in_reference": 0,
        "reference_flagged_not_selected": 0,
        "min_maf_among_reference": 1.0,
        "max_maf_among_list_snps_not_in_reference": 0.0,
        "duplicate_position_among_selected": 0,
        "reference_snps_outside_list": len(unflagged),
    }
    selected: set[str] = set()
    for chrom in range(1, 23):
        reader, bim, _fam = BedReader.from_prefix(args.plink.format(chrom=chrom))
        ids = bim["snp"].to_numpy(dtype=str)
        in_list = np.fromiter((s in hm3 for s in ids.tolist()), dtype=bool, count=ids.size)
        in_ref = np.fromiter(
            (s in flagged or s in unflagged for s in ids.tolist()), dtype=bool, count=ids.size
        )
        idx = np.flatnonzero(in_list | in_ref)
        freq, _ = allele_frequencies(reader.read(idx))
        maf = np.minimum(freq, 1 - freq)
        sub_ids = ids[idx]
        sub_list = in_list[idx]
        sub_ref = in_ref[idx]
        res["in_genotypes"] += int(sub_list.sum())
        ok = sub_list & (maf >= args.min_maf)
        res["pass_maf"] += int(ok.sum())
        selected.update(sub_ids[ok].tolist())
        pos = bim["pos"].to_numpy()[idx][ok]
        res["duplicate_position_among_selected"] += int(pos.size - np.unique(pos).size)
        res["min_maf_among_reference"] = min(
            res["min_maf_among_reference"], float(maf[sub_ref].min())
        )
        miss = sub_list & ~sub_ref
        if miss.any():
            res["max_maf_among_list_snps_not_in_reference"] = max(
                res["max_maf_among_list_snps_not_in_reference"], float(maf[miss].max())
            )
    res["pass_maf_and_in_reference"] = len(selected & flagged)
    res["pass_maf_not_in_reference"] = len(selected - flagged - unflagged)
    res["reference_flagged_not_selected"] = len(flagged - selected)
    res["selection_reproduces_flagged_reference_snps"] = selected == flagged or (
        res["pass_maf_not_in_reference"] == 0 and res["reference_flagged_not_selected"] == 0
    )
    print(json.dumps(res, indent=1))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
