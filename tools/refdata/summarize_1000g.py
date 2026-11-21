#!/usr/bin/env python3
"""Summarise the LDSC 1000 Genomes Phase 3 EUR PLINK files and their overlap with a bundle.

    python -I summarize_1000g.py --plink-dir .../1000G_EUR_Phase3_plink --bundle BUNDLE_DIR

Reads only the .bim/.fam text files and the 3-byte .bed header plus file size (structural
sanity: SNP-major PLINK 1 binary with ``3 + ceil(n_samples / 4) * n_snps`` bytes). No
genotypes are decoded.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # works under ``python -I`` too

import numpy as np  # noqa: E402

import refdata_common as rc  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plink-dir", required=True, type=Path)
    ap.add_argument("--bundle", required=True, type=Path)
    ap.add_argument("--prefix", default="1000G.EUR.QC.")
    args = ap.parse_args()

    with np.load(args.bundle / "snps.npz", allow_pickle=False) as z:
        rsid, chrom, pos = z["rsid"], z["chrom"].astype(np.int64), z["pos"].astype(np.int64)
    m = rsid.size

    fam_ref = None
    names, chroms, poss = [], [], []
    print(f"{'chr':>3} {'bim SNPs':>10} {'samples':>8} {'bed bytes ok':>13} {'ref SNPs':>9} "
          f"{'by rsid':>8} {'by chr:pos':>10}")
    total_bed = 0
    for c in range(1, 23):
        stem = args.plink_dir / f"{args.prefix}{c}"
        fam = [ln.split() for ln in (stem.with_name(stem.name + ".fam")).read_text().split("\n") if ln]
        iids = [f[1] for f in fam]
        if fam_ref is None:
            fam_ref = iids
        elif iids != fam_ref:
            print(f"WARNING: sample order of chromosome {c} differs from chromosome 1")
        n_s = len(fam)
        b_names, b_pos = [], []
        with open(stem.with_name(stem.name + ".bim"), "rb") as fh:
            for line in fh:
                f = line.split()
                if int(f[0]) != c:
                    raise SystemExit(f"{stem}.bim: row on chromosome {f[0].decode()}")
                b_names.append(f[1])
                b_pos.append(int(f[3]))
        n_snp = len(b_names)
        bed = stem.with_name(stem.name + ".bed")
        with open(bed, "rb") as fh:
            magic = fh.read(3)
        size = bed.stat().st_size
        total_bed += size
        ok = magic == b"\x6c\x1b\x01" and size == 3 + ((n_s + 3) // 4) * n_snp
        bn = np.array(b_names, dtype="S")
        bp = np.array(b_pos, dtype=np.int64)
        sel = chrom == c
        by_id = int(np.isin(rsid[sel], bn).sum())
        by_pos = int(np.isin(pos[sel], bp).sum())
        print(f"{c:>3} {n_snp:>10,} {n_s:>8} {str(ok):>13} {int(sel.sum()):>9,} {by_id:>8,} {by_pos:>10,}")
        names.append(bn)
        chroms.append(np.full(n_snp, c, dtype=np.int64))
        poss.append(bp)

    names = np.concatenate(names)
    chroms = np.concatenate(chroms)
    poss = np.concatenate(poss)
    print(f"total bim SNPs: {names.size:,}; unique rsids: {np.unique(names).size:,}; samples: {len(fam_ref)}; "
          f".bed bytes: {total_bed:,}")
    key_bim = chroms * (1 << 40) + poss
    print(f"bim rows sharing a chrom:pos with another row: {names.size - np.unique(key_bim).size:,}")

    key_ref = chrom * (1 << 40) + pos
    in_id = np.isin(rsid, names)
    in_pos = np.isin(key_ref, key_bim)
    # same rsid AND same chrom:pos
    order = np.argsort(names, kind="stable")
    sn = names[order]
    j = np.minimum(np.searchsorted(sn, rsid), sn.size - 1)
    same = (sn[j] == rsid) & (key_bim[order][j] == key_ref)
    print(f"reference SNPs: {m:,}")
    print(f"  present by rsid:                {int(in_id.sum()):,}  (missing {int((~in_id).sum()):,})")
    print(f"  present by chrom:pos (GRCh37):  {int(in_pos.sum()):,}  (missing {int((~in_pos).sum()):,})")
    print(f"  present by rsid AND chrom:pos:  {int(same.sum()):,}")
    print(f"  rsid found but at another position: {int((in_id & ~same).sum()):,}; "
          f"position found under another rsid only: {int((in_pos & ~in_id).sum()):,}; "
          f"found by neither: {int((~in_id & ~in_pos).sum()):,}")
    miss = np.flatnonzero(~in_id)[:10]
    if miss.size:
        print("  first reference SNPs missing by rsid:",
              [(rsid[i].decode(), int(chrom[i]), int(pos[i])) for i in miss])
    _ = rc  # shared module imported for the sys.path convention only


if __name__ == "__main__":
    main()
