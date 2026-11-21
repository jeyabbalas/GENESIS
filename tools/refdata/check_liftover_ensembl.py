#!/usr/bin/env python3
"""Independent check of the bundle's GRCh38 positions against Ensembl (network required).

    python -I check_liftover_ensembl.py --bundle BUNDLE_DIR --chain hg19ToHg38.over.chain.gz
        [--n-random 150] [--n-reverse 100] [--n-unmapped 50] [--seed 20261121]

``pos_grch38`` in snps.npz comes from a position liftover (UCSC chain) of the GRCh37
coordinate. Ensembl's variation database assigns coordinates to rsIDs from dbSNP on each
assembly independently of that chain, so agreement between the two is an end-to-end check
of the chain parser, the 0/1-based conversions and the reverse-strand arithmetic. The same
query against the GRCh37 archive server checks the bundle's GRCh37 ``pos`` as well.

Only public rsIDs are sent to rest.ensembl.org / grch37.rest.ensembl.org.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # works under ``python -I`` too

import numpy as np  # noqa: E402

import refdata_common as rc  # noqa: E402

WELL_KNOWN = [
    "rs3094315", "rs9939609", "rs7903146", "rs1421085", "rs12913832", "rs2981582", "rs1801133",
    "rs4988235", "rs1800562", "rs6025", "rs429358", "rs7412", "rs1333049", "rs10757278",
    "rs11209026", "rs2476601", "rs3184504", "rs13266634", "rs1260326", "rs4420638",
]
SERVERS = {"GRCh38": "https://rest.ensembl.org", "GRCh37": "https://grch37.rest.ensembl.org"}
AUTOSOMES = {str(c) for c in range(1, 23)}


def query(server: str, ids: list[str], retries: int = 4) -> dict:
    out: dict = {}
    for i in range(0, len(ids), 200):
        body = json.dumps({"ids": ids[i : i + 200]}).encode()
        req = urllib.request.Request(
            server + "/variation/homo_sapiens",
            data=body,
            headers={"Content-Type": "application/json", "Accept": "application/json",
                     "User-Agent": "refdata-liftover-check/1.0"},
        )
        for attempt in range(retries):
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    out.update(json.loads(resp.read()))
                break
            except Exception as exc:  # noqa: BLE001 - report and retry
                if attempt == retries - 1:
                    raise
                print(f"  retry after error: {exc}", file=sys.stderr)
                time.sleep(5 * (attempt + 1))
        time.sleep(0.5)
    return out


def primary_mappings(records: dict, assembly: str) -> dict[str, set[tuple[str, int]]]:
    """rsID (and synonyms) -> set of (chromosome, start) on the primary autosomes."""
    out: dict[str, set[tuple[str, int]]] = {}
    for key, rec in records.items():
        locs = {
            (m["seq_region_name"], int(m["start"]))
            for m in rec.get("mappings", [])
            if m.get("assembly_name") == assembly and m.get("seq_region_name") in AUTOSOMES
            and int(m["start"]) == int(m["end"])
        }
        for name in {key, rec.get("name", key), *rec.get("synonyms", [])}:
            out.setdefault(name, set()).update(locs)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", required=True, type=Path)
    ap.add_argument("--chain", required=True, type=Path)
    ap.add_argument("--n-random", type=int, default=150)
    ap.add_argument("--n-reverse", type=int, default=100)
    ap.add_argument("--n-unmapped", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20261121)
    ap.add_argument("--save", type=Path, help="write the raw Ensembl responses to this JSON file")
    args = ap.parse_args()

    with np.load(args.bundle / "snps.npz", allow_pickle=False) as z:
        rsid, chrom, pos, pos38 = z["rsid"], z["chrom"], z["pos"], z["pos_grch38"]
    m = rsid.size
    rng = np.random.default_rng(args.seed)

    # SNPs that sit in reverse-strand chain blocks (exercise the strand arithmetic)
    lo = rc.ChainLiftover(args.chain)
    in_rev = np.zeros(m, dtype=bool)
    for c in np.unique(chrom):
        b = lo.blocks[f"chr{int(c)}"]
        sel = np.flatnonzero(chrom == c)
        p0 = pos[sel].astype(np.int64) - 1
        i = np.searchsorted(b["t_start"], p0, side="right") - 1
        ic = np.maximum(i, 0)
        hit = (i >= 0) & (p0 < b["t_end"][ic])
        in_rev[sel] = hit & b["q_rev"][ic]
    mapped = pos38 > 0
    print(f"reference SNPs: {m:,}; mapped: {int(mapped.sum()):,}; of those through reverse-strand blocks: "
          f"{int((in_rev & mapped).sum()):,}; unmapped or other chromosome: {int((~mapped).sum()):,}")

    known = rc.index_in(np.array([k.encode() for k in WELL_KNOWN], dtype="S"), rsid)
    known = known[known >= 0]
    pick = [
        rng.choice(m, size=args.n_random, replace=False),
        rng.choice(np.flatnonzero(in_rev & mapped), size=min(args.n_reverse, int((in_rev & mapped).sum())),
                   replace=False),
        rng.choice(np.flatnonzero(~mapped), size=min(args.n_unmapped, int((~mapped).sum())), replace=False),
        known,
    ]
    idx = np.unique(np.concatenate(pick))
    ids = [rsid[i].decode() for i in idx]
    print(f"querying Ensembl for {len(ids)} rsIDs ({args.n_random} random, {len(pick[1])} reverse-strand, "
          f"{len(pick[2])} unmapped, {known.size} well-known present in the reference)")

    raw = {name: query(server, ids) for name, server in SERVERS.items()}
    if args.save:
        args.save.write_text(json.dumps(raw))
    map38 = primary_mappings(raw["GRCh38"], "GRCh38")
    map37 = primary_mappings(raw["GRCh37"], "GRCh37")

    def compare(label: str, table: dict, values: np.ndarray, subset: np.ndarray) -> None:
        n = agree = absent = 0
        bad = []
        for i in idx[subset[idx]]:
            name = rsid[i].decode()
            locs = table.get(name)
            if not locs:
                absent += 1
                continue
            n += 1
            if (str(int(chrom[i])), int(values[i])) in locs:
                agree += 1
            else:
                bad.append((name, int(chrom[i]), int(values[i]), sorted(locs)))
        print(f"{label}: compared {n}; concordant {agree}; discordant {len(bad)}; "
              f"no primary-assembly mapping returned by Ensembl {absent}")
        for item in bad[:15]:
            print("    discordant:", item)

    everything = np.ones(m, dtype=bool)
    compare("GRCh37 pos (bundle 'pos') vs Ensembl GRCh37", map37, pos, everything)
    compare("GRCh38 pos (bundle 'pos_grch38') vs Ensembl GRCh38, all mapped", map38, pos38, mapped)
    compare("  ... subset lifted through reverse-strand chain blocks", map38, pos38, mapped & in_rev)
    compare("  ... subset lifted through forward-strand chain blocks", map38, pos38, mapped & ~in_rev)

    # what does Ensembl say about SNPs the chain could not place on the same chromosome?
    n_un = n_has = n_same_chrom = 0
    for i in idx[~mapped[idx]]:
        n_un += 1
        locs = map38.get(rsid[i].decode())
        if locs:
            n_has += 1
            n_same_chrom += any(c == str(int(chrom[i])) for c, _ in locs)
    print(f"SNPs with pos_grch38 == -1 in the sample: {n_un}; Ensembl has a primary GRCh38 autosomal location "
          f"for {n_has} of them ({n_same_chrom} on the same chromosome)")
    print("well-known SNPs (rsid, chr, GRCh37, GRCh38 bundle, Ensembl GRCh38):")
    for i in known:
        name = rsid[i].decode()
        print(f"    {name:<12} chr{int(chrom[i]):<3} {int(pos[i]):>11,} {int(pos38[i]):>11,}  "
              f"{sorted(map38.get(name, []))}")


if __name__ == "__main__":
    main()
