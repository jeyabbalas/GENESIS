#!/usr/bin/env python3
"""Assemble a reference bundle (format_version 1) from an upstream R package's data.

    python -I build_bundle.py --raw RAW_DIR --chain hg19ToHg38.over.chain.gz --out BUNDLE_DIR
                              [--profile eur|eas]

``--profile eur`` (default) builds ``eur_1kg_hm3`` from upstream GENESIS, ``--profile eas``
builds ``eas_1kg_hm3`` from GENESISasian. RAW_DIR is the output of ``export_rdata.R``. The bundle layout and every array are
described in README.md. All arrays are aligned to the canonical SNP order, i.e. sorted
by (chrom, pos, rsid) ascending.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # works under ``python -I`` too

import numpy as np  # noqa: E402

import refdata_common as rc  # noqa: E402


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def scatter(values: np.ndarray, dest: np.ndarray, n: int, dtype) -> np.ndarray:
    """Return ``out`` with ``out[dest] = values`` (dest must be a permutation of 0..n-1)."""
    out = np.empty(n, dtype=dtype)
    out[dest] = values
    return out


def lift_positions(chain_gz: Path, chrom: np.ndarray, pos: np.ndarray):
    """GRCh37 -> GRCh38 for every SNP; -1 when unmapped or mapped to another chromosome."""
    lo = rc.ChainLiftover(chain_gz)
    if lo.n_overlapping_blocks:
        log(f"WARNING: chain has {lo.n_overlapping_blocks} overlapping source blocks")
    out = np.full(pos.shape, -1, dtype=np.int32)
    stats = {"mapped": 0, "deleted": 0, "duplicated": 0, "other_chromosome": 0}
    for c in np.unique(chrom):
        sel = np.flatnonzero(chrom == c)
        q_name, q_pos, status = lo.lift(f"chr{int(c)}", pos[sel])
        same = np.array([name == f"chr{int(c)}" for name in q_name], dtype=bool)
        ok = (status == 0) & same
        out[sel[ok]] = q_pos[ok].astype(np.int32)
        stats["mapped"] += int(ok.sum())
        stats["deleted"] += int((status == 1).sum())
        stats["duplicated"] += int((status == 2).sum())
        stats["other_chromosome"] += int(((status == 0) & ~same).sum())
    return out, stats


def build(raw: Path, chain: Path, out: Path, profile: str = "eur") -> None:
    prof = rc.PROFILES[profile]
    configs = rc.all_configs(profile)
    n_ref = int(prof["n_ref"])
    files: dict[str, dict] = {}

    def register(rel: str) -> None:
        p = out / rel
        files[rel] = {"sha256": rc.sha256_file(p), "bytes": p.stat().st_size}
        log(f"  wrote {rel}  {files[rel]['bytes']:,} bytes")

    # ------------------------------------------------------------------ SNP table
    base = next(c for c in configs if c["id"] == rc.PAIR_CFG)
    bdir = raw / base["raw_dir"]
    names0 = rc.read_names(bdir / "snpname.txt")
    chr0 = rc.read_i32(bdir / "chr.i32")
    bp0 = rc.read_i32(bdir / "bp.i32")
    m = names0.size
    if not (chr0.size == bp0.size == m):
        raise SystemExit("SNP table columns have different lengths")
    if chr0.min() < 1 or chr0.max() > 22 or bp0.min() < 1:
        raise SystemExit("unexpected chromosome/position values")
    perm = rc.canonical_order(chr0, bp0, names0)
    rsid = names0[perm]
    chrom = chr0[perm].astype(np.uint8)
    pos = bp0[perm].astype(np.int32)
    log(f"SNP table: M={m:,}; rsid dtype {rsid.dtype}")

    pos38, lift_stats = lift_positions(chain, chrom, pos)
    log(f"liftover hg19->hg38: {lift_stats}")

    hm3_lines = (raw / "hm3" / "w_hm3.noMHC.snplist.tsv").read_bytes().split(b"\n")
    if hm3_lines[0] != b"SNP\tA1\tA2":
        raise SystemExit("unexpected header in w_hm3.noMHC.snplist.tsv")
    hm3_names = np.array([ln.split(b"\t", 1)[0] for ln in hm3_lines[1:] if ln], dtype="S")
    hm3_flag = np.isin(rsid, hm3_names)
    log(f"hm3_no_mhc: {int(hm3_flag.sum()):,} of {m:,} reference SNPs are in the list ({hm3_names.size:,} entries)")

    rc.save_npz(
        out / "snps.npz",
        {"rsid": rsid, "chrom": chrom, "pos": pos, "pos_grch38": pos38, "hm3_no_mhc": hm3_flag},
    )
    register("snps.npz")

    # ------------------------------------------------------------------ LD configs
    ld_entries = []
    for cfg in configs:
        d = raw / cfg["raw_dir"]
        names = rc.read_names(d / "snpname.txt")
        u2c = rc.index_in(names, rsid)  # upstream row -> canonical row
        if names.size != m or (u2c < 0).any() or np.unique(u2c).size != m:
            raise SystemExit(f"{cfg['id']}: SNP set differs from the base configuration")
        if not (np.array_equal(rc.read_i32(d / "chr.i32"), chrom[u2c])
                and np.array_equal(rc.read_i32(d / "bp.i32"), pos[u2c])):
            raise SystemExit(f"{cfg['id']}: CHR/BP differ from the base configuration")
        nstar = scatter(rc.read_i32(d / "nstar.i32"), u2c, m, np.int32)
        ldscore = scatter(rc.read_f64(d / "ldscore.f64"), u2c, m, np.float64)
        tag_len = rc.read_i32(d / "tag_len.i32")
        tag_idx = rc.read_i32(d / "tag_idx0.i32")
        if tag_len.size != m or tag_idx.size != int(tag_len.sum(dtype=np.int64)):
            raise SystemExit(f"{cfg['id']}: tagging list sizes are inconsistent")
        rows = u2c[np.repeat(np.arange(m, dtype=np.int64), tag_len)]
        cols = u2c[tag_idx]
        indptr, indices = rc.csr_from_pairs(rows, cols, m)
        del rows, cols, tag_idx
        if not np.array_equal(np.diff(indptr), nstar):
            raise SystemExit(f"{cfg['id']}: row lengths differ from Nstar")
        rel = f"ld/{cfg['id']}.npz"
        rc.save_npz(
            out / rel,
            {
                "nstar": nstar,
                "ldscore": ldscore,
                "indptr": indptr,
                "indices": indices,
                "window_kb": np.float64(cfg["window_kb"]),
                "r2_min": np.float64(cfg["r2_min"]),
                "n_ref": np.int64(n_ref),
            },
        )
        register(rel)
        pair_rel = None
        if cfg["id"] == rc.PAIR_CFG:
            pdir = raw / "pair_w1MB_c0.1"
            plen = rc.read_i32(pdir / "len.i32")
            pidx = rc.read_i32(pdir / "tagidx0.i32")
            r2 = rc.read_f64(pdir / "r2.f64")
            if not (plen.size == m and pidx.size == r2.size == int(plen.sum(dtype=np.int64))):
                raise SystemExit("pairwise LD sizes are inconsistent")
            prow = u2c[np.repeat(np.arange(m, dtype=np.int64), plen)]
            pcol = u2c[pidx]
            pptr, pind, r2s = rc.csr_from_pairs(prow, pcol, m, r2)
            del prow, pcol, pidx, r2
            if not (np.array_equal(pptr, indptr) and np.array_equal(pind, indices)):
                raise SystemExit("TaggingSNPinx-derived lists differ from TaggingSNPs-derived lists")
            if r2s.min() < 0 or r2s.max() > 1:
                raise SystemExit("r2 outside [0, 1]")
            r2_u16 = np.rint(r2s * 65535.0).astype(np.uint16)
            pair_rel = f"pairld/{cfg['id']}.npz"
            rc.save_npz(out / pair_rel, {"r2_u16": r2_u16})
            register(pair_rel)
            del pptr, pind, r2s, r2_u16
        ld_entries.append(
            {"id": cfg["id"], "window_kb": cfg["window_kb"], "r2_min": cfg["r2_min"], "file": rel,
             "pairld": pair_rel}
        )
        del indptr, indices

    # ------------------------------------------------------------------ null z
    ndir = raw / "nullz"
    e2c = rc.index_in(rc.read_names(ndir / "snpname.txt"), rsid)
    if e2c.size != m or (e2c < 0).any() or np.unique(e2c).size != m:
        raise SystemExit("null-z SNP set differs from the reference SNP set")
    nullz_files = []
    for start in range(0, rc.N_NULLZ, rc.NULLZ_PER_FILE):
        block = np.empty((rc.NULLZ_PER_FILE, m), dtype=np.float16)
        for i in range(rc.NULLZ_PER_FILE):
            v = rc.read_f64(ndir / f"error_iter{start + i + 1:03d}.f64")
            if v.size != m or not np.isfinite(v).all():
                raise SystemExit(f"error_iter{start + i + 1}: bad length or non-finite values")
            block[i] = scatter(v, e2c, m, np.float64).astype(np.float16)
        rel = f"nullz/nullz_{start:03d}-{start + rc.NULLZ_PER_FILE - 1:03d}.npy"
        rc.save_npy(out / rel, block)
        register(rel)
        nullz_files.append(rel)

    # ------------------------------------------------------------------ example
    if prof["example"]:
        height = (raw / "height" / "heightGWAS.tsv").read_bytes()
        if not height.startswith(b"SNP\tZ\tN\n"):
            raise SystemExit("unexpected header in heightGWAS.tsv")
        rel = prof["example"]
        rc.gzip_deterministic(height, out / rel)
        register(rel)

    # ------------------------------------------------------------------ manifest
    manifest = {
        "format_version": rc.FORMAT_VERSION,
        "name": prof["name"],
        "version": prof["version"],
        "population": prof["population"],
        "panel": prof["panel"],
        "genome_build": "GRCh37",
        "n_ref": n_ref,
        "n_snps": int(m),
        "source": {"repo": prof["repo"], "commit": prof["commit"]},
        "ld_configs": ld_entries,
        "nullz": {"dtype": "float16", "n_replicates": rc.N_NULLZ, "files": nullz_files},
        "files": {k: files[k] for k in sorted(files)},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    total = sum(f["bytes"] for f in files.values())
    log(f"manifest.json written; {len(files)} data files, {total:,} bytes ({total / 2**20:.1f} MiB)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True, type=Path, help="output directory of export_rdata.R")
    ap.add_argument("--chain", required=True, type=Path, help="UCSC hg19ToHg38.over.chain.gz")
    ap.add_argument("--out", required=True, type=Path, help="bundle directory to create")
    ap.add_argument("--profile", choices=sorted(rc.PROFILES), default="eur",
                    help="upstream source: eur = GENESIS, eas = GENESISasian [eur]")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    build(args.raw, args.chain, args.out, args.profile)


if __name__ == "__main__":
    main()
