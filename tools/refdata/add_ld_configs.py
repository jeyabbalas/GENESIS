#!/usr/bin/env python3
"""Add LD configurations to a converted upstream bundle by recomputing them from genotypes.

    python add_ld_configs.py --bundle BUNDLE --plink 'kg3.{chrom}' --keep eas.samples \
        --window-kb 500,1000,2000 --r2 0.05,0.1,0.2 --genotype-source "..."

The original GENESISasian package ships a single LD configuration (1 Mb, r^2 >= 0.1). This
script computes further (window, threshold) configurations for the *same* SNPs from the
genotypes the upstream table was derived from, with ``genesis_gwas.ld.compute_ld`` and
``round_r2=6`` (the convention of the upstream tables; see tools/ld_validation).

It is deliberately conservative: every configuration the bundle already holds is
recomputed as well and must agree with the stored upstream table — identical N* and
tagging lists for at least ``--min-identical`` of the SNPs and LD scores within
``--ldscore-tol`` for those — before anything is written. Existing files are never
modified; new ``ld/*.npz`` files are added and ``manifest.json`` is extended, with the
provenance of the computed configurations recorded under ``computed_ld_configs``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import refdata_common as rc

import genesis_gwas as gg
from genesis_gwas.ld import BedReader, LDTable, compute_ld


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def floats(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", required=True, type=Path)
    ap.add_argument("--plink", required=True, help="per-chromosome PLINK prefix template with {chrom}")
    ap.add_argument("--keep", required=True, help="sample IDs of the panel (first column)")
    ap.add_argument("--window-kb", type=floats, default=[500.0, 1000.0, 2000.0])
    ap.add_argument("--r2", type=floats, default=[0.05, 0.1, 0.2])
    ap.add_argument("--round-r2", type=int, default=6)
    ap.add_argument("--min-identical", type=float, default=0.9999,
                    help="required fraction of SNPs whose recomputed tagging list equals the stored one")
    ap.add_argument("--ldscore-tol", type=float, default=1e-5,
                    help="largest accepted LD-score difference (decimal ties at the rounding digit move a score by 1e-6 each)")
    ap.add_argument("--genotype-source", required=True, help="description recorded in the manifest")
    ap.add_argument("--dry-run", action="store_true", help="verify only; write nothing")
    args = ap.parse_args()

    mpath = args.bundle / "manifest.json"
    manifest = json.loads(mpath.read_text())
    have = {(float(c["window_kb"]), float(c["r2_min"])): c for c in manifest["ld_configs"]}
    configs = [(w, r) for w in args.window_kb for r in args.r2]
    for cfg in have:
        if cfg not in configs:
            configs.append(cfg)
    stored = {cfg: gg.load_reference(args.bundle, window_kb=cfg[0], r2_min=cfg[1]) for cfg in have}
    base = next(iter(stored.values()))
    n_ref = int(manifest["n_ref"])
    keep_ids = {ln.split()[0] for ln in Path(args.keep).read_text().splitlines() if ln.strip()}

    per_cfg: dict[tuple[float, float], list[LDTable]] = {c: [] for c in configs}
    for chrom in range(1, 23):
        sel = np.flatnonzero(base.chrom == chrom)
        if sel.size == 0:
            continue
        off = int(sel[0])
        if not np.array_equal(sel, np.arange(off, off + sel.size)):
            raise SystemExit("bundle is not in canonical (chromosome-contiguous) order")
        reader, bim, fam = BedReader.from_prefix(args.plink.format(chrom=chrom))
        cols = np.flatnonzero(fam["iid"].astype(str).isin(keep_ids).to_numpy())
        if cols.size != n_ref:
            raise SystemExit(f"chr{chrom}: {cols.size} samples selected, the bundle records {n_ref}")
        lookup = {s: i for i, s in enumerate(bim["snp"].tolist())}
        try:
            vidx = np.array([lookup[s] for s in base.rsid[sel].astype("U").tolist()], dtype=np.int64)
        except KeyError as exc:
            raise SystemExit(f"chr{chrom}: reference SNP {exc} is not in the genotypes") from exc
        pos = base.pos[sel].astype(np.int64)
        if not np.array_equal(bim["pos"].to_numpy()[vidx], pos):
            raise SystemExit(f"chr{chrom}: positions in the .bim differ from the bundle")
        tables = compute_ld(reader.read(vidx, samples=cols), pos, configs=configs, round_r2=args.round_r2)
        for cfg, t in zip(configs, tables, strict=True):
            per_cfg[cfg].append(t)
        log(f"chr{chrom}: {sel.size:,} SNPs")
    full = {cfg: LDTable.concatenate(ts) for cfg, ts in per_cfg.items()}

    # ------------------------------------------------------------- gate on agreement
    checks = {}
    for cfg, ref in stored.items():
        t = full[cfg]
        nb = ref.neighborhoods
        same_n = t.nstar == ref.nstar
        rows_equal = same_n.copy()
        idx = np.flatnonzero(same_n)
        # compare the tagging lists of rows with equal length, element by element
        if idx.size:
            a_ptr, b_ptr = t.indptr, nb.indptr
            lens = ref.nstar[idx].astype(np.int64)
            ai = np.repeat(a_ptr[idx], lens) + (np.arange(lens.sum()) - np.repeat(np.cumsum(lens) - lens, lens))
            bi = np.repeat(b_ptr[idx], lens) + (np.arange(lens.sum()) - np.repeat(np.cumsum(lens) - lens, lens))
            diff = t.indices[ai] != nb.indices[bi]
            bad_rows = np.unique(np.repeat(idx, lens)[diff])
            rows_equal[bad_rows] = False
        frac = float(rows_equal.mean())
        absdiff = np.abs(t.ldscore[rows_equal] - ref.ldscore[rows_equal])
        dls = float(absdiff.max())
        checks[ref.config.id] = {
            "n_snps": int(ref.M),
            "nstar_equal": int(same_n.sum()),
            "tagging_rows_identical": int(rows_equal.sum()),
            "fraction_identical": frac,
            "max_abs_ldscore_diff_identical_rows": dls,
            "n_snps_ldscore_diff_gt_1e-9": int((absdiff > 1e-9).sum()),
            "pairs_stored": int(nb.indices.size),
            "pairs_recomputed": int(t.nnz),
        }
        log(f"check {ref.config.id}: {checks[ref.config.id]}")
        if frac < args.min_identical or dls > args.ldscore_tol:
            raise SystemExit(f"{ref.config.id}: recomputed table does not reproduce the stored one; nothing written")

    if args.dry_run:
        log("dry run: nothing written")
        return

    # ------------------------------------------------------------- write new configs
    added = []
    for cfg in configs:
        if cfg in have:
            continue
        t = full[cfg]
        cid = rc.cfg_id(int(cfg[0]) if float(cfg[0]).is_integer() else cfg[0], f"{cfg[1]:g}")
        rel = f"ld/{cid}.npz"
        rc.save_npz(
            args.bundle / rel,
            {
                "nstar": t.nstar.astype(np.int32),
                "ldscore": t.ldscore.astype(np.float64),
                "indptr": t.indptr.astype(np.int64),
                "indices": t.indices.astype(np.int32),
                "window_kb": np.float64(cfg[0]),
                "r2_min": np.float64(cfg[1]),
                "n_ref": np.int64(n_ref),
            },
        )
        p = args.bundle / rel
        manifest["files"][rel] = {"sha256": rc.sha256_file(p), "bytes": p.stat().st_size}
        manifest["ld_configs"].append(
            {"id": cid, "window_kb": int(cfg[0]) if float(cfg[0]).is_integer() else cfg[0],
             "r2_min": cfg[1], "file": rel, "pairld": None}
        )
        added.append(cid)
        log(f"wrote {rel}: {t.nnz:,} pairs, {manifest['files'][rel]['bytes']:,} bytes")
    manifest["ld_configs"].sort(key=lambda c: (float(c["window_kb"]), float(c["r2_min"])))
    manifest["files"] = {k: manifest["files"][k] for k in sorted(manifest["files"])}
    manifest["computed_ld_configs"] = {
        "ids": added,
        "built_with": f"genesis-gwas {gg.__version__} (compute_ld, round_r2={args.round_r2})",
        "genotypes": args.genotype_source,
        "n_samples": n_ref,
        "agreement_with_upstream": checks,
    }
    mpath.write_text(json.dumps(manifest, indent=2) + "\n")
    log(f"manifest updated: {len(added)} configuration(s) added")


if __name__ == "__main__":
    main()
