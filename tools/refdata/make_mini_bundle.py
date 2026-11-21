#!/usr/bin/env python3
"""Cut a single-chromosome test bundle out of a full reference bundle (same format).

    python -I make_mini_bundle.py --bundle FULL_DIR --out MINI_DIR \
        [--chrom 22] [--name mini_chr22] [--configs w1000kb_r0.1 w500kb_r0.2] \
        [--nullz-replicates 10]

Row indices are re-based to the subset. The script refuses to proceed if any neighbour of
a selected SNP lies outside the subset, so the result is a self-contained reference.
The example file keeps the rows of the full bundle's height example whose SNP is in the
subset (original order).
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # works under ``python -I`` too

import numpy as np  # noqa: E402

import refdata_common as rc  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", required=True, type=Path, help="full bundle directory")
    ap.add_argument("--out", required=True, type=Path, help="mini bundle directory to create")
    ap.add_argument("--chrom", type=int, default=22)
    ap.add_argument("--name", default=None, help="bundle name (default: mini_chr<chrom>)")
    ap.add_argument("--configs", nargs="+", default=["w1000kb_r0.1", "w500kb_r0.2"])
    ap.add_argument("--nullz-replicates", type=int, default=10)
    args = ap.parse_args()

    full, out = args.bundle, args.out
    name = args.name or f"mini_chr{args.chrom}"
    man = json.loads((full / "manifest.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict] = {}

    def register(rel: str) -> None:
        p = out / rel
        files[rel] = {"sha256": rc.sha256_file(p), "bytes": p.stat().st_size}
        print(f"  wrote {rel}  {files[rel]['bytes']:,} bytes")

    with np.load(full / "snps.npz", allow_pickle=False) as z:
        snps = {k: z[k] for k in z.files}
    sel = np.flatnonzero(snps["chrom"] == args.chrom)
    if sel.size == 0:
        raise SystemExit(f"no SNPs on chromosome {args.chrom}")
    lo, hi = int(sel[0]), int(sel[-1]) + 1
    if hi - lo != sel.size:
        raise SystemExit("selected SNPs are not contiguous in canonical order")
    m = hi - lo
    rc.save_npz(out / "snps.npz", {k: v[lo:hi] for k, v in snps.items()})
    register("snps.npz")

    by_id = {c["id"]: c for c in man["ld_configs"]}
    ld_entries = []
    for cid in args.configs:
        if cid not in by_id:
            raise SystemExit(f"configuration {cid} is not in the full bundle")
        entry = by_id[cid]
        with np.load(full / entry["file"], allow_pickle=False) as z:
            indptr = z["indptr"]
            a, b = int(indptr[lo]), int(indptr[hi])
            indices = z["indices"][a:b].astype(np.int64) - lo
            if indices.size and (indices.min() < 0 or indices.max() >= m):
                raise SystemExit(f"{cid}: a neighbour of a chr{args.chrom} SNP lies outside the subset")
            sub = {
                "nstar": z["nstar"][lo:hi],
                "ldscore": z["ldscore"][lo:hi],
                "indptr": (indptr[lo : hi + 1] - a).astype(np.int64),
                "indices": indices.astype(np.int32),
                "window_kb": z["window_kb"],
                "r2_min": z["r2_min"],
                "n_ref": z["n_ref"],
            }
        rel = f"ld/{cid}.npz"
        rc.save_npz(out / rel, sub)
        register(rel)
        pair_rel = None
        if entry.get("pairld"):
            with np.load(full / entry["pairld"], allow_pickle=False) as z:
                r2 = z["r2_u16"][a:b]
            pair_rel = f"pairld/{cid}.npz"
            rc.save_npz(out / pair_rel, {"r2_u16": r2})
            register(pair_rel)
        ld_entries.append(
            {"id": cid, "window_kb": entry["window_kb"], "r2_min": entry["r2_min"], "file": rel,
             "pairld": pair_rel}
        )

    nullz_files = []
    n_rep = args.nullz_replicates
    if n_rep:
        blocks, got = [], 0
        for rel in man["nullz"]["files"]:
            arr = np.load(full / rel, mmap_mode="r", allow_pickle=False)
            take = min(n_rep - got, arr.shape[0])
            blocks.append(np.array(arr[:take, lo:hi]))
            got += take
            if got >= n_rep:
                break
        if got < n_rep:
            raise SystemExit("not enough null-z replicates in the full bundle")
        rel = f"nullz/nullz_000-{n_rep - 1:03d}.npy"
        rc.save_npy(out / rel, np.concatenate(blocks, axis=0).astype(np.float16))
        register(rel)
        nullz_files.append(rel)

    src = next(k for k in man["files"] if k.startswith("examples/"))
    keep = set(snps["rsid"][lo:hi].tolist())
    with gzip.open(full / src, "rb") as fh:
        lines = fh.read().split(b"\n")
    body = [ln for ln in lines[1:] if ln and ln.split(b"\t", 1)[0] in keep]
    rel = f"examples/height_chr{args.chrom}.sumstats.tsv.gz"
    rc.gzip_deterministic(b"\n".join([lines[0], *body]) + b"\n", out / rel)
    register(rel)
    print(f"  example rows kept: {len(body):,}")

    manifest = {
        "format_version": rc.FORMAT_VERSION,
        "name": name,
        "version": man["version"],
        "population": man["population"],
        "panel": man["panel"] + f"; chromosome {args.chrom} only",
        "genome_build": man["genome_build"],
        "n_ref": man["n_ref"],
        "n_snps": int(m),
        "source": man["source"],
        "ld_configs": ld_entries,
        "nullz": {"dtype": "float16", "n_replicates": int(n_rep), "files": nullz_files},
        "files": {k: files[k] for k in sorted(files)},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    total = sum(f["bytes"] for f in files.values())
    print(f"mini bundle '{name}': M={m:,}; {len(files)} data files, {total:,} bytes ({total / 2**20:.2f} MiB)")


if __name__ == "__main__":
    main()
