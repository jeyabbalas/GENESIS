#!/usr/bin/env python3
"""Verify a GENESIS reference bundle (format_version 1).

    python -I verify_bundle.py --bundle BUNDLE_DIR
        [--raw RAW_DIR]            also compare against the export_rdata.R output
        [--spotcheck FILE]         also compare against spotcheck_export.R output (by rsID)
        [--parent PARENT_BUNDLE]   also check that BUNDLE is an exact chromosome subset

Every check prints one line starting with PASS / FAIL / INFO; the exit status is non-zero
if any check failed. Without --raw/--spotcheck only bundle-internal invariants are tested,
so the script can be pointed at any bundle (for example one built for another panel).
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

CHI2_MEDIAN_1DF = 0.454936423119572  # qchisq(0.5, 1)
N_FAIL = 0


def check(ok: bool, msg: str) -> bool:
    global N_FAIL
    if not ok:
        N_FAIL += 1
    print(("PASS  " if ok else "FAIL  ") + msg, flush=True)
    return bool(ok)


def info(msg: str) -> None:
    print("INFO  " + msg, flush=True)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def fmt(x: float) -> str:
    return format(float(x), ".17g")


# ------------------------------------------------------------------------- internal
def verify_internal(bundle: Path):
    man = json.loads((bundle / "manifest.json").read_text())
    print(f"=== bundle-internal checks: {bundle.name} (name={man['name']}, version={man['version']}) ===")
    check(man["format_version"] == rc.FORMAT_VERSION, f"format_version == {rc.FORMAT_VERSION}")
    need = ["format_version", "name", "version", "population", "panel", "genome_build", "n_ref",
            "n_snps", "source", "ld_configs", "nullz", "files"]
    check(all(k in man for k in need), "manifest has all required keys")

    # files: existence, size, sha256, and nothing unlisted
    bad = []
    for rel, meta in man["files"].items():
        p = bundle / rel
        if not p.is_file() or p.stat().st_size != meta["bytes"] or rc.sha256_file(p) != meta["sha256"]:
            bad.append(rel)
    check(not bad, f"all {len(man['files'])} listed files exist with matching size and sha256" +
          (f" (bad: {bad})" if bad else ""))
    on_disk = sorted(str(p.relative_to(bundle)) for p in bundle.rglob("*")
                     if p.is_file() and p.name != "manifest.json")
    check(on_disk == sorted(man["files"]), "no unlisted files in the bundle directory")

    # ---- snps
    s = load_npz(bundle / "snps.npz")
    m = int(man["n_snps"])
    check(sorted(s) == ["chrom", "hm3_no_mhc", "pos", "pos_grch38", "rsid"], "snps.npz keys")
    check(s["rsid"].dtype.kind == "S" and s["chrom"].dtype == np.uint8 and s["pos"].dtype == np.int32
          and s["pos_grch38"].dtype == np.int32 and s["hm3_no_mhc"].dtype == np.bool_,
          f"snps.npz dtypes (rsid {s['rsid'].dtype})")
    check(all(v.shape == (m,) for v in s.values()), f"snps.npz arrays have length n_snps = {m:,}")
    check(np.unique(s["rsid"]).size == m, "rsids are unique")
    chrom, pos, rsid = s["chrom"].astype(np.int64), s["pos"].astype(np.int64), s["rsid"]
    key = chrom * (1 << 40) + pos
    tie = key[1:] == key[:-1]
    strictly = (key[1:] > key[:-1]) | (tie & (rsid[1:] > rsid[:-1]))
    check(bool(strictly.all()), f"canonical order strictly sorted by (chrom, pos, rsid); "
          f"{int(tie.sum())} adjacent pairs share (chrom, pos)")
    check(int(pos.min()) >= 1, "positions are >= 1")
    p38 = s["pos_grch38"]
    info(f"chromosomes present: {int(chrom.min())}..{int(chrom.max())}; "
         f"pos_grch38 == -1 for {int((p38 == -1).sum()):,} SNPs; hm3_no_mhc true for "
         f"{int(s['hm3_no_mhc'].sum()):,}")
    check(bool(((p38 == -1) | (p38 >= 1)).all()), "pos_grch38 is -1 or a positive position")

    # ---- LD configurations
    results = {}
    for entry in man["ld_configs"]:
        cid = entry["id"]
        z = load_npz(bundle / entry["file"])
        ok_keys = sorted(z) == ["indices", "indptr", "ldscore", "n_ref", "nstar", "r2_min", "window_kb"]
        ok_dt = (z["nstar"].dtype == np.int32 and z["ldscore"].dtype == np.float64
                 and z["indptr"].dtype == np.int64 and z["indices"].dtype == np.int32
                 and z["window_kb"].dtype == np.float64 and z["r2_min"].dtype == np.float64
                 and z["n_ref"].dtype == np.int64 and z["window_kb"].shape == () and z["n_ref"].shape == ())
        check(ok_keys and ok_dt, f"[{cid}] keys and dtypes")
        check(cid == rc.cfg_id(int(z["window_kb"]), format(float(z["r2_min"]), "g"))
              and float(z["window_kb"]) == entry["window_kb"] and float(z["r2_min"]) == entry["r2_min"]
              and int(z["n_ref"]) == man["n_ref"], f"[{cid}] scalars agree with id and manifest")
        indptr, indices32, nstar = z["indptr"], z["indices"], z["nstar"]
        indices = indices32.astype(np.int64)
        nnz = indices.size
        check(indptr.shape == (m + 1,) and indptr[0] == 0 and indptr[-1] == nnz
              and bool((np.diff(indptr) >= 1).all()), f"[{cid}] indptr well formed, no empty rows; nnz = {nnz:,}")
        check(np.array_equal(np.diff(indptr), nstar), f"[{cid}] len(row) == nstar for every SNP")
        rows = rc.csr_rows(indptr)
        check(bool(nnz and indices.min() >= 0 and indices.max() < m), f"[{cid}] indices within [0, M)")
        inc = indices[1:] > indices[:-1]
        same_row = rows[1:] == rows[:-1]
        check(bool((inc | ~same_row).all()), f"[{cid}] every row strictly ascending (no duplicates)")
        self_count = np.bincount(rows[indices == rows], minlength=m)
        check(bool((self_count == 1).all()), f"[{cid}] self index present exactly once in every row")
        k1 = rows * m + indices  # sorted by construction
        k2 = np.sort(indices * m + rows)
        sym = np.array_equal(k1, k2)
        n_asym = 0 if sym else int(np.setdiff1d(k1, k2, assume_unique=True).size)
        check(sym, f"[{cid}] neighbour relation symmetric (asymmetric ordered pairs: {n_asym})")
        del k1, k2
        check(bool((chrom[rows] == chrom[indices]).all()), f"[{cid}] all neighbours on the same chromosome")
        dist = np.abs(pos[rows] - pos[indices])
        window_bp = int(round(float(z["window_kb"]) * 1000))
        dmax = int(dist.max())
        check(dmax <= window_bp, f"[{cid}] max |pos difference| = {dmax:,} bp <= window {window_bp:,} bp "
              f"(pairs at exactly the window: {int((dist == window_bp).sum())}; "
              f"within 1 kb of it: {int((dist > window_bp - 1000).sum()):,})")
        ld = z["ldscore"]
        info(f"[{cid}] nstar min/median/mean/max = {int(nstar.min())}/{float(np.median(nstar)):g}/"
             f"{float(nstar.mean()):.4f}/{int(nstar.max())}; ldscore min/mean/max = "
             f"{fmt(ld.min())}/{float(ld.mean()):.6f}/{fmt(ld.max())}; sum = {float(ld.sum()):.10f}")
        check(bool(np.isfinite(ld).all() and (ld > 0).all()), f"[{cid}] ldscore finite and positive")
        results[cid] = {"indptr": indptr, "indices": indices32, "nstar": nstar, "ldscore": ld}

        if entry.get("pairld"):
            r2u = load_npz(bundle / entry["pairld"])["r2_u16"]
            check(r2u.dtype == np.uint16 and r2u.shape == (nnz,), f"[{cid}] pairld r2_u16 aligned with indices")
            check(bool((r2u[indices == rows] == 65535).all()), f"[{cid}] pairld: r2 of every SNP with itself == 1")
            r2q = r2u.astype(np.float64) / 65535.0
            n_ref = int(z["n_ref"])
            sums = np.add.reduceat(r2q - (1.0 - r2q) / (n_ref - 2), indptr[:-1])
            d = np.abs(sums - ld)
            info(f"[{cid}] pairld: quantised r2 min = {fmt(r2q.min())} (u16 {int(r2u.min())}); "
                 f"|sum_row(adj r2_q) - ldscore| max = {d.max():.3e}, mean = {d.mean():.3e}")
            check(float(d.max()) < 1e-2, f"[{cid}] ldscore reproduced from quantised r2 within 1e-2")
            # symmetry of the stored values: r2(i,j) == r2(j,i)
            order = np.argsort(indices * m + rows, kind="stable")
            check(np.array_equal(r2u[order], r2u), f"[{cid}] pairld: r2(i,j) == r2(j,i) for every stored pair")
            results[cid]["r2_u16"] = r2u
        del rows, dist, indices

    # ---- null z
    nz = man["nullz"]
    reps = 0
    for rel in nz["files"]:
        a = np.load(bundle / rel, mmap_mode="r", allow_pickle=False)
        ok = a.dtype == np.float16 and a.ndim == 2 and a.shape[1] == m and a.flags["C_CONTIGUOUS"]
        check(ok and bool(np.isfinite(np.asarray(a, dtype=np.float32)).all()),
              f"[{rel}] float16, shape {a.shape}, C order, all finite")
        reps += a.shape[0]
    check(reps == nz["n_replicates"] and nz["dtype"] == "float16",
          f"nullz replicate count == {nz['n_replicates']}")

    # ---- examples
    for rel in man["files"]:
        if rel.startswith("examples/"):
            with gzip.open(bundle / rel, "rb") as fh:
                lines = fh.read().split(b"\n")
            body = [ln for ln in lines[1:] if ln]
            ex_snps = np.array([ln.split(b"\t", 1)[0] for ln in body], dtype="S")
            n_in = int(np.isin(ex_snps, rsid).sum())
            check(lines[0] == b"SNP\tZ\tN" and all(ln.count(b"\t") == 2 for ln in body[:1000]),
                  f"[{rel}] header SNP/Z/N; {len(body):,} rows; {n_in:,} rows are reference SNPs")
    return man, s, results


# ------------------------------------------------------------------------- vs raw
def verify_raw(bundle: Path, raw: Path, man: dict, s: dict, results: dict) -> None:
    print("=== comparison with the raw upstream exports ===")
    rsid, chrom, pos = s["rsid"], s["chrom"], s["pos"]
    m = rsid.size
    base_names = None
    for cfg in rc.all_configs():
        cid, d = cfg["id"], raw / cfg["raw_dir"]
        names = rc.read_names(d / "snpname.txt")
        if base_names is None:
            base_names = names
        same_set = names.size == m and np.array_equal(np.sort(names), np.sort(base_names))
        same_order = np.array_equal(names, base_names)
        check(same_set, f"[{cid}] upstream SNP set identical to the other configurations "
              f"(rows = {names.size:,}; same row order as first config: {same_order})")
        u2c = rc.index_in(names, rsid)
        check(bool((u2c >= 0).all()) and np.unique(u2c).size == m, f"[{cid}] every upstream rsid is in snps.npz")
        check(np.array_equal(rc.read_i32(d / "chr.i32"), chrom[u2c])
              and np.array_equal(rc.read_i32(d / "bp.i32"), pos[u2c]), f"[{cid}] CHR and BP match upstream")
        r = results[cid]
        check(np.array_equal(rc.read_i32(d / "nstar.i32"), r["nstar"][u2c]), f"[{cid}] nstar matches upstream")
        up_ld = rc.read_f64(d / "ldscore.f64")
        check(np.array_equal(up_ld.view(np.uint64), r["ldscore"][u2c].view(np.uint64)),
              f"[{cid}] ldscore is a bit-exact copy of upstream LD.score.correct")
        tag_len = rc.read_i32(d / "tag_len.i32")
        tag_idx = rc.read_i32(d / "tag_idx0.i32")
        up_rows = np.repeat(np.arange(m, dtype=np.int64), tag_len)
        k_up = np.sort(u2c[up_rows] * m + u2c[tag_idx])
        k_b = rc.csr_rows(r["indptr"]) * m + r["indices"]
        check(np.array_equal(k_up, k_b), f"[{cid}] neighbour lists equal upstream TaggingSNPs (as sets per row)")
        del k_up, k_b

        if cid == rc.PAIR_CFG:
            pdir = raw / "pair_w1MB_c0.1"
            plen = rc.read_i32(pdir / "len.i32")
            pidx = rc.read_i32(pdir / "tagidx0.i32")
            r2 = rc.read_f64(pdir / "r2.f64")
            check(np.array_equal(plen, tag_len), f"[{cid}] TaggingSNPinx row lengths == TaggingSNPs row lengths")
            a = np.sort(up_rows * m + tag_idx.astype(np.int64))
            b = np.sort(up_rows * m + pidx.astype(np.int64))
            check(np.array_equal(a, b), f"[{cid}] string-derived indices == TaggingSNPinx indices as sets per row "
                  f"(identical element order too: {np.array_equal(tag_idx, pidx)})")
            del a, b
            starts = np.concatenate(([0], np.cumsum(plen, dtype=np.int64)[:-1]))
            n_ref = man["n_ref"]
            sums = np.add.reduceat(r2 - (1.0 - r2) / (n_ref - 2), starts)
            d0 = np.abs(sums - up_ld)
            check(float(d0.max()) < 1e-9, f"[{cid}] ldscore == sum_row(r2 - (1 - r2)/(n_ref - 2)) with the "
                  f"original double r2: max abs diff = {d0.max():.3e}")
            raw_sum = np.add.reduceat(r2, starts)
            info(f"[{cid}] (for contrast) max |sum_row(raw r2) - ldscore| = {np.abs(raw_sum - up_ld).max():.6f}")
            rmin = float(r2.min())
            info(f"[{cid}] original r2: min = {fmt(rmin)} (== double 0.1: {rmin == 0.1}; count at min: "
                 f"{int((r2 == rmin).sum()):,}; count < 0.1: {int((r2 < 0.1).sum())}); max = {fmt(r2.max())}; "
                 f"count == 1: {int((r2 == 1.0).sum()):,}")
            scaled = r2 * 1e6
            n_6dp = int((np.abs(scaled - np.rint(scaled)) < 1e-6).sum())
            scaled8 = r2 * 1e8
            n_8dp = int((np.abs(scaled8 - np.rint(scaled8)) < 1e-6).sum())
            info(f"[{cid}] original r2 with at most 6 decimals: {n_6dp:,} of {r2.size:,} "
                 f"({100.0 * n_6dp / r2.size:.4f}%); at most 8 decimals: {100.0 * n_8dp / r2.size:.4f}%")
            # bundle r2 aligned to the bundle CSR order
            order = np.argsort(u2c[up_rows] * m + u2c[pidx], kind="stable")
            r2_can = r2[order]
            r2q = r["r2_u16"].astype(np.float64) / 65535.0
            check(np.array_equal(np.rint(r2_can * 65535.0).astype(np.uint16), r["r2_u16"]),
                  f"[{cid}] pairld r2_u16 == round(r2 * 65535) element for element")
            qerr = np.abs(r2q - r2_can)
            sums_q = np.add.reduceat(r2q - (1.0 - r2q) / (n_ref - 2), r["indptr"][:-1])
            dq = np.abs(sums_q - r["ldscore"])
            info(f"[{cid}] uint16 quantisation: max |r2_q - r2| = {qerr.max():.3e} (bound {0.5 / 65535:.3e}), "
                 f"mean = {qerr.mean():.3e}; effect on row LD score: max = {dq.max():.3e}, mean = {dq.mean():.3e}, "
                 f"max relative = {(dq / r['ldscore']).max():.3e}")
            info(f"[{cid}] quantised values below the cutoff: {int((r2q < 0.1).sum()):,} "
                 f"(smallest {fmt(r2q.min())}); |sqrt(r2_q) - sqrt(r2)| max = "
                 f"{np.abs(np.sqrt(r2q) - np.sqrt(r2_can)).max():.3e}")
            del r2, r2_can, r2q, order
        del up_rows, tag_idx

    # ---- null z
    ndir = raw / "nullz"
    e2c = rc.index_in(rc.read_names(ndir / "snpname.txt"), rsid)
    check(bool((e2c >= 0).all()) and np.unique(e2c).size == m, "nullz: upstream SNP set == reference SNP set")
    max_q = 0.0
    all_equal = True
    report = {1, 2, 50, 100}
    per_file = None
    for it in range(1, man["nullz"]["n_replicates"] + 1):
        v = rc.read_f64(ndir / f"error_iter{it:03d}.f64")
        canon = np.empty(m)
        canon[e2c] = v
        if per_file is None:
            per_file = np.load(bundle / man["nullz"]["files"][0], mmap_mode="r").shape[0]
        f, row = divmod(it - 1, per_file)
        stored = np.asarray(np.load(bundle / man["nullz"]["files"][f], mmap_mode="r")[row])
        all_equal &= bool(np.array_equal(stored, canon.astype(np.float16)))
        st = stored.astype(np.float64)
        max_q = max(max_q, float(np.abs(st - canon).max()))
        if it in report:
            info(f"nullz replicate {it:3d} (error_iter{it}): raw mean/sd/lambda = {canon.mean():+.6f}/"
                 f"{canon.std(ddof=1):.6f}/{np.median(canon**2) / CHI2_MEDIAN_1DF:.6f} | float16 = "
                 f"{st.mean():+.6f}/{st.std(ddof=1):.6f}/{np.median(st**2) / CHI2_MEDIAN_1DF:.6f} | "
                 f"max|err| = {np.abs(st - canon).max():.3e}, raw min/max = {canon.min():.4f}/{canon.max():.4f}")
    check(all_equal, "nullz: every stored replicate == float16(upstream error) in canonical order")
    info(f"nullz: max abs float16 quantisation error over all replicates = {max_q:.3e}")

    # ---- example
    src = (raw / "height" / "heightGWAS.tsv").read_bytes()
    rel = next(k for k in man["files"] if k.startswith("examples/"))
    with gzip.open(bundle / rel, "rb") as fh:
        got = fh.read()
    n_rows = src.count(b"\n") - 1
    check(got == src, f"[{rel}] decompresses to the exported heightGWAS table byte for byte ({n_rows:,} rows)")
    hm3 = np.array([ln.split(b"\t", 1)[0] for ln in
                    (raw / "hm3" / "w_hm3.noMHC.snplist.tsv").read_bytes().split(b"\n")[1:] if ln], dtype="S")
    check(np.array_equal(np.isin(rsid, hm3), s["hm3_no_mhc"]), "hm3_no_mhc == membership in w_hm3.noMHC.snplist")


# -------------------------------------------------------------------- vs spot check
def verify_spotcheck(bundle: Path, spot: Path, man: dict, s: dict, results: dict) -> None:
    print("=== independent by-rsID spot check against the upstream R objects ===")
    rsid = s["rsid"]
    recs: dict[str, list[list[str]]] = {}
    for line in spot.read_text().split("\n"):
        if line:
            f = line.split("\t")
            recs.setdefault(f[0], []).append(f[1:])
    snp_list = [r[0] for r in recs["SNP"]]
    idx = rc.index_in(np.array([x.encode() for x in snp_list], dtype="S"), rsid)
    check(bool((idx >= 0).all()), f"all {len(snp_list)} spot-check SNPs are in snps.npz")
    row_of = dict(zip(snp_list, idx.tolist()))
    name_of = lambda i: rsid[i].decode()  # noqa: E731

    by_cfg = {c["id"]: c for c in man["ld_configs"]}
    n_cfg = bad_cfg = 0
    for snp, cid, chrom, bp, nstar, ld, tags in recs["CFG"]:
        if cid not in by_cfg:
            continue
        n_cfg += 1
        i = row_of[snp]
        r = results[cid]
        a, b = int(r["indptr"][i]), int(r["indptr"][i + 1])
        names = {name_of(j) for j in r["indices"][a:b]}
        tag_list = tags.split(",")
        ok = (int(s["chrom"][i]) == int(chrom) and int(s["pos"][i]) == int(bp) and int(r["nstar"][i]) == int(nstar)
              and float(r["ldscore"][i]) == float(ld) and names == set(tag_list) and len(tag_list) == b - a
              and snp in names)
        bad_cfg += not ok
    check(bad_cfg == 0, f"CFG records: {n_cfg} (SNP x configuration) entries match on chr, bp, nstar, ldscore "
          f"(exact) and the tagging rsID set; mismatches: {bad_cfg}")

    n_pair = bad_pair = 0
    max_err = 0.0
    pair_cfg = next((c["id"] for c in man["ld_configs"] if c.get("pairld")), None)
    if pair_cfg:
        r = results[pair_cfg]
        for snp, names, r2s in recs["PAIR"]:
            n_pair += 1
            i = row_of[snp]
            a, b = int(r["indptr"][i]), int(r["indptr"][i + 1])
            got = {name_of(j): int(u) for j, u in zip(r["indices"][a:b], r["r2_u16"][a:b])}
            exp = dict(zip(names.split(","), (float(x) for x in r2s.split(","))))
            ok = set(got) == set(exp) and all(got[k] == int(np.rint(v * 65535.0)) for k, v in exp.items())
            if set(got) == set(exp):
                max_err = max(max_err, max(abs(got[k] / 65535.0 - v) for k, v in exp.items()))
            bad_pair += not ok
        check(bad_pair == 0, f"PAIR records: {n_pair} rows match neighbour-by-neighbour on round(r2*65535) "
              f"(max |r2_q - r2| = {max_err:.3e}); mismatches: {bad_pair}")

    n_null = bad_null = 0
    per_file = np.load(bundle / man["nullz"]["files"][0], mmap_mode="r").shape[0]
    cache: dict[int, np.ndarray] = {}
    for snp, it, val in recs["NULL"]:
        it = int(it)
        if it > man["nullz"]["n_replicates"]:
            continue
        n_null += 1
        f, row = divmod(it - 1, per_file)
        if f not in cache:
            cache[f] = np.load(bundle / man["nullz"]["files"][f], mmap_mode="r")
        bad_null += not (cache[f][row, row_of[snp]] == np.float16(float(val)))
    check(bad_null == 0, f"NULL records: {n_null} values == float16(upstream error); mismatches: {bad_null}")

    bad_hm3 = sum(bool(s["hm3_no_mhc"][row_of[snp]]) != (flag == "1") for snp, flag in recs["HM3"])
    check(bad_hm3 == 0, f"HM3 records: {len(recs['HM3'])} flags match; mismatches: {bad_hm3}")

    rel = next(k for k in man["files"] if k.startswith("examples/"))
    with gzip.open(bundle / rel, "rt") as fh:
        ex = {}
        want = set(snp_list)
        for line in fh:
            snp, _, rest = line.rstrip("\n").partition("\t")
            if snp in want:
                ex[snp] = rest.split("\t")
    n_present = bad_h = 0
    for snp, z, n in recs["HEIGHT"]:
        if z == "NA":
            bad_h += snp in ex
        else:
            n_present += 1
            bad_h += ex.get(snp) != [z, n]
    check(bad_h == 0, f"HEIGHT records: {len(recs['HEIGHT'])} SNPs ({n_present} present in the example) match "
          f"on Z and N text; mismatches: {bad_h}")


# ------------------------------------------------------------------------ vs parent
def verify_parent(bundle: Path, parent: Path, man: dict, s: dict, results: dict) -> None:
    print("=== subset consistency with the parent bundle ===")
    ps = load_npz(parent / "snps.npz")
    pman = json.loads((parent / "manifest.json").read_text())
    idx = rc.index_in(s["rsid"], ps["rsid"])
    lo, hi = int(idx[0]), int(idx[-1]) + 1
    check(bool((idx >= 0).all()) and np.array_equal(idx, np.arange(lo, hi)),
          f"subset SNPs form a contiguous block of the parent: rows {lo:,}..{hi - 1:,}")
    check(all(np.array_equal(s[k], ps[k][lo:hi]) for k in ps), "snps.npz equals the parent slice")
    chroms = np.unique(s["chrom"])
    check(int((np.isin(ps["chrom"], chroms)).sum()) == s["rsid"].size,
          f"subset holds every parent SNP on chromosome(s) {chroms.tolist()}")
    pby = {c["id"]: c for c in pman["ld_configs"]}
    for entry in man["ld_configs"]:
        cid = entry["id"]
        pz = load_npz(parent / pby[cid]["file"])
        a, b = int(pz["indptr"][lo]), int(pz["indptr"][hi])
        pind = pz["indices"][a:b].astype(np.int64)
        r = results[cid]
        inside = bool(pind.size and pind.min() >= lo and pind.max() < hi)
        check(inside, f"[{cid}] every neighbour of a subset SNP is inside the subset")
        check(np.array_equal(pind - lo, r["indices"]) and np.array_equal(pz["indptr"][lo:hi + 1] - a, r["indptr"])
              and np.array_equal(pz["nstar"][lo:hi], r["nstar"])
              and np.array_equal(pz["ldscore"][lo:hi], r["ldscore"]), f"[{cid}] ld arrays equal the re-based parent slice")
        if entry.get("pairld"):
            pr = load_npz(parent / pby[cid]["pairld"])["r2_u16"][a:b]
            check(np.array_equal(pr, r["r2_u16"]), f"[{cid}] pairld equals the parent slice")
    got = 0
    for rel in man["nullz"]["files"]:
        a = np.load(bundle / rel, mmap_mode="r")
        for row in range(a.shape[0]):
            f, prow = divmod(got, np.load(parent / pman["nullz"]["files"][0], mmap_mode="r").shape[0])
            parr = np.load(parent / pman["nullz"]["files"][f], mmap_mode="r")
            if not np.array_equal(np.asarray(a[row]), np.asarray(parr[prow, lo:hi])):
                check(False, f"nullz replicate {got} differs from the parent slice")
            got += 1
    check(got == man["nullz"]["n_replicates"], f"nullz: first {got} parent replicates, subset columns, identical")
    prel = next(k for k in pman["files"] if k.startswith("examples/"))
    rel = next(k for k in man["files"] if k.startswith("examples/"))
    keep = set(s["rsid"].tolist())
    with gzip.open(parent / prel, "rb") as fh:
        plines = fh.read().split(b"\n")
    exp = b"\n".join([plines[0], *[ln for ln in plines[1:] if ln and ln.split(b"\t", 1)[0] in keep]]) + b"\n"
    with gzip.open(bundle / rel, "rb") as fh:
        check(fh.read() == exp, f"[{rel}] == parent example rows whose SNP is in the subset")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", required=True, type=Path)
    ap.add_argument("--raw", type=Path)
    ap.add_argument("--spotcheck", type=Path)
    ap.add_argument("--parent", type=Path)
    args = ap.parse_args()

    man, s, results = verify_internal(args.bundle)
    if args.raw:
        verify_raw(args.bundle, args.raw, man, s, results)
    if args.spotcheck:
        verify_spotcheck(args.bundle, args.spotcheck, man, s, results)
    if args.parent:
        verify_parent(args.bundle, args.parent, man, s, results)

    print("=== file sizes ===")
    total = 0
    for rel in sorted(man["files"]):
        b = man["files"][rel]["bytes"]
        total += b
        print(f"  {b:>12,}  {rel}")
    mb = (args.bundle / "manifest.json").stat().st_size
    print(f"  {mb:>12,}  manifest.json")
    print(f"  {total + mb:>12,}  TOTAL ({(total + mb) / 2**20:.2f} MiB)")
    print(f"=== {'ALL CHECKS PASSED' if N_FAIL == 0 else f'{N_FAIL} CHECK(S) FAILED'} ===")
    sys.exit(1 if N_FAIL else 0)


if __name__ == "__main__":
    main()
