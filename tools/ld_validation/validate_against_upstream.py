#!/usr/bin/env python
"""Rebuild the LD tables of the original GENESIS package from genotypes and compare.

The original R package ships, for 1,070,777 HapMap3 SNPs, the number of tagged SNPs (N*),
bias-corrected LD scores and tagging lists for nine (window, r2 threshold) configurations,
but not the code that produced them. This script recomputes all nine tables from the
genotypes they were derived from (the LDSC "1000G_EUR_Phase3_plink" files, 489 samples) with
``genesis_gwas.ld.compute_ld`` and reports the agreement, both with the setting that emulates
the original processing (``round_r2=6``) and with the recommended default (exact r2).

Example
-------
    python validate_against_upstream.py \
        --plink "/path/1000G_EUR_Phase3_plink/1000G.EUR.QC.{chrom}" \
        --bundle /path/eur_1kg_hm3 --raw /path/raw --nullz --out report.json

``--bundle`` is the converted upstream reference (see tools/refdata); ``--raw`` (optional) is
the directory of raw upstream exports, used to compare pairwise r2 with the original doubles.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

import genesis_gwas as gg
from genesis_gwas.ld import BedReader, compute_ld, null_z, standardize_genotypes

WINDOWS = (500.0, 1000.0, 2000.0)
CUTOFFS = (0.05, 0.1, 0.2)
CONFIGS = [(w, c) for w in WINDOWS for c in CUTOFFS]
PAIR_CFG = (1000.0, 0.1)
NULL_KINDS = ("upstream", "t", "normal", "projection")


def cfg_name(cfg: tuple[float, float]) -> str:
    return f"w{int(cfg[0])}kb_r{cfg[1]:g}"


def peak_rss_gb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1 << 30) if sys.platform == "darwin" else rss / (1 << 20)


def parse_chroms(text: str) -> list[int]:
    out: list[int] = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def new_stats() -> dict:
    return {
        "n_snps": 0,
        "nstar_equal": 0,
        "rows_identical": 0,
        "pairs_ours": 0,
        "pairs_upstream": 0,
        "pairs_only_ours": 0,
        "pairs_only_upstream": 0,
        "max_abs_dld": 0.0,
        "n_dld_gt_1e-9": 0,
        "examples": [],
    }


def compare(
    stats: dict, table, up_nstar, up_ld, up_indptr, up_indices, rsid, pos, chrom, G, max_examples=25
):
    """Accumulate agreement statistics of one chromosome / configuration."""
    m = table.n_snps
    stats["n_snps"] += m
    stats["nstar_equal"] += int(np.sum(table.nstar == up_nstar))
    stats["pairs_ours"] += table.nnz
    stats["pairs_upstream"] += int(up_indices.size)
    dld = np.abs(table.ldscore - up_ld)
    stats["max_abs_dld"] = max(stats["max_abs_dld"], float(dld.max()))
    stats["n_dld_gt_1e-9"] += int(np.sum(dld > 1e-9))
    if (
        table.nnz == up_indices.size
        and np.array_equal(table.indices, up_indices)
        and np.array_equal(table.indptr, up_indptr)
    ):
        stats["rows_identical"] += m
        return
    ours = np.repeat(np.arange(m, dtype=np.int64), table.nstar) * m + table.indices
    theirs = np.repeat(np.arange(m, dtype=np.int64), np.diff(up_indptr)) * m + up_indices
    only_ours = np.setdiff1d(ours, theirs, assume_unique=True)
    only_theirs = np.setdiff1d(theirs, ours, assume_unique=True)
    stats["pairs_only_ours"] += int(only_ours.size)
    stats["pairs_only_upstream"] += int(only_theirs.size)
    bad_rows = np.union1d(only_ours // m, only_theirs // m)
    stats["rows_identical"] += m - int(bad_rows.size)
    Z = None
    for keys, who in ((only_ours, "ours"), (only_theirs, "upstream")):
        for key in keys[: max(0, max_examples - len(stats["examples"]))]:
            i, j = int(key // m), int(key % m)
            if Z is None:
                Z = standardize_genotypes(G)
            stats["examples"].append(
                {
                    "chrom": int(chrom),
                    "snp_i": rsid[i],
                    "snp_j": rsid[j],
                    "distance_bp": int(abs(int(pos[i]) - int(pos[j]))),
                    "r2_exact": float((Z[i] @ Z[j]) ** 2),
                    "only_in": who,
                }
            )


def load_raw_pair_r2(raw: Path, ref) -> tuple[np.ndarray, np.ndarray]:
    """Original double-precision pairwise r2 as (sorted canonical pair keys, values)."""
    d_ld = raw / "ld_w1MB_c0.1"
    d_pr = raw / "pair_w1MB_c0.1"
    names = np.array(d_ld.joinpath("snpname.txt").read_text().split(), dtype="S")
    canon = ref.lookup(names)
    assert np.all(canon >= 0)
    lens = np.fromfile(d_pr / "len.i32", dtype="<i4").astype(np.int64)
    idx0 = np.fromfile(d_pr / "tagidx0.i32", dtype="<i4")
    r2 = np.fromfile(d_pr / "r2.f64", dtype="<f8")
    rows = np.repeat(canon, lens)
    cols = canon[idx0]
    keys = rows * ref.M + cols
    order = np.argsort(keys, kind="stable")
    return keys[order], r2[order]


def nullz_check(G: np.ndarray, pos: np.ndarray, up_null: np.ndarray, n_rep: int, seed: int) -> dict:
    """Compare the local LD structure of our null replicates (default statistic) with upstream's.

    For random SNP pairs up to 50 SNPs apart, the product ``u_i * u_j`` averaged over replicates
    estimates the covariance of the two null statistics, which should equal the signed genotype
    correlation ``r_ij``: the regression slope of that average on ``r`` should be one.
    """
    Z = standardize_genotypes(G)
    ours = null_z(G, n_rep, seed=seed).astype(np.float64)
    m = G.shape[0]
    rng = np.random.default_rng(seed)
    # random pairs at short range: lag 1..50 in SNP index
    i = rng.integers(0, m - 60, 400_000)
    j = i + rng.integers(1, 51, i.size)
    r = np.einsum("ij,ij->i", Z[i], Z[j])
    out: dict = {"n_replicates": n_rep, "n_pairs": int(i.size), "n_replicates_precision_run": 4000}
    many = null_z(G, 4000, seed=seed + 1).astype(np.float64)
    for label, u in (
        ("ours", ours),
        ("upstream", up_null[:n_rep].astype(np.float64)),
        ("ours_4000_replicates", many),
    ):
        prod = np.zeros(i.size)
        for a in range(0, u.shape[0], 500):  # estimates cov(u_i, u_j) = r_ij
            prod += (u[a : a + 500][:, i] * u[a : a + 500][:, j]).sum(axis=0)
        prod /= u.shape[0]
        slope = float(r @ prod / (r @ r))
        slope_abs = float(np.abs(r) @ prod / (r @ r))
        strong = np.abs(r) > 0.5
        out[label] = {
            "mean": float(u.mean()),
            "variance": float(u.var()),
            "per_replicate_lambda_gc_mean": float(
                np.mean(np.median(u**2, axis=1) / 0.45493642311957283)
            ),
            "slope_of_cov_on_signed_r": slope,
            "slope_of_cov_on_abs_r": slope_abs,
            "corr_cov_vs_signed_r": float(np.corrcoef(r, prod)[0, 1]),
            "sign_agreement_when_abs_r_gt_0.5": float(
                np.mean(np.sign(prod[strong]) == np.sign(r[strong]))
            ),
            "frac_negative_r_among_strong": float(np.mean(r[strong] < 0)),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--plink", required=True, help="per-chromosome PLINK prefix template containing {chrom}"
    )
    ap.add_argument(
        "--bundle", required=True, help="converted upstream reference bundle (eur_1kg_hm3)"
    )
    ap.add_argument(
        "--raw", default=None, help="raw upstream exports (optional; exact pairwise r2 check)"
    )
    ap.add_argument("--chroms", default="1-22")
    ap.add_argument("--block-size", type=int, default=1024)
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--no-unrounded",
        action="store_true",
        help="skip the comparison of the default (exact r2) build",
    )
    ap.add_argument(
        "--nullz",
        action="store_true",
        help="statistical check of null_z against upstream replicates",
    )
    ap.add_argument("--nullz-chrom", type=int, default=22)
    ap.add_argument("--out", default=None, help="write the full report as JSON")
    args = ap.parse_args()
    chroms = parse_chroms(args.chroms)

    refs = {cfg: gg.load_reference(args.bundle, window_kb=cfg[0], r2_min=cfg[1]) for cfg in CONFIGS}
    base = refs[PAIR_CFG]
    M = base.M
    modes = [("round_r2=6", 6)] + ([] if args.no_unrounded else [("round_r2=None", None)])
    stats = {mode: {cfg_name(c): new_stats() for c in CONFIGS} for mode, _ in modes}
    seconds = {mode: 0.0 for mode, _ in modes}
    pair = {
        "n_pairs": 0,
        "max_abs_diff_vs_u16": 0.0,
        "n_compared_raw": 0,
        "n_exact_vs_raw": 0,
        "n_within_1ulp_vs_raw": 0,
        "n_diff_last_digit": 0,
        "max_abs_diff_vs_raw": 0.0,
        "last_digit_examples": [],
    }
    raw_keys = raw_vals = None
    if args.raw:
        raw_keys, raw_vals = load_raw_pair_r2(Path(args.raw), base)
    up_pair = base.pair_r2()
    null_report = None
    null_scale: dict | None = None
    if args.nullz:
        n_null = base.n_null_replicates
        null_scale = {
            "phenotypes": np.random.default_rng(12345).standard_normal((n_null, base.n_ref)),
            "upstream": np.stack([base.null_z(i) for i in range(n_null)]),
            "sum2": {k: np.zeros(n_null) for k in NULL_KINDS},
            "sum4": {k: np.zeros(n_null) for k in NULL_KINDS},
            "n_snps": 0,
        }
    read_seconds = 0.0
    t_all = time.perf_counter()
    for chrom in chroms:
        sel = np.flatnonzero(base.chrom == chrom)
        if sel.size == 0:
            continue
        off, m = int(sel[0]), int(sel.size)
        assert np.array_equal(sel, np.arange(off, off + m)), (
            "canonical order must be chromosome-contiguous"
        )
        t0 = time.perf_counter()
        reader, bim, _fam = BedReader.from_prefix(args.plink.format(chrom=chrom))
        lookup = {s: i for i, s in enumerate(bim["snp"].tolist())}
        rsid = base.rsid[sel].astype("U").tolist()
        vidx = np.array([lookup[s] for s in rsid], dtype=np.int64)
        pos = base.pos[sel].astype(np.int64)
        assert np.array_equal(bim["pos"].to_numpy()[vidx], pos), (
            "bim positions differ from the reference"
        )
        G = reader.read(vidx)
        read_seconds += time.perf_counter() - t0
        for mode, digits in modes:
            t0 = time.perf_counter()
            tables = compute_ld(
                G,
                pos,
                configs=CONFIGS,
                round_r2=digits,
                keep_r2=[c == PAIR_CFG for c in CONFIGS],
                block_size=args.block_size,
                device=args.device,
            )
            seconds[mode] += time.perf_counter() - t0
            for cfg, table in zip(CONFIGS, tables, strict=True):
                ref = refs[cfg]
                nb = ref.neighborhoods
                a, b = int(nb.indptr[off]), int(nb.indptr[off + m])
                compare(
                    stats[mode][cfg_name(cfg)],
                    table,
                    ref.nstar[sel],
                    ref.ldscore[sel],
                    nb.indptr[off : off + m + 1] - a,
                    nb.indices[a:b].astype(np.int64) - off,
                    rsid,
                    pos,
                    chrom,
                    G,
                )
                if (
                    digits == 6
                    and cfg == PAIR_CFG
                    and table.nnz == b - a
                    and np.array_equal(table.indices, nb.indices[a:b].astype(np.int64) - off)
                ):
                    pair["n_pairs"] += table.nnz
                    pair["max_abs_diff_vs_u16"] = max(
                        pair["max_abs_diff_vs_u16"], float(np.abs(table.r2 - up_pair[a:b]).max())
                    )
                    if raw_keys is not None:
                        keys = (np.repeat(np.arange(m, dtype=np.int64), table.nstar) + off) * M + (
                            table.indices.astype(np.int64) + off
                        )
                        loc = np.searchsorted(raw_keys, keys)
                        assert np.array_equal(raw_keys[loc], keys)
                        d = np.abs(raw_vals[loc] - table.r2)
                        pair["n_compared_raw"] += int(d.size)
                        pair["n_exact_vs_raw"] += int(np.sum(d == 0))
                        pair["n_within_1ulp_vs_raw"] += int(np.sum(d <= 2.3e-16))
                        pair["max_abs_diff_vs_raw"] = max(
                            pair["max_abs_diff_vs_raw"], float(d.max())
                        )
                        bad = np.flatnonzero(d > 1e-9)
                        pair["n_diff_last_digit"] += int(bad.size)
                        if bad.size:
                            zs = standardize_genotypes(G)
                            rr = np.repeat(np.arange(m, dtype=np.int64), table.nstar)
                            for q in bad[:20]:
                                i, j = int(rr[q]), int(table.indices[q])
                                pair["last_digit_examples"].append(
                                    {
                                        "chrom": int(chrom),
                                        "snp_i": rsid[i],
                                        "snp_j": rsid[j],
                                        "r2_exact": repr(float((zs[i] @ zs[j]) ** 2)),
                                        "ours": repr(float(table.r2[q])),
                                        "upstream": repr(float(raw_vals[loc][q])),
                                    }
                                )
        if null_scale is not None:
            null_scale["n_snps"] += m
            for kind in NULL_KINDS:
                if kind == "upstream":
                    u = null_scale["upstream"][:, sel].astype(np.float64)
                else:
                    u = null_z(G, phenotypes=null_scale["phenotypes"], statistic=kind).astype(
                        np.float64
                    )
                null_scale["sum2"][kind] += (u * u).sum(axis=1)
                null_scale["sum4"][kind] += (u**4).sum(axis=1)
            if chrom == args.nullz_chrom:
                null_report = nullz_check(
                    G, pos, null_scale["upstream"][:, sel], base.n_null_replicates, seed=1
                )
        print(
            f"chr{chrom:<2d} {m:>7,} SNPs done ({time.perf_counter() - t_all:6.1f} s elapsed)",
            flush=True,
        )

    scale_table = None
    if null_scale is not None and null_scale["n_snps"]:
        n_tot = null_scale["n_snps"]
        scale_table = {
            "n_snps": n_tot,
            "n_replicates": int(base.n_null_replicates),
            "n_ref": base.n_ref,
        }
        for kind in NULL_KINDS:
            v = null_scale["sum2"][kind] / n_tot
            scale_table[kind] = {
                "mean_of_replicate_mean_z2": float(v.mean()),
                "sd_of_replicate_mean_z2": float(v.std(ddof=1)),
                "min": float(v.min()),
                "max": float(v.max()),
                "mean_z4": float((null_scale["sum4"][kind] / n_tot).mean()),
            }
        scale_table["theory_t_variance"] = (base.n_ref - 2) / (base.n_ref - 4)

    report = {
        "plink": args.plink,
        "bundle": str(args.bundle),
        "chromosomes": chroms,
        "genesis_gwas_version": gg.__version__,
        "numpy": np.__version__,
        "block_size": args.block_size,
        "device": args.device,
        "seconds_reading_genotypes": read_seconds,
        "seconds_compute_ld_all_9_configs": seconds,
        "peak_rss_gb": peak_rss_gb(),
        "agreement": stats,
        "pairwise_r2_w1000kb_r0.1": pair,
        "null_z": null_report,
        "null_z_genome_wide_scale": scale_table,
    }
    for mode, _ in modes:
        print(
            f"\n=== {mode}: compute_ld, nine configurations in one pass, {seconds[mode]:.1f} s ==="
        )
        print(
            f"{'config':<15}{'SNPs':>10}{'N* equal':>12}{'rows identical':>16}{'pairs +ours':>13}{'pairs +upstr':>14}{'max|dLD|':>11}"
        )
        for cfg in CONFIGS:
            s = stats[mode][cfg_name(cfg)]
            print(
                f"{cfg_name(cfg):<15}{s['n_snps']:>10,}{s['nstar_equal']:>12,}{s['rows_identical']:>16,}"
                f"{s['pairs_only_ours']:>13,}{s['pairs_only_upstream']:>14,}{s['max_abs_dld']:>11.2e}"
            )
    print(f"\npairwise r2 (w1000kb_r0.1, round_r2=6): {pair}")
    if scale_table:
        print(
            f"\nnull statistics, {scale_table['n_replicates']} replicates over {scale_table['n_snps']:,} SNPs: "
            "mean chi-square of a replicate"
        )
        print(f"{'statistic':<14}{'mean':>9}{'SD':>9}{'min':>9}{'max':>9}{'mean z^4':>10}")
        for kind in NULL_KINDS:
            t = scale_table[kind]
            print(
                f"{kind:<14}{t['mean_of_replicate_mean_z2']:>9.4f}{t['sd_of_replicate_mean_z2']:>9.4f}"
                f"{t['min']:>9.4f}{t['max']:>9.4f}{t['mean_z4']:>10.4f}"
            )
        print(f"(theory: var of t with n-2 d.f. = {scale_table['theory_t_variance']:.4f})")
    if null_report:
        print("null_z local LD structure:", json.dumps(null_report, indent=1))
    print(f"\nread genotypes: {read_seconds:.1f} s; peak RSS: {report['peak_rss_gb']:.2f} GB")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
