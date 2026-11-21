# Reference-data conversion tools

These scripts convert the reference data that upstream GENESIS ships as `.RData` files
(<https://github.com/yandorazhang/GENESIS>, commit
`e4e6894d2e11ece6715d0572b0febceff1480f85`, about 1.6 GB inside the R package) into a compact,
checksummed **reference bundle**. They are provenance tooling: users of the package never
need to run them, because the finished bundle is distributed as release assets.

| Script | Language | What it does |
|---|---|---|
| `export_rdata.R` | base R | Dumps every upstream object to little-endian binaries / text (`raw/`). |
| `spotcheck_export.R` | base R | Writes all upstream fields *by rsID* for ~55 SNPs; an independent code path used only for verification. |
| `build_bundle.py` | Python + numpy | Assembles the full bundle `eur_1kg_hm3` from `raw/`, including GRCh38 positions. |
| `make_mini_bundle.py` | Python + numpy | Cuts a one-chromosome test bundle (`mini_chr22`) out of the full one. |
| `verify_bundle.py` | Python + numpy | Checks a bundle: internal invariants, against `raw/`, against the spot check, and against a parent bundle. |
| `check_liftover_ensembl.py` | Python + numpy, network | Compares `pos` / `pos_grch38` with Ensembl's rsID coordinates. |
| `ensembl_grch38_lookup.py` | Python + numpy, network | Optional side table of Ensembl GRCh38 positions for SNPs the chain could not lift. Does not change the bundle. |
| `summarize_1000g.py` | Python + numpy | SNP/sample counts of the LDSC 1000G EUR PLINK files and their overlap with the bundle. |
| `refdata_common.py` | Python + numpy | Shared helpers: raw readers, deterministic `.npz` writer, CSR builder, chain liftover. |

Every script takes its paths as arguments and can be run with `python -I` (isolated mode);
the scripts add only their own directory to `sys.path`.

## Reproducing the bundle

```bash
UP=/path/to/upstream/GENESIS/data            # clone of yandorazhang/GENESIS at e4e6894
W=/path/to/workdir
uv venv $W/venv --python 3.12 && uv pip install --python $W/venv/bin/python numpy
PY="$W/venv/bin/python -I -B"

Rscript --vanilla export_rdata.R     $UP $W/raw                      # ~2 min, 3.5 GB
Rscript --vanilla spotcheck_export.R $UP $W/raw_spotcheck.tsv        # ~1 min
curl -LO https://hgdownload.soe.ucsc.edu/goldenPath/hg19/liftOver/hg19ToHg38.over.chain.gz
                                                # md5 35887f73fe5e2231656504d1f6430900
$PY build_bundle.py     --raw $W/raw --chain hg19ToHg38.over.chain.gz --out $W/bundle/eur_1kg_hm3   # ~1 min, ~13 GB peak RAM
$PY make_mini_bundle.py --bundle $W/bundle/eur_1kg_hm3 --out $W/bundle/mini_chr22
$PY verify_bundle.py    --bundle $W/bundle/eur_1kg_hm3 --raw $W/raw --spotcheck $W/raw_spotcheck.tsv
$PY verify_bundle.py    --bundle $W/bundle/mini_chr22  --parent $W/bundle/eur_1kg_hm3
$PY check_liftover_ensembl.py --bundle $W/bundle/eur_1kg_hm3 --chain hg19ToHg38.over.chain.gz   # optional
```

`verify_bundle.py` prints one `PASS` / `FAIL` / `INFO` line per check and exits non-zero on
any failure. Run without `--raw` / `--spotcheck` it tests only bundle-internal invariants,
so it also works on bundles built for other reference panels.

### Byte reproducibility

`np.savez_compressed` stamps each zip member with the current time, so its output is not
byte-reproducible. `refdata_common.save_npz` writes the same container (a deflated zip of
`<key>.npy` members, no pickles, readable with `np.load(..., allow_pickle=False)`) with
constant timestamps and permissions, and the example table is gzipped with `mtime=0` and no
embedded file name. Rebuilding with the same Python, numpy and zlib reproduces every
`sha256` in `manifest.json` (checked: CPython 3.12.13, numpy 2.5.5, zlib 1.2.12, macOS
arm64). A different zlib implementation or compression level may produce different, equally
valid compressed bytes; the *array contents* are identical regardless, and
`verify_bundle.py --raw` proves that.

## Bundle format (`format_version` 1)

All arrays are aligned to the **canonical SNP order**: sorted by `(chrom, pos, rsid)`
ascending. `M` is the number of reference SNPs (1,070,777 for `eur_1kg_hm3`).

```
<bundle>/
  manifest.json
  snps.npz
  ld/w{window_kb}kb_r{cutoff}.npz          one per LD configuration
  pairld/w1000kb_r0.1.npz                  pairwise r^2, only where available
  nullz/nullz_{start:03d}-{stop:03d}.npy   LD-correlated null z-score replicates
  examples/*.sumstats.tsv.gz               example GWAS summary statistics
```

**`snps.npz`**

| key | dtype | meaning |
|---|---|---|
| `rsid` | `S<maxlen>` (ASCII bytes) | SNP identifier |
| `chrom` | `uint8` | chromosome (1–22) |
| `pos` | `int32` | GRCh37/hg19 position, 1-based (upstream `BP`) |
| `pos_grch38` | `int32` | GRCh38 position, 1-based, from UCSC liftOver of `(chrom, pos)`; `-1` if the base is not covered by the chain or lifts to another sequence |
| `hm3_no_mhc` | `bool` | rsID is in upstream `w_hm3.noMHC.snplist` (the list used by the upstream `filter=TRUE` step) |

**`ld/<cfg>.npz`** with `cfg = w{window_kb}kb_r{cutoff}`, e.g. `w1000kb_r0.1`, `w500kb_r0.05`

| key | dtype / shape | meaning |
|---|---|---|
| `nstar` | `int32 (M,)` | N*_k, number of reference SNPs tagged by SNP k (itself included) |
| `ldscore` | `float64 (M,)` | bias-corrected LD score, a bit-exact copy of upstream `LD.score.correct` |
| `indptr` | `int64 (M+1,)` | CSR row pointer |
| `indices` | `int32 (nnz,)` | CSR column indices = canonical rows of the tagged SNPs (upstream `TaggingSNPs`), ascending within a row, self included |
| `window_kb`, `r2_min` | `float64` scalars | configuration |
| `n_ref` | `int64` scalar | reference sample size (489) |

**`pairld/w1000kb_r0.1.npz`**: `r2_u16`, `uint16 (nnz,)`, aligned element for element with
`indices` of the matching `ld/` file; `r2 = r2_u16 / 65535`. Upstream ships pairwise r²
(`pairwiseLD`) only for the 1 Mb / r² ≥ 0.1 configuration.

**`nullz/*.npy`**: `float16`, shape `(10, M)`, C order. Row `i` of the file starting at
replicate `s` is upstream `error_iter{s+i+1}` in canonical SNP order. Memory-mappable.

**`examples/height_giant2010.sumstats.tsv.gz`**: columns `SNP`, `Z`, `N` — upstream
`heightGWAS` (GIANT height, Lango Allen et al. 2010), 1,037,172 rows in upstream order, with
the exact text of the upstream values (they are stored as factor labels upstream).

**`manifest.json`**: `format_version`, `name`, `version`, `population`, `panel`,
`genome_build`, `n_ref`, `n_snps`, `source` (`repo`, `commit`), `ld_configs` (list of
`id`, `window_kb`, `r2_min`, `file`, `pairld`), `nullz` (`dtype`, `n_replicates`, `files`),
and `files` (relative path → `sha256`, `bytes`).

## What the verification established

Numbers are from the run that produced the bundle (`verify_bundle.py`, all checks passed).

* All nine upstream tables hold the same 1,070,777 SNPs in the same row order; rsIDs are
  unique and no two SNPs share a `(chrom, pos)`.
* `nstar` equals the row length in every configuration; `ldscore` is bit-identical to
  upstream; the neighbour lists equal upstream `TaggingSNPs` as sets per row, and for
  1 Mb / 0.1 they also equal `TaggingSNPinx` (same element order upstream).
* Every row contains its own SNP exactly once. The neighbour relation is **symmetric** in all
  nine configurations (0 asymmetric pairs), and the stored r² satisfies r²(i,j) = r²(j,i).
* All neighbours are on the same chromosome. The **window is inclusive**: the 500 kb tables
  contain 18 / 12 / 4 ordered pairs at exactly 500,000 bp and the 1 Mb / 0.05 table 4 pairs at
  exactly 1,000,000 bp; no pair exceeds the window.
* `ldscore_k = Σ_{m ∈ row k} [ r²_km − (1 − r²_km) / (n_ref − 2) ]` with `n_ref = 489`
  reproduces upstream to 5.3e-13 (the unadjusted sum differs by up to 1.09). The sum runs over
  the tagged SNPs only (r² ≥ cutoff, within the window, self included).
* Upstream r² values all have at most six decimals (they were evidently read from a text
  table), the minimum is exactly `0.1` (176 pairs) and none is below it. At the stored
  precision the **cutoff is therefore r² ≥ 0.1**; whether a pair whose unrounded r² is a hair
  under 0.1 would have been kept cannot be decided from these files.
* Quantising r² to `uint16` changes it by at most 7.6e-6 (and √r² by at most 1.2e-5); LD
  scores recomputed from the quantised values differ from upstream by at most 7.3e-4
  (3.6e-5 relative). `ldscore` itself is stored exactly, so this affects only code that
  uses the pairwise values.
* `float16` storage of the null z-scores changes values by at most 1.95e-3; replicate means,
  SDs and median-χ² ratios agree with the double-precision originals to about 5e-4.
* The independent by-rsID spot check (55 SNPs × 9 configurations, pairwise r², six null
  replicates, HapMap3 flags, height rows) found no mismatch.
* GRCh38 positions: 1,070,185 SNPs lifted; 592 are `-1` (462 in chain gaps, 130 lift to
  alt/unplaced contigs, none to another primary chromosome). Against Ensembl's rsID
  coordinates: 262 of 262 sampled lifted SNPs agree on GRCh38 (including 100 of 100 lifted
  through reverse-strand chain blocks) and 311 of 311 agree on GRCh37. Ensembl has a unique
  same-chromosome GRCh38 location for 570 of the 592 unlifted SNPs
  (`ensembl_grch38_lookup.py`), should a caller want to fill them in.

## External resources

| Resource | Source | Checksum |
|---|---|---|
| `hg19ToHg38.over.chain.gz` | <https://hgdownload.soe.ucsc.edu/goldenPath/hg19/liftOver/> | md5 `35887f73fe5e2231656504d1f6430900` (matches UCSC `md5sum.txt`) |
| `1000G_Phase3_plinkfiles.tgz` (LDSC 1000G EUR, 489 samples, 9,997,231 SNPs) | Zenodo record 10515792, "S-LDSC reference files" (Gazal), DOI `10.5281/zenodo.10515792` | md5 `a7773ab485827b533cb300c76356d76b` (matches Zenodo) |

The PLINK files are not part of the bundle; they are the genotypes the upstream LD tables
were computed from, and all 1,070,777 reference SNPs are present in them by rsID and by
GRCh37 position.
