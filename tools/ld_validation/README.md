# Validation of the LD-reference builder against the original GENESIS tables

The original R package ships, for 1,070,777 HapMap3 SNPs, the number of tagged SNPs `N*`,
the bias-corrected LD score and the tagging lists for nine (window, r² threshold)
configurations, plus 100 LD-correlated null replicates — but not the code that produced
them. `genesis_gwas.ld` (`compute_ld`, `null_z`, `build_reference`) re-implements that
computation so that references can be built for other populations. The scripts here
establish, from the genotypes the original tables were derived from, exactly which
definitions the original used, and measure how well the re-implementation reproduces them.

| Script | What it does |
|---|---|
| `validate_against_upstream.py` | Recomputes all nine tables from the genotypes (once with `round_r2=6`, once with the default exact r²) and compares `N*`, tagging sets, LD scores and pairwise r² with the converted upstream reference; compares the null statistics with upstream's replicates. |
| `check_snp_selection.py` | Checks that "HapMap3 SNPs with MAF ≥ 0.05 in the panel" reproduces the reference SNP set. |
| `results_eur_1kg_hm3.json` | Full machine-readable output of the run reported below. |

## Inputs

* **Genotypes**: the LDSC `1000G_Phase3_plinkfiles.tgz` (Zenodo record 10515792; md5
  `a7773ab485827b533cb300c76356d76b`), i.e. `1000G.EUR.QC.{1..22}.{bed,bim,fam}`, 489
  European samples, 9,997,231 SNPs.
* **Ground truth**: the converted upstream reference bundle `eur_1kg_hm3` and, for the
  bit-level comparison of pairwise r², the raw exports — both produced by `tools/refdata`
  from upstream commit `e4e6894`.

```bash
PLINK="/path/1000G_EUR_Phase3_plink/1000G.EUR.QC.{chrom}"
python validate_against_upstream.py --plink "$PLINK" --bundle /path/eur_1kg_hm3 \
       --raw /path/raw --nullz --out results.json            # ~4 min, ~11 GB (it holds all 9 upstream tables)
python check_snp_selection.py --plink "$PLINK" --bundle /path/eur_1kg_hm3 \
       --hm3-list /path/raw/hm3/w_hm3.noMHC.snplist.tsv
# quick chromosome-22 version of the same check, as a unit test against the packaged mini bundle:
GENESIS_1000G_PLINK=/path/1000G_EUR_Phase3_plink pytest tests/ld/test_upstream_chr22.py
```

## What the original computed (established here)

* **Reference SNPs.** Of the 1,215,001 SNPs of `w_hm3.noMHC.snplist`, 1,188,097 are in the
  genotype files; keeping those with MAF ≥ 0.05 in the 489 samples selects 1,068,898 SNPs —
  *exactly* the reference SNPs carrying the HapMap3-no-MHC flag (0 extra, 0 missing; smallest
  MAF kept 49/978, largest dropped 48/978). The other 1,879 reference SNPs are HapMap3 SNPs
  in the MHC region, which that list omits. → `build_reference(min_maf=0.05)` with `>=`.
* **r.** Pearson correlation of allele dosages across the 489 samples (the genotypes have no
  missing values).
* **Tagging set.** Same chromosome, `|pos_m − pos_k| ≤ window` (**inclusive**), `r² ≥ r2_min`,
  the SNP itself included.
* **LD score.** `Σ_{m ∈ tagging set} [ r² − (1 − r²)/(n − 2) ]`, `n = 489`.
* **Rounding.** The tables were derived from a *printed* PLINK `--r2` table: every r² had
  first been rounded to **six significant digits** (`%g`: `0.123456` but `0.0512346`), and
  both the thresholds and the LD-score sums used the rounded values. Six *decimals* do not
  reproduce the r² ≥ 0.05 tables (LD scores off by up to 1.6e-5 on every SNP), and applying
  the thresholds to exact r² loses the pairs listed below. → `round_r2=6`.
* **Null replicates.** They are the regression *t* statistics of null GWAS in the 489
  samples (studentised), not raw projections `X̃ᵀy/√n`; see the last table. → `null_z`
  default `statistic="t"`.

## Agreement, all 22 chromosomes

`compute_ld(..., configs=<all nine>, round_r2=6)` — **the setting that reproduces upstream**:

| configuration | SNPs | `N*` equal | tagging rows identical | ordered pairs (ours = upstream) | pairs only ours / only upstream | max \|Δ LD score\| | SNPs with \|Δ\| > 1e-9 |
|---|---|---|---|---|---|---|---|
| w500kb_r0.05 | 1,070,777 | 1,070,777 | 1,070,777 | 72,302,185 | 0 / 0 | 9.1e-13 | 0 |
| w500kb_r0.1 | 1,070,777 | 1,070,777 | 1,070,777 | 48,465,097 | 0 / 0 | 9.1e-13 | 0 |
| w500kb_r0.2 | 1,070,777 | 1,070,777 | 1,070,777 | 30,046,219 | 0 / 0 | 8.0e-13 | 0 |
| w1000kb_r0.05 | 1,070,777 | 1,070,777 | 1,070,777 | 75,478,865 | 0 / 0 | 1.0e-06 | 2 |
| w1000kb_r0.1 | 1,070,777 | 1,070,777 | 1,070,777 | 49,789,689 | 0 / 0 | 1.0e-06 | 2 |
| w1000kb_r0.2 | 1,070,777 | 1,070,777 | 1,070,777 | 30,562,023 | 0 / 0 | 1.2e-12 | 0 |
| w2000kb_r0.05 | 1,070,777 | 1,070,777 | 1,070,777 | 76,512,765 | 0 / 0 | 1.0e-06 | 2 |
| w2000kb_r0.1 | 1,070,777 | 1,070,777 | 1,070,777 | 50,192,215 | 0 / 0 | 1.0e-06 | 2 |
| w2000kb_r0.2 | 1,070,777 | 1,070,777 | 1,070,777 | 30,686,323 | 0 / 0 | 1.2e-12 | 0 |

Every `N*` and every tagging list is reproduced exactly in all nine configurations, and so is
every LD score to rounding error — with **one residual**: the SNP pair rs2177254–rs11779181
(chr8, 547,822 bp apart) has r² = 9747/80000 = 0.1218375 *exactly*, a decimal tie at the
sixth significant digit. PLINK printed `0.121837`; our floating-point value lands a hair on
the other side and rounds to `0.121838`. That moves the LD scores of those two SNPs by
1.0e-6 in the four configurations that contain the pair (window ≥ 1 Mb, threshold ≤ 0.1).

Pairwise r² for w1000kb_r0.1 against the original doubles (49,789,689 ordered pairs):
49,776,837 are bit-identical, a further 12,850 differ by one unit in the last place (decimal
parsing on the original platform), and 2 — the tie pair above — by 1e-6. Against the
`uint16`-quantised copy in the bundle the largest difference is 7.7e-6, half a quantisation
step. A bundle written by `build_reference` for the nine configurations and read back with
`load_reference` has SNP table, `nstar`, `indptr`, `indices` and the stored pairwise r²
identical to the converted upstream bundle.

`compute_ld(..., round_r2=None)` — **the default, recommended for new panels**:

| configuration | SNPs with different `N*` | ordered pairs missing relative to upstream | their exact r² |
|---|---|---|---|
| w500kb / w1000kb / w2000kb, r² ≥ 0.05 | 37 / 41 / 41 | 46 / 50 / 50 | 0.04999995 … 0.05 |
| w500kb / w1000kb / w2000kb, r² ≥ 0.1 | 10 / 10 / 10 | 12 / 12 / 12 | 0.09999995 … 0.1 |
| w500kb / w1000kb / w2000kb, r² ≥ 0.2 | 48 / 50 / 50 | 60 / 62 / 62 | 0.1999995 … 0.2 |

Without rounding, the only pairs that differ are those whose exact r² is below the threshold
but *prints* as the threshold; no pair is added. For w1000kb_r0.1, 10 of 1,070,777 SNPs have
a different `N*` (nine by one, one by three, with LD scores lower by 0.10–0.29); for all
other SNPs the LD score differs by at most 6.0e-5 (median 1.3e-6, at most 2.2e-6 relative),
which is the accumulated rounding of the individual r² values. The default therefore gives,
for all practical purposes, the same reference without inheriting an artefact of a text
table.

## Null statistics

Mean χ² of a replicate over the 1,070,777 SNPs, 100 replicates each:

| statistic | mean | SD across replicates | min | max | mean z⁴ |
|---|---|---|---|---|---|
| upstream `error_iter1..100` | 1.0042 | 0.0111 | 0.987 | 1.066 | 3.038 |
| `null_z(statistic="t")` (default) | 1.0031 | 0.0099 | 0.984 | 1.035 | 3.030 |
| `null_z(statistic="normal")` | 0.9990 | 0.0098 | 0.980 | 1.031 | 2.993 |
| `null_z(statistic="projection")` | 0.9920 | 0.0614 | 0.861 | 1.150 | 2.950 |

(The variance of a *t* variable with 487 degrees of freedom is 1.0041.) Upstream's replicates
have the mean, the between-replicate spread and the fourth moment of studentised *t*
statistics. The raw projection `X̃ᵀy/√n`, although exactly N(0, R̂) over replicates, carries the
sample variance of each null phenotype as a common factor, so the genome-wide inflation of a
replicate fluctuates six times more; used in a QQ-plot simulation that would widen the
envelope for a reason unrelated to the GWAS being simulated.

Local LD structure on chromosome 22 (400,000 random SNP pairs at most 50 SNPs apart; the
replicate-average of `u_i·u_j` regressed on the signed genotype correlation `r_ij`): slope
1.005 for upstream and 0.975 for ours with 100 replicates each, and 1.001 (correlation
0.9989) for ours with 4,000 replicates; the sign of the covariance agrees with the sign of
`r` for every pair with |r| > 0.5 (16% of which have negative `r`) in both. Regressing on
|r| instead gives slopes of only 0.53–0.55, i.e. both sets of replicates carry the *signed* LD of this
panel.

## Cost

Apple M5 Pro laptop (18 cores, NumPy with Accelerate BLAS), all 22 chromosomes, 1,070,777
SNPs × 489 samples:

| step | wall time | peak memory |
|---|---|---|
| `compute_ld`, nine configurations in one pass (genotypes already in memory) | 42 s | — |
| reading the genotypes of the reference SNPs from the 22 `.bed` files | 7 s | — |
| `build_reference`, default configuration (1 Mb, r² ≥ 0.1, pairwise r² stored) | 45 s | 2.3 GB |
| the same plus 100 null replicates | 49 s | 3.1 GB |
| `build_reference`, nine configurations (`round_r2=6`) | 100 s | 4.9 GB |

`build_reference` times include parsing the 10-million-row `.bim` files and writing the
compressed bundle (111 MB, 315 MB and 266 MB respectively). The PyTorch code path
(`backend="torch"`) was run on the CPU and returns identical tagging sets and LD scores; it
could not be run on a CUDA device on this machine.

## Caveats

* The validation covers one panel (1000 Genomes EUR, no missing genotypes). Mean-imputation
  of missing genotypes, duplicate handling and the MAF filter are covered by unit tests on
  synthetic data only.
* PLINK's own number formatting was not re-implemented; `round_r2` rounds the exact double
  correctly to six significant digits. The single tie pair above is the only place where
  that differs from the printed table.
* The bundle format stores r² (as upstream does), not signed r, for the pairwise file.
