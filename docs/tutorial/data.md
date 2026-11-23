# Preparing the data

GENESIS needs three numbers per SNP — an identifier, a $z$ statistic and a sample size —
and an LD reference for the ancestry of the study. This chapter shows how the software
derives the three numbers from the files consortia actually distribute, which quality
filters it applies and why, and how the LD reference is obtained and matched.

Everything here happens automatically inside `genesis fit`. It is worth reading anyway:
almost every wrong GENESIS result traces back to this step, most often to the sample
size.

## The example file

The height GWAS of
[Lango Allen et al. (2010)](../reference/bibliography.md#ref-langoallen2010) is
distributed with the reference data in the simplest possible form:

```{literalinclude} ../_scripts/output/data_head.txt
:language: console
```

## Reading summary statistics

{func}`genesis_gwas.read_sumstats` — called for you by `gg.prepare` and `genesis fit` —
reads delimited text (tab, comma or whitespace; plain, `.gz`, `.bz2` or `.xz`, and
`.zst` if the `zstandard` package is installed)
and recognises common column names without regard to case.

| Quantity | Recognised column names (examples) | CLI option | Python argument |
|---|---|---|---|
| SNP identifier | `SNP`, `rsid`, `MarkerName`, `variant_id`, `ID` | `--snp` | `snp=` |
| $z$ statistic | `Z`, `zscore`, `tstat`, `stat` | `--z` | `z=` |
| effect | `BETA`, `b`, `effect`, `log_or`, `estimate` | `--beta` | `beta=` |
| standard error | `SE`, `stderr`, `standard_error` | `--se` | `se=` |
| odds ratio | `OR`, `odds_ratio` | `--or` | `odds_ratio=` |
| P-value | `P`, `pval`, `p_value` | `--p` | `p=` |
| sample size | `N`, `Neff`, `n_total`, `samplesize`, `weight` | `--n` | `n=` |
| cases, controls | `N_CASES`, `ncase`; `N_CONTROLS`, `ncontrol` | `--n-cases`, `--n-controls` | `n_cases=`, `n_controls=` |
| chromosome, position | `CHR`, `chrom`; `POS`, `BP`, `base_pair_location` | `--chr`, `--pos` | `chrom=`, `pos=` |
| allele frequency, imputation quality | `EAF`, `FRQ`, `MAF`; `INFO` | `--eaf`, `--info` | `eaf=`, `info=` |

The complete list is {data}`genesis_gwas.data.sumstats.COLUMN_ALIASES`. Any column can be
named explicitly, and `--n`, `--n-cases` and `--n-controls` also accept a number when the
sample size is the same for all SNPs.

**The $z$ statistic** is taken from the first of these that is available:

1. a $z$ column;
2. effect / standard error;
3. log(odds ratio) / standard error;
4. a P-value together with a signed effect: $z = \operatorname{sign}(\text{effect})\,
   |\Phi^{-1}(P/2)|$.

Only $z^2$ enters the likelihood, so the effect allele does not have to be aligned with
the reference, and strand does not matter.

**The sample size** is taken, in this order, from the options you pass, from case and
control counts (as the effective sample size, see below), or from the `N` column.

The script below writes the same three SNPs in five common layouts and shows what the
reader makes of each.

```{literalinclude} ../_scripts/examples/read_formats.py
:language: python
```

```{literalinclude} ../_scripts/output/read_formats.txt
:language: console
```

The line "Z from … ; N from …" is also the first thing `genesis fit` logs after reading
your file. **Read it every time.**

(tut-effective-n)=
## The sample size of a case–control study

GENESIS standardises effects as $\hat\beta_k = z_k/\sqrt{n_k}$. For a quantitative trait
$n_k$ is the number of individuals. For a case–control study it must be the **effective
sample size**

$$
n_\text{eff} = \frac{n_\text{case}\; n_\text{control}}{n_\text{case} + n_\text{control}} ,
$$

because the variance of a log odds ratio per standard deviation of genotype is
$1/n_\text{case} + 1/n_\text{control} = 1/n_\text{eff}$. With this choice $\hat\beta_k$ is
a log odds ratio per standard deviation of genotype, $\sigma^2$ is the variance of such
log odds ratios, and heritability is on the log-odds scale
([Chapter 6](binary-traits.md)).

:::{admonition} The factor-of-four trap
:class: warning

Many meta-analysis tools report a different "effective N":
$4/(1/n_\text{case} + 1/n_\text{control})$ — METAL's `Neff` and the PGC's convention — which
is **four times** the quantity above. Used as is, it makes every standardised effect half
as large as it should be: $\pi_c$ is unaffected, but $\sigma^2$, $a$ and the heritability
come out **four times too small**.

- If the file has case and control counts, use them: they are detected automatically, or
  pass `--n-cases` and `--n-controls` (column names or numbers).
- If it has a METAL-style effective N, pass `--n-scale 0.25` (fourth example above).
- If it has the *total* sample size and you know the fraction of cases $\phi$, pass
  `--case-fraction` $\phi$: the software uses $N\phi(1-\phi)$.
- A column named `Neff` (or `n_eff`, `n_effective`) is read as the sample size *as it
  stands*, with a warning, because the software cannot know which convention it
  follows. Say which it is with `--n-scale`: `0.25` for the METAL convention, `1` if the
  column already is $n_\text{case}n_\text{control}/(n_\text{case}+n_\text{control})$.
:::

When per-SNP sample sizes are not reported at all, `--n-from-se` (Python:
`n_from_se=True`) derives them from the standard error and the allele frequency,
$n_k = 1/\{2f_k(1-f_k)\,\mathrm{SE}_k^2\}$ — the last example above. The same identity is
a quick test of which convention a file follows; {ref}`tut-check-n` applies it.

For a quantitative trait analysed on its original scale the identity involves the trait
variance, so use it only for log odds ratios or for effects in standard-deviation units.

(tut-qc)=
## Quality control

{func}`genesis_gwas.prepare` filters the summary statistics and merges them with the
reference. The filters are those of
[Zhang et al. (2018)](../reference/bibliography.md#ref-zhang2018), who followed LD Hub
([Zheng et al., 2017](../reference/bibliography.md#ref-zheng2017)):

| Step | Default | Why | Change it with |
|---|---|---|---|
| drop rows with missing or non-finite $z$ or $N$, or $N \le 0$ | always | unusable | — |
| drop duplicated identifiers (first kept) | always | ambiguous | — |
| drop SNPs with $N$ below 0.67 × the 90th percentile of $N$ | on | SNPs covered by a fraction of the studies have different effective LD and noise | `--n-min-fraction`, `QCConfig(n_min_fraction=…)` |
| drop SNPs below an imputation-quality or allele-frequency threshold | off | optional | `--min-info`, `--min-maf` |
| keep SNPs that are in the LD reference | always | $N^*_k$ and $\ell_k$ are needed | — |
| keep the HapMap3 SNPs outside the MHC (chr6: 26–34 Mb) | on | LD in the MHC is too long-range for the model | `--keep-mhc`, `QCConfig(hm3_no_mhc=False, exclude_mhc=False)` |
| drop SNPs with $z^2 > 80$ | on | single very large effects are outliers to a normal mixture and dominate the fit | `--max-chisq`, `QCConfig(max_chisq=…)` |

`--no-qc` (Python: `qc=False`) keeps only the steps marked "always", which is what the
original software does by default (`filter=FALSE`).

```{literalinclude} ../_scripts/examples/prepare_height.py
:language: python
```

```{literalinclude} ../_scripts/output/prepare_height.txt
:language: console
```

The distributed height file had already been filtered, so only one step removes anything:
95,488 of its HapMap3 SNPs are not among the 1,070,777 of the reference, which keeps SNPs
with a minor-allele frequency of at least 5 % in the 1000 Genomes European sample. That
leaves $K = 941{,}684$ markers.

To see the filters at work, the next script damages a copy of the packaged
chromosome-22 subset of the same GWAS in known ways:

```{literalinclude} ../_scripts/examples/prepare_messy.py
:language: python
```

```{literalinclude} ../_scripts/output/prepare_messy.txt
:language: console
```

Each row accounts for exactly the damage done. The same table is written to
`PREFIX.summary.txt` and `PREFIX.log` by `genesis fit`, and is available as
`data.meta["qc"]` in Python.

:::{admonition} Two filters that deserve a decision
:class: tip

**Sample-size filter.** It assumes that most SNPs were measured in most of the sample.
If your file mixes genotyping arrays with very different coverage, look at the
distribution of $N$ before trusting the default.

**The $z^2 > 80$ filter.** For a well-powered study of a trait with a few strong loci —
most cancers, for example — this removes the top loci *and the markers in LD with them*.
The fitted distribution then describes the rest of the genome, and the heritability does
not include what was removed. {ref}`tut-chisq-filter` shows the size of the effect in a
simulation. Whatever you choose, report the number of markers removed.
:::

(tut-reference)=
## The LD reference

A reference is a *bundle* of files for one population:

- the list of SNPs with chromosome and position (GRCh37, and GRCh38 where the SNP could
  be lifted over);
- for each SNP, $N^*_k$, $\ell_k$ and the tagging neighbourhood, for each LD definition;
- optionally, pairwise $r^2$ (for simulating data with realistic LD) and LD-correlated
  null $z$ scores (for the envelopes of QQ plots).

The European reference `eur_1kg_hm3` is a conversion of the data of the original package:
1,070,777 HapMap3 SNPs with minor-allele frequency of at least 5 % in 489 European
individuals of the 1000 Genomes Project
([Auton et al., 2015](../reference/bibliography.md#ref-auton2015)), with nine LD
definitions — windows of 0.5, 1 and 2 Mb combined with $r^2$ thresholds of 0.05, 0.1 and
0.2. The default, and the definition used in the paper, is 1 Mb and 0.1.

References for East Asian (`eas_1kg_hm3`) and African ancestry (`afr_1kg_hm3`) have the
same structure and are used in the same way — name them with `--reference EAS` or
`--reference AFR`, or `genesis data fetch AFR`. [Chapter 7](other-ancestries.md)
describes them and shows how to build a reference for any other population. This
chapter continues with the European one.

### Getting it

Nothing needs to be done on a computer with internet access: the files an analysis needs
are downloaded when first used, verified against their checksums, and cached. To fetch
them ahead of time — on a cluster, do this once before submitting jobs — use
`genesis data fetch`:

```{literalinclude} ../_scripts/output/data_fetch.txt
:language: console
```

(`<download URL>` stands for the location of the data release. `--dest` is only used here
to show a fresh download; without it the files go to the data directory, which is
`$GENESIS_DATA_DIR` if set.) The default set is the SNP list and the default LD
definition, 32 MB. Further components are fetched by name:

```{literalinclude} ../_scripts/output/data_fetch_more.txt
:language: console
```

| `--components` | Content | Size | Needed for |
|---|---|---|---|
| `default` | SNP list, LD definition 1 Mb / 0.1 | 32 MB | fitting, projections, prediction, posterior |
| `ld` | all nine LD definitions | 196 MB | sensitivity analyses |
| `nullz` | 100 LD-correlated null $z$ replicates | 214 MB | QQ plots (`--qq`), `future_gc`, `simulate_ld_sumstats` |
| `pairld` | pairwise $r^2$ for the default definition | 70 MB | `simulate_ld_sumstats` |
| `examples` | the height summary statistics | 13 MB | this tutorial |
| `all` | everything | 504 MB | |

The sizes are those of the European reference; the other two are slightly smaller
([Installation](../getting-started/installation.md) has the table). `genesis data list`
shows what is installed, here with `--summary`, which leaves out the list of files:

```{literalinclude} ../_scripts/output/data_list.txt
:language: console
```

[Installation](../getting-started/installation.md) describes the data directory, offline
use and mirrors.

### Using it

```python
import genesis_gwas as gg

ref = gg.load_reference()                                   # eur_1kg_hm3, 1 Mb, r2 >= 0.1
ref = gg.load_reference(window_kb=2000, r2_min=0.05)        # another LD definition
ref = gg.load_reference("mini_chr22")                       # packaged chromosome 22, no download
ref = gg.load_reference("path/to/my_bundle")                # a reference you built
```

On the command line the same choices are `--reference`, `--ld-window-kb` and `--ld-r2`.
[Chapter 7](other-ancestries.md) builds a reference for another population.

## Matching SNPs to the reference

Markers are matched to the reference by identifier (rsID). If fewer than half of them
match and the file has chromosome and position columns, matching falls back to positions,
trying GRCh37 and GRCh38 and keeping the build with more matches. `--match-on position`
and `--build GRCh38` force the choice.

The example below removes the rsIDs from the chromosome-22 data and identifies variants by
GRCh38 coordinates instead, as many recent releases do:

```{literalinclude} ../_scripts/examples/match_position.py
:language: python
```

```{literalinclude} ../_scripts/output/match_position.txt
:language: console
```

All but 30 markers are recovered; those 30 have no GRCh38 position in the reference.

:::{admonition} Limits of position matching
:class: caution

The reference does not store alleles, so a position match cannot tell a SNP from another
variant at the same position. Restrict the file to bi-allelic SNPs before matching by
position. If several rows map to the same reference SNP, the first is kept and the
others are counted in the QC table.
:::

## Checklist

- [ ] The log line "Z from … ; N from …" says what you intended.
- [ ] For a binary trait, $N$ is $n_\text{case}n_\text{control}/(n_\text{case}+n_\text{control})$,
      not four times that and not the total.
- [ ] The number of markers matched to the reference is a large fraction of the
      reference (for imputed data, typically more than 85 % of its SNPs).
- [ ] The QC table shows no unexpectedly large removal, and you know how many markers
      the $z^2 > 80$ filter took out.
- [ ] The reference population matches the study population.
