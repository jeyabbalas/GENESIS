# Frequently asked questions

## Input data

### What does the summary-statistics file need to contain?

Three things per SNP: an identifier, a Z statistic and a sample size. The file is
delimited text (tab, comma or white space; plain or compressed with gzip, bzip2 or xz)
with a header line. Common column names are recognised without regard to case, and the Z statistic
is derived from whatever is available:

| Needed | Taken from (first that applies) |
|---|---|
| SNP | `SNP`, `rsid`, `MarkerName`, `variant_id`, ... — or chromosome and position (see below) |
| Z | `Z`; or `BETA`/`SE`; or log(`OR`)/`SE`; or a signed effect (`BETA` or `OR`) with a P-value |
| N | `N` (`Neff`, `sample_size`, ...); or `--n NUMBER`; or case and control counts; or `--n-from-se` |

Name a column explicitly with `--snp`, `--z`, `--beta`, `--se`, `--or`, `--p`, `--n`,
`--chr`, `--pos` when detection does not find it. The log file states which columns
were used ("Z from ...; N from ..."), and the result file records it under
`provenance.data.sumstats`.

Allele columns are not needed: the model is symmetric in the sign of the effect, so
neither the choice of effect allele nor the strand matters.

The statistics should **not** be corrected by genomic control, and should come from a
single-ancestry GWAS matching the LD reference.

### What sample size do I give for a case–control GWAS?

The effective sample size

$$n_\text{eff} = \frac{n_\text{cases}\, n_\text{controls}}{n_\text{cases} + n_\text{controls}},$$

which is the convention of the original software. With it, the standardised effect
$z/\sqrt{n}$ is a log odds ratio per standard deviation of genotype and heritability is
on the log-odds-ratio scale.

| Your file has ... | Use |
|---|---|
| case and control counts (columns or study-wide numbers) | `--n-cases`, `--n-controls` (columns named `n_cases`, `n_controls`, ... are detected) |
| the total sample size and you know the fraction of cases $\phi$ | `--case-fraction PHI` ($n_\text{eff} = N\phi(1-\phi)$) |
| an "effective N" as written by METAL or the PGC, $4/(1/n_\text{cases}+1/n_\text{controls})$ | `--n-scale 0.25` — that quantity is **four times** the one needed here |
| no sample size per SNP, but standard errors and allele frequencies | `--n-from-se` ($n_k = 1/\{2f_k(1-f_k)\,\mathrm{SE}_k^2\}$) |

Getting this wrong by a factor of four changes $\sigma^2$ and the heritability by the
same factor. The same convention applies to the sample sizes passed to
`genesis project` and `genesis predict`: to project a study with 100,000 cases and
100,000 controls, use `--n 50000`.

A sample-size column whose name says "effective" (`Neff`, `n_eff`) is used as it
stands, with a warning, until `--n-scale` says which convention it follows (`0.25` or
`1`).

Give `--population-prevalence` to have the heritability also reported on the observed
and liability scales. The proportion of cases in the study, `--sample-prevalence`, is
taken from the case and control counts when the file or the command line provides them,
and has to be given otherwise.

### My file has no rsIDs, or its positions are in GRCh38.

SNPs are matched to the reference by identifier and, if fewer than half of them match
and the file has chromosome and position columns, by position. The bundled European
reference carries GRCh37 and GRCh38 positions; the build that matches more SNPs is
used and recorded (`provenance.data.matching`). Force the behaviour with
`--match-on rsid|position` and `--build GRCh37|GRCh38`.

### Which SNPs are analysed, and why were so many removed?

Only SNPs of the LD reference can be analysed — for the bundled reference, 1,070,777
HapMap3 SNPs with minor-allele frequency of at least 5% in 1000 Genomes Europeans. The
default quality control then applies, in this order, the filters recommended by the
authors of the method:

1. rows with missing or non-finite Z or N, or N ≤ 0, and duplicated identifiers (always);
2. N below 0.67 times the 90th percentile of N (`--n-min-fraction`);
3. optional: imputation quality and minor-allele frequency (`--min-info`, `--min-maf`),
   if the file has such columns;
4. SNPs that are not in the reference;
5. SNPs outside the HapMap3-without-MHC list (`--keep-mhc` to retain the MHC);
6. $\chi^2 > 80$ (`--max-chisq`).

Each step and the number of SNPs it removed are listed in the log, in
`PREFIX.summary.txt` and in the result file. `--no-qc` skips steps 2, 3, 5 and 6, which
is the default of the original software (`filter=FALSE`).

A typical GWAS imputed to 1000 Genomes or HRC retains about a million SNPs. A warning
is issued if fewer than half of the reference SNPs remain; the usual causes are
identifiers in another format or positions in an unexpected build.

### Can I analyse a non-European GWAS?

Yes, with an LD reference from a matching population. Build one from PLINK
`.bed/.bim/.fam` files of a reference panel with `genesis ld build` (genome-wide
HapMap3 SNPs take about a minute) and pass the directory with `--reference`. The
estimates refer to the SNPs of that reference; $M$ is their number.

### Do I need individual-level data?

No. The LD reference replaces it; individual-level genotypes are needed only to build
a new LD reference.

## Models and interpretation

### What do the parameters mean?

| Parameter | Meaning |
|---|---|
| `pic` ($\pi_c$) | proportion of the $M$ reference SNPs with a non-zero effect (susceptibility SNPs) |
| `sigmasq` ($\sigma^2$) | variance of the effects of susceptibility SNPs, per standardised genotype and phenotype (two-component model) |
| `p1`, `sigmasq1`, `sigmasq2` | three-component model: proportion of susceptibility SNPs in the cluster with the larger variance, and the two variances |
| `a` | variance added to every marginal effect estimate by confounding (population stratification, cryptic relatedness), in units of $1/n$; $1 + \bar n a$ is the analogue of the LD-score-regression intercept |

"Susceptibility SNPs" are to be read relative to the reference panel: each stands for
the causal variation it tags best. The number of susceptibility SNPs is $M\pi_c$ and
the heritability is $M\pi_c$ times the mean effect variance.

### How should I interpret the number of susceptibility SNPs?

As a lower bound. In simulations with realistic LD the estimator of $\pi_c$ is biased
downwards and that of the effect variance upwards, while the heritability is nearly
unbiased — a pattern reported in the original paper and reproduced with this package
(in one scenario, $\pi_c$ was underestimated by about a quarter; see
[Validation](validation.md#simulation-study)). SNPs with very small effects are
absorbed into the null component unless the sample size is large enough to resolve
them. Standard errors describe the sampling variability of the estimate, not this
bias.

### What is `c0`, and why does the report show a value other than 10?

A marker with many SNPs in its tagging neighbourhood may tag several susceptibility
SNPs. The likelihood sums over the number it tags, up to a cap `c0` per mixture
component. The fit starts with `c0 = 10`, as the original software does; if, at the
solution, the probability mass beyond the cap is not negligible (more than $10^{-6}$
on average), `c0` is raised and the model is solved again, up to `--c0-max` (20). The
report shows the final value, and `diagnostics.c0_path` lists every level with the
estimates obtained there. See
[Differences from the original](differences.md#the-truncation-level-c0).

### Two components or three?

Fit both (the default of `genesis fit`) and use the model with the lower BIC; the
report states the choice. The three-component model allows a small cluster of SNPs with
larger effects next to a large cluster of SNPs with small effects, which most
well-powered traits in the original paper needed. For studies with little power the
three-component model may not be identified — see the next question.

### What does "the three-component model is not identified" mean?

The two variance components have merged (they agree to within 1%) or one of them has
collapsed to zero. The data then cannot distinguish two clusters: `p1` is arbitrary,
the cluster-specific quantities are meaningless and the standard errors unreliable.
The fit is stopped as soon as this is detected, flagged (`diagnostics.degenerate`),
given no AIC or BIC, never selected over a two-component fit, and should not be
reported. This is the expected outcome for small studies and for
subsets of the genome, such as the chromosome-22 example shipped with the package.

A related message, "parameter(s) ... ended on a bound", means that a proportion of the
three-component model reached the edge of its admissible range ($10^{-7}$ to 0.5); the
parameter is then treated as fixed, its standard error is not available, and the model
is probably over-parameterised for the data.

### On what scale is the heritability?

For a quantitative trait, the proportion of phenotypic variance explained by the
reference SNPs. For a binary trait analysed with the effective sample size (above), the
log-odds-ratio scale of the original paper; with `--population-prevalence` and
`--sample-prevalence` it is converted to the observed and the liability scale as well.
The "sibling relative risk" line of the report is meaningful for binary traits only.

### Can `a` be negative?

Yes. `a` is estimated without a sign constraint (it must only keep every variance
positive). A clearly negative estimate usually indicates summary statistics that were
corrected by genomic control, or sample sizes that are too large. `--no-stratification`
fixes `a = 0`.

### Why do my estimates differ from the original R package or from published tables?

Mainly because this package reports the maximiser of the composite likelihood, whereas
the original reports the point at which its EM iteration stops, which lies a fraction
of a standard error short of the maximum. Standard errors differ because of the
convention for the sandwich estimator. `genesis fit --legacy --no-qc` reproduces the
original's numbers to rounding error, and every result file contains the standard
errors under both conventions. Details and measured effects:
[Differences from the original](differences.md).

### How do I read the warnings?

| Message (abridged) | What to do |
|---|---|
| The three-component model is not identified | use the two-component model |
| Parameter(s) ... ended on a bound | as above, unless the estimate is of substantive interest; compare the models by BIC |
| Truncation still matters at the ceiling `c0` = ... | the estimate of $\pi_c$ is limited by the cap on the number of susceptibility SNPs that one marker may tag; refit with a larger `--c0-max` and check whether the estimates move. Occurs when the estimated proportion of susceptibility SNPs is large (several per cent), as in the chromosome-22 example |
| EM reached the iteration limit | raise `--max-iter`; if Newton's method then converged the result says so and is usable |
| Newton polishing was skipped / stopped early; the information matrix is not positive definite | the likelihood is flat in some direction — usually a model that is too rich for the data (M3 on a weak GWAS). Do not use the standard errors; prefer M2 |
| The estimate is about ... standard errors from the stationary point | appears only with `--legacy` or `--no-polish`: the EM stopped short of the maximum |
| Only ... of the reference SNPs are present | check identifiers and genome build; with genuinely sparse data the estimates are valid but imprecise |
| No LD-correlated null replicates are available | the QQ envelope is too narrow; fetch them with `genesis data fetch --components nullz`, or build the reference with `--n-null` |
| No tagging neighbourhoods were supplied | Python API only: standard errors ignore LD and are too small; prepare the data with an LD reference |

Every warning is also stored in the `warnings` list of the result file.

## Computation

### How long does a fit take, and how much memory does it need?

A complete genome-wide analysis — both models, model selection, projections, per-SNP
posteriors and a QQ plot for about one million SNPs — took about a minute on a laptop
with 12 threads; measured times are in [Validation](validation.md#run-times). The cost
of one pass over the data is proportional to the number of SNPs and, for the
three-component model, to $(c_0+1)^2$, so fits in which the truncation level has to be
raised take longer, and a three-component model that the data cannot identify can use
all of its EM iterations (several minutes). Data are processed in chunks, so memory
does not grow with the truncation level. QQ simulations hold `n_sim` replicates of all
SNPs in single precision (about 0.4 GB for 100 replicates of a million SNPs).

### CPU or GPU? How many threads?

Both give the same results. On CPUs the SNPs are processed in parallel chunks; by
default all CPUs available to the process are used (`--threads`, `$GENESIS_NUM_THREADS`
or, under Slurm, `$SLURM_CPUS_PER_TASK`). With PyTorch and CUDA available,
`--device auto` uses every visible GPU; `--device cpu` forces the CPU. For a single
trait a multi-core CPU node is usually sufficient; see the
[Biowulf guide](../biowulf/index.md) for sizing jobs.

GPU execution has not yet been run on CUDA hardware during development (see
[Validation](validation.md#what-has-not-been-verified)); run `genesis doctor` on the
GPU node first.

### Does it run on an Apple-silicon GPU?

No. All computation is in double precision, which the MPS backend does not provide;
`--device mps` stops with an explanation. Use the CPU on a Mac.

### Are results reproducible?

Yes. A fit is deterministic. For a given chunk size the result does not depend on the
number of threads; across chunk sizes, backends and devices results agree to about
$10^{-10}$ in relative terms (the order of summation differs). Analyses that simulate
(QQ envelopes, projection intervals) take `--seed`. Result files record the version,
settings, backend, number of threads and the QC report.

### How do I run on a cluster without internet access?

Fetch the reference data where there is access, copy the directory, set
`GENESIS_DATA_DIR`, and add `--no-download`; see
[Installation](../getting-started/installation.md#offline-and-restricted-networks).

### How do I analyse many traits?

Each `genesis fit` is an independent job. `genesis swarm` writes a swarm file for the
NIH Biowulf cluster, with one line per summary-statistics file, and prints the command
that submits it; the [Biowulf guide](../biowulf/index.md) covers this workflow. Results
of many traits are easily combined from the `PREFIX.estimates.tsv` files.

### The fit seems to depend on the starting value. Does it?

In default mode the reported estimate is a solution of the likelihood equations, and
different starting values normally lead to the same one (verified on the GIANT height
data). A mixture likelihood can have several local maxima, however: pass several
starting proportions (`--starting-pic 0.002,0.01,0.05`) to run one fit per value and
keep the solution with the highest likelihood; the result file lists them all
(`diagnostics.multistart`).

## Downstream analyses

### What do the projections assume?

That a future GWAS of the given size tests the same SNPs, declares discoveries at
$P < 5\times10^{-8}$ (`--alpha`), and that effect sizes follow the fitted distribution.
"Discoveries" are susceptibility SNPs expected to reach significance; the percentage of
heritability refers to the heritability explained by the reference SNPs. Confidence
intervals propagate the uncertainty of the estimated parameters.

### Why is the QQ envelope narrower than in the paper's figures?

The original simulation adds the same random shift to all SNPs of a replicate where the
model calls for independent noise per SNP; this widens its envelope. See
[Differences from the original](differences.md#qq-plots-and-genomic-control-projections).

### Where are the formulas?

In the [theory chapter](../theory/index.md), and in the docstrings of the
[API reference](../api/index.rst), which state the formula each function implements.
