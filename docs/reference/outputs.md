# Output files

This page specifies every file written by the command-line interface and the format of
LD-reference bundles. The Python API returns the same information as objects
({class}`~genesis_gwas.FitResult`, {class}`~genesis_gwas.analysis.Projection`, ...);
`FitResult.to_json` / `FitResult.from_json` convert between the two.

## Files written by `genesis fit`

`genesis fit SUMSTATS --out PREFIX` writes the following files (directories in `PREFIX`
are created as needed):

| File | Written | Content |
|---|---|---|
| `PREFIX.log` | always | the complete log: command line, reference, every QC step, EM progress, reports, warnings. Unaffected by `--quiet`; more detail with `--verbose`. |
| `PREFIX.summary.txt` | always | human-readable report: version, input, reference; one block per fitted model; the model comparison; the QC table |
| `PREFIX.M2.json`, `PREFIX.M3.json` | one per fitted model (`--model 2`, `3` or `both`) | [complete result](#result-json), including the EM trace |
| `PREFIX.best.json` | always | the selected model, without the EM trace; the file to pass to the downstream commands |
| `PREFIX.estimates.tsv` | always | [tidy table](#estimates-table) of all estimates of all fitted models |
| `PREFIX.projection.tsv` | with `--project N1,N2,...` | [projection](#projection-table) of the selected model at the given sample sizes |
| `PREFIX.posterior.tsv.gz` | with `--posterior` | [per-SNP posterior summaries](#posterior-table) under the selected model |
| `PREFIX.qq.png`, `PREFIX.qq.tsv.gz` | with `--qq` | QQ plot of the selected model and [the curves behind it](#qq-table) |

"Selected model" means the model with the lower BIC when both were fitted; a
three-component fit flagged as *not identified* is never selected over an identified
two-component fit. With `--no-variance` no information criterion exists, no selection is
made, and `PREFIX.best.json` is the first model that was fitted.

The per-model result files are written to a temporary name and then renamed, so an
interrupted job does not leave a truncated `PREFIX.M2.json` or `PREFIX.M3.json` behind.

(summary-report)=
### The report (`PREFIX.summary.txt`)

```text
GENESIS fit: M3 (three-component model)
  GWAS markers analysed (K)     : 941,684
  Reference-panel SNPs (M)      : 1,070,777
  Mean sample size              : 132,149.5
  Optimisation                  : converged after 51 EM iteration(s), c0 = 14; 3 Newton step(s)
  Composite log-likelihood      : 3,950,983.1388
  Model selection               : BIC = -7,900,568.377   AIC = -7,901,757.653   d_s = 104.312

  Parameter           Estimate              SE
  pic             9.579237e-03      1.5153e-03
  p1              9.506459e-02      1.7698e-02
  sigmasq1        1.415832e-04      1.2316e-05
  sigmasq2        2.023950e-05      3.5689e-06
  a               1.534863e-06      1.2164e-07

  Derived quantity                                              Estimate          SE
  Number of susceptibility SNPs                                    10257        1623
  Total heritability                                             0.32592      0.0153
  Sibling relative risk                                            1.177    0.009002
  Susceptibility SNPs in cluster 1 (larger variance)               975.1       177.7
  Susceptibility SNPs in cluster 2 (smaller variance)             9282.1        1558
  Heritability explained by cluster 1 (larger variance)          0.13806     0.01752
  Heritability explained by cluster 2 (smaller variance)         0.18787     0.01697

  Convergence check             : estimate is within 7.94e-08 SE of the likelihood stationary point
```

| Line | Meaning |
|---|---|
| GWAS markers analysed ($K$) | SNPs that passed QC and were matched to the reference |
| Reference-panel SNPs ($M$) | the number of SNPs that $\pi_c$ refers to (all SNPs of the reference unless `--M` is given) |
| Optimisation | status of the reported estimate, number of EM iterations (over all truncation levels, if `c0` was raised), final truncation level `c0`, and the number of Newton steps that completed the final solve |
| Composite log-likelihood | renormalised truncated composite log-likelihood at the estimate (`loglik`) |
| Model selection | modified BIC and AIC (lower is better) and the effective number of parameters $d_s = \operatorname{tr}(I^{-1}J)$ |
| Parameter table | estimates and sandwich standard errors; `nan` marks a parameter that sits on a bound |
| Derived quantities | functions of the parameters with delta-method standard errors |
| Heritability on other scales | only with `--population-prevalence` and `--sample-prevalence` |
| Convergence check | largest remaining Newton step in standard-error units (`newton_distance_se`); values far below 1 mean that the estimate solves the likelihood equations |
| Warnings | everything the analyst should read before using the numbers |

Parameters are `pic` ($\pi_c$, proportion of susceptibility SNPs), `sigmasq`
($\sigma^2$, variance of their effects) and `a` (variance inflation from confounding)
for the two-component model; `pic`, `p1` (proportion of susceptibility SNPs in cluster 1), `sigmasq1`,
`sigmasq2` and `a` for the three-component model. Cluster 1 is always the cluster
with the larger variance.

(result-json)=
## Result files (`PREFIX.M2.json`, `PREFIX.M3.json`, `PREFIX.best.json`)

One JSON object. Numbers are written with full double precision; non-finite values
(`NaN`, for instance the standard error of a parameter on a bound) are written as
`null`. Matrices are lists of rows in the order of `param_names`.

```python
import genesis_gwas as gg
fit = gg.FitResult.from_json("PREFIX.best.json")     # every field below as an attribute
```

### Estimates

| Field | Type | Meaning |
|---|---|---|
| `schema` | int | version of this file format (currently 1); a file with a newer schema is refused with a request to upgrade |
| `model`, `n_components` | str, int | `"M2"` / 2 or `"M3"` / 3 |
| `stratification` | bool | whether `a` was estimated (`false`: fixed at 0) |
| `param_names` | list | `["pic", "sigmasq", "a"]` or `["pic", "p1", "sigmasq1", "sigmasq2", "a"]` |
| `estimates` | list | parameter estimates |
| `se` | list | sandwich standard errors (`null` for a parameter on a bound, or if the variance step was skipped) |
| `covariance` | matrix | sandwich covariance matrix $I^{-1} J I^{-1}$ |
| `derived` | object | `{name: {"estimate": ..., "se": ...}}` — see the table below |
| `loglik` | float | renormalised truncated composite log-likelihood at the estimate (the quantity the original software reports, and the one used in AIC/BIC) |
| `loglik_unnorm` | float | unnormalised truncated composite log-likelihood (the objective the EM algorithm increases) |
| `information` | matrix | observed information $I$, summed over markers |
| `J` | matrix | variability matrix $J$ of the sandwich estimator |
| `ds` | float | effective number of parameters $\operatorname{tr}(I^{-1}J)$ |
| `aic`, `bic` | float | $\mathrm{AIC} = -2\ell + 2 d_s$ and the modified BIC of the original software, $\mathrm{BIC} = -2\ell + d_s\log\bar n + 2\gamma\, d_s \log 5$, with $\ell$ = `loglik` and $\gamma$ = `bic_gamma`; lower is better |
| `bic_gamma` | float | tuning constant $\gamma$ of the modified BIC (default 0.5) |
| `n_snps`, `M`, `mean_n` | int, int, float | $K$, $M$ and the mean sample size $\bar n$ |

Derived quantities (`derived`):

| Key | Meaning | Formula |
|---|---|---|
| `n_ssnp` | number of susceptibility SNPs | $M\pi_c$ |
| `heritability` | total heritability explained by the reference SNPs (observed scale; log-odds-ratio scale for case–control studies) | $M\pi_c\sum_h p_h\sigma_h^2$ |
| `sibling_risk` | sibling relative risk (meaningful for binary traits) | $\exp(h^2/2)$ |
| `n_ssnp_cluster1`, `n_ssnp_cluster2` | susceptibility SNPs in the larger- and smaller-variance cluster (M3) | $M\pi_c p_1$, $M\pi_c(1-p_1)$ |
| `heritability_cluster1`, `heritability_cluster2` | heritability explained by each cluster (M3) | $M\pi_c p_h\sigma_h^2$ |

### Optimisation

| Field | Type | Meaning |
|---|---|---|
| `starting` | list | starting values of the run that was selected |
| `n_iter` | int | number of EM iterations, summed over the truncation levels at which the model was solved (see `diagnostics.c0_path`) |
| `c0` | int | truncation level of the latent counts at which the reported estimate was obtained |
| `converged` | bool | whether the reported estimate is a converged solution: the EM met its stopping rule, or Newton's method converged |
| `status` | str | `"converged"`, `"max_iter"` (iteration limit reached without convergence), `"stalled"` (no ascent step found), `"not_identified"` (a three-component fit that collapsed onto the two-component model and was stopped early; see `degenerate`) or, in `legacy` mode, `"legacy_break"` (the original's early exit) |
| `history` | list | one record per EM iteration, as one continuous trace across truncation levels (absent from `PREFIX.best.json`): `iter`; `loglik` and `loglik_unnorm` at the parameters the iteration started from; `theta`, the parameters after the update; `c0`; `backtracks`, the number of step halvings; `modified_newton`, the number of inner Newton steps that needed a modified direction |

### Diagnostics

`diagnostics` is an object with the following entries.

| Key | Meaning |
|---|---|
| `em_status` | how the EM phase ended: `"converged"` (its own stopping rule), `"newton"` (ended by a successful Newton solve after the warm-up — the normal outcome in default mode), `"max_iter"`, `"stalled"`, `"not_identified"`, `"legacy_break"` |
| `polish` | the Newton phase of the final solve, or `null` if it did not run (`--no-polish`, `legacy`). `status`: `"converged"`, `"max_steps"`, `"no_ascent"` or `"not_positive_definite"`; `n_steps`: Newton steps taken; `last_step_naive_se`: size of the last step in units of $\sqrt{\operatorname{diag}(I^{-1})}$; `fixed`: parameters held on a bound; `em_estimates`, `em_loglik`: the EM iterate from which Newton started; `loglik_gain`: log-likelihood at the reported estimate minus `em_loglik`; `accelerated`: whether Newton was attempted before the EM met its own stopping rule |
| `newton_step` | one further Newton step from the reported estimate, per parameter |
| `newton_distance_se` | that step divided by the standard error: the distance to the stationary point of the likelihood. In default mode of the order $10^{-6}$ or smaller; in `legacy` mode or with `--no-polish` it shows how far short the EM stopped |
| `information_pd` | whether the observed information is positive definite at the estimate; `false` means the point is not a local maximum and standard errors are not meaningful |
| `information_condition` | condition number of the information matrix after scaling to unit diagonal; values above about $10^{8}$ indicate a nearly non-identified model |
| `degenerate` | `true` for a three-component fit whose variance components merged (within 1%) or collapsed to zero: the model is not identified for these data, `p1` and the cluster-specific quantities are meaningless, `ds`, `aic` and `bic` are `null`, and the fit is excluded from model selection |
| `at_bound` | names of proportions (`pic`, `p1`) that ended on a bound of the three-component model; their standard errors are `null` and those of the other parameters are conditional on the constraint |
| `labels_swapped` | whether the two clusters were exchanged at the end so that cluster 1 has the larger variance |
| `truncation` | effect of truncating the latent counts at the final `c0`, evaluated at the estimate. `log_mass_deficit`: `loglik` − `loglik_unnorm` ≥ 0; `mean_tail`, `max_tail`: mean and maximum over markers of the prior mass beyond `c0`; `frac_above_1e6`: fraction of markers for which that mass exceeds $10^{-6}$ |
| `c0_path` | the truncation levels at which the model was solved, in order. In default mode `c0` is raised, and the model solved again from the previous solution, while `mean_tail` at the solution exceeds the tolerance `settings.em.tail_tol` ($10^{-6}$) and `c0` is below `c0_max`. Each record has `c0`, `mean_tail`, `estimates` and `loglik`; a single record means that the initial level was sufficient. Empty in `legacy` mode |
| `ldsc` | LD-score regression on the analysed markers, used for starting values: `intercept`, `slope`, `mean_chi2`, `lambda_gc` |
| `multistart` | one record per starting value (`--starting-pic` with several values): `starting`, `estimates`, `loglik`, `n_iter`, `status`, `c0`, `selected` |
| `prefit_m2` | for a three-component fit started from scratch: the internal two-component fit that initialised it |
| `em_backtracks` | total number of step halvings of the EM safeguard (0 for well-behaved data) |
| `sandwich_alt` | standard errors, $d_s$ and BIC under the *other* convention for the sandwich estimator. In default mode `convention` is `"original_double_count"`: the values the original software would report at this estimate. With `--sandwich-double-count` or `--legacy` it is `"paper"`. See [Differences from the original](differences.md) |
| `liability` | binary traits, when both prevalences were given: `h2_observed`, `se_h2_observed`, `h2_liability`, `se_h2_liability` and the inputs used |

### Settings, provenance and timings

| Field | Meaning |
|---|---|
| `settings` | everything that determines the result besides the data: `em` (initial and maximum truncation level `c0`, `c0_max`; `tail_tol`; iteration limit; tolerances; bounds), `legacy`, `sandwich_double_count`, `bic_gamma`, `variance`, `polish`, `accelerate`, `newton_after`, `has_neighborhoods` |
| `provenance.software`, `provenance.version` | `"genesis-gwas"` and its version |
| `provenance.compute` | backend, devices, number of threads, chunk size, floating-point type |
| `provenance.data.source` | the summary-statistics file |
| `provenance.data.sumstats` | rows read, the columns that were used, and how Z and N were obtained (`z_from`, `n_from`) |
| `provenance.data.reference` | name, version, LD configuration, population and genome build of the reference |
| `provenance.data.matching` | how SNPs were matched (identifier or position, genome build, counts) |
| `provenance.data.qc_config`, `provenance.data.qc` | the QC settings, and every filtering step with the number of SNPs removed and remaining |
| `provenance.data.n_input`, `provenance.data.n_final` | SNPs read and SNPs analysed |
| `timings` | wall-clock seconds: `em_seconds` (EM and Newton), `variance_seconds`, `prefit_seconds` (internal M2 fit, if any), `total_seconds` |
| `warnings` | the messages also shown in the report |

(estimates-table)=
## Estimates table (`PREFIX.estimates.tsv`)

Tab-separated, one row per model and quantity — convenient for collecting many traits
with `cat`, pandas or R.

```text
model	selected	quantity	label	estimate	se
M2	0	pic	pic	0.004839258444	0.000515006
M2	0	sigmasq	sigmasq	5.793067107e-05	4.90394e-06
M2	0	a	a	1.87902412e-06	1.26398e-07
M2	0	n_ssnp	Number of susceptibility SNPs	5181.766638	551.457
M2	0	heritability	Total heritability	0.3001832187	0.0140732
M2	0	sibling_risk	Sibling relative risk	1.161940682	0.00817611
...
M3	1	pic	pic	0.009579236693	0.00151527
M3	1	p1	p1	0.09506458765	0.0176984
M3	1	sigmasq1	sigmasq1	0.0001415831949	1.23162e-05
...
```

| Column | Meaning |
|---|---|
| `model` | `M2` or `M3` |
| `selected` | 1 for the rows of the selected model |
| `quantity` | machine-readable name: a parameter; a derived quantity (keys above); `loglik`, `bic`, `aic`, `ds`; `n_snps`, `n_iter`, `c0`, `converged`; and, if prevalences were given, `heritability_observed` and `heritability_liability` |
| `label` | human-readable label |
| `estimate`, `se` | value and standard error (`nan` where not applicable) |

(projection-table)=
## Projection table (`PREFIX.projection.tsv`, `genesis project`)

One row per future sample size.

| Column | Meaning |
|---|---|
| `n` | sample size of the future GWAS (for case–control studies: the effective sample size, see the [FAQ](faq.md)) |
| `heritability` | total heritability under the fitted model |
| `n_discoveries` | expected number of susceptibility SNPs reaching the significance level (default $5\times10^{-8}$) |
| `gv_percentage` | percentage of the heritability explained by those SNPs |
| `pheno_variance` | the same on the absolute scale, `gv_percentage × heritability / 100` |
| `*_lower`, `*_upper` | limits of the confidence interval (95% by default) obtained by drawing parameters from their estimated sampling distribution; written by `genesis fit --project` (unless `--no-variance`), and by `genesis project` with `--ci` |

(posterior-table)=
## Posterior table (`PREFIX.posterior.tsv.gz`, `genesis posterior`)

One row per analysed marker, in reference order (chromosome, position).

| Column | Meaning |
|---|---|
| `snp` | SNP identifier |
| `z` | Z statistic |
| `p_null` | local false-discovery rate: posterior probability that the marker tags no susceptibility SNP, i.e. that its marginal effect is zero |
| `post_mean` | posterior mean of the marginal effect (per standardised genotype and phenotype) |
| `post_sd` | posterior standard deviation of the marginal effect |
| `expected_count` | posterior expected number of susceptibility SNPs tagged by the marker |

(qq-table)=
## QQ files (`PREFIX.qq.png`, `PREFIX.qq.tsv.gz`, `genesis qq`)

The table holds the four curves of the plot in units of $-\log_{10}P$, sorted by
increasing significance: `observed`; `expected`, the mean over simulations under the
fitted model; and `lower`, `upper`, the point-wise envelope (80% by default). To keep
the file small it is thinned to 20,000 points: the 10,000 most significant are kept
exactly and the rest evenly. The observed and expected genomic-control factors are
written to the log.

## Tables printed by `genesis predict`

One row per training sample size: `n`; `threshold` (`optimum` or `gwas`); `alpha`, the
P-value threshold for including SNPs in the polygenic score; `auc`, the expected area
under the ROC curve; and, for each risk multiple `k` given with `--k-fold` (default 3,
4, 5), `pop_frac_at_kx_risk` and `case_frac_at_kx_risk`: the proportions of the
population and of cases whose predicted risk is at least `k` times the average.

(reference-bundle-format)=
## Reference bundles

An LD reference is a directory (a *bundle*). The references that are downloaded
by `genesis data fetch` (European, East Asian, African ancestry), the chromosome-22 reference inside the package and the output
of `genesis ld build` all have this layout, and any such directory can be passed as
`--reference DIR` or `load_reference(DIR)`.

```text
eur_1kg_hm3/1.0/
├── manifest.json
├── snps.npz
├── ld/
│   ├── w1000kb_r0.1.npz          one file per LD configuration
│   └── ...
├── pairld/
│   └── w1000kb_r0.1.npz          optional
├── nullz/
│   ├── nullz_000-009.npy         optional
│   └── ...
└── examples/                     optional
    └── height_giant2010.sumstats.tsv.gz
```

| File | Arrays | Content |
|---|---|---|
| `snps.npz` | `rsid` (bytes), `chrom` (uint8), `pos` (int32); optional `pos_grch38` or `pos_grch37` (int32), `hm3_no_mhc` (bool) | the $M$ reference SNPs in canonical order (chromosome, position, identifier). `pos` is in the build named in the manifest; `pos_grch38` (in a GRCh37 bundle) or `pos_grch37` (in a GRCh38 bundle) holds the positions in the other build, −1 where a SNP could not be converted, and allows matching summary statistics by position in either build; `hm3_no_mhc` marks the SNPs of the HapMap3-without-MHC list used by the QC filter (without it the filter excludes chromosome 6, 26–34 Mb, by position) |
| `ld/w{W}kb_r{R}.npz` | `nstar` (int32), `ldscore` (float64), `indptr` (int64, length $M+1$), `indices` (int32); scalars `window_kb`, `r2_min`, `n_ref` | for LD window `W` kb and threshold $r^2 \ge$ `R`: the number of SNPs each SNP tags, $N^*_k$ (itself included); its LD score $\ell_k$; and its tagging neighbourhood in compressed-sparse-row form — the SNPs tagged by SNP `k` are `indices[indptr[k]:indptr[k+1]]` (row numbers of `snps.npz`) |
| `pairld/w{W}kb_r{R}.npz` | `r2_u16` (uint16) | pairwise $r^2$ aligned with `indices`, stored as `round(65535 · r²)`; needed only for simulating LD-dependent statistics and `future_gc` |
| `nullz/nullz_{a}-{b}.npy` | float16, shape (replicates, $M$) | Z statistics of null phenotypes computed from the reference genotypes: draws with the LD structure of the panel, used for QQ envelopes |
| `manifest.json` | — | `format_version`, `name`, `version`, `population`, `panel`, `genome_build`, `n_ref` (panel size), `n_snps`, the list of `ld_configs` (`id`, `window_kb`, `r2_min`, `file`, `pairld`), `nullz` (`dtype`, `n_replicates`, `files`), and for every file its size and SHA-256 checksum; `source` records where the data come from (the original package and commit, or the settings and filter counts of `genesis ld build`) |

The LD score stored for SNP $k$ is the sum over its tagged SNPs of the bias-corrected
squared correlation $r^2 - (1-r^2)/(n_\text{ref}-2)$.

```python
import genesis_gwas as gg

ref = gg.load_reference("mini_chr22")
ref.M, ref.n_ref, ref.config.id          # (15633, 489, 'w1000kb_r0.1')
ref.rsid, ref.chrom, ref.pos             # SNP table
ref.nstar, ref.ldscore                   # per-SNP LD summaries
ref.neighborhoods.indptr                 # tagging neighbourhoods (CSR)
```
