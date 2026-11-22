# Quick start

A first analysis without downloading anything: fit the model to the chromosome-22 slice
of the GIANT height GWAS that ships inside the package, read the result, then run a
complete analysis on genome-wide data.

## 1. Check the installation

```bash
genesis doctor     # versions, CPUs/GPUs, backend self-test
```

## 2. Fit the packaged example

The package contains a small LD reference (`mini_chr22`: the 15,633 chromosome-22 SNPs
of the European reference) and the matching rows of the GIANT height summary
statistics ([Lango Allen et al. 2010](#ref-langoallen2010)) — a file with three
columns, `SNP`, `Z` and `N`.

::::{tab-set}

:::{tab-item} Command line
```bash
# path of the example file inside the installed package
EX=$(python -c "import genesis_gwas as gg; print(gg.load_reference('mini_chr22').example_path())")

genesis fit "$EX" --reference mini_chr22 --model 2 --out out/chr22
```

`genesis fit` reads the file, applies the recommended quality-control filters, merges
with the reference and fits the model — here the two-component model (`--model 2`), in
which the effects of susceptibility SNPs follow one normal distribution. This takes a
few seconds. Progress is logged to the terminal and to `out/chr22.log`; the report is
`out/chr22.summary.txt`:

```console
$ cat out/chr22.summary.txt
genesis-gwas 0.1.0
input: .../site-packages/genesis_gwas/datasets/mini_chr22/examples/height_chr22.sumstats.tsv.gz
reference: mini_chr22 v1.0 (EUR, GRCh37, M = 15,633; window 1000 kb, r2 >= 0.1)

GENESIS fit: M2 (two-component model)
  GWAS markers analysed (K)     : 13,230
  Reference-panel SNPs (M)      : 15,633
  Mean sample size              : 130,795.0
  Optimisation                  : converged after 52 EM iteration(s), c0 = 20; 4 Newton step(s)
  Composite log-likelihood      : 55,825.9412
  Model selection               : BIC = -111,305.981   AIC = -111,600.220   d_s = 25.831

  Parameter           Estimate              SE
  pic             5.779742e-02      9.8183e-03
  sigmasq         6.326638e-06      1.3709e-06
  a               2.657100e-07      5.5298e-07

  Derived quantity                                              Estimate          SE
  Number of susceptibility SNPs                                   903.55       153.5
  Total heritability                                           0.0057164    0.001056
  Sibling relative risk                                           1.0029   0.0005296

  Convergence check             : estimate is within 1.91e-07 SE of the likelihood stationary point

  Warnings:
    - Truncation still matters at the ceiling c0 = 20: on average 4.4e-05 of the prior mass is discarded (tolerance 1e-06). Estimates, pi_c in particular, may be limited by the truncation; check their sensitivity with a larger c0_max.


Quality control
  missing / non-finite Z or N, or N <= 0                           removed          0   remaining     13,230
  duplicated SNP identifier (first kept)                           removed          0   remaining     13,230
  N < 0.67 x 0.9-quantile of N (89,665)                            removed          0   remaining     13,230
  not in the LD reference (mini_chr22, matched on rsid)            removed          0   remaining     13,230
  several input SNPs map to the same reference SNP (first kept)    removed          0   remaining     13,230
  outside the HapMap3-without-MHC list                             removed          0   remaining     13,230
  chi-square > 80                                                  removed          0   remaining     13,230
```
:::

:::{tab-item} Python
```python
import genesis_gwas as gg

ref = gg.load_reference("mini_chr22")           # LD reference
data = gg.prepare(ref.example_path(), ref)      # read + QC + merge -> ModelData
m2 = gg.fit(data, 2)                            # two-component model

print(m2.summary())
print(m2.derived["heritability"])               # estimate and standard error
print(gg.project(m2, n=[250_000, 500_000, 1_000_000]).to_frame())
```

```text
GENESIS fit: M2 (two-component model)
  GWAS markers analysed (K)     : 13,230
  Reference-panel SNPs (M)      : 15,633
  Mean sample size              : 130,795.0
  Optimisation                  : converged after 52 EM iteration(s), c0 = 20; 4 Newton step(s)
  Composite log-likelihood      : 55,825.9412
  Model selection               : BIC = -111,305.981   AIC = -111,600.220   d_s = 25.831

  Parameter           Estimate              SE
  pic             5.779742e-02      9.8183e-03
  sigmasq         6.326638e-06      1.3709e-06
  a               2.657100e-07      5.5298e-07

  Derived quantity                                              Estimate          SE
  Number of susceptibility SNPs                                   903.55       153.5
  Total heritability                                           0.0057164    0.001056
  Sibling relative risk                                           1.0029   0.0005296

  Convergence check             : estimate is within 1.91e-07 SE of the likelihood stationary point

  Warnings:
    - Truncation still matters at the ceiling c0 = 20: on average 4.4e-05 of the prior mass is discarded (tolerance 1e-06). Estimates, pi_c in particular, may be limited by the truncation; check their sensitivity with a larger c0_max.
Derived(estimate=0.00571641540951196, se=0.0010561268785992037)
           n  heritability  n_discoveries  gv_percentage  pheno_variance
0   250000.0      0.005716       0.625215       0.594223        0.000034
1   500000.0      0.005716       6.819724       5.320172        0.000304
2  1000000.0      0.005716      39.769489      22.661846        0.001295
```

Conditions the analyst should know about are raised as Python warnings
(`genesis_gwas.exceptions.GenesisWarning` and its subclasses) and are also listed in
`fit.warnings`.
:::

::::

### Reading the report

- **Parameters.** `pic` is the proportion of reference SNPs with a non-zero effect,
  `sigmasq` the variance of those effects (per standardised genotype and phenotype),
  and `a` the variance inflation attributed to confounding. Standard errors come from
  the sandwich estimator, which accounts for LD between markers.
- **Derived quantities.** The number of susceptibility SNPs ($M\pi_c$) and the
  heritability they explain, with delta-method standard errors. Here $M$ = 15,633
  because the reference is a single chromosome; with the full reference $M$ = 1,070,777.
- **Optimisation.** The number of EM iterations, the truncation level `c0` of the
  likelihood, and the Newton steps that completed the maximisation. "Convergence check"
  confirms that the reported estimate solves the likelihood equations.
- **Warnings.** Read them. This fit raises one: truncation still matters at the ceiling `c0` = 20. The likelihood sums over the number of susceptibility SNPs that a marker tags, up to a cap `c0`. With an estimated 6% of the SNPs being susceptibility SNPs, a marker that tags hundreds of SNPs would be expected to tag more of them than the cap allows, so the estimate of $\pi_c$ — and with it the number of susceptibility SNPs — is set by the cap rather than by this small data set. The heritability is hardly affected (it stays between 0.0054 and 0.0058 for caps from 10 to 60). Genome-wide data do not raise this warning (step 4).

One chromosome of one study is a toy data set whose purpose is to show the mechanics.
For the same reason this step fits only the two-component model: the three-component
model, a mixture of two normal distributions, is not identified by so little data.
(`genesis demo` fits it nevertheless, to show how such a fit is flagged, and step 4
compares the two models on genome-wide data.) The chapters of the
[tutorial](../tutorial/index.md) explain each element of the report and how to
interpret it.

## 3. What was written

```console
$ ls out/
chr22.M2.json
chr22.best.json
chr22.estimates.tsv
chr22.log
chr22.summary.txt
```

| File | Content |
|---|---|
| `chr22.summary.txt` | the report shown above |
| `chr22.M2.json` | complete result of the two-component model (estimates, covariance matrix, diagnostics, EM trace); with `--model both` there is a `chr22.M3.json` as well |
| `chr22.best.json` | the selected model — here the only one — without the trace |
| `chr22.estimates.tsv` | one row per model and quantity — convenient for combining many traits |
| `chr22.log` | the full log |

Formats are specified in [Output files](../reference/outputs.md). Any result can be
loaded again,

```python
fit = gg.FitResult.from_json("out/chr22.best.json")
fit.estimates, fit.se, fit.derived["heritability"]
```

and passed to the downstream commands:

```console
$ genesis project out/chr22.best.json --n 250000,500000,1000000
n               heritability    n_discoveries   gv_percentage   pheno_variance
250000          0.0057164154    0.62521479      0.59422332      3.3968273e-05
500000          0.0057164154    6.8197242       5.3201721       0.00030412314
1000000         0.0057164154    39.769489       22.661846       0.0012954453
$ genesis plot density out/chr22.best.json --out out/density.png
```

The first prints, for future GWAS of the given sizes, the expected number of
genome-wide significant susceptibility SNPs and the percentage of heritability they
explain; the second draws the fitted effect-size distribution.

## 4. A genome-wide analysis

Without `--reference` the European reference is used, which is downloaded and cached
on first use; without `--model` both models are fitted and one is selected by BIC. The
GIANT height file of the original package is available as an example data set:

```bash
# reference (32 MB), example GWAS (13 MB) and null replicates for the QQ plot (214 MB);
# prints the directory that holds them
DATA=$(genesis data fetch --components default,examples,nullz)

genesis fit "$DATA/examples/height_giant2010.sumstats.tsv.gz" \
    --out results/height --qq --project 250000,500000,1000000
```

The analysis uses 941,684 SNPs after quality control and merging and fits both models. On the laptop used for development it took about 80 seconds with 18 threads while other jobs were running, with a peak memory use of 3.0 GB. The report:

```console
$ cat results/height.summary.txt
genesis-gwas 0.1.0
input: ~/.cache/genesis-gwas/eur_1kg_hm3/1.0/examples/height_giant2010.sumstats.tsv.gz
reference: eur_1kg_hm3 v1.0 (EUR, GRCh37, M = 1,070,777; window 1000 kb, r2 >= 0.1)

GENESIS fit: M2 (two-component model)
  GWAS markers analysed (K)     : 941,684
  Reference-panel SNPs (M)      : 1,070,777
  Mean sample size              : 132,149.5
  Optimisation                  : converged after 50 EM iteration(s), c0 = 10; 6 Newton step(s)
  Composite log-likelihood      : 3,950,388.0029
  Model selection               : BIC = -7,899,524.616   AIC = -7,900,589.247   d_s = 93.379

  Parameter           Estimate              SE
  pic             4.839258e-03      5.1501e-04
  sigmasq         5.793067e-05      4.9039e-06
  a               1.879024e-06      1.2640e-07

  Derived quantity                                              Estimate          SE
  Number of susceptibility SNPs                                   5181.8       551.5
  Total heritability                                             0.30018     0.01407
  Sibling relative risk                                           1.1619    0.008176

  Convergence check             : estimate is within 1.74e-09 SE of the likelihood stationary point

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

Model comparison by BIC (lower is better)
  Model               logLik       d_s               AIC               BIC       delta
  M2           3,950,388.003    93.379    -7,900,589.247    -7,899,524.616    1043.761
  M3           3,950,983.139   104.312    -7,901,757.653    -7,900,568.377       0.000  <- selected

Quality control
  missing / non-finite Z or N, or N <= 0                           removed          0   remaining  1,037,172
  duplicated SNP identifier (first kept)                           removed          0   remaining  1,037,172
  N < 0.67 x 0.9-quantile of N (89,669)                            removed          0   remaining  1,037,172
  not in the LD reference (eur_1kg_hm3, matched on rsid)           removed     95,488   remaining    941,684
  several input SNPs map to the same reference SNP (first kept)    removed          0   remaining    941,684
  outside the HapMap3-without-MHC list                             removed          0   remaining    941,684
  chi-square > 80                                                  removed          0   remaining    941,684
```

For height the three-component model is identified and clearly preferred. The
projection for larger studies, with 95% confidence intervals
(`results/height.projection.tsv`):

```text
n        heritability  n_discoveries  n_discoveries_lower  n_discoveries_upper  gv_percentage  gv_percentage_lower  gv_percentage_upper  pheno_variance  pheno_variance_lower  pheno_variance_upper
250000   0.32592325    605.82016      515.975              724                  44.126191      39.034712            50.615355            0.14381752      0.1272232             0.16496721
500000   0.32592325    1454.7366      1272.95              1652.025             63.328131      56.906584            70.897537            0.2064011       0.18547179            0.23107156
1000000  0.32592325    2830.5283      2508.85              3144.05              80.661206      74.731313            86.052814            0.26289363      0.24356672            0.28046613
```

:::{figure} quickstart-height-qq.png
:alt: QQ plot of the height GWAS against the fitted three-component model
:width: 60%

`results/height.qq.png`: observed P-values against those expected under the fitted
model; the shaded band is the simulation envelope. Points on the diagonal mean that the
fitted effect-size distribution reproduces the observed distribution of test
statistics.
:::

For your own data, replace the file name. Column names are detected automatically
(`SNP`/`rsid`, `Z` or `BETA`+`SE` or `OR`+`SE`, `N` or case/control counts, ...);
`genesis fit --help` lists the options that name them explicitly, and the
[FAQ](../reference/faq.md) explains the sample-size convention for case–control
studies — the one setting that is easy to get wrong.

## Where next

::::{grid} 1 2 2 2
:gutter: 2

:::{grid-item-card} Tutorial
:link: ../tutorial/index
:link-type: doc

The model, the data, fitting, interpretation and projections, step by step with
figures.
:::

:::{grid-item-card} Theory
:link: ../theory/index
:link-type: doc

Derivations of the likelihood, the EM algorithm and the sandwich variance.
:::

:::{grid-item-card} Biowulf
:link: ../biowulf/index
:link-type: doc

Running on the NIH cluster: CPUs, GPUs and swarms of many traits.
:::

:::{grid-item-card} Differences from the R package
:link: ../reference/differences
:link-type: doc

What changed relative to the original software, with measured effects.
:::

::::
