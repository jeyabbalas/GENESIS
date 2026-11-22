# Interpreting the estimates

The fit returns three or five parameters. This chapter turns them into statements about
the trait, shows what the standard errors do and do not cover, and measures how much the
answers depend on the one tuning choice of the method, the definition of LD.

## From parameters to quantities

| Quantity | Formula | Name in `derived` and in `PREFIX.estimates.tsv` |
|---|---|---|
| number of susceptibility SNPs | $M\pi_c$ | `n_ssnp` |
| heritability | $M\pi_c\sum_h p_h\sigma_h^2$ | `heritability` |
| susceptibility SNPs in cluster $h$ (M3) | $M\pi_c p_h$ | `n_ssnp_cluster1`, `n_ssnp_cluster2` |
| heritability explained by cluster $h$ (M3) | $M\pi_c p_h\sigma_h^2$ | `heritability_cluster1`, `heritability_cluster2` |
| sibling relative risk (binary traits) | $\exp(h^2/2)$ | `sibling_risk` |

The script reads the two fits saved in the [previous chapter](fitting.md):

```{literalinclude} ../_scripts/examples/interpret.py
:language: python
```

```{literalinclude} ../_scripts/output/interpret.txt
:language: console
```

Reading the output for height under the selected model, M3:

- **About ten thousand susceptibility SNPs**, with a standard error of about 16 %. This
  is the number of SNPs *in the reference panel* with a non-zero effect. Under M2 the
  estimate is half as large: a single normal has no room for a large number of very small
  effects, so it classifies them as null. The paper describes these counts as lower
  bounds; the gap between M2 and M3 shows why.
- **Heritability of about 0.33**, with a standard error below 5 %. Unlike the count,
  it hardly depends on the model (0.30 under M2). Proportion and variance trade off
  against each other, and their product is what the data determine best.
- **Two clusters.** Roughly one susceptibility SNP in ten belongs to cluster 1, whose
  effects are 2.6 times larger in standard deviation, and those SNPs carry about 40 % of
  the heritability.
- **Sibling relative risk** is printed for every fit but is meaningful only for a
  disease analysed on the log-odds scale ([Chapter 6](binary-traits.md)). Ignore it for
  height.

## The distribution of effect sizes

A count of susceptibility SNPs is less informative than it sounds, because most of them
have tiny effects. The distribution is the real result.

```{figure} ../_static/figures/effect-density.svg
:name: fig-effect-density
:alt: Left, the fitted density of the effect sizes of susceptibility SNPs for height under the two- and three-component models; the three-component density has a sharper peak and heavier tails. Right, the expected number of susceptibility SNPs whose absolute effect exceeds a threshold, on a logarithmic scale, for both models.

**Fitted effect-size distributions for height.** (a) Density of the joint effects of
susceptibility SNPs. (b) Expected number of susceptibility SNPs with an absolute effect
larger than $x$. Effects are in standard deviations of height per standard deviation of
genotype.
```

Panel (b) is the more useful view. The two models disagree most about the smallest
effects, where a GWAS of this size has no power and the estimate is an extrapolation.
They agree to within 20 % on the number of SNPs with effects above 0.005, and M3 puts
about twice as many SNPs above 0.02.

{func}`genesis_gwas.n_in_interval` returns such counts directly, as in the last table of
the script above, and {func}`genesis_gwas.effect_density` the density itself.
`genesis plot density M2=results/height.M2.json M3=results/height.M3.json --out
density.png` draws panel (a) from the command line.

:::{admonition} Units
:class: note

Effects are standardised: a value of 0.01 means 0.01 standard deviations of the trait per
standard deviation of genotype. For a SNP with allele frequency $f$ the per-allele effect
is larger by the factor $1/\sqrt{2f(1-f)}$ — 1.54 at $f = 0.3$, 2.36 at $f = 0.1$. For a
disease, exponentiate the per-allele value to obtain an odds ratio.
:::

## Standard errors

The standard errors come from a sandwich estimator that sums each marker's contribution
to the likelihood equations with those of the markers in its tagging neighbourhood
({ref}`inf-sandwich`). Standard errors of derived quantities follow by the delta method
({ref}`inf-delta`). An approximate 95 % interval is the estimate ± 1.96 standard errors.

What they cover, and what they do not:

- They account for LD between markers within a neighbourhood. In the paper's simulations
  they were slightly conservative.
- They are conditional on the model. The difference between the M2 and M3 counts above
  is several M2 standard errors; no standard error reflects model choice.
- They are conditional on the LD definition ({ref}`tut-ld-sensitivity`).
- A parameter that ended on a bound has no standard error, and those of a fit that is not
  identified are meaningless ({ref}`tut-warnings`).

:::{admonition} Two conventions for the sandwich estimator
:class: note

The original software counts each marker's own contribution twice when it forms the
"meat" of the sandwich, which makes its standard errors a few per cent larger than the
formula in the paper. This package follows the paper and stores the other convention
alongside (`diagnostics["sandwich_alt"]`); the last line of the output above shows the
ratio for height. `--sandwich-double-count` makes the original convention the reported
one. See [Differences from the original](../reference/differences.md).
:::

(tut-ld-sensitivity)=
## Sensitivity to the LD definition

The neighbourhood of a marker — and with it $N^*_k$ and $\ell_k$ — depends on a window
and an $r^2$ threshold that are chosen, not estimated. The reference ships nine
combinations. Refitting M2 under each takes a few minutes:

```{literalinclude} ../_scripts/examples/ld_sensitivity.py
:language: python
```

```{literalinclude} ../_scripts/output/ld_sensitivity.txt
:language: console
```

```{figure} ../_static/figures/ld-sensitivity.svg
:name: fig-ld-sensitivity
:alt: Three panels showing, for LD windows of 0.5, 1 and 2 megabases and r-squared thresholds of 0.05, 0.1 and 0.2, the estimated number of susceptibility SNPs with 95 percent intervals, the estimated heritability with 95 percent intervals, and the mean neighbourhood size.

**Height, two-component model, under the nine LD definitions of the bundled reference.**
(a) Number of susceptibility SNPs and (b) heritability, with 95 % intervals; (c) mean
size of the tagging neighbourhood.
```

Three things stand out.

1. **The window hardly matters; the threshold does.** Going from 0.5 to 2 Mb changes the
   estimated number of susceptibility SNPs by less than 10 %. Going from $r^2 \ge 0.05$
   to $r^2 \ge 0.2$ raises it by about 70 %, more than the confidence interval at any one
   setting.
2. **The mechanism is visible in panel (c).** A stricter threshold makes neighbourhoods
   smaller. The model then needs a larger $\pi_c$ for the same number of markers to tag a
   susceptibility SNP.
3. **Heritability moves much less** — from 0.28 to 0.34 — and in the same direction.

The paper reports the same behaviour: the total number of susceptibility SNPs differed
notably between LD definitions for some traits, while the tails of the effect-size
distribution did not.

What to do with this:

- Use the default (1 Mb, $r^2 \ge 0.1$) for the headline analysis; it is the definition
  used for the primary analysis of the paper.
- Compare traits, ancestries or subtypes **under the same definition**. Differences
  between traits are far more stable than the absolute numbers.
- If the absolute count matters for your argument, show its range across definitions, as
  above.

## Comparing with the published results

Table 1 of [Zhang et al. (2018)](../reference/bibliography.md#ref-zhang2018) reports for
height 9.5 thousand susceptibility SNPs (standard error 1.2 thousand), of which 0.90
thousand in the cluster of larger effects, and a heritability of 0.324. The estimates
above — 10.3 thousand (1.6 thousand), 0.98 thousand and 0.326 — are close but not
identical, for two documented reasons:

- the original software stops the EM algorithm before the maximum is reached
  ({ref}`tut-polish`), which accounts for about 5 % in $\hat\pi_c$;
- it keeps the truncation level at 10 for these data, whereas the default here moves the
  three-component fit to 14 after checking the solution ({ref}`est-adaptive-c0`), which
  accounts for a further 2.5 % and for most of the increase in the standard error.

`--legacy` reproduces the output of the original software, which is what the paper
reports; [Validation](../reference/validation.md) shows the comparison, and
[Differences from the original](../reference/differences.md) explains each difference.

## What to report

- the selected model and the BIC difference;
- the number of susceptibility SNPs and the heritability with standard errors, stating
  the reference panel and LD definition;
- for M3, the size and share of heritability of each cluster;
- the QQ plot of the selected model;
- the number of markers analysed and the number removed by each filter;
- the software version and whether any non-default option was used (both are recorded in
  the result files).
