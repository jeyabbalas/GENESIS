# Projections, prediction and per-SNP results

An estimated effect-size distribution answers questions that the original GWAS cannot:
what a larger study would find, how well a polygenic score could ever predict, and how
much to believe any individual association. All four analyses in this chapter start from
the result file of [Chapter 3](fitting.md) and take seconds.

The derivations are in [Downstream analyses](../theory/downstream.md).

## What would a larger study find?

For a future GWAS of $n$ individuals, {func}`genesis_gwas.project` computes the expected
number of susceptibility SNPs reaching genome-wide significance and the share of
heritability they explain.

```{literalinclude} ../_scripts/examples/project.py
:language: python
```

```{literalinclude} ../_scripts/output/project.txt
:language: console
```

`discoveries` is the expected number of susceptibility SNPs with $P < 5\times10^{-8}$
(`alpha=` changes the level), `h2_explained_%` the percentage of the heritability they
account for, and `trait_variance_%` the same as a percentage of the trait variance. The
limits are 95 % intervals that combine the uncertainty of the estimated parameters with
the chance variation of the discovery count.

```{figure} ../_static/figures/projection.svg
:name: fig-projection
:alt: Two panels showing, as a function of GWAS sample size up to two million, the expected number of genome-wide significant susceptibility SNPs for height and the percentage of heritability they explain, under the two- and three-component models with 95 percent bands.

**Projected yield of larger height GWAS.** (a) Expected number of susceptibility SNPs
reaching $P < 5\times10^{-8}$ and (b) the percentage of heritability they explain, with
95 % intervals. The dotted line marks the sample size of the GWAS the model was fitted
to.
```

How to read this:

- **The model matters, and most far from the data.** At the current sample size the two
  models attribute the same share of heritability to the discoveries (30 %), although M2
  expects more of them (330 against 255). At two million M3 expects 39 % more
  discoveries than M2, because its additional small effects become detectable only then.
- **Heritability explained saturates slowly.** Going from 30 % to 90 % of the
  heritability takes fifteen times the sample size.
- **It is a projection under the fitted model.** It assumes that the future study is of
  the same population and trait definition, and that the effect-size distribution
  estimated from today's data is right about effects that are too small to see today.

The paper checked such projections against studies published later. For height, the
genome-wide significant SNPs of a GWAS of about 700,000 individuals
([Yengo et al., 2018](../reference/bibliography.md#ref-yengo2018)) explained 24.6 % of
the trait variance; the model fitted to the 2010 data had predicted 24.0 %. The row for
$n = 700{,}000$ above is this package's version of that prediction.

`genesis fit --project N1,N2,…` writes this table as `PREFIX.projection.tsv`, and
`genesis project` computes it from a saved fit:

```{literalinclude} ../_scripts/output/cli_project.txt
:language: console
```

`genesis plot discoveries` and `genesis plot variance` draw the two panels of
{numref}`fig-projection`; in Python use {func}`genesis_gwas.plotting.plot_projection`.

## How well could a polygenic score predict?

{func}`genesis_gwas.polyrisk_predict` implements the theory of
[Chatterjee et al. (2013)](../reference/bibliography.md#ref-chatterjee2013). It considers
a score that includes every SNP passing a P-value threshold in a GWAS of size $n$,
weighted by its estimated effect, and returns the expected correlation $\mu$ between that
score and the true genetic value. For a quantitative trait $\mu^2$ is the expected
out-of-sample $R^2$; for a disease $\mu$ determines the AUC
([next chapter](binary-traits.md)).

```{literalinclude} ../_scripts/examples/predict.py
:language: python
```

```{literalinclude} ../_scripts/output/predict.txt
:language: console
```

```{figure} ../_static/figures/prs-height.svg
:name: fig-prs-height
:alt: Left, the expected predictive R-squared of a polygenic score for height against GWAS sample size on a logarithmic axis, for the optimal P-value threshold and for genome-wide significant SNPs only, approaching the heritability. Right, the optimal threshold against sample size.

**Expected accuracy of a polygenic score for height.** (a) Expected $R^2$ of a score that
uses the optimal P-value threshold (orange) or only genome-wide significant SNPs
(black); the dashed line is the heritability, the ceiling. (b) The optimal threshold.
```

Two messages, both from the paper and both visible here:

- **Genome-wide significance is the wrong threshold for prediction.** At the sample size
  of this GWAS, the optimal threshold is near $P < 2\times10^{-3}$ and the expected $R^2$
  is about 70 % higher than with genome-wide significant SNPs alone.
- **Returns diminish.** Each doubling of the sample size adds less, and the heritability
  is a ceiling that even a study of millions does not reach.

:::{admonition} What the prediction assumes
:class: caution

- `M` is the number of *independent* SNPs among which the score has to find the
  susceptibility SNPs; null SNPs that pass the threshold add noise. The paper uses
  200,000. The default of the function, and of `genesis predict`, is the number of
  reference SNPs, which is more pessimistic.
- SNPs are treated as independent and weights are the marginal estimates: this is a
  "pruning and thresholding" score. Methods that model LD or shrink effects can do
  better.
- The score is applied to the population the GWAS came from.
:::

## Which associations are real, and how large are they?

Given the fitted model, each marker has a posterior distribution.
{func}`genesis_gwas.posterior_marginal` (CLI: `--posterior`, or `genesis posterior` for a
saved fit) returns for every marker

`p_null`
: the posterior probability that the marker tags *no* susceptibility SNP — its local
  false-discovery rate ([Efron & Tibshirani, 2002](../reference/bibliography.md#ref-efron2002));

`post_mean`, `post_sd`
: the posterior mean and standard deviation of the marker's true marginal effect, on the
  standardised scale — a shrunken estimate that corrects for the "winner's curse";

`expected_count`
: the posterior expected number of susceptibility SNPs the marker tags.

```{literalinclude} ../_scripts/examples/posterior.py
:language: python
```

```{literalinclude} ../_scripts/output/posterior.txt
:language: console
```

```{figure} ../_static/figures/posterior.png
:name: fig-posterior
:alt: Left, a scatter plot of the local false-discovery rate against the absolute z statistic for all markers of the height GWAS, coloured by LD score: at the same z, markers with high LD scores have much lower values. Right, the posterior mean effect against the observed z statistic, showing strong shrinkage towards zero for small z and little shrinkage for large z.

**Per-SNP posterior summaries for height under M3**, coloured by LD score. (a) The local
false-discovery rate depends on the LD score as well as on $z$. (b) Posterior mean of the
effect, expressed as a $z$ statistic, against the observed $z$; the dashed line is no
shrinkage.
```

What the figure shows:

- **The same $z$ does not mean the same evidence.** A marker with a large neighbourhood
  is likely to tag a susceptibility SNP before any data are seen, so a moderate $z$ is
  enough to make an association probable. For an isolated marker the same $z$ is most
  likely noise. A single P-value threshold ignores this.
- **Far more markers are probably real than are genome-wide significant** — compare the
  counts in the output. This reflects the polygenicity of height, and it is the basis of
  the more liberal optimal thresholds of the previous section.
- **Shrinkage is strong for small $z$ and mild for large $z$** (b). Estimated effects of
  barely significant markers are biased away from zero; the posterior mean is the
  model-based correction.

:::{admonition} Read `p_null` correctly
:class: caution

A small `p_null` says the marker is *in LD with* at least one susceptibility SNP. It does
not say the marker is causal, and neighbouring markers share the same signal. The values
also inherit the model's caveat: because the number of susceptibility SNPs is a lower
bound, `p_null` tends to be too large, which makes it conservative.
:::

## How inflated will test statistics be?

A polygenic trait inflates the genomic-control factor $\lambda_{GC}$ without any
confounding, and the more so the larger the study.
{func}`genesis_gwas.future_gc` predicts $\lambda_{GC}$ at a given sample size by
simulating a GWAS from the fitted model with the LD of the reference.

```{literalinclude} ../_scripts/examples/future_gc.py
:language: python
```

```{literalinclude} ../_scripts/output/future_gc.txt
:language: console
```

At the sample size of the data the prediction is close to the observed value (1.52
against 1.54), which is a check of the fit in its own right. The values for larger
studies say how much inflation to expect from polygenicity alone, and therefore how
misleading it would be to "correct" it away by genomic control.

## Summary of commands

| Question | Python | Command line |
|---|---|---|
| discoveries and heritability explained at sample size $n$ | `gg.project(fit, n, ci=True)` | `genesis project FIT.json --n … --ci` |
| accuracy of a polygenic score | `gg.polyrisk_predict(n, fit, M=200_000)` | `genesis predict FIT.json --n …` |
| local false-discovery rates and shrunken effects | `gg.posterior_marginal(data, fit)` | `genesis posterior …` |
| QQ plot against the fitted model | `gg.qq_simulation(data, fit, ref)` | `genesis qq …` |
| future genomic inflation | `gg.future_gc(fit, n, ref)` | — |
| plots of densities and projections | `genesis_gwas.plotting` | `genesis plot …` |
