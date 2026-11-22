# The model in pictures

This chapter explains what GENESIS assumes and estimates, with one real genomic region
and the height results as illustrations. Nothing here is needed to *run* the software,
but it is what you need to *read* its output. Derivations are in
[The statistical model](../theory/model.md).

## The question

A GWAS tests SNPs one at a time. Most associated SNPs are not causal themselves; they are
correlated with something that is. Counting genome-wide significant hits therefore says
little about how many variants affect a trait, and nothing about the thousands whose
effects are too small to be significant yet.

GENESIS asks instead: **what distribution of effect sizes, across all common SNPs, would
produce the association statistics we observed?** The answer is a handful of numbers —
how many SNPs have an effect, and how large those effects are — from which the practical
quantities follow.

## Joint and marginal effects

Two kinds of effect need to be kept apart.

- The **joint effect** $\beta^{(J)}_m$ of SNP $m$ is its effect when all SNPs are in the
  model together. It is the effect that belongs to the SNP itself. A SNP with a non-zero
  joint effect is called a *susceptibility SNP*.
- The **marginal effect** $\beta_k$ of marker $k$ is what a GWAS estimates: the effect of
  that marker analysed alone. It collects the joint effects of all SNPs the marker is
  correlated with,
  $\beta_k = \sum_m \rho_{km}\,\beta^{(J)}_m$, where $\rho_{km}$ is the LD correlation.

Throughout, effects are *standardised*: per standard deviation of genotype, in units of
the standard deviation of the trait (for a disease, a log odds ratio per standard
deviation of genotype). For a SNP with allele frequency $f$, the per-allele effect is the
standardised effect divided by $\sqrt{2f(1-f)}$.

```{figure} ../_static/figures/joint-vs-marginal.svg
:name: fig-joint-vs-marginal
:alt: Two aligned panels over a two-megabase region of chromosome 3. The top panel shows a single non-zero joint effect at one SNP. The bottom panel shows non-zero marginal effects at the 66 markers in linkage disequilibrium with that SNP.

**One susceptibility SNP produces many marginal associations.** A hypothetical joint
effect of 0.02 placed at rs724016 (a), and the marginal effects it induces at the 66
reference SNPs in LD with it (b), computed from the real LD of the region
($|\rho_{km}|$ from the bundled reference). Grey ticks are the other reference SNPs.
```

The model is about the *top* panel; the data are the *bottom* panel plus noise.

## The effect-size distribution

GENESIS assumes that the joint effects of the $M$ SNPs of the reference panel are
independent draws from a "spike and normal" distribution:

$$
\beta^{(J)}_m \;\sim\; \underbrace{(1-\pi_c)\,\delta_0}_{\text{no effect}}
\;+\; \underbrace{\pi_c \sum_{h=1}^{H} p_h\, N(0,\sigma_h^2)}_{\text{susceptibility SNPs}} .
$$

This is equation (1) of
[Zhang et al. (2018)](../reference/bibliography.md#ref-zhang2018). Two versions are
fitted:

**M2 — the two-component model** ($H=1$)
: A SNP either has no effect, or an effect drawn from one normal distribution
  $N(0,\sigma^2)$. Parameters: $\pi_c$, $\sigma^2$.

**M3 — the three-component model** ($H=2$)
: Susceptibility SNPs come in two clusters: a fraction $p_1$ with larger effects,
  $N(0,\sigma_1^2)$, and the rest with smaller effects, $N(0,\sigma_2^2)$. Parameters:
  $\pi_c$, $p_1$, $\sigma_1^2$, $\sigma_2^2$.

```{figure} ../_static/figures/effect-prior.svg
:name: fig-effect-prior
:alt: Two panels showing the fitted distribution of joint effect sizes for height. Each has an arrow at zero for the point mass of SNPs with no effect and a bell-shaped density for susceptibility SNPs; in the right panel the density is the sum of a wide low curve and a narrow tall curve.

**The fitted effect-size distributions for height.** The arrow is the point mass at zero
(more than 99 % of SNPs); the curve is the density of the effects of susceptibility SNPs.
Under M3 (b) the density is a mixture of a small cluster of larger effects and a large
cluster of smaller ones, which gives it a sharper peak and heavier tails than a single
normal (a).
```

Real effect-size distributions are not exactly mixtures of two normals. M3 is useful
because a two-normal mixture can imitate a wide range of peaked, heavy-tailed shapes, and
because — as the height example will show — a single normal often fits GWAS data visibly
worse.

## What a marker tags

Whether marker $k$ shows an association depends on how many susceptibility SNPs it is
correlated with, and how strongly. The LD reference summarises this with two numbers per
marker:

$N^*_k$ — the number of tagged SNPs
: the reference SNPs within 1 Mb of the marker with $r^2 \ge 0.1$, the marker included.
  This set is the marker's *tagging neighbourhood* $\mathcal N_k$.

$\ell_k$ — the LD score
: the sum of (bias-corrected) $r^2$ between the marker and the SNPs in its neighbourhood,
  the same quantity as in LD-score regression
  ([Bulik-Sullivan et al., 2015](../reference/bibliography.md#ref-bulik-sullivan2015)).

```{figure} ../_static/figures/ld-neighbourhood.svg
:name: fig-ld-neighbourhood
:alt: Squared correlation between the marker rs724016 and every reference SNP within one megabase, plotted against position on chromosome 3. Sixty-six SNPs in a 300-kilobase stretch around the marker exceed the threshold of 0.1; 676 other SNPs in the window do not.

**The tagging neighbourhood of one marker.** $r^2$ between rs724016 (chromosome 3) and the
reference SNPs within 1 Mb, from the bundled European reference. The 66 SNPs above the
threshold form the neighbourhood ($N^*_k = 66$); their bias-corrected $r^2$ values sum to
the LD score ($\ell_k = 26.2$). The 676 other SNPs in the window are not tagged.
```

Neighbourhoods vary enormously across the genome: in the bundled reference the median
marker tags 37 SNPs, one in ten tags more than 93, and the largest neighbourhood has 710.
This variation is what the method feeds on.

The window (1 Mb) and threshold (0.1) are choices, not estimates; the reference ships nine
combinations so that you can check how much they matter ({ref}`tut-ld-sensitivity`).

## The model for one marker

Put the two ingredients together. If each SNP is a susceptibility SNP with probability
$\pi_c$, the number $j$ of susceptibility SNPs among the $N^*_k$ that marker $k$ tags is
binomial. Given $j$, the marker's estimated effect is normal with a variance that has
three parts:

$$
\hat\beta_k \mid j \;\sim\; N\Bigl(0,\;\;
   \underbrace{j\,\sigma^2\,\frac{\ell_k}{N^*_k}}_{\text{signal}}
 \;+\; \underbrace{\frac{1}{n_k}}_{\text{sampling error}}
 \;+\; \underbrace{a}_{\text{inflation}} \Bigr),
\qquad j \sim \text{Binomial}(N^*_k, \pi_c) .
$$

- **Signal.** Each tagged susceptibility SNP contributes $\sigma^2$ times the average
  $r^2$ in the neighbourhood, $\ell_k/N^*_k$.
- **Sampling error.** $1/n_k$ for a standardised effect estimated from $n_k$ individuals.
- **Inflation.** A constant $a$ added to the variance of every marker, which absorbs
  population stratification and cryptic relatedness. In terms of test statistics, a null
  marker has $E[z^2] = 1 + n a$; this plays the role of the intercept of LD-score
  regression. For height $1 + \bar n a \approx 1.2$.

Averaging over $j$ gives a *mixture of normals* for each marker, with weights and
variances that depend on the marker's own $N^*_k$, $\ell_k$ and $n_k$ (equations 2 and 3
of the paper; for M3 the count $j$ becomes a pair of counts, one per cluster).

```{figure} ../_static/figures/marginal-mixture.svg
:name: fig-marginal-mixture
:alt: Four panels. The top row shows the probability of tagging zero, one, two or more susceptibility SNPs for a marker with a small neighbourhood and for one with a large neighbourhood. The bottom row shows the corresponding densities of the z statistic on a logarithmic scale, as component curves and their mixture, compared with the standard normal.

**What the model predicts for two markers**, using the two-component estimates for height
and $n = 130{,}000$. Top: the probability of tagging $j$ susceptibility SNPs. Bottom: the
density of the $z$ statistic (logarithmic scale). A marker that tags 5 SNPs (left) tags a
susceptibility SNP only 2 % of the time: its $z$ statistic is close to null, with a faint
heavy tail. A marker that tags 150 SNPs (right) tags at least one about half of the time,
and its $z$ statistic is clearly over-dispersed.
```

This is why GENESIS can estimate more than LD-score regression does. LD-score regression
uses only the *mean* of $z^2$ as a function of the LD score, which identifies the
heritability $M\pi_c\sigma^2$ but not $\pi_c$ and $\sigma^2$ separately. GENESIS uses the
whole *shape* of the distribution of $z$: few large effects produce heavy tails, many
small effects produce a broad bell, even at the same heritability.

## What is estimated

| Symbol | Name in the output | Meaning | Height, M3 |
|---|---|---|---|
| $\pi_c$ | `pic` | proportion of reference SNPs that are susceptibility SNPs | 0.0096 |
| $p_1$ | `p1` | proportion of susceptibility SNPs in cluster 1, the cluster with larger effects (M3 only) | 0.095 |
| $\sigma_1^2$, $\sigma_2^2$ | `sigmasq1`, `sigmasq2` | variances of the standardised joint effects in the two clusters (`sigmasq` in M2) | $1.4\times10^{-4}$, $2.0\times10^{-5}$ |
| $a$ | `a` | variance inflation common to all markers | $1.5\times10^{-6}$ |

Two derived quantities carry most of the interpretation:

$$
\underbrace{M\pi_c}_{\text{number of susceptibility SNPs}}
\qquad\text{and}\qquad
\underbrace{h^2 = M\pi_c\sum_h p_h\sigma_h^2}_{\text{heritability explained by the panel's SNPs}} ,
$$

where $M$ is the number of SNPs in the reference panel (1,070,777 common HapMap3 SNPs for
the bundled European reference). [Chapter 4](interpreting.md) reads them for height.

## How the model is fitted

Three ideas, each with a chapter in the theory section:

1. **Composite likelihood.** The mixture densities of all markers are multiplied as if
   the markers were independent. They are not — neighbouring markers share LD — but the
   resulting estimate is still consistent ([The statistical model](../theory/model.md)).
   It is maximised by an EM algorithm finished with Newton's method
   ([Estimation](../theory/estimation.md)).
2. **Standard errors that respect LD.** The dependence between markers is put back when
   computing uncertainty, with a "sandwich" estimator that sums the contributions of
   markers in the same neighbourhood ([Standard errors](../theory/inference.md)).
3. **Model choice.** M2 and M3 are compared with a modified BIC, and the fit of either
   can be checked against the data with a QQ plot ([Chapter 3](fitting.md)).

## What to keep in mind

:::{admonition} Assumptions and their consequences
:class: important

- **Common SNPs of the reference panel.** "Number of susceptibility SNPs" means the
  number among the panel's SNPs (HapMap3, minor-allele frequency of at least 5 %) needed
  to explain the common-variant signal, and $h^2$ is the heritability those SNPs explain.
  Rare variants are outside the model.
- **The count is a lower bound.** SNPs with very small effects cannot be told apart from
  SNPs with none at a finite sample size, and the model's summary of LD is an
  approximation. In the paper's simulations $\pi_c$ was biased downward and the variances
  upward, more so for small studies and when M2 was fitted to data that needed M3; the
  authors describe the estimated number of susceptibility SNPs as a lower bound. A
  simulation study with this package and real LD found the same: $\pi_c$ 27 % too low,
  the heritability within 3 % ([Validation](../reference/validation.md)). The
  heritability, and the distribution of the larger effects, are much more stable than
  the count.
- **Effects do not depend on allele frequency or local LD**, and susceptibility SNPs are
  scattered at random. Modest violations had little effect in the paper's simulations.
- **The LD reference matches the study population.** A mismatched reference distorts
  $N^*_k$ and $\ell_k$ and hence every estimate ([Chapter 7](other-ancestries.md)).
- **Three components need data.** The paper found M3 poorly identified for studies of
  fewer than about 25,000 individuals; the software detects and reports fits that are
  not identified ({ref}`tut-warnings`).
:::

## Notation

| Symbol | Meaning |
|---|---|
| $M$ | number of SNPs in the reference panel |
| $K$ | number of GWAS markers analysed (those also in the reference) |
| $z_k$, $n_k$ | $z$ statistic and sample size of marker $k$ |
| $\hat\beta_k = z_k/\sqrt{n_k}$ | standardised effect estimate; its sampling variance is $1/n_k$ |
| $\beta^{(J)}_m$ | joint effect of reference SNP $m$ |
| $\mathcal N_k$, $N^*_k$ | tagging neighbourhood of marker $k$ and its size |
| $\ell_k$ | LD score of marker $k$ |
| $\pi_c$ | proportion of susceptibility SNPs |
| $p_h$, $\sigma_h^2$ | weight and variance of normal component $h$ ($h = 1,\dots,H$) |
| $a$ | variance-inflation term |
| $h^2$ | heritability explained by the panel's SNPs |
| M2, M3 | the models with $H = 1$ and $H = 2$ normal components |
