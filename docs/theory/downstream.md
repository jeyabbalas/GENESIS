# Downstream quantities

Once the effect-size distribution has been estimated, several quantities of practical
interest follow from it by calculation or simulation. This chapter derives each one and
says which assumptions it adds to those of the [model](model.md). The corresponding
functions live in `genesis_gwas.analysis`; the tutorial shows them
[in use](../tutorial/downstream.md).

Throughout, $f(\beta) = \sum_h p_h\,N(\beta; 0, \sigma_h^2)$ is the fitted density of the
joint effect of a susceptibility SNP, and $h^2 = M\pi_c\sum_h p_h\sigma_h^2$.

## Effect-size distribution

Two direct read-outs of the fit need no further assumption.

The **density** $f(\beta)$ itself ({func}`genesis_gwas.effect_density`), plotted to
compare traits or models, and the **expected number of SNPs with effects in a range**
({func}`genesis_gwas.n_in_interval`):

$$
N(l, u) = M\pi_c\,\Pr(l < |\beta| < u)
        = 2M\pi_c\sum_{h} p_h\Bigl[\Phi\Bigl(-\frac{l}{\sigma_h}\Bigr) - \Phi\Bigl(-\frac{u}{\sigma_h}\Bigr)\Bigr],
\qquad 0 \le l < u \le \infty .
$$

Counts in the *tail* (say, effects above some threshold) are more stable across sample
sizes and model choices than the total $M\pi_c$, because they do not depend on how many
near-zero effects the model manages to separate from the null — the reason the paper
recommends comparing traits by them.

(down-projection)=
## Projections for future studies

### Definitions

A future GWAS of (effective) sample size $n$ estimates the effect of a susceptibility SNP
with $z$ statistic $W = \sqrt n\,\beta + Z$, $Z\sim N(0,1)$, and declares it at level
$\alpha$ if $|W| > c_\alpha = \Phi^{-1}(1-\alpha/2)$. Its power is

$$
\operatorname{pow}_\alpha(\beta) = 1 - \Phi\bigl(c_\alpha - \sqrt n\,\beta\bigr) + \Phi\bigl(-c_\alpha - \sqrt n\,\beta\bigr).
$$

The paper projects the expected number of discoveries and the fraction of heritability
they explain as

$$
E(\mathrm{ND}_\alpha) = M\pi_c\int \operatorname{pow}_\alpha(\beta)\, f(\beta)\,d\beta,
\qquad
E(\mathrm{GV}_\alpha) = \frac{M\pi_c}{h^2}\int \beta^2 \operatorname{pow}_\alpha(\beta)\, f(\beta)\,d\beta .
$$ (eq-projection)

Two idealisations are built in: each susceptibility SNP is tested by itself, as if
genotyped and free of LD with other susceptibility SNPs; and "discoveries" are
susceptibility SNPs, not loci. The projections are best read as how the yield *scales*
with sample size.

### Closed forms

The original software evaluates the integrals in {eq}`eq-projection` by adaptive
quadrature for every sample size. They have closed forms, which
{func}`genesis_gwas.project` uses.

Take one normal component, $\beta\sim N(0,\sigma^2)$. Then $W\sim N(0,\tau^2)$ with
$\tau^2 = 1 + n\sigma^2$, and the first integral is simply a normal tail:

$$
\int \operatorname{pow}_\alpha(\beta)\,N(\beta;0,\sigma^2)\,d\beta = \Pr(|W| > c_\alpha) = 2\,\Phi(-t),
\qquad t = \frac{c_\alpha}{\tau}.
$$

For the second, $(\beta, W)$ is bivariate normal with
$\operatorname{cov}(\beta, W) = \sqrt n\,\sigma^2$, so
$E[\beta\mid W] = \sqrt n\,\sigma^2 W/\tau^2$ and
$\operatorname{var}(\beta\mid W) = \sigma^2/\tau^2$, giving
$E[\beta^2\mid W] = \sigma^2/\tau^2 + n\sigma^4W^2/\tau^4$. With
$E[V^2\mathbb 1\{|V|>t\}] = 2\{\Phi(-t) + t\varphi(t)\}$ for a standard normal $V = W/\tau$,

$$
\int \beta^2 \operatorname{pow}_\alpha(\beta)\,N(\beta;0,\sigma^2)\,d\beta
 = E\bigl[\beta^2\,\mathbb 1\{|W|>c_\alpha\}\bigr]
 = 2\sigma^2\Bigl[\Phi(-t) + \frac{n\sigma^2}{1+n\sigma^2}\,t\,\varphi(t)\Bigr].
$$ (eq-power-integrals)

Summing over components with weights $p_h$ gives both projections exactly, for any vector
of sample sizes at once ({func}`genesis_gwas.analysis.projection.power_integrals`). The
results agree with the original's quadrature to its integration tolerance (relative
differences below $2\times10^{-11}$ in the test fixtures).

As $n\to\infty$, $t\to0$: every susceptibility SNP is eventually discovered and all of
$h^2$ explained. The approach to that limit is governed by $n\sigma_h^2$, which is why
traits with the same heritability but many small effects need far larger studies.

### Uncertainty

Intervals are obtained as in the original, by simulation: draw parameter vectors from a
normal approximation on a transformed scale, recompute {eq}`eq-projection` for each,
add binomial sampling variation to the number of discoveries
($\text{Binomial}(M, \mathrm{ND}/M)$), and take empirical quantiles. Two details differ
from the original, both to avoid failures: the inflation parameter $a$ — on which the
projection does not depend, and whose logarithm does not exist when its estimate is
negative — is left out; and $p_1$ is drawn on the logit scale rather than the log scale,
which could produce proportions above one.

(down-prs)=
## Performance of polygenic scores

### The score and its two moments

Following [Chatterjee et al. (2013)](../reference/bibliography.md#ref-chatterjee2013),
consider a polygenic score built from a GWAS of size $N$ by including every SNP whose
statistic exceeds a threshold,

$$
\hat S = \sum_{k:\ |z_k| > c} \hat\beta_k\,G_k ,
$$

over $M$ independent SNPs of which $M_1$ are susceptibility SNPs with effects drawn from
$f$. The true genetic value is $S = \sum_k \beta_kG_k$. For independent standardised
genotypes,

$$
\operatorname{cov}(\hat S, S) = \sum_k E\bigl[\beta_k\hat\beta_k\,\mathbb 1_k\bigr],
\qquad
\operatorname{var}(\hat S) = \sum_k E\bigl[\hat\beta_k^2\,\mathbb 1_k\bigr],
$$

with $\mathbb 1_k$ the indicator that SNP $k$ is selected. Writing
$\hat\beta = \beta + Z/\sqrt N$ and using the bivariate normality of $(\beta, W)$ and
$(Z, W)$ as above, with $\tau_h^2 = 1 + N\sigma_h^2$ and $t_h = c/\tau_h$, each
susceptibility SNP of component $h$ contributes

$$
\begin{aligned}
E[\beta^2\,\mathbb 1] &= 2\sigma_h^2\Bigl[\Phi(-t_h) + \tfrac{N\sigma_h^2}{\tau_h^2}t_h\varphi(t_h)\Bigr], &
E[\beta e\,\mathbb 1] &= \frac{2\sigma_h^2\,t_h\varphi(t_h)}{\tau_h^2}, \\
E[e^2\,\mathbb 1] &= \frac{1}{N}\Bigl[2\Phi(-t_h) + \frac{2t_h\varphi(t_h)}{\tau_h^2}\Bigr], &&
\end{aligned}
$$

where $e = Z/\sqrt N$ is the estimation error, and each null SNP contributes
$E[e^2\mathbb 1] = \{2\Phi(-c) + 2c\varphi(c)\}/N$ to the variance only. Collecting terms,

$$
\mu(c) = \frac{\operatorname{cov}(\hat S,S)}{\operatorname{sd}(\hat S)}
       = \frac{C_1 + C_4}{\sqrt{C_1 + C_2 + C_3 + C_4 + C_5}},
$$ (eq-mu)

$$
\begin{aligned}
C_1 &= M_1\textstyle\sum_h p_h\,2\sigma_h^2\bigl[\Phi(-t_h)+\tfrac{N\sigma_h^2}{\tau_h^2}t_h\varphi(t_h)\bigr], &
C_2 &= M_1\textstyle\sum_h p_h\,2\Phi(-t_h)/N, \\
C_3 &= M_1\textstyle\sum_h p_h\,2t_h\varphi(t_h)/N, &
C_4 &= M_1\textstyle\sum_h p_h\,2\sigma_h^2t_h\varphi(t_h)/\tau_h^2, \\
C_5 &= (M-M_1)\,\{2\Phi(-c)+2c\varphi(c)\}/N . &&
\end{aligned}
$$

(The grouping into $C_2$, $C_3$, $C_4$ is that of the original code. Using
$\tau_h^2 = 1 + N\sigma_h^2$ one checks that $C_1 + C_2 + C_3 + C_4$ equals
$M_1\sum_h p_h\{E[\beta^2\mathbb 1] + 2E[\beta e\mathbb 1] + E[e^2\mathbb 1]\}$, the
contribution of the susceptibility SNPs to $\operatorname{var}(\hat S)$.)
{func}`genesis_gwas.analysis.prediction.polyrisk_mu` evaluates {eq}`eq-mu`.

### From $\mu$ to accuracy

$\mu$ is the covariance between the score and the genetic value, per standard deviation of
the score. Three accuracy measures follow.

Quantitative trait
: With the trait standardised to unit variance, $\operatorname{cov}(\hat S, Y) =
  \operatorname{cov}(\hat S, S)$, so $\mu$ is the correlation between score and trait and
  **$\mu^2$ is the expected predictive $R^2$**. It cannot exceed $h^2$.

Binary trait, discrimination
: Under a log-linear risk model for a disease that is not too common, cases are an
  exponentially tilted sample of the population, which shifts the mean of the score by
  $\operatorname{cov}(\hat S, S)$ and leaves its variance unchanged to first order. The
  standardised mean difference between cases and controls is therefore $\mu$, and

  $$
  \mathrm{AUC} = \Phi\bigl(\mu/\sqrt2\bigr).
  $$

Binary trait, risk stratification
: The calibrated risk score is $E[S\mid\hat S] =
  \{\operatorname{cov}(\hat S,S)/\operatorname{var}(\hat S)\}\,\hat S$, whose standard
  deviation is $\operatorname{cov}(\hat S,S)/\operatorname{sd}(\hat S) = \mu$. If the log
  relative risk in the population is $N(-s^2/2, s^2)$ (so that the average relative risk
  is one), it is $N(s^2/2, s^2)$ among cases, and the proportions of the population and
  of cases whose risk is at least $k$ times the average are

  $$
  \mathrm{PPI}_k = 1 - \Phi\Bigl(\frac{\log k}{s} + \frac s2\Bigr), \qquad
  \mathrm{PCI}_k = 1 - \Phi\Bigl(\frac{\log k}{s} - \frac s2\Bigr), \qquad s = \mu .
  $$

:::{admonition} A difference from the original code
:class: important

The original `polyriskpredict()` uses $s = \sqrt\mu$ in the last formulas while using
$\mu$ itself in the AUC. The two are inconsistent — $\mu$ is homogeneous of degree one in
the effect sizes, as a standard deviation should be — and since $\mu < 1$ in practice,
$\sqrt\mu > \mu$ overstates the spread of risk. The default here is $s = \mu$;
`polyrisk_predict(..., legacy=True)` reproduces the original. AUC and the optimal
threshold are unaffected. See [Differences from the original](../reference/differences.md).
:::

### Choice of threshold and of $M$

`threshold="gwas"` evaluates $\mu$ at genome-wide significance. `threshold="optimum"`
maximises $\mu(c)$ over $c$, exactly as the original does: a 1000-point logarithmic grid
on $[10^{-8}, 7]$ followed by Brent's method between the neighbours of the best grid
point. The optimal threshold is typically far more liberal than genome-wide significance,
because including many true but individually unconvincing effects helps more than the
added noise hurts.

The theory assumes independent SNPs. `M` is therefore the number of *independent* SNPs —
the paper uses 200,000 — while `M1` is the estimated number of susceptibility SNPs,
$M_\text{panel}\pi_c$. By default `M` is the panel size of the fit; pass `M=200_000` to
follow the paper's convention. The choice matters for the optimal-threshold curve at
small sample sizes, where the noise from null SNPs dominates.

(down-posterior)=
## Posterior inference for individual markers

The fitted model is a prior for the marginal effect of each marker, so Bayes' rule gives
marker-level summaries — an empirical-Bayes use of the fit
([Efron & Tibshirani, 2002](../reference/bibliography.md#ref-efron2002);
[Stephens, 2017](../reference/bibliography.md#ref-stephens2017)).

Conditional on configuration $t$ the marginal effect is $N(0, \delta_{kt})$ with
$\delta_{kt} = r_kc_t$ (a point mass at zero for $t = 0$), and the estimate adds noise of
variance $s_k^2 + a$. The posterior configuration weights $w_{kt}$ are those of the
[E-step](estimation.md), and within a configuration the posterior of $\beta_k$ is normal
with mean $\hat\beta_k\,\delta_{kt}/(\delta_{kt}+s_k^2+a)$ and variance
$\delta_{kt}(s_k^2+a)/(\delta_{kt}+s_k^2+a)$. Hence
({func}`genesis_gwas.posterior_marginal`)

$$
\mathrm{lfdr}_k = \Pr(\beta_k = 0\mid\hat\beta_k) = w_{k0}, \qquad
E[\beta_k\mid\hat\beta_k] = \hat\beta_k\sum_t w_{kt}\,\frac{\delta_{kt}}{\delta_{kt}+s_k^2+a},
$$ (eq-posterior)

and the posterior variance follows from the second moment
$\sum_t w_{kt}\bigl[\kappa_{kt}^2\hat\beta_k^2 + \kappa_{kt}(s_k^2+a)\bigr]$, where
$\kappa_{kt} = \delta_{kt}/(\delta_{kt}+s_k^2+a)$ is the shrinkage factor of
configuration $t$.

Two properties are worth keeping in mind:

- The "null" here is that the marker tags **no** susceptibility SNP ($\beta_k = 0$), not
  that the marker itself is not causal. A marker in a large LD block has a small prior
  probability of tagging nothing, and correspondingly a smaller local false-discovery
  rate at the same $z$.
- Markers are treated one at a time; the posterior does not use the statistics of
  neighbouring markers, so it is not a fine-mapping tool.

The original `postmarginal()` omits $a$ from the noise variance. The default here
includes it, consistent with the likelihood under which the parameters were estimated;
`include_inflation=False` reproduces the original.

(down-simulation)=
## Simulation-based quantities

### Model-based QQ plot

The observed distribution of P-values is compared with the one expected under the fitted
model, obtained by simulation ({func}`genesis_gwas.qq_simulation`). In each replicate,
for every marker,

$$
z_k = \sqrt{n_k}\,(\beta_k + \nu_k) + e_k, \qquad
\beta_k \sim N\Bigl(0,\ r_k\textstyle\sum_h J_h\sigma_h^2\Bigr), \qquad \nu_k\sim N(0, a),
$$

with the counts $J$ drawn from the marker's truncated multinomial prior, and
$e = (e_1,\dots,e_K)$ one replicate of **LD-correlated null $z$ scores**. The latter are
obtained once per reference panel by running a GWAS of a random phenotype on the panel's
own genotypes, which reproduces the correlation between test statistics of neighbouring
markers; the bundled reference carries the 100 replicates of the original software. The
sorted $-\log_{10}P$ values are averaged over replicates to give the expected curve, and
their point-wise quantiles give the envelope (80 % by default, as in the paper's Fig. 1).
The genomic-control factor of each replicate gives $\lambda_\text{fit}$.

Note what is and is not correlated in this scheme: the estimation errors are, the
simulated *signals* $\beta_k$ are drawn independently for every marker, exactly as in the
original. The expected curve is unaffected by that, but the envelope does not reflect the
clustering of associated markers around a shared causal SNP and is narrower than it
would be under the LD-aware scheme below. The envelopes also come out narrower than those
of the original software, which adds a single common draw of $\nu$ to *all* markers of a
replicate instead of independent draws — an error that shifts every $z$ statistic of a
replicate by the same amount; see
[Differences from the original](../reference/differences.md).

### LD-aware simulation of a GWAS

{func}`genesis_gwas.simulate_ld_sumstats` implements the simulation design of the
paper's Methods on a real reference panel:

1. draw the joint effect of every one of the $M$ reference SNPs from the
   [effect-size distribution](model.md);
2. form marginal effects through the tagging neighbourhoods,
   $\beta_k = \sum_{m\in\mathcal N_k}|r_{km}|\,\beta^{(J)}_m$;
3. add $\nu_k\sim N(0,a)$ and an LD-correlated null replicate scaled by $1/\sqrt n$.

The absolute value in step 2 is forced by the data: the reference stores $r^2$, as the
original does, so the sign of $r$ is unavailable. For a single marker the distribution of
$\beta_k$ is unchanged (the joint effects are symmetric about zero); the correlation
between the *signals* of two markers that are in negative LD is not reproduced.

{func}`genesis_gwas.future_gc` uses the same generator to predict the genomic-control
factor $\lambda_{GC} = \operatorname{median}(z^2)/0.4549$ that a future study of size $n$
would show purely because of polygenicity. {func}`genesis_gwas.simulate_sumstats` is the
simpler generator that draws each marker independently from the composite model itself;
it is the right tool for checking that the estimator recovers known parameters.

(down-liability)=
## Heritability on other scales

For a binary trait analysed with effective sample sizes, $h^2$ is on the log-odds-ratio
scale. With $P$ the proportion of cases in the study, $K$ the population prevalence and
$\varphi(\Phi^{-1}(1-K))$ the normal density at the liability threshold
([Lee et al., 2011](../reference/bibliography.md#ref-lee2011);
{func}`genesis_gwas.h2_transfer`):

$$
h^2_\text{obs} = h^2\,P(1-P), \qquad
h^2_\text{liab} = h^2_\text{obs}\,\frac{K^2(1-K)^2}{\varphi\bigl(\Phi^{-1}(1-K)\bigr)^2\,P(1-P)} .
$$

Standard errors are scaled by the same factors. These are the formulas of the original
`h2transfer()`; they treat $K$ and $P$ as known.
