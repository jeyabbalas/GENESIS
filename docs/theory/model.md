# The statistical model

GENESIS answers one question: *given the marginal association statistics of a GWAS, what
distribution of effect sizes across the genome produced them?* This chapter states the
assumptions, derives the likelihood of one marker, and explains the two approximations
that make the computation tractable — the composite likelihood and the truncation of the
latent counts.

## Data and scale

For each marker $k = 1,\dots,K$ the input is a Z statistic $z_k$ and a sample size $n_k$.
They are converted to a standardised effect estimate and its sampling variance,

$$
\hat\beta_k = \frac{z_k}{\sqrt{n_k}}, \qquad s_k^2 = \frac{1}{n_k},
$$

which is what a regression of a unit-variance phenotype on a unit-variance genotype
returns (the residual variance is very nearly one because any single SNP explains almost
nothing). For a case–control study $n_k$ is the *effective* sample size
$n_\text{case} n_\text{control}/(n_\text{case}+n_\text{control})$, which makes
$\hat\beta_k$ a log odds ratio per standard deviation of genotype; see
[binary traits](../tutorial/binary-traits.md).

The second input is an LD reference panel with $M$ SNPs. For each marker it supplies the
*tagging neighbourhood* $\mathcal N_k$ — the reference SNPs within a window (default 1 Mb)
whose squared correlation with the marker is at least a threshold (default
$r^2 \ge 0.1$), the marker itself included — together with

$$
N^*_k = |\mathcal N_k| , \qquad
\ell_k = \sum_{m \in \mathcal N_k} \Bigl[ r^2_{km} - \frac{1 - r^2_{km}}{n_\text{ref} - 2} \Bigr] ,
$$

the number of tagged SNPs and the LD score. The bracket is the bias-corrected estimate of
$\rho^2_{km}$ used by LD-score regression
([Bulik-Sullivan et al., 2015](../reference/bibliography.md#ref-bulik-sullivan2015));
$n_\text{ref}$ is the number of individuals in the panel (489 for the bundled reference).

## The joint model and the effect-size distribution

Let $Y$ be the standardised phenotype and $G_m$ the standardised genotype of reference SNP
$m$. The *joint* model is

$$
Y = \sum_{m=1}^{M} \beta^{(J)}_m\, G_m + \epsilon ,
$$

and the quantity of interest is the distribution of the joint effects $\beta^{(J)}_m$
across SNPs. GENESIS assumes they are independent draws from a point mass at zero mixed
with $H$ zero-mean normals:

$$
\beta^{(J)}_m \;\sim\; \pi_c \sum_{h=1}^{H} p_h\, N(0, \sigma_h^2) \;+\; (1 - \pi_c)\,\delta_0 .
$$ (eq-prior)

This is equation (1) of the paper. A fraction $\pi_c$ of the SNPs are *susceptibility
SNPs*; among those, a fraction $p_h$ has effects of variance $\sigma_h^2$. Two models are
fitted:

| Model | $H$ | Parameter vector $\theta$ | Meaning |
|---|---|---|---|
| **M2** | 1 | $(\pi_c,\ \sigma^2,\ a)$ | one normal for all susceptibility SNPs |
| **M3** | 2 | $(\pi_c,\ p_1,\ \sigma_1^2,\ \sigma_2^2,\ a)$ | a cluster of larger effects ($\sigma_1^2 > \sigma_2^2$) holding a fraction $p_1$ of the susceptibility SNPs |

($a$ is introduced {ref}`below <model-sampling>`.) Two functionals of
$\theta$ carry most of the interpretation:

$$
M_\text{s} = M \pi_c \quad\text{(number of susceptibility SNPs)}, \qquad
h^2 = M \pi_c \sum_{h} p_h \sigma_h^2 \quad\text{(heritability explained by the panel)} .
$$

The second follows from $\operatorname{var}(\sum_m \beta^{(J)}_m G_m) = \sum_m
E[\beta^{(J)\,2}_m]$ for standardised, independent effects.

## From joint to marginal effects

A GWAS does not estimate $\beta^{(J)}_m$. It regresses the phenotype on one SNP at a time
and therefore estimates the *marginal* effect

$$
\beta_k = \sum_{m=1}^{M} \rho_{km}\, \beta^{(J)}_m ,
$$ (eq-marginal)

where $\rho_{km}$ is the correlation (LD) between marker $k$ and SNP $m$
([Yang et al., 2012](../reference/bibliography.md#ref-yang2012)). Every marker thus
reports a weighted sum of the effects of all SNPs it is in LD with
({numref}`fig-joint-vs-marginal` in the tutorial shows this for a real region).

The distribution of $\beta_k$ implied by {eq}`eq-prior` is obtained in four steps.

**Step 1 — keep the tagged SNPs.** SNPs outside $\mathcal N_k$ have $r^2$ below the
threshold or lie outside the window; their contribution to {eq}`eq-marginal` is dropped:
$\beta_k \approx \sum_{m\in\mathcal N_k} \rho_{km}\beta^{(J)}_m$.

**Step 2 — condition on which SNPs are causal.** Attach to each SNP a label
$C_m \in \{0, 1, \dots, H\}$ (null, or the component its effect comes from), independent
across SNPs with probabilities $(1-\pi_c,\ \pi_c p_1, \dots, \pi_c p_H)$. Given the labels,
$\beta_k$ is a sum of independent zero-mean normals:

$$
\beta_k \mid C \;\sim\; N\Bigl(0,\ \sum_{h=1}^{H} \sigma_h^2 \sum_{m \in \mathcal N_k:\, C_m = h} \rho^2_{km} \Bigr).
$$

**Step 3 — replace the inner sum by its mean.** Let $N_k^{(h)}$ be the number of SNPs in
$\mathcal N_k$ with label $h$. If the labels are independent of the LD pattern — the key
assumption of the method — then, given $N_k^{(h)}$, the SNPs carrying label $h$ are a
uniformly random subset of $\mathcal N_k$ and

$$
E\Bigl[\sum_{m \in \mathcal N_k:\, C_m = h} \rho^2_{km} \,\Big|\, N_k^{(h)}\Bigr]
 = N_k^{(h)} \cdot \frac{1}{N^*_k}\sum_{m\in\mathcal N_k}\rho^2_{km}
 = N_k^{(h)}\,\frac{\ell_k}{N^*_k}.
$$

GENESIS uses this mean in place of the sum: each tagged susceptibility SNP is credited
with the *average* squared correlation $\ell_k/N^*_k$ of the neighbourhood. What is
ignored is the spread of $\rho^2_{km}$ within the neighbourhood.

**Step 4 — average over the counts.** With independent labels the counts are multinomial,

$$
\bigl(N_k^{(0)}, N_k^{(1)}, \dots, N_k^{(H)}\bigr) \sim
 \text{Multinomial}\bigl(N^*_k;\ 1-\pi_c,\ \pi_c p_1, \dots, \pi_c p_H\bigr),
$$

(binomial when $H = 1$), so that

$$
\beta_k \mid \theta \;\sim\; \sum_{(N_k^{(0)},\dots,N_k^{(H)})}
   \Pr\bigl(N_k^{(0)},\dots,N_k^{(H)}\bigr)\;
   N\Bigl(0,\ \sum_{h=1}^{H} \frac{N_k^{(h)}}{N^*_k}\,\sigma_h^2\,\ell_k\Bigr).
$$ (eq-marginal-mixture)

This is equation (2) of the paper: the marginal effect of a marker is a mixture of
zero-mean normals, indexed by *how many susceptibility SNPs of each kind the marker tags*.
The component with all counts zero is a point mass at zero — the marker tags nothing.

:::{admonition} Consistency with LD-score regression
:class: note

Taking the expectation of the variance in {eq}`eq-marginal-mixture` gives
$\operatorname{var}(\beta_k) = \pi_c\sum_h p_h\sigma_h^2\,\ell_k = h^2\ell_k/M$, hence
$E[z_k^2] = 1 + n_k a + n_k h^2 \ell_k / M$ once sampling error is added: exactly the
LD-score regression equation. LD-score regression uses only this second moment; GENESIS
uses the whole distribution, which is what identifies $\pi_c$ separately from $\sigma^2$.
The package exploits the relationship to construct
{ref}`starting values <est-starting-values>`.
:::

(model-sampling)=
## Sampling error and the inflation term

The estimate differs from the true marginal effect by sampling error, plus any systematic
inflation of the test statistics from population stratification or cryptic relatedness.
As in LD-score regression the latter is absorbed by a single variance term $a$:

$$
\hat\beta_k \mid \beta_k \;\sim\; N\bigl(\beta_k,\ s_k^2 + a\bigr).
$$

Combining this with {eq}`eq-marginal-mixture` gives the likelihood of one marker,
equation (3) of the paper. Writing $t$ for a *configuration* — a vector of counts
$J_t = (J_{t1},\dots,J_{tH})$ — it reads

$$
L_k(\theta) = \sum_{t} \pi_{kt}(\theta)\, f_{kt}(\theta), \qquad
f_{kt} = N\bigl(\hat\beta_k;\ 0,\ v_{kt}\bigr), \qquad
v_{kt} = \frac{\ell_k}{N^*_k}\sum_{h=1}^{H} J_{th}\,\sigma_h^2 \;+\; s_k^2 \;+\; a ,
$$ (eq-likelihood)

with multinomial prior weights

$$
\pi_{kt} = \frac{N^*_k!}{\bigl(N^*_k - |J_t|\bigr)!\ \prod_h J_{th}!}\;
          \prod_{h=1}^{H} (\pi_c p_h)^{J_{th}}\;(1-\pi_c)^{N^*_k - |J_t|},
\qquad |J_t| = \sum_h J_{th}.
$$ (eq-prior-weights)

On the $z$ scale, $n_k a$ is the amount by which the variance of a null marker exceeds
one — the analogue of "LD-score intercept minus one". Nothing forces the estimate of $a$
to be positive (a negative value corresponds to an intercept below one); the model only
requires $s_k^2 + a > 0$ for every marker, and the optimiser enforces
$a > -\min_k s_k^2$. Fitting with `stratification=False` fixes $a = 0$.

{numref}`fig-marginal-mixture` in the tutorial draws {eq}`eq-likelihood` for a marker with
a small and one with a large neighbourhood.

## The composite likelihood

The exact joint likelihood of $(\hat\beta_1,\dots,\hat\beta_K)$ would require the full LD
matrix and a sum over all assignments of labels to a million SNPs. GENESIS instead
multiplies the marginal likelihoods as if the markers were independent,

$$
\mathrm{CL}(\theta) = \prod_{k=1}^{K} L_k(\theta), \qquad
\ell(\theta) = \sum_{k=1}^{K} \log L_k(\theta).
$$

This is a *composite likelihood*
([Lindsay, 1988](../reference/bibliography.md#ref-lindsay1988);
[Varin, Reid & Firth, 2011](../reference/bibliography.md#ref-varin2011)). Two facts about
it organise everything that follows.

1. **Point estimates remain valid.** Each $L_k$ is the correct density of
   $\hat\beta_k$, so the score of each term has mean zero at the true parameter. The
   maximiser solves an unbiased estimating equation and is consistent regardless of the
   correlation between markers.
2. **Its curvature overstates the information.** Correlated markers are counted as if
   independent, so the inverse Hessian is too small a variance. Standard errors need the
   {ref}`sandwich estimator <inf-sandwich>`, and information criteria
   need an {ref}`effective number of parameters <inf-model-choice>`.

## Truncation and the two log-likelihoods

A marker with $N^*_k = 300$ has 301 possible counts under M2 and more than 45,000 pairs
of counts under M3. Almost all carry negligible prior mass, because the expected number of
tagged susceptibility SNPs, $N^*_k\pi_c$, is usually well below one. The sum in
{eq}`eq-likelihood` is therefore truncated at $c_0$ counts *per component* (default
$c_0 = 10$, as in the original software):

$$
\mathcal T_k = \bigl\{ J \in \{0,1,\dots,c_0\}^H :\ |J| \le N^*_k \bigr\} .
$$

The software enumerates the $T = (c_0+1)^H$ configurations once (11 for M2 and 121 for
M3 at $c_0 = 10$; {func}`genesis_gwas.core.spec.config_table`) and gives impossible ones — more causal SNPs
than tagged SNPs — a log prior of $-\infty$, hence weight exactly zero.

Truncation removes prior mass, and there are two natural ways to write the resulting
log-likelihood. Both are computed in every pass
({class}`genesis_gwas.core.likelihood.CompositeLikelihood`):

`loglik_unnorm`
: $\displaystyle \tilde\ell(\theta) = \sum_k \log \sum_{t\in\mathcal T_k} \pi_{kt} f_{kt}$ —
  the truncated sum as it stands.

`loglik`
: $\displaystyle \ell(\theta) = \tilde\ell(\theta) - \sum_k \log y_k(\theta)$ with
  $y_k = \sum_{t\in\mathcal T_k}\pi_{kt} \le 1$ the retained prior mass — the truncated
  mixture renormalised to be a proper density. This is the value the original software
  prints and uses in its model-selection criterion, and it is what `FitResult.loglik`
  reports.

They differ by $G(\theta) = -\sum_k \log y_k(\theta) \ge 0$. The distinction matters
because **the EM algorithm of the method maximises $\tilde\ell$, not $\ell$**: its
closed-form update of $\pi_c$ is the exact M-step for the unnormalised sum (shown in the
{ref}`next chapter <est-mstep-proportions>`), and the scores and information used for
standard errors are derivatives of $\tilde\ell$ as well. The estimator is thus defined by
$\partial\tilde\ell/\partial\theta = 0$, while $\ell$ is the reported fit statistic.

In the regime the method is designed for, the two coincide to several digits.
{numref}`fig-truncation-mass` shows why: the discarded mass $1 - y_k$ is below
$10^{-6}$ as long as a marker is expected to tag fewer than about one susceptibility SNP,
and real markers almost all are in that regime.

```{figure} ../_static/figures/truncation-mass.svg
:name: fig-truncation-mass
:alt: Left, discarded prior mass against the expected number of tagged susceptibility SNPs for truncation levels 10 and 20. Right, histogram of that expected number over the reference SNPs for the two height fits.

**Truncation discards almost nothing for almost all markers.** (a) Prior mass beyond the
truncation level for a binomial count, against its mean $N^*_k\pi_c$. (b) Distribution of
$N^*_k\pi_c$ over the 1,070,777 reference SNPs at the fitted height estimates.
```

For the height example, with the level held at $c_0 = 10$
(`FitResult.diagnostics["truncation"]`):

| Fit at $c_0 = 10$ | mean $1-y_k$ | max $1-y_k$ | markers with $1-y_k > 10^{-6}$ | $G = \ell - \tilde\ell$ |
|---|---|---|---|---|
| M2 ($\hat\pi_c = 0.0048$) | $4.6\times10^{-8}$ | $8.5\times10^{-4}$ | 0.07 % | 0.04 |
| M3 ($\hat\pi_c = 0.0093$) | $3.3\times10^{-6}$ | $4.2\times10^{-2}$ | 0.65 % | 3.14 |

The maximum belongs to the marker with the largest neighbourhood ($N^*_k = 710$).

A mean of $3\times10^{-6}$ looks negligible, but the discarded mass is concentrated in
the markers with the largest neighbourhoods — the markers that tag several susceptibility
SNPs — and leaving out their highest counts makes a large $\pi_c$ look less likely than
it is. The estimate of $\pi_c$ therefore *increases* when terms are added. The software
does not treat $c_0$ as fixed: after solving at $c_0 = 10$ it evaluates the mean discarded
mass at the solution and, if it exceeds $10^{-6}$, solves again with more terms. For
height this leaves M2 at $c_0 = 10$ and moves M3 to $c_0 = 14$, where the mean discarded
mass is $7\times10^{-8}$ and $\hat\pi_c$ is 2.5 % larger. The rule, and a demonstration
of what happens when the data cannot determine $\pi_c$ at all, are in
{ref}`est-adaptive-c0`. The level can also be set directly (`c0=`, `--c0`, `--c0-max`).

## Notation used in the code

| Symbol | Meaning | In the code |
|---|---|---|
| $\hat\beta_k$, $s_k^2$ | standardised effect estimate and its sampling variance | `betahat`, `varbetahat` ({class}`genesis_gwas.ModelData`) |
| $N^*_k$, $\ell_k$ | number of tagged SNPs, LD score | `nstar`, `ldscore` |
| $r_k = \ell_k / N^*_k$ | LD score per tagged SNP | `r` (in `core/kernels.py`) |
| $M$ | number of reference SNPs | `M` |
| $\pi_c$ | proportion of susceptibility SNPs | `pic` |
| $p_1$ | share of susceptibility SNPs in cluster 1 (M3) | `p1` |
| $\sigma^2$; $\sigma_1^2, \sigma_2^2$ | effect-size variances | `sigmasq`; `sigmasq1`, `sigmasq2` |
| $a$ | variance inflation | `a` |
| $J_t$, $T$ | configuration (latent counts), number of configurations | `J`, `n_terms` |
| $c_t = \sum_h J_{th}\sigma_h^2$ | effect variance contributed by configuration $t$ | `cw`, `ce` |
| $v_{kt} = r_k c_t + s_k^2 + a$ | mixture variance | `v` |
| $c_0$ | truncation level | `c0` |
| $\tilde\ell$, $\ell$ | unnormalised / renormalised composite log-likelihood | `loglik_unnorm`, `loglik` |

The paper writes $s_k$ for the standard error (so its variance term is $s_k^2$), $\ell_k$
for the LD score and $N_k^*$ for the number of tagged SNPs; the correspondence is
one-to-one.

## What the model does not capture

These are properties of the method, stated in the paper's discussion, not of this
implementation.

- **Vanishing effects are invisible.** A SNP whose effect is far smaller than the sampling
  error is indistinguishable from a null SNP at any realistic sample size. Estimates of
  $\pi_c$ are therefore best read as a *lower bound* on the number of susceptibility
  SNPs — those with effects large enough to leave a trace — and they tend to increase with
  the sample size of the study.
- **The marginal model is an approximation under real LD.** Step 3 of the derivation
  replaces each marker's individual $r^2$ values by their average. Data generated with
  real LD are therefore not exactly distributed as the model assumes, even when the
  effect-size distribution is. In a simulation study carried out with this package (100
  genome-wide data sets with the LD of the reference; truth $\pi_c = 0.005$,
  $\sigma^2 = 6\times10^{-5}$, $n = 130{,}000$) the mean estimate of $\pi_c$ was 27 %
  below the truth and that of $\sigma^2$ 35 % above it, while the heritability was within
  3 %. Data drawn from the model's own marginal distribution
  ({func}`genesis_gwas.simulate_sumstats`) with the same parameters do not show this:
  three replicates gave $\hat\pi_c$ between 0.00499 and 0.00508. The bias therefore comes
  from the approximation, not from the estimator. It has the pattern of the paper's own
  simulations — proportions biased downward, variances upward, their product nearly
  unbiased. [Validation](../reference/validation.md) has the details.
- **Effects are assumed independent of LD and allele frequency.** If SNPs in low-LD or
  low-frequency regions carry systematically larger effects
  ([Gazal et al., 2017](../reference/bibliography.md#ref-gazal2017);
  [Speed et al., 2017](../reference/bibliography.md#ref-speed2017)), step 3 above is
  violated. The paper's simulations suggest the overall shape of the effect-size
  distribution is robust to moderate violations.
- **The tagging definition matters.** $N^*_k$ and $\ell_k$ depend on the window and the
  $r^2$ threshold, and so do the estimates — the number of susceptibility SNPs more than
  the heritability. The tutorial
  {ref}`quantifies this <tut-ld-sensitivity>` for the
  height example.
- **Only the reference panel's SNPs are modelled.** $\pi_c$ is a proportion of the $M$
  panel SNPs, and $h^2$ is the heritability they explain. Rare variants and SNPs not
  tagged by the panel are outside the model.
