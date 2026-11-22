# Standard errors and model choice

The composite likelihood treats correlated markers as independent, so its curvature
overstates how much information the data hold. This chapter derives the quantities needed
to correct for that — per-marker scores, the observed information, and the sandwich
covariance that accounts for linkage disequilibrium — and then the standard errors of
derived quantities and the criterion used to choose between M2 and M3.

(inf-scores)=
## Scores and observed information

All derivatives are those of the truncated composite log-likelihood
$\tilde\ell(\theta) = \sum_k \log \tilde L_k$, with
$\tilde L_k = \sum_t \pi_{kt} f_{kt}$ (see [The statistical model](model.md)); this is the
function whose stationary point defines the estimator.

For one configuration, let
$r_{kt} = \partial \log(\pi_{kt} f_{kt}) / \partial\theta$. With
$g_1 = \tfrac12(\hat\beta_k^2/v_{kt}^2 - 1/v_{kt})$ and $r_k = \ell_k/N^*_k$ as in the
[previous chapter](estimation.md), its components are

| Parameter | Component of $r_{kt}$ |
|---|---|
| $\pi_c$ | $\dfrac{\lvert J_t\rvert}{\pi_c} - \dfrac{N^*_k - \lvert J_t\rvert}{1-\pi_c}$ |
| $p_1$ (M3 only) | $\dfrac{J_{t1}}{p_1} - \dfrac{J_{t2}}{1-p_1}$ |
| $\sigma_h^2$ | $g_1(v_{kt})\; r_k\, J_{th}$ |
| $a$ | $g_1(v_{kt})$ |

Differentiating $\log\sum_t \pi_{kt}f_{kt}$ and using the posterior weights
$w_{kt} = \pi_{kt}f_{kt}/\tilde L_k$ gives the score and the observed information of
marker $k$:

$$
S_k(\theta) = \frac{\partial \log\tilde L_k}{\partial\theta} = \sum_t w_{kt}\, r_{kt},
$$ (eq-score)

$$
I_k(\theta) = -\frac{\partial^2 \log\tilde L_k}{\partial\theta\,\partial\theta^\top}
 = S_k S_k^\top - \sum_t w_{kt}\Bigl[r_{kt} r_{kt}^\top + \frac{\partial r_{kt}}{\partial\theta^\top}\Bigr].
$$ (eq-information)

The matrix $\partial r_{kt}/\partial\theta^\top$ is block diagonal: for the proportions
its only non-zero entries are
$-|J_t|/\pi_c^2 - (N^*_k - |J_t|)/(1-\pi_c)^2$ and $-J_{t1}/p_1^2 - J_{t2}/(1-p_1)^2$;
for the variance parameters it is $g_2(v_{kt})$ times the outer product of
$(r_kJ_{t1},\dots,r_kJ_{tH},1)$ with itself, where
$g_2 = \tfrac12(1/v^2 - 2\hat\beta^2/v^3)$.

The scores are stored for every marker (a $K\times P$ matrix) and the information is
summed, $I(\theta) = \sum_k I_k(\theta)$. Both come from a single pass
({func}`genesis_gwas.core.kernels.score_pass`,
{meth}`genesis_gwas.core.likelihood.CompositeLikelihood.score_info`). As in the EM pass,
every entry of {eq}`eq-information` factorises into a function of the configuration times
a sum over markers of $w$, $wg_1$, $wg_1^2$ or $wg_2$ weighted by $1$, $N^*_k$, $r_k$ or
their products; the kernel returns thirteen such vectors and the $P\times P$ matrices are
assembled on the host.

The analytic scores and information agree with numerical differentiation of the
log-likelihood to about nine significant digits, and with the original C++ (`SS`, `I`,
`SS3`, `I3`) to relative differences of order $10^{-13}$; both checks are part of the
test suite.

:::{note}
The original three-component routine `Ik3` contains the line `rr(2,0) = rr(2,0);` — a
self-assignment where the $(\sigma_1^2, \pi_c)$ entry of $r r^\top$ should be set — which
leaves that entry zero and makes its information matrix, and hence its published
covariance matrices, asymmetric. This implementation computes the symmetric matrix;
`legacy=True` reproduces the original. See
[Differences from the original](../reference/differences.md).
:::

(inf-sandwich)=
## The sandwich estimator

### Why the inverse information is not enough

Let $U(\theta) = \sum_k S_k(\theta)$ be the composite score. The estimator solves
$U(\hat\theta) = 0$, and a first-order expansion around the true value $\theta_0$ gives
$\hat\theta - \theta_0 \approx I(\theta_0)^{-1} U(\theta_0)$, hence

$$
\operatorname{var}(\hat\theta) \approx I^{-1}\, \operatorname{var}\bigl(U(\theta_0)\bigr)\, I^{-1} .
$$

For an ordinary likelihood, $\operatorname{var}(U) = I$ and the middle factor cancels.
Here it does not, for two reasons: the scores of markers in LD are correlated, and the
model for each marker is only an approximation. The general form is the
*Godambe information* or sandwich
([Godambe, 1960](../reference/bibliography.md#ref-godambe1960);
[White, 1982](../reference/bibliography.md#ref-white1982);
[Varin, Reid & Firth, 2011](../reference/bibliography.md#ref-varin2011)).

### Estimating the variance of the score

$\operatorname{var}(U) = \sum_k\sum_{k'} \operatorname{cov}(S_k, S_{k'})$ cannot be
estimated from products of all pairs — that sum is identically
$U U^\top = 0$ at the estimate. But scores of markers far apart are uncorrelated, so only
nearby pairs need to be kept, as in the window estimators developed for correlated data
([Heagerty & Lumley, 2000](../reference/bibliography.md#ref-heagerty2000);
[Lumley & Heagerty, 1999](../reference/bibliography.md#ref-lumley1999)). GENESIS defines
"nearby" by the same tagging relation as the LD score. With
$\mathcal N_k$ now restricted to markers *present in the GWAS*, and including $k$ itself,

$$
\bar U_k = \sum_{k'\in\mathcal N_k} S_{k'}, \qquad
\hat J = \sum_{k=1}^{K} S_k\, \bar U_k^\top, \qquad
\widehat{\operatorname{var}}(\hat\theta) = I(\hat\theta)^{-1}\,\hat J\; I(\hat\theta)^{-1}.
$$ (eq-sandwich)

This is equation (4) of the paper. Standard errors are the square roots of its diagonal.

```{figure} ../_static/figures/sandwich-neighbourhood.svg
:name: fig-sandwich-neighbourhood
:alt: Left, a matrix plot of which pairs among 300 consecutive markers on chromosome 3 tag each other, showing a block structure along the diagonal. Right, the positions of the markers in the neighbourhood of one marker along the chromosome.

**Which pairs of markers enter the "meat" of the sandwich.** (a) The tagging relation
($r^2\ge0.1$ within 1 Mb) among 300 consecutive reference SNPs on chromosome 3: a point at
$(k, k')$ means the product $S_kS_{k'}^\top$ contributes to $\hat J$. The blocks are LD
blocks. (b) The neighbourhood of one marker (row 150 of the matrix), whose scores are
summed to $\bar U_k$.
```

Because the tagging relation is symmetric, $\hat J$ is symmetric. It is computed as
$S^\top (A S)$ with $A$ the sparse 0/1 neighbourhood matrix, by gathering rows of the
score matrix block by block ({func}`genesis_gwas.core.variance.neighborhood_sums`), in
about a second for a million markers. Reference SNPs that are tagged but absent from the
GWAS simply do not contribute.

Inverting $I$ needs care only because the parameters live on very different scales
($\pi_c\sim10^{-2}$, $\sigma^2\sim10^{-5}$, $a\sim10^{-6}$), which makes the raw matrix
badly conditioned although the problem is not. It is rescaled to unit diagonal before
inversion
({func}`genesis_gwas.core.variance.scaled_inverse`); the condition number of the *scaled*
matrix is what `diagnostics["information_condition"]` reports.

:::{admonition} A difference from the original code
:class: important

The original R code adds each marker's own score to $\bar U_k$ a second time (its tagging
lists already contain the marker), so it computes $\hat J + \sum_k S_kS_k^\top$. The
default here follows {eq}`eq-sandwich`; `sandwich_double_count=True` or `legacy=True`
reproduces the original. Every fit also stores the standard errors, $d_s$ and BIC under
the *other* convention in `diagnostics["sandwich_alt"]`. For the height example the
original convention gives standard errors 2–5 % larger and $d_s$ of 96.9 instead of 93.4
(M2) and 108.9 instead of 104.3 (M3).
:::

If no neighbourhoods are available (data constructed by hand), markers are treated as
independent, $\hat J = \sum_k S_kS_k^\top$, with a warning: for real GWAS data that
understates the standard errors severely.

(inf-delta)=
## Derived quantities

Quantities of interest are smooth functions $g(\theta)$ of the parameters. Their standard
errors follow from the delta method,
$\operatorname{se}\{g(\hat\theta)\} = \sqrt{\nabla g^\top\, \widehat{\operatorname{var}}(\hat\theta)\, \nabla g}$
({func}`genesis_gwas.core.derived.derived_quantities`). With parameters ordered
$(\pi_c, \sigma^2, a)$ for M2 and $(\pi_c, p_1, \sigma_1^2, \sigma_2^2, a)$ for M3:

| Quantity | Definition | Gradient |
|---|---|---|
| number of susceptibility SNPs | $M\pi_c$ | $(M, 0, \dots)$ |
| heritability, M2 | $M\pi_c\sigma^2$ | $(M\sigma^2,\ M\pi_c,\ 0)$ |
| heritability, M3 | $M\pi_c\{p_1\sigma_1^2 + (1-p_1)\sigma_2^2\}$ | $\bigl(M\{p_1\sigma_1^2+(1-p_1)\sigma_2^2\},\ M\pi_c(\sigma_1^2-\sigma_2^2),\ M\pi_cp_1,\ M\pi_c(1-p_1),\ 0\bigr)$ |
| SNPs in cluster 1 | $M\pi_cp_1$ | $(Mp_1,\ M\pi_c,\ 0, 0, 0)$ |
| SNPs in cluster 2 | $M\pi_c(1-p_1)$ | $(M(1-p_1),\ -M\pi_c,\ 0, 0, 0)$ |
| heritability of cluster 1 | $M\pi_cp_1\sigma_1^2$ | $(Mp_1\sigma_1^2,\ M\pi_c\sigma_1^2,\ M\pi_cp_1,\ 0,\ 0)$ |
| heritability of cluster 2 | $M\pi_c(1-p_1)\sigma_2^2$ | $(M(1-p_1)\sigma_2^2,\ -M\pi_c\sigma_2^2,\ 0,\ M\pi_c(1-p_1),\ 0)$ |
| sibling relative risk | $\exp(h^2/2)$ | $\tfrac12\exp(h^2/2)\,\nabla h^2$ |

The sibling relative risk is meaningful for binary traits, where $h^2$ is on the
log-odds-ratio scale: under a log-normal risk model the risk to a sibling of a case is
$\exp(h^2/2)$ times the population risk. Conversion of $h^2$ to the observed and liability
scales is described under [Downstream quantities](downstream.md).

Estimates of $\pi_c$ and $\sigma^2$ are strongly negatively correlated — the ridge that
slows the EM — so the heritability is much better determined than either factor: for the
height M2 fit the relative standard errors are 10.6 % for $\pi_c$, 8.5 % for $\sigma^2$
and 4.7 % for $h^2$.

**Labelling of clusters.** The two normal components of M3 are exchangeable:
$(p_1,\sigma_1^2,\sigma_2^2)$ and $(1-p_1,\sigma_2^2,\sigma_1^2)$ describe the same
distribution. The paper defines cluster 1 as the one with the larger variance. If a fit
ends with $\sigma_1^2 < \sigma_2^2$, the labels are swapped and the covariance matrix is
transformed accordingly ({func}`genesis_gwas.core.derived.canonicalize_m3`), so "cluster
1" always means the larger-effect cluster in the output.

(inf-model-choice)=
## Model choice

M3 nests M2, so its likelihood is never lower; the question is whether the improvement is
worth two more parameters. For composite likelihoods the usual parameter count is replaced
by the *effective number of parameters*

$$
d_s = \operatorname{tr}\bigl(I^{-1}\hat J\bigr),
$$

which equals the number of parameters when the scores are independent and the model is
correct ($\hat J \approx I$), and grows with the amount of correlation between markers
([Varin, Reid & Firth, 2011](../reference/bibliography.md#ref-varin2011);
[Gao & Song, 2010](../reference/bibliography.md#ref-gao2010)). For the height data
$d_s \approx 93$ for the three parameters of M2: each parameter is "counted" about thirty
times because neighbouring markers carry largely the same information.

The criteria are ({func}`genesis_gwas.core.variance.information_criteria`)

$$
\mathrm{AIC} = -2\ell + 2d_s, \qquad
\mathrm{BIC} = -2\ell + d_s\log\bar n + 2\gamma\, d_s\log 5 ,
$$ (eq-bic)

where $\ell$ is the renormalised composite log-likelihood, $\bar n$ the mean sample size
and $\gamma$ = `bic_gamma` (default 0.5). The last term has the form of an extended-BIC
penalty ([Chen & Chen, 2008](../reference/bibliography.md#ref-chen2008)) and is
implemented exactly as in the original software (`2*BIC.gamma*log(5^ds)`); the paper
calls the result the "modified BIC". The model with the smaller BIC is preferred
({func}`genesis_gwas.select_model`).

For height: $\mathrm{BIC}_{M2} - \mathrm{BIC}_{M3} = 1044$, an overwhelming preference
for the three-component model, in line with the paper's finding for anthropometric
traits.

(inf-bounds)=
## Bounds and non-identified fits

Standard asymptotics assume the estimate lies in the interior of the parameter space and
the model is identified. Two situations violate this and are handled explicitly.

**A parameter on a bound.** If the M3 fit ends with $\pi_c$ or $p_1$ on its bound, or
with a variance component at zero, that parameter is treated as fixed: it is removed from
$I$ and $\hat J$ before inversion, its standard error is reported as `NaN`, and the
standard errors of the remaining parameters are conditional on the constraint. The
parameters concerned are listed in `diagnostics["at_bound"]`.

**A three-component model that is not identified.** When the data do not support two
clusters, the M3 likelihood is maximised where the two variances coincide (then $p_1$ is
arbitrary) or where one of them vanishes. The information matrix is singular in that
direction and nothing about $p_1$ or the cluster-specific quantities can be trusted. The
software flags a fit as `degenerate` when
$|\sigma_1^2 - \sigma_2^2| \le 0.01\max(\sigma_1^2,\sigma_2^2)$ or a variance is below
$10^{-11}$, attaches a warning and reports $d_s$, AIC and BIC as unavailable, since
$\operatorname{tr}(I^{-1}\hat J)$ has no meaning for a singular $I$.
{func}`genesis_gwas.select_model` never selects such a fit when a non-degenerate
alternative exists. The packaged chromosome-22 example is such a case: one chromosome
cannot separate two clusters, and M2 is selected.

In all other cases a matrix $I$ that is not positive definite at the estimate triggers a
warning and `diagnostics["information_pd"] = False`; it means the optimiser did not reach
a maximum (too few iterations) or the model is over-parameterised for the data.

## How good are these standard errors?

The paper's simulations found the sandwich standard errors — computed with the original
code, and hence with the double-counted self term — to be slightly conservative. A
simulation study carried out with this package (100 genome-wide data sets generated with
the LD of the reference from a two-component model, $n = 130{,}000$; reported under
[Validation](../reference/validation.md)) gives the ratio of the mean standard error to
the empirical standard deviation of the estimates as 1.01 for $\pi_c$, 0.97 for
$\sigma^2$, 0.93 for $h^2$ and 0.72 for $a$. Standard errors from the inverse information
alone, which ignore the dependence between markers, were five to six times too small; for
the height fits they would be three to six times smaller than the sandwich values.

Three caveats apply to any version of the estimator:

- **Uncertainty in the LD reference is ignored.** $N^*_k$ and $\ell_k$ are treated as
  known.
- **Only tagged pairs are counted.** Correlation between scores of markers with
  $r^2$ below the threshold, or further apart than the window, is neglected.
- **They are asymptotic.** With weak signal (small studies, or one chromosome as in the
  demonstration data) the likelihood surface is far from quadratic and normal-theory
  intervals for $\pi_c$ can be poor; intervals on the log scale are then preferable.

What was verified for this implementation is listed in
[Validation](../reference/validation.md).
