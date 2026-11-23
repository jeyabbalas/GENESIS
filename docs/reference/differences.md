# Differences from the original software

This package re-implements the method of [Zhang et al. (2018)](#ref-zhang2018) and was
validated against the original R/C++ package
([yandorazhang/GENESIS](https://github.com/yandorazhang/GENESIS), commit `e4e6894`).
The statistical model, the likelihood and the estimating equations are the same. The
defaults nevertheless differ from the original in a number of places, because close
reading and side-by-side testing showed behaviours of the original code that depart
from the published method or make results depend on incidental details.

This page lists every such difference: what the original does (with file and line at
commit `e4e6894`), what this package does by default, why, how large the effect is where
it was measured, and how to obtain the original behaviour. Numbers quoted for "height"
are for the GIANT height GWAS ([Lango Allen et al. 2010](#ref-langoallen2010)) that
ships with the original package (941,684 SNPs after merging; 1 Mb window, $r^2 \ge 0.1$),
the same data as the original's `examples/examples.md` and the height row of the
paper's Table 1.

:::{important}
`legacy=True` (command line: `--legacy`) reproduces the numerical behaviour of the
original fitting code — EM trajectory, stopping point, covariance matrix, information
criteria and derived quantities — to rounding error. It is tested against outputs of
the genuine original code (see [Validation](validation.md)). Use it to reproduce a
published analysis; use the defaults for new analyses.
:::

## What `legacy` mode covers

`gg.fit(..., legacy=True)` or `genesis fit --legacy` switches, together:

- the EM control flow of `EM_func` / `EM_func3` (resets, the `c0` rule, the early
  `break`, the handling of the proportion bounds) — [details below](#em-algorithm);
- no Newton steps: the estimate is the point at which the original EM stops;
- the original's rule for the truncation level `c0`;
- the sandwich "meat" with each marker's own score counted twice;
- the asymmetric three-component information matrix of the original `Ik3`;
- all parameters treated as free in the covariance even with `stratification=False`;
- the original formula for the standard error of the sibling relative risk;
- no relabelling of the two clusters of the three-component model.

It does **not** change data preparation or the downstream analyses, which have their
own switches:

| To mirror the original's ... | use |
|---|---|
| `filter=FALSE` (its default) | `prepare(..., qc=False)` / `--no-qc` |
| `postmarginal()` | `posterior_marginal(..., include_inflation=False)` / `--ignore-inflation` |
| `polyriskpredict()` risk stratification | `polyrisk_predict(..., legacy=True)` |
| only the doubled self-term of the sandwich | `fit(..., sandwich_double_count=True)` / `--sandwich-double-count` |
| the EM stopping point without the other legacy behaviours | `fit(..., polish=False)` / `--no-polish` |
| a fixed truncation level | `EMConfig(c0=10, c0_max=10)` / `--c0 10 --c0-max 10` |

Functions that draw random numbers (QQ simulation, projection intervals, `futuregc`)
cannot reproduce the original's R random-number stream; they agree statistically.

## Estimates

### The estimate is the maximiser, not the point where the EM stopped

**Original.** The EM iteration stops when every parameter changed by less than an
absolute tolerance in one iteration ($10^{-6}$ for $\pi_c$, $10^{-8}$ for
$\sigma^2$, $10^{-9}$ for $a$) and the log-likelihood changed by less than $10^{-6}$ in
relative terms (`R/genesis.R` 65–72; `src/EM_EM.cpp` 396, 976). EM converges linearly
and, for this likelihood, slowly: when the per-iteration change falls below those
tolerances the iterate is still a noticeable distance from the maximum, and where it
stops depends on where it started.

**This package.** The reported estimate solves the score equations of the composite
likelihood. By default the EM algorithm serves as a warm-up (`newton_after` = 50
iterations), after which Newton's method on the exact scores and the observed
information — the quantities needed anyway for the standard errors — is run to
convergence ({func}`~genesis_gwas.core.em.newton_polish`). A Newton run is accepted
only if the information matrix is positive definite, no step decreases the likelihood,
and the last step is smaller than $10^{-5}$ naive standard errors; if it is declined,
the EM continues for a further, longer block (100, 200, ... iterations) and Newton's
method is tried again. With `accelerate=False` (`--no-accelerate`) the EM is first run
to the original's stopping rule and Newton's method starts from there. Both routes end
at the same point (on height they agree to within $2\times10^{-9}$ standard errors
for both models), and in the cases examined the result did not depend on the starting
value (see below).

**Measured on height**, at the truncation level `c0 = 10` that the original run ended
with, so that the comparison isolates the stopping point:

| | Original (EM stop) | Maximiser | Shift / original SE | Relative shift |
|---|---:|---:|---:|---:|
| **M2** $\pi_c$ | 4.9068e-03 | 4.8393e-03 | −0.13 | −1.4% |
| $\sigma^2$ | 5.7330e-05 | 5.7931e-05 | +0.12 | +1.0% |
| $a$ | 1.8615e-06 | 1.8790e-06 | +0.13 | +0.9% |
| **M3** $\pi_c$ | 8.8998e-03 | 9.3425e-03 | +0.38 | +5.0% |
| $p_1$ | 9.4760e-02 | 9.5761e-02 | +0.05 | +1.1% |
| $\sigma_1^2$ | 1.4586e-04 | 1.4244e-04 | −0.28 | −2.3% |
| $\sigma_2^2$ | 2.2271e-05 | 2.0863e-05 | −0.46 | −6.3% |
| $a$ | 1.5676e-06 | 1.5466e-06 | −0.18 | −1.3% |

For the three-component model this moves the estimated number of susceptibility SNPs
from 9,530 to 10,004 and the total heritability from
0.3238 to 0.3252. The original values are those of the height row of the
paper's Table 1 (9.5 thousand SNPs, heritability 0.324), i.e. the published numbers are
EM stopping points. The direction of the shift — more susceptibility SNPs, smaller
variance components — is the direction of the finite-sample bias the paper reports in
its simulations (downward for $\pi_c$, upward for the variances).

Two checks that this point is the right one:

- With `accelerate=False`, starting from $\pi_c = 0.005$ (the original example) or
  from $0.01$ (the default here), the EM stops at different points (M2: $\pi_c$ =
  4.9068e-03 and 4.9075e-03), but Newton's method ends at the same point from both, to
  seven significant digits — the point that the default route reaches as well.
- Continuing the EM itself from the original stopping point with tolerances at least
  four orders of magnitude tighter (a further 633 iterations for M2) ends within
  0.001 SE of that point.

The convergence check is part of every result: `diagnostics["newton_distance_se"]`
is the remaining Newton step in standard-error units (2e-09 and 8e-08 for
the default fits of height, against 0.14 and 0.44 at the original stopping points for
M2 and M3), and `diagnostics["polish"]["em_estimates"]` keeps the EM iterate from which
Newton's method started.

**Original behaviour:** `legacy=True`, or `polish=False` for the EM stopping point with
the other defaults of this package.

### Fixing $a = 0$ (`stratification=FALSE`)

**Original.** With `stratification=FALSE` the inflation term is set to zero before and
after each Newton step, but the step itself is still the *joint* step in
$(\sigma^2, a)$ (`src/EM_EM.cpp` 354–357 and 930–933). The $\sigma^2$ component of a joint
step is not the Newton step of the constrained problem, so the iteration settles where
a combination of the two partial derivatives of $Q$ vanishes rather than where
$\partial Q/\partial\sigma^2 = 0$.

**This package.** The Newton step is taken in the variance components alone, and the
covariance matrix treats $a$ as fixed (zero variance) rather than as an estimated
parameter.

**Measured** (synthetic data, $K = 3{,}000$, fixture scenario `c_no_stratification`):
original $\hat\pi_c$ = 9.216e-03; EM of this package at the same tolerances 9.056e-03;
maximiser 8.969e-03 (the original is 2.7% or 0.22 SE away). Not measured on real data.

**Original behaviour:** `legacy=True`.

## Standard errors and model selection

### The marker's own score is counted once in the sandwich estimator

The paper's eq. (4) is $\widehat{\operatorname{var}}(\hat\theta) = I^{-1} J I^{-1}$
with $J = \sum_k U_k \bar U_k^\top$, where $\bar U_k$ sums the scores of the GWAS
markers in the tagging neighbourhood of marker $k$, *including itself*.

**Original.** The tagging lists of the reference tables already contain the marker
itself (verified for all 1,070,777 SNPs in all nine tables), and the code then adds the
marker's score once more: `m_Sbar = ... + m_S` (`R/genesis.R` 193 and 312). The result
is $J + \sum_k U_k U_k^\top$.

**This package.** $J$ as in eq. (4). The same matrices enter the effective number of
parameters $d_s = \operatorname{tr}(I^{-1}J)$ and therefore AIC and BIC.

**Measured on height** (both conventions evaluated at the same, default, estimate):

| | SE ratio, doubled / eq. (4) | $d_s$ eq. (4) | $d_s$ doubled |
|---|---|---:|---:|
| M2 ($\pi_c$, $\sigma^2$, $a$) | 1.017, 1.020, 1.030 | 93.38 | 96.92 |
| M3 ($\pi_c$, $p_1$, $\sigma_1^2$, $\sigma_2^2$, $a$) | 1.021, 1.034, 1.048, 1.030, 1.029 | 104.31 | 108.92 |

The doubled self-term makes standard errors 2–5% larger on these data;
the BIC difference between M2 and M3 changes from 1,044 to
1,029, with M3 selected either way. Both sets of standard errors are in
every result file (`diagnostics["sandwich_alt"]`). (For scale: ignoring LD altogether,
i.e. $J = \sum_k U_kU_k^\top$, would make the standard errors four to five times
smaller.)

**Which convention is better calibrated?** A simulation with realistic LD (100
genome-wide data sets from a two-component model; see
[Validation](validation.md#simulation-study)) gives the following ratios of the mean
estimated standard error to the empirical standard deviation of the estimates:

| | eq. (4), default | doubled self-term, original | inverse information, no sandwich |
|---|---:|---:|---:|
| $\pi_c$ | 1.01 | 1.03 | 0.17 |
| $\sigma^2$ | 0.97 | 0.99 | 0.18 |
| $a$ | 0.72 | 0.75 | 0.20 |

Both conventions are well calibrated for $\pi_c$ and $\sigma^2$ and differ by about 2%,
which 100 replicates cannot resolve (the Monte-Carlo standard error of each ratio is
about 0.07). The simulation therefore gives no empirical reason to prefer one; the
default is the formula as published. For $a$ both are too small in this simulation,
for a reason that may lie in the simulation design (discussed with the study).

**Original behaviour:** `legacy=True`, or `sandwich_double_count=True` alone.

### Symmetric information matrix for the three-component model

**Original.** In `Ik3`, the per-marker information of the three-component model, one
element of the outer product of first derivatives is assigned to itself,
`rr(2,0) = rr(2,0);` (`src/EM_EM.cpp` 1101), and therefore stays zero; the
corresponding line of the two-component code reads `rr(2,0) = r(2)*r(0);` (485). The
$(\sigma_1^2, \pi_c)$ entry of the information matrix thus lacks a term that its
transpose $(\pi_c, \sigma_1^2)$ has, the matrix is not symmetric, and neither is the
covariance matrix computed from it. This is visible in the original's own
`examples/examples.md`, whose three-component covariance has $v_{12}$ = −8.57e-06 but
$v_{21}$ = −1.13e-05.

**This package.** The symmetric matrix. `legacy=True` reproduces the original matrix
by removing the same term.

**Measured on height** (at the default estimate): the two entries differ by
0.9%; using the original matrix changes the standard errors of
($\pi_c$, $p_1$, $\sigma_1^2$, $\sigma_2^2$, $a$) by −2.3%, +2.2%, −3.6%, −4.2% and −0.2%.

**Original behaviour:** `legacy=True`.

### Standard error of the sibling relative risk

**Original.** The sibling relative risk is $\lambda = \exp(h^2/2)$ and its standard
error is computed as $\sqrt{\lambda\,\mathrm{se}(h^2)^2/2}$ (`R/genesis.R` 214, 346).

**This package.** The delta method, $\tfrac12\,\lambda\,\mathrm{se}(h^2)$.

**Measured on height (M2):** 0.0109 (original expression) versus 0.0083 (delta method)
for the same $h^2$ and standard error.

**Original behaviour:** `legacy=True`.

### Net effect on reported standard errors

Because the point at which the standard errors are evaluated (the maximiser, at the
truncation level chosen by the rule [described below](#the-truncation-level-c0)), the
"meat" (self-term) and, for M3, the information matrix all changed, default standard
errors differ from the original's as follows on height:

| | `c0` | $\pi_c$ | $p_1$ | $\sigma^2$ / $\sigma_1^2$ | $\sigma_2^2$ | $a$ | $d_s$ | BIC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M2 original | 10 | 5.357e-04 |  | 4.992e-06 |  | 1.312e-07 | 97.27 | −7,899,471.7 |
| M2 default | 10 | 5.150e-04 |  | 4.904e-06 |  | 1.264e-07 | 93.38 | −7,899,524.6 |
| M3 original | 10 | 1.152e-03 | 1.839e-02 | 1.232e-05 | 3.070e-06 | 1.186e-07 | 104.68 | −7,900,558.5 |
| M3 default | 14 | 1.515e-03 | 1.770e-02 | 1.232e-05 | 3.569e-06 | 1.216e-07 | 104.31 | −7,900,568.4 |

(em-algorithm)=
## EM algorithm

### Two likelihoods

The sum over the latent numbers of tagged susceptibility SNPs is truncated at `c0` per
component. Writing $\pi_{kt}$ for the multinomial prior of configuration $t$ and
$f_{kt}$ for the normal density of $\hat\beta_k$ under it, two quantities appear in the
code:

- the **unnormalised** truncated likelihood
  $\tilde\ell = \sum_k \log \sum_t \pi_{kt} f_{kt}$. The original M-step
  ($\pi_c = \sum w j / \sum w N^*$, `update_pi1`) and the scores and information used
  for the standard errors (`Sk`, `Ik`) are exactly those of $\tilde\ell$: it is the
  function the EM maximises;
- the **renormalised** likelihood
  $\ell = \tilde\ell - \sum_k \log \sum_t \pi_{kt}$, which divides by the truncated
  prior mass (`res += log(marginal_likelihood/y)`, `src/EM_EM.cpp` 231, 771). This is
  the value that is monitored during the iteration, reported, and used in the BIC.

The two differ by the truncated prior mass, which is negligible when few markers can
tag more than `c0` susceptibility SNPs (on height at `c0 = 10`:
$\ell - \tilde\ell$ = 0.04 for M2 and 3.1 for M3, out of 3.95 million). Both values
are returned (`loglik`, `loglik_unnorm`), and `diagnostics["truncation"]` reports the
discarded mass.

### The truncation level `c0`

**Original.** The iteration starts at `c0 = 10`. If the monitored log-likelihood
$\ell$ is lower than in the previous iteration, `c0` is increased by one, up to 20
(`src/EM_EM.cpp` 371, 949). Because the EM increases $\tilde\ell$, not $\ell$, this
fires for two unrelated reasons:

1. *Near convergence, when $\pi_c$ is decreasing.* The gain in $\tilde\ell$ becomes
   second-order small while the change of the truncated prior mass is first order in
   the change of $\pi_c$. In the $K = 3{,}000$ fixture (scenario
   `a_regular_startingpic0.01`), after 120 regular iterations $\tilde\ell$ rose by
   9.19e-05 while $\ell$ fell by 6.3e-07 as $\pi_c$ moved from 0.0078988 to 0.0078967;
   `c0` went from 10 to 11 and, in the next two iterations, to 13, where the run
   stopped. The same data started from $\pi_c = 0.002$, so that $\pi_c$ approaches
   its limit from below, keep `c0 = 10` and end at $\hat\pi_c$ = 0.007798 instead of
   0.007894.
2. *After a Newton step that overshoots*, which has nothing to do with truncation.

Whether and when `c0` changes thus depends on the path from the starting values, and
the estimate depends on `c0`: the truncation caps the number of susceptibility SNPs
that a marker can tag, which holds $\hat\pi_c$ down when markers with large $N^*$
would be expected to tag more than `c0` of them.

**This package.** `c0` never changes during an iteration. The model is solved at the
initial `c0` (10); then the prior mass that the truncation discards, averaged over
markers, is evaluated at the solution (`diagnostics["truncation"]["mean_tail"]`). If it
exceeds the tolerance `tail_tol` ($10^{-6}$; `--tail-tol`), `c0` is raised to the
smallest level at which that mass falls below a tenth of the tolerance, the model is
solved again from the previous solution, and the check is repeated, up to `c0_max`
(20; `--c0-max`). The rule depends on the data and the solution only. Every solve is
recorded in `diagnostics["c0_path"]`, and a warning is issued if the tolerance cannot
be met at `c0_max`. Set `c0_max = c0` for a fixed level. Raising `c0` has a price: the
cost of a pass over the data grows as $(c_0+1)^2$ for the three-component model.

**Measured.**

*Height, two-component model:* the discarded mass at the `c0 = 10` solution is 4.6e-08, and `c0` stays at 10. Fixing `c0` at 15, 20 or 30 instead moves no estimate by more than 0.0009 standard errors.

*Height, three-component model:* the tolerance is exceeded at `c0 = 10` and the model is solved again at `c0 = 14`:

| `c0` | Discarded prior mass (mean over markers) | $\hat\pi_c$ | Susceptibility SNPs | Heritability |
|---:|---:|---:|---:|---:|
| 10 | 3.3e-06 | 9.3425e-03 | 10,004 | 0.3252 |
| 14 | 6.8e-08 | 9.5792e-03 | 10,257 | 0.3259 |

The estimate of $\pi_c$ rises by 2.5% (0.16 standard errors) and the heritability by 0.0007. Fixing `c0` at 20 instead gives $\hat\pi_c$ = 9.5865e-03 (10,265 susceptibility SNPs, heritability 0.3259), a further 0.005 standard errors. The level chosen by the rule is sufficient here.

*Chromosome 22 of height, two-component model* (the example shipped with the package; $M$ = 15,633): here the truncation level determines the answer.

| `c0` | Discarded prior mass (mean over markers) | $\hat\pi_c$ | Susceptibility SNPs | Heritability |
|---:|---:|---:|---:|---:|
| 10 | 3.4e-04 | 2.8919e-02 | 452 | 0.0054 |
| 18 | 6.5e-05 | 5.2151e-02 | 815 | 0.0057 |
| 20 | 4.4e-05 | 5.7797e-02 | 904 | 0.0057 |

$\hat\pi_c$ changes by a factor of 2.0 between `c0 = 10` and `c0 = 20`, and the tolerance is still not met at the ceiling, which the fit reports as a warning. With several per cent of the SNPs estimated to be susceptibility SNPs, a marker that tags hundreds of SNPs is expected to tag more of them than any moderate `c0` allows. Fitted at fixed levels, the proportion keeps growing while the heritability hardly changes:

| `c0` (fixed) | $\hat\pi_c$ (SE) | Heritability (SE) |
|---:|---:|---:|
| 10 | 0.0289 (0.0031) | 0.00540 (0.00097) |
| 20 | 0.0578 (0.0098) | 0.00572 (0.00106) |
| 40 | 0.0883 (0.1041) | 0.00575 (0.00108) |
| 60 | 0.0883 (0.1045) | 0.00575 (0.00108) |

For these data $\pi_c$ is not identified: one chromosome of one study cannot distinguish many susceptibility SNPs with small effects from fewer with larger effects. The warning says as much, and the ceiling keeps the fit from running away.

The original's rule, on the same data, ends at whatever level its path happens to reach: `c0 = 10` for both height models, and 13 or 10 for the synthetic example above, depending on the starting value.

### The early exit at `c0 = 20`

**Original.** When `c0` reaches 20 the loop is left in the same iteration
(`src/EM_EM.cpp` 372, 950), before the result vector is updated: the returned
parameters and the returned `c0` (at most 19) are those of the previous iteration,
whereas the returned log-likelihood is evaluated afterwards at the newer parameters
with `c0 = 20` (399, 980). The exit is not governed by the convergence tolerances.

**This package.** There is no such exit, because `c0` does not change during an
iteration. `legacy=True` reproduces the original, including which values are returned
(`status = "legacy_break"`).

### Newton steps that leave the admissible region or overshoot

**Original.** The variance components are updated by raw Newton steps on the expected
complete-data log-likelihood $Q$, which is not concave. A component above 1 is reset
to $10^{-5}$ (to $10^{-6}$ for $a$ in the two-component code), a negative
$\sigma^2$ to $10^{-12}$, and $a < -\min_k s_k^2$ to $-\min_k s_k^2/2$
(`src/EM_EM.cpp` 360–363, 937–942). In the three-component code the second Newton
step is evaluated wherever the first one landed, including at negative variances.
Nothing checks that the likelihood increased.

**This package.**

- If the Hessian of $Q$ is not negative definite, the Newton direction is replaced by
  the direction obtained after flipping the sign of its positive eigenvalues, which is
  always an ascent direction ({func}`~genesis_gwas.core.em.newton_direction`).
- A negative $\sigma^2$ is set to $10^{-12}$ and an inadmissible $a$ to
  $-\min_k s_k^2/2$, as in the original; there are no resets for large values.
- After the M-step the new point is accepted only if $\tilde\ell$ did not decrease;
  otherwise the step is halved (up to 30 times). The iteration is therefore monotone.

Whenever none of this triggers, the EM iterates equal the original's to rounding
error for as long as the original leaves `c0` unchanged (in the synthetic example above the first 120 iterates of the two policies agree to 2e-15); that was
the case in every real-data run examined (height genome-wide and on chromosome 22:
`diagnostics["em_backtracks"]` is 0). In a fixture scenario with a
deliberately bad starting value ($\sigma^2 = 0.5$, three-component model, $K = 300$)
the original returns $a = 0.285$ and a heritability standard error of 28 after 12
iterations, without any message.

### Proportion bounds of the three-component model

**Original.** $\pi_c$ and $p_1$ are restricted to $[10^{-7}, 0.5]$. The closed-form
update is only carried out if both *current* values are inside the bounds; otherwise
the previous update is reused and the offending parameter is set to its bound
(`src/EM_EM.cpp` 920–926). A parameter that the data push beyond a bound therefore
alternates between the bound and the unconstrained update, and `c0` is raised every
other iteration until the early exit. If the *starting* value is outside the bounds,
$\pi_c$ is read from a vector that has not been assigned yet (`vec tem_p(2)`, line
894; zero-filled by current Armadillo versions) and the run returns NaN for all 3,000
iterations without an error (fixture scenarios `e2`, `e3`).

**This package.** The update is computed and then clipped to the bounds, which is the
exact constrained maximiser of $Q$ for these parameters. Starting values are clipped
as well. During polishing a parameter that reaches a bound is held there.

## Three-component fits that are not identified

**Original.** Nothing is checked. If the two variance components merge (as on a single
chromosome of the height data, where $\hat\sigma_1^2$ = 7.9072e-06 and
$\hat\sigma_2^2$ = 7.9067e-06 after 2,490 iterations), $p_1$ is arbitrary, the
information matrix is singular, and the reported standard errors include `NaN`. If the
fit ends with $\sigma_1^2 < \sigma_2^2$, the outputs labelled "cluster with larger
variance component" describe the smaller one.

**This package.**

- A fit whose variance components agree to within 1%, or one of which has collapsed to
  zero, is flagged (`diagnostics["degenerate"]`, a warning in the summary), and
  {func}`~genesis_gwas.select_model` never selects it when a well-identified
  alternative exists.
- Such a fit is also stopped early: once Newton's method has declined and the EM
  iterate has collapsed onto the two-component model, the remaining iterations would
  only drift along a flat direction of the likelihood (`status = "not_identified"`).
  The fit is not repeated at higher truncation levels, and $d_s$, AIC and BIC are
  reported as unavailable (`NaN`), because $\operatorname{tr}(I^{-1}J)$ has no meaning
  when the information matrix is singular. On chromosome 22 of height this ends the
  three-component fit after 150 iterations and a few seconds.
- A parameter that ends on a bound is treated as fixed in the covariance matrix: its
  standard error is `NaN`, the others are conditional on the constraint, and it does
  not count towards $d_s$ (`diagnostics["at_bound"]`).
- If the fit ends with $\sigma_1^2 < \sigma_2^2$ the two clusters are relabelled
  ($p_1 \to 1 - p_1$, variances swapped, covariance transformed accordingly) so that
  cluster 1 is always the larger-variance cluster, as the paper defines it
  (`diagnostics["labels_swapped"]`).

**Original behaviour:** `legacy=True` reports the original's numbers unchanged: the EM runs
to the original's stopping point, there is no relabelling, all parameters are treated
as free, and $d_s$, AIC and BIC are computed as the original computes them. The fit is
still flagged with a warning, and {func}`~genesis_gwas.select_model` still prefers an
identified alternative.

## Downstream analyses

### Posterior probabilities and shrinkage estimates

**Original.** `postmarginal()` computes the local false-discovery rate and the
posterior mean of each marginal effect with noise variance $s_k^2$ only; the fitted
inflation term $a$ does not appear (`R/postmarginal.R` 50–65, 74–94), although the model
under which the parameters were estimated is
$\hat\beta_k \mid \beta_k \sim N(\beta_k, s_k^2 + a)$.

**This package.** Noise variance $s_k^2 + a$, consistent with the likelihood.

**Measured** (original height M2 estimates; a marker with $N^* = 40$, $\ell = 15$,
$n = 130{,}000$):

| $z$ | lfdr, original | lfdr, default | shrinkage factor, original | default |
|---:|---:|---:|---:|---:|
| 3 | 0.24 | 0.40 | 0.57 | 0.43 |
| 4 | 0.022 | 0.082 | 0.74 | 0.65 |
| 5 | 0.0007 | 0.0066 | 0.76 | 0.72 |

With $a > 0$ the original is over-confident: it reports smaller false-discovery rates
and shrinks less. Also, the original always truncates at `c0 = 10`; this package uses
the `c0` at which the model was fitted.

**Original behaviour:** `include_inflation=False` (reproduces the original to
$10^{-10}$ on all 941,684 height SNPs).

### Polygenic risk prediction

**Original.** `polyriskpredict()` computes $\mu = \operatorname{cov}(\hat S, S)/
\operatorname{sd}(\hat S)$ for the polygenic score $\hat S$ and reports
$\mathrm{AUC} = \Phi(\mu/\sqrt2)$. For the proportions of the population and of cases
above $k$-fold risk it then uses $\sqrt{\mu}$ as the standard deviation of the log
relative risk (`R/polyriskpredict.R` 84–86).

**This package.** Under the log-normal risk model that gives
$\mathrm{AUC} = \Phi(\mu/\sqrt2)$, the calibrated score has standard deviation $\mu$,
so $\mu$ is used in the risk-stratification formulas as well. AUC and the optimal
threshold are unaffected.

**Measured** (the original example call at $N = 253{,}288$ with the two-component
height estimates; $\mu = 0.471$, AUC = 0.631 in both):

| $k$-fold risk | population above, original | default | cases above, original | default |
|---:|---:|---:|---:|---:|
| 3 | 2.60% | 0.51% | 10.4% | 1.8% |
| 4 | 0.91% | 0.075% | 4.7% | 0.34% |
| 5 | 0.36% | 0.013% | 2.3% | 0.074% |

Since $\mu < 1$ in practice, $\sqrt\mu > \mu$ and the original overstates the spread of
risk. The paper's risk-stratification numbers were produced with the original
expression: its type-2-diabetes example (1.3% versus 1.1% of the population and 9.4%
versus 7.4% of cases above five-fold risk, for 1,000,000 versus 100,000 individuals)
is reproduced from the rounded Table 1 estimates with $M$ = 200,000 by
`legacy=True` (1.30% / 1.06% and 9.40% / 7.43%), whereas the consistent formula gives
0.91% / 0.54% and 6.24% / 3.49% for the same inputs.

Separately, the worked example shipped with the original
(`examples/examples.md`, lines 58 and 61) passes `Sig2s = c(est[3], est[3])` for the
two-component model, which is the inflation term $a$ rather than the variance
`est[2]`; with a fitted model, this package takes the variance components from the
fit.

**Original behaviour:** `polyrisk_predict(..., legacy=True)`.

### QQ plots and genomic-control projections

**Original.** In `qqplotdata.simu()` the term that represents the inflation $a$ is a
single draw, `rnorm(1, mean = 0, sd = sqrt(a))`, added to every SNP of a simulation
(`R/qqplotdata.simu.R` 106): all effect estimates of a replicate are shifted by the same
amount instead of each receiving independent noise of variance $a$. `futuregc()` draws
it per SNP (`R/futuregc.R` 36), but its three-component branch seeds the generator
with `set.seed(iter*123)` and ignores the `seeds` argument (line 55).

**This package.** Independent draws per SNP. The latent configurations are sampled
with vectorised draws from the same truncated distribution the original samples from
one SNP at a time.

**Measured** (chromosome 22 of height, 13,230 SNPs, 100 simulations with the same
LD-correlated null replicates): the mean curve and the mean genomic-control factor
agree ($\lambda_\text{fit}$ = 1.548 here, 1.553 original; observed 1.648), but the
central 80% range of the simulated $\lambda$ is 0.15 wide here and 0.33 in the
original. The common shift is what widens the original's envelope; envelopes drawn by
this package are narrower than those in the paper's Figure 1.

### Projection intervals

**Original.** `projection(CI=TRUE)` draws parameter vectors as
$\exp\{N(\log\hat\theta, \cdot)\}$ for all parameters including $a$
(`R/projection.R` 57–63). It stops with an error when $a = 0$ exactly (the output of
`stratification=FALSE`) and when the covariance matrix is not positive definite
(e.g. the chromosome-22 three-component fit: `'Sigma' is not positive definite`); and
because $p_1$ is sampled on the log scale it can exceed one, which gives the second
component a negative weight.

**This package.** $a$ is not sampled (the projection does not depend on it),
$\pi_c$ and the variances are sampled on the log scale and $p_1$ on the logit scale,
and the covariance is symmetrised with negative eigenvalues clipped. The extra
binomial sampling of the number of discoveries is kept. Interval limits agree with the
original to within Monte-Carlo error (6% on the height examples). The point
projections are computed from closed-form integrals and agree with the original's
numerical integration to $2\times10^{-11}$.

## Data handling

**Original.** `preprocessing()` (`R/preprocessing.R`):

- missing values are removed only if there is more than one
  (`if(length(inx)>1)`, lines 54 and 56); a single missing $Z$ or $N$ stays in the data
  and propagates;
- with `filter=TRUE`, a missing sample size stops the function, because the 90th
  percentile is computed without `na.rm` (line 30);
- duplicated SNP identifiers are not removed and are duplicated again by the merge;
- filtering is off by default (`filter=FALSE`), although the documentation recommends
  it.

For the sandwich estimator, `genesis()` converts every identifier to the number after
`"rs"` and allocates a lookup vector as long as the largest such number
(`R/genesis.R` 190–191, `src/EM_EM.cpp` 30–37); identifiers that are not of the form
`rs<number>` cannot be used.

**This package.**

- Rows with missing or non-finite $Z$ or $N$, $N \le 0$, and duplicated identifiers
  (first occurrence kept) are always removed and counted in the QC report.
- The recommended filters — $N \ge 0.67 \times$ the 90th percentile, HapMap3 without
  the MHC, $\chi^2 \le 80$ — are **on by default**, applied in the original's order
  with the original's thresholds.
- Any identifier works; SNPs can also be matched by chromosome and position in GRCh37
  or GRCh38.
- $Z$ and $N$ can be derived from other columns (effect and standard error, odds
  ratio, P-value with sign, case and control counts).

On clean input, `qc=False` gives the same merged data as the original's default (the
height example: 941,684 SNPs in both).

**Original behaviour:** `prepare(..., qc=False)` / `--no-qc`.

## Numerical methods

| | Original | This package |
|---|---|---|
| Mixture densities | linear scale; where `long double` is 64-bit (arm64, MSVC) $(1-\pi_c)^{N^*}$ can underflow and produce NaN weights | log scale with log-sum-exp throughout |
| Precision | `long double` (80-bit on x86, 64-bit elsewhere), so results depend on the platform in the last digits | IEEE double everywhere, including GPUs |
| Projection integrals | adaptive quadrature (`integrate`) | closed form |
| Number of SNPs in an effect-size interval | difference of two CDF values (cancels for far tails) | upper-tail probabilities |
| Information-matrix inverse | `solve(m_I, tol = 1e-20)` on the raw matrix, whose entries span 6 (M2) to 10 (M3) orders of magnitude on height (condition number $9\times10^{10}$ for M3) | inverse after symmetric diagonal scaling (condition number 35 and 280 at the default estimates); definiteness and condition number reported |
| Variance-component Newton step | explicit cofactor formulas | linear solve (`legacy`: the same cofactor formulas) |

None of these changes results beyond rounding error where the original works.

## Engineering

| | Original | This package |
|---|---|---|
| Installation | R package with 1.6 GB of data inside the repository; C++ compiled at install time with OpenMP (with R 4.6 and Rcpp 1.1 the source only compiled after adding `-DPI=M_PI`) | pure Python wheel (2 MB); no compiler |
| Reference data | all nine LD tables and 100 null replicates installed with the package | about 30 MB per ancestry downloaded on first use for the default configuration; checksummed, resumable; other components on demand |
| Parallel workers | `genesis()` and `qqplotdata.simu()` each start `makeCluster(cores)` and never stop it (`R/genesis.R` 91, `R/qqplotdata.simu.R` 29); the cluster is not used for computation | one thread pool per fit, released when the fit ends |
| Memory | dense $K \times (c_0+1)^2$ weight matrix for the three-component model (1.0 GB at `c0 = 10`, 3.5 GB at `c0 = 20` per million SNPs), allocated and copied every iteration | chunked; working memory independent of the number of SNPs |
| Tagging neighbourhoods | comma-separated identifier strings parsed in R for every SNP at every fit | integer CSR arrays |
| Hardware | CPU threads (OpenMP) | CPU threads or CUDA GPUs |
| Reference panels | European (GENESIS) and East Asian (GENESISasian, one LD configuration), in separate packages | European, East Asian and African ancestry with nine LD configurations each, selected with `--reference`; any other population built from VCF or PLINK files with `genesis ld from-vcf` and `genesis ld build` |

Run times measured on one machine are in [Validation](validation.md#run-times).

(genesisasian)=
## GENESISasian

The authors distribute a second R package for East Asian studies,
[yandorazhang/GENESISasian](https://github.com/yandorazhang/GENESISasian) (examined at
commit `4df90ac`). Compared file by file with GENESIS at `e4e6894`:

| | GENESIS | GENESISasian |
|---|---|---|
| C++ source | | byte-identical |
| R code, `genesis()` and helpers | default `M = 1070777` | default `M = 953834`; otherwise the same |
| R code, `polyriskpredict()` | default `M = 1070777` | default `M = 200000`; line 67 passes `Sig2s=sig2s`, a variable the function does not define, where GENESIS passes `Sig2s=Sig2s` |
| LD data | nine configurations (0.5 / 1 / 2 Mb × $r^2 \ge$ 0.05 / 0.1 / 0.2), 489 European samples, 1,070,777 SNPs | one configuration (1 Mb, $r^2 \ge 0.1$), 504 East Asian samples, 953,834 SNPs |
| SNP set | HapMap3 without the MHC, MAF ≥ 0.05 in the panel | HapMap3, MAF ≥ 0.05 in the panel, **including** 1,775 SNPs in the MHC |
| Example GWAS and HapMap3 list | GIANT height (European) | the same two files, byte-identical |

So the method and its implementation are the same, and everything on this page applies
to GENESISasian as it does to GENESIS. In this package the East Asian data are the
reference `eas_1kg_hm3` (`--reference EAS`):

- The 1 Mb / 0.1 tables, the pairwise $r^2$ and the 100 null replicates are a lossless
  conversion of the original files. `legacy=True` with this reference reproduces
  GENESISasian to rounding error ([Validation](validation.md#east-asian-reference)).
- The other eight LD configurations do not exist in GENESISasian. They were computed
  with this software from the 504 `EAS` samples of 1000 Genomes Phase 3, after the
  same computation reproduced the original 1 Mb / 0.1 table (every $N^*$ and every
  tagging list identical).
- The 1,775 MHC SNPs stay in the reference and count towards $M$, as in the original.
  Quality control removes MHC markers from the GWAS by default, as for every reference
  (`--keep-mhc` to retain them, as GENESISasian with `filter=FALSE` does).
- The example GWAS inside GENESISasian is the European height file of GENESIS; it is
  not distributed with `eas_1kg_hm3`.
- Functions called with a bare parameter vector assume $M = 1{,}070{,}777$ unless `M=`
  is given, where GENESISasian's assume 953,834; pass the fitted model or `M=953_834`.
  `polyrisk_predict` takes `M` explicitly in both.

There is no original package for African ancestry; `afr_1kg_hm3` was built with this
software ([Other ancestries](../tutorial/other-ancestries.md)).

## Summary

| Item | Affects | Default here | Original behaviour via |
|---|---|---|---|
| Newton's method after the EM | estimates (up to 0.46 SE on height) | maximiser of the likelihood | `legacy=True` or `polish=False` |
| Constrained step for $a = 0$ | estimates with `stratification=False` | constrained Newton | `legacy=True` |
| Self-term of the sandwich | SEs (2–5% larger in the original), $d_s$, AIC/BIC | counted once (paper eq. 4) | `legacy=True` or `sandwich_double_count=True` |
| `Ik3` information matrix | M3 SEs (up to 4.2%), $d_s$, BIC | symmetric | `legacy=True` |
| Sibling-risk SE | one derived SE | delta method | `legacy=True` |
| Truncation level `c0` | estimates, $\pi_c$ in particular | raised until the discarded prior mass at the solution is below $10^{-6}$ | `legacy=True`; `c0_max = c0` for a fixed level |
| Early exit at `c0 = 20` | which values are returned | none | `legacy=True` |
| Newton safeguards | robustness | ascent direction, step halving | `legacy=True` |
| Proportion bounds | robustness, M3 estimates at a bound | clipped update | `legacy=True` |
| Non-identified M3 fits | model selection, reporting, run time | stopped early, flagged, not selectable, no AIC/BIC | `legacy=True` |
| Parameters on a bound | SEs, $d_s$ | fixed (SE = `NaN`) | `legacy=True` |
| Cluster labels | reporting of M3 | cluster 1 = larger variance | `legacy=True` |
| Posterior noise variance | lfdr, shrinkage | $s_k^2 + a$ | `include_inflation=False` |
| Risk SD in PPI/PCI | risk stratification | $\mu$ | `polyrisk_predict(legacy=True)` |
| Inflation draw in QQ simulation | envelope width | independent per SNP | — |
| Projection intervals | interval limits | $a$ excluded, logit scale for $p_1$ | — |
| Quality control | SNP set | recommended filters on | `qc=False` / `--no-qc` |
| Missing values, duplicates | robustness | removed and reported | — |
| Log-space arithmetic, scaled inverse | robustness | on | — |
