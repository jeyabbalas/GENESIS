# Estimation

The parameters are estimated by maximising the truncated composite likelihood
$\tilde\ell(\theta)$ defined in the [previous chapter](model.md). Two algorithms share the
work:

1. an **EM algorithm** — the one of the original software, with safeguards — which is
   robust far from the optimum but slow near it;
2. **Newton's method** on the likelihood itself, which uses exact first and second
   derivatives and converges in a handful of passes once the EM has brought the estimate
   into the neighbourhood of the maximum.

By default the EM is used as a warm-up and Newton's method finishes the job
({ref}`est-schedule`). The solution is then checked against the truncation of the
likelihood and, if necessary, recomputed with more terms ({ref}`est-adaptive-c0`). The
original software uses the EM alone and stops it by a rule that fires before the maximum
is reached; `legacy=True` reproduces that behaviour.

This chapter derives both algorithms, states the stopping rules, and explains how to read
the convergence diagnostics.

## The EM algorithm

### Complete data

Treat the configuration $t_k$ of each marker — the vector of counts
$J_{t_k} = (N_k^{(1)},\dots,N_k^{(H)})$ of tagged susceptibility SNPs of each kind — as
missing data. If it were observed, the log-likelihood contribution of marker $k$ would be
$\log \pi_{k t_k} + \log f_{k t_k}$ with the multinomial weight $\pi_{kt}$ and the normal
density $f_{kt}$ of the [marker likelihood](model.md):

$$
\log \pi_{kt} = \text{const} + \sum_{h} J_{th}\log(\pi_c p_h) + \bigl(N^*_k - |J_t|\bigr)\log(1-\pi_c),
\qquad
\log f_{kt} = -\tfrac12\log(2\pi v_{kt}) - \frac{\hat\beta_k^2}{2 v_{kt}} .
$$

The proportions $(\pi_c, p)$ appear only in the first term and the variance parameters
$(\sigma^2, a)$ only in the second, through $v_{kt} = r_k c_t + s_k^2 + a$ with
$r_k = \ell_k/N^*_k$ and $c_t = \sum_h J_{th}\sigma_h^2$. That separation is what makes
the M-step cheap.

(est-estep)=
### E-step

Given the current parameters $\theta^{\text{old}}$, the posterior probability of
configuration $t$ for marker $k$ is

$$
w_{kt} = \frac{\pi_{kt} f_{kt}}{\sum_{t'} \pi_{kt'} f_{kt'}} .
$$ (eq-weights)

The original code evaluates numerator and denominator as raw densities, which underflow to
zero for markers with large neighbourhoods on platforms without extended precision. Here
everything stays in log space
({func}`genesis_gwas.core.kernels.em_pass`):

$$
\lambda_{kt} = \log\pi_{kt} - \tfrac12\log v_{kt} - \tfrac12\,\frac{\hat\beta_k^2}{v_{kt}},
\qquad
w_{kt} = \exp\bigl(\lambda_{kt} - \max_{t'}\lambda_{kt'}\bigr) \Big/ \sum_{t''}\exp\bigl(\lambda_{kt''} - \max_{t'}\lambda_{kt'}\bigr),
$$

and the same quantities give the marker's log-likelihood as a log-sum-exp, so the E-step
and the evaluation of $\tilde\ell$ are one pass. Two observations keep the pass cheap:

- $\log\pi_{kt}$ depends on the marker only through $N^*_k$. It is tabulated once per
  iteration for the few hundred *distinct* values of $N^*$
  (`CompositeLikelihood._prior_tables`) and looked up per marker.
- Impossible configurations ($|J_t| > N^*_k$) have $\log\pi_{kt} = -\infty$ and hence
  $w_{kt} = 0$ exactly, with no special-casing in the inner loop.

(est-mstep-proportions)=
### M-step: proportions

The expected complete-data log-likelihood is
$Q(\theta) = \sum_k\sum_t w_{kt}\,[\log\pi_{kt}(\theta) + \log f_{kt}(\theta)]$. Its first
part is

$$
\sum_k\sum_t w_{kt}\Bigl[\,|J_t|\log\pi_c + \bigl(N^*_k - |J_t|\bigr)\log(1-\pi_c)
     + \sum_h J_{th}\log p_h \Bigr] + \text{const},
$$

a binomial log-likelihood in $\pi_c$ plus a multinomial one in $p$. With
$A_h = \sum_k\sum_t w_{kt} J_{th}$, the expected number of tagged susceptibility SNPs of
kind $h$ summed over markers, the maximisers are

$$
\pi_c^{\text{new}} = \frac{\sum_h A_h}{\sum_k N^*_k}, \qquad
p_h^{\text{new}} = \frac{A_h}{\sum_{h'} A_{h'}} .
$$ (eq-mstep-proportions)

In words: the new $\pi_c$ is the expected number of tagged susceptibility SNPs divided by
the number of tagged SNPs, both totalled over markers.

:::{admonition} Why this is the M-step for the *unnormalised* likelihood
:class: note

By Jensen's inequality, for any $\theta$,

$$
\log\sum_t \pi_{kt}(\theta) f_{kt}(\theta) \;\ge\;
\log\sum_t \pi_{kt}^{\text{old}} f_{kt}^{\text{old}}
 + \sum_t w_{kt}\log\frac{\pi_{kt}(\theta) f_{kt}(\theta)}{\pi_{kt}^{\text{old}} f_{kt}^{\text{old}}} ,
$$

and this holds whether or not the weights $\pi_{kt}$ sum to one over the retained
configurations. Increasing $Q$ therefore increases
$\tilde\ell(\theta)=\sum_k\log\sum_{t\in\mathcal T_k}\pi_{kt}f_{kt}$ — the truncated sum
*without* renormalisation. For the renormalised $\ell = \tilde\ell - \sum_k\log y_k$ the
update {eq}`eq-mstep-proportions` is not the exact M-step, because $y_k$ depends on
$\pi_c$. This is why the reported log-likelihood can decrease slightly from one iteration
to the next when truncation is active, and it is the origin of the original software's
rule for raising $c_0$ ({ref}`below <est-adaptive-c0>`).
:::

For M3 the proportions are confined to a box, $\pi_c \in [10^{-7}, 0.5]$ and
$p_1 \in [10^{-7}, 0.5]$ (the bounds of the original; `EMConfig.pic_bounds`,
`EMConfig.p1_bounds`). Because the binomial and multinomial parts are concave in their
own parameter, clipping the unconstrained update to the box *is* the constrained maximiser,
so monotonicity is preserved. The upper bound on $p_1$ encodes the convention that
cluster 1 — started with the larger variance — is the minority cluster.

(est-mstep-variance)=
### M-step: variance components

The second part of $Q$,

$$
Q_v(\sigma^2, a) = \sum_k\sum_t w_{kt}\Bigl[-\tfrac12\log v_{kt} - \frac{\hat\beta_k^2}{2v_{kt}}\Bigr],
\qquad v_{kt} = r_k\sum_h J_{th}\sigma_h^2 + s_k^2 + a ,
$$

has no closed-form maximiser. Following the original, the software takes a fixed number
of Newton–Raphson steps on it (default `newton_steps=2`) with the weights held fixed.
Write $\phi = (\sigma_1^2,\dots,\sigma_H^2, a)$ and

$$
g_1(v) = \frac{\partial \log f}{\partial v} = \frac12\Bigl(\frac{\hat\beta^2}{v^2} - \frac1v\Bigr),
\qquad
g_2(v) = \frac{\partial^2 \log f}{\partial v^2} = \frac12\Bigl(\frac1{v^2} - \frac{2\hat\beta^2}{v^3}\Bigr).
$$

Since $\partial v_{kt}/\partial\sigma_h^2 = r_k J_{th}$ and $\partial v_{kt}/\partial a = 1$,

$$
\begin{aligned}
\frac{\partial Q_v}{\partial \sigma_h^2} &= \sum_{k,t} w_{kt}\, g_1(v_{kt})\, r_k J_{th}, &
\frac{\partial Q_v}{\partial a} &= \sum_{k,t} w_{kt}\, g_1(v_{kt}), \\[4pt]
\frac{\partial^2 Q_v}{\partial \sigma_h^2\,\partial \sigma_{h'}^2} &= \sum_{k,t} w_{kt}\, g_2(v_{kt})\, r_k^2 J_{th} J_{th'}, &
\frac{\partial^2 Q_v}{\partial \sigma_h^2\,\partial a} &= \sum_{k,t} w_{kt}\, g_2(v_{kt})\, r_k J_{th}, \\[4pt]
&& \frac{\partial^2 Q_v}{\partial a^2} &= \sum_{k,t} w_{kt}\, g_2(v_{kt}),
\end{aligned}
$$ (eq-q-derivatives)

and one step is $\phi \leftarrow \phi - (\nabla^2 Q_v)^{-1}\nabla Q_v$. These are the
formulas of `onestep_varcomponent()` in the original C++, to which the kernel is tested
term by term.

Every sum in {eq}`eq-q-derivatives` has the form
$\sum_t (\text{function of } J_t) \times \sum_k (1, r_k \text{ or } r_k^2)\,w_{kt}\,g(v_{kt})$.
The kernel therefore returns five vectors of length $T$ per pass (`a0`, `a1`, `d0`, `d1`,
`d2`) and the small sums over configurations are done on the host
({meth}`genesis_gwas.core.likelihood.CompositeLikelihood.em_pass`); see
[Implementation](implementation.md) for why this matters.

The first Newton step is evaluated at the same $\phi$ that defined the weights, so it
re-uses $v_{kt}$ and is **fused with the E-step** into a single pass over the data. Each
further Newton step needs one more pass. An EM iteration with the default settings is
thus two passes, against four in the original (weights, proportions, and two Newton
steps).

(est-safeguards)=
### Safeguards

Two Newton steps are not an exact M-step, and $Q_v$ is not concave in the variances, so
the iteration is a *generalised* EM algorithm and can misbehave from poor starting values.
The original responds with ad-hoc resets (for example, a variance above 1 is replaced by
$10^{-5}$). The default policy here keeps the same iteration but guards it differently
({func}`genesis_gwas.core.em.run_em`):

Ascent direction
: If the Hessian of $Q_v$ is not negative definite, its eigenvalues (after diagonal
  scaling) are replaced by minus their absolute values. The result is a direction along
  which $Q_v$ increases; when the Hessian *is* negative definite it is the plain Newton
  step ({func}`genesis_gwas.core.em.newton_direction`).

Feasibility
: A negative variance component is set to $10^{-12}$ (the component has collapsed), and
  $a \le -\min_k s_k^2$ is moved to $-\min_k s_k^2/2$; both are the original's rules.

Monotonicity
: The next E-step yields $\tilde\ell$ at the proposed parameters at no extra cost. If it
  is lower than at the current parameters, the step is halved (up to 30 times) until it is
  not. Because the M-step direction is an ascent direction of $\tilde\ell$ at the current
  point, a short enough step always succeeds; if none is found the run ends with status
  `stalled`.

Constrained Newton step
: With `stratification=False` the Newton step is taken in $\sigma^2$ alone. (The original
  takes the joint step and then zeroes $a$, which does not converge to the constrained
  optimum.)

Whenever no safeguard is triggered — the normal case for real data — the iterates are
identical to those of the original to floating-point accuracy. `legacy=True` switches to a
literal transcription of the original control flow, used to reproduce its numbers exactly.

### The truncation level during the EM

The truncation level $c_0$ is held fixed while the EM runs. (The original software raises
it by one whenever the reported log-likelihood $\ell$ decreases from one iteration to the
next, and stops if that happens at $c_0 = 20$; `legacy=True` reproduces this.) How $c_0$ is
chosen in the default mode is the subject of {ref}`est-adaptive-c0`.

(est-convergence)=
### Stopping rule of the EM

Left to itself, the EM stops at the first iteration at which every parameter changed by
less than its absolute tolerance *and* the relative change of $\ell$ between the last two
E-steps is below `tol_loglik`. The defaults are those of the original software:

| | $\pi_c$ | $p_1$ | $\sigma_h^2$ | $a$ | $\ell$ (relative) | maximum iterations |
|---|---|---|---|---|---|---|
| tolerance | $10^{-6}$ | $10^{-5}$ | $10^{-8}$ | $10^{-9}$ | $10^{-6}$ | 3000 |

These are *absolute* tolerances on parameters of very different magnitude: for
$\sigma^2 \approx 6\times10^{-5}$ a change of $10^{-8}$ is a relative change of
$2\times10^{-4}$. More importantly, a small change per iteration says little about the
distance to the limit when convergence is slow ({ref}`est-polish`): the rule fires well
before the maximum is reached.

In the default mode this rule is rarely what ends a fit, because Newton's method takes
over earlier ({ref}`est-schedule`). It decides in `legacy` mode, with
`accelerate=False`, and whenever Newton's method is declined throughout.

### One EM iteration in pseudo-code

```text
Input:  data (β̂_k, s²_k, ℓ_k, N*_k), model (H), current θ, c0
        statistics from the E-step at θ: weights w, ℓ̃, ℓ, A, ∇Q_v, ∇²Q_v

π_c ← Σ_h A_h / Σ_k N*_k ;  p_h ← A_h / Σ A                    # closed form; clip to bounds (M3)
φ ← (σ², a)
for step = 1 … newton_steps
    φ ← φ + ascent_direction(∇Q_v, ∇²Q_v)                      # Newton step if ∇²Q_v is negative definite
    project φ onto {σ² ≥ 0, a > −min s²}
    if step < newton_steps: (∇Q_v, ∇²Q_v) ← pass(θ; evaluate at φ)     # weights still from θ
θ' ← (π_c, p, φ)
(w', ℓ̃', ℓ', …) ← pass(θ')                                      # E-step for the next iteration
while ℓ̃' < ℓ̃: θ' ← θ + (θ' − θ)/2 ; (w', ℓ̃', ℓ', …) ← pass(θ')        # monotonicity safeguard
em_converged ← all |θ' − θ| < tol  and  |ℓ − ℓ_previous| / |ℓ_previous| < tol_loglik
θ ← θ'
```

(est-starting-values)=
## Starting values

The starting point comes from an unweighted LD-score regression
({func}`genesis_gwas.core.start.ldsc_regression`): regress $z_k^2$ on
$x_k = \ell_k\,\bar n/M$, where $\bar n$ is the mean sample size. By the
[moment identity](model.md) $E[z_k^2] = 1 + na + n h^2\ell_k/M$, the slope estimates
$h^2 = M\pi_c\sigma^2$. Then, for a chosen starting proportion $\pi_c^{(0)}$ (default
0.01),

$$
\sigma^{2\,(0)} = \frac{\text{slope}}{M\,\pi_c^{(0)}}, \qquad
a^{(0)} = \frac{\text{intercept} - 1}{M}.
$$

For the height data the regression gives intercept 1.169 and slope 0.343, hence the start
$(0.01,\ 3.20\times10^{-5},\ 1.58\times10^{-7})$.

:::{note}
Dimensionally, the starting value of $a$ should divide by $\bar n$, not $M$. The
expression is kept as in the original because it only affects where the iteration
*starts*, and identical starts make trajectories comparable with the original's.
:::

**M3 from M2.** The three-component model is started from a two-component estimate
$(\hat\pi_c, \hat\sigma^2, \hat a)$ with one ninth of the susceptibility SNPs in a cluster
of five times the variance and the rest in a cluster of half the variance
({func}`genesis_gwas.core.start.m3_start_from_m2`):

$$
\bigl(\hat\pi_c,\ \tfrac19,\ 5\hat\sigma^2,\ \tfrac12\hat\sigma^2,\ \hat a\bigr).
$$

`fit(data, 3, starting=m2)` uses a finished M2 fit; `fit(data, 3)` alone first runs an
internal M2 fit (in `legacy` mode with the loosened tolerances of the original).

**Several starts.** A mixture likelihood can have more than one local maximum.
`starting_pic=[...]` (or `--starting-pic 0.001,0.01,0.05`) runs the optimisation from each
starting proportion and keeps the run with the highest log-likelihood; all runs are listed
in `FitResult.diagnostics["multistart"]`. The tutorial shows that for the height data all
starts lead to the same maximum ({ref}`tut-multistart`).

(est-polish)=
## Newton's method

### Why the EM alone is not enough

Near its limit the EM converges linearly, at a rate governed by the fraction of
information that is *missing* because the configurations are unobserved
([Dempster, Laird & Rubin, 1977](../reference/bibliography.md#ref-dempster1977)). Here
that fraction is large along one direction: making $\pi_c$ larger and $\sigma^2$ smaller
in proportion leaves the heritability — the best determined quantity — unchanged, and the
data separate the two only through the tails of the $z$ distribution. Along this ridge
successive EM iterates creep.

{numref}`fig-em-trajectory` shows the height fits when the EM is run to its own stopping
rule (`accelerate=False`, $c_0 = 10$). After 327 (M2) and 853 (M3) iterations all changes
are below
the tolerances and the EM stops, yet $\pi_c$ is still moving: the M3 estimate at that
point is 0.36 standard errors (5 %) below the value at the maximum, and $\sigma_2^2$ is
0.46 standard errors above it.

```{figure} ../_static/figures/em-trajectory.svg
:name: fig-em-trajectory
:alt: Four panels showing, for the two- and three-component height fits, the gap of the log-likelihood to its maximum and the estimate of the proportion of susceptibility SNPs against EM iteration, the point where the EM stopping rule fires, and the point where Newton's method takes over by default.

**The EM run to its own stopping rule, and where Newton's method takes over, for the
height GWAS.** Top: distance of the composite log-likelihood from its maximum (log
scale). Bottom: $\pi_c$ along the EM iterations (blue); the dashed line is the maximiser.
The open circle marks where the EM stopping rule fires. The orange arrow shows the
default: after 50 EM iterations Newton's method reaches the maximiser in a few steps. M2
is started from $\pi_c = 0.01$, M3 from the M2 estimate; the truncation level is held at
$c_0 = 10$ throughout.
```

Estimates stopped in this way depend on the starting value and on the number of
iterations, and they are biased towards the start. The direction seen here — $\pi_c$ too
small and variances too large when starting from a model with fewer components — is the
direction of the finite-sample bias reported in the paper's simulations.

### The Newton iteration

Newton's method is applied to $\tilde\ell$ itself
({func}`genesis_gwas.core.em.newton_polish`):

$$
\theta \leftarrow \theta + I(\theta)^{-1} \sum_{k} S_k(\theta),
$$ (eq-polish)

with $S_k$ the score of marker $k$ and $I$ the observed information, both derived in the
[next chapter](inference.md) and needed anyway for the standard errors. Convergence is
quadratic. The iteration is guarded so that it can only help:

- A step is taken only if the information matrix of the free parameters is positive
  definite, so that the step is an ascent direction and the iterate stays in a region
  where the likelihood is locally concave. Otherwise the attempt ends with status
  `not_positive_definite` (typical of a three-component model that is
  {ref}`not identified <inf-bounds>`, and of points far from the maximum).
- A step is halved until all parameters are feasible and $\tilde\ell$ does not decrease
  (status `no_ascent` if no such step exists).
- Parameters on a bound of the M3 box, and variance components that have collapsed to
  zero, are held fixed; a step that would cross a bound stops on it and fixes the
  parameter there (an active-set strategy).
- It stops with status `converged` when the Newton step is smaller than $10^{-5}$ naive
  standard errors, $\max_j |\Delta_j| / \sqrt{[I^{-1}]_{jj}}$, and gives up after 25 steps
  (`max_steps`).

(est-schedule)=
### When Newton's method takes over

Newton's method needs a starting point in the basin of the maximum; the EM needs no such
thing but is slow at the end. The default (`accelerate=True`) combines them
(`genesis_gwas.fit._em_then_newton`):

```{mermaid}
flowchart TD
    S([starting values]) --> EM["EM block<br/>50, then 100, 200, … iterations"]
    EM --> STALL{"EM stalled?"}
    STALL -- yes --> OUTE([last EM iterate, with a warning])
    STALL -- no --> NEWTON["Newton's method from the EM iterate<br/>(guarded: positive definite information,<br/>no decrease of the likelihood)"]
    NEWTON --> ACC{"Newton<br/>converged?"}
    ACC -- yes --> TAIL{"prior mass discarded by<br/>the truncation at θ̂<br/>above tolerance?"}
    TAIL -- no --> OUT([estimate θ̂])
    TAIL -- "yes, and c<sub>0</sub> below its ceiling" --> RAISE["raise c<sub>0</sub>;<br/>one EM iteration from θ̂"] --> NEWTON
    ACC -- no --> DONE{"EM stopping rule met,<br/>or iteration limit?"}
    DONE -- no --> EM
    DONE -- yes --> OUT2([best point found, with a note])
```

1. Run `newton_after` EM iterations (default 50).
2. Attempt Newton's method from the current EM iterate. If it converges, that is the
   solution at the current truncation level.
3. Otherwise discard the attempt and continue the EM *from where it was* for a block
   twice as long (100, 200, … iterations), then try again. A declined attempt usually
   costs two passes over the data: one to evaluate the likelihood and one to find that
   the information is not positive definite.
4. If the EM meets its own stopping rule, or the iteration limit, before Newton's method
   has converged, the search ends there. A Newton iterate that improved the likelihood
   without converging is reported with a note; if the information was not positive
   definite the EM iterate is reported.
5. Check the truncation at the solution ({ref}`est-adaptive-c0`). If the level has to be
   raised, one EM iteration from the solution is followed by Newton's method at the new
   level.

One exception shortens the search. If Newton's method declines and the three-component
iterate has collapsed onto the two-component model — the two variances equal to within
1 %, or one of them zero — the fit is stopped at once with status `not_identified`
({ref}`inf-bounds`). Continuing would only drift along a flat direction of the
likelihood, for thousands of iterations in the original software.

For the height data the first attempt succeeds for both models: 50 EM iterations followed
by 6 (M2) and 8 (M3) Newton steps, instead of 327 and 853 EM iterations. At the same
truncation level the estimates agree with those of the EM run to its stopping rule and
then finished by Newton's method — the sequence obtained with `accelerate=False` — to
within $2\times10^{-9}$ standard errors.

:::{admonition} What is and is not guaranteed
:class: note

An accepted Newton run ends at a stationary point of $\tilde\ell$ with positive definite
observed information — a local maximum — whose likelihood is at least that of the EM
iterate it started from. Like the EM, it finds *a* local maximum. If the likelihood has
several, an early Newton start could in principle end in a different one than the EM
would have crept to. Two tools check this: `starting_pic=[…]` compares several starts,
and `accelerate=False` reproduces the slower sequence.
:::

Related options: `polish=False` (`--no-polish`) turns Newton's method off and reports the
point where the EM stopped; `legacy=True` does the same and also restores the original's
control flow. `FitResult.diagnostics["polish"]` records the number of Newton steps, their
status, the EM iterate they started from (`em_estimates`) and the gain in log-likelihood;
if the truncation level was raised, these refer to the final level.

:::{admonition} What this changes compared with the original software
:class: important

The point estimate is the maximiser of the composite likelihood rather than the point
where the EM happened to stop. For the height example the two differ by up to half a
standard error; the measured differences are tabulated in
[Differences from the original](../reference/differences.md).
:::

(est-adaptive-c0)=
## Choosing the truncation level

The likelihood sums over at most $c_0$ tagged susceptibility SNPs per component
([The statistical model](model.md)). Whether that is enough depends on the answer: the
prior mass a marker places beyond the truncation grows with $N^*_k\pi_c$. The default
therefore checks the truncation *at the solution* (`genesis_gwas.fit`):

1. Solve at $c_0 = 10$ as described above.
2. Compute the prior mass discarded by the truncation at the estimate, averaged over
   markers, $\overline{1-y} = K^{-1}\sum_k \bigl(1 - y_k(\hat\theta)\bigr)$.
3. If it exceeds `tail_tol` $= 10^{-6}$, move to the smallest $c_0$ at which it would be
   at most a tenth of that, take one EM iteration from the current estimate, apply
   Newton's method, and repeat from step 2. Stop at `c0_max` $= 20$.

Each level is recorded in `FitResult.diagnostics["c0_path"]` (level, discarded mass,
estimates, log-likelihood), the final one is `FitResult.c0`, and a warning is issued if
the discarded mass is still above the tolerance at the ceiling. `c0_max` equal to `c0`
fixes the level.

The rule depends on the data only. The original's rule — raise $c_0$ when the
log-likelihood happens to decrease during the EM — depends on the path taken from the
starting values, so that two starts can end with different $c_0$ and therefore different
estimates; see [Differences from the original](../reference/differences.md).

### How much it matters

The script `docs/_scripts/examples/truncation.py` refits the height data at fixed levels:

```{literalinclude} ../_scripts/output/truncation.txt
:language: console
```

```{figure} ../_static/figures/truncation-sensitivity.svg
:name: fig-truncation-sensitivity
:alt: Two panels showing the estimated proportion of susceptibility SNPs against the truncation level. For the full height data the two-component estimate is flat and the three-component estimate rises slightly between 10 and 14 and is flat afterwards. For chromosome 22 alone the estimate keeps rising with the truncation level and its standard error grows, while the heritability stays constant.

**Sensitivity of the estimates to the truncation level.** (a) Full height GWAS:
$\hat\pi_c$ relative to its value at $c_0 = 20$, for M2 and M3; the open markers are the
levels chosen by the default rule. (b) Chromosome 22 alone, M2: $\hat\pi_c$ with its
standard error (left axis) and the heritability (right axis).
```

Three regimes:

- **Height, M2.** At $c_0 = 10$ the discarded mass is $5\times10^{-8}$, and higher
  levels change $\hat\pi_c$ by less than 0.01 %. The default stays at 10.
- **Height, M3.** $\hat\pi_c$ is twice as large as under M2, the discarded mass at
  $c_0 = 10$ is $3\times10^{-6}$, and the estimate is still moving: $\hat\pi_c$ rises by
  2.5 % between $c_0 = 10$ and 14 and by less than 0.1 % beyond. The default moves to
  14.
- **One chromosome.** With 13,230 markers the data determine the heritability but cannot
  tell many tiny effects from fewer larger ones. The likelihood is nearly flat in that
  direction, and the truncation — which caps the number of susceptibility SNPs a marker
  can tag — is what stops $\hat\pi_c$ from growing. The estimate then follows $c_0$,
  from 0.029 at $c_0 = 10$ to 0.088 at $c_0 = 40$, where its standard error is as large
  as the estimate; the heritability stays at 0.0054–0.0058 throughout. The default ends
  at the ceiling with the warning "Truncation still matters at the ceiling".

The third case is the signature of a model that is not identified by the data, not of a
numerical problem: no truncation level gives a meaningful $\hat\pi_c$. A dependence of
the estimate on $c_0$ together with a large standard error is the warning sign; the
heritability remains interpretable.

(est-diagnostics)=
## Convergence diagnostics

Every fit that computes standard errors also reports how far the estimate is from a
stationary point of $\tilde\ell$: the Newton step {eq}`eq-polish` at the final estimate,
divided by the standard errors (`FitResult.diagnostics["newton_distance_se"]`, shown as
"Convergence check" in the summary). After Newton's method has converged it is of the
order of $10^{-9}$ or smaller. Without it (`polish=False`, `legacy=True`) the number tells
you what the early stop cost; a warning is issued when it exceeds 0.25.

Other fields worth knowing:

| Field | Meaning |
|---|---|
| `status` | of the reported estimate: `converged`, `not_identified` (three-component fit collapsed onto M2), `max_iter` (limit reached), `stalled` (no ascent step found), `legacy_break` (the original's early exit) |
| `n_iter` | EM iterations performed (50 when the first Newton attempt succeeds, plus one per increase of $c_0$) |
| `diagnostics["em_status"]` | how the EM phase ended: `newton` (a Newton run ended the warm-up), `converged` (its own stopping rule), `max_iter`, `stalled`, `legacy_break` |
| `diagnostics["polish"]` | Newton's method: `status`, `n_steps`, `em_estimates` (where it started), `loglik_gain`, `fixed` parameters, `accelerated` |
| `diagnostics["em_backtracks"]` | step halvings by the monotonicity safeguard (0 for well-behaved data) |
| `c0`, `diagnostics["c0_path"]` | final truncation level; discarded prior mass, estimates and log-likelihood at each level tried |
| `diagnostics["truncation"]` | discarded prior mass at the final estimate (`mean_tail`, `max_tail`) and the gap $G$ between the two log-likelihoods |
| `diagnostics["multistart"]` | one row per starting value |
| `history` | per-iteration log-likelihoods, parameters and $c_0$ of the EM phase |
