# Fitting the model

This chapter fits both models to the height GWAS, from the command line and from Python,
and explains how to read the output, how the two models are compared, and what the
warnings mean.

## One command

`genesis fit` reads the summary statistics, applies the quality control of the
[previous chapter](data.md), fits M2 and then M3 (started from the M2 estimate), compares
them and writes the results. The optional flags add a QQ plot, the per-SNP posterior table
and projections for three future sample sizes:

```{literalinclude} ../_scripts/output/fit_cli.txt
:language: console
:end-before: Fitting the three-component model
```

The log continues with the three-component model and the comparison:

```{literalinclude} ../_scripts/output/fit_cli.txt
:language: text
:start-at: Fitting the three-component model
```

### What the log tells you

Reading the first block from the top:

`Reference`
: the LD reference and LD definition in use.

`Read … rows (Z from …; N from …)`
: how the $z$ statistic and the sample size were obtained. Check it
  ({ref}`tut-effective-n`).

`QC: …`
: one line per filter that removed anything, then the number of markers analysed.

`Compute`
: the backend and the number of CPU threads or GPUs.

`iter … logLik … [ … ]`
: progress of the EM algorithm: iteration, log-likelihood, truncation level and the
  current parameter values (every 50 iterations; `--print-every` changes that).

`Truncated prior mass … re-solving with c0 = …`
: the likelihood counts at most $c_0$ susceptibility SNPs per cluster among those a
  marker tags, starting with $c_0 = 10$. After solving, the software checks how much
  probability that leaves out *at the solution*; if it is more than one part in a million
  on average, it solves again with more terms. Here the three-component fit moves from 10
  to 14, which raises $\hat\pi_c$ by 2.5 %. The two-component fit needs no change.
  {ref}`est-adaptive-c0` has the rule and the reasons.

The summary block that follows is described in {ref}`tut-reading-summary`.

### Files written

```{literalinclude} ../_scripts/output/fit_files.txt
:language: console
```

| File | Content |
|---|---|
| `PREFIX.M2.json`, `PREFIX.M3.json` | the complete result of each fit: estimates, covariance matrix, diagnostics, settings, provenance. Everything downstream reads these. |
| `PREFIX.best.json` | a copy of the model selected by BIC |
| `PREFIX.estimates.tsv` | all estimates and derived quantities of both models in long format, for spreadsheets and for collecting many traits |
| `PREFIX.summary.txt` | the human-readable summaries, the model comparison and the QC table |
| `PREFIX.log` | the full log |
| `PREFIX.qq.png`, `PREFIX.qq.tsv.gz` | QQ plot of the selected model and the numbers behind it (`--qq`) |
| `PREFIX.posterior.tsv.gz` | per-SNP posterior summaries under the selected model (`--posterior`) |
| `PREFIX.projection.tsv` | projections for the requested sample sizes (`--project`) |

[Output files](../reference/outputs.md) documents every column. The long-format table
looks like this:

```{literalinclude} ../_scripts/output/fit_estimates.txt
:language: console
```

## The same in Python

```{literalinclude} ../_scripts/examples/fit_height.py
:language: python
```

```{literalinclude} ../_scripts/output/fit_height.txt
:language: console
```

{func}`genesis_gwas.fit` returns a {class}`genesis_gwas.FitResult`. The fields used most:

```python
m3.estimates, m3.se          # parameter vector and standard errors, in the order m3.param_names
m3.covariance                # sandwich covariance matrix of the parameters
m3.derived["n_ssnp"]         # derived quantities, each with .estimate and .se
m3.loglik, m3.bic, m3.ds     # composite log-likelihood, modified BIC, effective number of parameters
m3.status, m3.n_iter, m3.c0  # how the optimisation ended
m3.warnings                  # everything worth a second look, as text
m3.diagnostics               # convergence, truncation, starting values, …
```

`m3.to_json(path)` writes exactly the file the command line writes, and
`gg.FitResult.from_json(path)` reads it back.

(tut-reading-summary)=
## Reading the summary

`GWAS markers analysed (K)`
: markers that passed quality control and are in the reference. The likelihood is a
  product over these.

`Reference-panel SNPs (M)`
: the number of SNPs the proportions refer to: $\pi_c$ is a fraction of $M$, not of $K$.

`Optimisation`
: how the estimate was reached — the number of EM iterations, the final truncation level
  $c_0$ and the number of Newton steps at that level. `converged` is what you want to
  see.

`Composite log-likelihood`
: the maximised value. Only differences between models fitted to the same markers mean
  anything.

`Model selection`
: the modified BIC and AIC, and $d_s$, the *effective number of parameters*. $d_s$ is far
  larger than the three or five parameters of the model because neighbouring markers are
  correlated and each parameter is, in effect, counted many times
  ({ref}`inf-model-choice`).

`Parameter` table
: estimates and standard errors on the model's own scale. The standard errors come from
  the sandwich estimator and account for LD between markers.

`Derived quantity` table
: what you will usually report; [Chapter 4](interpreting.md) goes through it.

`Convergence check`
: the distance between the estimate and the nearest point where the likelihood equations
  are solved exactly, in units of the standard error. Anything below about 0.01 means the
  optimisation is not a source of error.

(tut-model-choice)=
## Choosing between M2 and M3

Two tools, to be used together.

**The modified BIC.**

$$
\mathrm{BIC} = -2\,\ell(\hat\theta) + d_s \log \bar n + 2\gamma\, d_s \log 5 ,
$$

where $\ell$ is the composite log-likelihood, $\bar n$ the mean sample size and
$\gamma = 0.5$ (`--bic-gamma`). The model with the *smaller* value is selected. For height
the three-component model wins by more than 1,000 units: the gain in log-likelihood of
about 595 dwarfs the penalty for an increase of $d_s$ by 11.

```python
choice = gg.select_model([m2, m3])
choice.best          # the selected FitResult
print(choice.summary())
```

**The QQ plot.** BIC ranks the two models against each other; it does not say whether
either describes the data. The QQ plot does. It compares the observed distribution of
P-values with the distribution *expected under the fitted model*, obtained by simulating
summary statistics from the model with the LD structure and sample sizes of the real
data.

```{figure} ../_static/figures/qq-height.png
:name: fig-qq-height
:alt: Two quantile-quantile plots of observed against expected minus log10 P-values for the height GWAS, one for the two-component model and one for the three-component model, each with a band of simulated data sets around the diagonal.

**Does the fitted model reproduce the observed P-values?** Observed $-\log_{10}P$ of the
height GWAS against the values expected under the fitted M2 (a) and M3 (b), with the 80 %
range of 100 data sets simulated from each model (a band barely wider than the line).
Points on the diagonal mean that the model generates data that look like the real data.
$\lambda_{obs}$ and $\lambda_{fit}$ are the observed and the model-predicted
genomic-control factors.
```

Under M2 the observed P-values leave the diagonal: a single normal cannot produce both
the bulk of small effects and the tail of larger ones, so the model predicts too many
moderately small P-values (the curve dips below the diagonal between $10^{-3}$ and
$10^{-5}$) and too few of the most extreme ones (it rises above beyond $10^{-6}$). Under
M3 the points follow the diagonal down to about $10^{-7}$ and stay close to it beyond. A
small excess at the extreme tail remains; the paper sees it for most traits and reads it
as a few loci whose effects are larger than the fitted distribution allows. For
psychiatric and cognitive traits the paper found M2 and M3 indistinguishable, and BIC
selected M2.

From the command line, `--qq` adds the plot for the selected model to a fit, and
`genesis qq` draws it for any saved fit; in Python use
{func}`genesis_gwas.qq_simulation` and {func}`genesis_gwas.plotting.qq_plot`.

:::{admonition} When M3 is selected but looks suspicious
:class: tip

Prefer M2 if the M3 fit carries a warning that it is *not identified*
({ref}`tut-warnings`), if its standard errors are as large as the estimates, or if the
two models' QQ plots are indistinguishable. {func}`genesis_gwas.select_model` already
refuses to select a three-component fit that is not identified.
:::

(tut-polish)=
## How the optimum is reached

The likelihood is maximised in two phases. An EM algorithm — the one of the original
software — moves from the starting values into the neighbourhood of the maximum. EM is
dependable but slow at the end, so after 50 iterations Newton's method takes over and
solves the likelihood equations in a handful of steps. The `Optimisation` line reports
both counts.

The original software runs the EM alone and stops when the parameters change by less than
a fixed amount per iteration. {numref}`fig-em-trajectory-tut` shows why that is not the
same as having arrived.

```{figure} ../_static/figures/em-trajectory.svg
:name: fig-em-trajectory-tut
:alt: Four panels showing, for the two- and three-component height fits, the gap of the log-likelihood to its maximum and the estimate of the proportion of susceptibility SNPs against EM iteration, the point where the EM stopping rule fires, and the point where Newton's method takes over by default.

**The EM run on its own, and where Newton's method takes over.** Top: distance of the
log-likelihood from its maximum. Bottom: $\pi_c$ along the EM iterations (blue) and its
value at the maximum (dashed). The open circle is where the EM stopping rule fires; the
orange arrow is the default path, Newton's method after 50 EM iterations. The truncation
level is held at 10 in this comparison.
```

When the stopping rule fires, $\pi_c$ is still drifting. You can see the cost with
`--no-polish`, which reports the point where the EM stops:

```{literalinclude} ../_scripts/output/fit_nopolish.txt
:language: console
```

The convergence check now reads 0.14 standard errors instead of $10^{-9}$. For the
three-component model the EM alone ends half a standard error from the maximum. The
script behind the figure runs both optimisers with the truncation level held at 10:

```{literalinclude} ../_scripts/examples/em_only.py
:language: python
```

```{literalinclude} ../_scripts/output/em_only.txt
:language: console
```

(The warning about truncation is expected here: the level is fixed at 10 on purpose,
which the default would have raised for the three-component model.)

Estimates that stop short depend on where the algorithm started. That is the reason this
package reports the maximiser, and it is the largest difference in the estimates from the
original software; see
[Differences from the original](../reference/differences.md) for the comparison and
{ref}`est-polish` for the algorithm.

Options that control this phase: `--newton-after N` (EM iterations before the first
Newton attempt), `--no-accelerate` (run the EM to its own stopping rule first; slower,
same answer), `--no-polish` (no Newton steps), `--c0`, `--c0-max` and `--tail-tol` (the
truncation level, its ceiling and the tolerance of the check) and `--legacy` (reproduce
the original software exactly).

(tut-multistart)=
## Starting values

Starting values come from an LD-score regression: its slope gives the heritability, and
with a starting proportion of susceptibility SNPs (default 0.01) that fixes the starting
variance. The three-component fit starts from the two-component estimate.

A mixture likelihood can have several local maxima, so it is worth checking that the
answer does not depend on the start. `starting_pic` (CLI: `--starting-pic`) accepts one
value or several; with several, each is run and the best is kept.

```{literalinclude} ../_scripts/examples/multistart.py
:language: python
```

```{literalinclude} ../_scripts/output/multistart.txt
:language: console
```

After 50 EM iterations the four runs are still far apart; after Newton's method they
agree to every digit shown, with the same log-likelihood. For height the maximum does not
depend on the starting value.

(tut-warnings)=
## Warnings and what to do about them

Warnings are shown on the terminal, stored in `FitResult.warnings` and written to
`PREFIX.summary.txt`.

### When the data are not enough: one chromosome

Small studies, and traits whose effects really follow one normal distribution, cannot
support everything the models ask for. Chromosome 22 of the height GWAS — one seventieth
of the data — shows both typical symptoms:

```{literalinclude} ../_scripts/examples/degenerate.py
:language: python
```

```{literalinclude} ../_scripts/output/degenerate.txt
:language: console
```

**M2: "Truncation still matters at the ceiling".** One chromosome determines the
heritability but cannot tell many tiny effects from fewer larger ones. The likelihood is
almost flat in that direction, $\hat\pi_c$ keeps growing as more terms are allowed in the
likelihood, and the software stops at its ceiling of $c_0 = 20$ and says so. The
heritability — 0.0057 with a standard error of 0.0011 — is the quantity to use;
$\hat\pi_c$ and the number of susceptibility SNPs are not determined by these data
({ref}`est-adaptive-c0` shows the estimate following $c_0$).

**M3: "not identified".** The two variances have merged, so the model has collapsed onto
M2 and the information matrix is singular. The software notices this as soon as Newton's
method declines to start, stops, and flags the fit: its standard errors are meaningless
(look at them), and no $d_s$, AIC or BIC is reported. Nothing about the clusters can be
learned. {func}`genesis_gwas.select_model` never selects a fit flagged in this way.
**Use the M2 results**, and of those the heritability.

### Other messages

| Message | Meaning | What to do |
|---|---|---|
| *Parameter(s) … ended on a bound* | In M3, $\pi_c$ or $p_1$ reached the edge of its allowed range ($10^{-7}$ to 0.5). Its standard error is not reported; the others are conditional on it. | Compare with M2 by BIC; usually M3 is over-parameterised. |
| *EM reached the iteration limit* | 3,000 EM iterations without meeting the stopping rule and without a successful Newton solve. | Rare. Raise `--max-iter`, try other `--starting-pic`, and check the data. |
| *EM stalled* | No step that increases the likelihood could be found. | Check the data (sample sizes, duplicated markers); try other starting values. |
| *about … standard errors from the likelihood stationary point* | The reported estimate is not at the maximum (only with `--no-polish`, or when Newton's method did not converge). | Use the defaults, or tighten the tolerances with `--tol-scale`. |
| *Newton polishing was skipped because the observed information is not positive definite* | The likelihood is flat in some direction at the EM estimate — typical of a model that is not identified. | Treat as "not identified". |
| *Truncation still matters at the ceiling* | Even at the largest allowed $c_0$ (20) the likelihood leaves out a non-negligible part of the prior, so $\hat\pi_c$ depends on $c_0$ (example above). | Rely on the heritability, which is stable. To see whether $\hat\pi_c$ settles, refit with a larger `--c0-max`; if its standard error grows as fast as the estimate, it is not identified. |
| *The three-component model is not identified* | The two variances merged or one cluster is empty (example above); status `not_identified`. | Use M2. |
| *No tagging neighbourhoods were supplied* | Standard errors treat markers as independent and are too small. | Prepare the data with an LD reference (the default). |

A **negative estimate of $a$** is not an error. $a$ absorbs any constant mis-scaling of
the test statistics, so it is negative when they are slightly under-dispersed, for
instance after genomic-control correction of the input.

## Options worth knowing

| Purpose | CLI | Python |
|---|---|---|
| fit only one model | `--model 2` or `--model 3` | `gg.fit(data, 2)` |
| fix $a = 0$ (no inflation term) | `--no-stratification` | `stratification=False` |
| CPU threads; GPU | `--threads 8`; `--device cuda` | `threads=8`; `device="cuda"` |
| another LD definition | `--ld-window-kb 2000 --ld-r2 0.05` | `gg.load_reference(window_kb=2000, r2_min=0.05)` |
| another ancestry | `--reference EAS`, `--reference AFR`, `--reference DIR` | `gg.load_reference("EAS")` |
| several starting values | `--starting-pic 0.001,0.01,0.05` | `starting_pic=[0.001, 0.01, 0.05]` |
| skip standard errors (faster exploration) | `--no-variance` | `variance=False` |
| reproduce the original software | `--legacy` | `legacy=True` |

The full list is in the [command-line reference](../reference/cli.md) and in
{func}`genesis_gwas.fit`.
