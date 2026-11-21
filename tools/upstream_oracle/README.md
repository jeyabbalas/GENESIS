# Upstream oracle

Scripts that run the **original** GENESIS implementation
([yandorazhang/GENESIS](https://github.com/yandorazhang/GENESIS), commit
`e4e6894d2e11ece6715d0572b0febceff1480f85`, 2020-09-06) and record its outputs as
golden fixtures. The Python package in this repository is tested against those
fixtures; nothing here re-implements upstream numerics.

Everything is produced by the genuine upstream code: `src/EM_EM.cpp` compiled
with `Rcpp::sourceCpp`, and the upstream `R/*.R` files `sys.source()`d into the
same environment. The upstream package is never installed (it bundles 1.6 GB of
data) and the upstream clone is never modified.

## Regenerating

Requirements: R with a C++ toolchain, `git`, `patch`, internet access for the
first run (CRAN binaries/sources of Rcpp, RcppArmadillo, jsonlite). Tested with
R 4.6.2 and Apple clang 21 on arm64 macOS only; the Linux/gcc branch of
`detect_openmp()` has not been exercised.

```bash
git clone https://github.com/yandorazhang/GENESIS.git /scratch/upstream   # ~3 GB
tools/upstream_oracle/run_all.sh /scratch/upstream /scratch/oracle_work 12
```

`run_all.sh` installs the three R packages into an **isolated library**
(`<work_dir>/rlib`; the user's and the system library are not touched), builds
upstream, and runs the four generators below. On macOS it also downloads the
OpenMP header matching R's bundled `libomp.dylib` from
<https://mac.r-project.org/openmp/> into `<work_dir>/omp` (never `/usr/local`).

| script | output | runtime (Apple M5 Pro) |
|---|---|---|
| `gen_synthetic.R` | `tests/fixtures/upstream/synthetic_{small,medium}.json.gz` | ~3 min + ~25 min, 1 thread |
| `gen_rfuncs.R` | `tests/fixtures/upstream/rfuncs.json` | ~1.5 min |
| `gen_mini_chr22.R` | `tests/data/mini_chr22/*`, `tests/fixtures/upstream/mini_chr22_{m2,m3}.json.gz`, `mini_chr22_stochastic.json` | ~22 min, 1 thread (two 3-component fits of ~9 min each) |
| `gen_height_full.R` | `tests/fixtures/upstream/height_full_m{2,3}.json` (summaries) and `<work_dir>/height_m{2,3}.json` + `*.f64` (per-SNP dumps, 189 MB / 265 MB, not committed) | ~25 min on 12 threads (M2 ~2 min, M3 ~23 min) |

Each script can be run on its own; see the usage line at the top of the file.
All paths are command-line arguments. Do not edit a generator while it is
running: `Rscript` reads the file incrementally.

Shared code lives in `lib/`:

* `lib/build_upstream.R` — patch-copy-compile-load (`build_upstream()`), the
  dataset override mechanism (`set_data_override()`), and
  `call_with_locals()`, which calls an upstream function and snapshots all of
  its local variables at exit through `base::trace(exit = ...)`. That is how
  quantities upstream computes but never returns (per-SNP scores `m_S`,
  neighbourhood sums `m_Sbar`, `inv_I`, `aic`, raw heritability and its SE,
  ...) are captured without touching the function body.
* `lib/oracle_utils.R` — fixture metadata, the JSON writer, and
  `ldsc_start_m2()`, a transcription of the starting-value block of
  `R/genesis.R` (lines 113-136). Every fixture that uses it also records that
  its result is `identical()` to the `starting` vector of the genuine
  `genesis()` run.

## How upstream is built

1. `src/EM_EM.cpp` is copied to `<work_dir>/src/` and
   `patches/EM_EM.cpp.patch` is applied. The patch removes one line, the
   `//[[Rcpp::plugins(openmp)]]` directive, because that plugin injects
   `-fopenmp`, which Apple clang rejects. No other byte of the file changes
   (pristine SHA-256 `cd48f975b1211804c2a8d06bffd4aed18ec038f664e1d7f4635b87316b8530ab`).
2. Flags are supplied through the environment instead:
   * `-DPI=M_PI` — `cdnorm()` uses R's legacy `PI` macro. `R_ext/Constants.h`
     used to provide exactly `#define PI M_PI`; the define is commented out in
     current R (seen with R 4.6.2), so without this flag upstream no longer
     compiles.
   * OpenMP, macOS: `-Xclang -fopenmp -I<work_dir>/omp/usr/local/include`,
     linked against `$R_HOME/lib/libomp.dylib` (shipped with CRAN's R).
     Linux: `-fopenmp` (untested). If no OpenMP runtime is found the build uses
     `stub/omp.h` (all loops serial, same numbers as one thread).
   * `FLIBS=` through a private `R_MAKEVARS_USER` when the gfortran directory
     recorded in R's configuration does not exist (CRAN's macOS R points at
     `/opt/gfortran`). The code needs no Fortran runtime at link time.
3. `Rcpp::sourceCpp()` compiles and loads the copy into a private environment;
   the upstream R files are `sys.source()`d into the same environment.

The build emits ten `-Wbitwise-instead-of-logical` warnings (upstream writes
`&`/`|` between booleans); they are harmless.

## Shims for upstream's side effects

The upstream functions are called as they are. Five names they reference are
replaced inside the private environment:

| upstream call | shim | why |
|---|---|---|
| `data("X")`, `data(list = ...)` | loads `<upstream>/data/X.RData` (or a registered in-memory override) into the environment enclosing the upstream functions | the package is not installed; overrides let the genuine `genesis()` run on synthetic and chr22-only LD tables |
| `library(doParallel)`, `library(foreach)` | no-op (other packages, e.g. `MASS`, are attached normally) | see next row |
| `makeCluster(cores)`, `registerDoParallel(cl)` | no-op | `genesis()` and `qqplotdata.simu()` create a PSOCK cluster of `cores` R processes, never use it for computation and never stop it |
| `foreach(i = ...) %do% {...}` | sequential loop evaluating the body in the calling frame | same semantics as foreach's `%do%`; avoids the dependency |

`cores` is still passed through to the C++ code, where it is the OpenMP thread
count.

## JSON conventions

* Doubles are written with 17 significant digits (`jsonlite::toJSON(digits = I(17))`)
  and round-trip exactly. (`digits = NA` keeps 15 digits only and does not.)
* Non-finite numbers are the strings `"NaN"`, `"Inf"`, `"-Inf"`, `"NA"`.
* Matrices are arrays of rows. Length-1 vectors are scalars.
* Every fixture has a `meta` block: upstream commit, R/compiler/package
  versions, `sizeof(long double)`, OpenMP mode and thread count, date.
* Result vectors of `EM_func` (10 values) / `EM_func3` (12 values) are stored
  whole: `i_last` (0-based index of the last completed iteration), final
  log-likelihood, the parameters, heritability, `c0`, and three CPU timings
  (ignore the timings; they differ on every run).

## What the fixtures contain

### `synthetic_small.json.gz` (K = 300), `synthetic_medium.json.gz` (K = 3000)

Inputs are generated in R with fixed seeds and **stored in the fixture**
(`dataset`): `SNP`, `Z`, `N`, and, in the row order of upstream's merged data
frame, `betahat = Z/sqrt(N)`, `varbetahat = 1/N`, `ldscore`, `Nstar`. `Nstar`
covers 1...710 (every value 1-12 is present), a few z-scores are extreme
(z² = 79.9, 25, exactly 0). `dataset.tagging_gwas_rows0` is the symmetric
neighbour structure among GWAS rows (CSR, 0-based, self included); for the
small set `dataset.reference` also holds the synthetic LD table in upstream's
format (comma-joined rsID strings, including SNPs absent from the GWAS and
rs numbers above the largest GWAS rs number).

* `m2.function_cases` — 4 parameter vectors × c0 ∈ {10, 3, 20}:
  `loglikelihood`, per-SNP log-likelihood (upstream `loglikelihood()` called on
  one SNP at a time), `weight` (full or leading rows; see `weight_rows`),
  posterior mean count `weight_Ej`, `update_pi1`, `onestep_varcomponent` and a
  second Newton step with the same weights, `S`, `SS`, `I`, `Ik` for the first
  five SNPs, `mixture_components_marginal` (some cases).
* `m3.function_cases` — 3 parameter vectors × c0 ∈ {10, 3, 5}: the `*3`
  analogues. Weight column `j1*(c0+1)+j2`. `S3` divides by K (upstream does).
  `I3` is stored exactly as upstream returns it, i.e. **asymmetric**.
* `m2.underflow_demo` — one input where upstream's linear-space densities
  underflow on platforms with 64-bit `long double`. Platform dependent; not a
  parity target.
* `m{2,3}.em_scenarios` — for each scenario: the exact call arguments,
  `trace` = result vectors of `EM_func(..., Meps = m)` for m = 1..60 (40 for
  the 3-component model on the medium set), `converged` = the default call
  (`Meps = 3000`), and `genesis` = everything the genuine `genesis()` computed
  for that scenario (estimates, `m_I`, `J`, `inv_I`, `var_est`, `sd_est`, `ds`,
  `aic`, `bic`, heritability/sSNP/sibling-risk values and SEs, 3-component
  extras, per-SNP scores `m_S`/`m_Sbar`: all rows for labels starting `a_`,
  `f_`, `e1_` on the small set, otherwise the first 50). Scenario labels:
  `a*` LD-score-regression start as in `genesis.R`; `b*` start with c0 = 3;
  `c*` `stratification = FALSE`; `d*` deliberately bad starts; `e*` (M3) p1 or
  pic at/above the 0.5 bound; `f*` user-supplied start at the generating
  values (the well-behaved baselines: for the 2-component model c0 never
  changes).

### `rfuncs.json`

`h2transfer`, `numInterval`, `dmixssnp`, `projection` (CI = FALSE),
`polyriskpredict`, `postmarginal` (synthetic 10-column data frame, inputs
included) and `preprocessing` (real LD table and HapMap3 list; documents the
filter rules and the NA handling). Calls that fail store `status = "error"`
and the R error message; warnings are stored too. The `stochastic` block holds
seeded single runs of `projection(CI = TRUE)` and `futuregc()`; they depend on
R's RNG stream and are for statistical sanity checks only.

### `mini_chr22_m2.json.gz`, `mini_chr22_m3.json.gz`, `mini_chr22_stochastic.json`

The genuine `genesis()` on `tests/data/mini_chr22/` (15,633 reference SNPs of
chromosome 22, 13,230 of them in the height GWAS), `M = 1070777`, c0 = 10,
`startingpic = NA` and `0.005`, both models, plus one 2-component fit with the
0.5 Mb / r² ≥ 0.2 table. Per fit: all estimates and intermediate matrices, the
first 200 rows of `m_S`/`m_Sbar`, the printed trace (6 digits) as a matrix, a
full-precision trace of the first 30 (M2) / 20 (M3) iterations, `postmarginal`
for the first 200 SNPs, `projection` at n = 253,288. `rows.snp_order` is the row
order (= order of the summary-statistics file). `mini_chr22_stochastic.json`:
seeded `projection(CI = TRUE)` and `qqplotdata.simu()` (100 simulations)
summaries.

All chr22 fits keep c0 = 10. The 2-component fits are well behaved (445 and
474 iterations). The 3-component fits (2490 iterations each) are degenerate:
chr22 alone does not support two clusters, the two variance components
coalesce (7.9072e-06 vs 7.9067e-06, p1 = 0.497), the information matrix is
nearly singular, the standard error of `sigmasq1` is `NaN`, and
`projection(CI = TRUE)` stops with "'Sigma' is not positive definite". They
are good targets for the EM path and the point estimates, poor ones for the
variance part; the well-identified real-data 3-component target is
`height_full_m3.json`.

### Scenarios in which upstream never changes c0

These are the natural targets when comparing an implementation whose
truncation handling differs from upstream's:

* real data: every chr22 fit, both full-scale height fits;
* small set, 2-component: `c_no_stratification`, `f*` (three scenarios);
  3-component: `e1_p1_0.49_near_upper_bound` (and `f_start_two_clusters`
  during the 60 traced iterations; it changes c0 at iteration 80);
* medium set, 2-component: `a3_regular_startingpic0.002`,
  `c_no_stratification`, `f2*`, `f3*`; 3-component: `c_no_stratification`,
  `f3_start_two_clusters_no_stratification`.

### `height_full_m2.json`, `height_full_m3.json`

The full-scale runs of `examples/examples.md` (941,684 SNPs after the merge).
Same content as the mini fits (first 200 per-SNP rows), plus the comparison
with the numbers printed in `examples.md` and the follow-up calls of that file.
The per-SNP dumps stay in `<work_dir>`: `height_m{2,3}.json` and
`height_m{2,3}_scores_rowmajor.f64` / `_scorebar_rowmajor.f64`
(little-endian doubles, K × P row-major).

## Results of the full-scale runs (this platform)

`genesis(heightGWAS, filter = F, LDcutoff = 0.1, LDwindow = 1, c0 = 10, ...)`, M = 1,070,777,
K = 941,684 SNPs after the merge, 12 OpenMP threads.

| | 2-component (`startingpic = 0.005`) | 3-component (`starting` built from the M2 values printed in `examples.md`) |
|---|---|---|
| EM iterations | 117 | 849 |
| final c0 | 10 (never changed) | 10 (never changed) |
| wall time, whole `genesis()` call | 90 s (EM alone 8.7 s; the rest is the merge and the R-level sandwich) | 1349 s (EM alone 1253 s, 1.48 s per iteration) |
| estimates | 4.906835485e-03, 5.732970955e-05, 1.861481368e-06 | 8.899809138e-03, 9.476025339e-02, 1.458649888e-04, 2.227117935e-05, 1.567643185e-06 |
| composite log-likelihood | 3950387.6611 | 3950980.6949 |
| `ds`, BIC | 97.27, -7899471.73 | 104.68, -7900558.54 |
| max rel. difference to `examples.md`, estimates | 7.8e-06 | 1.2e-07 |
| max rel. difference to `examples.md`, covariance | 2.5e-05 | 4.1e-07 |

`examples.md` prints 7 significant digits. The 3-component run reproduces it to
that precision, including the asymmetry of the covariance matrix. The
2-component run agrees only to about five digits: the run stops at the same
iteration, but its path differs slightly from the one that produced the printed
values (that run started from the LD-score-regression start, which the
3-component example bypasses with an explicit `starting` vector). Use the
fixture values, not the `examples.md` values, as the precise 2-component target.

The 3-component start follows the example literally: after the line
`est <- c(4.906797e-03, 5.733005e-05, 1.861491e-06)` the 7-digit values are what
`starting` is built from. Pass `computed` as the last argument of
`gen_height_full.R` to start from the oracle's own 2-component estimate instead.

## Platform caveats

* **`long double`.** Upstream accumulates in `long double`. On arm64 macOS
  (and MSVC) that is a 64-bit double; on x86 Linux/macOS it is the 80-bit
  extended type. Fixtures generated here use the 64-bit variant
  (`meta.sizeof_long_double = 8`). No x86 build was run. It should agree
  with these fixtures up to rounding on well-conditioned quantities, and it
  should stay finite in situations where the 64-bit build underflows (see
  `underflow_demo`); decisions that hinge on the sign of a tiny
  log-likelihood difference (the c0 increments, item 8 below) can in principle
  fall the other way.
* **Threads.** The synthetic and chr22 fixtures use one thread. Regenerating
  both synthetic fixtures on the same machine reproduced every stored number
  exactly (389,679 values compared; only timings and dates differ). The
  full-scale runs use OpenMP reductions over 12 threads and were run once;
  expect agreement up to rounding, not bitwise, on a repeat.
* **Early stopping.** Upstream stops EM when the per-iteration change of every
  parameter is below an absolute tolerance. The estimates are therefore a
  point on the EM path, not the maximiser, and they depend on the start in
  the 5th-6th significant digit (compare the two `startingpic` fits in
  `mini_chr22_m2.json.gz`). The total score at the returned estimate is far
  from zero (`function_values_at_est.S`).
* **RNG-based functions** (`projection(CI = TRUE)`, `qqplotdata.simu()`,
  `futuregc()`) cannot be reproduced outside R.

## Upstream behaviours observed while building the oracle

Worked around (build / harness):

1. Does not compile against current R: `PI` is no longer defined by R's
   headers (fixed with `-DPI=M_PI`).
2. `//[[Rcpp::plugins(openmp)]]` + unconditional `#include <omp.h>` cannot be
   built with Apple clang without extra setup (patch + flags above).
3. `genesis()` and `qqplotdata.simu()` call `makeCluster(cores)` and never
   `stopCluster()`: every call leaks `cores` idle R worker processes for the
   life of the session. The cluster is not used for any computation
   (`foreach ... %do%` is sequential). Shimmed to no-ops.
4. `data()` calls inside functions require the installed package; replaced by
   the loader shim.
5. The rs-number dictionary (`modification_loc`) allocates a double vector as
   long as the largest rs number in the GWAS (17.9 million entries for the
   height data) and requires every SNP name to be `rs<digits>`.

Numerical / logical behaviours captured in the fixtures (see the fixture named
in parentheses):

6. **Sandwich self term counted twice** (`synthetic_*`: `genesis.m_Sbar_rows`,
   `J`). `TaggingSNPs` already contains the SNP itself, and `genesis()` adds
   `m_S` once more: `m_Sbar[k] = sum_{k' in tagging(k), k' in GWAS} S[k'] + S[k]`,
   `J = t(m_S) %*% m_Sbar`. Verified to 1e-13 against the stored neighbour
   lists.
7. **`Ik3` typo makes the 3-component information matrix asymmetric**
   (`I3`, `m_I`, `var_est`): entry `[3,1]` (R indexing) lacks the
   `E_w[r3*r1]` term (`rr(2,0) = rr(2,0);`). `inv_I %*% J %*% inv_I` is then
   asymmetric too, as in `examples.md`.
8. **c0 grows on any decrease of the log-likelihood, whatever its cause**
   (`em_scenarios[*].c0_path_first_iterations`, `genesis.printed_c0_changes`).
   Two triggers show up in the fixtures. (i) Early, non-monotone Newton
   updates: on the small set the LD-score-regression start sends c0 from 10 to
   16/17 within the first ten iterations. (ii) Near convergence, when pic
   approaches its limit from above: on the medium set the scenarios `a`, `a2`
   and `f` (pic decreasing along the path) end with c0 = 13 after three
   consecutive increments in their last three iterations, while `a3` (same
   data, pic increasing along the path) stays at c0 = 10. The monitored
   likelihood is renormalised by the truncated prior mass, `sum_k log y_k`
   with `y_k = P(Binomial(Nstar_k, pic) <= c0)`, whereas the EM updates
   maximise the un-renormalised one. Checked on scenario `a` at iteration 119:
   the un-renormalised log-likelihood rises by +9.19e-05, `sum_k log y_k` rises
   by +9.25e-05 because pic shrinks, so the monitored value falls by 6.3e-07
   and c0 is incremented; in `a3` the same term falls and no decrease ever
   occurs. The decisions rest on differences of about 1e-7 in a
   log-likelihood of about 1.2e4. The final c0, and through it the estimates,
   therefore depend on the start. On the real data (chr22 and full height) c0
   never moved.
9. **Exit through the `c0 >= 20` break** (e.g. small / M3 /
   `a_regular_startingpic0.01`). When c0 goes from 19 to 20 the `break` fires
   in the same iteration. The returned parameters and c0 (= 19) are those of
   the previous iteration, but the returned log-likelihood is evaluated with
   c0 = 20 at the parameters produced by the breaking iteration's M-step, which
   are not returned (and can violate the p1 <= 0.5 bound). `genesis()` then
   computes scores and information with c0 = 19. So the largest c0 a fit can
   report is 19.
10. **3-component bound handling** (small / M3 / `a`, `d2` for the bound;
    `e2`, `e3` for starts above it). At the p1 = 0.5
    bound the update alternates between "unconstrained step" and "clip", the
    log-likelihood drops every other iteration and c0 is incremented every
    other iteration until the break. A *starting* value with p1 or pic above
    0.5 skips `update_p3` in the first iteration, leaves `tem_p` at its
    initial value (zero with current Armadillo, uninitialised memory with old
    versions), sets pic = 0 and yields NaN for all 3000 iterations; `genesis()`
    returns NaN estimates without an error.
11. **3-component fits can end with sigmasq1 < sigmasq2** (small / M3 / `a`,
    `d2`): the outputs labelled "cluster with larger variance component" then
    describe the smaller one. Negative diagonal entries of `var_est` give
    `NaN` standard errors with only a warning.
12. **Bad starts return garbage silently** (small / M3 / `d1`: "converged"
    after 12 iterations with a = 0.285 and a heritability SE of 28).
13. **`stratification = FALSE`** keeps a at 0 but takes the Newton step from
    the joint (sigmasq, a) Hessian (`c*`, `f3*` scenarios); the sandwich still
    reports a standard error for a.
14. **Early stopping.** The total score at the returned estimate is far from
    zero and two starts give estimates that differ in the 6th digit
    (`mini_chr22_m2`: `function_values_at_est.S`, the two `fits`).
15. **`weight_loglikelihood3` and `loglikelihood3` differ in the last bits**
    for the same input (different summation expression); the reported value
    always comes from `loglikelihood*()`. `S3()` divides by K, `S()` does not.
16. **Underflow** (`m2.underflow_demo`): with 64-bit `long double`,
    `(1-pic)^Nstar` underflows for large Nstar, weights become NaN and the
    log-likelihood NaN.
17. **`preprocessing()`** (`rfuncs.preprocessing`): output rows follow the
    input order; a single `NA` z or N is kept (`length(inx) > 1`); with
    `filter = TRUE` any `NA` in N is an error (`quantile` without `na.rm`);
    the effect-size rule is `Z^2 <= 80`, and `sqrt(80)^2` evaluates to
    80.00000000000001, so such a row is removed.
18. **`postmarginal()` ignores a** (the last element of `est`) altogether.
19. **`projection(CI = TRUE)`** (`rfuncs.stochastic`): a < 0 only produces a
    warning (the NaN sits in a slot the power integrals never read); a == 0,
    which is what `stratification = FALSE` returns, is an error
    (`infinite or missing values in 'x'`). The asymmetric 3-component
    covariance is accepted because `MASS::mvrnorm` reads the lower triangle;
    a covariance that is not positive definite (degenerate 3-component fit,
    `mini_chr22_stochastic.projection_CI_m3`) is an error.
20. **`futuregc()`**: the 3-component branch calls `set.seed(iter*123)` and
    ignores `seeds`.
21. **`polyriskpredict` call in `examples.md`** passes `Sig2s = c(est[3], est[3])`
    for the 2-component model, i.e. a instead of sigmasq; the result is an AUC
    of 0.501 (`height_full_m2.examples_md_calls` stores both variants).
22. **Sibling-risk SE** is `sqrt(risk * sd_h2^2 / 2)` (verified against the
    captured values), not the delta-method `risk * sd_h2 / 2`.
23. **Starting value of a** is `(LDSC intercept - 1) / M`, not `/ mean(N)`:
    negligible for M = 1,070,777, large for small M (5e-5 on the synthetic
    sets, which is what triggers item 8 there).
24. **`numInterval()`** with `lower < 0` includes the point mass at zero and
    returns more than M SNPs.
