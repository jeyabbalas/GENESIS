# Validation

This page documents how the correctness of the implementation was established, the
measured agreement with the original software, and — explicitly — what has *not* been
verified. The numbers can be regenerated from the repository: the test suite, the
fixtures in `tests/fixtures/upstream`, and the tools in `tools/`.

## Approach

A re-implementation of a statistical method can be wrong in ways that still produce
plausible numbers, so three independent sources of truth are used.

1. **The original software itself.** The original R/C++ package
   ([yandorazhang/GENESIS](https://github.com/yandorazhang/GENESIS), commit `e4e6894`)
   was built in isolation and run on synthetic and real data (`tools/upstream_oracle`);
   so was its East Asian sibling
   ([yandorazhang/GENESISasian](https://github.com/yandorazhang/GENESISasian), commit
   `4df90ac`).
   Its outputs — function values, complete EM trajectories, everything `genesis()`
   computes, and the results of the pure-R downstream functions — are stored as
   fixtures with 17 significant digits. The C++ source is compiled unchanged except for
   the removal of one build directive (the OpenMP plugin line, which the compiler used
   rejects); the R functions are called as they are, with the package's data loading
   and its unused parallel-cluster set-up replaced by shims.
2. **A literal port.** `tests/upstream_port.py` transcribes the original C++ line by
   line into scalar Python (linear-space densities, the same loops, the same resets,
   the same typo). It shares no code with the package, is itself checked against the
   fixtures, and serves as an oracle for inputs that the fixtures do not cover.
3. **Mathematics.** Scores and the information matrix are compared with numerical
   derivatives of the log-likelihood; closed-form projection integrals with numerical
   quadrature; samplers with their theoretical moments; the LD builder with a
   brute-force computation over all pairs; and so on.

`legacy=True` is the bridge between the two code bases: it must reproduce the original
*exactly*, which pins down every kernel. The default behaviour differs from `legacy`
only by the changes listed in [Differences from the original](differences.md), each of
which has its own tests.

## Test suite

478 tests. Continuous integration runs them on every push for Python 3.10 to
3.13 on Linux and for Python 3.12 on macOS (`.github/workflows/ci.yml`); tests that
need external data or hardware — the full reference bundle, the 1000 Genomes genotypes,
a GPU — are skipped where these are absent (2 on the development machine, which lacks the LDSC European genotype files).

| Area | Tests | What is checked |
|---|---:|---|
| `tests/core` | 80 | kernels against the literal port and the upstream fixtures; numerical derivatives; EM policies; Newton's method; sandwich estimator; engine (threads, chunking, backends, shards) |
| `tests/analysis` | 256 | density, projections, risk prediction, posterior summaries, simulation, plots; comparisons with the original R functions |
| `tests/ld` | 68 | PLINK reader; LD computation against brute force; bundle round trip; null statistics; VCF conversion (by identifier and by position); chain-file liftover |
| `tests/io` | 39 | parsing of summary statistics, QC, matching by identifier and position, reference bundles (GRCh37 and GRCh38), download logic |
| `tests/test_fit.py`, `tests/test_cli.py` | 30 | end-to-end fits and the command line on real chromosome-22 data |
| `tests/test_full_scale.py` | 5 | genome-wide parity with GENESIS (European reference) and GENESISasian (East Asian reference); these need the full references, see below |

```bash
pytest -q                                              # everything that needs no external data
GENESIS_DATA_DIR=/path/to/data pytest tests/test_full_scale.py        # genome-wide parity (see below)
GENESIS_1000G_PLINK=/path/to/1000G_EUR_Phase3_plink pytest tests/ld   # LD builder against the original tables, chromosome 22
```

The genome-wide tests need the reference bundle with the example GWAS
(`genesis data fetch --components default,examples --dest /path/to/data`); the
three-component one is additionally marked `slow` (849 EM iterations on 941,684 SNPs).
The two East Asian tests need `eas_1kg_hm3` with its null replicates and pairwise $r^2$
(`genesis data fetch EAS --components default,nullz,pairld`, or `GENESIS_EAS_BUNDLE`
pointing at a bundle directory).

## Parity with the original fitting code

### Function level

Synthetic data sets with $K = 300$ and $K = 3{,}000$ markers that include the awkward
cases ($N^*$ below `c0`, $z = 0$, $z^2 \approx 80$); 12 combinations of parameters and
`c0` for the two-component model and 9 for the three-component model. Maximum relative
difference from the original over all combinations:

| Quantity (original function) | M2, $K$ = 300 / 3,000 | M3, $K$ = 300 / 3,000 |
|---|---|---|
| log-likelihood (`loglikelihood`, `loglikelihood3`) | 1.4e-15 / 3.5e-15 | 1.2e-15 / 2.7e-15 |
| posterior weights, absolute difference (`weight`, `weight3`) | 8.8e-14 / 9.8e-14 | 7.5e-14 / 8.8e-14 |
| proportion update (`update_pi1`, `update_p3`) | 7.4e-15 / 1.4e-14 | 1.1e-14 / 2.4e-14 |
| Newton step for the variances (`onestep_varcomponent`, `onestep_varcomponent3`) | 3.9e-14 / 2.5e-13 | 7.7e-14 / 2.2e-13 |
| per-marker scores (`SS`, `SS3`) | 9.2e-14 / 2.1e-13 | 1.4e-13 / 2.3e-13 |
| information matrix (`I`, `I3`) | 6.9e-13 / 4.9e-12 | 5.8e-14 / 2.2e-14 |

For `I3` the comparison is with the original's asymmetric matrix, reproduced by leaving
out the term that the original omits; the package's own, symmetric matrix equals the
original with that one entry replaced by its transpose.

### EM trajectories

For 12 two-component and 12 three-component scenarios per data set — regular starts,
`c0 = 3` (the truncation binds, so `c0` changes), `stratification = FALSE`, absurd
starting values (the reset branches), starts on and beyond the proportion bounds — the
original was run for 1, 2, ..., 60 iterations (40 for the three-component model on the
larger data set) and to convergence. In `legacy` mode:

| | M2, $K$ = 300 / 3,000 | M3, $K$ = 300 / 3,000 |
|---|---|---|
| scenarios compared | 12 / 12 | 10 / 10 |
| iterations compared | 720 / 720 | 523 / 363 |
| parameters after every iteration, max. rel. diff. | 5.3e-12 / 1.6e-12 | 2.8e-08 / 5.4e-10 |
| `c0` after every iteration | identical (41 / 29 changes) | identical (19 / 15 changes) |
| stopping iteration and final `c0` | identical in all | identical in all 9 |
| parameters at the stopping point | 1.4e-13 / 5.2e-13 | 2.6e-10 / 1.5e-11 |

Three three-component scenarios per data set are excluded or restricted, for stated
reasons: two start outside the proportion bounds, where the original reads a vector it
has not assigned and returns NaN; and one starts at $\sigma^2 = 0.5$, where the
original's unsafeguarded Newton step is evaluated at negative variances and the outcome
is rounding noise (compared for the first three iterations only).

### Complete `genesis()` output

Everything the original `genesis()` function computes — starting values from LD-score
regression, EM, scores and information, the sandwich covariance from its tagging
strings, AIC and BIC, heritability and the other derived quantities — against
`gg.fit(..., legacy=True)`. Maximum relative difference over scenarios, $K$ = 300 /
3,000:

| | M2 (12 scenarios each) | M3 (9 scenarios each) |
|---|---|---|
| estimates | 1.4e-13 / 5.2e-13 | 4.9e-09 / 1.5e-11 |
| log-likelihood | 9.7e-16 / 2.4e-15 | 1.5e-13 / 2.4e-15 |
| information matrix | 3.3e-13 / 3.9e-13 | 7.0e-09 / 4.4e-09 |
| $J$ | 3.2e-10 / 4.5e-13 | 5.5e-09 / 4.9e-09 |
| covariance matrix | 2.7e-13 / 6.8e-13 | 9.4e-08 / 5.9e-10 |
| standard errors | 1.3e-13 / 3.4e-13 | 1.8e-08 / 9.4e-12 |
| $d_s$ | 3.3e-14 / 8.8e-14 | 2.6e-09 / 1.6e-12 |
| BIC | 2.0e-15 / 2.6e-15 | 9.9e-11 / 4.9e-15 |
| derived quantities and their standard errors | 1.3e-13 / 3.4e-13 | 1.8e-08 / 9.4e-12 |

The larger values for the three-component model come from scenarios whose information
matrix is nearly singular, where rounding differences are amplified by the inversion.

### Genome-wide: the GIANT height example

The worked example of the original package (941,684 SNPs; 1 Mb window, $r^2 \ge 0.1$;
`filter = FALSE`), original with 12 OpenMP threads against `legacy=True`, both started
from the same values:

| | Two-component model | Three-component model |
|---|---|---|
| EM iterations (original = this package) | 117 = 117 | 849 = 849 |
| final `c0` | 10 = 10 | 10 = 10 |
| estimates, max. rel. diff. | 3.1e-12 | 3.6e-11 |
| log-likelihood | 3.1e-15 | 3.1e-15 |
| information matrix | 6.2e-12 | 8.7e-11 |
| $J$ | 6.6e-12 | 1.0e-10 |
| covariance matrix | 1.2e-11 | 2.5e-09 |
| standard errors | 6.2e-12 | 5.8e-11 |
| $d_s$ / BIC | 1.4e-12 / 2.9e-15 | 1.7e-12 / 2.7e-15 |
| derived quantities / their standard errors | 3.1e-12 / 6.2e-12 | 4.7e-11 / 3.7e-11 |

The two-component run starts from the LD-score-regression values that each program
computes itself (they agree to $10^{-9}$); the three-component run starts from the
vector the original used, stored in the fixture. The trajectory is not sensitive to the
last digits of the start: begun instead from values that agree with the original's only
to seven digits, the three-component run also ends after 849 iterations, with estimates
within 4e-07.

The original's own `examples/examples.md` prints these fits with seven significant
digits. The fixture values agree with the printed estimates to 1e-07 for the
three-component model (whose printed covariance matrix shows the same asymmetry) and to
8e-06 for the two-component model. The cause of the latter, small, difference was not
investigated; that document was presumably produced on an x86 platform, where the
original computes in 80-bit `long double` (see below).

(east-asian-reference)=
### GENESISasian and the East Asian reference

GENESISasian contains no East Asian GWAS (its example is the European height file), so
a GWAS was simulated for the comparison: 858,450 markers of `eas_1kg_hm3` with sample
sizes between 135,000 and 150,000, effects from a three-component model spread through
the signed LD of the panel (`tools/upstream_oracle/make_eas_sim.py`). The genuine
GENESISasian code was run on it with all its defaults ($M = 953{,}834$) and the results
stored as fixtures; `fit(..., legacy=True)` with `eas_1kg_hm3` is compared with them in
`tests/test_full_scale.py`.

| | Two-component model | Three-component model |
|---|---|---|
| EM iterations (original = this package) | 163 = 163 | 521 = 521 |
| Estimates, largest relative difference | 6e-14 | 3.6e-11 |
| Standard errors, largest relative difference | 5e-13 | 6.3e-11 |
| Composite log-likelihood, relative difference | 5e-15 | |

(Iterations are counted as in the European table above; the original's own counter
starts at zero and reads 162 and 520.) A projection of `projection()` and a count of
`numInterval()` computed by GENESISasian from these fits are reproduced to within
$10^{-8}$ by `project` and `n_in_interval` given the fitted model, which carries
$M = 953{,}834$.

## Parity of the downstream functions

Against the original R functions (fixtures `rfuncs.json`, `mini_chr22_*.json`,
`height_full_*.json`):

| Function (original) | Agreement |
|---|---|
| `effect_density` (`dmixssnp`) | 3e-15 |
| `n_in_interval` (`numInterval`) | within the original's own cancellation error |
| `project` without intervals (`projection`) | ≤ 2e-11 (closed form against adaptive quadrature) |
| `polyrisk_predict(legacy=True)` (`polyriskpredict`) | AUC ≤ 1e-9; optimal threshold and proportions above $k$-fold risk ≤ 5e-8 |
| `posterior_marginal(include_inflation=False)` (`postmarginal`) | ≤ 2e-13 on synthetic and chromosome-22 fits; ≤ 1e-10 on all 941,684 height SNPs |
| `h2_transfer` (`h2transfer`) | 7e-16 |

Functions that draw random numbers cannot reproduce R's stream and are compared
statistically: the mean QQ curve and the mean genomic-control factor on chromosome 22
(1.548 against 1.553), projection intervals (limits within 6% on the height examples),
and the predicted genomic-control factor (`future_gc`: 1.836 / 1.855 / 1.498 against
1.821 / 1.857 / 1.491 for three settings on the full reference).

## LD-reference builder

`genesis ld build` was validated by recomputing the original's reference tables from
the genotypes they were derived from (LDSC 1000 Genomes Phase 3, 489 European samples,
1,070,777 HapMap3 SNPs; `tools/ld_validation`). With `round_r2=6`, which mimics the
six-significant-digit PLINK table that the original tables were built from:

| Configuration | $N^*$ equal | Tagging lists identical | Max. abs. difference in LD score |
|---|---|---|---|
| 500 kb, $r^2 \ge$ 0.05 / 0.1 / 0.2 | all 1,070,777 SNPs | all | 9.1e-13 / 9.1e-13 / 8.0e-13 |
| 1 Mb, $r^2 \ge$ 0.05 / 0.1 / 0.2 | all | all | 1.0e-06 / 1.0e-06 / 1.2e-12 |
| 2 Mb, $r^2 \ge$ 0.05 / 0.1 / 0.2 | all | all | 1.0e-06 / 1.0e-06 / 1.2e-12 |

The single residual ($10^{-6}$ in the LD score of two SNPs) is one SNP pair whose
$r^2$ is an exact decimal tie at the sixth digit. Of the 49,789,689 pairwise $r^2$
values of the 1 Mb / 0.1 configuration, 49,776,837 are bit-identical to the original's
and the others differ in the last binary digit, apart from that pair.

Without rounding (`round_r2=None`, the default for new panels) the only differences are
pairs whose exact $r^2$ lies just below the threshold but prints as the threshold: 12
ordered pairs for 1 Mb / 0.1 (10 SNPs with a different $N^*$); all other LD scores
agree to 6e-5.

This establishes the definitions that the original used — tagging: same chromosome,
within the window (inclusive), $r^2 \ge$ threshold, the SNP itself included; LD score:
the sum of $r^2 - (1-r^2)/(n-2)$ over the tagged SNPs; reference SNPs: the
HapMap3-without-MHC list restricted to MAF ≥ 0.05 in the panel — and that the builder
implements them. The validation covers one panel without missing genotypes; the
handling of missing genotypes (mean imputation), of duplicates and of the MAF filter is
covered by unit tests on synthetic data only.

The same was done for the East Asian table of GENESISasian, from the 1000 Genomes
Phase 3 VCF files converted with `genesis ld from-vcf` and the 504 samples labelled
`EAS`:

| Configuration | $N^*$ equal | Tagging lists identical | Ordered pairs | Max. abs. difference in LD score |
|---|---|---|---|---|
| 1 Mb, $r^2 \ge$ 0.1 | all 953,834 SNPs | all | 42,020,856 in both | 4.0e-06 |

(41 SNPs have an LD-score difference above $10^{-9}$: decimal ties at the sixth digit,
as above.) The SNP set is reproduced as well: "HapMap3 SNPs with MAF ≥ 0.05 in the 504
samples" selects 952,124 SNPs of the HapMap3-without-MHC list, among them all 952,104
such SNPs of the original reference; the 20 additional ones are not in the original,
and the original's other 1,730 SNPs lie in the MHC, which that list omits. This is a
second, independent data set — different genotype files, different samples, a
different conversion path — on which the builder reproduces tables made by the
method's authors.

### The distributed references

| Reference | Origin | How it was checked |
|---|---|---|
| `eur_1kg_hm3` | lossless conversion of the tables of GENESIS | LD scores bit-identical, tagging lists identical as sets, 55 spot-checked SNPs agree field by field in all nine configurations (`tools/refdata/verify_bundle.py`); rebuilt from genotypes as above |
| `eas_1kg_hm3`, 1 Mb / 0.1, pairwise $r^2$, null replicates | lossless conversion of the tables of GENESISasian | the same checks; rebuilt from genotypes as above; parity of fits with GENESISasian |
| `eas_1kg_hm3`, the other eight configurations | computed with this package (`round_r2=6`) from the 504 `EAS` samples | written only after the recomputed 1 Mb / 0.1 table reproduced the original (`tools/refdata/add_ld_configs.py`); internal consistency checks |
| `afr_1kg_hm3` | computed with this package from the 661 `AFR` samples (exact $r^2$) | internal consistency checks; the download and a fit from the downloaded copy tested from local files; simulations below. **No original tables exist to compare with.** |

The internal consistency checks of `verify_bundle.py` (unique SNPs, every SNP in its
own neighbourhood, symmetric neighbourhoods within the window and the chromosome,
checksums, and others) pass for all three: 126, 125 and 125 checks.

## Backends

The same kernel code runs on NumPy and on PyTorch. Tests assert that

- results are **bit-identical** across thread counts for a fixed chunk size, and agree
  to $10^{-10}$ across chunk sizes (the order of summation differs);
- PyTorch on the CPU agrees with NumPy to $10^{-10}$, also when the SNPs are split into
  several shards — the code path used for several GPUs;
- on an Apple MPS device in single precision (a smoke test of the device handling only;
  MPS is not a supported device) the log-likelihood agrees to $2 \times 10^{-4}$.

`genesis doctor` runs the comparison for every backend and device of the machine at
hand.

(run-times)=
## Run times

:::{admonition} Conservative timings
:class: note

All timings were taken on one laptop while other jobs were running on it. Treat them
as upper bounds; no timings on cluster nodes or GPUs exist yet.
:::

Apple M5 Pro laptop (18 cores, of which 6 are of the fastest kind; 48 GB), GIANT height GWAS,
941,684 SNPs. The original was built with OpenMP and run with 12 threads.

| Task | Original | This package |
|---|---:|---:|
| M2, EM to the original's stopping rule (117 iterations) | 8.7 s | 7.7 s (`legacy`) |
| M2, whole run | 90 s (R session: loading the reference tables, merging, EM, covariance) | 9.6 s for the fit and its covariance (`legacy`), plus a few seconds to read and merge the data |
| M3, EM to the original's stopping rule (849 iterations) | 1,253 s | 403 s (`legacy`) |
| M3, whole run | 1,349 s | 405 s for the fit and its covariance (`legacy`) |
| **Complete analysis in default mode**: both models, model selection, projections with intervals, posterior summaries for all SNPs, QQ plot with 100 simulations (`genesis fit --qq --posterior --project ...`) | 1,439 s for the two fits alone | **63 s** (12 threads) |
| The quick-start command (`genesis fit --qq --project ...`, all 18 cores) | — | 84 s with 18 threads; peak memory 3.0 GB, of which M2 6 s (50 EM iterations, 6 Newton steps, `c0` = 10) and M3 55 s (final `c0` = 14) |
| QQ simulation, 100 replicates, chromosome 22 (13,230 SNPs) | 70 s | 0.6 s |
| LD reference for 1,070,777 SNPs and 489 samples from PLINK files | not available | 45 s |

Why the default mode is faster than `legacy`: the EM algorithm converges linearly and
needs hundreds of passes over the data; the default uses it for 50 warm-up iterations
and then Newton's method, which converges in a handful of passes (see
[Differences from the original](differences.md)).

Caveats:

- One machine, one data set. On x86 clusters the comparison will differ: there the
  original computes in 80-bit `long double`, which is slower than the 64-bit arithmetic
  it uses on arm64.
- In the original, most of the wall time of the two-component analysis is spent outside
  the EM (loading the reference tables and parsing the tagging strings in R).
- Scaling with the number of threads beyond the six fastest cores of this laptop, and
  on many-core nodes, has not been measured. `genesis doctor --benchmark` times one
  pass over a genome-wide data set on every backend and device of the machine at hand.
- There are no GPU timings (next section).

## What has not been verified

- **No CUDA GPU was available during development.** The GPU path has been exercised
  only indirectly: the kernels are device-independent PyTorch code that is tested on
  the CPU, including execution in several shards, and a single-precision smoke test on
  Apple MPS confirms that data are moved to and from a non-CPU device correctly. Before
  relying on GPU results, run `genesis doctor` on the GPU node: it evaluates the
  likelihood, the scores and the information matrix on every visible device and
  reports the maximum relative difference from the single-threaded CPU result (for
  PyTorch on the CPU it is $2\times10^{-14}$; a GPU should give a value of similar
  magnitude).
- **Multi-GPU execution** on real hardware (the shard logic is tested on the CPU).
- **x86 builds of the original.** The fixtures were generated on arm64, where the
  original's `long double` is a 64-bit double. On x86 it is an 80-bit type, so the
  original's results there can differ in the trailing digits; this was not examined.
- **Biowulf.** The job scripts and instructions of the Biowulf guide follow the NIH HPC
  documentation but were not run on the cluster.
- **Other Python and NumPy versions on the development machine.** Tests were run
  locally with Python 3.12, NumPy 2.5 and pandas 3; the other Python versions are
  covered by continuous integration only. On Apple silicon with NumPy 2.2, NumPy's
  Accelerate-backed matrix product emits spurious floating-point warnings, which the
  test configuration treats as errors; results are not affected.
- **Reference panels with missing genotypes** in the LD builder (unit tests on
  synthetic data only).
- **Real East Asian or African-ancestry GWAS.** No such data were analysed. For East
  Asian data the numerical agreement with GENESISasian is established on a simulated
  GWAS; for African ancestry there is no original software to agree with, and the
  evidence is the builder's reproduction of the European and East Asian tables plus
  the simulations below.
- **Reference panels in GRCh38.** Selecting SNPs by position (`genesis ld from-vcf
  --positions`) gives, on chromosome 22 of the Phase 3 files with GRCh37 positions, the
  same 17,757 SNPs and identical genotypes as selecting by identifier, and GRCh38
  references are covered by unit tests; no reference was built from a GRCh38 panel.
- **Downloads of the East Asian and African-ancestry references from GitHub.** The
  download path was exercised with the release files served from a local directory.
- **Calibration of the standard errors beyond one setting.** The simulation study below
  covers a single two-component scenario; the three-component model was not studied.

(simulation-study)=
## Simulation study

How accurate are the estimates and how well calibrated are the standard errors when
the data have realistic LD? One hundred genome-wide data sets were simulated with the
scheme of the original paper's simulation studies, using the signed LD of the reference
panel:

- joint effects for all 1,070,777 reference SNPs from the two-component model with
  $\pi_c = 0.005$ and $\sigma^2 = 6\times10^{-5}$ (5,354 susceptibility SNPs,
  heritability 0.321);
- marginal effects $\beta_k = \sum_{m \in \mathcal N_k} r_{km}\,\beta^{(J)}_m$, summed
  over the tagging neighbourhood of each SNP (1 Mb, $r^2 \ge 0.1$), with signed
  correlations $r_{km}$ computed from the 1000 Genomes European genotypes (49.8 million
  pairs);
- over-dispersion $N(0, a)$ with $a = 10^{-6}$, drawn independently for every SNP;
- estimation error $e_k/\sqrt n$ with $n = 130{,}000$, where $e$ is one of the 100
  LD-correlated null replicates of the original package, a different one for each data
  set.

Each data set was fitted with the two-component model and default settings. All fits
converged, at `c0 = 10`; the mean genomic-control factor of the simulated statistics
was 1.38.

| | Truth | Mean estimate | Bias | Empirical SD | Mean SE (default) | SE / SD | Mean SE (original's convention) | SE / SD | SE / SD without sandwich |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| $\pi_c$ | 5.00e-03 | 3.63e-03 | −27.5% | 2.53e-04 | 2.56e-04 | 1.01 | 2.61e-04 | 1.03 | 0.17 |
| $\sigma^2$ | 6.00e-05 | 8.07e-05 | +34.5% | 4.78e-06 | 4.64e-06 | 0.97 | 4.73e-06 | 0.99 | 0.18 |
| $a$ | 1.00e-06 | 1.17e-06 | +16.9% | 1.10e-07 | 7.93e-08 | 0.72 | 8.24e-08 | 0.75 | 0.20 |
| Susceptibility SNPs | 5,354 | 3,883 | −27.5% | 271 | 274 | 1.01 | | | |
| Heritability | 0.3212 | 0.3124 | −2.7% | 0.0138 | 0.0128 | 0.93 | | | |

"Default" is the sandwich estimator of the paper's equation (4); "original's
convention" counts each marker's own score twice (see
[Differences from the original](differences.md)); "without sandwich" uses the inverse
information matrix alone. With 100 replicates the Monte-Carlo standard error of each
ratio is about 0.07.

What the study shows:

1. **The sandwich standard errors are calibrated for $\pi_c$ and $\sigma^2$.** The
   mean estimated standard error is within 3% of the empirical standard deviation of
   the estimates, and intervals of ±1.96 SE around the mean estimate contain 95% and
   96% of the estimates. Standard errors from the inverse information alone, which
   ignore LD, are five to six times too small.
2. **The two conventions for the sandwich estimator cannot be told apart.** They differ
   by about 2%, far less than 100 replicates can resolve. The default is the formula
   of the paper; every result file also contains the original's values
   (`diagnostics.sandwich_alt`). (The paper describes its sandwich estimator as
   "slightly conservative" in its own simulations.)
3. **The standard error of $a$ is too small by about a quarter in this simulation**
   (and that of the heritability by 7%, which is within Monte-Carlo error). A possible
   explanation — a hypothesis that has not been tested — lies in the simulation rather
   than in the estimator: the null replicates used as estimation error were computed
   from a panel of 489 individuals and therefore share genome-wide sample correlations
   (their genome-wide mean $\chi^2$ varies across replicates with a standard deviation
   of 0.011) that the statistics of a large GWAS do not have, and that an estimator
   built from local tagging neighbourhoods cannot see.
4. **The estimates are biased under realistic LD.** $\pi_c$ is too low by about a
   quarter and $\sigma^2$ too high by about a third, while the heritability is nearly
   unbiased. This is the pattern that the original paper reports for its own
   simulations — "some downward bias in estimates of proportions of susceptibility
   SNPs ... and upward bias in corresponding variance-component parameters" — and it
   agrees with the paper's description of the estimated number of susceptibility SNPs
   as a lower bound ([Zhang et al. 2018](#ref-zhang2018)).
5. **Calibrated standard errors do not make the intervals cover the truth.** Because of
   that bias, the nominal 95% interval contained the true $\pi_c$ or $\sigma^2$ in
   none of the 100 data sets, and the true heritability in 87. Standard errors
   describe the sampling variability of the estimator, not its distance from the
   truth.

Limitations: one parameter setting, one sample size and a two-component truth only;
estimation errors from a small reference panel (point 3); calibration was assessed
around the mean estimate. The three-component model, for which the paper reports
distinctly larger standard errors, was not studied.

(simulation-ancestries)=
### The same study with East Asian and African-ancestry LD

The study was repeated with the other two references — same parameter values
($\pi_c = 0.005$, $\sigma^2 = 6\times10^{-5}$, $a = 10^{-6}$, $n = 130{,}000$), same
scheme, 50 data sets each. Because the references differ in $M$, the true number of
susceptibility SNPs and the true heritability differ: 4,769 and 0.2862 for
`eas_1kg_hm3`, 5,127 and 0.3076 for `afr_1kg_hm3`. The estimation errors are the null
replicates of each reference; those of GENESISasian are coded towards the minor allele
and were given the sign convention of the genotypes first.

| LD of | Data sets | $\pi_c$: bias (SE / SD) | $\sigma^2$: bias (SE / SD) | $a$: bias (SE / SD) | Heritability: truth, mean estimate, bias | Empirical SD | Mean SE | SE / SD |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| `eur_1kg_hm3` | 100 | −27.5% (1.01) | +34.5% (0.97) | +16.9% (0.72) | 0.3212, 0.3124, −2.7% | 0.0138 | 0.0128 | 0.93 |
| `eas_1kg_hm3` | 50 | −27.2% (1.01) | +32.1% (1.09) | +18.9% (0.58) | 0.2862, 0.2742, −4.2% | 0.0108 | 0.0118 | 1.09 |
| `afr_1kg_hm3` | 50 | −31.7% (0.89) | +41.3% (0.94) | +12.0% (0.41) | 0.3076, 0.2955, −3.9% | 0.0146 | 0.0143 | 0.98 |

With 50 data sets the Monte-Carlo standard error of a ratio SE / SD is about 0.10.

The pattern is the same in all three populations: $\pi_c$ too low by between a quarter
and a third, $\sigma^2$ too high correspondingly, heritability within a few per cent
of the truth, and standard errors of $\pi_c$, $\sigma^2$ and the heritability
consistent with the spread of the estimates. The standard error of $a$ is too small
everywhere, and more so with the African-ancestry panel (0.41). Nothing in these
results is specific to an ancestry, which is what one expects of a method whose only
population-specific input is the reference — but note what the simulations cannot
show: the data are generated from the same panel that serves as the reference, so they
say nothing about how well a 1000 Genomes panel represents the LD of a real study
population. That question matters most for admixed samples, the African-ancestry
reference included.
