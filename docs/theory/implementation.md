# Implementation

This chapter explains how the mathematics of the previous chapters is organised in code:
which module does what, why the inner loops look the way they do, what they cost, and how
correctness is established. It is meant for people who want to read, extend or audit the
source.

(impl-architecture)=
## Architecture

```{mermaid}
flowchart LR
    subgraph datalayer ["data/ · ld/"]
        SS["sumstats.py<br/>read, derive Z and N"]
        REF["reference.py · download.py<br/>LD bundles, verified downloads"]
        PREP["prepare.py<br/>QC and merge"]
        LDB["ld/build.py · ld/plink.py<br/>build a reference from genotypes"]
    end
    subgraph core ["core/"]
        MD["data.py<br/>ModelData"]
        SPEC["spec.py<br/>parameter layout, configurations"]
        KER["kernels.py<br/>per-chunk maths"]
        ENG["engine.py<br/>chunked map-reduce"]
        LIK["likelihood.py<br/>log-likelihoods, scores, information"]
        EM["em.py · start.py<br/>EM, Newton's method, starting values"]
        VAR["variance.py · derived.py<br/>sandwich, BIC, delta method"]
    end
    subgraph backends ["backends/ · compute.py"]
        NP["NumPy<br/>CPU threads"]
        TO["PyTorch<br/>CUDA GPUs"]
    end
    FIT["fit.py<br/>orchestration"]
    RES["result.py<br/>FitResult, model choice"]
    subgraph user ["analysis/ · plotting/ · cli/ · hpc/"]
        AN["projections, PRS,<br/>posterior, simulation"]
        CLI["genesis command"]
    end
    SS --> PREP
    REF --> PREP
    LDB --> REF
    PREP --> MD --> ENG
    SPEC --> LIK
    KER --> ENG --> LIK --> EM --> FIT
    LIK --> VAR --> FIT
    NP --> ENG
    TO --> ENG
    FIT --> RES --> AN
    RES --> CLI
    REF --> VAR
```

The layers have narrow interfaces:

- **`ModelData`** ({class}`genesis_gwas.ModelData`) is the complete input of the statistical
  model: five aligned arrays ($\hat\beta_k$, $s_k^2$, $\ell_k$, $N^*_k$, $n_k$), the panel
  size $M$, and — for the sandwich estimator — each marker's row in the reference and the
  reference's neighbourhoods. Everything before it is file handling.
- **`CompositeLikelihood`** ({class}`genesis_gwas.core.likelihood.CompositeLikelihood`)
  turns a parameter vector into numbers: log-likelihoods, M-step statistics, scores,
  information, posteriors. It knows the model but not the hardware.
- **`Engine`** ({class}`genesis_gwas.core.engine.Engine`) evaluates a kernel over all
  markers on the configured devices. It knows the hardware but not the model.
- **`fit`** ({func}`genesis_gwas.fit`) sequences starting values, the EM warm-up,
  Newton's method, the truncation check, the variance step and the derived quantities,
  and returns a
  {class}`genesis_gwas.FitResult` that serialises to JSON.

(impl-kernels)=
## The kernels

### One formula for all models

Every heavy computation has the same shape: for each marker $k$ and each configuration
$t$, evaluate something, then sum over markers. The mixture variance is
$v_{kt} = r_kc_t + s_k^2 + a$, where $c_t = \sum_hJ_{th}\sigma_h^2$ is a vector of length
$T$ computed on the host. The kernels in `core/kernels.py` receive $c$, the scalar $a$ and
a table of log priors — and nothing else about the model. They do not know whether there
are one or two normal components.

What makes this possible is that every model-specific quantity factorises into a function
of the configuration times a sum over markers. For example, the gradient of the M-step
objective with respect to $\sigma_h^2$ ([Estimation](estimation.md)) is

$$
\frac{\partial Q_v}{\partial\sigma_h^2}
 = \sum_{t} J_{th} \underbrace{\sum_{k} r_k\, w_{kt}\, g_1(v_{kt})}_{\texttt{a1}[t]} .
$$

The kernel returns the inner sum as a vector over configurations (`a1`), and the outer sum
— a dot product of length $T$ — is done on the host. The complete list for an EM pass is
five vectors plus the expected configuration counts:

| Kernel output | Definition | Used for |
|---|---|---|
| `lse` | $\sum_k \log\sum_t \exp\lambda_{kt}$ | $\tilde\ell$ and $\ell$ |
| `n_t` | $\sum_k w_{kt}$ | $A_h = \sum_t J_{th}\,\texttt{n\_t}[t]$, the proportion updates |
| `a0`, `a1` | $\sum_k \{1, r_k\}\,w_{kt}\,g_1$ | gradient of $Q_v$ |
| `d0`, `d1`, `d2` | $\sum_k \{1, r_k, r_k^2\}\,w_{kt}\,g_2$ | Hessian of $Q_v$ |

The score/information pass returns thirteen such vectors and two small per-marker
matrices. Because results are vectors of length $T$ (11 or 121), the amount of data that
leaves a compute device per pass is a few kilobytes regardless of the number of markers.

The practical consequences:

- **One implementation, tested once.** M2 and M3 share every line of numerical code; the
  original has separate, near-duplicate routines for the two models (and the two had
  diverged: see the note on `Ik3` in [Standard errors](inference.md)).
- **Backend independence.** A kernel is twenty lines of array arithmetic written against a
  namespace `xp` that is either `numpy` or `torch`. Only operations with identical names
  and semantics in both are used.

### Log space

Posterior weights are computed with the log-sum-exp identity, never from raw densities.
The original multiplies a binomial probability by a normal density in linear space; for a
marker tagging several hundred SNPs, $(1-\pi_c)^{N^*}$ times a density underflows unless
`long double` has extended precision, which it has on x86 Linux but not on ARM or with
Microsoft compilers. In log space there is nothing to underflow: impossible
configurations carry $-\infty$ and receive weight exactly zero.

### Fused passes

The first Newton step of each M-step is evaluated at the parameters that defined the
weights, so the E-step and that Newton step share $v_{kt}$, $1/v_{kt}$ and
$\hat\beta_k^2/v_{kt}$ and are computed together. One EM iteration with the default two
Newton steps therefore costs two passes over the data (the original makes four: weights,
proportions, and each Newton step). The second pass recomputes the weights instead of
storing them — storing would need $8KT$ bytes, about 0.9 GB for M3 on a million markers —
which keeps memory proportional to the chunk size.

(impl-engine)=
## The engine

```{mermaid}
flowchart LR
    T["tables for this pass<br/>(log priors, c, a) — a few kB"] --> D1 & D2
    subgraph D1 ["device 1 (or CPU)"]
        direction TB
        C11["chunk"] --> K11["kernel"] --> P11["partial sums"]
        C12["chunk"] --> K12["kernel"] --> P11
    end
    subgraph D2 ["device 2"]
        direction TB
        C21["chunk"] --> K21["kernel"] --> P21["partial sums"]
    end
    P11 --> R["reduce on host<br/>in marker order"]
    P21 --> R
    R --> H["host: model-specific<br/>assembly (T-vectors → θ)"]
```

**Chunks.** Markers are processed in contiguous chunks so that the working arrays
(chunk size × $T$ doubles each) fit in cache or in device memory. On the CPU a chunk
holds about $2^{18}$ marker–configuration pairs (2 MB per array; 2,166 markers for M3,
23,831 for M2). On a GPU it is sized from the free device memory.

**CPU threads.** NumPy's element-wise loops run on one core but release Python's global
interpreter lock, so chunks are dispatched to a thread pool and `n` threads use `n`
cores. No BLAS routine is in the hot path — sums over markers use `einsum` loops — so
there is no nested threading to oversubscribe a Slurm allocation. The number of threads
is taken from the *allocation* (`GENESIS_NUM_THREADS`, then `SLURM_CPUS_PER_TASK`, then
the CPU affinity mask), never from the size of the machine
({func}`genesis_gwas.compute.available_cpus`).

**GPUs.** With the PyTorch backend the markers are split into one contiguous shard per
device. Each device processes its shard from its own host thread and accumulates partial
sums on the device; the host adds the per-device results. Several GPUs therefore work on
one fit concurrently. Because communication per pass is a few kilobytes, scaling is
limited only by the number of markers per device.

**Precision.** All inference is in double precision. Apple's MPS backend has no 64-bit
floats and is refused for that reason; CUDA devices compute in `float64`.

**Determinism.** Partial results are always combined in marker order, scalars with an
exactly rounded sum. For a given plan (backend, devices, chunk size) results are
bit-for-bit reproducible, and on the CPU they do not depend on the number of threads.
Different chunkings or devices change the order of floating-point additions and hence the
last few bits; the EM is self-correcting, so this does not accumulate.

### Cost

| Quantity | Value |
|---|---|
| work per pass | $K \times T$ terms, $T = (c_0+1)^H$ |
| passes per EM iteration | `newton_steps` (default 2) |
| passes per Newton step | 1 score/information pass + 1 or more likelihood passes |
| memory | data: 56 bytes per marker; working arrays: threads × chunk × $T$ × 8 bytes × about 12 |

Measured on one Apple M5 Pro laptop for the height data (941,684 markers), one fused
E/M pass takes 0.86 s on one thread for M3 at $c_0 = 10$ (7.5 ns per marker–configuration
pair) and 0.17 s on 12 threads; for M2, 0.12 s and 0.02 s. A complete fit of both models
peaked at 2.7 GB of resident memory with 12 threads. GPU timings have not been measured
(no CUDA device was available); `genesis doctor` reports what your own hardware does and
checks that every backend returns the same numbers.

With the default optimisation ({ref}`est-schedule`) a fit needs on the order of a hundred
passes in total: 50 EM iterations at two passes each, and a handful of Newton steps. If
the truncation check raises $c_0$ ({ref}`est-adaptive-c0`), a few more passes follow at
the larger $T$. The sandwich step adds one pass plus a sparse gather over the tagging
neighbourhoods (about 50 million pairs for the bundled reference; about a second).

### Measured run time

```{figure} ../_static/figures/runtime.svg
:name: fig-runtime
:alt: Horizontal bar chart on a logarithmic time axis comparing, for the two- and three-component models, the wall-clock time of the original software, of this package running the original algorithm, and of this package with default settings; and the time of a complete analysis with this package.

**Wall-clock time for the height GWAS (941,684 markers) on one Apple M5 Pro laptop**, with
12 threads for both programs. "Original algorithm" is `--legacy`, which performs the same
EM iterations as the original software, so the difference from the first bar is
implementation and the difference from the third is the optimiser. The last bar is one
`genesis fit` command that fits both models and writes the QQ plot (100 simulated data
sets), the posterior table and projections; it was timed while the machine was busy.
GPU timings were not measured.
```

Where the time went in the original: its EM for M3 took 1,253 of the 1,349 seconds; for M2
most of the 90 seconds were spent outside the EM, loading data and computing standard
errors. These are single measurements on one machine; `runtime_measurements.json` in
`docs/_scripts` records them.

(impl-numerics)=
## Numerical details

- **Tables by distinct $N^*$.** The log multinomial coefficients and the retained prior
  mass depend on a marker only through $N^*_k$, which takes a few hundred distinct values.
  They are computed per distinct value ({func}`genesis_gwas.core.spec.log_multinomial_table`)
  and looked up per marker; the renormalisation term $\sum_k\log y_k$ is then a dot product
  with the counts of each value and costs nothing per marker.
- **Scaled linear algebra.** Hessians and information matrices are rescaled to unit
  diagonal before being inverted or tested for definiteness
  ({func}`genesis_gwas.core.variance.scaled_inverse`,
  {func}`genesis_gwas.core.em.newton_direction`).
- **Feasibility is checked on the host** before a pass is started, so a kernel never sees
  a non-positive variance and never emits `NaN`.
- **Non-finite numbers do not reach files.** `FitResult.to_json` writes them as `null`.

(impl-legacy)=
## Legacy mode

`legacy=True` reproduces the original software number for number. It is not a second
implementation: it runs the same kernels and changes only the layers above them.

| Layer | Default | `legacy=True` |
|---|---|---|
| EM control flow | safeguarded | literal transcription of `EM_func` / `EM_func3` (resets, early exit) |
| Newton step for $\sigma^2, a$ | ascent direction via scaled eigen-decomposition | the original's explicit 2×2 / 3×3 cofactor formulas |
| Newton's method | on, after a short EM warm-up | off |
| truncation level | checked at the solution, raised if needed | the original's rule inside the EM |
| information matrix (M3) | symmetric | with the original's missing term |
| sandwich "meat" | paper's eq. (4) | marker's own score counted twice |
| sibling-risk standard error | delta method | the original's expression |

Each row is described in [Differences from the original](../reference/differences.md).

(impl-reference)=
## Reference data

The original package ships 1.6 GB of `.RData` files inside its git repository, which is
what makes it slow and fragile to install. Here a reference panel is a *bundle*: a
directory of NumPy archives with a manifest of SHA-256 checksums
({mod}`genesis_gwas.data.reference`).

| File | Content |
|---|---|
| `snps.npz` | identifier, chromosome, position (GRCh37 and, where liftable, GRCh38), HapMap3-without-MHC flag |
| `ld/w{window}kb_r{r2}.npz` | $N^*_k$, $\ell_k$ and the tagging neighbourhoods (compressed sparse rows) for one LD definition |
| `pairld/…npz` | pairwise $r^2$ aligned with the neighbourhoods (for simulation) |
| `nullz/…npy` | LD-correlated null $z$ scores (for QQ envelopes) |
| `manifest.json` | names, provenance, sizes, checksums |

The bundled European reference is a lossless conversion of the original's data (LD scores
are bit-identical), with SNPs sorted by position so that a marker's neighbours are nearby
rows. Files are fetched on first use, one at a time and only those an analysis needs —
32 MB for a standard fit — with resumable transfers, checksum verification before a file
becomes visible, and a lock that makes concurrent fetches from many cluster jobs safe
({mod}`genesis_gwas.data.download`).

`genesis ld build` writes the same format from PLINK genotypes
({func}`genesis_gwas.ld.build_reference`). Built from the 1000 Genomes European genotypes
it reproduces the bundled tables, which validates both the definitions and the builder;
see [the tutorial](../tutorial/other-ancestries.md) and
[Validation](../reference/validation.md).

(impl-tests)=
## How correctness is established

The test suite is organised around independent oracles, each catching a different kind of
error.

1. **A literal port of the original C++** (`tests/upstream_port.py`): scalar loops,
   linear-space densities, the same truncation logic — sharing no code with the package.
   The optimised kernels must match it for weights, log-likelihoods, M-step statistics,
   scores and information.
2. **Outputs of the genuine original software** (`tests/fixtures/upstream/`, generated by
   `tools/upstream_oracle/` from the original repository): function-level values, EM
   trajectories iteration by iteration, and complete `genesis()` results on synthetic
   data, on chromosome 22 and on the full height GWAS. `legacy=True` must reproduce them.
3. **Mathematics**: analytic scores and information against numerical differentiation;
   closed-form projection integrals against quadrature; monotonicity of the EM objective;
   Newton's method against a very long EM run.
4. **Statistics**: recovery of known parameters from simulated data.
5. **Consistency across hardware**: NumPy versus PyTorch, one thread versus many, one
   shard versus several, with a single-precision device run as a plumbing check.

[Validation](../reference/validation.md) reports the agreement achieved.
