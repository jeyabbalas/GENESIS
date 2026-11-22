# GENESIS

[![CI](https://github.com/jeyabbalas/GENESIS/actions/workflows/ci.yml/badge.svg)](https://github.com/jeyabbalas/GENESIS/actions/workflows/ci.yml)
[![Docs](https://github.com/jeyabbalas/GENESIS/actions/workflows/docs.yml/badge.svg)](https://jeyabbalas.github.io/GENESIS/)
[![License: GPL-2.0-or-later](https://img.shields.io/badge/license-GPL--2.0--or--later-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](pyproject.toml)

**GEN**etic **E**ffect-**S**ize distribution **I**nference from **S**ummary-level data,
re-implemented in Python for CPUs and GPUs.

GENESIS estimates, from GWAS summary statistics and an LD reference panel, the number
of susceptibility SNPs of a trait, the distribution of their effect sizes and the
heritability they explain, and projects the yield of future, larger studies (number of
discoveries, heritability explained, accuracy of polygenic risk scores). The method is
described in

> Zhang Y, Qi G, Park J-H, Chatterjee N (2018). Estimation of complex effect-size
> distributions using summary-level statistics from genome-wide association studies
> across 32 complex traits. *Nature Genetics* 50:1318–1326.
> <https://doi.org/10.1038/s41588-018-0193-x>

This repository is a ground-up re-implementation of the authors'
[R package](https://github.com/yandorazhang/GENESIS) as the Python package
`genesis-gwas`.

**Documentation: <https://jeyabbalas.github.io/GENESIS/>**

## Features

- Two- and three-component mixture models fitted by composite likelihood, with
  sandwich standard errors that account for LD and model selection by BIC.
- The same statistical model as the original; a `legacy` mode reproduces the original
  code's results to rounding error and is tested against it.
- Multi-threaded CPU execution (NumPy) and single- or multi-GPU execution (PyTorch),
  in double precision.
- Flexible input: automatic detection of columns and effect scales, a step-by-step QC
  report, matching by rsID or by position (GRCh37/GRCh38).
- Downstream analyses: projections of future discoveries, expected AUC of polygenic
  scores, per-SNP posterior probabilities and shrunken effects, model-based QQ plots.
- LD references for any population, built from PLINK files with `genesis ld build`.
- A command-line interface and a Python API; results as JSON and TSV.

## Installation

Python 3.10 or newer; no compiler is needed.

```bash
pip install "genesis-gwas @ git+https://github.com/jeyabbalas/GENESIS.git"
genesis doctor     # versions, CPUs/GPUs, backend self-test
genesis demo       # complete analysis of a packaged chromosome-22 example
```

For GPUs, install a CUDA build of [PyTorch](https://pytorch.org) as well (or use the
`[gpu]` extra). See the
[installation guide](https://jeyabbalas.github.io/GENESIS/getting-started/installation.html),
and the [Biowulf guide](BIOWULF.md) for the NIH HPC cluster.

## Quick start

Command line — the European LD reference (32 MB) is downloaded on first use:

```bash
genesis fit trait.sumstats.tsv.gz --out results/trait --qq --project 500000,1000000
cat results/trait.summary.txt
```

The input is a text file with a SNP identifier, a Z statistic (or effect and standard
error, or odds ratio and standard error) and a sample size per SNP; common column names
are recognised. For case–control studies the sample size is the effective sample size
`n_cases * n_controls / (n_cases + n_controls)` (`--n-cases`, `--n-controls`).

Python:

```python
import genesis_gwas as gg

ref = gg.load_reference()                           # European reference (cached download)
data = gg.prepare("trait.sumstats.tsv.gz", ref)     # QC + merge with the LD reference
m2 = gg.fit(data, 2)                                # two-component model
m3 = gg.fit(data, 3, starting=m2)                   # three-component model
best = gg.select_model([m2, m3]).best               # by BIC
print(best.summary())                               # estimates, SEs, heritability, diagnostics
print(gg.project(best, n=[500_000, 1_000_000]).to_frame())
```

## How it differs from the original R package

| | Original R package | This package |
|---|---|---|
| Statistical model, likelihood, estimating equations | — | the same |
| Reported estimate | point at which the EM iteration stops | maximiser of the composite likelihood (EM warm-up, then Newton's method); `--legacy` for the original |
| Truncation level of the likelihood | raised during the iteration, depending on its path | chosen from the data: raised until the discarded probability mass is negligible |
| Sandwich standard errors | a marker's own score counted twice | counted once, as in the paper's formula; both conventions reported |
| Run time, GIANT height GWAS (both models) | about 24 minutes | about one minute, including projections, posteriors and a QQ plot (12 threads, same laptop) |
| Non-identified three-component fits | reported without warning | flagged, not selected |
| Quality control | optional, off by default | on by default, with a report |
| Installation | R package with 1.6 GB of data, compiled C++ | pure Python; 32 MB of reference data downloaded on demand |
| Hardware | CPU threads (OpenMP) | CPU threads or CUDA GPUs |
| LD reference panels | European | European, plus a builder for any population |

Every difference is documented with its reason and measured effect in
[Differences from the original](https://jeyabbalas.github.io/GENESIS/reference/differences.html);
how the implementation was verified, and what has not been verified, is in
[Validation](https://jeyabbalas.github.io/GENESIS/reference/validation.html).

## Documentation

- [Quick start](https://jeyabbalas.github.io/GENESIS/getting-started/quickstart.html)
- [Tutorial](https://jeyabbalas.github.io/GENESIS/tutorial/index.html) — the model, the
  data, fitting, interpretation, projections
- [Theory](https://jeyabbalas.github.io/GENESIS/theory/index.html) — likelihood, EM
  algorithm, sandwich variance
- [Biowulf guide](https://jeyabbalas.github.io/GENESIS/biowulf/index.html) (also
  [BIOWULF.md](BIOWULF.md))
- [Command-line reference](https://jeyabbalas.github.io/GENESIS/reference/cli.html) and
  [Python API](https://jeyabbalas.github.io/GENESIS/api/index.html)
- [Output files](https://jeyabbalas.github.io/GENESIS/reference/outputs.html) and
  [FAQ](https://jeyabbalas.github.io/GENESIS/reference/faq.html)

## Citation

Please cite Zhang et al. (2018), above, and state the version of `genesis-gwas` used
(`genesis --version`; it is also recorded in every result file). See
[CITATION.cff](CITATION.cff).

## License and acknowledgement

GPL-2.0-or-later (see [LICENSE](LICENSE)). This package is a derivative work of the
R package [GENESIS](https://github.com/yandorazhang/GENESIS) by Yan Zhang, Guanghao Qi,
Ju-Hyun Park and Nilanjan Chatterjee, distributed under GPL (≥ 2). The method, the
European LD reference (derived by its authors from the 1000 Genomes Project) and the
example data are taken from that package.

## Reporting problems

Please open an [issue](https://github.com/jeyabbalas/GENESIS/issues) with the output of
`genesis doctor`, the command, and the log file (`PREFIX.log`). Contributions are
welcome; see [CONTRIBUTING.md](CONTRIBUTING.md) and the [changelog](CHANGELOG.md).
