# GENESIS

**GEN**etic **E**ffect-**S**ize distribution **I**nference from **S**ummary-level data —
a Python implementation for CPUs and GPUs.

GENESIS estimates, from the summary statistics of a genome-wide association study and
an LD reference panel alone, how the heritability of a trait is distributed over the
genome:

- the **number of susceptibility SNPs** — the proportion $\pi_c$ of common variants
  with a non-zero effect;
- the **distribution of their effect sizes** — one normal distribution, or a mixture of
  two that allows a cluster of larger effects;
- the **heritability** they explain, separated from confounding by population
  stratification or cryptic relatedness.

From the fitted distribution it **projects what larger studies will yield**: the number
of genome-wide significant discoveries, the heritability they will explain, and the
accuracy of polygenic risk scores, as functions of sample size.

The method is that of [Zhang, Qi, Park and Chatterjee (2018)](#ref-zhang2018): a
mixture model for the joint effects of SNPs, fitted to marginal summary statistics by
composite likelihood with an EM algorithm, with standard errors from a sandwich
estimator that accounts for linkage disequilibrium. This package is a re-implementation
of the authors' [R package](https://github.com/yandorazhang/GENESIS).

## What this implementation adds

- **The same model, verified against the original.** The `legacy` mode reproduces the
  estimates, standard errors and information criteria of the original R/C++ code to
  rounding error; the default mode departs from it only where documented, each
  departure with its reason and measured effect
  ([Differences from the original](reference/differences.md),
  [Validation](reference/validation.md)).
- **CPUs and GPUs.** One code path runs multi-threaded on CPUs (NumPy) and on one or
  several NVIDIA GPUs (PyTorch), in double precision on both.
- **Estimates that solve the likelihood equations.** The EM algorithm is completed by
  Newton's method, every result carries a convergence check, and three-component fits
  that the data cannot identify are flagged and excluded from model selection.
- **Practical data handling.** Column names and effect scales are detected, quality
  control is reported step by step, SNPs are matched by identifier or by position in
  GRCh37 or GRCh38. LD references for European, East Asian and African ancestry are
  ready to use, and one for any other population can be built from VCF or PLINK files.
- **Simple installation.** A pure-Python package; the reference data needed for a
  standard analysis are a download of about 30 MB per ancestry, verified by checksum.
- **A command line, a Python API and machine-readable results**, with helpers for
  running many traits on the NIH Biowulf cluster.

## Quick example

::::{tab-set}

:::{tab-item} Command line
```bash
pip install "genesis-gwas @ git+https://github.com/jeyabbalas/GENESIS.git"
genesis demo                                    # end-to-end check on packaged data
genesis fit trait.sumstats.tsv.gz --out results/trait --qq --project 500000,1000000
```
:::

:::{tab-item} Python
```python
import genesis_gwas as gg

ref = gg.load_reference()                           # European; "EAS" or "AFR" for other ancestries
data = gg.prepare("trait.sumstats.tsv.gz", ref)     # QC + merge with the LD reference
m2 = gg.fit(data, 2)                                # two-component model
m3 = gg.fit(data, 3, starting=m2)                   # three-component model
best = gg.select_model([m2, m3]).best               # by BIC
print(best.summary())
print(gg.project(best, n=[500_000, 1_000_000]).to_frame())
```
:::

::::

## Where to go

::::{grid} 1 2 3 3
:gutter: 2

:::{grid-item-card} Installation
:link: getting-started/installation
:link-type: doc

pip, uv or conda; GPU support; reference data; offline use.
:::

:::{grid-item-card} Quick start
:link: getting-started/quickstart
:link-type: doc

A first analysis in five minutes on data shipped with the package.
:::

:::{grid-item-card} Tutorial
:link: tutorial/index
:link-type: doc

The model, the data, fitting, interpretation and projections — step by step, with
figures.
:::

:::{grid-item-card} Theory
:link: theory/index
:link-type: doc

The likelihood, the EM algorithm and the sandwich variance, derived in full.
:::

:::{grid-item-card} Biowulf guide
:link: biowulf/index
:link-type: doc

CPU and GPU jobs and swarms of many traits on the NIH HPC cluster.
:::

:::{grid-item-card} Command line
:link: reference/cli
:link-type: doc

Every `genesis` command and option.
:::

:::{grid-item-card} Python API
:link: api/index
:link-type: doc

Functions and classes of `genesis_gwas`.
:::

:::{grid-item-card} Output files
:link: reference/outputs
:link-type: doc

Every file and every field that is written.
:::

:::{grid-item-card} Differences from the R package
:link: reference/differences
:link-type: doc

What changed, why, and by how much.
:::

:::{grid-item-card} Validation
:link: reference/validation
:link-type: doc

How correctness was established, and what has not been verified.
:::

:::{grid-item-card} FAQ
:link: reference/faq
:link-type: doc

Sample sizes for case–control studies, other ancestries, warnings, run times.
:::

:::{grid-item-card} References
:link: reference/bibliography
:link-type: doc

The method, its statistical background, and related approaches.
:::

::::

(how-to-cite)=
## How to cite

Please cite the paper that introduced the method:

> Zhang Y, Qi G, Park J-H, Chatterjee N (2018). Estimation of complex effect-size
> distributions using summary-level statistics from genome-wide association studies
> across 32 complex traits. *Nature Genetics* 50:1318–1326.
> [doi:10.1038/s41588-018-0193-x](https://doi.org/10.1038/s41588-018-0193-x)

and state the software and version used, for example: "analyses were run with
genesis-gwas {{ version }} (<https://github.com/jeyabbalas/GENESIS>)". The version is
printed by `genesis --version` and recorded in every result file. A `CITATION.cff` file
with this information is in the repository.

## License and acknowledgement

`genesis-gwas` is free software under the GNU General Public License, version 2 or
(at your option) any later version (GPL-2.0-or-later). It is a derivative work of the
R package [GENESIS](https://github.com/yandorazhang/GENESIS) by Yan Zhang, Guanghao Qi,
Ju-Hyun Park and Nilanjan Chatterjee, which is distributed under GPL (≥ 2). The method,
the European LD reference (computed by its authors from 1000 Genomes Project data), the
East Asian LD reference (from the same authors'
[GENESISasian](https://github.com/yandorazhang/GENESISasian) package) and
the example data set (GIANT height GWAS, [Lango Allen et al. 2010](#ref-langoallen2010))
are taken from that package.

```{toctree}
:hidden:
:caption: Getting started

getting-started/installation
getting-started/quickstart
```

```{toctree}
:hidden:
:caption: User guide

tutorial/index
theory/index
biowulf/index
```

```{toctree}
:hidden:
:caption: Reference

reference/cli
api/index
reference/outputs
reference/differences
reference/validation
reference/faq
reference/bibliography
reference/changelog
```
