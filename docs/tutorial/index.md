# Tutorial

This tutorial takes you from a file of GWAS summary statistics to an estimated
effect-size distribution and to what follows from it: the number of susceptibility
SNPs, heritability, projections for larger studies, the expected accuracy of polygenic
scores, and per-SNP posterior summaries.

It is written for analysts who know what a GWAS is and can run a command or a few lines
of Python. No knowledge of mixture models is assumed: [the first chapter](concepts.md)
explains the model in pictures, and the [theory section](../theory/index.md) has the
derivations for readers who want them.

The running example is the GIANT height GWAS of
[Lango Allen et al. (2010)](../reference/bibliography.md#ref-langoallen2010), the example
of the original software, so that every number can be compared with
[Zhang et al. (2018)](../reference/bibliography.md#ref-zhang2018).

## What GENESIS does

GENESIS fits a model in which a small fraction $\pi_c$ of common SNPs are *susceptibility
SNPs* whose effects follow a normal distribution (or a mixture of two), and all other
SNPs have no effect. It needs only

- one row per SNP of **summary statistics** — a $z$ statistic (or an effect with its
  standard error, or a P-value with a sign) and a sample size; and
- a **linkage-disequilibrium (LD) reference** for the ancestry of the study, which the
  package downloads on first use.

```{mermaid}
flowchart LR
    subgraph inputs ["You provide"]
        SS["GWAS summary statistics<br/>SNP, Z (or BETA/SE, OR, P), N"]
    end
    subgraph ref ["Downloaded on first use"]
        LD["LD reference<br/>N*, LD score, neighbourhoods"]
    end
    SS --> PREP["<b>Prepare</b><br/>derive Z and N,<br/>quality control,<br/>match to the reference"]
    LD --> PREP
    PREP --> M2["<b>Fit M2</b><br/>one normal component"]
    M2 --> M3["<b>Fit M3</b><br/>two normal components"]
    M2 --> SEL{"<b>Choose</b><br/>by BIC,<br/>check the QQ plot"}
    M3 --> SEL
    SEL --> EST["<b>Estimates</b><br/>susceptibility SNPs,<br/>heritability,<br/>effect-size distribution"]
    EST --> PROJ["Projections for<br/>larger GWAS"]
    EST --> PRS["Expected accuracy of<br/>polygenic scores"]
    EST --> POST["Per-SNP posterior<br/>probabilities and effects"]
```

One command performs the whole middle part of the diagram:

```console
$ genesis fit height_giant2010.sumstats.tsv.gz --out results/height
```

and the same steps are available as Python functions
({func}`genesis_gwas.prepare`, {func}`genesis_gwas.fit`,
{func}`genesis_gwas.select_model`, …) when you want to work interactively.

## Chapters

::::{grid} 1 2 2 2
:gutter: 2

:::{grid-item-card} 1 · The model in pictures
:link: concepts
:link-type: doc

Joint and marginal effects, tagging neighbourhoods, the mixture model, and what each
parameter means.
:::

:::{grid-item-card} 2 · Preparing the data
:link: data
:link-type: doc

Input formats, deriving $z$ and $N$, the effective sample size, quality control, and the
LD reference.
:::

:::{grid-item-card} 3 · Fitting the model
:link: fitting
:link-type: doc

`genesis fit` and `gg.fit`, reading the output, convergence, model choice, and the QQ
plot.
:::

:::{grid-item-card} 4 · Interpreting the estimates
:link: interpreting
:link-type: doc

From parameters to numbers of SNPs and heritability, standard errors, and how much the
answers depend on the LD definition.
:::

:::{grid-item-card} 5 · Projections, prediction and per-SNP results
:link: downstream
:link-type: doc

Discoveries in larger studies, polygenic-score accuracy, local false-discovery rates,
shrunken effects, and future genomic inflation.
:::

:::{grid-item-card} 6 · Binary traits
:link: binary-traits
:link-type: doc

Effective sample size for case–control studies, the scale of heritability, sibling
relative risk, and AUC.
:::

:::{grid-item-card} 7 · Other ancestries
:link: other-ancestries
:link-type: doc

Building an LD reference from your own genotypes with `genesis ld build`, and what has
and has not been validated.
:::

:::{grid-item-card} 8 · Many traits and ancestries
:link: consortium
:link-type: doc

Organising a consortium-scale analysis: one fit per trait and ancestry, batch jobs,
collecting results, and a pre-flight checklist.
:::
::::

## Before you start

**Install** the package as described in
[Installation](../getting-started/installation.md) and check that it runs:

```console
$ genesis --version
```

**Reference data.** The European reference used throughout (`eur_1kg_hm3`) is fetched
automatically the first time it is needed; the files for a standard fit total 32 MB.
[Chapter 2](data.md) shows how to download it ahead of time, which you will want on a
cluster or behind a firewall. A chromosome-22 subset (`mini_chr22`) is packaged with the
software and needs no download; the tutorial uses it where speed matters.

**Working directory.** All commands are run from one empty directory that contains the
example file and a `results/` folder:

```console
$ mkdir genesis-tutorial && cd genesis-tutorial && mkdir results
$ cp "$(python -c 'import genesis_gwas as gg; print(gg.load_reference().example_path())')" \
     height_giant2010.sumstats.tsv.gz
```

## How long things take

Measured on one laptop (Apple M5 Pro, 12 CPU threads, no GPU) with the height GWAS
(941,684 markers after matching). Treat the numbers as orders of magnitude.

| Step | Command | Time |
|---|---|---|
| two-component model, with standard errors | `genesis fit --model 2` | 5 s |
| three-component model, with standard errors | `genesis fit --model 3` | 35 s |
| both models, QQ plot with 100 simulated data sets, posterior table for all markers, projections | `genesis fit --qq --posterior --project …` | 1–2 min |
| projections with intervals, polygenic-score predictions | `genesis project`, `genesis predict` | < 1 s |

A fit of both models used 2.7 GB of memory at its peak. Fitting time is proportional to
the number of markers, and the three-component model does at least eleven times more work
per pass over the data than the two-component model. More CPU cores help — a pass was
about five times faster with 12 threads than with one on this machine — and so does a
CUDA GPU, which was not available for these measurements. On a cluster one trait is a
small job; see [Running on Biowulf](../biowulf/index.md) for many traits at once.

## About the examples

Every command and script in this tutorial was executed, and the output shown is what it
printed. The scripts are in `docs/_scripts/examples/` of the repository, and
`docs/_scripts/run_examples.sh` runs all of them and regenerates the outputs and figures.
Output may differ in the last digits across machines and package versions.

```{toctree}
:hidden:
:maxdepth: 1

concepts
data
fitting
interpreting
downstream
binary-traits
other-ancestries
consortium
```
