# How GENESIS works

This part of the documentation is for readers who want to understand, check or extend the
method: every formula the software evaluates is derived here and tied to the function that
implements it. Nothing in it is needed to *run* an analysis — for that, start with the
[tutorial](../tutorial/index.md).

The method is that of
[Zhang, Qi, Park & Chatterjee (2018)](../reference/bibliography.md#ref-zhang2018). Where
this implementation deliberately departs from the original R/C++ package, the departure is
catalogued, with its measured effect, in
[Differences from the original](../reference/differences.md); the chapters below describe
what the software does by default and point there rather than repeating the argument.

## The pieces and how they fit

```{mermaid}
flowchart TB
    subgraph inputs [Inputs]
        Z["GWAS summary statistics<br/>z<sub>k</sub>, n<sub>k</sub> for K markers"]
        LD["LD reference<br/>N*<sub>k</sub>, ℓ<sub>k</sub>, tagging neighbourhoods"]
    end
    subgraph model [1 · Model]
        P["Effect-size distribution<br/>π<sub>c</sub>, p<sub>h</sub>, σ²<sub>h</sub>  (eq. 1)"]
        MX["Mixture for each marker<br/>over latent counts  (eq. 2–3)"]
        CL["Composite likelihood<br/>product over markers"]
    end
    subgraph est [2 · Estimation]
        EM["EM warm-up<br/>closed form + Newton M-step"]
        NP["Newton's method<br/>exact maximiser,<br/>truncation check"]
    end
    subgraph inf [3 · Inference]
        SC["Scores and information"]
        SW["Sandwich covariance<br/>with LD neighbourhoods  (eq. 4)"]
        BIC["Modified BIC<br/>M2 versus M3"]
    end
    subgraph down [4 · Downstream]
        PR["Projections for future GWAS"]
        PRS["Polygenic-score performance"]
        PO["Posterior: local FDR, shrinkage"]
        QQ["Simulation: QQ envelope, λ"]
    end
    Z --> MX
    LD --> MX
    P --> MX --> CL --> EM --> NP --> SC --> SW --> BIC
    LD --> SW
    NP --> PR & PRS & PO & QQ
    SW --> PR
```

| Chapter | Question it answers | Main source files |
|---|---|---|
| [The statistical model](model.md) | What is assumed, and where does the mixture for a marker's effect come from? | `core/spec.py`, `core/likelihood.py` |
| [Estimation](estimation.md) | How is the composite likelihood maximised, and when is the answer "converged"? | `core/kernels.py`, `core/em.py`, `core/start.py` |
| [Standard errors and model choice](inference.md) | How are standard errors obtained when markers are correlated, and how are M2 and M3 compared? | `core/variance.py`, `core/derived.py` |
| [Downstream quantities](downstream.md) | How do projections, polygenic-score performance, posteriors and QQ envelopes follow from a fit? | `analysis/*.py` |
| [Implementation](implementation.md) | How is all of this organised for speed, accuracy and testability on CPUs and GPUs? | `core/engine.py`, `backends/`, `data/` |

## Conventions

- Effects are on the **standardised scale**: genotypes and phenotype have unit variance, so
  an effect $\beta$ is in phenotype standard deviations per standard deviation of genotype.
  For a case–control study analysed with the [effective sample size](../tutorial/data.md),
  $\beta$ is a log odds ratio per standard deviation of genotype.
- $k = 1,\dots,K$ indexes the GWAS **markers** that enter the analysis; $m = 1,\dots,M$
  indexes the SNPs of the **reference panel** (for the bundled European reference,
  $M = 1{,}070{,}777$ HapMap3 SNPs).
- $H$ is the number of normal components for non-null effects: $H = 1$ for the
  two-component model **M2**, $H = 2$ for the three-component model **M3** (the names count
  the null component, as in the paper).
- Equations numbered **(1)–(4)** in the text are the paper's equations (1)–(4).

```{toctree}
:maxdepth: 2
:hidden:

model
estimation
inference
downstream
implementation
```
