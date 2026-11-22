# Analysing many traits and ancestries

A consortium rarely has one GWAS. It has an overall analysis and several subtypes, each
in several ancestries — the breast-cancer analyses of the Confluence project are the
application this software was prepared for. This chapter organises such a project: how to
lay out the files, check them before fitting, run one fit per trait and ancestry as batch
jobs, collect the results into one table, and compare with the original software.

The mechanics are demonstrated on three small simulated files, so that everything shown
here was actually run. They are toy data on chromosome 22 and say nothing about any real
trait.

## The plan

One fit per **trait × ancestry**, each with the reference of its own ancestry.

- Fit each ancestry separately. A meta-analysis across ancestries has no single LD
  structure, and no reference is correct for it ([Chapter 7](other-ancestries.md)).
- Use the same LD definition, SNP set and options for every fit that will be compared.
- Give every fit its own output prefix, `results/TRAIT.ANCESTRY`. All outputs of a fit
  share the prefix, and the collection step below reads trait and ancestry from it.

```text
project/
├── gwas/                        summary statistics, one file per trait and ancestry
│   ├── overall.EUR.tsv.gz
│   ├── subtypeA.EUR.tsv.gz
│   └── overall.AFR.tsv.gz
├── refs/                        references you built (the European one is downloaded)
│   └── afr_1kg_hm3/
├── results/                     TRAIT.ANCESTRY.{M2,M3,best}.json, .estimates.tsv, .summary.txt, .log
├── genesis.swarm                one command per fit
└── scripts/                     preflight.py, collect_estimates.py
```

## Toy data

```{literalinclude} ../_scripts/examples/make_toy_gwas.py
:language: python
```

```{literalinclude} ../_scripts/output/toy_gwas.txt
:language: console
```

## Step 1 — check every file before fitting anything

Most problems are visible without fitting: a wrong sample-size convention, a file in
another genome build, half of the SNPs missing. The script below reads and
quality-controls each file and prints one line per file. It takes seconds per file.

```{literalinclude} ../_scripts/examples/preflight.py
:language: python
```

```{literalinclude} ../_scripts/output/preflight.txt
:language: console
```

What to look for:

- **`Z from; N from`** — the derivation is the intended one, and for case–control data
  the sample size is the effective one ({ref}`tut-effective-n`). Run the standard-error
  check of {ref}`tut-check-n` on at least one file per source.
- **`kept`** — a large fraction of the reference. A low number means an identifier or
  build problem, or a file restricted to a genotyping array.
- **`chi2>80`** — how many markers the outlier filter removes
  ({ref}`tut-chisq-filter`).
- **`median N`** — plausible for the study, and consistent across files that should have
  similar sizes.
- **`mean chi2`, `lambda_GC`** — comparable with what the consortium reports. A mean
  $\chi^2$ close to 1 means there is little signal to fit; the three-component model, in
  particular, needs a well-powered study.

## Step 2 — one command per fit

`genesis swarm` writes a file with one `genesis fit` command per input file. Everything
after `--` is passed on to every fit. **It submits nothing**; it prints the command that
would, and logs the dry-run form of that command (`swarm --devel`), which shows what
would be submitted without submitting it.

```{literalinclude} ../_scripts/output/swarm.txt
:language: console
```

```{literalinclude} ../_scripts/output/swarm_file.txt
:language: console
```

The lines do not depend on where the swarm is submitted from: paths are written out in
full (shown here for a project in `/data/jdoe/project`), and the directory of the
reference data — `GENESIS_DATA_DIR` when the command was run, or `--data-dir` — is part
of every line. The directory for the swarm's own log files, `OUTDIR/logs`, is created,
because `swarm` expects it to exist.

The options before `--` describe the jobs: CPUs, memory and wall time per fit,
`--bundle` to run several short fits one after another in each subjob, `--partition`,
`--gpu` to request a GPU, and `--activate` for a line that activates your software
environment inside each job. The options after `--` are ordinary `genesis fit` options;
here they select the packaged chromosome-22 reference so that the toy example needs no
download.

Because the options after `--` apply to every file, write one swarm file per ancestry
and concatenate them:

```console
$ genesis swarm gwas/*.EUR.tsv.gz --outdir results --swarmfile eur.swarm -- --model both
$ genesis swarm gwas/*.AFR.tsv.gz --outdir results --swarmfile afr.swarm -- --model both --reference refs/afr_1kg_hm3
$ cat eur.swarm afr.swarm > genesis.swarm
```

(These three lines are a template and were not run.)

**On Biowulf**, submit the swarm file yourself from the login node with the command that
`genesis swarm` printed. [Running on Biowulf](../biowulf/index.md) covers the environment,
reference data on the cluster, resources, GPUs and monitoring.

**Anywhere else**, a swarm file is just a list of commands. For the toy example they are
run one after another on the local machine:

```{literalinclude} ../_scripts/output/swarm_local.txt
:language: console
```

:::{admonition} Resources for a real fit
:class: tip

A fit of both models to the height GWAS (about 940,000 markers) took 40 seconds with 8
CPU threads on a laptop and used 2.6 GB of memory at its peak. Requesting 8 CPUs, 8 GB
and an hour per trait leaves a wide margin. Time grows in proportion to the number of
markers; add a few minutes for `--qq`. Fetch the reference data once before submitting
(`genesis data fetch`, {ref}`tut-reference`) so that jobs do not download it
concurrently.
:::

## Step 3 — collect the results

Every fit writes `PREFIX.estimates.tsv`, a long table with one row per quantity and
model. Stacking these files gives one table for the whole project:

```{literalinclude} ../_scripts/examples/collect_estimates.py
:language: python
```

```{literalinclude} ../_scripts/output/collect.txt
:language: console
```

`all_estimates.tsv` holds everything — both models, all parameters and derived
quantities, log-likelihoods, BIC, convergence status. The summary printed here is one
view of it. (With one chromosome and a few dozen detectable effects the toy estimates are
noisy, as their standard errors say; BIC selects the two-component model, from which
the data were simulated, for all three.) Before interpreting anything, check the whole column `converged`, and read
`PREFIX.summary.txt` of any fit that is flagged or whose three-component model was set
aside as not identified ({ref}`tut-warnings`).

## Step 4 — compare traits with care

- **Compare like with like.** Same reference SNP set, same LD definition, same quality
  control. The absolute number of susceptibility SNPs depends on these choices
  ({ref}`tut-ld-sensitivity`); differences between traits depend on them much less.
- **Sample size matters.** The estimated number of susceptibility SNPs is a lower bound
  that rises with power ({doc}`concepts`). A subtype with a fifth of the cases will show
  fewer susceptibility SNPs than the overall analysis for that reason alone. Heritability
  and the numbers of SNPs with effects above a fixed size (`gg.n_in_interval`) are the
  fairer comparison.
- **Model choice may differ between traits.** If BIC selects M3 for one trait and M2 for
  another, also compare them under the same model.
- **Estimates are not independent.** Subtype analyses usually share controls, and an
  overall analysis contains its subtypes. The standard errors of two such fits cannot be
  combined into a test of their difference as if the fits were independent.
- **Scale.** For binary traits everything is on the log-odds scale unless converted
  ({ref}`tut-h2-scales`); conversions need the prevalence of each outcome.

## Comparing with the original software

`--legacy` reproduces the numerical behaviour of the original R package. Running one
trait both ways shows how much the documented differences matter for your data:

```{literalinclude} ../_scripts/output/legacy_fit.txt
:language: console
```

```{literalinclude} ../_scripts/output/legacy_compare.txt
:language: console
```

Column A is the default, column B the original algorithm. The defaults differ from the
original in the point estimate (the maximiser instead of the point where the EM
algorithm stops; the truncation level) and in the standard errors (the sandwich
convention). For this toy file the estimates differ by less than half a per cent and the
standard errors by 3 %; for the height data the differences are larger.
[Differences from the original](../reference/differences.md) quantifies each of them, and
[Validation](../reference/validation.md) documents the agreement of `--legacy` with the
original software itself.

For results that will be compared with published GENESIS analyses, state which mode was
used.

## Pre-flight checklist

Data

- [ ] One file per trait and ancestry; no multi-ancestry meta-analysis.
- [ ] `Z from …; N from …` is right for every file; effective sample size for binary
      traits, verified against standard errors where possible.
- [ ] Number of matched markers is high for every file; genome build understood.
- [ ] Number of markers removed by each filter recorded.

Reference

- [ ] A reference for each ancestry, built from a matching panel of at least 200
      individuals, with the same kind of SNP set and the same LD definitions.
- [ ] Reference data present on the cluster before jobs start.

Fitting

- [ ] Same options for all fits that will be compared; software version recorded (it is
      in every result file).
- [ ] One fit run interactively and read end to end before the batch is submitted.

Results

- [ ] `converged` for every fit; warnings read.
- [ ] QQ plot of the selected model inspected for the main traits.
- [ ] Sensitivity to the LD definition examined for the headline numbers.
- [ ] Scale of heritability, and the prevalence used for any conversion, stated.
