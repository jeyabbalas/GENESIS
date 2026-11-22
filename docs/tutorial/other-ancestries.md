# Other ancestries: building an LD reference

The LD reference must describe the population of the GWAS. $N^*_k$ and $\ell_k$ enter
the likelihood of every marker, so a reference from the wrong population biases every
estimate. The software bundles a European reference only. For any other population you
build one from genotypes with `genesis ld build`; this chapter shows how, and verifies
the builder by reconstructing the bundled European reference from the 1000 Genomes
genotypes it came from.

## What you need

1. **Genotypes of a reference panel** from the population of the GWAS, in PLINK
   `.bed/.bim/.fam` format — one fileset, or one per chromosome. A few hundred unrelated
   individuals are enough (see {ref}`tut-panel-size`). Public panels such as the 1000
   Genomes Project ([Auton et al., 2015](../reference/bibliography.md#ref-auton2015)) are
   the usual source; a random subset of the study's own genotypes is better still.
2. **A list of SNPs** that defines the panel. The estimated proportion $\pi_c$ is a
   proportion *of these SNPs*, so choose them deliberately. The convention of the paper is
   the HapMap3 SNPs ([Altshuler et al., 2010](../reference/bibliography.md#ref-altshuler2010))
   with a minor-allele frequency of at least 5 % in the panel, outside the MHC.

The builder computes, for every SNP, its tagging neighbourhood, $N^*_k$ and $\ell_k$ as
defined in [Chapter 1](concepts.md), and writes a bundle in the same format as the
bundled reference ({ref}`impl-reference`).

## Rebuilding the European reference

The genotypes are the 489 European individuals of 1000 Genomes Phase 3 in the per-chromosome
PLINK files distributed with LD-score regression
([Bulik-Sullivan et al., 2015](../reference/bibliography.md#ref-bulik-sullivan2015)), here
in a directory `1000G_EUR_Phase3_plink/`. To make the comparison exact the SNP list is
that of the bundled reference:

```{literalinclude} ../_scripts/examples/write_snplist.py
:language: python
```

```{literalinclude} ../_scripts/output/ld_snplist.txt
:language: console
```

`{chrom}` in the file prefix stands for the chromosome number. The command builds all
nine LD definitions of the bundled reference — three windows times three thresholds — in
one pass:

```{literalinclude} ../_scripts/output/ld_build.txt
:language: console
```

The whole genome takes little more than a minute on a laptop. The log reports, per
chromosome, the SNPs kept and the number of tagging pairs under the most inclusive
definition, and at the end how many variants of the genotype files were set aside and
why (here, the 8.9 million that are not on the SNP list).

### Does it reproduce the bundled reference?

```{literalinclude} ../_scripts/examples/ld_compare.py
:language: python
```

```{literalinclude} ../_scripts/output/ld_compare.txt
:language: console
```

Under all nine definitions every neighbourhood and every $N^*_k$ is identical, and LD
scores agree to within $10^{-6}$. A fit with the rebuilt reference therefore gives the
same answer as with the bundled one (column A against column B):

```{literalinclude} ../_scripts/output/ld_fit.txt
:language: console
```

```{literalinclude} ../_scripts/output/ld_fit_compare.txt
:language: console
```

:::{admonition} Why `--keep-mhc` in this comparison
:class: note

The bundled reference carries the HapMap3-without-MHC list of the original software, and
quality control uses it. A reference you build has no such list, so quality control
falls back to excluding chromosome 6, 26–34 Mb, by position. The two definitions are not
the same: the original list retains 70 SNPs inside that interval, 58 of them in the
height file. `--keep-mhc` switches the positional filter off so that both fits analyse
exactly the same markers; the height file contains no other MHC SNPs. For a reference of
your own, the cleanest approach is to leave the MHC out of the SNP list you build from.
The positional filter is only applied to references in GRCh37 coordinates.
:::

### What `--round-r2 6` is for

The command above rounds $r^2$ to six significant digits before applying the threshold.
That detail matters only for reproducing the original tables, which were evidently
derived from a printed table of $r^2$ values. Without rounding:

```{literalinclude} ../_scripts/output/ld_build_exact.txt
:language: console
```

```{literalinclude} ../_scripts/output/ld_compare_exact.txt
:language: console
```

A handful of SNP pairs whose $r^2$ lies within rounding error of the threshold fall on
the other side of it, which changes $N^*_k$ for 10 of the 1,070,777 SNPs. The effect on
any analysis is nil. **For a new reference, leave `--round-r2` out:** rounding is an
artefact of how the original tables were made, not part of the method.

## Building a reference for another population

```console
$ genesis ld build \
      --bfile panels/1000G.AFR.{chrom} \
      --keep afr_unrelated.txt \
      --snps hapmap3_noMHC.snplist \
      --window-kb 500,1000,2000 --r2 0.05,0.1,0.2 \
      --n-null 100 \
      --name afr_1kg_hm3 --population AFR --genome-build GRCh38 \
      --out refs/afr_1kg_hm3

$ genesis fit my_gwas.tsv.gz --reference refs/afr_1kg_hm3 --out results/my_gwas
```

(This command is a template and was not run for this tutorial; the file names are
placeholders.) What the options do:

`--keep`
: the samples to use, one identifier per line. Remove close relatives and population
  outliers first.

`--snps`
: the SNP list. Without it, every autosomal SNP of the fileset that passes the frequency
  filter is used.

`--min-maf`
: minimum minor-allele frequency *in the selected samples* (default 0.05). The number of
  SNPs that survive is the $M$ of the reference and is reported in the log.

`--window-kb`, `--r2`
: one or several values each; every combination is computed in a single pass. Building
  the nine combinations of the bundled reference lets you repeat the sensitivity analysis
  of {ref}`tut-ld-sensitivity`.

`--n-null`
: number of LD-correlated null $z$ replicates to store. They are needed for the envelopes
  of QQ plots and for `future_gc`; 100 replicates take about 0.2 GB per million SNPs.
  For a panel of a few hundred individuals or fewer, add `--null-statistic normal`.

`--genome-build`
: the build of the positions in the `.bim` files. It matters only when summary
  statistics are matched by position.

The bundle is a directory; copy it wherever the analysis runs and pass its path to
`--reference` or {func}`genesis_gwas.load_reference`. `genesis data verify DIR` checks its
checksums.

(tut-panel-size)=
## Choices that matter

**The SNP set.** Results are stated relative to the SNPs of the reference: the number of
susceptibility SNPs is $M\pi_c$ and the heritability is that explained by those $M$ SNPs.
Estimates for two populations are only comparable if their references were built from
the same kind of SNP set. The 5 % frequency filter is applied within each panel, so the
sets — and $M$ — will differ somewhat between populations; report $M$.

**The size of the panel.** The threshold is applied to the *sample* $r^2$. For two SNPs
that are not in LD at all, $n\,r^2$ is approximately $\chi^2_1$ in a panel of $n$
individuals, so the chance of exceeding the threshold of 0.1 by accident is about
$3\times10^{-12}$ for $n = 489$, $8\times10^{-6}$ for $n = 200$, $1.6\times10^{-3}$ for
$n = 100$ and $2.5\times10^{-2}$ for $n = 50$. With some 700 reference SNPs within a
megabase of a typical marker, that is a negligible number of spurious neighbours at
$n = 200$, about one per marker at $n = 100$, and about 18 per marker at $n = 50$ —
against a median neighbourhood of 37 in Europeans. Use at least 200 individuals; below
that, raise the threshold or expect $N^*_k$ to be inflated by noise.

**Window and threshold.** The defaults (1 Mb, $r^2 \ge 0.1$) were chosen for Europeans.
LD extends over shorter distances in populations of African ancestry, so neighbourhoods
will be smaller at the same settings. Whether other settings are preferable there has not
been examined here; if results are to be compared across ancestries, use the same
settings for all and treat the sensitivity analysis as part of the result.

**Admixed populations.** The model assumes one LD structure for the whole sample. In an
admixed sample LD depends on the admixture proportions, and a panel from one of the
source populations is the wrong reference. The best available option is a panel with the
same admixture — ideally a random subset of the study's own genotypes.

## What has and has not been validated

Validated:

- The builder reproduces all nine tables of the original European reference from the
  1000 Genomes genotypes, as shown above; the definitions of $N^*_k$ and $\ell_k$ in this
  software are therefore those of the original.
- A fit with the rebuilt reference equals the fit with the bundled one.

Not validated:

- The statistical behaviour of the method in non-European populations. The analyses and
  simulations of [Zhang et al. (2018)](../reference/bibliography.md#ref-zhang2018) are
  European. Nothing in the method is specific to one ancestry, but its accuracy with a
  different LD structure, and with the smaller studies typical of other ancestries, has
  not been established here.
- Analyses of summary statistics from multi-ancestry meta-analyses. A meta-analysis
  across ancestries has no single LD structure, and no reference panel is correct for it.
  Fit each ancestry separately.

[Validation](../reference/validation.md) lists what is tested and how.
