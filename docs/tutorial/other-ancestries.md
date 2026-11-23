# Other ancestries

The LD reference must describe the population of the GWAS. $N^*_k$ and $\ell_k$ enter
the likelihood of every marker, so a reference from the wrong population biases every
estimate. This chapter covers the three ready-made references — European, East Asian and
African ancestry — and then shows how to build a reference for any other population from
genotypes, starting from the VCF files of the 1000 Genomes Project.

## Ready-made references

| Name | Short name | Panel (1000 Genomes Phase 3) | Samples | SNPs ($M$) | Origin |
|---|---|---|---:|---:|---|
| `eur_1kg_hm3` | `EUR` | European | 489 | 1,070,777 | the data of the original GENESIS package, converted |
| `eas_1kg_hm3` | `EAS` | East Asian | 504 | 953,834 | the data of the original GENESISasian package, converted; eight further LD definitions computed with this software |
| `afr_1kg_hm3` | `AFR` | African ancestry | 661 | 1,025,309 | built with this software, as shown below |

All three contain the HapMap3 SNPs
([Altshuler et al., 2010](../reference/bibliography.md#ref-altshuler2010)) with a
minor-allele frequency of at least 5 % in their own panel, offer the same nine LD
definitions (windows of 0.5, 1 and 2 Mb, $r^2$ thresholds of 0.05, 0.1 and 0.2) and 100
LD-correlated null replicates, and carry positions in GRCh37 and GRCh38. Select one with
`--reference`, by name or short name; the files are downloaded on first use:

```console
$ genesis fit gwas/overall.AFR.tsv.gz --reference AFR --out results/overall.AFR
$ genesis fit gwas/overall.EAS.tsv.gz --reference EAS --out results/overall.EAS
$ genesis data fetch afr_1kg_hm3 --components default,nullz      # ahead of time, e.g. on a cluster
```

```python
ref = gg.load_reference("afr_1kg_hm3")          # or "AFR"
```

(These lines are templates with placeholder file names.) Fetching ahead of time looks the
same as for the European reference ({ref}`tut-reference`):

```{literalinclude} ../_scripts/output/data_fetch_ancestries.txt
:language: console
```

`genesis data list` shows the references and what is installed:

```{literalinclude} ../_scripts/output/data_list_all.txt
:language: console
```

### How the three populations differ

The same LD definition gives very different neighbourhoods:

| Reference | Median $N^*_k$ | Mean $N^*_k$ | Median $\ell_k$ | Mean $\ell_k$ |
|---|---:|---:|---:|---:|
| `eur_1kg_hm3` | 37 | 46.5 | 13.0 | 17.2 |
| `eas_1kg_hm3` | 35 | 44.1 | 13.2 | 17.5 |
| `afr_1kg_hm3` | 17 | 23.3 | 5.8 | 8.0 |

(1 Mb, $r^2 \ge 0.1$.) A marker in the African-ancestry panel tags half as many SNPs as
one in the European or East Asian panel. Fitting an African-ancestry GWAS with a European
reference would attribute each marker's signal to twice as many SNPs as actually
contribute to it — which is why the reference has to match.

### The East Asian reference and GENESISasian

The authors of GENESIS published a second R package,
[GENESISasian](https://github.com/yandorazhang/GENESISasian), for East Asian studies. Its
fitting code is the same as that of GENESIS — the C++ source is byte-for-byte identical —
and the R code differs only in the default number of reference SNPs, 953,834 instead of
1,070,777. What differs is the data: LD tables from the 504 East Asian samples of 1000
Genomes Phase 3, for a single LD definition (1 Mb, $r^2 \ge 0.1$).

`eas_1kg_hm3` is a lossless conversion of those tables, so `--reference EAS --legacy`
reproduces GENESISasian, and without `--legacy` an East Asian GWAS gets the same
treatment as any other ([Validation](../reference/validation.md) has the comparison).
The other eight LD definitions were computed with this software from the same
genotypes, after checking that it reproduces the original table exactly (below). Two
things to know:

- The reference includes 1,775 SNPs in the MHC region, as the original does. They count
  towards $M$; quality control removes MHC markers from the GWAS as usual.
- The example data inside GENESISasian is the *European* height GWAS of the GENESIS
  package, so it is not distributed with this reference.

### The African-ancestry reference

`afr_1kg_hm3` uses all 661 samples of the 1000 Genomes `AFR` super-population: Yoruba
(YRI, 108), Luhya (LWK, 99), Gambian (GWD, 113), Mende (MSL, 85) and Esan (ESN, 99) from
Africa, and African Caribbean (ACB, 96) and African American (ASW, 61) samples, who have
some European admixture. That composition suits GWAS of African-American and other
African-diaspora populations better than a purely continental panel would, and a GWAS
within one African population less well. If your study calls for a different set of
samples, build a reference from them: it is the command below with another `--keep`
file.

No original tables exist for this population, so the reference cannot be compared with
anything the authors produced. What supports it is that the same builder reproduces the
European and East Asian tables from the same kind of genotypes, and that simulations
with the LD of this panel behave as they do for the other two
({ref}`tut-ancestry-validation`).

## Building a reference for another population

Two commands turn public genotypes into a reference. They are shown as they were run to
build `afr_1kg_hm3`; for another population only the list of samples changes.

### What you need

1. **Genotypes of a reference panel** from the population of the GWAS. A few hundred
   unrelated individuals are enough (see {ref}`tut-panel-size`). The usual source is the
   1000 Genomes Project ([Auton et al., 2015](../reference/bibliography.md#ref-auton2015));
   a random subset of the study's own genotypes is better still.
2. **A list of SNPs** that defines the panel. The estimated proportion $\pi_c$ is a
   proportion *of these SNPs*, so choose them deliberately. The convention of the paper is
   the HapMap3 SNPs with a minor-allele frequency of at least 5 % in the panel, outside
   the MHC. The list used here — the HapMap3 list of the original package without the
   SNPs in chromosome 6, 26–34 Mb — is distributed with the reference data as
   `hapmap3_noMHC.snplist.gz` (1,214,893 identifiers, one per line):
   <https://github.com/jeyabbalas/GENESIS/releases/download/refdata-v2/hapmap3_noMHC.snplist.gz>.
   For panels whose variants have no rsIDs there is a table of positions next to it
   ({ref}`tut-vcf-positions`).
3. Optionally, a **UCSC chain file** to add positions in the other genome build
   (`hg19ToHg38.over.chain.gz` for a GRCh37 panel, from
   <https://hgdownload.soe.ucsc.edu/goldenPath/hg19/liftOver/>). With it the reference
   matches summary statistics by position in either build, which matters when a file
   has no rsIDs.

### Step 1 — from VCF to PLINK files

The 1000 Genomes Phase 3 genotypes are 22 VCF files with 81 million variants and all
2,504 samples (`ALL.chr*.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf.gz`,
15 GB, from the project's FTP site or its mirror in the AWS Open Data registry).
`genesis ld from-vcf` reads them once and writes the SNPs on a list as PLINK files; it
needs no other software:

```{literalinclude} ../_scripts/output/kg_from_vcf.txt
:language: console
```

This took under two minutes with 14 processes. (The list used in this run,
`hm3_union.snplist`, is the HapMap3 list plus the MHC SNPs of the European and East Asian
references, so that the same files could serve the comparisons below.) All samples are
kept, so the same files serve every population. Only biallelic SNPs on chromosomes 1–22
are written; indels, multi-allelic records and repeated identifiers are counted and set
aside.

If you have PLINK 2, `plink2 --vcf … --extract … --make-bed` produces equivalent files,
and genotypes that are already in PLINK `.bed/.bim/.fam` format need no conversion at all.

(tut-vcf-positions)=
#### Panels without rsIDs

Newer panels — the high-coverage 1000 Genomes release, HGDP, TOPMed — are in GRCh38 and
often name their variants by position rather than by rsID, so a list of identifiers
selects nothing. `--positions FILE` selects by position instead (it cannot be combined
with `--snps`). The file has the columns `CHR POS` and optionally `ID`, with positions
in the build of the VCF; lines whose position is not a positive integer, such as a
header or `NA`, are ignored. With the third column each SNP is written under that name,
otherwise as `chrom:pos:ref:alt`. Only the first biallelic SNP at a position is written.

The positions of the HapMap3 list are distributed as
[`hapmap3_noMHC.positions.tsv.gz`](https://github.com/jeyabbalas/GENESIS/releases/download/refdata-v2/hapmap3_noMHC.positions.tsv.gz)
(13.8 MB; columns `CHR POS_GRCh37 POS_GRCh38 SNP`): the 1,203,399 SNPs of the list that
are biallelic SNPs in 1000 Genomes Phase 3, with `NA` in `POS_GRCh38` for the 626 that
the UCSC chain does not convert. For a GRCh38 panel:

```console
$ gzip -dc hapmap3_noMHC.positions.tsv.gz | awk 'NR>1 && $3!="NA" {print $1, $3, $4}' > hm3.grch38.positions
$ genesis ld from-vcf --vcf 'panel.chr{chrom}.vcf.gz' --positions hm3.grch38.positions --out 'plink/panel.{chrom}'
$ genesis ld build --bfile 'plink/panel.{chrom}' --keep panel.samples --genome-build GRCh38 \
      --liftover-chain hg38ToHg19.over.chain.gz --out refs/my_panel --name my_panel --population XYZ
```

(A template that was not run.) Because the SNPs are written under their rsIDs, the
reference matches summary statistics by identifier as well as by position in either
build.

:::{admonition} How far this route has been tested
:class: note

Selecting chromosome 22 of the Phase 3 files by GRCh37 position from this table gives
the same 17,757 SNPs and identical genotypes as selecting by identifier, and the
handling of GRCh38 references is covered by unit tests. A complete reference has not
been built from a GRCh38 panel.
:::

### Step 2 — compute the reference

The samples are those labelled `AFR` in the project's sample table
(`hm3_noMHC_strict.snplist` is the decompressed `hapmap3_noMHC.snplist.gz`); the builder
computes all nine LD definitions in one pass:

```{literalinclude} ../_scripts/output/afr_build.txt
:language: console
```

A minute and a half on a laptop. The log reports, per chromosome, the SNPs kept and the
number of tagging pairs under the most inclusive definition, then why variants were set
aside — here 173,216 HapMap3 SNPs with a minor-allele frequency below 5 % in this panel —
and how many positions could be converted to the other build.

What the options do:

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
  the nine combinations lets you repeat the sensitivity analysis of
  {ref}`tut-ld-sensitivity`.

`--n-null`
: number of LD-correlated null $z$ replicates to store. They are needed for the envelopes
  of QQ plots and for `future_gc`; 100 replicates take about 0.2 GB per million SNPs.
  For a panel of a few hundred individuals or fewer, add `--null-statistic normal`.

`--genome-build`, `--liftover-chain`
: the build of the positions in the `.bim` files, and the chain file from that build to
  the other one. Without a chain file the reference matches by position in its own build
  only.

`--name`, `--population`, `--panel`
: labels recorded in the bundle and in every result that uses it.

The bundle is a directory; copy it wherever the analysis runs and pass its path to
`--reference` or {func}`genesis_gwas.load_reference`. `genesis data verify DIR` checks its
checksums.

### Step 3 — use it

Summary statistics simulated with the LD of this panel — one two-component data set with
60,000 effective samples, positions in GRCh38 and no rsIDs — are matched by position and
fitted like any other file:

```{literalinclude} ../_scripts/output/afr_fit.txt
:language: console
```

(The simulated truth is 6,163 susceptibility SNPs and a heritability of 0.49; the
estimate of the number of susceptibility SNPs is too low, in the way described in
{doc}`concepts` and quantified below.)

## Does the builder reproduce the original references?

The original packages ship their LD tables but not the code that produced them. Rebuilding
those tables from the genotypes is therefore the test of the builder — and it is what
establishes that references built for other populations follow the same definitions.

### East Asian

From the PLINK files of step 1 and the 504 samples labelled `EAS`:

```{literalinclude} ../_scripts/output/eas_rebuild.txt
:language: console
```

Every $N^*_k$ and every tagging list of the 953,834 SNPs is identical to the original's,
and LD scores agree to $4\times10^{-6}$. The SNP set itself is reproduced by the rule
"HapMap3 SNPs with a minor-allele frequency of at least 5 % in the panel": the rule
selects 952,124 SNPs of the HapMap3-without-MHC list, of which 952,104 are in the
original reference (the original has no SNP that the rule rejects; its remaining 1,730
SNPs are in the MHC, which that list omits).

### European

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

#### Does it reproduce the bundled reference?

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
The positional filter uses the coordinates of the interval in the build of the
reference, GRCh37 or GRCh38.
:::

#### What `--round-r2 6` is for

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
same admixture — ideally a random subset of the study's own genotypes. For Hispanic/Latino
studies the 1000 Genomes `AMR` samples (347) are the nearest public panel, with the
caveat that admixture proportions vary widely between and within these populations.

**The genome build of the panel.** Newer panels (the 1000 Genomes high-coverage
re-sequencing, HGDP, TOPMed) are in GRCh38. Select their SNPs by position
({ref}`tut-vcf-positions`) and build with `--genome-build GRCh38` and
`--liftover-chain hg38ToHg19.over.chain.gz`; the MHC filter of quality control is applied
in the right coordinates either way. No reference has been built from such a panel here.

(tut-ancestry-validation)=
## What has and has not been validated

Validated:

- The builder reproduces all nine tables of the original European reference and the
  table of the original East Asian reference from 1000 Genomes genotypes, as shown above;
  the definitions of $N^*_k$ and $\ell_k$ in this software are therefore those of the
  originals.
- A fit with the rebuilt European reference equals the fit with the bundled one.
- `--legacy` with `eas_1kg_hm3` reproduces the GENESISasian package
  ([Validation](../reference/validation.md)).
- In simulations with the real LD of each panel, the method behaves the same way in all
  three populations:

  | LD of | Data sets | $\pi_c$: bias | Heritability: bias | Heritability: mean SE / empirical SD |
  |---|---:|---:|---:|---:|
  | `eur_1kg_hm3` | 100 | −27.5 % | −2.7 % | 0.93 |
  | `eas_1kg_hm3` | 50 | −27.2 % | −4.2 % | 1.09 |
  | `afr_1kg_hm3` | 50 | −31.7 % | −3.9 % | 0.98 |

  (Two-component truth with $\pi_c = 0.005$, $\sigma^2 = 6\times10^{-5}$ and 130,000
  samples; details in {ref}`simulation-ancestries`.) The number of susceptibility SNPs is
  underestimated by about the same amount everywhere, heritability is close to the
  truth, and its standard error is about right.

Not validated:

- **Real GWAS of non-European populations.** The analyses of
  [Zhang et al. (2018)](../reference/bibliography.md#ref-zhang2018) are European, and no
  East Asian or African-ancestry GWAS was analysed here. The simulations above generate
  data from the same panel that serves as the reference; a real study's LD differs from
  that of any panel, and more so the less well the panel matches — the main concern for
  admixed samples.
- **Small studies.** GWAS of non-European populations are often smaller. With little
  power the three-component model is frequently not identified and the two-component
  estimates have wide intervals; neither is specific to ancestry.
- **Panels in GRCh38.** Tested on chromosome 22 and by unit tests only
  ({ref}`tut-vcf-positions`).
- **The NIH Biowulf cluster and CUDA GPUs.** Nothing in this chapter was run on either.
- **Summary statistics from multi-ancestry meta-analyses.** A meta-analysis across
  ancestries has no single LD structure, and no reference panel is correct for it. Fit
  each ancestry separately.

[Validation](../reference/validation.md) lists what is tested and how.
