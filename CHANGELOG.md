# Changelog

All notable changes to `genesis-gwas` are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html). Until version 1.0 the
public API (the names exported by `genesis_gwas`, the `genesis` command line and the
result-file schema) may still change between minor versions; such changes will be
listed under "Changed".

## 0.2.0 - 2026-11-23

References for East Asian and African ancestry, and a path from public genotypes to a
reference for any population.

### Added

- **East Asian reference `eas_1kg_hm3`**: the reference data of the authors' second R
  package, [GENESISasian](https://github.com/yandorazhang/GENESISasian) (commit
  `4df90ac`; 1000 Genomes Phase 3, 504 East Asian samples, 953,834 HapMap3 SNPs),
  converted without loss. With it, `legacy` mode reproduces GENESISasian to rounding
  error (tested against fixtures generated with that package). GENESISasian ships one LD
  configuration (1 Mb, r² ≥ 0.1); the other eight were computed with this software from
  the same genotypes, after confirming that it reproduces the original table exactly.
- **African-ancestry reference `afr_1kg_hm3`**: built with this software from the 661
  `AFR` samples of 1000 Genomes Phase 3 (1,025,309 HapMap3 SNPs outside the MHC with
  MAF ≥ 0.05; nine LD configurations; 100 null replicates; GRCh37 and GRCh38
  positions).
- Both are downloaded on demand like the European reference (release `refdata-v2`).
  The population codes `EUR`, `EAS` and `AFR` are accepted wherever a reference name is
  (`--reference AFR`, `genesis data fetch eas`, `load_reference("AFR")`).
- **`genesis ld from-vcf`** (`genesis_gwas.ld.vcf_to_plink`): converts VCF files, such
  as those of the 1000 Genomes Project, to the PLINK filesets that `genesis ld build`
  reads, keeping the SNPs and samples on a list; pure Python, no external tools.
- **`genesis ld build --liftover-chain`** (`genesis_gwas.ld.liftover`): a built
  reference can carry its positions in the other genome build as well, so that summary
  statistics are matched by position in GRCh37 or GRCh38; panels in GRCh38 are
  supported (`pos_grch37`). `--panel` records a description of the panel.
- `genesis ld from-vcf --positions` selects SNPs by position instead of identifier,
  for panels whose variants have no rsIDs.
- The HapMap3-without-MHC SNP list used for the African-ancestry reference is
  distributed with the reference data (`hapmap3_noMHC.snplist.gz`), together with a
  table of its positions in GRCh37 and GRCh38 (`hapmap3_noMHC.positions.tsv.gz`).
- `genesis data list --summary`; `genesis data list` shows population, number of SNPs,
  panel size and genome build of every reference.
- Simulations with the LD of the East Asian and African-ancestry panels
  ([Validation](https://jeyabbalas.github.io/GENESIS/reference/validation.html)).

### Changed

- `genesis ld build` stores pairwise r² for the 1 Mb / r² ≥ 0.1 configuration when it
  is among those built (previously for the first configuration listed).
- The positional MHC filter of quality control also applies to references whose native
  build is GRCh38 (previously it was skipped with a warning).
- The log of `genesis fit` states how markers were matched to the reference (by
  identifier, or by position and in which build).
- `examples/biowulf/fetch_refdata.sh` fetches any number of references;
  `project_swarm.py` uses the downloadable references for the ancestries `EUR`, `EAS`
  and `AFR`.

### Fixed

- Case or control counts that are zero or negative are treated as missing instead of
  entering the effective sample size (a negative count could give a positive effective
  sample size); a file in which no row has valid counts is an error.

### Known limitations

- No real East Asian or African-ancestry GWAS was analysed; the support rests on
  reproducing the original reference tables, on parity with GENESISasian for a
  simulated GWAS, and on simulations with the LD of each panel.
- Building a reference from a panel in GRCh38 (`genesis ld from-vcf --positions`,
  `--genome-build GRCh38`) is covered by unit tests and a chromosome-22 check only.
- Functions called with a bare parameter vector instead of a fitted model assume
  M = 1,070,777, the European reference; with another reference pass the fit or `M=`.
- As before, nothing was run on CUDA hardware or on the Biowulf cluster.

## 0.1.0 - 2026-11-21

First release: a re-implementation in Python of the GENESIS method of Zhang, Qi, Park
and Chatterjee (*Nature Genetics* 2018) and of the functionality of the authors'
R package ([yandorazhang/GENESIS](https://github.com/yandorazhang/GENESIS), commit
`e4e6894`).

### Added

- **Model fitting** (`genesis_gwas.fit`, `genesis fit`): two- and three-component
  mixture models fitted by composite likelihood; EM algorithm completed by Newton's
  method; a data-driven rule for the truncation level of the likelihood, with a warning
  when the estimates are limited by it; sandwich standard errors over LD tagging
  neighbourhoods, reported under both conventions in use; composite-likelihood
  AIC and modified BIC; model selection (`select_model`); derived quantities (number of
  susceptibility SNPs, heritability, per-cluster quantities) with delta-method standard
  errors; liability-scale heritability for binary traits.
- **`legacy` mode** that reproduces the estimates, covariance matrices and information
  criteria of the original R/C++ implementation to rounding error, tested against
  fixtures generated with the original code.
- **Compute backends**: multi-threaded NumPy on CPUs and PyTorch on one or several CUDA
  GPUs, sharing one implementation of the kernels, in double precision; `genesis doctor`
  to verify that every available backend agrees.
- **Data layer**: summary-statistics reader with detection of column names and effect
  scales, and of case and control counts (effective sample size; a warning when a
  sample-size column may follow another convention); quality control with a
  step-by-step report; matching to the LD reference by identifier or by position
  (GRCh37 and GRCh38).
- **Reference data**: the European reference of the original package (1000 Genomes
  Phase 3, 1,070,777 HapMap3 SNPs, nine LD configurations) converted to a compact
  format and downloaded on demand with checksum verification (`genesis data`); a
  chromosome-22 subset and example data shipped inside the package.
- **LD reference builder** (`genesis_gwas.ld`, `genesis ld build`) for any population
  from PLINK genotype files, validated against the original reference tables.
- **Downstream analyses**: effect-size density, projections of future discoveries with
  confidence intervals, expected accuracy of polygenic risk scores, per-SNP posterior
  summaries, model-based QQ plots, predicted genomic-control factors, simulation of
  summary statistics.
- **Plotting** functions for QQ plots, effect-size densities, projections and AUC.
- **Command-line interface** `genesis` with the sub-commands `fit`, `project`,
  `predict`, `posterior`, `qq`, `plot`, `ld build`, `data`, `swarm`, `demo` and
  `doctor`; machine-readable result files (JSON and TSV).
- **Documentation**: tutorial, theory, guide for the NIH Biowulf cluster, API and
  command-line reference, and a description of every difference from the original
  package (<https://jeyabbalas.github.io/GENESIS/>).

### Changed relative to the original R package

The defaults differ from the original where its code departs from the published method
or makes results depend on incidental details; `legacy=True` (`--legacy`) restores the
original behaviour of the fitting code. The main differences — each documented with its
reason and measured effect in
[Differences from the original](https://jeyabbalas.github.io/GENESIS/reference/differences.html) —
are:

- the reported estimate is the maximiser of the composite likelihood rather than the
  point at which the EM iteration stops; it is reached by a short EM warm-up followed
  by Newton's method, which needs far fewer passes over the data;
- the truncation level of the likelihood (`c0`) is no longer changed during the
  iteration, which made it depend on the starting values; it is raised after a fit
  until the probability mass discarded by the truncation is negligible at the solution;
- a marker's own score is counted once in the sandwich estimator, as in the paper's
  formula (the original counts it twice);
- the information matrix of the three-component model is symmetric;
- three-component fits that are not identified by the data are stopped early, flagged
  and not selected;
- per-SNP posterior summaries include the fitted inflation term in the noise variance;
- quality-control filters are applied by default.

### Known limitations

- GPU execution has not been run on CUDA hardware; the device-independent kernels are
  tested on CPUs. Run `genesis doctor` on a GPU node before relying on GPU results.
- The job scripts of the Biowulf guide follow the NIH HPC documentation but have not
  been executed on the cluster.
- Only a European LD reference is distributed; other populations require
  `genesis ld build`.
