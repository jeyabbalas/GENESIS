# Command-line reference

The package installs one command, `genesis` (also available as `genesis-gwas`), with a
sub-command per task. `genesis <command> --help` prints the options described on this
page.

## Which command for which task

| Task | Command |
|---|---|
| Check the installation and the compute devices | `genesis doctor` |
| Run a complete analysis on packaged example data | `genesis demo` |
| Download or inspect reference data | `genesis data fetch`, `genesis data list`, `genesis data verify` |
| **Fit the effect-size distribution** (QC, merge, M2 and/or M3, model selection) | `genesis fit` |
| ... and in the same run a QQ plot, per-SNP posteriors, projections | `genesis fit --qq --posterior --project N1,N2,...` |
| Project discoveries of future GWAS from a fitted model | `genesis project` |
| Predict the accuracy of polygenic risk scores | `genesis predict` |
| Per-SNP local false-discovery rates and shrunken effects | `genesis posterior` |
| QQ plot of observed P-values against the fitted model | `genesis qq` |
| Plot effect-size densities or projection curves of one or more fits | `genesis plot` |
| Use the East Asian or African-ancestry reference | `genesis fit --reference EAS`, `--reference AFR` |
| Build an LD reference for another population from VCF or PLINK files | `genesis ld from-vcf`, `genesis ld build` |
| Write a Biowulf swarm file that fits many traits | `genesis swarm` |

## Typical invocations

```bash
# Quantitative trait; columns are detected (SNP, Z or BETA+SE, N)
genesis fit height.tsv.gz --out results/height

# Case-control study: with both counts, N = N_cases x N_controls / (N_cases + N_controls)
genesis fit t2d.tsv.gz --out results/t2d --n-cases 26676 --n-controls 132532 \
    --population-prevalence 0.08 --sample-prevalence 0.168

# Only the two-component model, another LD configuration, eight threads
# (give the window and the threshold together)
genesis fit trait.tsv.gz --out results/trait --model 2 --ld-window-kb 2000 --ld-r2 0.1 --threads 8

# Reproduce the original R package's worked example (two-component model, no QC
# filters, its starting value): same estimates and standard errors, to rounding error
genesis fit height.tsv.gz --out results/height_legacy --model 2 --legacy --no-qc --starting-pic 0.005

# Downstream analyses from a saved fit
genesis project   results/height.best.json --n 250000,500000,1000000 --ci
genesis predict   results/t2d.best.json --n 100000,1000000
genesis posterior height.tsv.gz --fit results/height.best.json --out results/height.posterior.tsv.gz
genesis qq        height.tsv.gz --fit results/height.best.json --out results/height
genesis plot density M2=results/height.M2.json M3=results/height.M3.json --out results/density.png

# East Asian or African ancestry: ready-made references, downloaded on first use
genesis fit trait_eas.tsv.gz --reference EAS --out results/trait_eas
genesis fit trait_afr.tsv.gz --reference AFR --out results/trait_afr

# Any other population: convert VCF files of a panel to PLINK files, build a
# reference from the chosen samples ({chrom} stands for the chromosome number), use it
genesis ld from-vcf --vcf 'ALL.chr{chrom}.vcf.gz' --snps hapmap3_noMHC.snplist --out 'plink/kg.{chrom}'
genesis ld build --bfile 'plink/kg.{chrom}' --keep sas.samples --snps hapmap3_noMHC.snplist \
    --n-null 100 --liftover-chain hg19ToHg38.over.chain.gz --out refs/sas --name sas --population SAS
genesis fit trait_sas.tsv.gz --reference refs/sas --out results/trait_sas

# Many traits on Biowulf: write a swarm file (nothing is submitted)
genesis swarm sumstats/*.tsv.gz --outdir results -- --qq --project 500000,1000000
```

File names and sample sizes in these examples are placeholders; each form of command
was run on the example data of this package.

Conventions shared by all commands:

- **Exit status** is 0 on success and non-zero on error. Errors are reported as one
  explanatory line; add `--debug` for the Python traceback.
- **Log messages** go to standard error (`-q` for warnings and errors only, `-v` for
  more detail); tables that a command prints go to standard output.
- **Threads.** `--threads`, else `$GENESIS_NUM_THREADS`, else `$SLURM_CPUS_PER_TASK`,
  else the number of CPUs available to the process.
- **Devices.** `--device auto` uses all visible CUDA GPUs if PyTorch with CUDA is
  installed, otherwise the CPU.
- **Reference data** are looked up in `--data-dir`, else `$GENESIS_DATA_DIR`, else
  `~/.cache/genesis-gwas`.

The files written by each command are specified in [Output files](outputs.md).

## All commands and options

The remainder of this page is generated from the program's argument parser and is
therefore always in step with the installed version.

```{argparse}
:module: genesis_gwas.cli.main
:func: build_parser
:prog: genesis
```
