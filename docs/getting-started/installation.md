# Installation

GENESIS is a pure-Python package: installing it needs no compiler, no R, and nothing
beyond `pip`. It runs on Linux and macOS with Python 3.10 or newer.

| You want to ... | Install |
|---|---|
| analyse data on CPUs (laptop, workstation, cluster) | the base package |
| also use NVIDIA GPUs | the base package plus PyTorch with CUDA (`[gpu]` extra) |
| run it on NIH Biowulf | see the [Biowulf guide](../biowulf/index.md) |
| develop or build the documentation | an editable clone with the `dev` / `docs` extras |

## Install

::::{tab-set}

:::{tab-item} pip
```bash
python -m pip install "genesis-gwas @ git+https://github.com/jeyabbalas/GENESIS.git"
```
:::

:::{tab-item} uv
```bash
uv venv --python 3.12
uv pip install "genesis-gwas @ git+https://github.com/jeyabbalas/GENESIS.git"
```
:::

:::{tab-item} conda / mamba
```bash
mamba create -n genesis -c conda-forge python=3.12 numpy scipy pandas matplotlib threadpoolctl pip
mamba activate genesis
python -m pip install "genesis-gwas @ git+https://github.com/jeyabbalas/GENESIS.git"
```
The scientific dependencies come from conda-forge; `pip` then only adds GENESIS
itself.
:::

:::{tab-item} from a clone (development)
```bash
git clone https://github.com/jeyabbalas/GENESIS.git
cd GENESIS
python -m pip install -e ".[dev,docs]"
```
:::

::::

This installs the `genesis` command (also available as `genesis-gwas`) and the Python
package `genesis_gwas`. The dependencies are NumPy, SciPy, pandas, Matplotlib and
threadpoolctl.

:::{note}
The package is not on PyPI; install it from GitHub as shown. The name `genesis` on PyPI
belongs to an unrelated project.
:::

## GPU support (optional)

GPU execution uses [PyTorch](https://pytorch.org). Install a CUDA build of PyTorch that
matches your system, then GENESIS:

```bash
# example: CUDA 12.8 wheels from the PyTorch index; choose the tag that matches your
# driver from https://pytorch.org/get-started/locally/
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
python -m pip install "genesis-gwas @ git+https://github.com/jeyabbalas/GENESIS.git"
```

or let pip choose the default PyTorch build with the `gpu` extra:

```bash
python -m pip install "genesis-gwas[gpu] @ git+https://github.com/jeyabbalas/GENESIS.git"
```

Then check that the GPU is visible to PyTorch and agrees with the CPU:

```bash
genesis doctor
```

Things to know:

- Without PyTorch, or without a visible CUDA device, GENESIS runs on the CPU. The CPU
  backend (NumPy) is the reference implementation and is not a fallback of lower
  quality.
- All computation is in double precision. NVIDIA GPUs support this natively.
- **Apple-silicon GPUs (MPS) are not supported**, because the MPS backend has no 64-bit
  floating point; `--device mps` is rejected with an explanation. Use `--device cpu` on
  a Mac.
- `--device auto` (the default) uses every CUDA device visible to the process — on a
  Slurm cluster, the GPUs granted to the job — and otherwise the CPU.

## Verify the installation

`genesis doctor` prints the versions, the CPUs and GPUs available to the process, and
runs the same likelihood, score and information computation on every available
backend, comparing each with the single-threaded NumPy result:

```console
$ genesis doctor
genesis-gwas 0.2.0
  python 3.12.13  numpy 2.5.5  scipy 1.18.1  torch 2.15.0
  platform: macOS-27.0.1-arm64-arm-64bit
  CPUs available to this process: 18 (machine: 18)
  PyTorch CUDA build: none (CPU-only build); CUDA_VISIBLE_DEVICES=None
  GPUs: none visible

Self-test (three-component likelihood, scores and information on the packaged chr22 data):
  ok    numpy / cpu, 1 thread                        max rel. diff 0.0e+00   0.21 s
  ok    numpy / cpu, 18 thread(s)                    max rel. diff 0.0e+00   0.06 s
  ok    torch / cpu                                  max rel. diff 2.3e-14   0.07 s

All backends agree.
```

`genesis demo` runs a complete analysis (QC, both models, model selection) on the
chromosome-22 example that ships inside the package; it needs no download and takes ten to twenty seconds on a laptop:

```console
$ genesis demo
GENESIS fit: M2 (two-component model)
  GWAS markers analysed (K)     : 13,230
  Reference-panel SNPs (M)      : 15,633
  Mean sample size              : 130,795.0
  Optimisation                  : converged after 52 EM iteration(s), c0 = 20; 4 Newton step(s)
  Composite log-likelihood      : 55,825.9412
  Model selection               : BIC = -111,305.981   AIC = -111,600.220   d_s = 25.831

  Parameter           Estimate              SE
  pic             5.779742e-02      9.8183e-03
  sigmasq         6.326638e-06      1.3709e-06
  a               2.657100e-07      5.5298e-07

  Derived quantity                                              Estimate          SE
  Number of susceptibility SNPs                                   903.55       153.5
  Total heritability                                           0.0057164    0.001056
  Sibling relative risk                                           1.0029   0.0005296

  Convergence check             : estimate is within 1.91e-07 SE of the likelihood stationary point

  Warnings:
    - Truncation still matters at the ceiling c0 = 20: on average 4.4e-05 of the prior mass is discarded (tolerance 1e-06). Estimates, pi_c in particular, may be limited by the truncation; check their sensitivity with a larger c0_max.

[...]

Model comparison by BIC (lower is better)
  Model               logLik       d_s               AIC               BIC       delta
  M2              55,825.941    25.831      -111,600.220      -111,305.981       0.000  <- selected
  M3              55,827.159    34.955      -111,584.407      -111,186.236     119.745  (not identified)

Note: this is a smoke test. One chromosome carries too little information to separate two clusters of effect sizes, so the three-component fit is expected to be flagged as not identified and M2 to be selected; with M = 15,633 the numbers refer to chromosome 22 alone.

Results: /tmp/genesis-demo-nvpgf53w/demo.M2.json, /tmp/genesis-demo-nvpgf53w/demo.M3.json
```

(The report of the three-component model, omitted here, has the same layout.)

On a GPU node, `genesis doctor` additionally lists each GPU and reports its maximum
relative difference from the CPU result; anything other than `ok` on every line means
the installation should not be used.

## Reference data

Analyses need an LD reference: for every SNP of a reference panel, the number of SNPs
it tags, its LD score and its tagging neighbourhood. Three references, all computed
from 1000 Genomes Phase 3 genotypes, are available for download:

| Reference | Alias | Population | Panel | SNPs | Source |
|---|---|---|---:|---:|---|
| `eur_1kg_hm3` (the default) | `EUR` | European | 489 | 1,070,777 | the original GENESIS package |
| `eas_1kg_hm3` | `EAS` | East Asian | 504 | 953,834 | the authors' GENESISasian package |
| `afr_1kg_hm3` | `AFR` | African ancestry | 661 | 1,025,309 | built with this software |

The reference an analysis names (`--reference`; European if none is named) is
downloaded automatically the first time it is used and cached. To fetch one explicitly:

```bash
genesis data fetch                       # European, default configuration: 32 MB
genesis data fetch AFR                   # African ancestry: 30 MB
genesis data fetch EAS                   # East Asian: 27 MB
genesis data list --summary              # what is installed, and where
```

:::{dropdown} Output of `genesis data list --summary` after `genesis data fetch`
```text
Data directory: ~/.cache/genesis-gwas

eur_1kg_hm3 v1.0: 1000 Genomes Phase 3, LDSC EUR subset (n=489), HapMap3 SNPs with MAF>=0.05
  population EUR, 1,070,777 SNPs, 489 samples, GRCh37
  installed 2/22 files (33 of 504 MB) in ~/.cache/genesis-gwas/eur_1kg_hm3/1.0
  LD configurations: w500kb_r0.05, w500kb_r0.1, w500kb_r0.2, w1000kb_r0.05, w1000kb_r0.1, w1000kb_r0.2, w2000kb_r0.05, w2000kb_r0.1, w2000kb_r0.2

eas_1kg_hm3 v1.0: 1000 Genomes Phase 3, EAS (n=504), HapMap3 SNPs with MAF>=0.05
  population EAS, 953,834 SNPs, 504 samples, GRCh37
  installed 0/21 files (0 of 411 MB) in ~/.cache/genesis-gwas/eas_1kg_hm3/1.0
  LD configurations: w500kb_r0.05, w500kb_r0.1, w500kb_r0.2, w1000kb_r0.05, w1000kb_r0.1, w1000kb_r0.2, w2000kb_r0.05, w2000kb_r0.1, w2000kb_r0.2

afr_1kg_hm3 v1.0: 1000 Genomes Phase 3, AFR (n=661), HapMap3 SNPs outside the MHC with MAF>=0.05
  population AFR, 1,025,309 SNPs, 661 samples, GRCh37
  installed 0/21 files (0 of 439 MB) in ~/.cache/genesis-gwas/afr_1kg_hm3/1.0
  LD configurations: w500kb_r0.05, w500kb_r0.1, w500kb_r0.2, w1000kb_r0.05, w1000kb_r0.1, w1000kb_r0.2, w2000kb_r0.05, w2000kb_r0.1, w2000kb_r0.2

Packaged (no download needed): mini_chr22
```
:::

Without `--summary` the listing also shows every file of each reference and whether it
is installed. Only what an analysis needs is downloaded:

| Component | `--components` | EUR | EAS | AFR | Needed for |
|---|---|---:|---:|---:|---|
| SNP table + LD configuration 1 Mb / $r^2 \ge 0.1$ | `default` | 32 MB | 27 MB | 30 MB | fitting with the paper's primary settings |
| all nine LD configurations (0.5 / 1 / 2 Mb × 0.05 / 0.1 / 0.2) | `ld` | 196 MB | 153 MB | 183 MB | sensitivity analyses (`--ld-window-kb`, `--ld-r2`) |
| one other configuration | e.g. `w2000kb_r0.05` | 17–27 MB | 14–20 MB | 15–26 MB | |
| LD-correlated null statistics (100 replicates) | `nullz` | 214 MB | 191 MB | 205 MB | QQ-plot envelopes, `future_gc` |
| pairwise $r^2$ | `pairld` | 70 MB | 57 MB | 41 MB | `future_gc`, `simulate_ld_sumstats` |
| GIANT height example | `examples` | 13 MB | – | – | the tutorial |
| everything | `all` | 504 MB | 411 MB | 439 MB | |

Every download is verified against a SHA-256 checksum that ships inside the package,
interrupted downloads resume, and a file appears under its final name only once it is
complete and verified — so a partial or corrupted download cannot be used by accident.
Components that an analysis needs but that are missing (for example the null statistics
for `--qq`) are fetched on demand. The files are release assets of the GitHub
repository (tags `refdata-v1` for the European reference and `refdata-v2` for the
others); set `GENESIS_DATA_URL` to download them from a mirror
that serves the same file names.

### Where the data go

By default: `~/.cache/genesis-gwas` (or `$XDG_CACHE_HOME/genesis-gwas`). Set
`GENESIS_DATA_DIR` to choose another place — on clusters with a small home quota, a
directory on project storage:

```bash
export GENESIS_DATA_DIR=/data/$USER/genesis-data
genesis data fetch
```

`genesis data fetch --dest DIR` and the `--data-dir DIR` option of `genesis fit` do the
same for one command.

### Offline and restricted networks

- **Behind a proxy:** the standard `HTTPS_PROXY` / `https_proxy` variables are honoured.
- **No internet on the machine that runs the analysis:** fetch on a machine that has
  access, copy the directory, and point GENESIS at it:

  ```bash
  # on a machine with internet
  genesis data fetch --components all --dest ./genesis-data
  rsync -a ./genesis-data/ cluster:/data/me/genesis-data/

  # on the cluster
  export GENESIS_DATA_DIR=/data/me/genesis-data
  genesis fit trait.tsv.gz --out results/trait --no-download
  ```

  With `--no-download`, any missing reference file — including the optional pieces
  that are otherwise fetched the first time they are used — is an immediate, explicit
  error instead of a download attempt. A bundle directory can also be passed
  directly: `--reference /data/me/genesis-data/eur_1kg_hm3/1.0`. For offline use fetch
  every component you will need (`nullz` for QQ envelopes, `pairld` for
  `future_gc`), or simply `all`.
- **Checking a copied bundle:** `genesis data verify DIR` re-hashes every file.

### Other populations

For a population other than the three above, build a reference from the genotypes of
an ancestry-matched panel: `genesis ld from-vcf` converts VCF files (for example those
of the 1000 Genomes Project) to PLINK files, and `genesis ld build` computes the
reference from them (a genome-wide HapMap3 build takes one to two minutes). Pass the
resulting directory with `--reference`. The tutorial chapter on
[other ancestries](../tutorial/other-ancestries.md) goes through every step; the
HapMap3 SNP list it uses is the file `hapmap3_noMHC.snplist.gz` of the `refdata-v2`
release.

## Upgrading and uninstalling

```bash
python -m pip install --upgrade --force-reinstall --no-deps \
    "genesis-gwas @ git+https://github.com/jeyabbalas/GENESIS.git"   # upgrade
python -m pip uninstall genesis-gwas                                  # uninstall
rm -rf ~/.cache/genesis-gwas                                          # remove cached reference data
```

Result files record the version that produced them (`provenance.version`).
