# Biowulf job scripts for GENESIS

Ready-to-edit scripts for running GENESIS on the NIH Biowulf cluster. The full guide,
with the reasoning behind every number in these files, is [BIOWULF.md](../../BIOWULF.md)
(also on the [documentation site](https://jeyabbalas.github.io/GENESIS/biowulf/)).

| File | What it does | How it runs |
|---|---|---|
| `setup_env.sh` | creates the conda environment (`CPU_ONLY=1` for a build without PyTorch) | `bash setup_env.sh` inside an `sinteractive` session |
| `fetch_refdata.sh` | downloads and verifies the LD reference into `/data` | `sbatch fetch_refdata.sh default,nullz` |
| `gpu_check.sh` | test of the GPU path (`nvidia-smi`, PyTorch, `genesis doctor --benchmark`, demo fit) | `sbatch gpu_check.sh` |
| `benchmark.sh` | times one pass over a genome-wide data set for several thread counts and for the GPU(s) of the job (`genesis doctor --benchmark`) | `sbatch benchmark.sh` |
| `fit_cpu.sh` | one analysis on CPUs | `sbatch fit_cpu.sh SUMSTATS OUT_PREFIX [genesis fit options]` |
| `fit_gpu.sh` | one analysis on one GPU | `sbatch fit_gpu.sh SUMSTATS OUT_PREFIX [genesis fit options]` |
| `fit_multigpu.sh` | one analysis on two GPUs of one node | `sbatch fit_multigpu.sh SUMSTATS OUT_PREFIX [genesis fit options]` |
| `ld_build.sh` | builds an LD reference for another population from PLINK files | `sbatch ld_build.sh BFILE OUT_DIR NAME POPULATION SNPLIST` |
| `project_swarm.py` | writes one swarm file for every trait of every ancestry of a project, skipping finished fits | `python project_swarm.py PROJECT -- [genesis fit options]`, then the `swarm` command it prints |
| `collect_results.py` | gathers all results into two tables and lists unfinished fits | `python collect_results.py PROJECT/results --out PROJECT/collected/all` |
| `subset_to_reference.py` | shrinks a genome-wide summary-statistics file to the SNPs of the LD reference | `python subset_to_reference.py IN OUT` |

Conventions shared by all of them:

- **Nothing here submits a job.** `sbatch` and `swarm` are commands you type on the
  Biowulf login node; every job script ends with a comment showing the exact command.
- **Environment.** The scripts activate the conda environment `genesis` under
  `/data/$USER/conda`. For any other setup, pass the activation line when submitting, for
  example
  `GENESIS_ACTIVATE='module load python/3.12 && source /data/$USER/venvs/genesis/bin/activate' sbatch fit_cpu.sh ...`.
- **Reference data.** `GENESIS_DATA_DIR` defaults to `/data/$USER/genesis-ref`; export
  another value before submitting to use a shared copy.
- **Threads.** GENESIS reads `$SLURM_CPUS_PER_TASK`; no script hard-codes a thread count.
- **Sizes are starting points.** The scripts were tested on a laptop with Slurm's
  environment simulated; nothing has been run on Biowulf by the developers, and no
  Biowulf run time has been measured. Run `benchmark.sh` and one trait, read
  `jobhist JOBID`, then set `--time` and `--mem` for the rest.

Copy the directory next to your project (for example `git clone` the repository into
`/data/$USER`) and edit the `#SBATCH` lines to taste; command-line flags to `sbatch`
override them without editing.
