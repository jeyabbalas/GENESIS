#!/bin/bash
#SBATCH --job-name=genesis-fit
#SBATCH --cpus-per-task=16
#SBATCH --mem=8g
#SBATCH --time=00:30:00
# fit_cpu.sh -- one GENESIS analysis on CPUs: QC, two- and three-component fits, model
# selection by BIC, standard errors. Everything after the two required arguments is
# passed to `genesis fit` unchanged.
#
#   sbatch fit_cpu.sh SUMSTATS OUT_PREFIX [genesis fit options]
#
# Sizing (BIOWULF.md, section 4): 8-16 CPUs; 8 GB is enough for inputs of up to about
# 5 million rows, use --mem=16g for 10 million (or shrink the file first with
# subset_to_reference.py). The run needs no scratch space. Thirty minutes is a generous
# first walltime for one trait (a complete analysis took about a minute on the laptop
# the software was developed on; Biowulf has not been timed): calibrate --time and --mem
# from the first run with `jobhist JOBID`.
set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "usage: sbatch fit_cpu.sh SUMSTATS OUT_PREFIX [genesis fit options]" >&2
    exit 2
fi
SUMSTATS=$1
OUT=$2
shift 2

set +u    # conda's activate scripts are not written for 'set -u'
# Default: the conda environment from BIOWULF.md, section 2. For another setup submit with
#   GENESIS_ACTIVATE='<shell line that puts genesis on PATH>' sbatch fit_cpu.sh ...
eval "${GENESIS_ACTIVATE:-source /data/$USER/conda/etc/profile.d/conda.sh && conda activate genesis}"
set -u
export GENESIS_DATA_DIR=${GENESIS_DATA_DIR:-/data/$USER/genesis-ref}
# GENESIS runs its own pool of worker threads, one per allocated CPU ($SLURM_CPUS_PER_TASK).
# The numerical libraries underneath stay at one thread each, so the two levels cannot multiply.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

echo "host=$(hostname -s)  job=${SLURM_JOB_ID:-none}  cpus=${SLURM_CPUS_PER_TASK:-unset}  mem=${SLURM_MEM_PER_NODE:-unset}MB"
echo "sumstats=$SUMSTATS  out=$OUT  GENESIS_DATA_DIR=$GENESIS_DATA_DIR"
genesis --version

genesis fit "$SUMSTATS" --out "$OUT" --device cpu "$@"

# Submit (on Biowulf, login node), for example:
#   sbatch fit_cpu.sh /data/$USER/proj/sumstats/trait.sumstats.tsv.gz /data/$USER/proj/results/trait
#   sbatch fit_cpu.sh /data/$USER/proj/sumstats/bc.tsv.gz /data/$USER/proj/results/bc \
#          --n-cases NCASE --n-controls NCONTROL --population-prevalence 0.12 --sample-prevalence 0.45 --qq
#   sbatch --partition=quick,norm fit_cpu.sh ...     # runs under 4 h may also use the quick partition
# Outputs: OUT_PREFIX.{summary.txt,estimates.tsv,M2.json,M3.json,best.json,log}; Slurm's own
# log is slurm-JOBID.out in the directory you submitted from.
