#!/bin/bash
#SBATCH --job-name=genesis-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpua100|gpuv100x"
#SBATCH --cpus-per-task=8
#SBATCH --mem=16g
#SBATCH --time=00:15:00
# fit_gpu.sh -- one GENESIS analysis on one GPU.
#
#   sbatch fit_gpu.sh SUMSTATS OUT_PREFIX [genesis fit options]
#
# The GPU path has NOT been run on Biowulf by the developers (BIOWULF.md, section 9):
# submit gpu_check.sh once before the first real job, and compare one fit with fit_cpu.sh
# before choosing GPUs for production -- a complete analysis took about a minute on the
# CPUs of a laptop. Fifteen minutes is a generous first walltime; calibrate it from
# `jobhist JOBID`. The request takes an A100 or a V100x, whichever is free first. 8 CPUs
# is within the CPUs-per-GPU cap of both (A100 16, V100x 18). GPU memory is not a
# constraint: the engine sizes its chunks from the free device memory.
set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "usage: sbatch fit_gpu.sh SUMSTATS OUT_PREFIX [genesis fit options]" >&2
    exit 2
fi
SUMSTATS=$1
OUT=$2
shift 2

set +u    # conda's activate scripts are not written for 'set -u'
# Default: the conda environment from BIOWULF.md, section 2. For another setup submit with
#   GENESIS_ACTIVATE='<shell line that puts genesis on PATH>' sbatch fit_gpu.sh ...
eval "${GENESIS_ACTIVATE:-source /data/$USER/conda/etc/profile.d/conda.sh && conda activate genesis}"
set -u
export GENESIS_DATA_DIR=${GENESIS_DATA_DIR:-/data/$USER/genesis-ref}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1    # GENESIS threads itself (see fit_cpu.sh)

echo "host=$(hostname -s)  job=${SLURM_JOB_ID:-none}  cpus=${SLURM_CPUS_PER_TASK:-unset}  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "sumstats=$SUMSTATS  out=$OUT  GENESIS_DATA_DIR=$GENESIS_DATA_DIR"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv
genesis --version

# --device cuda fails with a clear message if PyTorch cannot see a GPU, instead of
# silently running on the CPUs of a GPU node.
genesis fit "$SUMSTATS" --out "$OUT" --device cuda "$@"

# Submit (on Biowulf, login node), for example:
#   sbatch fit_gpu.sh /data/$USER/proj/sumstats/trait.sumstats.tsv.gz /data/$USER/proj/results/trait
#   sbatch --constraint=gpua100 fit_gpu.sh ...        # one specific GPU type
#   (if sbatch objects to the quoted constraint line above, delete it and pass
#    --constraint="gpua100|gpuv100x" on the sbatch command line instead)
# Check afterwards that the GPU was used:
#   dashboard_cli jobs --jobid JOBID --fields jobid,state,elapsed_time,gpus,gpu_util,cpu_util,mem_util
