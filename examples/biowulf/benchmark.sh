#!/bin/bash
#SBATCH --job-name=genesis-bench
#SBATCH --cpus-per-task=32
#SBATCH --mem=8g
#SBATCH --time=00:20:00
# benchmark.sh -- measure GENESIS on the node this job lands on: how the time for one pass
# over a genome-wide data set changes with the number of threads, and how the CPUs compare
# with the GPU(s) if the job has any. It runs `genesis doctor --benchmark`, which times the
# three-component model (c0 = 10) on 1,000,000 synthetic SNPs, so it needs no data.
# Use it to choose --cpus-per-task, to decide whether a GPU is worth queueing for, and to
# set --time. It answers the questions the developers could not: Biowulf was never timed.
#
#   sbatch benchmark.sh
#
# Settings (environment variables):
#   THREADS="4 8 16"   thread counts to try [4 8 16 32 64, below the allocation]
#   SNPS=1000000       number of synthetic SNPs
set -euo pipefail

set +u    # conda's activate scripts are not written for 'set -u'
eval "${GENESIS_ACTIVATE:-source /data/$USER/conda/etc/profile.d/conda.sh && conda activate genesis}"
set -u
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1    # GENESIS threads itself (see fit_cpu.sh)
CPUS=${SLURM_CPUS_PER_TASK:-2}
THREADS=${THREADS:-4 8 16 32 64}
SNPS=${SNPS:-1000000}

echo "host=$(hostname -s)  job=${SLURM_JOB_ID:-none}  cpus=$CPUS  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
grep -m1 'model name' /proc/cpuinfo 2>/dev/null || true
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
fi

echo
echo "--- CPU threads: one pass over $SNPS synthetic SNPs, three-component model, c0 = 10"
for t in $THREADS; do
    if [ "$t" -lt "$CPUS" ]; then
        # GENESIS_NUM_THREADS overrides the allocation for this one command
        GENESIS_NUM_THREADS=$t genesis doctor --benchmark --benchmark-snps "$SNPS" --quiet \
            | grep 'numpy / cpu' | grep 'per pass' || echo "  $t threads: failed"
    fi
done

echo
echo "--- every backend and device of this job, with the full allocation ($CPUS CPUs)"
genesis doctor --benchmark --benchmark-snps "$SNPS" --quiet      # exit status 1 if a backend fails or disagrees

echo
echo "Reading the numbers: scale 'ms per pass' by your number of SNPs after QC (in millions),"
echo "multiply by about 150 for a default three-component fit, and add a minute for reading,"
echo "QC, the two-component fit, standard errors and output (--qq and --posterior add more)."
echo "Past the thread count where the time per pass stops falling, extra CPUs are wasted."

# Submit (on Biowulf, login node), once on CPUs and once with a GPU, and keep both outputs:
#   sbatch benchmark.sh                                   # 32 CPUs of whatever node type is free
#   sbatch --constraint=e7543 benchmark.sh                # one specific CPU type (freen lists them)
#   sbatch --partition=gpu --gres=gpu:a100:1 --cpus-per-task=16 --mem=16g benchmark.sh
# The table is in slurm-JOBID.out.
