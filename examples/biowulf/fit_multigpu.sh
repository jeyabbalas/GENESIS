#!/bin/bash
#SBATCH --job-name=genesis-multigpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=24g
#SBATCH --time=00:15:00
# fit_multigpu.sh -- one GENESIS analysis spread over several GPUs of one node.
#
#   sbatch fit_multigpu.sh SUMSTATS OUT_PREFIX [genesis fit options]
#
# `--device cuda` uses every GPU granted to the job: the SNPs are split into one
# contiguous shard per GPU and the sufficient statistics are added up on the host.
# Read BIOWULF.md, section 5.2, before using this. One fit of about a million SNPs is
# expected to be a small problem for a single GPU, and reading, QC and output do not use
# the GPU at all, so several GPUs per fit can only pay off for much larger SNP sets or a
# large truncation level (--c0). This path has not been run on real hardware by the
# developers: compare it with one GPU first (`genesis doctor --benchmark` prints both).
# To use many GPUs productively, run many fits side by side (genesis swarm --gpu).
# The request below asks for 2 of the 4 A100s of one node; for V100x use gpu:v100x:2.
set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "usage: sbatch fit_multigpu.sh SUMSTATS OUT_PREFIX [genesis fit options]" >&2
    exit 2
fi
SUMSTATS=$1
OUT=$2
shift 2

set +u    # conda's activate scripts are not written for 'set -u'
eval "${GENESIS_ACTIVATE:-source /data/$USER/conda/etc/profile.d/conda.sh && conda activate genesis}"
set -u
export GENESIS_DATA_DIR=${GENESIS_DATA_DIR:-/data/$USER/genesis-ref}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1    # GENESIS threads itself (see fit_cpu.sh)

echo "host=$(hostname -s)  job=${SLURM_JOB_ID:-none}  cpus=${SLURM_CPUS_PER_TASK:-unset}  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "sumstats=$SUMSTATS  out=$OUT  GENESIS_DATA_DIR=$GENESIS_DATA_DIR"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv
python -c 'import torch; print("GPUs visible to PyTorch:", torch.cuda.device_count())'
genesis --version

genesis fit "$SUMSTATS" --out "$OUT" --device cuda "$@"

# Submit (on Biowulf, login node):
#   sbatch fit_multigpu.sh /data/$USER/proj/sumstats/trait.sumstats.tsv.gz /data/$USER/proj/results/trait
# The log line "Compute: torch:cuda:0 (...), torch:cuda:1 (...)" confirms that both GPUs are in use;
# afterwards compare gpu_util for 1 and 2 GPUs before making this your default:
#   dashboard_cli jobs --jobid JOBID --fields jobid,state,elapsed_time,gpus,gpu_avg,gpu_util
