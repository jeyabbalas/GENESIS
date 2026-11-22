#!/bin/bash
#SBATCH --job-name=genesis-gpu-check
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpua100|gpuv100x"
#SBATCH --cpus-per-task=4
#SBATCH --mem=8g
#SBATCH --time=00:15:00
# gpu_check.sh -- short test of the GPU path. Run it once after installing, once per GPU
# type you intend to use, and again whenever PyTorch or the environment changes. It prints
# the GPU the job got, checks that PyTorch sees it, and runs `genesis doctor --benchmark`,
# which (a) verifies that the GPU reproduces the CPU reference numbers (maximum relative
# difference below 1e-8) and (b) times one pass over 1,000,000 synthetic SNPs on the CPUs
# and on the GPU, so you can see whether the GPU is worth queueing for. It ends by fitting
# the packaged chromosome-22 example on the GPU.
set -euo pipefail

set +u    # conda's activate scripts are not written for 'set -u'
# Default: the conda environment from BIOWULF.md, section 2. For another setup submit with
#   GENESIS_ACTIVATE='<shell line that puts genesis on PATH>' sbatch gpu_check.sh
eval "${GENESIS_ACTIVATE:-source /data/$USER/conda/etc/profile.d/conda.sh && conda activate genesis}"
set -u
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1    # GENESIS threads itself (see fit_cpu.sh)

echo "host=$(hostname -s)  job=${SLURM_JOB_ID:-none}  cpus=${SLURM_CPUS_PER_TASK:-unset}  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi
python -c 'import torch; print("torch", torch.__version__, "| CUDA build:", torch.version.cuda, "| available:", torch.cuda.is_available(), "| devices:", torch.cuda.device_count())'

genesis --version
genesis doctor --benchmark           # exit status 1 if any backend fails or disagrees
# The demo below warns about the truncation ceiling and an unidentified three-component
# model: both are expected for a single chromosome and are not problems.
genesis demo --device cuda           # fails if PyTorch cannot use the GPU

# Submit (on Biowulf, login node):
#   sbatch gpu_check.sh
# then read slurm-JOBID.out in the directory you submitted from, and keep it. Expect
#   ok    torch / cuda:0 (<GPU model>)    max rel. diff <a number below 1e-8>
# in the self-test, and in the benchmark one "ms per pass" line for the CPUs and one per GPU.
# To test one GPU type, override the constraint:
#   sbatch --constraint=gpua100 gpu_check.sh
#   sbatch --constraint=gpuv100x gpu_check.sh
#   sbatch --constraint=gpul40 gpu_check.sh        # see BIOWULF.md, section 4.3, before relying on L40s
