#!/bin/bash
# setup_env.sh -- create the conda environment that runs GENESIS on Biowulf.
#
# Run it INSIDE an interactive session, never on the login node:
#
#   # on Biowulf (login node)
#   sinteractive --cpus-per-task=4 --mem=20g --gres=lscratch:20 --time=2:00:00
#   # on the compute node (sinteractive session)
#   bash setup_env.sh                 # GPU-capable: CUDA build of PyTorch + genesis-gwas[gpu]
#   CPU_ONLY=1 bash setup_env.sh      # CPU only: no PyTorch (smaller, faster to build)
#
# It needs a personal Miniforge under /data (BIOWULF.md, section 2.2). Settings, all
# optional, are read from the environment:
#   ENV_NAME      name of the environment                       [genesis]
#   CONDA_ROOT    where Miniforge lives                         [/data/$USER/conda]
#   GENESIS_SPEC  what pip installs [git+https://github.com/jeyabbalas/GENESIS.git]. Pin a
#                 tag or commit for reproducibility (...GENESIS.git@<tag-or-commit>), or
#                 install from a clone on disk (file:///data/$USER/GENESIS)
#   CUDA_VERSION  CUDA series conda should assume when it resolves PyTorch [12.4]
set -euo pipefail

if [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "Run this inside an sinteractive session (or a batch job), not on the login node." >&2
    exit 1
fi

ENV_NAME=${ENV_NAME:-genesis}
CONDA_ROOT=${CONDA_ROOT:-/data/$USER/conda}
GENESIS_SPEC=${GENESIS_SPEC:-git+https://github.com/jeyabbalas/GENESIS.git}
CUDA_VERSION=${CUDA_VERSION:-12.4}

if [ -d "/lscratch/$SLURM_JOB_ID" ]; then
    export TMPDIR=/lscratch/$SLURM_JOB_ID      # build and download temp files stay off /tmp
fi
if [ ! -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]; then
    echo "No conda installation found at $CONDA_ROOT." >&2
    echo "Install Miniforge first:  module load mamba_install && mamba_install   (BIOWULF.md, section 2.2)" >&2
    exit 1
fi

set +u    # conda's shell functions are not written for 'set -u'
# shellcheck disable=SC1091
source "$CONDA_ROOT/etc/profile.d/conda.sh"
set -u
if [ -d "$CONDA_ROOT/envs/$ENV_NAME" ]; then
    echo "Environment '$ENV_NAME' already exists in $CONDA_ROOT/envs." >&2
    echo "Remove it first (conda env remove -n $ENV_NAME) or choose another ENV_NAME." >&2
    exit 1
fi
if command -v mamba >/dev/null 2>&1; then CREATE=mamba; else CREATE=conda; fi

echo "== creating environment '$ENV_NAME' in $CONDA_ROOT/envs (TMPDIR=${TMPDIR:-unset})"
BASE_PKGS=(python=3.12 numpy scipy pandas matplotlib threadpoolctl pip git)
if [ -n "${CPU_ONLY:-}" ]; then
    "$CREATE" create -y -n "$ENV_NAME" -c conda-forge "${BASE_PKGS[@]}"
    EXTRA=""
else
    # A CPU session has no GPU driver, so tell conda which CUDA to assume; otherwise it
    # can silently pick a CPU-only PyTorch build.
    CONDA_OVERRIDE_CUDA="$CUDA_VERSION" "$CREATE" create -y -n "$ENV_NAME" -c conda-forge \
        "${BASE_PKGS[@]}" 'pytorch=*=cuda*'
    EXTRA="[gpu]"
fi

set +u
conda activate "$ENV_NAME"
set -u

echo "== installing genesis-gwas$EXTRA from $GENESIS_SPEC"
# --no-cache-dir: pip's cache would otherwise land in ~/.cache/pip, inside the 16 GB /home quota
python -m pip install --no-cache-dir "genesis-gwas$EXTRA @ $GENESIS_SPEC"
python -m pip check || echo "pip check reported problems (see above)" >&2

python -c "import matplotlib.font_manager"     # build matplotlib's font cache once, not in every job
if [ -z "${CPU_ONLY:-}" ]; then
    python -c 'import torch; print("torch", torch.__version__, "| CUDA build:", torch.version.cuda)'
    if python -c 'import sys, torch; sys.exit(0 if torch.version.cuda else 1)'; then
        echo "PyTorch has CUDA support (it can only be exercised in a GPU job: sbatch gpu_check.sh)."
    else
        echo "WARNING: PyTorch was installed WITHOUT CUDA support; GPU jobs will not work." >&2
        echo "Remove the environment and re-run, or run this script inside a GPU session." >&2
    fi
fi

echo "== self-test"
genesis --version
genesis doctor
echo
echo "Done. Activate the environment in job scripts with:"
echo "  source $CONDA_ROOT/etc/profile.d/conda.sh && conda activate $ENV_NAME"
