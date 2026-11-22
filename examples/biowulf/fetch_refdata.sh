#!/bin/bash
#SBATCH --job-name=genesis-refdata
#SBATCH --cpus-per-task=2
#SBATCH --mem=4g
#SBATCH --time=00:30:00
# fetch_refdata.sh -- download the LD reference once, into /data, before any fit runs.
# Compute nodes reach the internet through the HPC proxy (https works), so this runs as a
# batch job or inside an sinteractive session; do not run it on the login node.
#
#   sbatch fetch_refdata.sh                      # default: SNP table + 1 Mb / r2>=0.1 (32 MB)
#   sbatch fetch_refdata.sh default,nullz        # + null Z-scores for QQ envelopes (246 MB)
#   sbatch fetch_refdata.sh all                  # everything (504 MB)
set -euo pipefail

COMPONENTS=${1:-default}
DEST=${GENESIS_DATA_DIR:-/data/$USER/genesis-ref}     # a group directory works too: /data/GROUP/genesis-ref

set +u    # conda's activate scripts are not written for 'set -u'
eval "${GENESIS_ACTIVATE:-source /data/$USER/conda/etc/profile.d/conda.sh && conda activate genesis}"
set -u

umask 027                                             # group-readable, never world-readable
echo "host=$(hostname -s)  job=${SLURM_JOB_ID:-none}  dest=$DEST  components=$COMPONENTS"
env | grep -i '_proxy=' || echo "note: no proxy variables are set"

# the command logs to stderr and prints the bundle directory on stdout
BUNDLE=$(genesis data fetch eur_1kg_hm3 --components "$COMPONENTS" --dest "$DEST")
genesis data verify "$BUNDLE"                         # re-hashes every file ('missing' = not fetched)
du -sh "$DEST"
echo "Add to your job scripts:  export GENESIS_DATA_DIR=$DEST"

# Submit (on Biowulf, login node):
#   sbatch fetch_refdata.sh default,nullz
