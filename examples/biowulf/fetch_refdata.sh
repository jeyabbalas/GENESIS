#!/bin/bash
#SBATCH --job-name=genesis-refdata
#SBATCH --cpus-per-task=2
#SBATCH --mem=4g
#SBATCH --time=00:30:00
# fetch_refdata.sh -- download the LD reference(s) once, into /data, before any fit runs.
# Compute nodes reach the internet through the HPC proxy (https works), so this runs as a
# batch job or inside an sinteractive session; do not run it on the login node.
#
#   sbatch fetch_refdata.sh [COMPONENTS] [REFERENCE ...]
#
#   sbatch fetch_refdata.sh                      # European; SNP table + 1 Mb / r2>=0.1 (32 MB)
#   sbatch fetch_refdata.sh default,nullz        # + null Z-scores for QQ envelopes (247 MB)
#   sbatch fetch_refdata.sh all                  # everything (504 MB)
#   sbatch fetch_refdata.sh default,nullz eur_1kg_hm3 eas_1kg_hm3 afr_1kg_hm3
#                                                # the three downloadable references
#                                                # (European, East Asian, African ancestry)
set -euo pipefail

COMPONENTS=${1:-default}
shift || true
if [ "$#" -gt 0 ]; then REFERENCES=("$@"); else REFERENCES=(eur_1kg_hm3); fi
DEST=${GENESIS_DATA_DIR:-/data/$USER/genesis-ref}     # a group directory works too: /data/GROUP/genesis-ref

set +u    # conda's activate scripts are not written for 'set -u'
eval "${GENESIS_ACTIVATE:-source /data/$USER/conda/etc/profile.d/conda.sh && conda activate genesis}"
set -u

umask 027                                             # group-readable, never world-readable
echo "host=$(hostname -s)  job=${SLURM_JOB_ID:-none}  dest=$DEST  components=$COMPONENTS  references=${REFERENCES[*]}"
env | grep -i '_proxy=' || echo "note: no proxy variables are set"

# the command logs to stderr and prints the bundle directory on stdout
for REFERENCE in "${REFERENCES[@]}"; do
    BUNDLE=$(genesis data fetch "$REFERENCE" --components "$COMPONENTS" --dest "$DEST")
    genesis data verify "$BUNDLE"                     # re-hashes every file ('missing' = not fetched)
done
du -sh "$DEST"
echo "Add to your job scripts:  export GENESIS_DATA_DIR=$DEST"

# Submit (on Biowulf, login node):
#   sbatch fetch_refdata.sh default,nullz
