#!/bin/bash
#SBATCH --job-name=genesis-ld
#SBATCH --cpus-per-task=16
#SBATCH --mem=16g
#SBATCH --time=02:00:00
# ld_build.sh -- build an LD reference bundle (N*, LD scores, tagging neighbourhoods, null
# Z-scores) for one population from PLINK genotypes, in the format of the downloadable
# references. Needed only for a population that the downloadable references (European,
# East Asian, African ancestry) do not serve, or to use a panel of your own; every fit for
# that ancestry then uses `--reference OUT_DIR`.
# Genotypes in VCF format (e.g. the 1000 Genomes Project's) are converted to PLINK files
# first, with `genesis ld from-vcf` (see the "Other ancestries" chapter of the tutorial).
#
#   sbatch ld_build.sh BFILE OUT_DIR NAME POPULATION SNPLIST [genesis ld build options]
#
#   BFILE       PLINK prefix; write {chrom} where the chromosome number goes when there is
#               one fileset per chromosome, e.g. /data/$USER/panels/SAS/chr{chrom}
#   OUT_DIR     directory of the new bundle, e.g. /data/$USER/proj/ref/SAS
#   NAME        name recorded in the bundle and in every result, e.g. sas_1kg_hm3
#   POPULATION  label, e.g. SAS
#   SNPLIST     text file, one SNP identifier per line: the SNPs that define the reference
#               panel (HapMap3 without the MHC is the paper's choice)
#
# Measured on a laptop for 489 samples x 1.07 million SNPs: under a minute and 2-5 GB.
# Time and memory grow with the number of samples; calibrate from the first run.
set -euo pipefail

if [ "$#" -lt 5 ]; then
    echo "usage: sbatch ld_build.sh BFILE OUT_DIR NAME POPULATION SNPLIST [genesis ld build options]" >&2
    exit 2
fi
BFILE=$1
OUT_DIR=$2
NAME=$3
POPULATION=$4
SNPLIST=$5
shift 5

set +u    # conda's activate scripts are not written for 'set -u'
eval "${GENESIS_ACTIVATE:-source /data/$USER/conda/etc/profile.d/conda.sh && conda activate genesis}"
set -u
# The LD computation is matrix products in the BLAS library: `genesis ld build` caps the
# BLAS threads at --threads (default: $SLURM_CPUS_PER_TASK); the variables do the same for
# libraries it cannot reach.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS MKL_NUM_THREADS=$OMP_NUM_THREADS
umask 027

echo "host=$(hostname -s)  job=${SLURM_JOB_ID:-none}  cpus=${SLURM_CPUS_PER_TASK:-unset}  mem=${SLURM_MEM_PER_NODE:-unset}MB"
echo "bfile=$BFILE  out=$OUT_DIR  name=$NAME  population=$POPULATION  snps=$SNPLIST"
genesis --version

# --genome-build must describe the positions in the .bim file(s): pass `--genome-build GRCh38`
# as an extra option if they are GRCh38. Add `--liftover-chain FILE` (UCSC
# hg19ToHg38.over.chain.gz for a GRCh37 panel, hg38ToHg19.over.chain.gz for a GRCh38 one)
# so that the reference matches summary statistics by position in either build.
# --n-null stores LD-correlated null Z-scores, which the QQ-plot envelopes need (100
# replicates cost about 0.2 GB per million SNPs).
genesis ld build \
    --bfile "$BFILE" \
    --out "$OUT_DIR" \
    --name "$NAME" \
    --population "$POPULATION" \
    --snps "$SNPLIST" \
    --window-kb 1000 --r2 0.1 \
    --min-maf 0.05 \
    --n-null 100 \
    --device cpu \
    "$@"

genesis data verify "$OUT_DIR"
du -sh "$OUT_DIR"

# Submit (on Biowulf, login node), for example:
#   sbatch ld_build.sh /data/$USER/proj/panels/SAS/chr{chrom} /data/$USER/proj/ref/SAS sas_1kg_hm3 SAS \
#          /data/$USER/proj/ref/hapmap3_noMHC.snplist
# ({chrom} is typed literally: the shell leaves it alone and GENESIS substitutes 1..22.)
# Options after the five arguments go to `genesis ld build`, e.g. --genome-build GRCh38,
# --liftover-chain hg19ToHg38.over.chain.gz, --keep samples.txt, or
# --window-kb 500,1000,2000 --r2 0.05,0.1,0.2 for all nine configurations.
