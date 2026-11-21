#!/usr/bin/env bash
# run_all.sh -- regenerate every upstream fixture.
#
#   tools/upstream_oracle/run_all.sh <upstream_clone> <work_dir> [threads_for_full_runs]
#
#   <upstream_clone>  a checkout of https://github.com/yandorazhang/GENESIS
#                     (fixtures in this repository: commit e4e6894)
#   <work_dir>        scratch directory (isolated R library, build products,
#                     logs, large per-SNP dumps); several GB are written here
#
# Outputs (relative to the repository root):
#   tests/fixtures/upstream/   synthetic_*.json.gz, rfuncs.json, mini_chr22_*.json(.gz), height_full_m{2,3}.json
#   tests/data/mini_chr22/     chr22 slice of the upstream reference + height GWAS
#   <work_dir>/height_m{2,3}.json (+ *.f64)   full-scale dumps incl. per-SNP scores (not committed)
set -euo pipefail
if [ $# -lt 2 ]; then echo "usage: $0 <upstream_clone> <work_dir> [threads]" >&2; exit 2; fi
UPSTREAM=$(cd "$1" && pwd); mkdir -p "$2"; WORK=$(cd "$2" && pwd); THREADS=${3:-8}
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../.." && pwd)
FX="$REPO/tests/fixtures/upstream"; MINI="$REPO/tests/data/mini_chr22"
mkdir -p "$FX" "$MINI" "$WORK/logs"
R="Rscript --vanilla"

$R "$HERE/setup_rlib.R" "$WORK"

# macOS + Apple clang: fetch the OpenMP header that matches R's bundled libomp
# into the work dir (NOT /usr/local).  Without it the build falls back to the
# serial stub in stub/omp.h, which is fine for everything but slow for step 4.
if [ "$(uname -s)" = "Darwin" ] && [ ! -f "$WORK/omp/usr/local/include/omp.h" ]; then
  mkdir -p "$WORK/omp"
  if curl -fsS -m 120 -o "$WORK/omp/openmp.tar.gz" https://mac.r-project.org/openmp/openmp-19.1.5-darwin20-Release.tar.gz \
     && echo "5b44175bcbaa334b0c57391482e068ea185c95a2  $WORK/omp/openmp.tar.gz" | shasum -a 1 -c - >/dev/null; then
    tar xzf "$WORK/omp/openmp.tar.gz" -C "$WORK/omp"
  else
    echo "[run_all] could not fetch/verify libomp tarball; building without OpenMP" >&2
    rm -f "$WORK/omp/openmp.tar.gz"
  fi
fi

echo "[run_all] 1/4 synthetic function-level + EM-trace fixtures"
$R "$HERE/gen_synthetic.R" "$HERE" "$UPSTREAM" "$WORK" "$FX" small,medium 2> "$WORK/logs/synthetic.log"
echo "[run_all] 2/4 pure-R function fixtures"
$R "$HERE/gen_rfuncs.R" "$HERE" "$UPSTREAM" "$WORK" "$FX" 2> "$WORK/logs/rfuncs.log"
echo "[run_all] 3/4 chr22 mini pipeline"
$R "$HERE/gen_mini_chr22.R" "$HERE" "$UPSTREAM" "$WORK" "$MINI" "$FX" 100 2> "$WORK/logs/mini_chr22.log"
echo "[run_all] 4/4 full-scale height GWAS (2- then 3-component; ~25 min on 12 threads)"
$R "$HERE/gen_height_full.R" "$HERE" "$UPSTREAM" "$WORK" 2,3 "$THREADS" "$WORK" "$FX" "$WORK/logs/height_full.stdout.log" \
   > "$WORK/logs/height_full.stdout.log" 2> "$WORK/logs/height_full.stderr.log"
echo "[run_all] done"; ls -la "$FX"
