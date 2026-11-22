#!/usr/bin/env bash
# Run every command and script shown in the tutorial and record what it printed.
#
#   docs/_scripts/run_examples.sh [WORKDIR]
#
# Each step is stored as a terminal transcript ("$ command" followed by its output) in
# docs/_scripts/output/NAME.txt, which the tutorial pages include verbatim. Fitted models
# and tables go to WORKDIR/results (default ./genesis-docs-work), from where the figure
# scripts read them (see README.md).
#
# Requirements
#   * the genesis-gwas package on PATH (`genesis`, and `python` importing genesis_gwas);
#   * network access to the reference-data release (about 500 MB in total), or a
#     populated $GENESIS_DATA_DIR;
#   * optional: GENESIS_1000G_PLINK=<directory holding 1000G.EUR.QC.{1..22}.bed/bim/fam>
#     for the "ancestry" section (skipped otherwise).
#
# Environment
#   GENESIS_DOCS_SECTIONS   space-separated subset of
#                           "data fit interpret downstream binary consortium ancestry slow"
#                           [all but "slow"]. "slow" runs the EM to its own stopping rule
#                           and the truncation-level series, for two figures of the
#                           theory section (about fifteen minutes).
#   GENESIS_DOCS_ONLY       space-separated transcript names: within the selected sections,
#                           run only these steps (e.g. GENESIS_DOCS_SECTIONS=fit
#                           GENESIS_DOCS_ONLY=degenerate)
#   GENESIS_DOCS_OUTPUT     directory for the transcripts [docs/_scripts/output]
#   PYTHON                  Python interpreter [python]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EX="$HERE/examples"
OUT="${GENESIS_DOCS_OUTPUT:-$HERE/output}"
WORK="${1:-genesis-docs-work}"
PY="${PYTHON:-python}"
SECTIONS="${GENESIS_DOCS_SECTIONS:-data fit interpret downstream binary consortium ancestry}"

mkdir -p "$WORK/results" "$OUT"
OUT="$(cd "$OUT" && pwd)"
cd "$WORK"
WORKDIR="$PWD"

section() { [[ " $SECTIONS " == *" $1 "* ]]; }
wanted() { [[ -z "${GENESIS_DOCS_ONLY:-}" || " $GENESIS_DOCS_ONLY " == *" $1 "* ]]; }

# Replace machine-specific locations in transcripts. With SHOWN_ROOT set, the working
# directory is shown as that (illustrative) absolute path instead of being removed, for
# steps whose point is that they write absolute paths.
sanitise() {
    local sed_args=(-e "s#$WORKDIR/##g" -e "s#$WORKDIR#.#g" -e "s#$EX/##g")
    [[ -n "${SHOWN_ROOT:-}" ]] && sed_args=(-e "s#$WORKDIR#$SHOWN_ROOT#g" -e "s#$EX/##g")
    [[ -n "${GENESIS_DATA_URL:-}" ]] && sed_args+=(-e "s#${GENESIS_DATA_URL%/}#<download URL>#g")
    [[ -n "${GENESIS_1000G_PLINK:-}" ]] && sed_args+=(-e "s#${GENESIS_1000G_PLINK%/}#1000G_EUR_Phase3_plink#g")
    if [[ -n "${GENESIS_DATA_DIR:-}" ]]; then
        if [[ -n "${SHOWN_ROOT:-}" ]]; then
            sed_args+=(-e "s#${GENESIS_DATA_DIR%/}#$(dirname "$SHOWN_ROOT")/genesis-ref#g")
        else
            sed_args+=(-e "s#${GENESIS_DATA_DIR%/}#<data directory>#g")
        fi
    fi
    sed "${sed_args[@]}"
}

# Print "$ command", wrapped with backslash continuations.
show_command() {
    echo "$*" | sanitise | fold -s -w 88 | awk '
        { sub(/[ ]+$/, ""); lines[NR] = $0 }
        END { for (i = 1; i <= NR; i++) printf "%s%s%s\n", (i == 1 ? "$ " : "      "), lines[i], (i < NR ? " \\" : "") }'
}

# capture NAME COMMAND...      run COMMAND; transcript in $OUT/NAME.txt
capture() {
    local name="$1"; shift
    wanted "$name" || return 0
    echo "== $name" >&2
    local shown=("$@")
    [[ "${shown[0]}" == "$PY" ]] && shown[0]="python"
    { show_command "${shown[*]}"; "$@" 2>&1 | sanitise; } > "$OUT/$name.txt"
}

# capture_sh NAME 'SHELL COMMAND'   the same for pipelines and compound commands
capture_sh() {
    local name="$1"; shift
    wanted "$name" || return 0
    echo "== $name" >&2
    { show_command "$1"; bash -c "$1" 2>&1 | sanitise; } > "$OUT/$name.txt"
}

# ------------------------------------------------------------------------ data
if section data; then
    "$PY" -c "import shutil, genesis_gwas as gg; shutil.copy(gg.load_reference().example_path(), 'height_giant2010.sumstats.tsv.gz')"
    capture_sh data_head         "gzip -dc height_giant2010.sumstats.tsv.gz | head -4"
    rm -rf refdata-demo
    capture    data_fetch        genesis data fetch --dest refdata-demo
    capture    data_fetch_more   genesis data fetch --dest refdata-demo --components w500kb_r0.2,examples
    capture    data_list         genesis data list --data-dir refdata-demo
    capture    prepare_height    "$PY" "$EX/prepare_height.py"
    capture    prepare_messy     "$PY" "$EX/prepare_messy.py"
    capture    read_formats      "$PY" "$EX/read_formats.py"
    capture    match_position    "$PY" "$EX/match_position.py"
fi

# ------------------------------------------------------------------------- fit
if section fit; then
    capture    fit_cli           genesis fit height_giant2010.sumstats.tsv.gz --out results/height \
                                     --qq --posterior --project 250000,500000,1000000
    capture_sh fit_files         "ls results | grep '^height\.'"
    capture_sh fit_estimates     "grep -E '^(model|M3)' results/height.estimates.tsv | cut -f1-3,5,6"
    capture    fit_height        "$PY" "$EX/fit_height.py"
    capture    multistart        "$PY" "$EX/multistart.py"
    capture    degenerate        "$PY" "$EX/degenerate.py"
    capture_sh fit_nopolish      "genesis fit height_giant2010.sumstats.tsv.gz --out results/height_nopolish --model 2 --no-polish --quiet && grep -E 'Optimisation|pic  |Convergence check' results/height_nopolish.summary.txt"
fi

# ------------------------------------------------------------------- interpret
if section interpret; then
    capture    interpret         "$PY" "$EX/interpret.py"
    genesis data fetch --components ld > /dev/null 2>&1
    capture    ld_sensitivity    "$PY" "$EX/ld_sensitivity.py"
fi

# ------------------------------------------------------------------ downstream
if section downstream; then
    capture    project           "$PY" "$EX/project.py"
    capture    cli_project       genesis project results/height.best.json --n 250000,500000,1000000 --ci
    capture    predict           "$PY" "$EX/predict.py"
    capture    posterior         "$PY" "$EX/posterior.py"
    capture    future_gc         "$PY" "$EX/future_gc.py"
    capture    cli_plot_density  genesis plot density M2=results/height.M2.json M3=results/height.M3.json \
                                     --out results/height.density.png
    capture    cli_plot_disc     genesis plot discoveries M2=results/height.M2.json M3=results/height.M3.json \
                                     --out results/height.discoveries.png
fi

# ---------------------------------------------------------------------- binary
if section binary; then
    capture    simulate_disease  "$PY" "$EX/simulate_disease.py"
    capture    check_effective_n "$PY" "$EX/check_effective_n.py"
    capture    fit_disease       genesis fit sim_disease.sumstats.tsv.gz --out results/sim_disease \
                                     --population-prevalence 0.10 --sample-prevalence 0.5455
    capture    fit_disease_all   genesis fit sim_disease.sumstats.tsv.gz --out results/sim_disease_all \
                                     --max-chisq inf --population-prevalence 0.10 --sample-prevalence 0.5455 --quiet
    capture    disease           "$PY" "$EX/disease.py"
    capture    cli_predict       genesis predict results/sim_disease.best.json --n 52000,250000 --k-fold 2,3
fi

# ------------------------------------------------------------------ consortium
if section consortium; then
    rm -rf gwas results/consortium genesis.swarm
    capture    toy_gwas          "$PY" "$EX/make_toy_gwas.py"
    capture_sh preflight         "python $EX/preflight.py mini_chr22 gwas/*.EUR.tsv.gz"
    SHOWN_ROOT=/data/jdoe/project \
    capture_sh swarm             "genesis swarm gwas/*.EUR.tsv.gz --outdir results/consortium --swarmfile genesis.swarm --cpus 8 --mem-gb 8 --time 01:00:00 -- --reference mini_chr22 --model both"
    SHOWN_ROOT=/data/jdoe/project \
    capture    swarm_file        cat genesis.swarm
    capture_sh swarm_local       "bash genesis.swarm > results/consortium.local.log 2>&1; ls results/consortium | grep -c . ; ls results/consortium | grep overall"
    capture    collect           "$PY" "$EX/collect_estimates.py"
    capture    legacy_fit        genesis fit gwas/overall.EUR.tsv.gz --reference mini_chr22 --model 2 \
                                     --out results/consortium-legacy/overall.EUR --legacy --quiet
    capture    legacy_compare    "$PY" "$EX/compare_estimates.py" results/consortium/overall.EUR.estimates.tsv \
                                     results/consortium-legacy/overall.EUR.estimates.tsv
fi

# -------------------------------------------------------------------- ancestry
if section ancestry; then
    if [[ -n "${GENESIS_1000G_PLINK:-}" ]]; then
        genesis data fetch --components ld > /dev/null 2>&1
        capture ld_snplist       "$PY" "$EX/write_snplist.py" panel.snplist
        rm -rf refs/eur_rebuilt refs/eur_unrounded
        capture ld_build         genesis ld build --bfile "$GENESIS_1000G_PLINK/1000G.EUR.QC.{chrom}" \
                                     --snps panel.snplist --window-kb 500,1000,2000 --r2 0.05,0.1,0.2 --round-r2 6 \
                                     --name eur_rebuilt --population EUR --genome-build GRCh37 --out refs/eur_rebuilt
        capture ld_compare       "$PY" "$EX/ld_compare.py" refs/eur_rebuilt
        capture ld_build_exact   genesis ld build --bfile "$GENESIS_1000G_PLINK/1000G.EUR.QC.{chrom}" \
                                     --snps panel.snplist --name eur_unrounded --population EUR \
                                     --genome-build GRCh37 --out refs/eur_unrounded --quiet
        capture ld_compare_exact "$PY" "$EX/ld_compare.py" refs/eur_unrounded
        capture ld_fit           genesis fit height_giant2010.sumstats.tsv.gz --reference refs/eur_rebuilt \
                                     --model 2 --keep-mhc --out results/height_rebuilt --quiet
        capture ld_fit_compare   "$PY" "$EX/compare_estimates.py" results/height.estimates.tsv \
                                     results/height_rebuilt.estimates.tsv
    else
        echo "GENESIS_1000G_PLINK is not set: skipping the LD-building example" >&2
    fi
fi

# ------------------------------------------------------------------------ slow
if section slow || section em; then
    capture    em_only           "$PY" "$EX/em_only.py"
fi
if section slow || section truncation; then
    capture    truncation        "$PY" "$EX/truncation.py"
fi

# ------------------------------------------------- cheap listings (re-run alone)
if section listings; then
    capture_sh fit_files         "ls results | grep '^height\.'"
    capture_sh fit_estimates     "grep -E '^(model|M3)' results/height.estimates.tsv | cut -f1-3,5,6"
fi

echo "done: transcripts in $OUT, results in $WORKDIR/results" >&2
