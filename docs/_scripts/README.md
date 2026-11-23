# Regenerating the tutorial outputs and figures

Everything the tutorial and theory pages show — terminal transcripts and figures — is
produced by the scripts in this directory. Nothing is edited by hand, so the pages can be
brought up to date after a change in the package by running three commands.

```
docs/_scripts/
├── run_examples.sh     runs every command and script of the tutorial
├── examples/           the Python scripts shown in the tutorial pages
├── output/             transcripts ("$ command" + what it printed), included by the pages
├── fig_*.py            figure scripts; write to docs/_static/figures/
├── _common.py          shared plotting helpers
└── runtime_measurements.json   timings behind the run-time figure (edit by hand)
```

Sphinx does not treat this directory as documentation source (`exclude_patterns` in
`docs/conf.py`); the pages pull files in with `literalinclude`.

## Requirements

- the package installed with its plotting extra, with `genesis` and `python` on `PATH`;
- the European reference with all components (about 500 MB; fetched automatically, or
  once with `genesis data fetch --components all`);
- for the chapter on other ancestries: the 1000 Genomes Phase 3 European PLINK files
  distributed with LD-score regression (`1000G_EUR_Phase3_plink/1000G.EUR.QC.{1..22}.*`),
  with their directory in `GENESIS_1000G_PLINK`. Without it that section is skipped and
  its transcripts keep their previous content.

Not produced by `run_examples.sh`: the transcripts of the multi-ancestry part of the
chapter on other ancestries. `kg_from_vcf.txt`, `afr_build.txt`, `afr_fit.txt` and
`eas_rebuild.txt` were saved from the runs that built and checked the East Asian and
African-ancestry references (they need the 15 GB of 1000 Genomes Phase 3 VCF files; the
commands are in the transcripts and in `tools/refdata/README.md`), and
`data_list_all.txt` and `data_fetch_ancestries.txt` are the output of the commands in
their first lines with the data directory shown as `~/.cache/genesis-gwas` or
`refdata-demo`.

## Steps

```bash
# 1. transcripts and fitted models (about 20 minutes on a laptop; "slow" adds about 15)
export GENESIS_1000G_PLINK=/path/to/1000G_EUR_Phase3_plink
export GENESIS_DOCS_SECTIONS="data fit interpret downstream binary consortium ancestry slow"
docs/_scripts/run_examples.sh /path/to/workdir

# 2. figures
export GENESIS_DOCS_RUNS=/path/to/workdir/results
for script in concepts model fit downstream runtime; do
    python docs/_scripts/fig_$script.py
done

# 3. build the documentation
sphinx-build -W --keep-going -b html docs docs/_build/html
```

`run_examples.sh` takes the working directory as its argument (default
`./genesis-docs-work`); fitted models go to `WORKDIR/results`, which is what the figure
scripts read through `--runs` or `GENESIS_DOCS_RUNS`. `GENESIS_DOCS_SECTIONS` selects
parts of the run, so a single chapter can be refreshed, for example
`GENESIS_DOCS_SECTIONS="binary"`. Later sections read results of earlier ones (`fit`
before `interpret`, `downstream` and `ancestry`). The two parts of `slow` can be run
separately as `em` and `truncation`, and `listings` refreshes the two file listings of
the fitting chapter without refitting.

Each figure script accepts `--out` (default `docs/_static/figures`), `--runs` and
`--reference`; `fig_fit.py --skip-qq` and `fig_downstream.py --skip-posterior` skip the
two slow figures.

| Script | Figures | Needs |
|---|---|---|
| `fig_concepts.py` | `ld-neighbourhood`, `joint-vs-marginal`, `effect-prior`, `marginal-mixture` | `height.M2.json`, `height.M3.json`; reference with pairwise LD |
| `fig_model.py` | `truncation-mass`, `truncation-sensitivity`, `sandwich-neighbourhood` | the same; `truncation.json` (section `slow`) |
| `fig_fit.py` | `em-trajectory`, `effect-density`, `ld-sensitivity`, `qq-height` | also `height_em.M{2,3}.json` and `height_em_newton.M{2,3}.json` (section `slow`), `ld_sensitivity.json` (section `interpret`), null replicates |
| `fig_downstream.py` | `projection`, `prs-height`, `auc-disease`, `posterior` | also `sim_disease.best.json` (section `binary`) |
| `fig_runtime.py` | `runtime` | `runtime_measurements.json` |

## Conventions

- Transcripts are sanitised by `run_examples.sh`: the working directory, the data
  directory, the download location and the genotype directory are replaced by short
  placeholders, so that no machine-specific path reaches the documentation.
- Random seeds are fixed in every script. Outputs still differ in the last digits between
  machines and library versions.
- Figures are SVG, except dense scatter plots (PNG, 200 dpi); colours come from
  `genesis_gwas.plotting.PALETTE`.
- The run-time figure is drawn from numbers recorded in `runtime_measurements.json`,
  together with the machine they were measured on. Update that file when you re-measure.
