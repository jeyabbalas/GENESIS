#!/usr/bin/env python
"""Collect the results of many ``genesis fit`` runs into tidy tables.

Every ``genesis fit --out PREFIX`` writes ``PREFIX.estimates.tsv`` (all estimates of all
fitted models, long format), ``PREFIX.best.json`` (the selected model) and ``PREFIX.log``.
This script walks one or more result directories and writes

``OUT.estimates.tsv``
    every row of every ``*.estimates.tsv``, with two leading columns: ``group`` (the
    directory of the fit relative to the directory you passed, e.g. the ancestry) and
    ``trait`` (the file-name prefix).
``OUT.summary.tsv``
    one row per fit: selected model, numbers of SNPs, convergence, headline estimates
    with standard errors, warnings and provenance (software version, reference, device).

It also lists fits that started (a ``.log`` exists) but did not finish (no
``.best.json``), which is what you need to build a re-run swarm.

Usage (inside an sinteractive session or a batch job, with the environment activated)::

    python collect_results.py /data/$USER/project/results --out /data/$USER/project/all

Only pandas and the standard library are needed; nothing is submitted or deleted.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

SUFFIX = ".estimates.tsv"


def _derived(fit: dict, key: str) -> tuple[float | None, float | None]:
    entry = (fit.get("derived") or {}).get(key) or {}
    return entry.get("estimate"), entry.get("se")


def _model_field(prefix: Path, model: str, field: str) -> object:
    path = Path(f"{prefix}.{model}.json")
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text()).get(field)
    except (OSError, json.JSONDecodeError):
        return None


def summarise(prefix: Path, group: str, trait: str) -> dict:
    """One summary row for the fit whose output prefix is ``prefix``."""
    row: dict = {"group": group, "trait": trait}
    best_path = Path(f"{prefix}.best.json")
    if not best_path.exists():
        row["finished"] = False
        return row
    best = json.loads(best_path.read_text())
    prov = best.get("provenance") or {}
    ref = (prov.get("data") or {}).get("reference") or {}
    compute = prov.get("compute") or {}
    h2, h2_se = _derived(best, "heritability")
    n_ssnp, n_ssnp_se = _derived(best, "n_ssnp")
    liab = (best.get("diagnostics") or {}).get("liability") or {}
    warnings = best.get("warnings") or []
    row.update(
        finished=True,
        selected_model=best.get("model"),
        n_snps=best.get("n_snps"),
        M=best.get("M"),
        mean_n=best.get("mean_n"),
        em_status=best.get("status"),
        converged=best.get("converged"),
        polish=((best.get("diagnostics") or {}).get("polish") or {}).get("status"),
        c0=best.get("c0"),
        n_ssnp=n_ssnp,
        n_ssnp_se=n_ssnp_se,
        heritability=h2,
        heritability_se=h2_se,
        heritability_liability=liab.get("h2_liability"),
        heritability_liability_se=liab.get("se_h2_liability"),
        bic_M2=_model_field(prefix, "M2", "bic"),
        bic_M3=_model_field(prefix, "M3", "bic"),
        n_warnings=len(warnings),
        warnings=" | ".join(warnings),
        reference=f"{ref.get('name')} v{ref.get('version')} {ref.get('config')}",
        genesis_gwas=prov.get("version"),
        compute=compute.get("description"),
        seconds=(best.get("timings") or {}).get("total_seconds"),
    )
    return row


def main(argv: list[str] | None = None) -> int:
    """Collect every fit under the given directories into two tables."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "results", nargs="+", type=Path, help="directories containing genesis fit outputs"
    )
    parser.add_argument(
        "--out", required=True, help="output prefix (writes OUT.estimates.tsv and OUT.summary.tsv)"
    )
    args = parser.parse_args(argv)

    tables, rows, unfinished = [], [], []
    for root in args.results:
        if not root.is_dir():
            parser.error(f"not a directory: {root}")
        for est in sorted(root.rglob(f"*{SUFFIX}")):
            trait = est.name[: -len(SUFFIX)]
            group = est.parent.relative_to(root).as_posix()
            group = "" if group == "." else group
            table = pd.read_csv(est, sep="\t")
            table.insert(0, "trait", trait)
            table.insert(0, "group", group)
            tables.append(table)
            rows.append(summarise(est.parent / trait, group, trait))
        for log in sorted(root.rglob("*.log")):
            prefix = log.with_suffix("")
            if not Path(f"{prefix}.best.json").exists() and log.parent.name != "logs":
                unfinished.append(str(prefix))

    if not tables:
        print(
            f"No *{SUFFIX} files found under: {', '.join(map(str, args.results))}", file=sys.stderr
        )
        return 1
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    estimates = pd.concat(tables, ignore_index=True)
    summary = pd.DataFrame(rows)
    estimates.to_csv(f"{out}.estimates.tsv", sep="\t", index=False)
    summary.to_csv(f"{out}.summary.tsv", sep="\t", index=False)

    print(
        f"{len(rows)} finished fit(s) -> {out}.estimates.tsv ({len(estimates)} rows), {out}.summary.tsv"
    )
    show = [
        "group",
        "trait",
        "selected_model",
        "n_snps",
        "n_ssnp",
        "heritability",
        "heritability_se",
        "n_warnings",
    ]
    print(summary[[c for c in show if c in summary.columns]].to_string(index=False))
    flagged = summary[summary.get("n_warnings", 0) > 0]
    if len(flagged):
        print(
            f"\n{len(flagged)} fit(s) carry warnings; read the `warnings` column before using their numbers."
        )
    if unfinished:
        print(f"\n{len(unfinished)} fit(s) started but did not finish (no .best.json):")
        for prefix in unfinished:
            print(f"  {prefix}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
