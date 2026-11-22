"""Gather PREFIX.estimates.tsv of many fits into one table (files are named TRAIT.ANCESTRY.*)."""
from pathlib import Path

import pandas as pd

tables = []
for path in sorted(Path("results/consortium").glob("*.estimates.tsv")):
    trait, ancestry = path.name.split(".")[:2]
    table = pd.read_csv(path, sep="\t")
    table.insert(0, "ancestry", ancestry)
    table.insert(0, "trait", trait)
    tables.append(table)
estimates = pd.concat(tables, ignore_index=True)
estimates.to_csv("results/consortium/all_estimates.tsv", sep="\t", index=False)
print(f"{len(tables)} fits, {len(estimates)} rows -> results/consortium/all_estimates.tsv\n")

# One row per fit: the headline numbers of the selected model.
selected = estimates[estimates["selected"] == 1]
keys = ["trait", "ancestry"]
value = selected.pivot(index=keys, columns="quantity", values="estimate")
error = selected.pivot(index=keys, columns="quantity", values="se")
summary = pd.DataFrame({
    "model": selected.groupby(keys)["model"].first(),
    "markers": value["n_snps"].astype(int),
    "sSNPs": value["n_ssnp"].round(0).astype(int),
    "sSNPs_se": error["n_ssnp"].round(0).astype(int),
    "h2": value["heritability"].round(3),
    "h2_se": error["heritability"].round(3),
    "converged": value["converged"].astype(int),
})
print(summary.to_string())
