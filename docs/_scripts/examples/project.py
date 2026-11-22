import genesis_gwas as gg

best = gg.FitResult.from_json("results/height.best.json")      # the model selected by BIC

proj = gg.project(best, n=[132_000, 250_000, 500_000, 700_000, 1_000_000, 2_000_000], ci=True)

table = proj.to_frame()
table["pheno_variance"] *= 100                                  # per cent of the trait variance
table = table.rename(columns={
    "n_discoveries": "discoveries", "n_discoveries_lower": "lower", "n_discoveries_upper": "upper",
    "gv_percentage": "h2_explained_%", "gv_percentage_lower": "lower_%", "gv_percentage_upper": "upper_%",
    "pheno_variance": "trait_variance_%",
})
table["n"] = table["n"].astype(int)
columns = ["n", "discoveries", "lower", "upper", "h2_explained_%", "lower_%", "upper_%", "trait_variance_%"]
print(table[columns].round(1).to_string(index=False))
