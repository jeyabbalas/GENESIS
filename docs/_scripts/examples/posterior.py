import genesis_gwas as gg

ref = gg.load_reference()
best = gg.FitResult.from_json("results/height.best.json")
data = gg.prepare("height_giant2010.sumstats.tsv.gz", ref)

post = gg.posterior_marginal(data, best)          # one row per marker
print(post.sort_values("p_null").head(5).round(5).to_string(index=False))
print("...")
print(post[post["z"].abs().between(1.9, 2.0)].head(3).round(5).to_string(index=False, header=False))

genome_wide = post["z"].abs() > 5.4513            # P < 5e-8
print(f"\nmarkers with P < 5e-8    : {genome_wide.sum():>7,}")
for cut in (0.01, 0.05, 0.5):
    print(f"markers with lfdr < {cut:<4} : {(post['p_null'] < cut).sum():>7,}")
selected = post["p_null"] < 0.05
print(f"expected number of false positives among the markers with lfdr < 0.05: "
      f"{post.loc[selected, 'p_null'].sum():,.0f} of {selected.sum():,}")
