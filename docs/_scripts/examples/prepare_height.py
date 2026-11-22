import genesis_gwas as gg

ref = gg.load_reference()          # "eur_1kg_hm3", 1 Mb window, r2 >= 0.1
print(ref.describe())

data = gg.prepare("height_giant2010.sumstats.tsv.gz", ref)   # recommended QC, then merge

print(f"\n{'QC step':<62}{'removed':>10}{'remaining':>12}")
for step in data.meta["qc"]:
    print(f"{step['step']:<62}{step['removed']:>10,}{step['remaining']:>12,}")

print(f"\nmarkers for analysis : {data.K:,}")
print(f"reference SNPs (M)   : {data.M:,}")
print(f"mean chi-square      : {(data.z ** 2).mean():.3f}")
print(f"N* per marker        : median {int(sorted(data.nstar)[data.K // 2])}, max {data.nstar.max()}")
