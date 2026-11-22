import numpy as np
import genesis_gwas as gg

ref = gg.load_reference()
best = gg.FitResult.from_json("results/height.best.json")
data = gg.prepare("height_giant2010.sumstats.tsv.gz", ref)

observed = np.median(data.z ** 2) / 0.4549364
print(f"observed lambda_GC of this GWAS (mean n = {data.n.mean():,.0f}): {observed:.3f}")

sizes = [132_000, 250_000, 500_000, 1_000_000]
for n, value in zip(sizes, gg.future_gc(best, sizes, ref, n_sim=5)):
    print(f"predicted lambda_GC at n = {n:>9,}: {value:.3f}")
