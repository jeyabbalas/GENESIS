import json
import warnings
import numpy as np
import genesis_gwas as gg

rows = []
for window_kb in (500, 1000, 2000):
    for r2_min in (0.05, 0.1, 0.2):
        ref = gg.load_reference(window_kb=window_kb, r2_min=r2_min)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = gg.prepare("height_giant2010.sumstats.tsv.gz", ref)
            fit = gg.fit(data, 2)
        d = fit.derived
        rows.append({
            "window_kb": window_kb, "r2_min": r2_min,
            "mean_nstar": float(np.mean(data.nstar)), "mean_ldscore": float(np.mean(data.ldscore)),
            "n_ssnp": d["n_ssnp"].estimate, "n_ssnp_se": d["n_ssnp"].se,
            "h2": d["heritability"].estimate, "h2_se": d["heritability"].se,
            "a": fit.a, "loglik": fit.loglik,
        })
        r = rows[-1]
        print(f"window {window_kb:>4} kb, r2 >= {r2_min:<4}: mean N* {r['mean_nstar']:5.1f}, mean LD score {r['mean_ldscore']:5.2f}"
              f"  ->  sSNPs {r['n_ssnp']:6.0f} ({r['n_ssnp_se']:.0f}),  h2 {r['h2']:.3f} ({r['h2_se']:.3f})", flush=True)

with open("results/ld_sensitivity.json", "w") as fh:
    json.dump(rows, fh, indent=1)
