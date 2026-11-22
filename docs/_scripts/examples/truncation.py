"""How much do the estimates depend on the truncation level c0?

Each fit below holds c0 fixed (c0_max = c0), which switches the automatic rule off.
The three-component fits of the full height data take a few minutes.
"""
import json
import warnings

import genesis_gwas as gg

warnings.simplefilter("ignore")        # fixed levels deliberately provoke the truncation warning


def fits_at(data, n_components, levels, start=None):
    rows = []
    for c0 in levels:
        fit = gg.fit(data, n_components, starting=start, em=gg.EMConfig(c0=c0, c0_max=c0),
                     newton_after=50 if not rows else 1)
        start = fit                                   # the next level starts from this solution
        h2 = fit.derived["heritability"]
        rows.append({"c0": c0, "mean_tail": fit.diagnostics["truncation"]["mean_tail"],
                     "pic": fit.pic, "pic_se": float(fit.se[0]), "h2": h2.estimate, "h2_se": h2.se,
                     "loglik": fit.loglik, "status": fit.status})
        r = rows[-1]
        print(f"  c0 = {c0:<3} discarded prior mass {r['mean_tail']:.1e}   pic {r['pic']:.4e} (SE {r['pic_se']:.1e})"
              f"   h2 {r['h2']:.4f} (SE {r['h2_se']:.4f})", flush=True)
    return rows, fit


results = {}

chr22 = gg.load_reference("mini_chr22")
print("chromosome 22 of the height GWAS (13,230 markers), M2")
results["chr22_M2"], _ = fits_at(gg.prepare(chr22.example_path(), chr22), 2, (10, 12, 15, 20, 25, 30, 40))

ref = gg.load_reference()
height = gg.prepare("height_giant2010.sumstats.tsv.gz", ref)
print("\nfull height GWAS (941,684 markers), M2")
results["height_M2"], m2 = fits_at(height, 2, (10, 12, 14, 16, 20))
print("\nfull height GWAS, M3")
results["height_M3"], _ = fits_at(height, 3, (10, 12, 14, 16, 20), start=m2)

with open("results/truncation.json", "w") as handle:
    json.dump(results, handle, indent=1)
