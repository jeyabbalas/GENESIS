import numpy as np
import genesis_gwas as gg

ref = gg.load_reference()
data = gg.prepare("height_giant2010.sumstats.tsv.gz", ref)

print("starting pic   after 50 EM iterations     after Newton's method     log-likelihood")
print("               pic        sigmasq         pic        sigmasq")
for pic0 in (0.001, 0.005, 0.01, 0.05):
    fit = gg.fit(data, 2, starting_pic=pic0, variance=False)
    em = np.asarray(fit.diagnostics["polish"]["em_estimates"])
    print(f"{pic0:<13g}  {em[0]:.6f}   {em[1]:.4e}      {fit.pic:.6f}   {fit.estimates[1]:.4e}      {fit.loglik:.4f}")

# The same in one call: the start with the highest log-likelihood is kept.
best = gg.fit(data, 2, starting_pic=[0.001, 0.005, 0.01, 0.05], variance=False)
print("\nruns recorded in diagnostics['multistart']:", len(best.diagnostics["multistart"]),
      "| log-likelihoods:", sorted({round(run["loglik"], 4) for run in best.diagnostics["multistart"]}))
