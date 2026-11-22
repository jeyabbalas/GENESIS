import numpy as np

import genesis_gwas as gg

truth_theta = [0.004, 0.10, 5.0e-4, 4.0e-5, 1.0e-6]            # as simulated
truth = {"n_ssnp": 4283, "n_ssnp_cluster1": 428, "heritability": 0.3683,
         "heritability_cluster1": 0.2142, "heritability_cluster2": 0.1542}
fits = {"default QC": gg.FitResult.from_json("results/sim_disease.best.json"),
        "no chi-square filter": gg.FitResult.from_json("results/sim_disease_all.best.json")}
M = fits["default QC"].M

header = f"{'':<26}{'truth':>8}" + "".join(f"{name + ' (' + fit.model + ')':>28}" for name, fit in fits.items())
print(header)
for key, value in truth.items():
    cells = "".join(f"{fit.derived[key].estimate:.4g} ({fit.derived[key].se:.3g})".rjust(28) for fit in fits.values())
    print(f"{key:<26}{value:>8.4g}{cells}")
print(f"{'markers analysed':<26}{'':>8}" + "".join(f"{fit.n_snps:>28,}" for fit in fits.values()))

print("\nexpected number of SNPs with an absolute effect (log odds ratio per SD of genotype) above x")
for x in (0.005, 0.01, 0.02, 0.03):
    cells = "".join(f"{float(gg.n_in_interval(x, np.inf, fit)):>28.0f}" for fit in fits.values())
    print(f"{'x = ' + str(x):<26}{float(gg.n_in_interval(x, np.inf, truth_theta, M=M)):>8.0f}{cells}")

fit = fits["default QC"]
h2 = fit.derived["heritability"]
scales = gg.h2_transfer(h2.estimate, h2.se, population_prevalence=0.10, sample_prevalence=120 / 220)
print(f"\nheritability, log-odds scale : {h2.estimate:.3f} (SE {h2.se:.3f})")
print(f"heritability, observed scale : {scales['h2_observed']:.3f} (SE {scales['se_h2_observed']:.3f})")
print(f"heritability, liability scale: {scales['h2_liability']:.3f} (SE {scales['se_h2_liability']:.3f})")
risk = fit.derived["sibling_risk"]
print(f"sibling relative risk        : {risk.estimate:.3f} (SE {risk.se:.3f})")

# Polygenic risk score built from GWAS of increasing effective size (200,000 independent SNPs)
print("\n effective n     AUC    people at >= 2x average risk: % of population, % of future cases")
for n in (52_000, 100_000, 250_000, 1_000_000):
    pred = gg.polyrisk_predict(n, fit, M=200_000, k_fold=[2])
    print(f"{n:>12,}   {pred.auc:.3f}   {100 * pred.ppi[0]:>41.2f}   {100 * pred.pci[0]:>18.2f}")
