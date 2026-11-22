import numpy as np
import genesis_gwas as gg

m2 = gg.FitResult.from_json("results/height.M2.json")
m3 = gg.FitResult.from_json("results/height.M3.json")

print("M3 parameters")
for name, est, se in zip(m3.param_names, m3.estimates, m3.se):
    print(f"  {name:<9} {est:.4e}   (SE {se:.1e})")

print("\nM3 derived quantities")
for key, q in m3.derived.items():
    print(f"  {key:<22} {q.estimate:>10.5g}   (SE {q.se:.4g})")

weights, variances = m3.mixture
print("\nSD of effects in cluster 1 and 2:", np.sqrt(variances).round(5), " shares:", weights.round(4))

# Expected number of SNPs by absolute effect size (in SD of height per SD of genotype)
edges = np.array([0.0, 0.0025, 0.005, 0.01, 0.02, np.inf])
print("\n|effect| range        M2 count   M3 count")
for lo, hi in zip(edges[:-1], edges[1:]):
    print(f"  {lo:<7g}- {hi:<8g} {float(gg.n_in_interval(lo, hi, m2)):>9.0f} {float(gg.n_in_interval(lo, hi, m3)):>10.0f}")

# Standard errors under the original software's sandwich convention, for comparison
alt = m3.diagnostics["sandwich_alt"]
print("\nSE, original convention / default:", (np.array(alt["se"]) / m3.se).round(3))
