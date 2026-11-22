"""Simulate a case-control GWAS with realistic LD and write it the way consortia report results."""
import numpy as np
import pandas as pd
from scipy.stats import norm
import genesis_gwas as gg

ref = gg.load_reference()
M = ref.M

# Generating model (log-odds-ratio scale): 4,283 susceptibility SNPs in two clusters.
truth = np.array([0.004, 0.10, 5.0e-4, 4.0e-5, 1.0e-6])   # pic, p1, sigmasq1, sigmasq2, a
n_cases, n_controls = 120_000, 100_000

rng = np.random.default_rng(2026)
coverage = rng.uniform(0.9, 1.0, M)                # not every SNP is available in every study
nca = np.round(n_cases * coverage)
nco = np.round(n_controls * coverage)
n_eff = nca * nco / (nca + nco)                    # effective sample size per SNP

sim = gg.simulate_ld_sumstats(ref, truth, n_eff, seed=11, replicate=0)

eaf = rng.uniform(0.05, 0.95, M)[sim.ref_index]    # illustrative allele frequencies
se = 1.0 / np.sqrt(2 * eaf * (1 - eaf) * sim.n)    # SE of a per-allele log odds ratio
table = pd.DataFrame({
    "SNP": sim.snp,
    "CHR": ref.chrom[sim.ref_index],
    "BP": ref.pos[sim.ref_index],
    "EAF": np.round(eaf, 4),
    "BETA": sim.z * se,
    "SE": se,
    "P": 2 * norm.sf(np.abs(sim.z)),
    "N_CASES": nca[sim.ref_index].astype(int),
    "N_CONTROLS": nco[sim.ref_index].astype(int),
})
table.to_csv("sim_disease.sumstats.tsv.gz", sep="\t", index=False, float_format="%.6g")

h2 = M * truth[0] * (truth[1] * truth[2] + (1 - truth[1]) * truth[3])
print(table.head(4).to_string(index=False))
print(f"\nrows: {len(table):,}   susceptibility SNPs: {M * truth[0]:,.0f}   heritability (log-odds scale): {h2:.4f}")
print(f"mean chi-square: {np.mean(sim.z ** 2):.3f}   lambda_GC: {np.median(sim.z ** 2) / 0.4549364:.3f}")
