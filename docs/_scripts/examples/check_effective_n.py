"""Which sample-size convention does a case-control file follow?

For a log odds ratio, SE^2 is close to 1 / (2 f (1 - f) n_eff) with
n_eff = n_cases * n_controls / (n_cases + n_controls), the GENESIS convention.
Comparing that with the sample size you intend to use reveals a factor-of-four mismatch.
"""
import numpy as np
import pandas as pd

df = pd.read_csv("sim_disease.sumstats.tsv.gz", sep="\t")

n_from_se = 1.0 / (2 * df["EAF"] * (1 - df["EAF"]) * df["SE"] ** 2)
n_genesis = df["N_CASES"] * df["N_CONTROLS"] / (df["N_CASES"] + df["N_CONTROLS"])
n_metal = 4.0 / (1.0 / df["N_CASES"] + 1.0 / df["N_CONTROLS"])
n_total = df["N_CASES"] + df["N_CONTROLS"]

print(f"median N implied by SE and allele frequency : {np.median(n_from_se):>10,.0f}")
for label, n in (("cases x controls / total (GENESIS)", n_genesis),
                 ("METAL / PGC effective N", n_metal),
                 ("cases + controls", n_total)):
    print(f"median {label:<36}: {np.median(n):>10,.0f}   ratio to the implied N: {np.median(n / n_from_se):.2f}")
