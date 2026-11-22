"""What the QC filters do, on a deliberately corrupted copy of the chromosome-22 example."""
import numpy as np
import pandas as pd
import genesis_gwas as gg

ref = gg.load_reference("mini_chr22")
df = pd.read_csv(ref.example_path(), sep="\t")

rng = np.random.default_rng(1)
rows = rng.permutation(len(df))
df.loc[rows[:5], "Z"] = np.nan                    # 5 missing statistics
df.loc[rows[5:8], "N"] = 0                        # 3 impossible sample sizes
df.loc[rows[8:308], "N"] = 40_000                 # 300 SNPs covered by a third of the studies
df.loc[rows[308:312], "Z"] = 9.5                  # 4 extreme associations (chi2 = 90)
df = pd.concat([df, df.iloc[rows[312:322]]])      # 10 duplicated rows
extra = pd.DataFrame({"SNP": [f"rs_unknown_{i}" for i in range(20)], "Z": 0.3, "N": 130_000})
df = pd.concat([df, extra], ignore_index=True)    # 20 SNPs that are not in the reference

data = gg.prepare(df, ref)
print(f"{'QC step':<62}{'removed':>10}{'remaining':>12}")
for step in data.meta["qc"]:
    print(f"{step['step']:<62}{step['removed']:>10,}{step['remaining']:>12,}")
