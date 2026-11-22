"""Three small simulated case-control files on the packaged chromosome-22 reference (toy data)."""
from pathlib import Path

import numpy as np
import pandas as pd

import genesis_gwas as gg

ref = gg.load_reference("mini_chr22")
Path("gwas").mkdir(exist_ok=True)
designs = {                       # name: (pic, sigmasq, cases, controls)
    "overall":  (0.010, 1.8e-4, 60_000, 60_000),
    "subtypeA": (0.010, 2.2e-4, 40_000, 60_000),
    "subtypeB": (0.005, 4.5e-4, 15_000, 60_000),
}
for i, (name, (pic, sigmasq, nca, nco)) in enumerate(designs.items()):
    n_eff = nca * nco / (nca + nco)
    sim = gg.simulate_ld_sumstats(ref, [pic, sigmasq, 5e-7], n_eff, seed=100 + i, replicate=i)
    se = np.full(sim.K, 1.0 / np.sqrt(0.42 * n_eff))          # per-allele scale at allele frequency 0.3
    pd.DataFrame({"SNP": sim.snp, "BETA": sim.z * se, "SE": se, "N_CASES": nca, "N_CONTROLS": nco}).to_csv(
        f"gwas/{name}.EUR.tsv.gz", sep="\t", index=False, float_format="%.6g"
    )
    print(f"gwas/{name}.EUR.tsv.gz: {nca:,} cases, {nco:,} controls; simulated truth: "
          f"{ref.M * pic:.0f} susceptibility SNPs, heritability {ref.M * pic * sigmasq:.3f}")
