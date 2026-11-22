"""Read and quality-control every file without fitting anything:  preflight.py REFERENCE FILES..."""
import sys
import warnings
from pathlib import Path

import numpy as np

import genesis_gwas as gg

warnings.simplefilter("ignore")        # QC messages are summarised in the table below
ref = gg.load_reference(sys.argv[1])
print(f"reference: {ref.describe()}\n")
print(f"{'file':<22}{'rows':>8}{'kept':>8}{'chi2>80':>9}{'median N':>10}{'mean chi2':>11}{'lambda_GC':>11}   Z from; N from")
for path in sys.argv[2:]:
    sumstats = gg.read_sumstats(path)
    data = gg.prepare(sumstats, ref)
    removed = {step["step"]: step["removed"] for step in data.meta["qc"]}
    big = next((n for step, n in removed.items() if step.startswith("chi-square")), 0)
    chi2 = data.z ** 2
    print(f"{Path(path).name:<22}{sumstats.info['rows_read']:>8,}{data.K:>8,}{big:>9,}{np.median(data.n):>10,.0f}"
          f"{chi2.mean():>11.3f}{np.median(chi2) / 0.4549364:>11.3f}   {sumstats.info['z_from']}; {sumstats.info['n_from']}")
