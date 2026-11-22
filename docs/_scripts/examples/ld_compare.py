"""Compare a reference rebuilt from PLINK genotypes with the bundled European reference,
for every LD definition the rebuilt bundle contains:  ld_compare.py BUNDLE_DIR"""
import sys

import numpy as np

import genesis_gwas as gg

path = sys.argv[1]
first = gg.load_reference(path)
print(first.describe())
print("same SNPs in the same order:", np.array_equal(first.rsid, gg.load_reference().rsid), "\n")

print("window   r2 >=   SNPs with identical N*   neighbourhoods identical   largest LD-score difference")
for config in first.configs:
    rebuilt = gg.load_reference(path, window_kb=config.window_kb, r2_min=config.r2_min)
    bundled = gg.load_reference(window_kb=config.window_kb, r2_min=config.r2_min)
    same_n = int((rebuilt.nstar == bundled.nstar).sum())
    a, b = rebuilt.neighborhoods, bundled.neighborhoods
    same_sets = np.array_equal(a.indptr, b.indptr) and np.array_equal(a.indices, b.indices)
    diff = np.abs(rebuilt.ldscore - bundled.ldscore).max()
    print(f"{config.window_kb:>4g} kb   {config.r2_min:<5g}   {same_n:>9,} of {bundled.M:,}   {same_sets!s:>24}   {diff:>27.1e}")
