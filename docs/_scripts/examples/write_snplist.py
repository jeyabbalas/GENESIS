"""Write the SNP identifiers of the bundled reference, one per line:  write_snplist.py OUT"""
import sys

import numpy as np

import genesis_gwas as gg

ref = gg.load_reference()
identifiers = ref.subset_ids(np.arange(ref.M))
with open(sys.argv[1], "w") as handle:
    handle.write("\n".join(identifiers) + "\n")
print(f"{ref.M:,} SNP identifiers written to {sys.argv[1]}")
