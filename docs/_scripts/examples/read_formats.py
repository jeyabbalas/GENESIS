"""The same three SNPs written the way different consortia report them."""
import io
import pandas as pd
import genesis_gwas as gg

def show(title, text, **options):
    table = gg.read_sumstats(pd.read_csv(io.StringIO(text), sep=r"\s+"), **options)
    print(f"--- {title}")
    print(f"    Z from: {table.info['z_from']}")
    print(f"    N from: {table.info['n_from']}")
    print(table.table[["SNP", "Z", "N"]].round(4).to_string(index=False), "\n")

show("Z statistic and sample size", """
SNP        Z       N
rs1    2.5000  100000
rs2   -1.2000   98000
rs3    0.3000  100000
""")

show("effect and standard error; one sample size for all SNPs", """
rsid   beta     se
rs1    0.0250   0.0100
rs2   -0.0120   0.0100
rs3    0.0030   0.0100
""", n=100_000)

show("odds ratio, its standard error, cases and controls", """
MarkerName  OR      SE      N_cases  N_controls
rs1         1.0253  0.0100  60000    40000
rs2         0.9881  0.0100  60000    38000
rs3         1.0030  0.0100  60000    40000
""")

show("P-value with a signed effect; METAL-style effective N (4x the GENESIS convention)", """
SNP   BETA     P         Neff
rs1    0.025   0.012419  96000
rs2   -0.012   0.230139  93061
rs3    0.003   0.764177  96000
""", n="Neff", n_scale=0.25)

show("per-SNP effective N from the standard error and the allele frequency", """
SNP   BETA       SE        EAF
rs1    0.036085  0.014434  0.20
rs2   -0.015119  0.012599  0.30
rs3    0.004000  0.013333  0.75
""", n_from_se=True)
