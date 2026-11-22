import genesis_gwas as gg

best = gg.FitResult.from_json("results/height.best.json")

# A score built from the SNPs that pass a P-value threshold in a GWAS of size n, weighted by
# their estimated effects. M = 200,000 independent SNPs, as in the paper.
print("  GWAS size   best threshold   R2 at best threshold   R2 with P < 5e-8 only")
for n in (132_000, 250_000, 500_000, 1_000_000, 2_000_000):
    best_threshold = gg.polyrisk_predict(n, best, M=200_000)
    gwas_only = gg.polyrisk_predict(n, best, M=200_000, threshold="gwas")
    print(f"{n:>11,}   {best_threshold.alpha:>14.1e}   {best_threshold.mu ** 2:>20.3f}   {gwas_only.mu ** 2:>21.3f}")
print(f"\nheritability (the ceiling for R2): {best.heritability:.3f}")
