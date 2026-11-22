"""A three-component model the data cannot support: one chromosome of the height GWAS."""
import textwrap
import warnings

import genesis_gwas as gg

warnings.simplefilter("ignore")        # the same messages are kept in FitResult.warnings (printed below)

ref = gg.load_reference("mini_chr22")
data = gg.prepare(ref.example_path(), ref)

m2 = gg.fit(data, 2)
m3 = gg.fit(data, 3, starting=m2)

for fit in (m2, m3):
    print(f"{fit.model}: status {fit.status}, c0 = {fit.c0}")
    print("   estimates:", ", ".join(f"{name} = {value:.3g}" for name, value in zip(fit.param_names, fit.estimates)))
    print("   standard errors:", ", ".join(f"{se:.2g}" for se in fit.se))
    print(f"   heritability: {fit.heritability:.4f} (SE {fit.derived['heritability'].se:.4f})")
    for message in fit.warnings:
        print(textwrap.fill(message, 96, initial_indent="   - ", subsequent_indent="     "))
print("\nM3 not identified:", m3.diagnostics["degenerate"], "| parameters on a bound:", m3.diagnostics["at_bound"])
print()
print(gg.select_model([m2, m3]).summary())
