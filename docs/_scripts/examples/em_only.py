"""The EM algorithm on its own, against the default (short EM, then Newton's method).

The truncation level is held at c0 = 10 for both, so that the two differ in the optimiser only.
Running the EM to its own stopping rule takes several minutes for the three-component model.
"""
import numpy as np

import genesis_gwas as gg

ref = gg.load_reference()
data = gg.prepare("height_giant2010.sumstats.tsv.gz", ref)
fixed_c0 = gg.EMConfig(c0_max=10)

start = None
for n_components in (2, 3):
    fast = gg.fit(data, n_components, starting=start, em=fixed_c0)                      # default optimiser
    slow = gg.fit(data, n_components, starting=start, em=fixed_c0, accelerate=False)   # EM to its stopping rule
    start = fast

    polish = slow.diagnostics["polish"]
    shift = (slow.estimates - np.asarray(polish["em_estimates"])) / slow.se
    worst = slow.param_names[int(np.argmax(np.abs(shift)))]
    print(f"{slow.model}: the EM alone stops after {slow.n_iter} iterations, {np.abs(shift).max():.2f} standard errors "
          f"from the maximum (largest for {worst}; pic {shift[0]:+.2f} SE), log-likelihood {polish['loglik_gain']:.2f} below it.")
    print(f"    default: {fast.n_iter} EM iterations + {fast.diagnostics['polish']['n_steps']} Newton steps; "
          f"largest difference between the two final estimates: {np.abs((fast.estimates - slow.estimates) / slow.se).max():.0e} SE")
    slow.to_json(f"results/height_em.{slow.model}.json")
    fast.to_json(f"results/height_em_newton.{fast.model}.json")
