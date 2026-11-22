Downstream analyses
===================

.. automodule:: genesis_gwas.analysis
   :no-members:

Every function accepts either a :class:`~genesis_gwas.FitResult` or a raw parameter
vector (three values for the two-component model, five for the three-component model).

Effect-size distribution
------------------------

.. automodule:: genesis_gwas.analysis.density
   :no-members:

.. autofunction:: genesis_gwas.effect_density

.. autofunction:: genesis_gwas.analysis.effect_cdf

.. autofunction:: genesis_gwas.n_in_interval

.. autofunction:: genesis_gwas.analysis.effect_mixture

.. autoclass:: genesis_gwas.analysis.density.EffectModel

.. autofunction:: genesis_gwas.analysis.density.resolve_model

.. autodata:: genesis_gwas.analysis.density.DEFAULT_M

Projections for future studies
------------------------------

.. automodule:: genesis_gwas.analysis.projection
   :no-members:

.. autofunction:: genesis_gwas.project

.. autoclass:: genesis_gwas.analysis.Projection

.. autofunction:: genesis_gwas.analysis.projection.power_integrals

Polygenic risk prediction
-------------------------

.. automodule:: genesis_gwas.analysis.prediction
   :no-members:

.. autofunction:: genesis_gwas.polyrisk_predict

.. autoclass:: genesis_gwas.analysis.PolygenicPrediction

.. autofunction:: genesis_gwas.analysis.prediction.polyrisk_mu

Posterior inference for individual markers
------------------------------------------

.. automodule:: genesis_gwas.analysis.posterior
   :no-members:

.. autofunction:: genesis_gwas.posterior_marginal

Simulation, QQ plots and genomic control
----------------------------------------

.. automodule:: genesis_gwas.analysis.simulate
   :no-members:

.. autofunction:: genesis_gwas.qq_simulation

.. autoclass:: genesis_gwas.analysis.QQData

.. autofunction:: genesis_gwas.future_gc

.. autofunction:: genesis_gwas.analysis.lambda_gc

.. autofunction:: genesis_gwas.simulate_sumstats

.. autofunction:: genesis_gwas.simulate_ld_sumstats

Heritability scales
-------------------

.. automodule:: genesis_gwas.analysis.heritability
   :no-members:

.. autofunction:: genesis_gwas.h2_transfer
