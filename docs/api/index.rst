.. _api:

Python API
==========

Everything most analyses need is importable from the top-level package:

.. code-block:: python

   import genesis_gwas as gg

   ref = gg.load_reference()                        # LD reference (cached download)
   data = gg.prepare("trait.sumstats.tsv.gz", ref)  # QC + merge
   m2 = gg.fit(data, 2)                             # two-component model
   m3 = gg.fit(data, 3, starting=m2)                # three-component model
   best = gg.select_model([m2, m3]).best

The tables below list every public object by task. Each object is documented exactly
once, under the name you import it by; the modules that *define* them
(``genesis_gwas.core``, ``genesis_gwas.backends``, ...) are described under
:doc:`internals/index` for readers who want to follow the algorithm through the code.

.. note::

   Names not listed on these pages (anything starting with an underscore, and
   everything in ``genesis_gwas.cli.commands``) are implementation details and may
   change without notice.

.. module:: genesis_gwas
   :synopsis: GENetic Effect-Size distribution Inference from Summary-level data.

Preparing data
--------------

See :doc:`data`.

.. autosummary::
   :nosignatures:

   read_sumstats
   SumStats
   prepare
   QCConfig
   ModelData
   load_reference
   LDReference
   fetch_bundle
   save_reference

Fitting and results
-------------------

See :doc:`fitting`.

.. autosummary::
   :nosignatures:

   fit
   EMConfig
   ModelSpec
   Neighborhoods
   FitResult
   select_model
   ModelComparison

Downstream analyses
-------------------

See :doc:`analysis`.

.. autosummary::
   :nosignatures:

   effect_density
   n_in_interval
   project
   polyrisk_predict
   posterior_marginal
   qq_simulation
   future_gc
   h2_transfer
   simulate_sumstats
   simulate_ld_sumstats

Plotting
--------

See :doc:`plotting`.

.. autosummary::
   :nosignatures:

   plotting.qq_plot
   plotting.plot_effect_density
   plotting.plot_projection
   plotting.plot_auc
   plotting.save_figure

LD references for other populations
-----------------------------------

See :doc:`ld`.

.. autosummary::
   :nosignatures:

   ld.build_reference
   ld.compute_ld
   ld.null_z
   ld.BedReader

Compute devices
---------------

See :doc:`compute`.

.. autosummary::
   :nosignatures:

   resolve_plan
   ComputePlan
   compute.available_cpus
   compute.describe_environment

Errors and warnings
-------------------

See :doc:`exceptions`.

.. autosummary::
   :nosignatures:

   GenesisError
   exceptions.InputError
   exceptions.DataError
   exceptions.ConvergenceWarning

.. toctree::
   :hidden:
   :maxdepth: 2

   data
   fitting
   analysis
   plotting
   ld
   compute
   exceptions
   internals/index
