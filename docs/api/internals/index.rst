Internals
=========

These modules implement the method. They are documented for readers who want to
audit or extend the algorithm; analyses do not need to import them. The
:doc:`theory pages </theory/index>` derive the mathematics that each of them
implements.

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Module
     - Role
   * - :doc:`spec`
     - Parameter layout of the two- and three-component models; tables of latent
       configurations and log-multinomial coefficients.
   * - :doc:`kernels`
     - The inner loops: log-space E-step, M-step moments, scores, information and
       posterior summaries for one chunk of SNPs, written once for NumPy and PyTorch.
   * - :doc:`engine`
     - Chunked map-reduce over SNPs on CPU threads or GPU shards.
   * - :doc:`likelihood`
     - The composite likelihood, its scores and observed information, assembled from
       the kernel outputs.
   * - :doc:`em`
     - Expectation-maximisation with safeguards, the literal ``legacy`` control flow,
       and Newton polishing.
   * - :doc:`variance`
     - Sandwich covariance over tagging neighbourhoods; information criteria.
   * - :doc:`start`
     - Starting values from LD-score regression.
   * - :doc:`derived`
     - Derived quantities (number of susceptibility SNPs, heritability, ...) with
       delta-method standard errors.
   * - :doc:`hpc`
     - Swarm-file generation for Slurm clusters.
   * - :doc:`cli`
     - Entry point of the ``genesis`` command.

.. toctree::
   :hidden:

   spec
   kernels
   engine
   likelihood
   em
   variance
   start
   derived
   hpc
   cli
