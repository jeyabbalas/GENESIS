Building LD references
======================

.. automodule:: genesis_gwas.ld
   :no-members:

Building a bundle
-----------------

.. automodule:: genesis_gwas.ld.build
   :no-members:

.. autofunction:: genesis_gwas.ld.build_reference

.. autofunction:: genesis_gwas.ld.compute_ld

.. autoclass:: genesis_gwas.ld.LDTable

.. autofunction:: genesis_gwas.ld.null_z

.. autofunction:: genesis_gwas.ld.standardize_genotypes

.. autofunction:: genesis_gwas.ld.allele_frequencies

Converting VCF files
--------------------

.. automodule:: genesis_gwas.ld.vcf
   :no-members:

.. autofunction:: genesis_gwas.ld.vcf_to_plink

Positions in a second genome build
----------------------------------

.. automodule:: genesis_gwas.ld.liftover
   :no-members:

.. autoclass:: genesis_gwas.ld.liftover.ChainLiftover
   :members: lift

.. autofunction:: genesis_gwas.ld.liftover.lift_positions

Reading PLINK files
-------------------

.. automodule:: genesis_gwas.ld.plink
   :no-members:

.. autoclass:: genesis_gwas.ld.BedReader

.. autofunction:: genesis_gwas.ld.read_bim

.. autofunction:: genesis_gwas.ld.read_fam

.. autofunction:: genesis_gwas.ld.resolve_fileset
