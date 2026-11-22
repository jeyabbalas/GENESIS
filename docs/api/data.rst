Preparing data
==============

Three steps turn a GWAS results file into model inputs:

1. :func:`~genesis_gwas.read_sumstats` parses the file into ``SNP`` / ``Z`` / ``N``.
2. :func:`~genesis_gwas.load_reference` opens an LD reference bundle.
3. :func:`~genesis_gwas.prepare` filters the SNPs, merges them with the reference and
   returns a :class:`~genesis_gwas.ModelData`.

``prepare`` accepts a file path directly, so steps 1 and 3 are usually one call.

.. module:: genesis_gwas.data
   :synopsis: Summary statistics, LD reference bundles, quality control and merging.

Summary statistics
------------------

.. automodule:: genesis_gwas.data.sumstats
   :no-members:

.. autofunction:: genesis_gwas.read_sumstats

.. autoclass:: genesis_gwas.SumStats

.. autodata:: genesis_gwas.data.sumstats.COLUMN_ALIASES
   :no-value:

Quality control and merging
---------------------------

.. automodule:: genesis_gwas.data.prepare
   :no-members:

.. autofunction:: genesis_gwas.prepare

.. autoclass:: genesis_gwas.QCConfig

.. autoclass:: genesis_gwas.ModelData

LD reference bundles
--------------------

.. automodule:: genesis_gwas.data.reference
   :no-members:

.. autofunction:: genesis_gwas.load_reference

.. autoclass:: genesis_gwas.LDReference

.. autoclass:: genesis_gwas.data.LDConfig

.. autofunction:: genesis_gwas.save_reference

.. autofunction:: genesis_gwas.data.reference.config_id

.. autofunction:: genesis_gwas.data.reference.default_data_dir

.. autofunction:: genesis_gwas.data.reference.list_packaged_references

.. autodata:: genesis_gwas.data.reference.DEFAULT_REFERENCE

Downloading and verifying reference data
----------------------------------------

.. automodule:: genesis_gwas.data.download
   :no-members:

.. autofunction:: genesis_gwas.fetch_bundle

.. autofunction:: genesis_gwas.data.verify_bundle

.. autofunction:: genesis_gwas.data.download.known_bundles

.. autofunction:: genesis_gwas.data.download.load_packaged_manifest

.. autofunction:: genesis_gwas.data.download.asset_name

.. autofunction:: genesis_gwas.data.download.asset_url

.. autofunction:: genesis_gwas.data.download.ensure_file

.. autofunction:: genesis_gwas.data.download.sha256_file
