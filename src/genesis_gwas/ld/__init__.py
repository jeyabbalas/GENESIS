"""Building LD reference bundles from genotypes.

Use :func:`build_reference` to create a reference for any population from PLINK
genotypes, and :func:`vcf_to_plink` to obtain those from VCF files such as the 1000
Genomes Project's; :func:`compute_ld` and :func:`null_z` are the underlying computations
and :class:`~genesis_gwas.ld.plink.BedReader` a dependency-free PLINK ``.bed`` reader.
"""

from genesis_gwas.ld.build import (
    LDTable,
    allele_frequencies,
    build_reference,
    compute_ld,
    null_z,
    standardize_genotypes,
)
from genesis_gwas.ld.plink import BedReader, read_bim, read_fam, resolve_fileset
from genesis_gwas.ld.vcf import vcf_to_plink

__all__ = [
    "BedReader",
    "LDTable",
    "allele_frequencies",
    "build_reference",
    "compute_ld",
    "null_z",
    "read_bim",
    "read_fam",
    "resolve_fileset",
    "standardize_genotypes",
    "vcf_to_plink",
]
