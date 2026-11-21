"""GENESIS: GENetic Effect-Size distribution Inference from Summary-level data.

A ground-up reimplementation of the method of Zhang, Qi, Park & Chatterjee
(Nature Genetics 2018) with CPU and GPU execution.

Typical use::

    import genesis_gwas as gg

    ref = gg.load_reference()                       # bundled EUR reference (cached download)
    data = gg.prepare("trait.sumstats.tsv.gz", ref) # QC + merge with the LD reference
    m2 = gg.fit(data, 2)                            # two-component model
    m3 = gg.fit(data, 3, starting=m2)               # three-component model
    best = gg.select_model([m2, m3]).best
    print(best.summary())
    print(gg.project(best, n=[250_000, 500_000, 1_000_000]).to_frame())
"""

from genesis_gwas._version import __version__
from genesis_gwas.analysis import (
    effect_density,
    future_gc,
    h2_transfer,
    n_in_interval,
    polyrisk_predict,
    posterior_marginal,
    project,
    qq_simulation,
    simulate_ld_sumstats,
    simulate_sumstats,
)
from genesis_gwas.compute import ComputePlan, resolve_plan
from genesis_gwas.core.data import ModelData
from genesis_gwas.core.em import EMConfig
from genesis_gwas.core.spec import ModelSpec
from genesis_gwas.core.variance import Neighborhoods
from genesis_gwas.data import (
    LDReference,
    QCConfig,
    SumStats,
    fetch_bundle,
    load_reference,
    prepare,
    read_sumstats,
    save_reference,
)
from genesis_gwas.exceptions import GenesisError
from genesis_gwas.fit import fit
from genesis_gwas.result import FitResult, ModelComparison, select_model

__all__ = [
    "ComputePlan",
    "EMConfig",
    "FitResult",
    "GenesisError",
    "LDReference",
    "ModelComparison",
    "ModelData",
    "ModelSpec",
    "Neighborhoods",
    "QCConfig",
    "SumStats",
    "__version__",
    "effect_density",
    "fetch_bundle",
    "fit",
    "future_gc",
    "h2_transfer",
    "load_reference",
    "n_in_interval",
    "polyrisk_predict",
    "posterior_marginal",
    "prepare",
    "project",
    "qq_simulation",
    "read_sumstats",
    "resolve_plan",
    "save_reference",
    "select_model",
    "simulate_ld_sumstats",
    "simulate_sumstats",
]
