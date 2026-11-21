"""Downstream analyses of a fitted effect-size distribution.

===================================  ==================================================
:func:`effect_density`               density of joint effect sizes of susceptibility SNPs
:func:`n_in_interval`                number of SNPs with effects in a range
:func:`project`                      discoveries / variance explained in future GWAS
:func:`polyrisk_predict`             expected performance of polygenic risk scores
:func:`posterior_marginal`           local false-discovery rates and shrunken effects
:func:`qq_simulation`                model-based QQ plot with simulation envelope
:func:`future_gc`                    genomic-control factor expected at a sample size
:func:`simulate_sumstats`            draw summary statistics from the fitted model
:func:`simulate_ld_sumstats`         ... with realistic linkage disequilibrium
:func:`h2_transfer`                  heritability on the observed / liability scale
===================================  ==================================================
"""

from genesis_gwas.analysis.density import effect_cdf, effect_density, effect_mixture, n_in_interval
from genesis_gwas.analysis.heritability import h2_transfer
from genesis_gwas.analysis.posterior import posterior_marginal
from genesis_gwas.analysis.prediction import PolygenicPrediction, polyrisk_predict
from genesis_gwas.analysis.projection import Projection, project
from genesis_gwas.analysis.simulate import (
    QQData,
    future_gc,
    lambda_gc,
    qq_simulation,
    simulate_ld_sumstats,
    simulate_sumstats,
)

__all__ = [
    "PolygenicPrediction",
    "Projection",
    "QQData",
    "effect_cdf",
    "effect_density",
    "effect_mixture",
    "future_gc",
    "h2_transfer",
    "lambda_gc",
    "n_in_interval",
    "polyrisk_predict",
    "posterior_marginal",
    "project",
    "qq_simulation",
    "simulate_ld_sumstats",
    "simulate_sumstats",
]
