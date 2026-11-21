r"""Empirical-Bayes posterior inference for individual GWAS markers.

Given the fitted effect-size distribution, the marginal effect :math:`\beta_k` of marker
:math:`k` has a mixture prior: conditional on the latent configuration :math:`t` (the
numbers of tagged susceptibility SNPs of each type, :math:`J_t`),

.. math::

    \beta_k \mid t \sim N(0, \delta_{kt}), \qquad
    \delta_{kt} = \frac{\ell_k}{N^*_k}\sum_h J_{th}\,\sigma_h^2 ,
    \qquad \hat\beta_k \mid \beta_k \sim N(\beta_k,\; s_k^2 + a),

and configuration :math:`t=0` (no susceptibility SNP tagged) is a point mass at zero.
With posterior configuration weights :math:`w_{kt}\propto \Pr(t)\,N(\hat\beta_k;0,
\delta_{kt}+s_k^2+a)`:

.. math::

    \mathrm{lfdr}_k = \Pr(\beta_k = 0\mid\hat\beta_k) = w_{k0}, \qquad
    E[\beta_k\mid\hat\beta_k] = \hat\beta_k \sum_t w_{kt}\,
        \frac{\delta_{kt}}{\delta_{kt}+s_k^2+a}.

These are the local false-discovery rate and the shrinkage estimator of Zhang et al.
(2018; "False discovery rate and shrinkage estimation"), computed by ``postmarginal()``
in the original package.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from genesis_gwas.compute import ComputePlan, resolve_plan
from genesis_gwas.core.data import ModelData
from genesis_gwas.core.engine import Engine
from genesis_gwas.core.likelihood import CompositeLikelihood
from genesis_gwas.exceptions import InputError
from genesis_gwas.result import FitResult

__all__ = ["posterior_marginal"]


def posterior_marginal(
    data: ModelData,
    fit: FitResult,
    *,
    include_inflation: bool = True,
    c0: int | None = None,
    device: str | list[str] | tuple[str, ...] = "auto",
    threads: int | None = None,
    plan: ComputePlan | None = None,
) -> pd.DataFrame:
    r"""Local false-discovery rates and posterior-mean effect sizes for every marker.

    Parameters
    ----------
    data
        The markers to evaluate (normally the data the model was fitted to).
    fit
        Fitted two- or three-component model.
    include_inflation
        How the fitted variance-inflation term :math:`a` is treated.

        ``True`` (default)
            :math:`a` is part of the noise variance, :math:`\hat\beta_k\mid\beta_k\sim
            N(\beta_k, s_k^2+a)` — the sampling model under which the parameters were
            estimated (eq. 3 of the paper).
        ``False``
            Reproduces the original ``postmarginal()``, which silently drops :math:`a`
            (noise variance :math:`s_k^2`). When :math:`a>0` this makes the original
            over-confident for markers with real evidence of association: it reports a
            smaller lfdr and shrinks less. For example, with the height estimates of the
            original's worked example, a marker with :math:`z=3`, :math:`N^*=40`,
            :math:`\ell=15` and :math:`n=130{,}000` gets lfdr 0.24 instead of 0.40.
    c0
        Truncation level of the latent counts (default: the level at which the model was
        fitted, ``fit.c0``; the original defaults to 10 regardless of the fit).
    device, threads
        Compute options, see :func:`genesis_gwas.compute.resolve_plan`.
    plan
        A ready-made :class:`~genesis_gwas.compute.ComputePlan` (overrides ``device`` and
        ``threads``).

    Returns
    -------
    pandas.DataFrame
        One row per marker, in the order of ``data``, with columns

        ``snp``
            Marker identifier (only if ``data.snp`` is available).
        ``z``
            Z statistic :math:`\hat\beta_k/s_k`.
        ``p_null``
            Local false-discovery rate :math:`\Pr(\beta_k=0\mid\hat\beta_k)` (``prob.beta0``
            in the original).
        ``post_mean``
            Posterior mean :math:`E[\beta_k\mid\hat\beta_k]` of the marginal effect
            (``posteriormean.marginal`` in the original).
        ``post_sd``
            Posterior standard deviation of the marginal effect.
        ``expected_count``
            Posterior expected number of susceptibility SNPs tagged by the marker,
            :math:`\sum_t w_{kt}\,|J_t|`.
    """
    if not isinstance(fit, FitResult):
        raise InputError("fit must be a FitResult")
    level = int(fit.c0 if c0 is None else c0)
    if level < 1:
        raise InputError("c0 must be >= 1")
    if plan is None:
        plan = resolve_plan(device, threads=threads)
    with Engine(data, plan) as engine:
        lik = CompositeLikelihood(fit.spec, engine, level)
        post = lik.posterior(
            np.asarray(fit.estimates, dtype=np.float64), include_inflation=include_inflation
        )
    columns: dict[str, np.ndarray] = {}
    if data.snp is not None:
        columns["snp"] = np.asarray(data.snp)
    columns["z"] = data.z
    columns["p_null"] = post.p_null
    columns["post_mean"] = post.post_mean
    columns["post_sd"] = post.post_sd
    columns["expected_count"] = post.expected_count
    return pd.DataFrame(columns)
