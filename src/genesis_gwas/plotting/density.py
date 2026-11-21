"""Plot of the estimated effect-size distribution of susceptibility SNPs."""

from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.axes import Axes

from genesis_gwas.analysis.density import effect_density
from genesis_gwas.exceptions import InputError
from genesis_gwas.plotting import as_labelled, curve_style, new_axes, style_axes

__all__ = ["plot_effect_density"]


def plot_effect_density(
    fits: Any,
    ax: Axes | None = None,
    *,
    xlim: tuple[float, float] = (-0.02, 0.02),
    n_points: int = 1001,
    title: str | None = None,
) -> Axes:
    r"""Plot the density of joint effect sizes of susceptibility SNPs (the paper's Fig. 2).

    Each curve is :math:`f(\beta)=\sum_h p_h N(\beta;0,\sigma_h^2)` of one fitted model,
    evaluated with :func:`genesis_gwas.analysis.density.effect_density`. Fatter tails
    mean relatively more susceptibility SNPs with large effects.

    Parameters
    ----------
    fits
        A :class:`~genesis_gwas.result.FitResult` (or parameter vector), or a mapping
        ``{label: fit}`` to overlay several traits or models.
    ax
        Axes to draw on; a new figure is created if omitted.
    xlim
        Range of effect sizes (standardised scale; default ``(-0.02, 0.02)`` as for the
        quantitative traits in the paper).
    n_points
        Number of evaluation points.
    title
        Optional panel title.

    Returns
    -------
    matplotlib.axes.Axes
    """
    lo, hi = float(xlim[0]), float(xlim[1])
    if not lo < hi:
        raise InputError("xlim must be increasing")
    if n_points < 2:
        raise InputError("n_points must be >= 2")
    items = as_labelled(fits)
    if not items:
        raise InputError("No fits to plot")
    x = np.linspace(lo, hi, int(n_points))
    ax = new_axes((5.2, 3.6)) if ax is None else ax
    for i, (label, fit) in enumerate(items):
        ax.plot(x, effect_density(x, fit), linewidth=1.4, label=label or None, **curve_style(i))
    ax.set_xlim(lo, hi)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Joint effect size (per SD of genotype)")
    ax.set_ylabel("Probability density")
    if title:
        ax.set_title(title, fontsize=10)
    if any(label for label, _ in items):
        ax.legend(frameon=False, fontsize=8)
    return style_axes(ax)
