"""Plots of projected GWAS yield and of polygenic-prediction performance."""

from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.ticker import FuncFormatter

from genesis_gwas.analysis.prediction import polyrisk_predict
from genesis_gwas.analysis.projection import project
from genesis_gwas.exceptions import InputError
from genesis_gwas.plotting import PALETTE, as_labelled, curve_style, new_axes, style_axes
from genesis_gwas.result import FitResult

__all__ = ["plot_auc", "plot_projection"]


def _thousands(value: float, _pos: int) -> str:
    return f"{value / 1000.0:,.0f}"


def _grouped(value: float, _pos: int) -> str:
    return f"{value:,.0f}"


def _level(alpha: float) -> str:
    r"""Format a significance level as mathtext, e.g. ``5e-8`` -> ``$5\times10^{-8}$``."""
    mantissa, exponent = f"{alpha:.3e}".split("e")
    m = f"{float(mantissa):g}"
    e = int(exponent)
    if e == 0:
        return f"${m}$"
    return f"$10^{{{e}}}$" if m == "1" else rf"${m}\times10^{{{e}}}$"


def plot_projection(
    fits: Any,
    ax: Axes | None = None,
    *,
    what: str = "discoveries",
    n_grid: Any = None,
    alpha: float = 5e-8,
    M: int | None = None,
    ci: bool = False,
    coverage: float = 0.95,
    n_sim: int = 1000,
    seed: int | None = 123,
    title: str | None = None,
) -> Axes:
    r"""Plot projected discoveries or heritability explained against sample size (Fig. 3).

    Parameters
    ----------
    fits
        A :class:`~genesis_gwas.result.FitResult` (or parameter vector), or a mapping
        ``{label: fit}``.
    ax
        Axes to draw on; a new figure is created if omitted.
    what
        ``"discoveries"`` — expected number of susceptibility SNPs reaching significance;
        ``"gv"`` — percentage of GWAS heritability they explain.
    n_grid
        Sample sizes (default: 200 points from 10,000 to 1,000,000). For disease traits
        with a 1:1 case–control ratio the total number of subjects is four times the
        effective sample size used here.
    alpha
        Significance level of a discovery (default :math:`5\times10^{-8}`).
    M
        Number of reference SNPs (default: that of each fit).
    ci
        Shade simulation-based confidence bands (needs the covariance matrix, i.e.
        :class:`~genesis_gwas.result.FitResult` inputs with standard errors).
    coverage, n_sim, seed
        Settings of the confidence bands; see :func:`genesis_gwas.analysis.projection.project`.
    title
        Optional panel title.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if what not in ("discoveries", "gv"):
        raise InputError("what must be 'discoveries' or 'gv'")
    grid = np.linspace(1e4, 1e6, 200) if n_grid is None else np.asarray(n_grid, dtype=float).ravel()
    if grid.size < 2:
        raise InputError("n_grid needs at least two sample sizes")
    items = as_labelled(fits)
    if not items:
        raise InputError("No fits to plot")
    ax = new_axes((4.8, 3.8)) if ax is None else ax
    for i, (label, fit) in enumerate(items):
        with_ci = ci and isinstance(fit, FitResult) and fit.covariance is not None
        proj = project(
            fit, grid, alpha=alpha, M=M, ci=with_ci, coverage=coverage, n_sim=n_sim, seed=seed
        )
        style = curve_style(i)
        if what == "discoveries":
            y, lo, hi = proj.n_discoveries, proj.n_discoveries_lower, proj.n_discoveries_upper
        else:
            y, lo, hi = proj.gv_percentage, proj.gv_percentage_lower, proj.gv_percentage_upper
        if with_ci and lo is not None and hi is not None:
            ax.fill_between(grid, lo, hi, color=style["color"], alpha=0.18, linewidth=0)
        ax.plot(grid, y, linewidth=1.4, label=label or None, **style)
    ax.set_xlim(0, float(np.max(grid)))
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_formatter(FuncFormatter(_thousands))
    ax.set_xlabel("Sample size (in 1,000)")
    if what == "discoveries":
        ax.set_ylabel(f"Expected number of discoveries ($P$ < {_level(alpha)})")
        ax.yaxis.set_major_formatter(FuncFormatter(_grouped))
    else:
        ax.set_ylabel("GWAS heritability explained (%)")
        ax.set_ylim(0, 100)
    if title:
        ax.set_title(title, fontsize=10)
    if any(label for label, _ in items):
        ax.legend(frameon=False, fontsize=8)
    return style_axes(ax)


def plot_auc(
    fits: Any,
    ax: Axes | None = None,
    *,
    n_grid: Any = None,
    M: float | None = None,
    alpha_gwas: float = 5e-8,
    title: str | None = None,
) -> Axes:
    r"""Plot the expected AUC of polygenic risk scores against GWAS sample size (Fig. 4).

    For every fit two curves are drawn: SNPs included at the **optimal** significance
    threshold (solid) and at the **genome-wide** level ``alpha_gwas`` (dashed), both from
    :func:`genesis_gwas.analysis.prediction.polyrisk_predict`.

    Parameters
    ----------
    fits
        A :class:`~genesis_gwas.result.FitResult` (or parameter vector), or a mapping
        ``{label: fit}``.
    ax
        Axes to draw on; a new figure is created if omitted.
    n_grid
        Sample sizes (default: 25 log-spaced points from 10,000 to 1,000,000).
    M
        Number of independent SNPs assumed by the prediction theory (default: ``M`` of
        each fit; the paper uses 200,000).
    alpha_gwas
        Genome-wide significance level of the dashed curves.
    title
        Optional panel title.

    Returns
    -------
    matplotlib.axes.Axes
    """
    grid = np.logspace(4, 6, 25) if n_grid is None else np.asarray(n_grid, dtype=float).ravel()
    if grid.size < 2:
        raise InputError("n_grid needs at least two sample sizes")
    items = as_labelled(fits)
    if not items:
        raise InputError("No fits to plot")
    single = len(items) == 1
    ax = new_axes((4.8, 3.8)) if ax is None else ax
    top = 0.5
    for i, (label, fit) in enumerate(items):
        # A single fit is drawn as in the paper: optimum in vermilion, GWAS level in black.
        color = PALETTE[2] if single else PALETTE[(i + 1) % len(PALETTE)]
        opt = polyrisk_predict(grid, fit, M=M, threshold="optimum")
        gws = polyrisk_predict(grid, fit, M=M, threshold="gwas", alpha_gwas=alpha_gwas)
        prefix = f"{label}: " if label else ""
        ax.plot(grid, opt.auc, color=color, linewidth=1.5, label=f"{prefix}optimal threshold")
        ax.plot(
            grid,
            gws.auc,
            color=PALETTE[0] if single else color,
            linewidth=1.5,
            linestyle="-" if single else "--",
            label=f"{prefix}$P$ < {_level(alpha_gwas)}",
        )
        top = max(top, float(np.max(opt.auc)), float(np.max(gws.auc)))
    ax.set_xscale("log")
    ax.set_xlim(float(np.min(grid)), float(np.max(grid)))
    ax.set_ylim(0.5, min(1.0, max(0.7, float(np.ceil((top + 0.02) * 20.0) / 20.0))))
    ax.xaxis.set_major_formatter(FuncFormatter(_grouped))
    ax.set_xlabel("GWAS sample size")
    ax.set_ylabel("Expected AUC")
    if title:
        ax.set_title(title, fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    return style_axes(ax)
