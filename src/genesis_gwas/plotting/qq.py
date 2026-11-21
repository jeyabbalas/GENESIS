"""Quantile–quantile plot of observed P-values against the fitted model."""

from __future__ import annotations

from matplotlib.axes import Axes

from genesis_gwas.analysis.simulate import QQData
from genesis_gwas.exceptions import InputError
from genesis_gwas.plotting import new_axes, style_axes

__all__ = ["qq_plot"]


def qq_plot(
    qqdata: QQData,
    ax: Axes | None = None,
    *,
    axis_max: float = 10.0,
    max_points: int = 20000,
    title: str | None = None,
    annotate: bool = True,
    line_color: str = "#000000",
    band_color: str = "#BFBFBF",
) -> Axes:
    r"""Draw the model-based QQ plot of a fit (the paper's Fig. 1; ``qqplot.plot()``).

    The y axis shows the observed :math:`-\log_{10}P` values, the x axis the values
    *expected under the fitted effect-size model* (mean over simulations), so a
    well-fitting model follows the diagonal even when the test statistics are strongly
    inflated by polygenicity. The shaded band is the point-wise simulation envelope
    (drawn horizontally, i.e. as a range of expected values for each observed order
    statistic, as in the original).

    Parameters
    ----------
    qqdata
        Output of :func:`genesis_gwas.analysis.simulate.qq_simulation`.
    ax
        Axes to draw on; a new figure is created if omitted.
    axis_max
        Upper limit of both axes (default 10, as in the paper).
    max_points
        The curves are thinned to at most this many points (the most significant half is
        always kept exactly).
    title
        Optional panel title (e.g. the trait).
    annotate
        Print the observed genomic-control factor :math:`\lambda_{obs}` and the mean
        simulated one :math:`\lambda_{fit}` in the lower right corner.
    line_color, band_color
        Colours of the curve and of the envelope.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if not isinstance(qqdata, QQData):
        raise InputError("qq_plot expects the QQData returned by qq_simulation()")
    if not axis_max > 0:
        raise InputError("axis_max must be > 0")
    q = qqdata.thin(max_points) if qqdata.observed.size > max_points else qqdata
    ax = new_axes((4.2, 4.2)) if ax is None else ax
    ax.fill_betweenx(q.observed, q.lower, q.upper, color=band_color, linewidth=0, zorder=1)
    ax.plot([0, axis_max], [0, axis_max], color="#7F7F7F", linestyle="--", linewidth=0.9, zorder=2)
    ax.plot(q.expected, q.observed, color=line_color, linewidth=1.2, zorder=3)
    ax.set_xlim(0, axis_max)
    ax.set_ylim(0, axis_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"Expected $-\log_{10}(P)$ under the fitted model")
    ax.set_ylabel(r"Observed $-\log_{10}(P)$")
    if title:
        ax.set_title(title, fontsize=10)
    if annotate:
        ax.text(
            0.97,
            0.05,
            rf"$\lambda_{{obs}}$ = {qqdata.lambda_obs:.3f}"
            "\n"
            rf"$\lambda_{{fit}}$ = {qqdata.lambda_mean:.3f}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            linespacing=1.5,
        )
    return style_axes(ax)
