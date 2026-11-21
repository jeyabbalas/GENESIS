"""Publication-style figures for GENESIS results (matplotlib).

Every plotting function

* takes an optional ``ax`` (a new figure is created when it is omitted),
* returns the :class:`matplotlib.axes.Axes` it drew on, and
* never calls ``plt.show()`` nor touches the global pyplot state, so it is safe in batch
  jobs and on head-less cluster nodes (figures are built directly on
  :class:`matplotlib.figure.Figure`, which needs no display).

Use :func:`save_figure` (or ``ax.figure.savefig``) to write the result to disk::

    from genesis_gwas.plotting import qq_plot, save_figure

    ax = qq_plot(qqdata)
    save_figure(ax, "qq.png")
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from matplotlib.axes import Axes
from matplotlib.figure import Figure

#: Colour-blind-safe qualitative palette (Okabe & Ito, 2008), black first.
PALETTE: tuple[str, ...] = (
    "#000000",
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#F0E442",
)

#: Line styles cycled together with :data:`PALETTE` when many curves share one panel.
LINESTYLES: tuple[Any, ...] = ("-", "--", ":", "-.", (0, (5, 1, 1, 1, 1, 1)))


def new_axes(figsize: tuple[float, float] = (4.4, 4.2)) -> Axes:
    """Create a single-panel figure without pyplot and return its axes."""
    fig = Figure(figsize=figsize, layout="constrained")
    return fig.add_subplot()


def style_axes(ax: Axes) -> Axes:
    """Apply the house style: no top/right spines, outward ticks, modest font sizes."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3.5, labelsize=9)
    ax.xaxis.label.set_fontsize(10)
    ax.yaxis.label.set_fontsize(10)
    return ax


def curve_style(index: int) -> dict[str, Any]:
    """Colour and line style of the ``index``-th curve of a panel."""
    return {
        "color": PALETTE[index % len(PALETTE)],
        "linestyle": LINESTYLES[(index // len(PALETTE)) % len(LINESTYLES)],
    }


def as_labelled(fits: Any, default_label: str = "") -> list[tuple[str, Any]]:
    """Normalise ``fit | {label: fit}`` to a list of ``(label, fit)`` pairs."""
    if isinstance(fits, Mapping):
        return [(str(k), v) for k, v in fits.items()]
    return [(default_label, fits)]


def save_figure(ax_or_fig: Axes | Figure, path: str | os.PathLike[str], dpi: int = 200) -> Path:
    """Write the figure containing ``ax_or_fig`` to ``path`` (format from the extension).

    Returns the path written. Parent directories are created as needed.
    """
    fig = ax_or_fig if isinstance(ax_or_fig, Figure) else ax_or_fig.figure
    out = Path(os.fspath(path)).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi)  # type: ignore[union-attr]
    return out


# The submodules use the helpers defined above, hence the late imports.
from genesis_gwas.plotting.density import plot_effect_density  # noqa: E402
from genesis_gwas.plotting.projection import plot_auc, plot_projection  # noqa: E402
from genesis_gwas.plotting.qq import qq_plot  # noqa: E402

__all__ = [
    "LINESTYLES",
    "PALETTE",
    "as_labelled",
    "curve_style",
    "new_axes",
    "plot_auc",
    "plot_effect_density",
    "plot_projection",
    "qq_plot",
    "save_figure",
    "style_axes",
]
