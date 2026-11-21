"""Publication-style figures for GENESIS results (matplotlib).

Every plotting function

* takes an optional ``ax`` (a new figure is created when it is omitted),
* returns the :class:`matplotlib.axes.Axes` it drew on, and
* never calls ``plt.show()`` or touches the global pyplot state, so it is safe in batch
  jobs and on head-less cluster nodes.

Use :func:`save_figure` (or ``ax.figure.savefig``) to write the result to disk.
"""

from genesis_gwas.plotting._common import PALETTE, new_axes, save_figure
from genesis_gwas.plotting.density import plot_effect_density
from genesis_gwas.plotting.projection import plot_auc, plot_projection
from genesis_gwas.plotting.qq import qq_plot

__all__ = [
    "PALETTE",
    "new_axes",
    "plot_auc",
    "plot_effect_density",
    "plot_projection",
    "qq_plot",
    "save_figure",
]
