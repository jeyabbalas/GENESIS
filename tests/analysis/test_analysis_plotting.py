"""Smoke tests of the plotting functions (head-less; nothing is ever shown)."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from genesis_gwas.analysis.simulate import QQData
from genesis_gwas.exceptions import InputError
from genesis_gwas.plotting import (
    PALETTE,
    new_axes,
    plot_auc,
    plot_effect_density,
    plot_projection,
    qq_plot,
    save_figure,
)

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def assert_png(path) -> None:
    data = path.read_bytes()
    assert data[:8] == PNG_MAGIC and len(data) > 5000


@pytest.fixture()
def qqdata() -> QQData:
    rng = np.random.default_rng(0)
    obs = np.sort(-np.log10(rng.uniform(size=50000))) * 1.3
    exp = np.sort(-np.log10(rng.uniform(size=50000))) * 1.25
    return QQData(
        observed=obs,
        expected=exp,
        lower=exp * 0.95,
        upper=exp * 1.05,
        lambda_obs=1.234,
        lambda_mean=1.2,
        lambda_lower=1.1,
        lambda_upper=1.3,
        n_sim=20,
        n_snps=50000,
    )


def test_new_axes_needs_no_display_and_does_not_register_with_pyplot():
    ax = new_axes()
    assert isinstance(ax, Axes) and isinstance(ax.figure, Figure)
    assert len(PALETTE) >= 6 and all(c.startswith("#") for c in PALETTE)
    code = (
        "import sys, genesis_gwas.plotting as p; ax = p.new_axes(); "
        "import matplotlib; assert 'matplotlib.pyplot' not in sys.modules, 'pyplot imported'"
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert res.returncode == 0, res.stderr


def test_qq_plot(qqdata, tmp_path):
    ax = qq_plot(qqdata, title="Trait")
    assert isinstance(ax, Axes)
    assert ax.get_xlim() == (0.0, 10.0) and ax.get_ylim() == (0.0, 10.0)
    assert "Expected" in ax.get_xlabel() and "Observed" in ax.get_ylabel()
    assert ax.get_title() == "Trait"
    texts = " ".join(t.get_text() for t in ax.texts)
    assert "1.234" in texts and "1.200" in texts
    curve = ax.lines[-1]
    assert curve.get_xdata().size <= 20000  # thinned for plotting
    np.testing.assert_array_equal(curve.get_ydata()[-100:], qqdata.observed[-100:])
    assert_png(save_figure(ax, tmp_path / "sub" / "qq.png"))

    plain = qq_plot(qqdata, axis_max=6, annotate=False, max_points=500)
    assert plain.get_xlim() == (0.0, 6.0) and not plain.texts
    assert plain.lines[-1].get_xdata().size <= 500


def test_qq_plot_on_existing_axes_and_validation(qqdata):
    fig = Figure()
    axes = fig.subplots(1, 2)
    out = qq_plot(qqdata, ax=axes[1])
    assert out is axes[1] and not axes[0].lines
    with pytest.raises(InputError):
        qq_plot({"observed": [1, 2]})
    with pytest.raises(InputError):
        qq_plot(qqdata, axis_max=0)


def test_plot_effect_density(fit2, fit3, est3, tmp_path):
    ax = plot_effect_density({"Height M2": fit2, "Height M3": fit3})
    assert isinstance(ax, Axes) and len(ax.lines) == 2
    assert ax.get_legend() is not None
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["Height M2", "Height M3"]
    x, y = ax.lines[1].get_data()
    assert x[0] == -0.02 and x[-1] == 0.02 and np.all(y > 0)
    assert float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x))) == pytest.approx(
        0.97, abs=0.03
    )  # most of the mass
    assert_png(save_figure(ax, tmp_path / "density.png"))

    single = plot_effect_density(est3, xlim=(-0.05, 0.05), n_points=51, title="one")
    assert len(single.lines) == 1 and single.get_legend() is None
    assert single.lines[0].get_xdata().size == 51
    with pytest.raises(InputError):
        plot_effect_density(est3, xlim=(0.1, -0.1))
    with pytest.raises(InputError):
        plot_effect_density({})


@pytest.mark.parametrize("what", ["discoveries", "gv"])
def test_plot_projection(fit2, fit3, what, tmp_path):
    ax = plot_projection({"M2": fit2, "M3": fit3}, what=what)
    assert isinstance(ax, Axes) and len(ax.lines) == 2
    for line in ax.lines:
        y = line.get_ydata()
        assert np.all(np.diff(y) >= 0) and y[0] >= 0
    if what == "gv":
        assert ax.get_ylim() == (0.0, 100.0) and "%" in ax.get_ylabel()
    else:
        assert "discoveries" in ax.get_ylabel()
    assert "1,000" in ax.get_xlabel()
    assert_png(save_figure(ax, tmp_path / f"projection_{what}.png"))


def test_plot_projection_with_confidence_band(fit2, est3):
    ax = plot_projection(fit2, n_grid=np.linspace(5e4, 5e5, 12), ci=True, n_sim=200)
    assert len(ax.collections) == 1  # the shaded band
    no_cov = plot_projection(est3, n_grid=[1e5, 2e5, 4e5], ci=True)
    assert len(no_cov.collections) == 0  # raw vectors carry no covariance: curve only
    with pytest.raises(InputError):
        plot_projection(fit2, what="power")
    with pytest.raises(InputError):
        plot_projection(fit2, n_grid=[1e5])


def test_plot_auc(fit2, fit3, tmp_path):
    grid = np.logspace(4, 6, 7)
    ax = plot_auc(fit3, n_grid=grid, M=200000)
    assert isinstance(ax, Axes) and len(ax.lines) == 2 and ax.get_xscale() == "log"
    optimum, gwas = (line.get_ydata() for line in ax.lines)
    assert np.all(optimum >= gwas) and np.all(np.diff(optimum) > 0)
    lo, hi = ax.get_ylim()
    assert lo == 0.5 and hi >= optimum.max()
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert "optimal threshold" in labels[0] and "10^{-8}" in labels[1]
    assert_png(save_figure(ax, tmp_path / "auc.png"))

    both = plot_auc({"M2": fit2, "M3": fit3}, n_grid=grid)
    assert len(both.lines) == 4
    with pytest.raises(InputError):
        plot_auc(fit3, n_grid=[1e5])


def test_save_figure_accepts_figures_and_formats(fit2, tmp_path):
    ax = plot_effect_density(fit2)
    pdf = save_figure(ax.figure, tmp_path / "density.pdf")
    assert pdf.read_bytes()[:4] == b"%PDF"
    svg = save_figure(ax, tmp_path / "density.svg")
    assert b"<svg" in svg.read_bytes()[:400]


def test_plot_argument_validation(fit2):
    with pytest.raises(InputError):
        plot_effect_density(fit2, n_points=1)
    with pytest.raises(InputError):
        plot_projection({})
    with pytest.raises(InputError):
        plot_auc({})
    titled = plot_projection(fit2, n_grid=[1e5, 2e5], title="Height")
    assert titled.get_title() == "Height" and titled.get_legend() is None
    assert plot_auc(fit2, n_grid=[1e5, 2e5], title="Height").get_title() == "Height"


def test_significance_level_formatting():
    from genesis_gwas.plotting.projection import _level

    assert _level(5e-8) == r"$5\times10^{-8}$"
    assert _level(1e-5) == "$10^{-5}$"
    assert _level(1.0) == "$1$"
