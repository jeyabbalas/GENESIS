"""Full-scale parity with the original software on the GIANT height GWAS (941,684 SNPs).

Needs the complete European reference bundle, so these tests are skipped unless
``GENESIS_DATA_DIR`` points at a directory that holds it (``genesis data fetch
eur_1kg_hm3 --components default,examples``). The fixtures are summaries of genuine runs
of the original R/C++ package (``tools/upstream_oracle``); the two-component run is the
worked example of the original ``examples/examples.md``.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pytest

import genesis_gwas as gg
from tests.helpers import load_fixture

pytestmark = [pytest.mark.refdata]


@pytest.fixture(scope="module")
def height():
    if not os.environ.get("GENESIS_DATA_DIR"):
        pytest.skip("set GENESIS_DATA_DIR to a directory holding the eur_1kg_hm3 bundle")
    try:
        ref = gg.load_reference("eur_1kg_hm3", download=False)
        path = ref.example_path()
        if not path.exists():
            raise FileNotFoundError(path)
    except Exception as exc:
        pytest.skip(f"full reference bundle not usable offline: {exc}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return gg.prepare(path, ref, qc=False)  # the original example uses filter=FALSE


def _rel(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.max(np.abs(a - b) / np.abs(b)))


def test_two_component_legacy_reproduces_the_original_example(height):
    fx = load_fixture("height_full_m2.json")
    assert height.K == fx["K"] == 941684
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = gg.fit(height, 2, starting_pic=0.005, legacy=True)
    assert res.n_iter == int(fx["fit"][0]) + 1 == 117 and res.c0 == fx["c0_final"] == 10
    assert _rel(res.starting, fx["starting"]) < 1e-9
    assert _rel(res.estimates, fx["est"]) < 1e-9
    assert _rel(res.loglik, fx["llk"]) < 1e-12
    assert _rel(res.information, fx["m_I"]) < 1e-8
    assert _rel(res.J, fx["J"]) < 1e-8
    assert _rel(res.covariance, fx["var_est"]) < 1e-7
    assert _rel(res.se, fx["sd_est"]) < 1e-8
    assert _rel(res.ds, fx["ds"]) < 1e-8 and _rel(res.bic, fx["bic"]) < 1e-10
    d = res.derived
    assert (
        _rel(d["n_ssnp"].estimate, fx["causalnum"]) < 1e-9
        and _rel(d["n_ssnp"].se, fx["sd_causalnum"]) < 1e-8
    )
    assert _rel(d["heritability"].estimate, fx["heritability"]) < 1e-9
    assert _rel(d["heritability"].se, fx["sd_heritability"]) < 1e-8
    assert _rel(d["sibling_risk"].se, fx["sd_risk"]) < 1e-8
    # ... and therefore the numbers printed in the original's examples.md (7 digits, x86 build)
    assert _rel(res.estimates, fx["examples_md"]["est"]) < 2e-5


def test_two_component_default_reaches_the_maximiser(height):
    """Default mode: same model, but the estimate is the stationary point, not the EM stop."""
    fx = load_fixture("height_full_m2.json")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = gg.fit(height, 2, starting_pic=0.005)
    assert res.converged and res.diagnostics["polish"]["status"] == "converged"
    np.testing.assert_allclose(res.estimates, [4.839258e-03, 5.793067e-05, 1.879024e-06], rtol=2e-6)
    assert np.max(np.abs(res.diagnostics["newton_distance_se"])) < 1e-6
    assert res.loglik > fx["llk"]  # higher likelihood than the original's stopping point
    shift = (res.estimates - np.asarray(fx["est"])) / np.asarray(fx["sd_est"])
    assert 0.05 < np.max(np.abs(shift)) < 0.25  # the original stops ~0.13 SE short
    # a different start reaches the same optimum (start-independent, unlike the original)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        other = gg.fit(height, 2, starting_pic=0.02, variance=False)
    np.testing.assert_allclose(other.estimates, res.estimates, rtol=1e-6)


@pytest.mark.slow
def test_three_component_legacy_reproduces_the_original_example(height):
    """849 EM iterations; the asymmetric covariance of the original is reproduced as well."""
    fx = load_fixture("height_full_m3.json")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = gg.fit(height, 3, starting=np.asarray(fx["starting"]), legacy=True)
    assert res.n_iter == int(fx["fit"][0]) + 1 == 849 and res.c0 == fx["c0_final"] == 10
    assert _rel(res.estimates, fx["est"]) < 1e-8
    assert _rel(res.loglik, fx["llk"]) < 1e-11
    assert _rel(res.information, fx["m_I"]) < 1e-7
    cov, ref_cov = res.covariance, np.asarray(fx["var_est"])
    assert _rel(cov, ref_cov) < 1e-6
    assert (
        abs(ref_cov[0, 1] - ref_cov[1, 0]) > 1e-7 and abs(cov[0, 1] - cov[1, 0]) > 1e-7
    )  # asymmetric
    assert _rel(res.se, fx["sd_est"]) < 1e-7 and _rel(res.bic, fx["bic"]) < 1e-10
    assert _rel(res.derived["n_ssnp_cluster1"].se, fx["sd_largenum"]) < 1e-6
    assert _rel(res.estimates, fx["examples_md"]["est"]) < 5e-7


# ---------------------------------------------------------------------------
# East Asian reference: parity with the GENESISasian R package
# ---------------------------------------------------------------------------
_EAS_SIM_MD5 = "37ab848c1d1b069cf180612645dec2b3"  # of the uncompressed table of make_eas_sim.py


@pytest.fixture(scope="module")
def eas_sim(tmp_path_factory):
    """The simulated East Asian GWAS of the ``eas_sim_m*.json`` fixtures, matched to ``eas_1kg_hm3``.

    The bundle is taken from ``GENESIS_EAS_BUNDLE`` (a bundle directory) or, failing that,
    from ``GENESIS_DATA_DIR`` by name. The GWAS is regenerated with
    ``tools/upstream_oracle/make_eas_sim.py`` (or read from ``GENESIS_EAS_SIM``) and must
    have the checksum of the file the original software was run on.
    """
    import hashlib
    import importlib.util
    from pathlib import Path

    bundle = os.environ.get("GENESIS_EAS_BUNDLE")
    if not bundle and not os.environ.get("GENESIS_DATA_DIR"):
        pytest.skip(
            "set GENESIS_EAS_BUNDLE (bundle directory) or GENESIS_DATA_DIR holding eas_1kg_hm3"
        )
    try:
        ref = gg.load_reference(bundle or "eas_1kg_hm3", download=False)
        ref.null_z(0)
        if ref.config.pairld is None:  # the simulation needs the pairwise LD
            raise FileNotFoundError("no pairwise LD in the bundle")
    except Exception as exc:
        pytest.skip(
            f"eas_1kg_hm3 bundle (with null replicates and pairwise LD) not usable offline: {exc}"
        )
    path = os.environ.get("GENESIS_EAS_SIM")
    if not path:
        script = (
            Path(__file__).resolve().parents[1] / "tools" / "upstream_oracle" / "make_eas_sim.py"
        )
        spec = importlib.util.spec_from_file_location("make_eas_sim", script)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        path = tmp_path_factory.mktemp("eas") / "eas_sim.tsv"
        mod.write(mod.simulate(ref.root)[1], path)
        if hashlib.md5(Path(path).read_bytes()).hexdigest() != _EAS_SIM_MD5:
            pytest.skip(
                "the regenerated simulation differs from the file the original was run on "
                "(platform-dependent rounding); point GENESIS_EAS_SIM at the original file"
            )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return gg.prepare(path, ref, qc=False)


def test_east_asian_two_component_legacy_reproduces_genesisasian(eas_sim):
    """All defaults on both sides: M = 953,834 comes from the reference, LDSC start with pic 0.01."""
    fx = load_fixture("eas_sim_m2.json")
    assert eas_sim.K == fx["K"] == 858450 and eas_sim.M == fx["M_used"] == 953834
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = gg.fit(eas_sim, 2, legacy=True)
    assert res.M == 953834
    assert res.n_iter == int(fx["fit"][0]) + 1 and res.c0 == fx["c0_final"]
    assert _rel(res.starting, fx["starting"]) < 1e-9
    assert _rel(res.estimates, fx["est"]) < 1e-9
    assert _rel(res.loglik, fx["llk"]) < 1e-12
    assert _rel(res.information, fx["m_I"]) < 1e-8 and _rel(res.J, fx["J"]) < 1e-8
    assert _rel(res.covariance, fx["var_est"]) < 1e-7 and _rel(res.se, fx["sd_est"]) < 1e-8
    assert _rel(res.ds, fx["ds"]) < 1e-8 and _rel(res.bic, fx["bic"]) < 1e-10
    d = res.derived
    assert _rel(d["n_ssnp"].estimate, fx["causalnum"]) < 1e-9
    assert _rel(d["n_ssnp"].se, fx["sd_causalnum"]) < 1e-8
    assert _rel(d["heritability"].estimate, fx["heritability"]) < 1e-9
    assert _rel(d["heritability"].se, fx["sd_heritability"]) < 1e-8
    # downstream functions take M from the fit, as GENESISasian's defaults do
    proj = gg.project(res, 300000)
    assert proj.M == 953834
    assert _rel(proj.n_discoveries, fx["projection_n300000"]["Numdiscoveries"]) < 1e-8
    assert _rel(proj.gv_percentage, fx["projection_n300000"]["GVpercentage"]) < 1e-8
    assert _rel(gg.n_in_interval(0.005, np.inf, res), fx["numInterval_0.005_Inf"]) < 1e-8


@pytest.mark.slow
def test_east_asian_three_component_legacy_reproduces_genesisasian(eas_sim):
    """Default start of the original: an internal two-component fit, then the EM for M3."""
    fx = load_fixture("eas_sim_m3.json")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = gg.fit(eas_sim, 3, legacy=True)
    assert res.M == fx["M_used"] == 953834
    assert res.n_iter == int(fx["fit"][0]) + 1 and res.c0 == fx["c0_final"]
    assert _rel(res.starting, fx["starting"]) < 1e-7
    assert _rel(res.estimates, fx["est"]) < 1e-7
    assert _rel(res.loglik, fx["llk"]) < 1e-11
    assert _rel(res.information, fx["m_I"]) < 1e-6
    assert _rel(res.covariance, fx["var_est"]) < 1e-5 and _rel(res.se, fx["sd_est"]) < 1e-6
    assert _rel(res.bic, fx["bic"]) < 1e-10
    assert _rel(res.derived["heritability"].estimate, fx["heritability"]) < 1e-7
