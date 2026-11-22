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
