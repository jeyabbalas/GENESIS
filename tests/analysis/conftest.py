"""Fixtures shared by the downstream-analysis tests."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

import genesis_gwas as gg
from genesis_gwas.core.derived import derived_quantities
from genesis_gwas.core.spec import ModelSpec
from genesis_gwas.result import FitResult

# Parameter estimates and covariance matrices printed in upstream's examples/examples.md
# (height GWAS, Lango Allen et al. 2010).
EST2 = np.array([4.906797e-03, 5.733005e-05, 1.861491e-06])
V2 = np.array(
    [
        [2.870170e-07, -2.426575e-09, -6.077209e-11],
        [-2.426575e-09, 2.491765e-11, 4.715283e-13],
        [-6.077209e-11, 4.715283e-13, 1.720162e-14],
    ]
)
EST3 = np.array([8.899809e-03, 9.476025e-02, 1.458650e-04, 2.227118e-05, 1.567643e-06])
# upstream prints this matrix column by column; it is asymmetric (see the fixtures README)
V3 = np.array(
    [
        1.327856e-06, -1.131049e-05, 6.912489e-10, -2.901301e-09, -9.388865e-11,
        -8.568269e-06, 3.380985e-04, -1.543479e-07, 1.771036e-09, 8.989668e-10,
        -1.846216e-09, -1.353542e-07, 1.517545e-10, 1.166563e-11, 1.901686e-14,
        -3.068113e-09, 9.520417e-09, 5.633801e-12, 9.427492e-12, 1.220433e-13,
        -9.111521e-11, 1.063848e-09, -1.526281e-13, 1.042123e-13, 1.406410e-14,
    ]
).reshape(5, 5, order="F")  # fmt: skip


def build_fit(
    theta: Any,
    *,
    M: int = 1_070_777,
    c0: int = 10,
    covariance: Any = None,
    stratification: bool = True,
) -> FitResult:
    """A FitResult carrying given parameter values (no fitting involved)."""
    th = np.asarray(theta, dtype=float)
    spec = ModelSpec(2 if th.size == 3 else 3, stratification)
    cov = None if covariance is None else np.asarray(covariance, dtype=float)
    se = np.full(th.size, np.nan) if cov is None else np.sqrt(np.abs(np.diag(cov)))
    return FitResult(
        n_components=spec.n_components,
        stratification=stratification,
        param_names=spec.param_names,
        estimates=th,
        se=se,
        covariance=cov,
        loglik=0.0,
        loglik_unnorm=0.0,
        c0=c0,
        n_iter=0,
        converged=True,
        status="converged",
        n_snps=0,
        M=M,
        mean_n=1.0,
        derived=derived_quantities(spec, th, cov, M),
    )


@pytest.fixture(scope="session")
def make_fit() -> Callable[..., FitResult]:
    """Factory for FitResult objects with prescribed parameters."""
    return build_fit


@pytest.fixture(scope="session")
def est2() -> np.ndarray:
    return EST2.copy()


@pytest.fixture(scope="session")
def est3() -> np.ndarray:
    return EST3.copy()


@pytest.fixture(scope="session")
def cov2() -> np.ndarray:
    """Covariance matrix of EST2 as printed by upstream."""
    return V2.copy()


@pytest.fixture(scope="session")
def cov3_raw() -> np.ndarray:
    """Covariance matrix of EST3 exactly as printed by upstream (asymmetric)."""
    return V3.copy()


@pytest.fixture(scope="session")
def fit2() -> FitResult:
    """Two-component height estimates of upstream's worked example, with covariance."""
    return build_fit(EST2, covariance=V2)


@pytest.fixture(scope="session")
def fit3() -> FitResult:
    """Three-component height estimates of upstream's worked example (symmetrised covariance)."""
    return build_fit(EST3, covariance=0.5 * (V3 + V3.T))


@pytest.fixture(scope="session")
def mini_ref() -> gg.LDReference:
    """Packaged chromosome-22 reference (pairwise r2 and 10 null replicates included)."""
    return gg.load_reference("mini_chr22")


@pytest.fixture(scope="session")
def mini_data(mini_ref: gg.LDReference) -> gg.ModelData:
    """Height GWAS on chromosome 22, merged with the mini reference (no filtering)."""
    return gg.prepare(mini_ref.example_path(), mini_ref, qc=False)


@pytest.fixture(scope="session")
def mini_fit(mini_data: gg.ModelData) -> FitResult:
    """A real two-component fit to the chromosome-22 data."""
    return gg.fit(mini_data, 2, threads=2)
