"""Shared fixtures for the test suite."""

from __future__ import annotations

import pytest
from tests.helpers import simulate

from genesis_gwas.compute import resolve_plan
from genesis_gwas.core.data import ModelData


@pytest.fixture(scope="session")
def small_data() -> ModelData:
    """240 markers including the awkward cases (N* below c0, z = 0, z^2 ~ 80)."""
    return simulate(240, seed=11)


@pytest.fixture(scope="session")
def medium_data() -> ModelData:
    """20k markers simulated from a three-component truth (for EM behaviour tests)."""
    return simulate(
        20000, seed=5, pic=0.02, sig=(4e-4, 4e-5), p=(0.15, 0.85), a=1e-6, M=200000, edge=False
    )


@pytest.fixture(scope="session")
def plan1():
    """Single-threaded NumPy plan (deterministic, fastest for tiny inputs)."""
    return resolve_plan("cpu", threads=1)
