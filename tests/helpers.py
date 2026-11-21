"""Helper functions shared by the tests (importable as ``tests.helpers``)."""

from __future__ import annotations

import gzip
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from genesis_gwas.core.data import ModelData

FIXTURES = Path(__file__).parent / "fixtures" / "upstream"


def _decode(x: Any) -> Any:
    """Upstream fixtures encode non-finite doubles as strings."""
    if isinstance(x, dict):
        return {k: _decode(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_decode(v) for v in x]
    if isinstance(x, str):
        return {"NaN": math.nan, "NA": math.nan, "Inf": math.inf, "-Inf": -math.inf}.get(x, x)
    return x


def load_fixture(name: str) -> dict[str, Any]:
    """Load an upstream fixture (``.json`` or ``.json.gz``); skip the test if it is absent."""
    for cand in (FIXTURES / name, FIXTURES / (name + ".gz")):
        if cand.exists():
            opener = gzip.open if cand.suffix == ".gz" else open
            with opener(cand, "rt") as fh:
                return _decode(json.load(fh))
    pytest.skip(f"upstream fixture {name} not present")


def simulate(
    K: int, seed: int, *, pic=0.01, sig=(5e-5,), p=(1.0,), a=1e-6, M=20000, edge=True
) -> ModelData:
    """Simulate from the fitted-model form (independent markers)."""
    rng = np.random.default_rng(seed)
    nstar = np.clip(rng.gamma(1.6, 29.0, K).astype(np.int64), 1, 700)
    if edge:
        head = np.array([1, 1, 2, 3, 5, 9, 10, 11, 12, 20, 21, 700])
        nstar[: head.size] = head
    ld = np.maximum(1.0, nstar * rng.uniform(0.15, 0.7, K))
    ld[nstar == 1] = 1.0
    n = rng.uniform(5e4, 1.5e5, K)
    probs = np.array([pic * q for q in p] + [1 - pic])
    counts = np.stack([rng.multinomial(int(m), probs) for m in nstar])[:, :-1]
    var = (counts @ np.asarray(sig)) * ld / nstar + 1.0 / n + a
    b = rng.normal(0.0, np.sqrt(var))
    if edge:
        b[0] = 0.0
        b[1] = math.sqrt(79.9 / n[1])
        b[2] = -math.sqrt(25.0 / n[2])
    return ModelData(betahat=b, varbetahat=1.0 / n, ldscore=ld, nstar=nstar, n=n, M=M)
