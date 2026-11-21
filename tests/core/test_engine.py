"""The compute engine: backends, threads, chunking, shards and planning."""

from __future__ import annotations

import numpy as np
import pytest

from genesis_gwas.backends import get_backend, torch_available
from genesis_gwas.compute import available_cpus, resolve_plan
from genesis_gwas.core.engine import Engine
from genesis_gwas.core.likelihood import CompositeLikelihood
from genesis_gwas.core.spec import ModelSpec
from genesis_gwas.exceptions import BackendError, InputError
from tests.helpers import simulate

needs_torch = pytest.mark.skipif(not torch_available(), reason="PyTorch not installed")
TH3 = np.array([0.012, 0.2, 2e-4, 2e-5, 1.2e-6])
TH2 = np.array([0.012, 6e-5, 1.2e-6])


@pytest.fixture(scope="module")
def data():
    return simulate(3000, seed=21)


def _everything(data, plan, spec, theta, c0=10):
    with Engine(data, plan) as eng:
        lik = CompositeLikelihood(spec, eng, c0)
        em = lik.em_pass(theta)
        si = lik.score_info(theta)
        po = lik.posterior(theta)
    return {
        "loglik": np.array([em.loglik, em.loglik_unnorm]),
        "counts": em.counts,
        "grad": em.grad,
        "hess": em.hess,
        "scores": si.scores,
        "info": si.information,
        "p_null": po.p_null,
        "post_mean": po.post_mean,
        "post_sd": po.post_sd,
    }


def _assert_close(a, b, rtol):
    for key in a:
        np.testing.assert_allclose(a[key], b[key], rtol=rtol, atol=1e-11, err_msg=key)


@pytest.mark.parametrize("spec,theta", [(ModelSpec(2), TH2), (ModelSpec(3), TH3)], ids=["M2", "M3"])
def test_results_do_not_depend_on_threads_or_chunking(data, spec, theta):
    base = _everything(data, resolve_plan("cpu", threads=1), spec, theta)
    # same chunking, different thread counts: bit-identical
    a = _everything(data, resolve_plan("cpu", threads=1, chunk_size=257), spec, theta)
    b = _everything(data, resolve_plan("cpu", threads=5, chunk_size=257), spec, theta)
    for key in a:
        np.testing.assert_array_equal(a[key], b[key], err_msg=key)
    # different chunking: equal up to floating-point reassociation of the sums
    _assert_close(a, base, rtol=1e-10)
    _assert_close(
        _everything(data, resolve_plan("cpu", threads=3, chunk_size=1), spec, theta), base, 1e-10
    )
    _assert_close(
        _everything(data, resolve_plan("cpu", threads=2, chunk_size=10**6), spec, theta),
        base,
        1e-10,
    )


@needs_torch
@pytest.mark.torch
@pytest.mark.parametrize("spec,theta", [(ModelSpec(2), TH2), (ModelSpec(3), TH3)], ids=["M2", "M3"])
def test_torch_backend_matches_numpy(data, spec, theta):
    base = _everything(data, resolve_plan("cpu", threads=1), spec, theta)
    torch_cpu = _everything(data, resolve_plan("cpu", backend="torch", threads=2), spec, theta)
    _assert_close(torch_cpu, base, rtol=1e-10)
    # several shards (stand-in for several GPUs) with small chunks inside each
    shards = _everything(
        data,
        resolve_plan(["cpu", "cpu", "cpu"], backend="torch", threads=2, chunk_size=401),
        spec,
        theta,
    )
    _assert_close(shards, base, rtol=1e-10)


@needs_torch
@pytest.mark.mps
def test_device_plumbing_on_mps_in_single_precision(data):
    """No CUDA here: exercise the non-CPU device path on Apple MPS (float32, loose tolerance)."""
    import torch

    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")
    base = _everything(data, resolve_plan("cpu", threads=1), ModelSpec(3), TH3)
    plan = resolve_plan("mps", backend="torch", dtype="float32")
    with Engine(data, plan) as eng:
        lik = CompositeLikelihood(ModelSpec(3), eng, 10)
        em = lik.em_pass(TH3)
        po = lik.posterior(TH3)
    assert em.loglik == pytest.approx(base["loglik"][0], rel=2e-4)
    np.testing.assert_allclose(em.counts, base["counts"], rtol=5e-3)
    np.testing.assert_allclose(po.p_null, base["p_null"], atol=5e-3)


def test_mps_is_refused_for_double_precision():
    with pytest.raises(BackendError, match="64-bit"):
        resolve_plan("mps")


def test_plan_resolution_rules(monkeypatch):
    monkeypatch.delenv("GENESIS_NUM_THREADS", raising=False)
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "7")
    assert available_cpus() == 7
    assert resolve_plan("cpu").n_threads == 7
    monkeypatch.setenv("GENESIS_NUM_THREADS", "3")
    assert available_cpus() == 3
    monkeypatch.setenv("GENESIS_NUM_THREADS", "not-a-number")
    assert available_cpus() == 7
    assert resolve_plan("cpu", threads=2).n_threads == 2

    plan = resolve_plan("cpu")
    assert plan.backend == "numpy" and plan.devices == ("cpu",) and not plan.is_gpu
    assert "numpy" in plan.describe()
    assert plan.to_dict()["dtype"] == "float64"
    auto = resolve_plan("auto", backend="numpy")
    assert auto.backend == "numpy"

    with pytest.raises(InputError):
        resolve_plan("cpu", threads=0)
    with pytest.raises(InputError):
        resolve_plan("cpu", chunk_size=0)
    with pytest.raises(BackendError):
        resolve_plan("tpu")
    with pytest.raises(BackendError):
        resolve_plan("cuda:0", backend="numpy")
    with pytest.raises(BackendError):
        get_backend("jax")


def test_cuda_request_without_cuda_fails_clearly():
    try:
        import torch

        if torch.cuda.is_available():
            pytest.skip("CUDA present")
    except ImportError:
        pass
    with pytest.raises(BackendError, match="CUDA"):
        resolve_plan("cuda")
    with pytest.raises(BackendError):
        resolve_plan("cuda:0")


def test_engine_lifecycle(data):
    eng = Engine(data, resolve_plan("cpu", threads=2))
    assert eng.K == data.K and eng.sum_nstar == float(data.nstar.sum())
    assert "numpy" in eng.describe()
    eng.close()
    eng.close()  # idempotent
    with pytest.raises(RuntimeError):
        CompositeLikelihood(ModelSpec(2), eng, 10).em_pass(TH2)
