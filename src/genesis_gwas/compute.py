"""Compute planning: which backend, which devices, how many threads.

The rules are deliberately boring and HPC-safe:

* The number of CPU threads is taken from the *allocation*, never from the size of the
  node: ``GENESIS_NUM_THREADS`` → ``SLURM_CPUS_PER_TASK`` → the process CPU-affinity
  mask → ``os.cpu_count()``. On a shared cluster node ``os.cpu_count()`` reports every
  core of the machine, which is the classic way to oversubscribe a Slurm job.
* ``device="auto"`` uses every CUDA device visible to the process (Slurm exposes only
  the GPUs granted to the job) when PyTorch with CUDA is available, else the CPU.
* All inference is double precision. Apple MPS has no ``float64`` and is rejected.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from genesis_gwas.exceptions import BackendError, InputError

__all__ = ["ComputePlan", "available_cpus", "describe_environment", "resolve_plan"]


def available_cpus() -> int:
    """Number of CPUs this process may use (allocation-aware)."""
    for var in ("GENESIS_NUM_THREADS", "SLURM_CPUS_PER_TASK"):
        val = os.environ.get(var, "").strip()
        if val:
            try:
                n = int(val)
            except ValueError:
                continue
            if n >= 1:
                return n
    if hasattr(os, "sched_getaffinity"):
        try:
            return max(1, len(os.sched_getaffinity(0)))
        except OSError:  # pragma: no cover - exotic platforms
            pass
    return max(1, os.cpu_count() or 1)


@dataclass(frozen=True)
class ComputePlan:
    """Resolved execution plan.

    Attributes
    ----------
    backend
        ``"numpy"`` or ``"torch"``.
    devices
        Device strings, e.g. ``("cpu",)`` or ``("cuda:0", "cuda:1")``. With several
        devices the SNPs are split into contiguous shards, one per device, and the
        sufficient statistics are reduced on the host.
    n_threads
        Worker threads for the NumPy backend / intra-op threads for torch on CPU.
    chunk_size
        SNPs per chunk, or ``None`` to size chunks automatically from the number of
        mixture terms (CPU: cache-sized; GPU: from free device memory).
    dtype
        ``"float64"`` (always, for inference).
    """

    backend: str = "numpy"
    devices: tuple[str, ...] = ("cpu",)
    n_threads: int = 1
    chunk_size: int | None = None
    dtype: str = "float64"

    @property
    def is_gpu(self) -> bool:
        """Whether any device is a CUDA GPU."""
        return any(d.startswith("cuda") for d in self.devices)

    def describe(self) -> str:
        """One-line summary for logs."""
        if self.backend == "numpy":
            return f"numpy on CPU, {self.n_threads} thread(s)"
        devs = ", ".join(self.devices)
        extra = f", {self.n_threads} thread(s)" if not self.is_gpu else ""
        return f"torch on {devs}{extra}"

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable representation (stored with every result for provenance)."""
        return {
            "backend": self.backend,
            "devices": list(self.devices),
            "n_threads": self.n_threads,
            "chunk_size": self.chunk_size,
            "dtype": self.dtype,
        }


def _cuda_devices() -> list[str]:
    try:
        import torch
    except Exception:
        return []
    if not torch.cuda.is_available():
        return []
    return [f"cuda:{i}" for i in range(torch.cuda.device_count())]


def resolve_plan(
    device: str | Sequence[str] = "auto",
    *,
    threads: int | None = None,
    chunk_size: int | None = None,
    backend: str | None = None,
    dtype: str = "float64",
) -> ComputePlan:
    """Turn user-facing device/thread options into a :class:`ComputePlan`.

    Parameters
    ----------
    device
        ``"auto"`` (default), ``"cpu"``, ``"cuda"`` (all visible GPUs), ``"cuda:0"``, a
        comma-separated list such as ``"cuda:0,cuda:1"``, or a list of device strings.
    threads
        CPU threads; default :func:`available_cpus`.
    chunk_size
        SNPs per chunk (``None`` = automatic).
    backend
        Force ``"numpy"`` or ``"torch"``. By default CPU work uses NumPy and GPU work
        uses PyTorch; ``backend="torch"`` with ``device="cpu"`` runs the PyTorch code
        path on the CPU (useful for testing an installation without a GPU).
    dtype
        Leave at ``"float64"``.

    Raises
    ------
    BackendError
        If the requested device is unavailable or cannot do double precision.
    """
    if threads is not None and threads < 1:
        raise InputError("threads must be >= 1")
    if chunk_size is not None and chunk_size < 1:
        raise InputError("chunk_size must be >= 1")
    n_threads = int(threads) if threads is not None else available_cpus()

    if isinstance(device, str):
        requested = [d.strip() for d in device.split(",") if d.strip()]
    else:
        requested = [str(d).strip() for d in device]
    if not requested:
        raise InputError("device must not be empty")

    if requested == ["auto"]:
        if backend == "numpy":
            devices = ["cpu"]
        else:
            devices = _cuda_devices() or ["cpu"]
    elif requested == ["cuda"] or requested == ["gpu"]:
        devices = _cuda_devices()
        if not devices:
            raise BackendError(
                "device='cuda' was requested but no CUDA device is available to PyTorch. "
                "Check that the job was granted a GPU (nvidia-smi) and that a CUDA build of "
                "PyTorch is installed (python -c 'import torch; print(torch.version.cuda)')."
            )
    else:
        devices = requested

    for d in devices:
        if d == "mps" and dtype == "float64":
            raise BackendError(
                "Apple MPS does not support 64-bit floating point, which GENESIS needs for "
                "reliable inference. Use --device cpu on Apple-silicon machines."
            )
        if not (d == "cpu" or d == "mps" or d.startswith("cuda")):
            raise BackendError(f"Unrecognised device {d!r} (expected 'cpu', 'cuda' or 'cuda:N').")
    if len(set(devices)) != len(devices) and any(d != "cpu" for d in devices):
        raise BackendError(f"Duplicate devices requested: {devices}")

    on_gpu = any(d != "cpu" for d in devices)
    if on_gpu and any(d == "cpu" for d in devices):
        raise BackendError("Mixing 'cpu' with GPU devices in one plan is not supported.")
    if backend is None:
        backend = "torch" if on_gpu else "numpy"
    if backend == "numpy" and on_gpu:
        raise BackendError("The numpy backend cannot use GPU devices.")
    if backend == "numpy":
        devices = ["cpu"]

    if backend == "torch" and on_gpu:
        visible = set(_cuda_devices())
        for d in devices:
            if d.startswith("cuda") and (d if ":" in d else "cuda:0") not in visible:
                raise BackendError(
                    f"Device {d!r} is not available (visible CUDA devices: {sorted(visible) or 'none'})."
                )
        devices = [d if ":" in d or d == "mps" else "cuda:0" for d in devices]

    return ComputePlan(
        backend=backend,
        devices=tuple(devices),
        n_threads=n_threads,
        chunk_size=chunk_size,
        dtype=dtype,
    )


def describe_environment() -> dict[str, Any]:
    """Collect facts about the execution environment (used by ``genesis doctor``)."""
    import platform

    import numpy as np
    import scipy

    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "available_cpus": available_cpus(),
        "os_cpu_count": os.cpu_count(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch": None,
        "cuda_devices": [],
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["torch_cuda_build"] = torch.version.cuda
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                info["cuda_devices"].append(
                    {
                        "index": i,
                        "name": props.name,
                        "memory_gib": round(props.total_memory / 2**30, 1),
                    }
                )
    except Exception as exc:  # pragma: no cover - depends on the environment
        info["torch_error"] = f"{type(exc).__name__}: {exc}"
    return info
