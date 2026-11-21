"""Chunked map-reduce execution of kernels over SNPs on CPUs or GPUs.

Every heavy computation in GENESIS has the same shape: evaluate a kernel on each SNP
(times each latent configuration) and either *sum* the result over SNPs or keep one row
per SNP. The :class:`Engine` owns the SNP arrays on the compute device(s) and runs a
kernel over them in chunks:

NumPy backend
    Cache-sized chunks are dispatched to a thread pool. NumPy releases the GIL inside
    its loops, so ``n_threads`` workers use ``n_threads`` cores, and the peak working
    memory is ``n_threads x chunk x T`` doubles regardless of the number of SNPs.
PyTorch backend
    The SNPs are split into one contiguous shard per device. Each device processes its
    shard in memory-sized chunks from its own host thread, so several GPUs work
    concurrently; partial sums are reduced on the host in a fixed order.

Results are deterministic for a fixed plan (backend, devices, chunk size): partial sums
are always combined in SNP order, independent of thread scheduling.
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from genesis_gwas.backends import Backend, get_backend
from genesis_gwas.compute import ComputePlan
from genesis_gwas.core.data import ModelData
from genesis_gwas.core.kernels import Chunk

__all__ = ["Engine"]

Kernel = Callable[[Any, Chunk, dict[str, Any]], tuple[dict[str, Any], dict[str, Any]]]

# Number of simultaneously live (n, T) temporaries in the most demanding kernel; used to
# size GPU chunks from free device memory.
_LIVE_TEMPORARIES = 12
_FIELDS = ("b", "b2", "s2", "r", "r2", "nstar", "uidx")


def _cpu_chunk_elements() -> int:
    """Target number of (SNP x configuration) elements per CPU chunk (~2 MB of doubles)."""
    try:
        return max(1024, int(os.environ.get("GENESIS_CHUNK_ELEMENTS", 1 << 18)))
    except ValueError:
        return 1 << 18


class _Shard:
    """SNP rows ``[start, stop)`` resident on one device."""

    __slots__ = ("arrays", "device", "start", "stop")

    def __init__(self, device: str, start: int, stop: int, arrays: dict[str, Any]) -> None:
        self.device = device
        self.start = start
        self.stop = stop
        self.arrays = arrays

    def chunk(self, i0: int, i1: int) -> Chunk:
        a = self.arrays
        return Chunk(*(a[f][i0:i1] for f in _FIELDS))


class Engine:
    """Run kernels over all SNPs according to a :class:`~genesis_gwas.compute.ComputePlan`.

    Parameters
    ----------
    data
        The model inputs. Derived per-SNP quantities (:math:`\\hat\\beta^2`,
        :math:`\\ell/N^*`, the index into the table of distinct :math:`N^*`) are computed
        once here and kept on the device(s).
    plan
        Execution plan from :func:`genesis_gwas.compute.resolve_plan`.

    Notes
    -----
    Use as a context manager (or call :meth:`close`) to release worker threads and
    device memory promptly.
    """

    def __init__(self, data: ModelData, plan: ComputePlan) -> None:
        self.plan = plan
        self.K = data.K
        self.backend: Backend = get_backend(plan.backend, plan.dtype)

        nstar_f = data.nstar.astype(np.float64)
        self.nstar_unique, uidx = np.unique(data.nstar, return_inverse=True)
        self.nstar_counts = np.bincount(uidx, minlength=self.nstar_unique.size).astype(np.float64)
        r = data.ldscore / nstar_f
        host = {
            "b": data.betahat,
            "b2": data.betahat * data.betahat,
            "s2": data.varbetahat,
            "r": r,
            "r2": r * r,
            "nstar": nstar_f,
            "uidx": uidx.astype(np.int64),
        }
        self.sum_nstar = float(nstar_f.sum())
        self.min_varbetahat = float(data.varbetahat.min())
        # Host-side float64 copies used when assembling per-SNP results.
        self.host_nstar = nstar_f
        self.host_r = r

        n_dev = len(plan.devices)
        bounds = np.linspace(0, self.K, n_dev + 1).astype(np.int64)
        self._shards: list[_Shard] = []
        for d, device in enumerate(plan.devices):
            lo, hi = int(bounds[d]), int(bounds[d + 1])
            arrays = {f: self.backend.to_device(host[f][lo:hi], device) for f in _FIELDS}
            self._shards.append(_Shard(device, lo, hi, arrays))

        if plan.backend == "numpy":
            workers = max(1, plan.n_threads)
        else:
            workers = n_dev
            if not plan.is_gpu:
                import torch

                torch.set_num_threads(max(1, plan.n_threads))
        self._pool: ThreadPoolExecutor | None = (
            ThreadPoolExecutor(max_workers=workers, thread_name_prefix="genesis")
            if workers > 1
            else None
        )

    # ----------------------------------------------------------------- lifecycle
    def close(self) -> None:
        """Release worker threads and drop references to device arrays."""
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None
        self._shards = []

    def __enter__(self) -> Engine:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------ chunking
    def chunk_size(self, n_terms: int, device: str) -> int:
        """SNPs per chunk for a kernel with ``n_terms`` mixture configurations."""
        if self.plan.chunk_size is not None:
            return max(1, int(self.plan.chunk_size))
        if device.startswith("cuda"):
            free = self.backend.free_memory(device)
            if free:
                itemsize = 8 if self.plan.dtype == "float64" else 4
                per_snp = n_terms * itemsize * _LIVE_TEMPORARIES
                return int(min(self.K, max(4096, (free // 2) // per_snp)))
            return 1 << 17
        target = _cpu_chunk_elements()
        if self.plan.backend == "torch":
            target *= 8  # torch parallelises inside each (larger) chunk
        return int(max(256, target // max(1, n_terms)))

    @staticmethod
    def _ranges(n: int, size: int) -> Iterator[tuple[int, int]]:
        for i0 in range(0, n, size):
            yield i0, min(n, i0 + size)

    # ----------------------------------------------------------------------- run
    def run(
        self,
        kernel: Kernel,
        tables: dict[str, Any],
        n_terms: int,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Evaluate ``kernel`` on every SNP.

        Parameters
        ----------
        kernel
            One of the functions in :mod:`genesis_gwas.core.kernels`.
        tables
            Small host-side inputs. NumPy arrays are moved to each device; scalars,
            booleans and ``None`` are passed through unchanged.
        n_terms
            Number of mixture configurations ``T`` (used only to size chunks).

        Returns
        -------
        (sums, per_snp)
            ``sums[name]`` is the kernel's sum output added over all SNPs (``float64``
            NumPy arrays; scalars become Python floats). ``per_snp[name]`` has one row
            per SNP, in SNP order.
        """
        if not self._shards:
            raise RuntimeError("Engine is closed")
        if self.plan.backend == "numpy":
            parts = self._run_numpy(kernel, tables, n_terms)
        else:
            parts = self._run_torch(kernel, tables, n_terms)
        return self._reduce(parts)

    def _device_tables(self, tables: dict[str, Any], device: str) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, val in tables.items():
            out[key] = self.backend.to_device(val, device) if isinstance(val, np.ndarray) else val
        return out

    def _run_numpy(
        self, kernel: Kernel, tables: dict[str, Any], n_terms: int
    ) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        shard = self._shards[0]
        tb = self._device_tables(tables, "cpu")
        size = self.chunk_size(n_terms, "cpu")
        ranges = list(self._ranges(shard.stop - shard.start, size))
        xp = self.backend.xp

        def work(rng: tuple[int, int]) -> tuple[dict[str, Any], dict[str, Any]]:
            return kernel(xp, shard.chunk(*rng), tb)

        if self._pool is None or len(ranges) == 1:
            return [work(rng) for rng in ranges]
        return list(self._pool.map(work, ranges))

    def _run_torch(
        self, kernel: Kernel, tables: dict[str, Any], n_terms: int
    ) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        import torch

        backend = self.backend
        xp = backend.xp

        def work(shard: _Shard) -> tuple[dict[str, Any], dict[str, Any]]:
            tb = self._device_tables(tables, shard.device)
            size = self.chunk_size(n_terms, shard.device)
            acc: dict[str, Any] = {}
            rows: dict[str, list[Any]] = {}
            with torch.inference_mode():
                for i0, i1 in self._ranges(shard.stop - shard.start, size):
                    sums, per = kernel(xp, shard.chunk(i0, i1), tb)
                    for key, val in sums.items():
                        # Accumulate in float64 on the device (matters only for the
                        # float32 smoke-test mode).
                        val = val.to(torch.float64) if backend.dtype != "float64" else val
                        acc[key] = val if key not in acc else acc[key] + val
                    for key, val in per.items():
                        rows.setdefault(key, []).append(backend.to_numpy(val))
            sums_np = {k: backend.to_numpy(v) for k, v in acc.items()}
            per_np = {k: np.concatenate(v, axis=0) for k, v in rows.items()}
            return sums_np, per_np

        if self._pool is None or len(self._shards) == 1:
            return [work(s) for s in self._shards]
        return list(self._pool.map(work, self._shards))

    def _reduce(
        self, parts: list[tuple[dict[str, Any], dict[str, Any]]]
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        sums: dict[str, Any] = {}
        per: dict[str, np.ndarray] = {}
        if not parts:
            return sums, per
        to_np = self.backend.to_numpy
        for key in parts[0][0]:
            vals = [to_np(p[0][key]) for p in parts]
            if vals[0].ndim == 0:
                # Exactly rounded sum of the per-chunk partial sums.
                sums[key] = math.fsum(float(v) for v in vals)
            else:
                total = np.array(vals[0], dtype=np.float64, copy=True)
                for v in vals[1:]:
                    total += v
                sums[key] = total
        for key in parts[0][1]:
            per[key] = np.concatenate([to_np(p[1][key]) for p in parts], axis=0)
        return sums, per

    # ------------------------------------------------------------------- reports
    def describe(self) -> str:
        """Human-readable description of the devices in use."""
        if self.plan.backend == "numpy":
            return self.plan.describe()
        return ", ".join(self.backend.describe(d) for d in self.plan.devices)
