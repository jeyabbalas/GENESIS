"""NumPy (CPU) backend."""

from __future__ import annotations

from typing import Any

import numpy as np

from genesis_gwas.backends.base import Backend
from genesis_gwas.exceptions import BackendError

__all__ = ["NumpyBackend"]


class NumpyBackend(Backend):
    """CPU backend built on NumPy.

    NumPy's element-wise loops are single-threaded but release the GIL, so the engine
    obtains multi-core scaling by evaluating independent, cache-sized chunks of SNPs in
    a thread pool. No BLAS call sits in the hot path, which avoids nested-threading
    oversubscription regardless of how the linked BLAS is configured.
    """

    name = "numpy"
    xp = np

    def __init__(self, dtype: str = "float64") -> None:
        if dtype not in ("float64", "float32"):
            raise BackendError(f"Unsupported dtype {dtype!r}")
        self.dtype = dtype
        self._float = np.dtype(dtype)

    def to_device(self, array: np.ndarray, device: str) -> Any:  # noqa: D102
        if device != "cpu":
            raise BackendError(f"The numpy backend only runs on 'cpu' (got {device!r}).")
        arr = np.asarray(array)
        if arr.dtype.kind == "f":
            return np.ascontiguousarray(arr, dtype=self._float)
        return np.ascontiguousarray(arr)

    def to_numpy(self, array: Any) -> np.ndarray:  # noqa: D102
        arr = np.asarray(array)
        return arr.astype(np.float64, copy=False) if arr.dtype.kind == "f" else arr

    def synchronize(self, device: str) -> None:  # noqa: D102
        return None

    def free_memory(self, device: str) -> int | None:  # noqa: D102
        return None
