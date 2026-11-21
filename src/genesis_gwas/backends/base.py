"""Backend interface shared by the NumPy and PyTorch implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

__all__ = ["Backend"]


class Backend(ABC):
    """Minimal array-backend abstraction used by the compute engine.

    Attributes
    ----------
    name
        ``"numpy"`` or ``"torch"``.
    xp
        The array namespace handed to the kernels (the :mod:`numpy` or :mod:`torch`
        module). Kernels only use functions that exist with the same name and keyword
        arguments in both.
    dtype
        Floating-point precision, ``"float64"`` for all inference. ``"float32"`` exists
        solely so device plumbing can be smoke-tested on hardware without double
        precision (Apple MPS); it is never used for estimation.
    """

    name: str
    xp: Any
    dtype: str

    @abstractmethod
    def to_device(self, array: np.ndarray, device: str) -> Any:
        """Copy a NumPy array to ``device`` (floats are cast to the backend precision)."""

    @abstractmethod
    def to_numpy(self, array: Any) -> np.ndarray:
        """Bring a backend array back to host memory as a ``float64``/integer NumPy array."""

    @abstractmethod
    def synchronize(self, device: str) -> None:
        """Block until queued work on ``device`` has finished (no-op on CPU)."""

    @abstractmethod
    def free_memory(self, device: str) -> int | None:
        """Bytes of free memory on ``device`` (``None`` if unknown / host memory)."""

    def describe(self, device: str) -> str:
        """Human-readable description of ``device`` for logs and provenance."""
        return f"{self.name}:{device}"
