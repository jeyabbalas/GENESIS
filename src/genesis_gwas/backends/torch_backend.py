"""PyTorch backend (CUDA GPUs; also CPU)."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from genesis_gwas.backends.base import Backend
from genesis_gwas.exceptions import BackendError

__all__ = ["TorchBackend"]


class TorchBackend(Backend):
    """Backend built on PyTorch tensors.

    All inference runs in ``float64``. Data-centre NVIDIA GPUs execute double precision
    natively; Apple's MPS backend has no ``float64`` and is therefore rejected for
    estimation (see :func:`genesis_gwas.compute.resolve_plan`).
    """

    name = "torch"
    xp = torch

    def __init__(self, dtype: str = "float64") -> None:
        if dtype not in ("float64", "float32"):
            raise BackendError(f"Unsupported dtype {dtype!r}")
        self.dtype = dtype
        self._float = torch.float64 if dtype == "float64" else torch.float32

    def to_device(self, array: np.ndarray, device: str) -> Any:  # noqa: D102
        arr = np.ascontiguousarray(array)
        if arr.dtype.kind == "f":
            # Cast on the host first: MPS cannot even receive a float64 tensor.
            host = torch.from_numpy(arr.astype(np.float64, copy=False)).to(self._float)
            return host.to(device)
        if arr.dtype.kind in "iu":
            return torch.from_numpy(arr.astype(np.int64, copy=False)).to(device)
        if arr.dtype.kind == "b":
            return torch.from_numpy(arr).to(device)
        raise BackendError(f"Cannot move array of dtype {arr.dtype} to a torch device")

    def to_numpy(self, array: Any) -> np.ndarray:  # noqa: D102
        if isinstance(array, torch.Tensor):
            out = array.detach().to("cpu")
            if out.is_floating_point():
                out = out.to(torch.float64)
            return out.numpy()
        return np.asarray(array)

    def synchronize(self, device: str) -> None:  # noqa: D102
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
        elif device == "mps":  # pragma: no cover - hardware specific
            torch.mps.synchronize()

    def free_memory(self, device: str) -> int | None:  # noqa: D102
        if device.startswith("cuda"):
            free, _total = torch.cuda.mem_get_info(device)
            return int(free)
        return None

    def describe(self, device: str) -> str:  # noqa: D102
        if device.startswith("cuda"):
            idx = torch.device(device).index or 0
            props = torch.cuda.get_device_properties(idx)
            return f"torch:{device} ({props.name}, {props.total_memory / 2**30:.0f} GiB)"
        return f"torch:{device}"
