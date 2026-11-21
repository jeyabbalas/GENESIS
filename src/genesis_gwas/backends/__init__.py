"""Array backends.

The numerical kernels in :mod:`genesis_gwas.core.kernels` are written once against a
small, NumPy-flavoured subset of array operations that both :mod:`numpy` and
:mod:`torch` implement with identical names and semantics (``log``, ``exp``, ``sum``,
``amax``, ``einsum``, broadcasting, integer-array row indexing). A
:class:`~genesis_gwas.backends.base.Backend` supplies that namespace (``backend.xp``)
plus the handful of operations that genuinely differ: moving data to a device and back.

``numpy``
    Default CPU backend. Multi-core execution comes from running independent SNP
    chunks in a thread pool (NumPy releases the GIL inside its inner loops).
``torch``
    GPU backend (one or more CUDA devices), also usable on CPU. PyTorch is an optional
    dependency: it is imported only when this backend is requested.
"""

from __future__ import annotations

from genesis_gwas.backends.base import Backend
from genesis_gwas.backends.numpy_backend import NumpyBackend
from genesis_gwas.exceptions import BackendError

__all__ = ["Backend", "NumpyBackend", "get_backend", "torch_available"]


def torch_available() -> bool:
    """Return ``True`` if PyTorch can be imported."""
    try:
        import torch  # noqa: F401
    except Exception:  # pragma: no cover - depends on the environment
        return False
    return True


def get_backend(name: str, dtype: str = "float64") -> Backend:
    """Instantiate a backend by name (``"numpy"`` or ``"torch"``)."""
    if name == "numpy":
        return NumpyBackend(dtype=dtype)
    if name == "torch":
        try:
            from genesis_gwas.backends.torch_backend import TorchBackend
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise BackendError(
                "The 'torch' backend needs PyTorch, which is not installed. Install it with "
                "`pip install 'genesis-gwas[gpu]'` (or a CUDA build matching your cluster), "
                "or use the default CPU backend (--device cpu)."
            ) from exc
        return TorchBackend(dtype=dtype)
    raise BackendError(f"Unknown backend {name!r}; expected 'numpy' or 'torch'.")
