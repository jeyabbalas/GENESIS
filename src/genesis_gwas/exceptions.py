"""Exception hierarchy for :mod:`genesis_gwas`.

Every error raised deliberately by the package derives from :class:`GenesisError`,
so callers (and the command-line interface) can distinguish "the tool is telling
you something about your inputs" from genuine programming errors.
"""

from __future__ import annotations


class GenesisError(Exception):
    """Base class for all errors raised deliberately by ``genesis_gwas``."""


class InputError(GenesisError, ValueError):
    """An argument or an input file is invalid (wrong shape, type, range or format)."""


class DataError(GenesisError):
    """Summary statistics or reference data are unusable for the requested analysis."""


class ReferenceDataError(DataError):
    """A reference bundle is missing, corrupt, or does not contain a requested component."""


class DownloadError(ReferenceDataError):
    """A reference-data download failed or did not match its recorded checksum."""


class BackendError(GenesisError):
    """The requested compute backend or device is unavailable or unsupported."""


class ConvergenceError(GenesisError):
    """The optimiser could not make progress (raised only when ``strict=True``)."""


class GenesisWarning(UserWarning):
    """Base class for warnings emitted by ``genesis_gwas``."""


class ConvergenceWarning(GenesisWarning):
    """The EM algorithm stopped before meeting its convergence criteria."""


class NumericalWarning(GenesisWarning):
    """A numerically fragile situation was detected (e.g. a near-singular information matrix)."""


class DataWarning(GenesisWarning):
    """Something about the input data deserves the analyst's attention."""
