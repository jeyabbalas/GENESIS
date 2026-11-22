"""Generate Biowulf *swarm* files for fitting many traits.

A swarm file has one shell command per line; Biowulf's ``swarm`` utility turns it into
a Slurm job array, giving every line its own allocation. Fitting many traits (or the
same trait under several settings) is embarrassingly parallel, so this is the most
efficient way to use many CPUs or GPUs: each fit gets a modest allocation and the
scheduler runs as many as the cluster has room for.

This module only *writes* files and returns the command the user should run; it never
submits anything. The generated lines use absolute paths and name the reference-data
directory explicitly, so that they do not depend on the directory or the environment
from which the swarm is submitted.
"""

from __future__ import annotations

import logging
import os
import re
import shlex
from collections.abc import Sequence
from pathlib import Path

from genesis_gwas.exceptions import InputError

__all__ = ["DEFAULT_GPU_CONSTRAINT", "swarm_command", "trait_name", "write_swarm"]

log = logging.getLogger(__name__)

#: GPU types requested by default. GENESIS computes in double precision, which the
#: compute-class cards (A100, V100x) execute at full speed; graphics-class cards may not.
DEFAULT_GPU_CONSTRAINT = "gpua100|gpuv100x"

_SUFFIXES = (
    ".gz",
    ".bgz",
    ".bz2",
    ".xz",
    ".zst",
    ".txt",
    ".tsv",
    ".csv",
    ".sumstats",
    ".assoc",
    ".ma",
    ".tbl",
)


def trait_name(path: str | os.PathLike[str]) -> str:
    """Derive a job/output name from a summary-statistics file name.

    ``/data/x/BCAC_overall.sumstats.tsv.gz`` → ``BCAC_overall``.
    """
    name = Path(os.fspath(path)).name
    changed = True
    while changed:
        changed = False
        for suf in _SUFFIXES:
            if name.lower().endswith(suf) and len(name) > len(suf):
                name = name[: -len(suf)]
                changed = True
    return re.sub(r"[^A-Za-z0-9_.+-]", "_", name) or "trait"


def _abspath(path: str | os.PathLike[str]) -> str:
    """Absolute path without resolving symbolic links (``/data/$USER`` must stay as typed)."""
    return os.path.abspath(os.path.expanduser(os.fspath(path)))


def swarm_command(
    swarmfile: str | os.PathLike[str],
    *,
    gpu: bool = False,
    cpus: int = 16,
    mem_gb: int = 16,
    walltime: str = "04:00:00",
    logdir: str | os.PathLike[str] | None = None,
    bundle: int | None = None,
    partition: str | None = None,
    gpu_constraint: str | None = DEFAULT_GPU_CONSTRAINT,
    devel: bool = False,
) -> str:
    """The ``swarm`` command line that submits ``swarmfile`` with the given resources.

    Parameters
    ----------
    swarmfile
        Path of the swarm file (given last, the form current versions of ``swarm`` expect).
    gpu
        Request one GPU per job in the ``gpu`` partition.
    cpus
        ``-t``: CPUs (hyperthreads) per fit. GENESIS uses all of them.
    mem_gb
        ``-g``: memory per fit in GB.
    walltime
        ``--time`` per fit.
    logdir
        ``--logdir`` for the per-job output files. The directory must exist when the swarm
        is submitted.
    bundle
        ``-b``: run this many fits one after another in each subjob (for short fits; the
        walltime of a subjob is ``walltime`` times ``bundle``).
    partition
        ``--partition`` (e.g. ``"quick,norm"``). With ``gpu`` the default is ``"gpu"``.
    gpu_constraint
        Node features accepted for GPU jobs, e.g. ``"gpua100|gpuv100x"``; ``None`` or an
        empty string accepts any GPU type.
    devel
        Return the dry-run form (``swarm --devel``), which submits nothing.
    """
    if cpus < 1 or mem_gb < 1:
        raise InputError("cpus and mem_gb must be >= 1")
    if bundle is not None and bundle < 1:
        raise InputError("bundle must be >= 1")
    parts = ["swarm"]
    if devel:
        parts.append("--devel")
    parts += ["-g", str(int(mem_gb)), "-t", str(int(cpus)), "--time", walltime]
    if bundle is not None and bundle > 1:
        parts += ["-b", str(int(bundle))]
    if gpu:
        parts += ["--partition", partition or "gpu", "--gres=gpu:1"]
        if gpu_constraint:
            parts += ["--sbatch", shlex.quote(f'--constraint="{gpu_constraint}"')]
    elif partition:
        parts += ["--partition", partition]
    if logdir is not None:
        parts += ["--logdir", shlex.quote(os.fspath(logdir))]
    parts.append(shlex.quote(os.fspath(swarmfile)))
    return " ".join(parts)


def write_swarm(
    sumstats: Sequence[str | os.PathLike[str]],
    *,
    outdir: str | os.PathLike[str],
    swarmfile: str | os.PathLike[str] = "genesis.swarm",
    fit_args: Sequence[str] = (),
    gpu: bool = False,
    cpus: int | None = None,
    mem_gb: int = 16,
    walltime: str = "04:00:00",
    activate: str = "",
    executable: str = "genesis",
    data_dir: str | os.PathLike[str] | None = None,
    bundle: int | None = None,
    partition: str | None = None,
    gpu_constraint: str | None = DEFAULT_GPU_CONSTRAINT,
    devel: bool = False,
) -> str:
    """Write a swarm file with one ``genesis fit`` per summary-statistics file.

    Parameters
    ----------
    sumstats
        Summary-statistics files; each becomes one job writing ``OUTDIR/<trait>.*``.
        Relative paths are made absolute.
    outdir
        Directory for the results. It and its ``logs`` sub-directory (for the output of
        the swarm itself) are created if possible, because ``swarm`` does not create its
        log directory.
    swarmfile
        Where to write the swarm file.
    fit_args
        Extra arguments appended to every ``genesis fit`` command. A ``--reference`` that
        names an existing directory is made absolute.
    gpu, cpus, mem_gb, walltime, bundle, partition, gpu_constraint
        Resources per job (see :func:`swarm_command`). ``cpus`` defaults to 16 for CPU
        jobs and to 8 for GPU jobs (within the CPUs-per-GPU limit of every GPU type).
    activate
        Shell snippet that makes ``genesis`` available inside each job, e.g.
        ``"source /data/$USER/conda/etc/profile.d/conda.sh && conda activate genesis"``.
        A swarm inherits the environment of the shell that submits it, in which an
        environment is usually not active.
    executable
        Name or path of the ``genesis`` executable.
    data_dir
        Directory holding the reference bundles, written into every line as
        ``--data-dir``. Defaults to ``$GENESIS_DATA_DIR`` at the time of the call; without
        either, the jobs fall back to the default cache in the home directory.
    devel
        Return the dry-run command instead of the submitting one.

    Returns
    -------
    str
        The ``swarm`` command the user should run on the login node.
    """
    if not sumstats:
        raise InputError("No summary-statistics files given")
    names = [trait_name(p) for p in sumstats]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise InputError(
            f"These inputs would write to the same output prefix: {dupes}. Rename the files."
        )
    n_cpus = (8 if gpu else 16) if cpus is None else int(cpus)
    out = Path(_abspath(outdir))
    logdir = out / "logs"
    device = "cuda" if gpu else "cpu"
    extra = [str(a) for a in fit_args]
    for i, arg in enumerate(extra):
        # A reference given as a directory (rather than by name) must not depend on the
        # directory the swarm is submitted from either.
        if arg == "--reference" and i + 1 < len(extra) and os.path.isdir(extra[i + 1]):
            extra[i + 1] = _abspath(extra[i + 1])
        elif arg.startswith("--reference=") and os.path.isdir(arg.split("=", 1)[1]):
            extra[i] = "--reference=" + _abspath(arg.split("=", 1)[1])
    if data_dir is None:
        data_dir = os.environ.get("GENESIS_DATA_DIR", "").strip() or None
    if data_dir is not None and not any(a.split("=")[0] == "--data-dir" for a in extra):
        extra = ["--data-dir", _abspath(data_dir), *extra]
    missing = [os.fspath(p) for p in sumstats if not os.path.exists(_abspath(p))]
    if missing:
        log.warning(
            "WARNING: %d input file(s) do not exist on this machine (first: %s). Generate the "
            "swarm file where the data are, or check the paths.",
            len(missing),
            missing[0],
        )
    lines = []
    for path, name in zip(sumstats, names, strict=True):
        cmd = [
            executable,
            "fit",
            _abspath(path),
            "--out",
            str(out / name),
            "--device",
            device,
            *extra,
        ]
        line = " ".join(shlex.quote(c) for c in cmd)
        if activate.strip():
            line = f"{activate.strip()} && {line}"
        if "#" in line:
            raise InputError(
                "A '#' in a path or option starts a comment in a swarm file and would cut the "
                f"command short: {line!r}. Rename the file or remove the character."
            )
        lines.append(line)
    sf = Path(_abspath(swarmfile))
    sf.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# Swarm file written by `genesis swarm`; one fit per line.\n"
        "# Threads are taken from $SLURM_CPUS_PER_TASK automatically.\n"
    )
    sf.write_text(header + "\n".join(lines) + "\n")
    try:
        logdir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        log.warning(
            "WARNING: could not create %s (%s). Create it before submitting: swarm does not "
            "create its log directory.",
            logdir,
            exc.strerror or exc,
        )
    return swarm_command(
        sf,
        gpu=gpu,
        cpus=n_cpus,
        mem_gb=mem_gb,
        walltime=walltime,
        logdir=logdir,
        bundle=bundle,
        partition=partition,
        gpu_constraint=gpu_constraint,
        devel=devel,
    )
