"""Generate Biowulf *swarm* files for fitting many traits.

A swarm file has one shell command per line; Biowulf's ``swarm`` utility turns it into
a Slurm job array, giving every line its own allocation. Fitting many traits (or the
same trait under several settings) is embarrassingly parallel, so this is the most
efficient way to use many CPUs or GPUs: each fit gets a modest allocation and the
scheduler runs as many as the cluster has room for.

This module only *writes* files and returns the command the user should run; it never
submits anything.
"""

from __future__ import annotations

import os
import re
import shlex
from collections.abc import Sequence
from pathlib import Path

from genesis_gwas.exceptions import InputError

__all__ = ["swarm_command", "trait_name", "write_swarm"]

_SUFFIXES = (".gz", ".bgz", ".bz2", ".xz", ".zst", ".txt", ".tsv", ".csv", ".sumstats", ".assoc", ".ma", ".tbl")


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


def swarm_command(
    swarmfile: str | os.PathLike[str],
    *,
    gpu: bool = False,
    cpus: int = 16,
    mem_gb: int = 16,
    walltime: str = "04:00:00",
    logdir: str | os.PathLike[str] | None = None,
) -> str:
    """The ``swarm`` command line that submits ``swarmfile`` with the given resources.

    Parameters
    ----------
    swarmfile
        Path of the swarm file.
    gpu
        Request one GPU per job in the ``gpu`` partition (any of V100x/A100/L40).
    cpus
        ``-t``: CPUs (hyperthreads) per job. GENESIS uses all of them.
    mem_gb
        ``-g``: memory per job in GB.
    walltime
        ``--time`` per job.
    logdir
        ``--logdir`` for the per-job output files.
    """
    if cpus < 1 or mem_gb < 1:
        raise InputError("cpus and mem_gb must be >= 1")
    parts = ["swarm", "-f", shlex.quote(os.fspath(swarmfile)), "-g", str(int(mem_gb)), "-t", str(int(cpus)),
             "--time", walltime]
    if gpu:
        parts += ["--partition", "gpu", "--gres", "gpu:1", "--sbatch", shlex.quote('--constraint="gpuv100x|gpua100|gpul40"')]
    if logdir is not None:
        parts += ["--logdir", shlex.quote(os.fspath(logdir))]
    return " ".join(parts)


def write_swarm(
    sumstats: Sequence[str | os.PathLike[str]],
    *,
    outdir: str | os.PathLike[str],
    swarmfile: str | os.PathLike[str] = "genesis.swarm",
    fit_args: Sequence[str] = (),
    gpu: bool = False,
    cpus: int = 16,
    mem_gb: int = 16,
    walltime: str = "04:00:00",
    activate: str = "",
    executable: str = "genesis",
) -> str:
    """Write a swarm file with one ``genesis fit`` per summary-statistics file.

    Parameters
    ----------
    sumstats
        Summary-statistics files; each becomes one job writing ``OUTDIR/<trait>.*``.
    outdir
        Directory for the results.
    swarmfile
        Where to write the swarm file.
    fit_args
        Extra arguments appended to every ``genesis fit`` command.
    gpu, cpus, mem_gb, walltime
        Resources per job (see :func:`swarm_command`).
    activate
        Shell snippet that makes ``genesis`` available inside each job, e.g.
        ``"source /data/$USER/conda/etc/profile.d/conda.sh && conda activate genesis"``.
    executable
        Name or path of the ``genesis`` executable.

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
        raise InputError(f"These inputs would write to the same output prefix: {dupes}. Rename the files.")
    out = Path(os.fspath(outdir))
    device = "cuda" if gpu else "cpu"
    lines = []
    for path, name in zip(sumstats, names, strict=True):
        cmd = [executable, "fit", os.fspath(path), "--out", str(out / name), "--device", device, *fit_args]
        line = " ".join(shlex.quote(c) for c in cmd)
        if activate.strip():
            line = f"{activate.strip()} && {line}"
        lines.append(line)
    sf = Path(os.fspath(swarmfile))
    sf.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# Swarm file written by `genesis swarm`; one fit per line.\n"
        "# Threads are taken from $SLURM_CPUS_PER_TASK automatically.\n"
    )
    sf.write_text(header + "\n".join(lines) + "\n")
    return swarm_command(sf, gpu=gpu, cpus=cpus, mem_gb=mem_gb, walltime=walltime, logdir=out / "logs")
