#!/usr/bin/env python
"""Write ONE swarm file that fits every trait of every ancestry of a project.

Expected layout (PROJECT is the first argument)::

    PROJECT/sumstats/<ANCESTRY>/<trait>.<ext>   one summary-statistics file per trait
    PROJECT/ref/<ANCESTRY>/                     LD reference built with ld_build.sh
    PROJECT/fit_options.tsv                     optional per-trait options (see below)

For every input that has no ``PROJECT/results/<ANCESTRY>/<trait>.best.json`` yet, one
``genesis fit`` line is written to ``PROJECT/swarm/genesis.swarm``. Run the script again
after a swarm and the file contains exactly the fits that are still missing.

The LD reference of an ancestry is ``PROJECT/ref/<ANCESTRY>`` when that directory holds a
bundle; otherwise the ancestries ``EUR``, ``EAS`` and ``AFR`` use the downloadable
references ``eur_1kg_hm3``, ``eas_1kg_hm3`` and ``afr_1kg_hm3`` (fetch them first with
fetch_refdata.sh); anything else is skipped with a message.
``--reference ANC=NAME_OR_DIR`` overrides the choice.

Options for ``genesis fit`` come from two places and are combined:

* everything after a literal ``--`` on the command line goes to every fit;
* ``PROJECT/fit_options.tsv`` (or ``--options FILE``) adds options to the fits whose
  ``ANCESTRY/trait`` matches a shell-style pattern. Every matching line applies, in file
  order::

      # pattern      extra `genesis fit` options
      */*            --n-cases NCASE --n-controls NCONTROL
      EUR/overall    --population-prevalence 0.10 --sample-prevalence 0.45
      EAS/*          --match-on position --build GRCh38

Nothing is submitted: the script ends by printing the ``swarm`` commands for you to run
on the Biowulf login node. Run it inside an sinteractive session with the environment
active::

    python project_swarm.py /data/$USER/proj -- --qq
"""

from __future__ import annotations

import argparse
import fnmatch
import getpass
import os
import shlex
import sys
from pathlib import Path

from genesis_gwas.hpc.swarm import trait_name

IGNORED_SUFFIXES = (".tbi", ".csi", ".md5", ".sha256", ".log", ".md", ".pdf", ".png", ".json")


# Ancestry labels served by a downloadable reference when PROJECT/ref/<ANCESTRY> is absent.
DOWNLOADABLE = {"EUR": "eur_1kg_hm3", "EAS": "eas_1kg_hm3", "AFR": "afr_1kg_hm3"}


def read_options(path: Path) -> list[tuple[str, list[str]]]:
    """Parse ``pattern  options...`` lines; ``#`` starts a comment."""
    rules = []
    for number, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split(None, 1)
        try:
            options = shlex.split(parts[1]) if len(parts) > 1 else []
        except ValueError as exc:
            sys.exit(f"{path}:{number}: cannot parse the options ({exc})")
        if "/" not in parts[0]:
            sys.exit(f"{path}:{number}: the pattern must look like ANCESTRY/trait, e.g. */ERneg")
        rules.append((parts[0], options))
    return rules


def main() -> int:
    """Write the swarm file and print the commands that submit it."""
    argv = sys.argv[1:]
    fit_args: list[str] = []
    if "--" in argv:
        split = argv.index("--")
        argv, fit_args = argv[:split], argv[split + 1 :]

    user = os.environ.get("USER") or getpass.getuser()
    data_dir = os.environ.get("GENESIS_DATA_DIR", f"/data/{user}/genesis-ref")
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        epilog="Options after a literal `--` are passed to every `genesis fit`.",
    )
    parser.add_argument("project", type=Path, help="project directory")
    parser.add_argument("--options", type=Path, help="per-trait options [PROJECT/fit_options.tsv]")
    parser.add_argument("--pattern", default="*", help="file-name pattern of the inputs [*]")
    parser.add_argument(
        "--reference",
        action="append",
        default=[],
        metavar="ANC=REF",
        help="LD reference (name or directory) for one ancestry; repeatable",
    )
    parser.add_argument("--cpus", type=int, default=16, help="CPUs per fit [16]")
    parser.add_argument("--mem-gb", type=int, default=8, help="memory per fit in GB [8]")
    parser.add_argument("--time", default="00:30:00", help="walltime per fit [00:30:00]")
    parser.add_argument(
        "--bundle",
        type=int,
        default=1,
        metavar="B",
        help="run B fits one after another in each subjob (swarm -b B); use it when one fit "
        "takes less than about 15 minutes [1]",
    )
    parser.add_argument("--partition", help="CPU partition(s), e.g. quick,norm [norm]")
    parser.add_argument("--gpu", action="store_true", help="one GPU per fit (partition gpu)")
    parser.add_argument(
        "--gpu-constraint",
        default="gpua100|gpuv100x",
        help="GPU types to accept with --gpu [gpua100|gpuv100x]",
    )
    parser.add_argument(
        "--activate",
        default=(
            f"export GENESIS_DATA_DIR={shlex.quote(data_dir)} && "
            f"source /data/{user}/conda/etc/profile.d/conda.sh && conda activate genesis"
        ),
        help="shell line that makes `genesis` available in each subjob",
    )
    parser.add_argument("--force", action="store_true", help="also re-run finished fits")
    args = parser.parse_args(argv)

    project = args.project.resolve()  # swarm lines must not depend on a working directory
    sumstats_root = project / "sumstats"
    if not sumstats_root.is_dir():
        sys.exit(f"{sumstats_root} does not exist; expected PROJECT/sumstats/<ANCESTRY>/<files>")

    options_file = args.options or project / "fit_options.tsv"
    rules = read_options(options_file) if options_file.is_file() else []
    if args.options and not options_file.is_file():
        sys.exit(f"{options_file} does not exist")
    overrides = dict(item.split("=", 1) for item in args.reference if "=" in item)
    if len(overrides) != len(args.reference):
        sys.exit("--reference takes ANCESTRY=NAME_OR_DIRECTORY")

    jobs: list[tuple[int, str]] = []  # (input size, swarm line)
    for anc_dir in sorted(p for p in sumstats_root.iterdir() if p.is_dir()):
        anc = anc_dir.name
        if anc in overrides:
            reference = overrides[anc]
        elif (project / "ref" / anc / "manifest.json").is_file():
            reference = str(project / "ref" / anc)
        elif anc.upper() in DOWNLOADABLE:
            reference = DOWNLOADABLE[anc.upper()]
        else:
            print(
                f"{anc}: SKIPPED, no LD reference at {project / 'ref' / anc} and no "
                f"downloadable reference for this label (those are {', '.join(DOWNLOADABLE)}); "
                "build one with ld_build.sh or pass --reference"
            )
            continue

        files = sorted(
            p
            for p in anc_dir.iterdir()
            if p.is_file()
            and not p.name.startswith(".")
            and fnmatch.fnmatch(p.name, args.pattern)
            and not p.name.lower().endswith(IGNORED_SUFFIXES)
        )
        names = [trait_name(p) for p in files]
        clashes = sorted({n for n in names if names.count(n) > 1})
        if clashes:
            sys.exit(f"{anc}: several files map to the same trait name {clashes}; rename them")

        n_done = n_new = 0
        for path, trait in zip(files, names, strict=True):
            prefix = project / "results" / anc / trait
            best = Path(f"{prefix}.best.json")
            if not args.force and best.is_file() and best.stat().st_size > 0:
                n_done += 1
                continue
            extra = [
                o
                for pattern, opts in rules
                if fnmatch.fnmatch(f"{anc}/{trait}", pattern)
                for o in opts
            ]
            device = "cuda" if args.gpu else "cpu"
            command = ["genesis", "fit", str(path), "--out", str(prefix), "--device", device]
            command += ["--reference", reference, *fit_args, *extra]
            line = " ".join(shlex.quote(c) for c in command)
            if args.activate.strip():
                line = f"{args.activate.strip()} && {line}"
            jobs.append((path.stat().st_size, line))
            n_new += 1
        print(f"{anc}: reference {reference}; {n_new} fit(s) to run, {n_done} already finished")

    if not jobs:
        print("Nothing to do: every input already has a .best.json (or no inputs were found).")
        return 0

    swarm_dir, log_dir = project / "swarm", project / "logs" / "swarm"
    swarm_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    swarmfile = swarm_dir / "genesis.swarm"
    jobs.sort(key=lambda job: -job[0])  # largest input first: a `head` pilot sees the worst case
    swarmfile.write_text("".join(line + "\n" for _, line in jobs))

    opts = f"-g {args.mem_gb} -t {args.cpus} --time {args.time}"
    if args.bundle > 1:
        opts += f" -b {args.bundle}"
    opts += f" --logdir {shlex.quote(str(log_dir))}"
    if args.gpu:
        constraint = shlex.quote(f'--constraint="{args.gpu_constraint}"')
        opts += f" --partition gpu --gres=gpu:1 --sbatch {constraint}"
    elif args.partition:
        opts += f" --partition {args.partition}"
    print(f"\nWrote {swarmfile} with {len(jobs)} fit(s), largest input first.")
    print("Read it, then run on the Biowulf login node:")
    print(f"  swarm --devel {opts} {shlex.quote(str(swarmfile))}    # dry run: submits nothing")
    print(f"  swarm {opts} {shlex.quote(str(swarmfile))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
