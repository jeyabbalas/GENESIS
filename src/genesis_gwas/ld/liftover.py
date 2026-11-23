r"""Coordinate conversion between genome builds through a UCSC chain file.

A reference built from genotypes knows its SNPs' positions in one genome build, while
summary statistics may come in another. :func:`lift_positions` converts the positions
of a SNP table through a UCSC ``*.over.chain`` file (for example
``hg19ToHg38.over.chain.gz`` or ``hg38ToHg19.over.chain.gz`` from
https://hgdownload.soe.ucsc.edu/goldenPath/), so that a reference bundle can carry both
and match summary statistics by position in either build. The implementation is pure
Python/NumPy and reproduces UCSC ``liftOver`` for single bases.
"""

from __future__ import annotations

import gzip
import os
from pathlib import Path
from typing import Any

import numpy as np

from genesis_gwas.exceptions import DataError

__all__ = ["ChainLiftover", "lift_positions"]


class ChainLiftover:
    """Point liftover through a UCSC ``*.over.chain`` file (pure Python/numpy).

    Chain coordinates are 0-based half-open; reverse-strand query coordinates are
    relative to the reverse-complemented sequence, so a base at reverse coordinate ``x``
    sits at forward 0-based position ``qSize - 1 - x``. ``lift`` takes and returns
    1-based positions.
    """

    def __init__(self, chain: str | os.PathLike[str]) -> None:
        per_t: dict[str, list[tuple[int, int, int, bool, int, int]]] = {}
        qnames: list[str] = []
        qcode: dict[str, int] = {}
        n_chains = 0
        path = Path(os.fspath(chain)).expanduser()
        if not path.exists():
            raise DataError(f"chain file not found: {path}")
        opener: Any = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt") as fh:
            t_name = q_name = ""
            t = q = t_end = q_end = q_size = 0
            q_rev = False
            closed = True
            for line in fh:
                if line.startswith("chain"):
                    if not closed:
                        raise DataError("chain block not terminated")
                    f = line.split()
                    t_name, t_strand, t, t_end = f[2], f[4], int(f[5]), int(f[6])
                    q_name, q_size, q_strand, q, q_end = (
                        f[7],
                        int(f[8]),
                        f[9],
                        int(f[10]),
                        int(f[11]),
                    )
                    if t_strand != "+":
                        raise DataError("target strand '-' is not supported")
                    q_rev = q_strand == "-"
                    if q_name not in qcode:
                        qcode[q_name] = len(qnames)
                        qnames.append(q_name)
                    closed = False
                    n_chains += 1
                    continue
                f = line.split()
                if not f:
                    continue
                size = int(f[0])
                per_t.setdefault(t_name, []).append((t, t + size, q, q_rev, q_size, qcode[q_name]))
                if len(f) == 3:
                    t += size + int(f[1])
                    q += size + int(f[2])
                elif len(f) == 1:
                    t += size
                    q += size
                    if t != t_end or q != q_end:
                        raise DataError("chain block sizes do not add up to the header span")
                    closed = True
                else:
                    raise DataError(f"malformed chain line: {line!r}")
        self.n_chains = n_chains
        self.qnames = np.array(qnames, dtype=object)
        self.blocks: dict[str, dict[str, np.ndarray]] = {}
        self.n_overlapping_blocks = 0
        for name, rows in per_t.items():
            arr = np.array(rows, dtype=np.int64)
            arr = arr[np.argsort(arr[:, 0], kind="stable")]
            cummax_end = np.maximum.accumulate(arr[:, 1])
            self.n_overlapping_blocks += int((arr[1:, 0] < cummax_end[:-1]).sum())
            self.blocks[name] = {
                "t_start": arr[:, 0],
                "t_end": arr[:, 1],
                "q_start": arr[:, 2],
                "q_rev": arr[:, 3].astype(bool),
                "q_size": arr[:, 4],
                "q_code": arr[:, 5],
                "prev_max_end": np.concatenate(([-1], cummax_end[:-1])),
            }

    def lift(self, t_name: str, pos1: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Lift 1-based positions on ``t_name``.

        Returns ``(q_name, q_pos1, status)`` where ``status`` is 0 = mapped,
        1 = not covered by any chain block (liftOver: "Deleted in new"),
        2 = covered by more than one block (liftOver: "Duplicated in new").
        Unmapped entries have ``q_name`` None and ``q_pos1`` -1.
        """
        pos1 = np.asarray(pos1, dtype=np.int64)
        q_name = np.full(pos1.shape, None, dtype=object)
        q_pos = np.full(pos1.shape, -1, dtype=np.int64)
        status = np.ones(pos1.shape, dtype=np.int8)
        b = self.blocks.get(t_name)
        if b is None or pos1.size == 0:
            return q_name, q_pos, status
        p0 = pos1 - 1
        i = np.searchsorted(b["t_start"], p0, side="right") - 1
        valid = i >= 0
        ic = np.where(valid, i, 0)
        hit = valid & (p0 < b["t_end"][ic])
        # An earlier-starting block that also reaches past p0 means double coverage.
        earlier = valid & (b["prev_max_end"][ic] > p0)
        dup = hit & earlier
        only_earlier = ~hit & earlier  # covered solely by an earlier, longer block
        if only_earlier.any():
            # Resolve by scanning backwards for the covering block(s).
            for k in np.flatnonzero(only_earlier):
                cover = np.flatnonzero(
                    (b["t_start"][: ic[k] + 1] <= p0[k]) & (b["t_end"][: ic[k] + 1] > p0[k])
                )
                if cover.size == 1:
                    ic[k] = cover[0]
                    hit[k] = True
                elif cover.size > 1:
                    dup[k] = True
        good = hit & ~dup
        off = p0 - b["t_start"][ic]
        fwd = b["q_start"][ic] + off
        q0 = np.where(b["q_rev"][ic], b["q_size"][ic] - 1 - fwd, fwd)
        q_pos[good] = q0[good] + 1
        q_name[good] = self.qnames[b["q_code"][ic][good]]
        status[good] = 0
        status[dup] = 2
        return q_name, q_pos, status


def lift_positions(
    chain: str | os.PathLike[str] | ChainLiftover, chrom: np.ndarray, pos: np.ndarray
) -> tuple[np.ndarray, dict[str, int]]:
    """Convert autosomal positions to the other genome build.

    Parameters
    ----------
    chain
        Path of a UCSC chain file (optionally gzipped) from the build of ``pos`` to the
        wanted build, or a :class:`ChainLiftover`.
    chrom, pos
        Chromosome numbers (1–22) and 1-based positions.

    Returns
    -------
    numpy.ndarray
        ``int32`` positions in the other build; ``-1`` where the base is not covered by
        the chain, is covered more than once, or maps to a different chromosome.
    dict
        Counts: ``mapped``, ``deleted``, ``duplicated``, ``other_chromosome``.
    """
    lo = chain if isinstance(chain, ChainLiftover) else ChainLiftover(chain)
    chrom_a = np.asarray(chrom, dtype=np.int64)
    pos_a = np.asarray(pos, dtype=np.int64)
    out = np.full(pos_a.shape, -1, dtype=np.int32)
    stats = {"mapped": 0, "deleted": 0, "duplicated": 0, "other_chromosome": 0}
    for c in np.unique(chrom_a):
        sel = np.flatnonzero(chrom_a == c)
        q_name, q_pos, status = lo.lift(f"chr{int(c)}", pos_a[sel])
        same = np.array([name == f"chr{int(c)}" for name in q_name], dtype=bool)
        ok = (status == 0) & same
        out[sel[ok]] = q_pos[ok].astype(np.int32)
        stats["mapped"] += int(ok.sum())
        stats["deleted"] += int((status == 1).sum())
        stats["duplicated"] += int((status == 2).sum())
        stats["other_chromosome"] += int(((status == 0) & ~same).sum())
    return out, stats
