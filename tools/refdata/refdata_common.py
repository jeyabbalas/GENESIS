"""Shared helpers for the reference-data conversion tools (numpy + stdlib only).

These scripts turn the reference data bundled with upstream GENESIS
(https://github.com/yandorazhang/GENESIS, commit e4e6894) into the compact
"reference bundle" (format_version 1) consumed by the refactored package.
See README.md in this directory for the format specification.
"""

from __future__ import annotations

import gzip
import hashlib
import zipfile
from pathlib import Path

import numpy as np

FORMAT_VERSION = 1
N_REF = 489
UPSTREAM_REPO = "https://github.com/yandorazhang/GENESIS"
UPSTREAM_COMMIT = "e4e6894d2e11ece6715d0572b0febceff1480f85"
PANEL = "1000 Genomes Phase 3, LDSC EUR subset (n=489), HapMap3 SNPs with MAF>=0.05"

# (label used in upstream file names, window in kb)
WINDOWS = (("0.5", 500), ("1", 1000), ("2", 2000))
CUTOFFS = ("0.05", "0.1", "0.2")
PAIR_CFG = "w1000kb_r0.1"  # the only configuration upstream ships pairwise r^2 for
NULLZ_PER_FILE = 10
N_NULLZ = 100


def cfg_id(window_kb: int, cutoff: str) -> str:
    """Bundle id of an LD configuration, e.g. ``w1000kb_r0.1``."""
    return f"w{window_kb}kb_r{cutoff}"


def raw_ld_dir(window_label: str, cutoff: str) -> str:
    """Directory name written by export_rdata.R for an LD configuration."""
    return f"ld_w{window_label}MB_c{cutoff}"


def all_configs() -> list[dict]:
    """All nine upstream configurations, ordered by (window, cutoff)."""
    out = []
    for w_label, w_kb in WINDOWS:
        for c in CUTOFFS:
            out.append(
                {
                    "id": cfg_id(w_kb, c),
                    "window_kb": w_kb,
                    "r2_min": float(c),
                    "cutoff_label": c,
                    "raw_dir": raw_ld_dir(w_label, c),
                }
            )
    return out


# --------------------------------------------------------------------------- raw I/O
def read_names(path: Path) -> np.ndarray:
    """Read one identifier per line into a fixed-width ASCII bytes array."""
    data = Path(path).read_bytes()
    names = data.split(b"\n")
    if names and names[-1] == b"":
        names.pop()
    if any(n.endswith(b"\r") for n in names[:1000]):
        raise ValueError(f"{path}: unexpected CR line endings")
    arr = np.array(names, dtype="S")
    if arr.size and (np.char.str_len(arr) == 0).any():
        raise ValueError(f"{path}: empty identifier")
    return arr


def read_i32(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype="<i4")


def read_f64(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype="<f8")


def index_in(query: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Positions ``idx`` with ``ref[idx] == query`` (``ref`` unique; -1 where absent)."""
    order = np.argsort(ref, kind="stable")
    sorted_ref = ref[order]
    if sorted_ref.size > 1 and (sorted_ref[1:] == sorted_ref[:-1]).any():
        raise ValueError("reference identifiers are not unique")
    pos = np.searchsorted(sorted_ref, query)
    pos = np.minimum(pos, sorted_ref.size - 1)
    found = sorted_ref[pos] == query
    out = np.where(found, order[pos], -1).astype(np.int64)
    return out


def canonical_order(chrom: np.ndarray, pos: np.ndarray, rsid: np.ndarray) -> np.ndarray:
    """Permutation that sorts SNPs by (chrom, pos, rsid) ascending."""
    return np.lexsort((rsid, pos, chrom))


# ------------------------------------------------------------------- deterministic npz
_ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)


def save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write a ``np.savez_compressed``-compatible archive with fixed metadata.

    ``np.savez_compressed`` stamps every member with the current time, so its output is
    not byte-reproducible. This writer produces the same container (a deflated zip of
    ``<key>.npy`` members, no pickles) with constant timestamps and permissions, so the
    bytes depend only on the arrays, the numpy ``.npy`` header format and the zlib build.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        for key, value in arrays.items():
            arr = np.asanyarray(value)
            if arr.dtype.hasobject:
                raise TypeError(f"{key}: object arrays are not allowed")
            info = zipfile.ZipInfo(key + ".npy", date_time=_ZIP_EPOCH)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            with zf.open(info, "w", force_zip64=True) as fh:
                np.lib.format.write_array(fh, arr, allow_pickle=False)
    tmp.replace(path)


def save_npy(path: Path, arr: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.lib.format.write_array(fh, np.ascontiguousarray(arr), allow_pickle=False)
    tmp.replace(path)


def gzip_deterministic(src_bytes: bytes, dst: Path, level: int = 9) -> None:
    """gzip with no embedded file name and mtime=0 so the output is reproducible."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    with open(tmp, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0,
                                               compresslevel=level) as gz:
        gz.write(src_bytes)
    tmp.replace(dst)


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


# ------------------------------------------------------------------------ CSR helpers
def csr_from_pairs(rows: np.ndarray, cols: np.ndarray, n: int, *extra: np.ndarray):
    """Build CSR (indptr, indices) with rows ascending and columns ascending within a row.

    ``extra`` arrays (aligned with the input pairs) are returned in the same final order.
    """
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    if rows.size and (rows.min() < 0 or rows.max() >= n or cols.min() < 0 or cols.max() >= n):
        raise ValueError("row/column index out of range")
    key = rows * np.int64(n) + cols  # one int64 key: row-major order
    order = np.argsort(key, kind="stable")
    key = key[order]
    if key.size > 1 and (key[1:] == key[:-1]).any():
        raise ValueError("duplicate (row, column) pair in neighbour lists")
    del key
    indices = cols[order].astype(np.int32)
    counts = np.bincount(rows, minlength=n)
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    return (indptr, indices, *[e[order] for e in extra])


def csr_rows(indptr: np.ndarray) -> np.ndarray:
    """Row id of every stored element of a CSR structure."""
    return np.repeat(np.arange(indptr.size - 1, dtype=np.int64), np.diff(indptr))


# ---------------------------------------------------------------------------- liftover
class ChainLiftover:
    """Point liftover through a UCSC ``*.over.chain`` file (pure Python/numpy).

    Chain coordinates are 0-based half-open; reverse-strand query coordinates are
    relative to the reverse-complemented sequence, so a base at reverse coordinate ``x``
    sits at forward 0-based position ``qSize - 1 - x``. ``lift`` takes and returns
    1-based positions.
    """

    def __init__(self, chain_gz: Path):
        per_t: dict[str, list[tuple]] = {}
        qnames: list[str] = []
        qcode: dict[str, int] = {}
        n_chains = 0
        with gzip.open(chain_gz, "rt") as fh:
            t_name = q_name = None
            t = q = t_end = q_end = q_size = 0
            q_rev = False
            closed = True
            for line in fh:
                if line.startswith("chain"):
                    if not closed:
                        raise ValueError("chain block not terminated")
                    f = line.split()
                    t_name, t_strand, t, t_end = f[2], f[4], int(f[5]), int(f[6])
                    q_name, q_size, q_strand, q, q_end = f[7], int(f[8]), f[9], int(f[10]), int(f[11])
                    if t_strand != "+":
                        raise ValueError("target strand '-' is not supported")
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
                        raise ValueError("chain block sizes do not add up to the header span")
                    closed = True
                else:
                    raise ValueError(f"malformed chain line: {line!r}")
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

    def lift(self, t_name: str, pos1: np.ndarray):
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
                cover = np.flatnonzero((b["t_start"][: ic[k] + 1] <= p0[k]) & (b["t_end"][: ic[k] + 1] > p0[k]))
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
