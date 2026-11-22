r"""LD reference bundles.

A *reference bundle* is a directory holding, for one reference panel (e.g. 1000 Genomes
EUR restricted to HapMap3 SNPs):

``snps.npz``
    The panel's SNP table in canonical order (sorted by chromosome, position, id):
    ``rsid``, ``chrom``, ``pos`` and optionally ``pos_grch38``, ``hm3_no_mhc``.
``ld/w{window_kb}kb_r{r2_min}.npz``
    One file per LD configuration: for every reference SNP the number of tagged SNPs
    ``nstar`` (:math:`N^*_k`), the bias-corrected LD score ``ldscore`` (:math:`\ell_k`),
    and the tagging neighbourhoods in CSR form (``indptr``, ``indices``; self included).
``pairld/...npz`` (optional)
    Pairwise :math:`r^2` aligned with ``indices`` (used to simulate marginal effects).
``nullz/*.npy`` (optional)
    LD-correlated null Z-score replicates (used for QQ-plot envelopes and
    genomic-control projections).
``manifest.json``
    Names, provenance and SHA-256 checksums.

The bundled European reference is a lossless conversion of the data shipped with the
original R package; :func:`genesis_gwas.ld.build_reference` creates bundles in the same
format from PLINK genotypes for any other population.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from functools import cached_property
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np

from genesis_gwas.core.variance import Neighborhoods
from genesis_gwas.exceptions import InputError, ReferenceDataError

__all__ = [
    "DEFAULT_REFERENCE",
    "LDConfig",
    "LDReference",
    "config_id",
    "default_data_dir",
    "list_packaged_references",
    "load_reference",
    "save_reference",
]

FORMAT_VERSION = 1
DEFAULT_REFERENCE = "eur_1kg_hm3"
PACKAGED = ("mini_chr22",)


def config_id(window_kb: float, r2_min: float) -> str:
    """Canonical identifier of an LD configuration, e.g. ``"w1000kb_r0.1"``."""
    w = float(window_kb)
    w_txt = str(int(w)) if w == int(w) else repr(w)
    return f"w{w_txt}kb_r{float(r2_min):g}"


def default_data_dir() -> Path:
    """Directory where downloaded reference bundles are cached.

    ``$GENESIS_DATA_DIR`` if set, otherwise ``$XDG_CACHE_HOME/genesis-gwas`` (normally
    ``~/.cache/genesis-gwas``). On HPC systems with small home quotas, point
    ``GENESIS_DATA_DIR`` at project storage.
    """
    env = os.environ.get("GENESIS_DATA_DIR", "").strip()
    if env:
        return Path(env).expanduser()
    base = os.environ.get("XDG_CACHE_HOME", "").strip() or str(Path.home() / ".cache")
    return Path(base).expanduser() / "genesis-gwas"


def list_packaged_references() -> tuple[str, ...]:
    """Names of the small reference bundles shipped inside the package (for demos/tests)."""
    return PACKAGED


def _packaged_path(name: str) -> Path:
    return Path(str(resources.files("genesis_gwas") / "datasets" / name))


@dataclass(frozen=True)
class LDConfig:
    r"""One LD configuration of a bundle.

    Attributes
    ----------
    id
        Identifier such as ``"w1000kb_r0.1"``.
    window_kb
        Half-width of the window in kb: SNPs within this distance (inclusive) are
        candidates for tagging.
    r2_min
        Tagging threshold: a reference SNP is tagged if :math:`r^2 \ge` ``r2_min``.
    file
        Bundle-relative path of the ``.npz`` file.
    pairld
        Bundle-relative path of the pairwise-:math:`r^2` file, if available.
    """

    id: str
    window_kb: float
    r2_min: float
    file: str
    pairld: str | None = None


@dataclass
class LDReference:
    r"""A reference panel with one selected LD configuration.

    Do not construct directly; use :func:`load_reference` or
    :func:`genesis_gwas.ld.build_reference`.

    Attributes
    ----------
    name, version
        Bundle identity.
    root
        Directory of the bundle on disk.
    config
        The selected :class:`LDConfig`.
    rsid
        ``(M,)`` SNP identifiers as fixed-width ASCII bytes.
    chrom, pos
        ``(M,)`` chromosome and base-pair position in ``genome_build`` coordinates.
    pos_grch38
        ``(M,)`` GRCh38 positions (``-1`` where unavailable), or ``None``.
    hm3_no_mhc
        ``(M,)`` flag of the HapMap3-without-MHC list used by the QC filter, or ``None``.
    download
        Whether files of the bundle that are missing locally may be downloaded on first use.
    nstar, ldscore
        ``(M,)`` :math:`N^*_k` and :math:`\ell_k` under ``config``.
    """

    name: str
    version: str
    root: Path
    manifest: dict[str, Any]
    config: LDConfig
    rsid: np.ndarray
    chrom: np.ndarray
    pos: np.ndarray
    nstar: np.ndarray
    ldscore: np.ndarray
    pos_grch38: np.ndarray | None = None
    hm3_no_mhc: np.ndarray | None = None
    download: bool = True
    _cache: dict[str, Any] = field(default_factory=dict, repr=False)

    # ----------------------------------------------------------------- basics
    @property
    def M(self) -> int:
        """Number of SNPs in the reference panel."""
        return int(self.rsid.size)

    @property
    def n_ref(self) -> int:
        """Sample size of the reference panel."""
        return int(self.manifest.get("n_ref", 0))

    @property
    def genome_build(self) -> str:
        """Genome build of ``pos`` (e.g. ``"GRCh37"``)."""
        return str(self.manifest.get("genome_build", "unknown"))

    @property
    def population(self) -> str:
        """Population label of the panel."""
        return str(self.manifest.get("population", "unknown"))

    @property
    def configs(self) -> list[LDConfig]:
        """All LD configurations available in the bundle."""
        return [_parse_config(c) for c in self.manifest.get("ld_configs", [])]

    def describe(self) -> str:
        """One-line description for logs."""
        return (
            f"{self.name} v{self.version} ({self.population}, {self.genome_build}, "
            f"M = {self.M:,}; window {self.config.window_kb:g} kb, r2 >= {self.config.r2_min:g})"
        )

    # ------------------------------------------------------------ lazy pieces
    @cached_property
    def neighborhoods(self) -> Neighborhoods:
        """Tagging neighbourhoods (loaded on first use; ~200 MB for the full panel)."""
        with np.load(self._resolve(self.config.file), allow_pickle=False) as z:
            return Neighborhoods(indptr=z["indptr"], indices=z["indices"])

    def pair_r2(self) -> np.ndarray:
        """Pairwise :math:`r^2` aligned with ``neighborhoods.indices`` (``float32``)."""
        if self.config.pairld is None:
            raise ReferenceDataError(
                f"Reference {self.name!r} has no pairwise r2 for configuration {self.config.id}."
            )
        if "pair_r2" not in self._cache:
            with np.load(self._resolve(self.config.pairld), allow_pickle=False) as z:
                if "r2_u16" in z.files:
                    self._cache["pair_r2"] = (z["r2_u16"].astype(np.float32)) / np.float32(65535.0)
                else:
                    self._cache["pair_r2"] = z["r2"].astype(np.float32)
        return self._cache["pair_r2"]

    @property
    def n_null_replicates(self) -> int:
        """Number of stored LD-correlated null Z-score replicates."""
        return int((self.manifest.get("nullz") or {}).get("n_replicates", 0))

    def null_z(self, replicate: int) -> np.ndarray:
        """Return null replicate ``replicate`` (0-based) as ``float32`` of length ``M``."""
        info = self.manifest.get("nullz") or {}
        n = int(info.get("n_replicates", 0))
        if not 0 <= replicate < n:
            raise ReferenceDataError(
                f"Null replicate {replicate} requested but reference {self.name!r} has {n}."
            )
        offset = 0
        for rel in info["files"]:
            arr = np.load(self._resolve(rel), mmap_mode="r", allow_pickle=False)
            if replicate < offset + arr.shape[0]:
                return np.asarray(arr[replicate - offset], dtype=np.float32)
            offset += arr.shape[0]
        raise ReferenceDataError("nullz manifest is inconsistent with its files")

    def example_path(self, name: str | None = None) -> Path:
        """Path of an example summary-statistics file shipped with the bundle."""
        ex = sorted(k for k in self.manifest.get("files", {}) if k.startswith("examples/"))
        if not ex:
            raise ReferenceDataError(f"Reference {self.name!r} ships no example data.")
        if name is None:
            rel = ex[0]
        else:
            matches = [k for k in ex if Path(k).name == name or k == name]
            if not matches:
                raise ReferenceDataError(f"No example named {name!r}; available: {ex}")
            rel = matches[0]
        return self._resolve(rel)

    def _resolve(self, rel: str) -> Path:
        path = self.root / rel
        if not path.exists():
            from genesis_gwas.data.download import ensure_file, known_bundles

            if self.name not in known_bundles():
                raise ReferenceDataError(f"Reference file {path} is missing.")
            if not self.download:
                raise ReferenceDataError(
                    f"Reference file {path} is missing and downloads are disabled. Fetch it "
                    f"beforehand on a machine with internet access (`genesis data fetch "
                    f"{self.name} --components all`)."
                )
            return ensure_file(self.name, rel, self.root)
        return path

    # ---------------------------------------------------------------- lookups
    @cached_property
    def _rsid_order(self) -> np.ndarray:
        return np.argsort(self.rsid, kind="stable")

    def lookup(self, ids: Any) -> np.ndarray:
        """Map SNP identifiers to reference rows (``-1`` where absent).

        Parameters
        ----------
        ids
            Sequence of identifiers (``str`` or ``bytes``).
        """
        q = _to_bytes(ids)
        order = self._rsid_order
        sorted_ids = self.rsid[order]
        pos = np.searchsorted(sorted_ids, q)
        pos[pos >= sorted_ids.size] = 0
        hit = sorted_ids[pos] == q
        out = np.full(q.shape, -1, dtype=np.int64)
        out[hit] = order[pos[hit]]
        return out

    def lookup_position(self, chrom: Any, pos: Any, build: str | None = None) -> np.ndarray:
        """Map (chromosome, position) pairs to reference rows (``-1`` where absent).

        Parameters
        ----------
        chrom, pos
            Chromosome numbers (1–22; ``"chr"`` prefixes are accepted) and positions.
        build
            ``"GRCh37"``/``"hg19"`` or ``"GRCh38"``/``"hg38"``. Defaults to the bundle's
            native build.
        """
        native = _norm_build(self.genome_build)
        want = native if build is None else _norm_build(build)
        if want == native:
            ref_pos = self.pos
        elif want == "GRCh38" and self.pos_grch38 is not None:
            ref_pos = self.pos_grch38
        else:
            raise ReferenceDataError(
                f"Reference {self.name!r} has no {want} positions (native build {native})."
            )
        ok = ref_pos > 0
        key_ref = self.chrom.astype(np.int64)[ok] * (1 << 32) + ref_pos.astype(np.int64)[ok]
        rows = np.flatnonzero(ok)
        order = np.argsort(key_ref, kind="stable")
        key_sorted = key_ref[order]
        if key_sorted.size > 1 and np.any(key_sorted[1:] == key_sorted[:-1]):
            dup = np.zeros(key_sorted.size, dtype=bool)
            same = key_sorted[1:] == key_sorted[:-1]
            dup[1:] |= same
            dup[:-1] |= same
        else:
            dup = np.zeros(key_sorted.size, dtype=bool)
        c = _chrom_to_int(chrom)
        p = np.asarray(pos, dtype=np.int64)
        q = c * (1 << 32) + p
        loc = np.searchsorted(key_sorted, q)
        loc[loc >= key_sorted.size] = 0
        hit = (key_sorted[loc] == q) & ~dup[loc] & (c > 0)
        out = np.full(q.shape, -1, dtype=np.int64)
        out[hit] = rows[order[loc[hit]]]
        return out

    def subset_ids(self, rows: np.ndarray) -> np.ndarray:
        """Identifiers of the given reference rows as ``str``."""
        return self.rsid[rows].astype("U")


def _norm_build(build: str) -> str:
    b = str(build).strip().lower()
    if b in ("grch37", "hg19", "b37", "37"):
        return "GRCh37"
    if b in ("grch38", "hg38", "b38", "38"):
        return "GRCh38"
    raise InputError(f"Unknown genome build {build!r} (use GRCh37/hg19 or GRCh38/hg38)")


def _chrom_to_int(chrom: Any) -> np.ndarray:
    arr = np.asarray(chrom)
    if arr.dtype.kind in "iu":
        return arr.astype(np.int64)
    if arr.dtype.kind == "f":
        out = np.where(np.isfinite(arr), arr, 0).astype(np.int64)
        return out
    txt = np.char.lower(np.asarray(arr, dtype="U"))
    txt = np.char.replace(txt, "chr", "")
    out = np.zeros(txt.shape, dtype=np.int64)
    for i in range(1, 23):
        out[txt == str(i)] = i
    return out


def _to_bytes(ids: Any) -> np.ndarray:
    arr = np.asarray(ids)
    if arr.dtype.kind == "S":
        return arr
    if arr.dtype.kind == "U":
        try:
            return np.char.encode(arr, "ascii")
        except UnicodeEncodeError:
            return np.array([s.encode("utf-8", "replace") for s in arr.tolist()], dtype="S")
    return np.array([str(s).encode("utf-8", "replace") for s in arr.tolist()], dtype="S")


def _parse_config(c: dict[str, Any]) -> LDConfig:
    return LDConfig(
        id=str(c["id"]),
        window_kb=float(c["window_kb"]),
        r2_min=float(c["r2_min"]),
        file=str(c["file"]),
        pairld=c.get("pairld"),
    )


def _select_config(
    manifest: dict[str, Any], window_kb: float | None, r2_min: float | None, name: str
) -> LDConfig:
    configs = [_parse_config(c) for c in manifest.get("ld_configs", [])]
    if not configs:
        raise ReferenceDataError(f"Reference {name!r} declares no LD configurations.")
    if window_kb is None and r2_min is None:
        preferred = [c for c in configs if c.window_kb == 1000 and c.r2_min == 0.1]
        if preferred:
            return preferred[0]
        if len(configs) == 1:
            return configs[0]
        raise InputError(
            f"Reference {name!r} has several LD configurations and none is the default "
            f"(1000 kb, r2 >= 0.1); choose one of {[c.id for c in configs]}."
        )
    matches = [
        c
        for c in configs
        if (window_kb is None or np.isclose(c.window_kb, window_kb))
        and (r2_min is None or np.isclose(c.r2_min, r2_min))
    ]
    if len(matches) > 1:
        # Only one of the two was given: the other takes its default (1000 kb / 0.1).
        preferred = [
            c
            for c in matches
            if (window_kb is not None or c.window_kb == 1000)
            and (r2_min is not None or c.r2_min == 0.1)
        ]
        if len(preferred) == 1:
            return preferred[0]
        raise InputError(
            f"Reference {name!r}: window_kb={window_kb}, r2_min={r2_min} matches several LD "
            f"configurations ({[c.id for c in matches]}); give both window_kb and r2_min."
        )
    if matches:
        return matches[0]
    raise InputError(
        f"Reference {name!r} has no LD configuration with window_kb={window_kb}, r2_min={r2_min}; "
        f"available: {[c.id for c in configs]}."
    )


def load_reference(
    reference: str | os.PathLike[str] = DEFAULT_REFERENCE,
    *,
    window_kb: float | None = None,
    r2_min: float | None = None,
    data_dir: str | os.PathLike[str] | None = None,
    download: bool = True,
) -> LDReference:
    r"""Load an LD reference bundle.

    Parameters
    ----------
    reference
        One of

        * a path to a bundle directory (containing ``manifest.json``),
        * ``"eur_1kg_hm3"`` — the European reference of the original software
          (1000 Genomes Phase 3 EUR, 1,070,777 HapMap3 SNPs); downloaded on first use
          into :func:`default_data_dir` and verified against recorded SHA-256 checksums,
        * ``"mini_chr22"`` — a 2 MB chromosome-22 slice of the above, shipped inside the
          package for demonstrations and tests.
    window_kb, r2_min
        LD configuration (default: 1000 kb and :math:`r^2\ge0.1`, the paper's primary
        choice). The bundled reference offers windows 500/1000/2000 kb and thresholds
        0.05/0.1/0.2.
    data_dir
        Where downloaded bundles live (default :func:`default_data_dir`).
    download
        Permit downloading missing files, now and later for the optional pieces that are
        only fetched when first used (null replicates, pairwise :math:`r^2`, examples).
        With ``False`` a missing file raises
        :class:`~genesis_gwas.exceptions.ReferenceDataError`.
    """
    ref_txt = os.fspath(reference)
    path = Path(ref_txt).expanduser()
    if path.is_dir() and (path / "manifest.json").exists():
        root = path
        manifest = json.loads((root / "manifest.json").read_text())
        name = str(manifest.get("name", path.name))
    elif ref_txt in PACKAGED:
        root = _packaged_path(ref_txt)
        manifest = json.loads((root / "manifest.json").read_text())
        name = ref_txt
    else:
        from genesis_gwas.data.download import known_bundles, load_packaged_manifest

        if ref_txt not in known_bundles():
            raise ReferenceDataError(
                f"{ref_txt!r} is neither a bundle directory (no manifest.json found) nor a known "
                f"reference name. Known names: {sorted((*known_bundles(), *PACKAGED))}."
            )
        name = ref_txt
        manifest = load_packaged_manifest(name)
        base = Path(data_dir).expanduser() if data_dir is not None else default_data_dir()
        root = base / name / str(manifest["version"])

    if int(manifest.get("format_version", 0)) > FORMAT_VERSION:
        raise ReferenceDataError(
            f"Reference {name!r} uses bundle format {manifest.get('format_version')}, newer than "
            f"this version of genesis-gwas supports ({FORMAT_VERSION}). Please upgrade."
        )
    config = _select_config(manifest, window_kb, r2_min, name)

    def need(rel: str) -> Path:
        p = root / rel
        if p.exists():
            return p
        if not download:
            raise ReferenceDataError(
                f"Reference file {p} is missing and download=False. Run "
                f"`genesis data fetch {name}` on a machine with internet access."
            )
        from genesis_gwas.data.download import ensure_file

        return ensure_file(name, rel, root)

    with np.load(need("snps.npz"), allow_pickle=False) as z:
        rsid = z["rsid"]
        chrom = z["chrom"]
        pos = z["pos"]
        pos38 = z["pos_grch38"] if "pos_grch38" in z.files else None
        hm3 = z["hm3_no_mhc"] if "hm3_no_mhc" in z.files else None
    with np.load(need(config.file), allow_pickle=False) as z:
        nstar = z["nstar"].astype(np.int64)
        ldscore = z["ldscore"].astype(np.float64)
    if not (rsid.size == chrom.size == pos.size == nstar.size == ldscore.size):
        raise ReferenceDataError(
            f"Reference {name!r} is corrupt: arrays have inconsistent lengths."
        )
    return LDReference(
        name=name,
        version=str(manifest.get("version", "0")),
        root=root,
        manifest=manifest,
        config=config,
        rsid=rsid,
        chrom=chrom,
        pos=pos,
        nstar=nstar,
        ldscore=ldscore,
        pos_grch38=pos38,
        hm3_no_mhc=hm3,
        download=download,
    )


def save_reference(
    out_dir: str | os.PathLike[str],
    *,
    name: str,
    rsid: Any,
    chrom: Any,
    pos: Any,
    configs: list[dict[str, Any]],
    version: str = "1.0",
    population: str = "unknown",
    panel: str = "",
    genome_build: str = "GRCh37",
    n_ref: int = 0,
    pos_grch38: Any = None,
    hm3_no_mhc: Any = None,
    null_z: np.ndarray | None = None,
    source: dict[str, Any] | None = None,
) -> Path:
    """Write a reference bundle to ``out_dir``.

    Parameters
    ----------
    out_dir
        Destination directory (created if needed).
    name, version, population, panel, genome_build, n_ref, source
        Metadata recorded in ``manifest.json``.
    rsid, chrom, pos
        SNP table. Rows must already be in canonical order (chromosome, position).
    configs
        One dict per LD configuration with keys ``window_kb``, ``r2_min``, ``nstar``,
        ``ldscore``, ``indptr``, ``indices`` and optionally ``r2`` (pairwise
        :math:`r^2` aligned with ``indices``).
    pos_grch38, hm3_no_mhc
        Optional extra SNP columns.
    null_z
        Optional ``(n_replicates, M)`` array of LD-correlated null Z-scores.

    Returns
    -------
    pathlib.Path
        The bundle directory.
    """
    import hashlib

    root = Path(out_dir).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    ids = _to_bytes(rsid)
    chrom_a = np.asarray(chrom, dtype=np.uint8)
    pos_a = np.asarray(pos, dtype=np.int32)
    M = ids.size
    if not (chrom_a.size == pos_a.size == M):
        raise InputError("rsid, chrom and pos must have the same length")
    if np.unique(ids).size != M:
        raise InputError("SNP identifiers must be unique")
    key = chrom_a.astype(np.int64) * (1 << 32) + pos_a.astype(np.int64)
    if np.any(np.diff(key) < 0):
        raise InputError("SNPs must be sorted by chromosome and position")

    files: dict[str, dict[str, Any]] = {}

    def record(rel: str) -> None:
        p = root / rel
        h = hashlib.sha256()
        with p.open("rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
        files[rel] = {"sha256": h.hexdigest(), "bytes": p.stat().st_size}

    snp_arrays: dict[str, np.ndarray] = {"rsid": ids, "chrom": chrom_a, "pos": pos_a}
    if pos_grch38 is not None:
        snp_arrays["pos_grch38"] = np.asarray(pos_grch38, dtype=np.int32)
    if hm3_no_mhc is not None:
        snp_arrays["hm3_no_mhc"] = np.asarray(hm3_no_mhc, dtype=bool)
    np.savez_compressed(root / "snps.npz", **snp_arrays)  # type: ignore[arg-type]
    record("snps.npz")

    cfg_entries = []
    (root / "ld").mkdir(exist_ok=True)
    for cfg in configs:
        cid = config_id(cfg["window_kb"], cfg["r2_min"])
        nstar = np.asarray(cfg["nstar"], dtype=np.int32)
        ldscore = np.asarray(cfg["ldscore"], dtype=np.float64)
        indptr = np.asarray(cfg["indptr"], dtype=np.int64)
        indices = np.asarray(cfg["indices"], dtype=np.int32)
        if nstar.size != M or ldscore.size != M or indptr.size != M + 1:
            raise InputError(f"LD configuration {cid}: arrays do not match the SNP table")
        if not np.array_equal(np.diff(indptr), nstar):
            raise InputError(f"LD configuration {cid}: nstar must equal the neighbourhood sizes")
        rel = f"ld/{cid}.npz"
        np.savez_compressed(
            root / rel,
            nstar=nstar,
            ldscore=ldscore,
            indptr=indptr,
            indices=indices,
            window_kb=np.float64(cfg["window_kb"]),
            r2_min=np.float64(cfg["r2_min"]),
            n_ref=np.int64(n_ref),
        )
        record(rel)
        pair_rel = None
        if cfg.get("r2") is not None:
            r2 = np.asarray(cfg["r2"], dtype=np.float64)
            if r2.size != indices.size:
                raise InputError(f"LD configuration {cid}: r2 must align with indices")
            (root / "pairld").mkdir(exist_ok=True)
            pair_rel = f"pairld/{cid}.npz"
            np.savez_compressed(
                root / pair_rel, r2_u16=np.round(np.clip(r2, 0, 1) * 65535.0).astype(np.uint16)
            )
            record(pair_rel)
        cfg_entries.append(
            {
                "id": cid,
                "window_kb": float(cfg["window_kb"]),
                "r2_min": float(cfg["r2_min"]),
                "file": rel,
                "pairld": pair_rel,
            }
        )

    nullz_entry = None
    if null_z is not None:
        nz = np.asarray(null_z)
        if nz.ndim != 2 or nz.shape[1] != M:
            raise InputError("null_z must have shape (n_replicates, M)")
        (root / "nullz").mkdir(exist_ok=True)
        names = []
        for start in range(0, nz.shape[0], 10):
            stop = min(nz.shape[0], start + 10)
            rel = f"nullz/nullz_{start:03d}-{stop - 1:03d}.npy"
            np.save(root / rel, nz[start:stop].astype(np.float16), allow_pickle=False)
            record(rel)
            names.append(rel)
        nullz_entry = {"dtype": "float16", "n_replicates": int(nz.shape[0]), "files": names}

    manifest = {
        "format_version": FORMAT_VERSION,
        "name": name,
        "version": version,
        "population": population,
        "panel": panel,
        "genome_build": genome_build,
        "n_ref": int(n_ref),
        "n_snps": int(M),
        "source": source or {},
        "ld_configs": cfg_entries,
        "nullz": nullz_entry,
        "files": files,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return root
