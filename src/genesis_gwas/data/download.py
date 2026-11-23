"""Reliable, verifiable retrieval of reference-bundle files.

Design goals (each addresses a way the original package's 1.6 GB monolithic download
fails in practice):

* **Small and lazy** — only the files an analysis needs are fetched (32 MB for the
  default configuration instead of 1.6 GB).
* **Resumable** — interrupted transfers continue with an HTTP ``Range`` request.
* **Verified** — every file is checked against the SHA-256 recorded in the manifest that
  ships *inside* the package, so a truncated or tampered download can never be used.
* **Atomic and concurrency-safe** — files appear under their final name only once
  complete and verified, and a lock file serialises concurrent fetches (e.g. many Slurm
  array tasks starting at once on a shared file system).
* **Proxy-friendly** — standard ``HTTPS_PROXY``/``https_proxy`` variables are honoured,
  which is all that compute nodes behind a proxy need.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from importlib import resources
from pathlib import Path
from typing import Any

from genesis_gwas._version import __version__
from genesis_gwas.exceptions import DownloadError, ReferenceDataError

__all__ = [
    "asset_name",
    "asset_url",
    "ensure_file",
    "fetch_bundle",
    "known_bundles",
    "load_packaged_manifest",
    "resolve_bundle_name",
    "sha256_file",
    "verify_bundle",
]

log = logging.getLogger("genesis_gwas")

# Bundles that can be downloaded: name -> release tag holding their assets.
_REMOTE_BUNDLES: dict[str, str] = {
    "eur_1kg_hm3": "refdata-v1",
    "eas_1kg_hm3": "refdata-v2",
    "afr_1kg_hm3": "refdata-v2",
}
# Short names accepted wherever a bundle name is (case-insensitive).
_ALIASES: dict[str, str] = {"eur": "eur_1kg_hm3", "eas": "eas_1kg_hm3", "afr": "afr_1kg_hm3"}
_DEFAULT_BASE_URL = "https://github.com/jeyabbalas/GENESIS/releases/download"
_CHUNK = 1 << 20


def known_bundles() -> tuple[str, ...]:
    """Names of the reference bundles that can be downloaded."""
    return tuple(_REMOTE_BUNDLES)


def resolve_bundle_name(name: str) -> str:
    """Canonical bundle name of a name or population alias (``"AFR"`` → ``"afr_1kg_hm3"``).

    Names that are not recognised are returned unchanged.
    """
    key = name.strip().lower()
    if key in _REMOTE_BUNDLES:
        return key
    return _ALIASES.get(key, name)


def load_packaged_manifest(name: str) -> dict[str, Any]:
    """Manifest (file list with checksums) of a downloadable bundle, as shipped in the package."""
    name = resolve_bundle_name(name)
    if name not in _REMOTE_BUNDLES:
        raise ReferenceDataError(
            f"Unknown reference bundle {name!r}; known: {sorted(_REMOTE_BUNDLES)}"
        )
    text = (resources.files("genesis_gwas") / "data" / "manifests" / f"{name}.json").read_text()
    return json.loads(text)


def asset_name(bundle: str, version: str, rel: str) -> str:
    """Flat release-asset file name of a bundle-relative path.

    ``("eur_1kg_hm3", "1.0", "ld/w1000kb_r0.1.npz")`` → ``"eur_1kg_hm3-1.0-ld-w1000kb_r0.1.npz"``.
    """
    return f"{bundle}-{version}-{rel.replace('/', '-')}"


def asset_url(bundle: str, rel: str) -> str:
    """Download URL of a bundle file (override the host with ``GENESIS_DATA_URL``)."""
    bundle = resolve_bundle_name(bundle)
    manifest = load_packaged_manifest(bundle)
    base = os.environ.get("GENESIS_DATA_URL", "").strip().rstrip("/")
    name = asset_name(bundle, str(manifest["version"]), rel)
    if base:
        return f"{base}/{name}"
    return f"{_DEFAULT_BASE_URL}/{_REMOTE_BUNDLES[bundle]}/{name}"


def sha256_file(path: Path) -> str:
    """Hex SHA-256 digest of a file."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(_CHUNK), b""):
            h.update(block)
    return h.hexdigest()


@contextlib.contextmanager
def _file_lock(target: Path, timeout: float = 3600.0, stale_after: float = 900.0) -> Iterator[None]:
    """Cross-node advisory lock based on atomic ``O_EXCL`` file creation.

    Works on network file systems where ``flock`` is unreliable. A lock whose file has
    not been touched for ``stale_after`` seconds is assumed to belong to a dead process.
    """
    lock = target.with_name(target.name + ".lock")
    deadline = time.monotonic() + timeout
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{os.getpid()}\n".encode())
            os.close(fd)
            break
        except FileExistsError:
            try:
                age = time.time() - lock.stat().st_mtime
            except FileNotFoundError:
                continue
            if age > stale_after:
                with contextlib.suppress(FileNotFoundError):
                    lock.unlink()
                continue
            if time.monotonic() > deadline:
                raise DownloadError(
                    f"Timed out waiting for the lock {lock}. If no other download is running, "
                    "delete that file and retry."
                ) from None
            time.sleep(1.0)
    try:
        yield
    finally:
        with contextlib.suppress(FileNotFoundError):
            lock.unlink()


def _download(
    url: str,
    dest: Path,
    expected_bytes: int | None,
    retries: int = 5,
    heartbeat: Path | None = None,
) -> None:
    """Download ``url`` to ``dest`` (a ``.part`` file), resuming partial content.

    ``heartbeat`` is a lock file whose modification time is refreshed while data
    arrives, so that a long transfer is never mistaken for a stale lock.
    """
    last: Exception | None = None
    for attempt in range(1, retries + 1):
        have = dest.stat().st_size if dest.exists() else 0
        if expected_bytes is not None and have == expected_bytes:
            return
        if expected_bytes is not None and have > expected_bytes:
            dest.unlink()
            have = 0
        req = urllib.request.Request(url, headers={"User-Agent": f"genesis-gwas/{__version__}"})
        if have:
            req.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                resumed = have > 0 and resp.status == 206
                mode = "ab" if resumed else "wb"
                with dest.open(mode) as out:
                    while True:
                        block = resp.read(_CHUNK)
                        if not block:
                            break
                        out.write(block)
                        if heartbeat is not None:
                            with contextlib.suppress(OSError):
                                os.utime(heartbeat)
            if expected_bytes is None or dest.stat().st_size == expected_bytes:
                return
            last = DownloadError(
                f"incomplete transfer: got {dest.stat().st_size} of {expected_bytes} bytes"
            )
        except urllib.error.HTTPError as exc:
            if exc.code == 416:  # range not satisfiable: start over
                with contextlib.suppress(FileNotFoundError):
                    dest.unlink()
                last = exc
            elif exc.code in (401, 403, 404):
                raise DownloadError(
                    f"The server refused {url} (HTTP {exc.code}). If you are behind a proxy or "
                    "firewall, download the bundle on another machine with "
                    "`genesis data fetch --dest DIR` and copy DIR here (see the documentation)."
                ) from exc
            else:
                last = exc
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            last = exc
        if attempt < retries:
            wait = min(60.0, 2.0**attempt)
            log.warning(
                "Download attempt %d/%d failed (%s); retrying in %.0f s",
                attempt,
                retries,
                last,
                wait,
            )
            time.sleep(wait)
    raise DownloadError(f"Could not download {url} after {retries} attempts: {last}")


def ensure_file(bundle: str, rel: str, root: Path) -> Path:
    """Make sure bundle file ``rel`` exists under ``root`` and matches its checksum.

    Returns the path. Downloads (or resumes) the file if necessary.
    """
    manifest = load_packaged_manifest(bundle)
    entry = (manifest.get("files") or {}).get(rel)
    if entry is None:
        raise ReferenceDataError(f"Bundle {bundle!r} has no file {rel!r}.")
    dest = root / rel
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    with _file_lock(dest):
        if dest.exists():  # another process finished while we waited
            return dest
        url = asset_url(bundle, rel)
        part = dest.with_name(dest.name + ".part")
        log.info("Downloading %s (%.1f MB)", url, entry["bytes"] / 1e6)
        _download(url, part, int(entry["bytes"]), heartbeat=dest.with_name(dest.name + ".lock"))
        digest = sha256_file(part)
        if digest != entry["sha256"]:
            part.unlink()
            raise DownloadError(
                f"Checksum mismatch for {rel} downloaded from {url}: expected "
                f"{entry['sha256']}, got {digest}. The partial file was removed; please retry."
            )
        part.replace(dest)
    return dest


def _select_files(manifest: dict[str, Any], components: str | list[str]) -> list[str]:
    files = sorted((manifest.get("files") or {}).keys())
    if components == "all":
        return files
    wanted = [components] if isinstance(components, str) else list(components)
    out: list[str] = ["snps.npz"]
    for comp in wanted:
        if comp == "default":
            out.append("ld/w1000kb_r0.1.npz")
        elif comp == "ld":
            out += [f for f in files if f.startswith("ld/")]
        elif comp == "pairld":
            out += [f for f in files if f.startswith("pairld/")]
        elif comp == "nullz":
            out += [f for f in files if f.startswith("nullz/")]
        elif comp == "examples":
            out += [f for f in files if f.startswith("examples/")]
        elif comp in files:
            out.append(comp)
        elif f"ld/{comp}.npz" in files:
            out.append(f"ld/{comp}.npz")
        else:
            raise ReferenceDataError(
                f"Unknown component {comp!r}. Use 'default', 'ld', 'pairld', 'nullz', 'examples', "
                f"'all', an LD configuration id, or a file path from: {files}"
            )
    seen: set[str] = set()
    return [f for f in out if not (f in seen or seen.add(f))]  # type: ignore[func-returns-value]


def fetch_bundle(
    bundle: str = "eur_1kg_hm3",
    components: str | list[str] = "default",
    data_dir: str | os.PathLike[str] | None = None,
) -> Path:
    r"""Download (parts of) a reference bundle into the data directory.

    Parameters
    ----------
    bundle
        Bundle name.
    components
        ``"default"`` (SNP table + the 1 Mb / :math:`r^2\ge0.1` configuration, 32 MB),
        ``"ld"`` (all nine LD configurations), ``"pairld"``, ``"nullz"`` (null Z-score
        replicates for QQ envelopes, 214 MB), ``"examples"``, ``"all"`` (504 MB), an LD
        configuration id such as ``"w2000kb_r0.05"``, or a list of these.
    data_dir
        Destination (default :func:`genesis_gwas.data.reference.default_data_dir`).

    Returns
    -------
    pathlib.Path
        The bundle directory, usable as ``reference=`` everywhere.
    """
    from genesis_gwas.data.reference import default_data_dir

    bundle = resolve_bundle_name(bundle)
    manifest = load_packaged_manifest(bundle)
    base = Path(data_dir).expanduser() if data_dir is not None else default_data_dir()
    root = base / bundle / str(manifest["version"])
    root.mkdir(parents=True, exist_ok=True)
    for rel in _select_files(manifest, components):
        ensure_file(bundle, rel, root)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return root


def verify_bundle(root: str | os.PathLike[str]) -> dict[str, str]:
    """Re-hash every file of a bundle directory against its manifest.

    Returns
    -------
    dict
        ``{relative path: "ok" | "missing" | "corrupt"}``.
    """
    rootp = Path(root).expanduser()
    mpath = rootp / "manifest.json"
    if not mpath.exists():
        raise ReferenceDataError(f"{rootp} is not a reference bundle (no manifest.json)")
    manifest = json.loads(mpath.read_text())
    status: dict[str, str] = {}
    for rel, entry in sorted((manifest.get("files") or {}).items()):
        p = rootp / rel
        if not p.exists():
            status[rel] = "missing"
        elif p.stat().st_size != entry["bytes"] or sha256_file(p) != entry["sha256"]:
            status[rel] = "corrupt"
        else:
            status[rel] = "ok"
    return status
