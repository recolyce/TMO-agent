"""Content hashes for code, data, splits, configs, and the environment.

Every scientific artifact that can change a result must be recorded. Hashes
are SHA-256 hex digests. They are not a substitute for git history.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from omics_agent import __version__


def sha256_bytes(data: bytes) -> str:
    """Return the SHA-256 hex digest of ``data``."""

    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    """Hash UTF-8 text."""

    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path) -> str:
    """Hash a file in 1 MiB chunks.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist. The caller should turn this into a
        beginner-facing error.
    """

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def md5_file(path: Path) -> str:
    """MD5 hex digest used only to verify a publisher-provided MD5.

    Do not use MD5 as the pipeline's own integrity hash. Record SHA-256
    alongside any official MD5 check.
    """

    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(payload: Any) -> bytes:
    """Serialize ``payload`` with sorted keys for stable hashing."""

    return json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")


def hash_mapping(payload: dict[str, Any]) -> str:
    """Hash a JSON-serializable mapping."""

    return sha256_bytes(canonical_json(payload))


def hash_yaml_file(path: Path) -> str:
    """Parse YAML then hash the canonical JSON form so comments do not matter."""

    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    return hash_mapping(loaded if isinstance(loaded, dict) else {"_raw": loaded})


def hash_dataframe(frame: pd.DataFrame) -> str:
    """Hash a DataFrame by sorted column names and row-wise CSV bytes.

    Index values are included as a column so sample identity is part of the
    hash. Floating-point values use a fixed format to avoid noisy hashes.
    """

    ordered = frame.copy()
    if ordered.index.name or not ordered.index.equals(pd.RangeIndex(len(ordered))):
        ordered = ordered.reset_index()
    ordered = ordered.reindex(sorted(ordered.columns.astype(str)), axis=1)
    csv_bytes = ordered.to_csv(index=False, float_format="%.10g").encode("utf-8")
    return sha256_bytes(csv_bytes)


def hash_source_tree(root: Path) -> str:
    """Hash all ``*.py`` files under ``root`` in sorted path order."""

    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*.py") if path.is_file())
    if not files:
        raise FileNotFoundError(f"No Python files under {root}")
    for path in files:
        rel = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(rel)
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def git_commit(repo: Path) -> str | None:
    """Return HEAD SHA if ``repo`` is a git checkout, otherwise ``None``."""

    try:
        result = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def environment_hash(lock_path: Path | None) -> str:
    """Hash ``uv.lock`` when present; otherwise hash the package version."""

    if lock_path is not None and lock_path.is_file():
        return sha256_file(lock_path)
    return sha256_text(f"omics-agent=={__version__};lock=missing")


def collect_run_hashes(
    *,
    package_root: Path,
    repo_root: Path,
    data_frames: dict[str, pd.DataFrame],
    split_frame: pd.DataFrame | None,
    config_payload: dict[str, Any],
    seed: int,
) -> dict[str, str]:
    """Collect the hashes that every MLflow run must record.

    Returns a flat string mapping suitable for tags and a JSON sidecar.
    """

    hashes = {
        "package_version": __version__,
        "code_hash": hash_source_tree(package_root),
        "config_hash": hash_mapping(config_payload),
        "seed": str(seed),
        "environment_hash": environment_hash(repo_root / "uv.lock"),
    }
    commit = git_commit(repo_root)
    if commit is not None:
        hashes["git_commit"] = commit
    for name, frame in sorted(data_frames.items()):
        hashes[f"data_hash.{name}"] = hash_dataframe(frame)
    if data_frames:
        hashes["data_hash"] = hash_mapping(
            {name: hash_dataframe(frame) for name, frame in sorted(data_frames.items())}
        )
    if split_frame is not None:
        hashes["split_hash"] = hash_dataframe(split_frame)
    return hashes
