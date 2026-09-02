"""Classify filenames as processed matrix, archive, or forbidden raw data.

Classification uses the filename only. A publisher label that says
"processed" is recorded as untrusted metadata; it does not override a
``.fastq`` / ``.mzML`` suffix.
"""

from __future__ import annotations

from omics_agent.errors import UnsupportedRawDataError
from omics_agent.schemas.enums import FileRole

_COMPRESSION = (".gz", ".bz2", ".xz", ".zip", ".tar")
_RAW_MARKERS = (
    ".fastq",
    ".fq",
    ".sra",
    ".bam",
    ".cram",
    ".raw",
    ".mzml",
    ".mzxml",
    ".wiff",
    ".d",
    ".lcd",
    ".baf",
)
_MATRIX_MARKERS = (
    ".tsv",
    ".csv",
    ".txt",
    ".tab",
    ".parquet",
    "series_matrix",
    "_series_matrix.txt",
)
_ARCHIVE_MARKERS = (".zip", ".tar", ".tar.gz", ".tgz", ".7z")


def _strip_compression(name: str) -> str:
    lower = name.lower()
    if lower.endswith(".tar.gz") or lower.endswith(".tgz"):
        return lower[: -len(".tar.gz")] if lower.endswith(".tar.gz") else lower[: -len(".tgz")]
    for suffix in (".gz", ".bz2", ".xz"):
        if lower.endswith(suffix):
            return lower[: -len(suffix)]
    return lower


def is_raw_filename(filename: str) -> bool:
    """Return True for FASTQ, SRA, BAM, and vendor raw mass-spec names."""

    core = _strip_compression(filename)
    lower = filename.lower()
    if any(lower.endswith(ext) for ext in (".fastq.gz", ".fq.gz", ".fastq.bz2", ".fq.bz2")):
        return True
    return any(core.endswith(marker) for marker in _RAW_MARKERS)


def is_archive_filename(filename: str) -> bool:
    lower = filename.lower()
    return any(lower.endswith(marker) for marker in _ARCHIVE_MARKERS)


def classify_filename(filename: str) -> FileRole:
    """Assign a file role from the name. Unknown is not promoted to matrix."""

    if is_raw_filename(filename):
        return FileRole.REJECTED_RAW
    if is_archive_filename(filename):
        return FileRole.ARCHIVE
    core = _strip_compression(filename)
    if any(marker in core for marker in _MATRIX_MARKERS):
        return FileRole.MATRIX
    if "sample" in core and any(core.endswith(ext) for ext in (".tsv", ".csv", ".txt")):
        return FileRole.SAMPLE_SHEET
    return FileRole.UNKNOWN


def reject_raw(filename: str) -> None:
    """Raise if ``filename`` is raw sequencing or raw mass-spec."""

    if not is_raw_filename(filename):
        return
    raise UnsupportedRawDataError(
        f"'{filename}' looks like raw sequencing or raw mass-spec. "
        "Milestone 2 only accepts author-provided processed matrices.",
        how_to_fix=(
            "Use the processed count/abundance table the authors deposited "
            "(series matrix, protein groups, quantified metabolites). "
            "Do not point the pipeline at FASTQ, SRA, mzML, or vendor .raw files."
        ),
    )
