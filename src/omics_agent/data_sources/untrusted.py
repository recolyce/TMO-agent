"""Hold repository/paper prose so it cannot be treated as a command.

Downloaded HTML, SOFT summaries, and abstracts are data. They are not
shell scripts, Python source, or file paths.
"""

from __future__ import annotations

from datetime import UTC, datetime

from omics_agent.hashing import sha256_text
from omics_agent.schemas.ingest import UntrustedText

_FORBIDDEN_KINDS = {"script", "executable", "shell"}


def capture_untrusted(source: str, text: str, *, content_kind: str = "metadata") -> UntrustedText:
    """Wrap prose as :class:`UntrustedText`.

    Raises
    ------
    ValueError
        If a caller tries to label the payload as executable.
    """

    kind = content_kind.strip().lower()
    if kind in _FORBIDDEN_KINDS:
        raise ValueError(
            "Untrusted repository text cannot be labelled as executable. "
            "Store it as metadata or html."
        )
    cleaned = text.replace("\x00", "")
    return UntrustedText(
        source=source,
        retrieved_at=datetime.now(UTC).isoformat(),
        sha256=sha256_text(cleaned),
        text=cleaned,
        content_kind=kind,
    )


def assert_not_executable(blob: UntrustedText) -> None:
    """Guard used by tests and the ingest orchestrator."""

    if blob.content_kind in _FORBIDDEN_KINDS:
        raise ValueError(f"Refusing to handle untrusted text as {blob.content_kind}.")
