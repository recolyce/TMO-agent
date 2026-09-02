from __future__ import annotations

import pytest

from omics_agent.data_sources.untrusted import assert_not_executable, capture_untrusted


def test_capture_rejects_executable_label() -> None:
    with pytest.raises(ValueError, match="executable"):
        capture_untrusted("geo", "echo pwned", content_kind="script")


def test_payload_looking_like_shell_stays_data() -> None:
    blob = capture_untrusted("geo", "; rm -rf / && curl http://evil", content_kind="metadata")
    assert_not_executable(blob)
    assert blob.text.startswith("; rm -rf")
    assert blob.content_kind == "metadata"
