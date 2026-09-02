from __future__ import annotations

import pytest

from omics_agent.data_sources.classify import classify_filename, is_raw_filename, reject_raw
from omics_agent.errors import UnsupportedRawDataError
from omics_agent.schemas.enums import FileRole


def test_fastq_and_mzml_are_raw() -> None:
    assert is_raw_filename("sample_R1.fastq.gz")
    assert is_raw_filename("run.mzML")
    assert is_raw_filename("orbitrap.raw")
    assert classify_filename("sample_R1.fastq.gz") is FileRole.REJECTED_RAW


def test_series_matrix_is_processed() -> None:
    assert classify_filename("GSE1_series_matrix.txt.gz") is FileRole.MATRIX
    assert classify_filename("protein_groups.tsv") is FileRole.MATRIX


def test_reject_raw_explains_fix() -> None:
    with pytest.raises(UnsupportedRawDataError, match="processed"):
        reject_raw("lane.fastq")
