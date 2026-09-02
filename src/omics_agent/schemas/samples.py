"""Sample sheet contract.

One row is one modality measurement of one biospecimen. Alignment across
modalities is proven by ``biospecimen_id``, never by similar file names.
"""

from __future__ import annotations

from typing import Self

import pandas as pd
from pydantic import Field, field_validator, model_validator

from omics_agent.errors import NeedsReviewError, SchemaError
from omics_agent.schemas.dataset import StrictModel
from omics_agent.schemas.enums import ReplicateType, SamplingDesign

REQUIRED_SAMPLE_COLUMNS: tuple[str, ...] = (
    "sample_id",
    "observation_id",
    "experimental_unit_id",
    "subject_id",
    "biospecimen_id",
    "time",
    "time_unit",
    "condition",
    "batch",
    "modality",
    "file_id",
    "replicate_type",
)

NA_TOKENS = {"", "NA", "NaN", "nan", "None", "null", "."}


class SampleRow(StrictModel):
    """One validated sample-sheet row."""

    sample_id: str
    observation_id: str = Field(
        description="Shared key for modalities taken from the same biospecimen/time."
    )
    experimental_unit_id: str
    subject_id: str
    biospecimen_id: str
    time: float
    time_unit: str
    condition: str
    batch: str
    modality: str
    file_id: str
    replicate_type: ReplicateType

    @field_validator(
        "sample_id",
        "observation_id",
        "experimental_unit_id",
        "subject_id",
        "biospecimen_id",
        "condition",
        "batch",
        "modality",
        "file_id",
        "time_unit",
    )
    @classmethod
    def not_placeholder(cls, value: str) -> str:
        if value.strip() in NA_TOKENS:
            raise NeedsReviewError(
                f"A required sample-sheet field is missing or NA ({value!r}).",
                how_to_fix=(
                    "Fill the field with the real identifier from the paper or GEO metadata. "
                    "If it is truly unknown, stop and add it to human_review.unresolved. "
                    "Do not invent donor, time, or biospecimen IDs."
                ),
            )
        return value


class SampleSheet(StrictModel):
    """Validated long-format sample table plus design checks."""

    rows: list[SampleRow] = Field(min_length=1)
    sampling_design: SamplingDesign
    declared_modalities: list[str]

    @model_validator(mode="after")
    def identities_are_unique_and_design_legal(self) -> Self:
        sample_ids = [row.sample_id for row in self.rows]
        if len(sample_ids) != len(set(sample_ids)):
            raise SchemaError(
                "sample_id values are not unique.",
                how_to_fix="Give every modality measurement its own sample_id, e.g. S01_t0_rna.",
            )
        seen_keys: set[tuple[str, str, float, str]] = set()
        for row in self.rows:
            key = (row.subject_id, row.biospecimen_id, row.time, row.modality)
            if key in seen_keys:
                raise NeedsReviewError(
                    "Duplicate (subject_id, biospecimen_id, time, modality) with no explanation: "
                    f"{key}.",
                    how_to_fix=(
                        "If these are technical replicates, give them distinct sample_id values "
                        "and set replicate_type: technical. If you cannot explain the duplicate, "
                        "add it to human_review.unresolved instead of dropping a row."
                    ),
                )
            seen_keys.add(key)
            if row.modality not in self.declared_modalities:
                raise SchemaError(
                    f"Sample '{row.sample_id}' uses modality '{row.modality}' which is not in the manifest.",
                    how_to_fix=f"Declared modalities are {self.declared_modalities}. Fix the sheet or the manifest.",
                )

        if self.sampling_design is SamplingDesign.LONGITUDINAL:
            times_per_unit: dict[str, set[float]] = {}
            for row in self.rows:
                times_per_unit.setdefault(row.experimental_unit_id, set()).add(row.time)
            short = [unit for unit, times in times_per_unit.items() if len(times) < 2]
            if short:
                raise SchemaError(
                    "Longitudinal design has experimental units with only one time point: "
                    f"{short[:8]}.",
                    how_to_fix=(
                        "A longitudinal unit must be the same subject/animal sampled at multiple "
                        "times. If each time point is a different animal, set "
                        "sampling_design: repeated_cross_sectional."
                    ),
                )
        elif self.sampling_design is SamplingDesign.REPEATED_CROSS_SECTIONAL:
            rcs_times: dict[str, set[float]] = {}
            subjects_per_unit: dict[str, set[str]] = {}
            for row in self.rows:
                rcs_times.setdefault(row.experimental_unit_id, set()).add(row.time)
                subjects_per_unit.setdefault(row.experimental_unit_id, set()).add(row.subject_id)
            multi_time = [unit for unit, times in rcs_times.items() if len(times) > 1]
            if multi_time:
                raise NeedsReviewError(
                    "Repeated cross-sectional experimental units appear at multiple times: "
                    f"{multi_time[:8]}. That would fabricate a longitudinal trajectory.",
                    how_to_fix=(
                        "Give each sacrificed animal / culture dish its own experimental_unit_id "
                        "and a single time. Do not reuse an animal ID across time points."
                    ),
                )
            multi_subject = [unit for unit, subs in subjects_per_unit.items() if len(subs) > 1]
            if multi_subject:
                raise NeedsReviewError(
                    "An experimental_unit_id maps to multiple subject_id values.",
                    how_to_fix="experimental_unit_id and subject_id must be 1:1 for RCS animals.",
                )
        return self

    def to_frame(self) -> pd.DataFrame:
        """Return a DataFrame with the required columns."""

        return pd.DataFrame([row.model_dump() for row in self.rows])


def load_sample_sheet(
    frame: pd.DataFrame,
    *,
    sampling_design: SamplingDesign,
    declared_modalities: list[str],
) -> SampleSheet:
    """Validate a sample-sheet DataFrame.

    Missing required columns and NA identity fields fail loudly.
    """

    missing = [col for col in REQUIRED_SAMPLE_COLUMNS if col not in frame.columns]
    if missing:
        raise SchemaError(
            f"Sample sheet is missing required columns: {missing}.",
            how_to_fix=(
                "The header must include:\n  "
                + " ".join(REQUIRED_SAMPLE_COLUMNS)
                + "\nobservation_id is the shared biospecimen/time key used to align modalities."
            ),
        )
    records = [
        {str(key): value for key, value in row.items()}
        for row in frame[list(REQUIRED_SAMPLE_COLUMNS)].to_dict(orient="records")
    ]
    return SampleSheet(
        rows=[SampleRow.model_validate(record) for record in records],
        sampling_design=sampling_design,
        declared_modalities=declared_modalities,
    )
