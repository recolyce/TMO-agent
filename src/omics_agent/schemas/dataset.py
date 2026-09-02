"""Dataset manifest: the only legal description of an input study.

A manifest must be human-reviewed before training. Unresolved pairing,
time, license, or sample-identity questions become ``needs_review``.
The pipeline does not guess those fields.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from omics_agent.errors import NeedsReviewError, SchemaError
from omics_agent.schemas.enums import (
    AssayType,
    ChecksumAlg,
    FeatureIdType,
    FileRole,
    PairingLevel,
    ReviewStatus,
    SamplingDesign,
    SourceType,
    UnitOfIndependence,
    ValueType,
)


class StrictModel(BaseModel):
    """Shared base: extra YAML keys are errors, not silently dropped."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class SourceSpec(StrictModel):
    """Provenance of the files. Real download adapters arrive in milestone 2."""

    type: SourceType
    accession: str | None = None
    paper_doi: str | None = None
    landing_page: str | None = None
    local_dir: Path | None = None
    generator: str | None = Field(
        default=None,
        description="For synthetic data, the generator module and version.",
    )


class LicenseSpec(StrictModel):
    """Redistribution must be explicit. Unknown licenses block training."""

    name: str
    redistributable: bool
    notes: str | None = None


class OrganismSpec(StrictModel):
    """NCBI Taxonomy identity of the sampled organism.

    ``taxon_id`` may be omitted only while human review is still open.
    An approved training manifest must have a real NCBI taxon ID.
    """

    taxon_id: int | None = Field(default=None, ge=1)
    name: str


class DesignSpec(StrictModel):
    """Experimental design. Illegal combinations fail validation."""

    unit_of_independence: UnitOfIndependence
    sampling_design: SamplingDesign
    pairing_level: PairingLevel
    paired_modalities: bool
    time_unit: str
    longitudinal: bool | None = Field(
        default=None,
        description="Deprecated convenience mirror of sampling_design; must agree if set.",
    )

    @model_validator(mode="after")
    def design_is_internally_consistent(self) -> Self:
        if self.sampling_design is SamplingDesign.UNDECLARED:
            if self.longitudinal is not None:
                raise SchemaError(
                    "design.sampling_design is undeclared but longitudinal is set.",
                    how_to_fix=(
                        "Leave longitudinal empty until a person confirms the sampling design. "
                        "Do not guess longitudinal vs repeated cross-section."
                    ),
                )
            return self
        if self.longitudinal is not None:
            expected = self.sampling_design is SamplingDesign.LONGITUDINAL
            if self.longitudinal != expected:
                raise SchemaError(
                    "design.longitudinal does not match design.sampling_design.",
                    how_to_fix=(
                        "Set sampling_design to 'longitudinal' and longitudinal: true, "
                        "or sampling_design to 'repeated_cross_sectional' and "
                        "longitudinal: false. Do not invent a hybrid."
                    ),
                )
        if (
            self.sampling_design is SamplingDesign.REPEATED_CROSS_SECTIONAL
            and self.unit_of_independence is UnitOfIndependence.DONOR
        ):
            raise SchemaError(
                "Repeated cross-sectional data cannot use unit_of_independence='donor' "
                "as if each animal were a longitudinal donor.",
                how_to_fix=(
                    "Use experimental_unit or experiment_batch. "
                    "Never stitch different animals into one subject trajectory."
                ),
            )
        if self.pairing_level is PairingLevel.UNDECLARED:
            return self
        if self.pairing_level is PairingLevel.GROUP_LEVEL_ONLY and self.paired_modalities:
            raise SchemaError(
                "pairing_level='group_level_only' cannot be marked paired_modalities=true. "
                "That would pretend sample-level RNA–protein pairing exists.",
                how_to_fix=(
                    "Set paired_modalities: false, or upgrade pairing_level only after "
                    "biospecimen IDs prove same-aliquot / same-biospecimen / same-subject-time."
                ),
            )
        return self


class ModalitySpec(StrictModel):
    """One bulk assay matrix."""

    assay: AssayType
    value_type: ValueType
    feature_id_type: FeatureIdType
    n_features: int | None = Field(default=None, ge=1)


class FileSpec(StrictModel):
    """One input file. sha256 is required after the first real download."""

    path: Path | None = None
    url: str | None = None
    sha256: str | None = None
    modality: str
    role: FileRole = FileRole.MATRIX
    official_checksum: str | None = None
    official_checksum_alg: ChecksumAlg | None = None
    size_bytes: int | None = Field(default=None, ge=0)
    retrieved_at: str | None = None

    @model_validator(mode="after")
    def has_locator(self) -> Self:
        if self.path is None and self.url is None:
            raise SchemaError(
                f"File for modality '{self.modality}' has neither path nor url.",
                how_to_fix="Add files[].path for local/synthetic data, or files[].url for a download.",
            )
        return self


class HumanReview(StrictModel):
    """Gate that a person, not the pipeline, must close."""

    status: ReviewStatus
    unresolved: list[str] = Field(default_factory=list)
    reviewer: str | None = None
    notes: str | None = None

    @model_validator(mode="after")
    def unresolved_requires_review(self) -> Self:
        if self.unresolved and self.status is ReviewStatus.APPROVED:
            raise SchemaError(
                "human_review.status is 'approved' but unresolved items remain.",
                how_to_fix=(
                    "Either resolve every item in human_review.unresolved, or set "
                    "status to 'required'. The pipeline will not guess the answers."
                ),
            )
        if self.status is ReviewStatus.REQUIRED and not self.unresolved:
            raise SchemaError(
                "human_review.status is 'required' but unresolved is empty.",
                how_to_fix="List the open questions, or set status to 'approved' after a person checked the mapping.",
            )
        return self


class DatasetManifest(StrictModel):
    """Validated ``dataset.yaml``.

    This object is the contract between curation and every later stage.
    """

    schema_version: str = Field(pattern=r"^\d+\.\d+$")
    dataset_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    title: str
    source: SourceSpec
    license: LicenseSpec
    organism: OrganismSpec
    design: DesignSpec
    modalities: dict[str, ModalitySpec] = Field(min_length=1)
    files: list[FileSpec] = Field(min_length=1)
    sample_sheet: Path
    human_review: HumanReview
    notes: str | None = None

    @field_validator("modalities")
    @classmethod
    def modality_names_are_stable(cls, value: dict[str, ModalitySpec]) -> dict[str, ModalitySpec]:
        for name in value:
            if not name.isidentifier():
                raise SchemaError(
                    f"Modality name '{name}' is not a valid identifier.",
                    how_to_fix="Use names like rna, protein, metabolite (letters, digits, underscore).",
                )
        return value

    @model_validator(mode="after")
    def files_match_modalities(self) -> Self:
        matrix_modalities = {item.modality for item in self.files if item.role is FileRole.MATRIX}
        missing = set(self.modalities) - matrix_modalities
        if missing:
            raise SchemaError(
                f"Modalities {sorted(missing)} have no matrix file.",
                how_to_fix="Add a files[] entry with role: matrix and matching modality for each modalities key.",
            )
        unknown = matrix_modalities - set(self.modalities)
        if unknown:
            raise SchemaError(
                f"Matrix files refer to unknown modalities {sorted(unknown)}.",
                how_to_fix="Add those keys under modalities: or fix the files[].modality spelling.",
            )
        return self

    @model_validator(mode="after")
    def approved_requires_resolved_biology(self) -> Self:
        if self.human_review.status is not ReviewStatus.APPROVED:
            return self
        if self.organism.taxon_id is None:
            raise SchemaError(
                "An approved manifest must include organism.taxon_id.",
                how_to_fix="Look up the NCBI Taxonomy ID and set organism.taxon_id before approving.",
            )
        if self.design.sampling_design is SamplingDesign.UNDECLARED:
            raise SchemaError(
                "An approved manifest cannot leave sampling_design undeclared.",
                how_to_fix="Set longitudinal or repeated_cross_sectional after a person inspects the sample sheet.",
            )
        if self.design.pairing_level is PairingLevel.UNDECLARED:
            raise SchemaError(
                "An approved manifest cannot leave pairing_level undeclared.",
                how_to_fix="Prove pairing with biospecimen IDs, or set group_level_only.",
            )
        return self

    def require_approved_for_training(self) -> None:
        """Block modeling when review is open or rejected."""

        if self.human_review.status is ReviewStatus.APPROVED:
            return
        if self.human_review.status is ReviewStatus.REJECTED:
            raise NeedsReviewError(
                f"Dataset '{self.dataset_id}' was rejected by a reviewer.",
                how_to_fix="Do not train. Fix the listed issues and set human_review.status to approved.",
            )
        open_items = "\n".join(f"- {item}" for item in self.human_review.unresolved)
        raise NeedsReviewError(
            f"Dataset '{self.dataset_id}' still needs human review.\n{open_items}",
            how_to_fix=(
                "A person must confirm sample, time, pairing, and license fields. "
                "Edit the manifest, set human_review.status: approved, and clear unresolved. "
                "The pipeline will not infer missing donor/time/biospecimen links."
            ),
        )

    def resolve_path(self, manifest_path: Path, raw: Path) -> Path:
        """Resolve a manifest-relative path against the YAML location."""

        if raw.is_absolute():
            return raw
        return (manifest_path.parent / raw).resolve()


def load_manifest(path: Path) -> DatasetManifest:
    """Load and validate a dataset YAML.

    Parameters
    ----------
    path:
        Path to ``dataset.yaml``.
    """

    import yaml

    if not path.is_file():
        raise SchemaError(
            f"Dataset manifest not found: {path}",
            how_to_fix="Pass an existing YAML path, or run: omics-agent generate-synthetic --output-dir <dir>",
        )
    with path.open("r", encoding="utf-8") as handle:
        payload: dict[str, Any] = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise SchemaError(
            f"{path} is not a YAML mapping.",
            how_to_fix="The file must start with keys like schema_version, dataset_id, source.",
        )
    return DatasetManifest.model_validate(payload)
