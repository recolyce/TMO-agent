"""Controlled vocabularies used by every schema.

These enums are the only allowed spellings in YAML. A typo should fail
validation, not be silently coerced.
"""

from __future__ import annotations

from enum import StrEnum


class SourceType(StrEnum):
    """Where the processed matrices came from."""

    GEO = "geo"
    SRA = "sra"
    BIOSTUDIES = "biostudies"
    PRIDE = "pride"
    MW = "mw"
    URL = "url"
    LOCAL = "local"
    SYNTHETIC = "synthetic"


class SamplingDesign(StrEnum):
    """Biological sampling design. This changes the legal prediction task."""

    LONGITUDINAL = "longitudinal"
    REPEATED_CROSS_SECTIONAL = "repeated_cross_sectional"


class PairingLevel(StrEnum):
    """How RNA and protein (or other modalities) are aligned."""

    SAME_ALIQUOT = "same_aliquot"
    SAME_BIOSPECIMEN = "same_biospecimen"
    SAME_SUBJECT_TIME = "same_subject_time"
    GROUP_LEVEL_ONLY = "group_level_only"


class UnitOfIndependence(StrEnum):
    """The biological unit that must not cross train/val/test."""

    DONOR = "donor"
    SUBJECT = "subject"
    EXPERIMENTAL_UNIT = "experimental_unit"
    EXPERIMENT_BATCH = "experiment_batch"


class ReplicateType(StrEnum):
    """Biological vs technical replicate. They are not interchangeable."""

    BIOLOGICAL = "biological"
    TECHNICAL = "technical"


class ReviewStatus(StrEnum):
    """Human review state of a dataset manifest."""

    REQUIRED = "required"
    APPROVED = "approved"
    REJECTED = "rejected"


class Stage(StrEnum):
    """Pipeline stage. Transitions are explicit; there is no hidden skip."""

    INGESTED = "ingested"
    NEEDS_REVIEW = "needs_review"
    READY = "ready"
    PREPROCESSED = "preprocessed"
    SPLIT_LOCKED = "split_locked"
    BASELINED = "baselined"
    OPTIMIZED = "optimized"
    FROZEN = "frozen"
    TESTED = "tested"
    EXPLAINED = "explained"


class TaskKind(StrEnum):
    """Registered prediction tasks.

    ``subject_forecast`` is legal only for longitudinal designs: the same
    experimental unit is observed at multiple times.

    ``group_time_forecast`` is the RCS-safe task: predict later independent
    biological replicates from the training population. Animals are never
    concatenated into a fake individual trajectory.
    """

    SUBJECT_FORECAST = "subject_forecast"
    GROUP_TIME_FORECAST = "group_time_forecast"


class SplitName(StrEnum):
    """The three isolated partitions. Membership is not Agent-editable."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class AssayType(StrEnum):
    """Bulk assay class. Preprocessing is registered per assay later."""

    BULK_RNASEQ = "bulk_rnaseq"
    MASS_SPEC = "mass_spec"
    MICROARRAY = "microarray"
    METABOLOMICS = "metabolomics"
    SYNTHETIC = "synthetic"


class ValueType(StrEnum):
    """Meaning of the numeric matrix. Do not log-transform twice."""

    RAW_COUNTS = "raw_counts"
    LOG1P = "log1p"
    LOG2_INTENSITY = "log2_intensity"
    INTENSITY = "intensity"
    ZSCORE = "zscore"
    SYNTHETIC_ABUNDANCE = "synthetic_abundance"


class FeatureIdType(StrEnum):
    """Stable feature identifier scheme."""

    ENSEMBL_GENE_ID = "ensembl_gene_id"
    UNIPROT_ACCESSION = "uniprot_accession"
    SYNTHETIC_GENE = "synthetic_gene"
    SYNTHETIC_PROTEIN = "synthetic_protein"


class HistoryPolicy(StrEnum):
    """How much past is visible when constructing a forecast instance."""

    PREVIOUS_ALL = "previous_all"
    LAST_OBSERVATION = "last_observation"
