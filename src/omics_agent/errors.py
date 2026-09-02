"""Exceptions that tell a beginner what went wrong and how to fix it."""

from __future__ import annotations


class OmicsAgentError(Exception):
    """Base error with an explicit repair hint.

    Parameters
    ----------
    message:
        What failed, in plain language.
    how_to_fix:
        Concrete next steps. Shown after the message.
    """

    def __init__(self, message: str, how_to_fix: str) -> None:
        self.message = message
        self.how_to_fix = how_to_fix
        super().__init__(f"{message}\n\nHow to fix:\n{how_to_fix}")


class SchemaError(OmicsAgentError):
    """A YAML/TSV field is missing, mistyped, or scientifically illegal."""


class NeedsReviewError(OmicsAgentError):
    """Sample mapping, time, pairing, or license is unresolved.

    The pipeline must stop. Guessing a mapping is not allowed.
    """


class SplitLeakageError(OmicsAgentError):
    """The same independent biological unit appears in more than one split."""


class PreprocessingLeakageError(OmicsAgentError):
    """A transformer was fitted on validation, test, or the full dataset."""


class TaskDesignError(OmicsAgentError):
    """The prediction task is incompatible with the sampling design."""


class ManifestError(OmicsAgentError):
    """Dataset manifest failed validation or points at unusable files."""


class MetricError(OmicsAgentError):
    """Predictions and targets cannot be scored (shape, alignment, or empty)."""


class TrackingError(OmicsAgentError):
    """MLflow or hash recording failed."""


class DownloadError(OmicsAgentError):
    """A download failed, exceeded policy, or was aborted before verify."""


class ChecksumMismatchError(OmicsAgentError):
    """Computed digest does not match the official or recorded checksum."""


class UnsupportedRawDataError(OmicsAgentError):
    """FASTQ / raw mass-spec is outside milestone 2."""


class TestLockError(OmicsAgentError):
    """The one-shot final test for this experiment_id was already consumed."""


class ArtifactIntegrityError(OmicsAgentError):
    """A frozen artifact (config, checkpoint, split, evaluator) was modified."""


class OdeSolverError(OmicsAgentError):
    """ODE integration produced non-finite latent states."""


class TrainingDivergedError(OmicsAgentError):
    """A training loss became NaN/inf. The run must stop, not report garbage."""


class PriorError(OmicsAgentError):
    """A prior bundle is missing, mislabelled, or scientifically illegal."""
