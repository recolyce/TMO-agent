"""Milestone-5 validation-only HPO, freeze, and one-shot test lock.

The Optuna objective receives train and val data only. Test rows are never
materialized inside the tuner, and a consumed test lock forbids further
tuning under the same experiment_id.
"""

from omics_agent.optimization.lock import run_final_test
from omics_agent.optimization.tuner import run_tuning

__all__ = ["run_final_test", "run_tuning"]
