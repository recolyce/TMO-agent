"""Registered ModelPlugins. Evaluator is imported from ``omics_agent.evaluation``."""

import contextlib

from omics_agent.models.base import ModelPlugin, get_model, list_models
from omics_agent.models.last_value import LastValueModel
from omics_agent.models.mlp import MlpModel
from omics_agent.models.ridge import RidgeModel
from omics_agent.models.time_spline import TimeSplineModel

# Dynamics plugins need the optional torch extra; without it they simply stay
# unregistered and get_model('gru') explains how to install torch.
with contextlib.suppress(ImportError):
    from omics_agent.models.dynamics import plugin as _dynamics_plugin  # noqa: F401

__all__ = [
    "LastValueModel",
    "MlpModel",
    "ModelPlugin",
    "RidgeModel",
    "TimeSplineModel",
    "get_model",
    "list_models",
]
