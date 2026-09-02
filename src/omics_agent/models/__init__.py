"""Registered ModelPlugins. Evaluator is imported from ``omics_agent.evaluation``."""

from omics_agent.models.base import ModelPlugin, get_model, list_models
from omics_agent.models.last_value import LastValueModel
from omics_agent.models.ridge import RidgeModel
from omics_agent.models.time_spline import TimeSplineModel

__all__ = [
    "LastValueModel",
    "ModelPlugin",
    "RidgeModel",
    "TimeSplineModel",
    "get_model",
    "list_models",
]
