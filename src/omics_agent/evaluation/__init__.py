"""Independent evaluator. Models do not compute official metrics."""

from omics_agent.evaluation.evaluator import evaluate_predictions
from omics_agent.evaluation.metrics import correlation, mae, mse, r2_score, rmse

__all__ = ["correlation", "evaluate_predictions", "mae", "mse", "r2_score", "rmse"]
