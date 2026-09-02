"""Deterministic bulk temporal multi-omics prediction pipeline.

This package is a research pipeline with a constrained orchestration layer.
It is not an autonomous agent that executes untrusted network code.

Milestone 1 implements schemas, synthetic bulk data, leakage-safe splits,
LastValue / Ridge / time-spline baselines, a unified evaluator, and local
MLflow tracking. LLM calls, real downloads, ODE models, and a web UI are
intentionally absent.
"""

__version__ = "0.1.0"
