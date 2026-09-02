"""Leakage-safe group splits. Membership is not Agent-editable."""

from omics_agent.splitting.guard import assert_no_group_leakage
from omics_agent.splitting.split import assign_splits

__all__ = ["assert_no_group_leakage", "assign_splits"]
