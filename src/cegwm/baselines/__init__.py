"""Interfaces for the four generative-watermark comparison baselines.

This package records method identity and observations; it does not implement or
execute an external method.
"""

from cegwm.baselines.registry import PRIMARY_BASELINES, baseline_by_id
from cegwm.baselines.records import BaselineObservation, validate_observation
from cegwm.baselines.adapters import AdapterPlan, adapter_plan

__all__ = [
    "PRIMARY_BASELINES",
    "AdapterPlan",
    "BaselineObservation",
    "baseline_by_id",
    "adapter_plan",
    "validate_observation",
]
