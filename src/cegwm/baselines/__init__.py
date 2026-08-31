"""Interfaces for the four generative-watermark comparison baselines.

This package records method identity and observations; it does not implement or
execute an external method.
"""

from cegwm.baselines.registry import PRIMARY_BASELINES, baseline_by_id
from cegwm.baselines.records import BaselineObservation, validate_observation
from cegwm.baselines.adapters import AdapterPlan, adapter_plan
from cegwm.baselines.table import BaselineTableRow, build_baseline_table_row
from cegwm.baselines.protocol import (
    CLEAN_CONFIRMATION_NEGATIVES,
    EVALUATION_PHYSICAL_UNITS,
    FORMAL_ATTACK_CONDITIONS,
    TARGET_FPR_UPPER_BOUND,
    THRESHOLD_FREEZE_NEGATIVES,
    evaluate_clean_confirmation,
    one_sided_clopper_pearson_upper,
    operating_point_violation,
    per_method_scale,
    rotation_execution_blocker,
)
from cegwm.baselines.attacks import ROTATION_ATTACK_ID, RotationAttackResult, rotation_10_bicubic_reflect_center_crop

__all__ = [
    "PRIMARY_BASELINES",
    "AdapterPlan",
    "BaselineTableRow",
    "ROTATION_ATTACK_ID",
    "RotationAttackResult",
    "CLEAN_CONFIRMATION_NEGATIVES",
    "EVALUATION_PHYSICAL_UNITS",
    "FORMAL_ATTACK_CONDITIONS",
    "TARGET_FPR_UPPER_BOUND",
    "THRESHOLD_FREEZE_NEGATIVES",
    "BaselineObservation",
    "baseline_by_id",
    "adapter_plan",
    "build_baseline_table_row",
    "evaluate_clean_confirmation",
    "one_sided_clopper_pearson_upper",
    "operating_point_violation",
    "per_method_scale",
    "rotation_execution_blocker",
    "rotation_10_bicubic_reflect_center_crop",
    "validate_observation",
]
