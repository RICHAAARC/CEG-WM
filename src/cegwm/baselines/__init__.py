"""Interfaces for the four generative-watermark comparison baselines.

This package records method identity and observations; it does not implement or
execute an external method.
"""

from cegwm.baselines.registry import PRIMARY_BASELINES, baseline_by_id
from cegwm.baselines.records import BaselineObservation, validate_observation
from cegwm.baselines.adapters import AdapterPlan, adapter_plan
from cegwm.baselines.table import (
    FINAL_BASELINE_LONG_TABLE_FIELDS,
    FINAL_BASELINE_PRIMARY_CONDITION_ORDER,
    BaselineTableRow,
    build_baseline_table_row,
)
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
from cegwm.baselines.attacks import (
    CENTER_CROP_80_RESTORE_ATTACK_ID,
    GAUSSIAN_BLUR_SIGMA_1PX_ATTACK_ID,
    JPEG_Q50_ATTACK_ID,
    RESIZE_50_BICUBIC_RESTORE_ATTACK_ID,
    ROTATION_ATTACK_ID,
    FrozenAttackResult,
    RotationAttackResult,
    center_crop_80_restore,
    gaussian_blur_sigma_1px,
    jpeg_q50,
    resize_50_bicubic_restore,
    rotation_10_bicubic_reflect_center_crop,
)
from cegwm.baselines.t2smark import (
    DEFAULT_KEY_LENGTH,
    DEFAULT_MESSAGE_LENGTH,
    DEFAULT_NUM_INVERSION_STEPS,
    DEFAULT_TAU,
    KEY_CHANNELS,
    MESSAGE_CHANNELS,
    SD35_LATENT_SHAPE,
    T2SMarkCodec,
    embed_t2smark_sd35,
    score_t2smark_rgb,
    t2smark_sd35_codecs,
)

__all__ = [
    "PRIMARY_BASELINES",
    "AdapterPlan",
    "BaselineTableRow",
    "FINAL_BASELINE_LONG_TABLE_FIELDS",
    "FINAL_BASELINE_PRIMARY_CONDITION_ORDER",
    "ROTATION_ATTACK_ID",
    "JPEG_Q50_ATTACK_ID",
    "RESIZE_50_BICUBIC_RESTORE_ATTACK_ID",
    "CENTER_CROP_80_RESTORE_ATTACK_ID",
    "GAUSSIAN_BLUR_SIGMA_1PX_ATTACK_ID",
    "FrozenAttackResult",
    "RotationAttackResult",
    "T2SMarkCodec",
    "DEFAULT_KEY_LENGTH",
    "DEFAULT_MESSAGE_LENGTH",
    "DEFAULT_TAU",
    "DEFAULT_NUM_INVERSION_STEPS",
    "SD35_LATENT_SHAPE",
    "KEY_CHANNELS",
    "MESSAGE_CHANNELS",
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
    "jpeg_q50",
    "resize_50_bicubic_restore",
    "center_crop_80_restore",
    "gaussian_blur_sigma_1px",
    "t2smark_sd35_codecs",
    "embed_t2smark_sd35",
    "score_t2smark_rgb",
    "validate_observation",
]
