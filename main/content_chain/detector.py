"""CEG-WM 正式 HF-only D_M 与未晋升 LF/HF 组合诊断。"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from math import floor, isfinite, sqrt
from typing import Literal, Sequence

from main.shared.key_schedule import (
    NORMAL_QUANTILE_TABLE_SHA256,
    KeyScheduleError,
    normal_quantile_table_lookup,
    stable_json_utf8,
)

from .hf_detector import HfDetectionResult
from .lf_detector import LfDetectionResult

CONTENT_DETECTOR_CANDIDATE_IDS = (
    "hf_sparse_tail",
    "lf_low_pass",
    "content_combination_calibrated",
)
FROZEN_COMBINATION_WEIGHTS = (0.25, 0.50, 0.75)
BranchName = Literal["hf", "lf"]
CombinationFunction = Literal["C0", "C1", "C2"]


class ContentDetectorError(ValueError):
    """分支统计、CDF 身份、组合公式或密钥语义无效。"""


@dataclass(frozen=True, slots=True)
class NullScoreRecord:
    """primary-null 的 float64 分支分数与稳定排序身份。"""

    score: float
    source_cluster_id: str
    sample_id: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.score, bool)
            or not isinstance(self.score, (int, float))
            or not isfinite(float(self.score))
        ):
            raise ContentDetectorError("null score must be a finite float64 value")
        if (
            type(self.source_cluster_id) is not str
            or not self.source_cluster_id
            or type(self.sample_id) is not str
            or not self.sample_id
        ):
            raise ContentDetectorError(
                "null record identities must be non-empty strings"
            )
        object.__setattr__(self, "score", float(self.score))


@dataclass(frozen=True, slots=True)
class BranchNullCalibration:
    """与分支 detector/partition 绑定的有限 primary-null multiset。"""

    branch: BranchName
    detector_identity: str
    partition_identity: str
    records: tuple[NullScoreRecord, ...]
    calibration_identity: str = field(init=False)

    def __post_init__(self) -> None:
        if self.branch not in {"hf", "lf"}:
            raise ContentDetectorError("CDF branch must be hf or lf")
        for role, value in (
            ("detector_identity", self.detector_identity),
            ("partition_identity", self.partition_identity),
        ):
            if type(value) is not str or not value:
                raise ContentDetectorError(f"{role} must be a non-empty string")
        if len(self.records) < 2 or any(
            type(record) is not NullScoreRecord for record in self.records
        ):
            raise ContentDetectorError(
                "branch empirical CDF requires at least two null records"
            )
        ordered = tuple(
            sorted(
                self.records,
                key=lambda record: (
                    record.score,
                    record.source_cluster_id,
                    record.sample_id,
                ),
            )
        )
        record_ids = [
            (record.source_cluster_id, record.sample_id) for record in ordered
        ]
        if len(set(record_ids)) != len(record_ids):
            raise ContentDetectorError(
                "branch empirical CDF record identities must be unique"
            )
        object.__setattr__(self, "records", ordered)
        identity = {
            "branch": self.branch,
            "candidate_id": "content_combination_calibrated",
            "detector_identity": self.detector_identity,
            "normal_quantile_table_sha256": NORMAL_QUANTILE_TABLE_SHA256,
            "partition_identity": self.partition_identity,
            "records": [
                {
                    "sample_id": record.sample_id,
                    "score_float64_hex": record.score.hex(),
                    "source_cluster_id": record.source_cluster_id,
                }
                for record in ordered
            ],
        }
        object.__setattr__(
            self,
            "calibration_identity",
            sha256(stable_json_utf8(identity)).hexdigest(),
        )


@dataclass(frozen=True, slots=True)
class BranchStandardizationResult:
    """mid-rank、tail clipping 与冻结表索引的完整可观测结果。"""

    branch: BranchName
    raw_score: float
    less_count: int
    equal_count: int
    null_count: int
    u_raw: float
    epsilon_n: float
    u_clipped: float
    quantile_index: int
    z_score: float
    calibration_identity: str


@dataclass(frozen=True, slots=True)
class CalibratedCombinationResult:
    """未晋升的 C0/C1/C2 诊断统计及冻结公式身份。"""

    candidate_id: str
    function_id: str
    weight: float | None
    hf_standardization: BranchStandardizationResult
    lf_standardization: BranchStandardizationResult | None
    combined_score: float
    formula_identity: str
    combination_identity: str
    diagnostic_only: bool
    promoted: bool


@dataclass(frozen=True, slots=True)
class ContentDetectionResult:
    """正式 HF-only D_M，并独立保留 LF/HF/combined 诊断。"""

    candidate_ids: tuple[str, ...]
    formal_mode: str
    content_score: float
    hf_score: float
    lf_score: float | None
    combined_score: float | None
    detector_identity: str
    content_config_digest: str
    hf_result: HfDetectionResult
    lf_result: LfDetectionResult | None
    diagnostic_combination: CalibratedCombinationResult | None
    diagnostic_identity: str | None


def _validate_hf_result(result: object) -> HfDetectionResult:
    if type(result) is not HfDetectionResult:
        raise ContentDetectorError(
            "content detector requires HfDetectionResult"
        )
    if result.candidate_id != "hf_sparse_tail" or not isfinite(result.hf_score):
        raise ContentDetectorError("HF branch identity or score is invalid")
    return result


def _validate_lf_result(result: object) -> LfDetectionResult:
    if type(result) is not LfDetectionResult:
        raise ContentDetectorError(
            "LF diagnostic branch requires LfDetectionResult"
        )
    if result.candidate_id != "lf_low_pass" or not isfinite(result.lf_score):
        raise ContentDetectorError("LF branch identity or score is invalid")
    return result


def _validate_shared_key_semantics(
    hf_result: HfDetectionResult,
    lf_result: LfDetectionResult,
) -> None:
    if (
        hf_result.root_key_public_digest != lf_result.root_key_public_digest
        or hf_result.key_role != lf_result.key_role
        or hf_result.wrong_key_index != lf_result.wrong_key_index
    ):
        raise ContentDetectorError(
            "LF/HF branch key semantics differ; combination is forbidden"
        )
    if hf_result.observation_digest != lf_result.observation_digest:
        raise ContentDetectorError(
            "LF/HF observation digests differ; cross-image combination is forbidden"
        )


def _standardize_branch(
    *,
    branch: BranchName,
    score: float,
    detector_identity: str,
    calibration: BranchNullCalibration,
) -> BranchStandardizationResult:
    if type(calibration) is not BranchNullCalibration:
        raise ContentDetectorError("branch standardization requires frozen CDF")
    if calibration.branch != branch:
        raise ContentDetectorError("branch CDF identity mismatch")
    if calibration.detector_identity != detector_identity:
        raise ContentDetectorError(
            "branch CDF cannot be reused across detector identities"
        )
    query = float(score)
    if not isfinite(query):
        raise ContentDetectorError("branch query score must be finite")
    less_count = sum(record.score < query for record in calibration.records)
    equal_count = sum(record.score == query for record in calibration.records)
    null_count = len(calibration.records)
    u_raw = (less_count + 0.5 * equal_count) / null_count
    epsilon_n = 1.0 / (2.0 * null_count)
    u_clipped = min(max(u_raw, epsilon_n), 1.0 - epsilon_n)
    quantile_index = min((1 << 20) - 1, floor(u_clipped * (1 << 20)))
    try:
        z_score = float(normal_quantile_table_lookup(quantile_index))
    except KeyScheduleError as exc:
        raise ContentDetectorError(
            "frozen normal quantile lookup failed"
        ) from exc
    if not isfinite(z_score):
        raise ContentDetectorError("standardized branch score must be finite")
    return BranchStandardizationResult(
        branch=branch,
        raw_score=query,
        less_count=less_count,
        equal_count=equal_count,
        null_count=null_count,
        u_raw=u_raw,
        epsilon_n=epsilon_n,
        u_clipped=u_clipped,
        quantile_index=quantile_index,
        z_score=z_score,
        calibration_identity=calibration.calibration_identity,
    )


def _combine_diagnostic(
    *,
    hf_result: HfDetectionResult,
    lf_result: LfDetectionResult | None,
    hf_null: BranchNullCalibration,
    lf_null: BranchNullCalibration | None,
    function: CombinationFunction,
    weight: float | None,
) -> CalibratedCombinationResult:
    if function not in {"C0", "C1", "C2"}:
        raise ContentDetectorError("combination function must be C0, C1, or C2")
    hf_standardization = _standardize_branch(
        branch="hf",
        score=hf_result.hf_score,
        detector_identity=hf_result.detector_identity,
        calibration=hf_null,
    )
    lf_standardization: BranchStandardizationResult | None = None
    if function in {"C1", "C2"}:
        if lf_result is None or lf_null is None:
            raise ContentDetectorError(
                f"{function} requires independent LF score and LF CDF"
            )
        if hf_null.partition_identity != lf_null.partition_identity:
            raise ContentDetectorError(
                "LF/HF CDFs must share one frozen partition identity"
            )
        lf_standardization = _standardize_branch(
            branch="lf",
            score=lf_result.lf_score,
            detector_identity=lf_result.detector_identity,
            calibration=lf_null,
        )

    z_hf = hf_standardization.z_score
    if function == "C0":
        if weight is not None:
            raise ContentDetectorError("C0 does not accept a weight")
        combined_score = z_hf
        function_id = "C0"
        formula = "z_hf"
        normalized_weight = None
    elif function == "C1":
        if (
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or float(weight) not in FROZEN_COMBINATION_WEIGHTS
        ):
            raise ContentDetectorError(
                "C1 weight must be one of 0.25, 0.50, 0.75"
            )
        normalized_weight = float(weight)
        z_lf = lf_standardization.z_score
        combined_score = (
            normalized_weight * z_hf
            + sqrt(1.0 - normalized_weight * normalized_weight) * z_lf
        )
        function_id = f"C1_w{int(normalized_weight * 100):03d}"
        formula = "w*z_hf+sqrt(1-w^2)*z_lf"
    else:
        if weight is not None:
            raise ContentDetectorError("C2 does not accept a weight")
        normalized_weight = None
        z_lf = lf_standardization.z_score
        combined_score = max(z_hf, z_lf)
        function_id = "C2"
        formula = "max(z_hf,z_lf)"
    if not isfinite(combined_score):
        raise ContentDetectorError("combined diagnostic score must be finite")

    formula_identity = sha256(
        stable_json_utf8(
            {
                "candidate_id": "content_combination_calibrated",
                "formula": formula,
                "function_id": function_id,
                "normal_quantile_table_sha256": NORMAL_QUANTILE_TABLE_SHA256,
                "weight_float64_hex": (
                    normalized_weight.hex()
                    if normalized_weight is not None
                    else None
                ),
            }
        )
    ).hexdigest()
    combination_identity = sha256(
        stable_json_utf8(
            {
                "formula_identity": formula_identity,
                "hf_calibration_identity": (
                    hf_standardization.calibration_identity
                ),
                "lf_calibration_identity": (
                    lf_standardization.calibration_identity
                    if lf_standardization is not None
                    else None
                ),
                "promotion_status": "diagnostic_not_promoted",
            }
        )
    ).hexdigest()
    return CalibratedCombinationResult(
        candidate_id="content_combination_calibrated",
        function_id=function_id,
        weight=normalized_weight,
        hf_standardization=hf_standardization,
        lf_standardization=lf_standardization,
        combined_score=combined_score,
        formula_identity=formula_identity,
        combination_identity=combination_identity,
        diagnostic_only=True,
        promoted=False,
    )


def content_detector(
    hf_result: HfDetectionResult,
    lf_result: LfDetectionResult | None = None,
    *,
    hf_null: BranchNullCalibration | None = None,
    lf_null: BranchNullCalibration | None = None,
    combination: CombinationFunction | None = None,
    weight: float | None = None,
) -> ContentDetectionResult:
    """保持正式 HF-only D_M，同时可计算明确未晋升的组合诊断。"""

    hf_result = _validate_hf_result(hf_result)
    normalized_lf_result = (
        _validate_lf_result(lf_result) if lf_result is not None else None
    )
    if normalized_lf_result is not None:
        _validate_shared_key_semantics(hf_result, normalized_lf_result)

    diagnostic: CalibratedCombinationResult | None = None
    if combination is not None:
        if hf_null is None:
            raise ContentDetectorError(
                "diagnostic combination requires frozen HF CDF"
            )
        diagnostic = _combine_diagnostic(
            hf_result=hf_result,
            lf_result=normalized_lf_result,
            hf_null=hf_null,
            lf_null=lf_null,
            function=combination,
            weight=weight,
        )
    elif hf_null is not None or lf_null is not None or weight is not None:
        raise ContentDetectorError(
            "CDFs and weight require an explicit diagnostic function"
        )

    # This is deliberately byte-for-byte the batch-2 formal HF-only identity.
    content_config = {
        "branch_detector_identity": hf_result.detector_identity,
        "candidate_id": "hf_sparse_tail",
        "content_detector_role": "hf_only_direct_score",
        "lf_combination_enabled": False,
    }
    content_config_digest = sha256(
        stable_json_utf8(content_config)
    ).hexdigest()
    detector_identity = sha256(
        stable_json_utf8(
            {
                "branch_detector_identity": hf_result.detector_identity,
                "content_config_digest": content_config_digest,
                "detector_name": "ceg_wm_hf_only_content_detector",
            }
        )
    ).hexdigest()
    diagnostic_identity = (
        sha256(
            stable_json_utf8(
                {
                    "candidate_ids": list(CONTENT_DETECTOR_CANDIDATE_IDS),
                    "combination_identity": diagnostic.combination_identity,
                    "formal_detector_identity": detector_identity,
                    "formal_mode": "hf_only",
                }
            )
        ).hexdigest()
        if diagnostic is not None
        else None
    )
    return ContentDetectionResult(
        candidate_ids=CONTENT_DETECTOR_CANDIDATE_IDS,
        formal_mode="hf_only",
        content_score=hf_result.hf_score,
        hf_score=hf_result.hf_score,
        lf_score=(
            normalized_lf_result.lf_score
            if normalized_lf_result is not None
            else None
        ),
        combined_score=(
            diagnostic.combined_score if diagnostic is not None else None
        ),
        detector_identity=detector_identity,
        content_config_digest=content_config_digest,
        hf_result=hf_result,
        lf_result=normalized_lf_result,
        diagnostic_combination=diagnostic,
        diagnostic_identity=diagnostic_identity,
    )
