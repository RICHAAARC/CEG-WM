"""Frozen CPU-only selective-reliability contract for Geometry-V7 R2."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

from cegwm.geometry_v7.contracts import CANONICAL_CORNERS_NORMALIZED


R2_FEATURE_ORDER = ("raw_logit", "kappa_f", "coverage", "area_ratio")
R2_QUANTILES = (0.20, 0.40, 0.60, 0.80)
R2_CONDITION_IDS = (
    "core_rotation_neg15", "core_rotation_pos15", "core_fixed_canvas_zoom_0_8",
    "core_fixed_canvas_zoom_1_2", "core_translation_pos32_x",
    "core_translation_neg32_x", "core_translation_pos32_y",
    "core_translation_neg32_y", "core_offset_crop_rescale",
    "core_composite_c0_85_t16_neg16_r10",
)
R2_DEV_UNIT_IDS = tuple(f"content-v6-iss-eval-{index:04d}" for index in range(1, 5))
R2_TEST_UNIT_IDS = tuple(f"content-v6-iss-eval-{index:04d}" for index in range(5, 9))
R2_DEV_NO_FEASIBLE = "R2_DEV_NO_FEASIBLE_CANDIDATE"
R2_PASSED_ALL = "R2_SELECTIVE_RISK_PASSED_ALL_FAMILY_COVERAGE"
R2_PASSED_PARTIAL = "R2_SELECTIVE_RISK_PASSED_PARTIAL_FAMILY_COVERAGE"
R2_FAILED = "R2_SELECTIVE_RISK_FAILED"
R2_OPERATIONAL_FAILURE = "OPERATIONAL_FAILURE_RETAINED_FIXED_DENOMINATOR"
R2_CLAIM_CEILING = (
    "small_sample_selective_reliability_engineering_canary_with_prior_aggregate_visibility_only"
)


@dataclass(frozen=True, slots=True)
class FeatureRow:
    split: str
    condition_id: str
    unit_id: str
    mandatory_valid: bool
    raw_logit: float | None
    kappa_f: float | None
    coverage: float | None
    area_ratio: float | None
    validity_errors: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class OutcomeRow:
    split: str
    condition_id: str
    unit_id: str
    membership: str
    complete: bool
    safe: bool
    safe_rescue: bool
    baseline_positive: bool
    post_positive: bool | None
    observed_negative_false_positive: bool | None
    errors: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Stump:
    feature: str
    direction: str
    threshold: float
    source_quantiles: tuple[float, ...]

    @property
    def candidate_id(self) -> str:
        return f"B|{self.feature}|{self.direction}|{self.threshold.hex()}"


@dataclass(frozen=True, slots=True)
class Candidate:
    candidate_id: str
    components: tuple[Stump, ...]

    @property
    def complexity(self) -> int:
        return len(self.components)


@dataclass(frozen=True, slots=True)
class AttackMetrics:
    condition_id: str
    accepted_count: int
    unsafe_accept_count: int
    safe_rescue_count: int
    selected_negative_control_fp_count: int
    coverage: float


@dataclass(frozen=True, slots=True)
class CandidateMetrics:
    candidate_id: str
    split: str
    accepted_count: int
    unsafe_accept_count: int
    safe_rescue_count: int
    net_rescue_change: int
    selected_negative_control_fp_count: int
    selected_negative_control_denominator: int
    selected_negative_control_known_denominator: int
    rejected_count: int
    covered_attack_count: int
    coverage: float
    selective_risk: float | None
    unsafe_fixed_denominator_rate: float
    rejection_rate: float
    per_attack: tuple[AttackMetrics, ...]
    gates_passed: bool


@dataclass(frozen=True, slots=True)
class Selection:
    status: str
    selected: Candidate | None
    selected_metrics: CandidateMetrics | None
    candidates: tuple[Candidate, ...]
    candidate_table: tuple[CandidateMetrics, ...]


def _finite(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("value must be a finite real scalar")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("value must be a finite real scalar")
    return value


def _matrix_inverse(matrix: Sequence[Sequence[float]]) -> tuple[tuple[float, ...], ...]:
    values = tuple(_finite(value) for row in matrix for value in row)
    if len(values) != 9:
        raise ValueError("matrix must be finite 3x3")
    a, b, c, d, e, f, g, h, i = values
    determinant = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    if not math.isfinite(determinant) or determinant == 0.0:
        raise ValueError("matrix must be invertible")
    inverse = (
        (e * i - f * h, c * h - b * i, b * f - c * e),
        (f * g - d * i, a * i - c * g, c * d - a * f),
        (d * h - e * g, b * g - a * h, a * e - b * d),
    )
    return tuple(tuple(value / determinant for value in row) for row in inverse)


def _apply_h(matrix: Sequence[Sequence[float]], points: Sequence[Sequence[float]]):
    result = []
    for x, y in points:
        denominator = matrix[2][0] * x + matrix[2][1] * y + matrix[2][2]
        if not math.isfinite(denominator) or denominator == 0.0:
            raise ValueError("homography point is undefined")
        result.append(((matrix[0][0] * x + matrix[0][1] * y + matrix[0][2]) / denominator,
                       (matrix[1][0] * x + matrix[1][1] * y + matrix[1][2]) / denominator))
    return tuple(result)


def _strict_convex(points: Sequence[Sequence[float]]) -> bool:
    if len(points) != 4 or any(len(point) != 2 for point in points):
        return False
    crosses = []
    for index in range(4):
        a, b, c = points[index], points[(index + 1) % 4], points[(index + 2) % 4]
        crosses.append((b[0]-a[0])*(c[1]-b[1]) - (b[1]-a[1])*(c[0]-b[0]))
    return all(value > 0 for value in crosses) or all(value < 0 for value in crosses)


def _shoelace(points: Sequence[Sequence[float]]) -> float:
    return 0.5 * sum(
        points[index][0] * points[(index + 1) % len(points)][1]
        - points[(index + 1) % len(points)][0] * points[index][1]
        for index in range(len(points))
    )


def _clip_polygon(points: Sequence[Sequence[float]]) -> tuple[tuple[float, float], ...]:
    polygon = tuple((float(x), float(y)) for x, y in points)
    boundaries = (
        (lambda p: p[0] >= -1.0, lambda a, b: (-1.0, a[1] + (b[1]-a[1])*(-1.0-a[0])/(b[0]-a[0]))),
        (lambda p: p[0] <= 1.0, lambda a, b: (1.0, a[1] + (b[1]-a[1])*(1.0-a[0])/(b[0]-a[0]))),
        (lambda p: p[1] >= -1.0, lambda a, b: (a[0] + (b[0]-a[0])*(-1.0-a[1])/(b[1]-a[1]), -1.0)),
        (lambda p: p[1] <= 1.0, lambda a, b: (a[0] + (b[0]-a[0])*(1.0-a[1])/(b[1]-a[1]), 1.0)),
    )
    for inside, intersection in boundaries:
        if not polygon:
            break
        output = []
        previous = polygon[-1]
        for current in polygon:
            if inside(current):
                if not inside(previous):
                    output.append(intersection(previous, current))
                output.append(current)
            elif inside(previous):
                output.append(intersection(previous, current))
            previous = current
        polygon = tuple(output)
    return polygon


def feature_row_from_geometry(*, split: str, condition_id: str, unit_id: str,
                              geometry: Mapping[str, object]) -> FeatureRow:
    errors: list[str] = []
    try:
        if geometry.get("status") not in ("RELIABLE", "UNRELIABLE"):
            raise ValueError("geometry_status")
        if geometry.get("legal") is not True or geometry.get("error") is not None:
            raise ValueError("geometry_legality")
        raw_logit = _finite(geometry.get("uncalibrated_sync_logit"))
        raw_points = geometry.get("observed_corners_in_canonical_normalized")
        if not isinstance(raw_points, (list, tuple)) or len(raw_points) != 4:
            raise ValueError("predicted_correspondences")
        points = tuple(tuple(_finite(axis) for axis in point) for point in raw_points)
        if any(len(point) != 2 for point in points) or not _strict_convex(points):
            raise ValueError("predicted_correspondences")
        raw_matrix = geometry.get("homography_observed_to_canonical")
        if not isinstance(raw_matrix, (list, tuple)) or len(raw_matrix) != 3:
            raise ValueError("homography")
        matrix = tuple(tuple(_finite(axis) for axis in row) for row in raw_matrix)
        if any(len(row) != 3 for row in matrix):
            raise ValueError("homography")
        inverse = _matrix_inverse(matrix)
        mapped = _apply_h(matrix, CANONICAL_CORNERS_NORMALIZED)
        if any(abs(a-b) > 1e-9 for p, q in zip(mapped, points, strict=True)
               for a, b in zip(p, q, strict=True)):
            raise ValueError("homography_correspondence")
        kappa = math.sqrt(sum(v*v for row in matrix for v in row)) * math.sqrt(
            sum(v*v for row in inverse for v in row)
        )
        area = abs(_shoelace(points)) / 4.0
        clipped = _clip_polygon(points)
        coverage = 0.0 if len(clipped) < 3 else abs(_shoelace(clipped)) / 4.0
        return FeatureRow(split, condition_id, unit_id, True, raw_logit, kappa, coverage, area)
    except (TypeError, ValueError, ZeroDivisionError) as error:
        errors.append(str(error))
        return FeatureRow(split, condition_id, unit_id, False, None, None, None, None, tuple(errors))


def outcome_row_from_repair(*, split: str, condition_id: str, unit_id: str,
                            membership: str, record: Mapping[str, object]) -> OutcomeRow:
    errors = tuple(str(item) for item in record.get("errors", ()) if isinstance(item, str))
    complete = not errors
    required_numeric = ("positive_gate_a_delta", "positive_gate_b_delta", "positive_score_delta")
    try:
        if not all(math.isfinite(_finite(record.get(name))) for name in required_numeric):
            complete = False
    except ValueError:
        complete = False
    scores = record.get("scores")
    positive = None
    if isinstance(scores, Mapping):
        decision = scores.get("positive_cg_vs_g")
        complete = complete and all(name in scores for name in (
            "u", "g", "cg", "positive_cg_vs_g", "negative_g_vs_u"
        ))
        if isinstance(decision, Mapping) and isinstance(decision.get("positive"), bool):
            positive = bool(decision["positive"])
    fp = record.get("observed_negative_false_positive")
    if not isinstance(fp, bool) or positive is None:
        complete = False
        fp = None
    gain = record.get("positive_score_delta")
    gain_valid = isinstance(gain, (int, float)) and not isinstance(gain, bool) and math.isfinite(float(gain))
    improved = record.get("improved")
    complete = complete and isinstance(improved, bool) and gain_valid
    if complete and improved != (float(gain) > 0.0):
        complete = False
    if membership == "N_recovery_negative":
        recovered = record.get("recovered_negative")
        complete = complete and isinstance(recovered, bool) and gain_valid
        safe = complete and float(gain) > 0 and recovered is True and fp is False
        baseline = False
    elif membership == "B_boundary":
        complete = complete and gain_valid
        safe = complete and float(gain) > 0 and fp is False
        baseline = True
    elif membership == "D_damage_only":
        harm = record.get("decision_harm")
        complete = complete and isinstance(harm, bool)
        safe = complete and harm is False and fp is False
        baseline = True
    else:
        complete, safe, baseline = False, False, False
    return OutcomeRow(split, condition_id, unit_id, membership, complete, safe,
                      membership == "N_recovery_negative" and safe, baseline,
                      positive, fp, errors)


def _validate_rows(rows: Sequence[FeatureRow], split: str) -> tuple[FeatureRow, ...]:
    rows = tuple(rows)
    units = R2_DEV_UNIT_IDS if split == "dev" else R2_TEST_UNIT_IDS
    if len(rows) != 40 or any(row.split != split for row in rows):
        raise ValueError(f"{split} feature rows must be exact fixed 40")
    conditions = tuple(dict.fromkeys(row.condition_id for row in rows))
    if conditions != R2_CONDITION_IDS:
        raise ValueError(f"{split} feature rows must contain exact ten attacks")
    expected = tuple((condition, unit) for condition in conditions for unit in units)
    actual = tuple((row.condition_id, row.unit_id) for row in rows)
    if actual != expected or len({(row.condition_id, row.unit_id) for row in rows}) != 40:
        raise ValueError(f"{split} feature row identity/order differs")
    return rows


def _type7(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    h = (len(ordered) - 1) * quantile
    lower = math.floor(h)
    upper = math.ceil(h)
    return ordered[lower] + (h-lower) * (ordered[upper]-ordered[lower])


def _row_is_mandatory_valid(row: FeatureRow) -> bool:
    return row.mandatory_valid and all(
        isinstance(getattr(row, feature), (int, float))
        and not isinstance(getattr(row, feature), bool)
        and math.isfinite(float(getattr(row, feature)))
        for feature in R2_FEATURE_ORDER
    )


def generate_candidates(dev_rows: Sequence[FeatureRow]) -> tuple[Candidate, ...]:
    rows = _validate_rows(dev_rows, "dev")
    stumps: list[Stump] = []
    for feature in R2_FEATURE_ORDER:
        values = [getattr(row, feature) for row in rows if _row_is_mandatory_valid(row)]
        values = [float(value) for value in values if value is not None and math.isfinite(value)]
        grouped: dict[str, tuple[float, list[float]]] = {}
        for quantile in R2_QUANTILES:
            if not values:
                continue
            threshold = _type7(values, quantile)
            grouped.setdefault(threshold.hex(), (threshold, []))[1].append(quantile)
        for threshold, quantiles in sorted(grouped.values(), key=lambda item: (item[0], item[0].hex())):
            for direction in ("le", "ge"):
                stumps.append(Stump(feature, direction, threshold, tuple(quantiles)))
    stumps.sort(key=lambda stump: (R2_FEATURE_ORDER.index(stump.feature), 0 if stump.direction == "le" else 1,
                                   stump.threshold, stump.threshold.hex()))
    candidates = [Candidate("A|LEGAL_ONLY", ())]
    candidates.extend(Candidate(stump.candidate_id, (stump,)) for stump in stumps)
    for index, left in enumerate(stumps):
        for right in stumps[index+1:]:
            if left.feature == right.feature:
                continue
            components = tuple(sorted((left, right), key=lambda s: (R2_FEATURE_ORDER.index(s.feature),
                                0 if s.direction == "le" else 1, s.threshold, s.threshold.hex())))
            candidates.append(Candidate(f"C|{components[0].candidate_id}&{components[1].candidate_id}", components))
    return tuple(candidates)


def _accept(candidate: Candidate, row: FeatureRow) -> bool:
    if not _row_is_mandatory_valid(row):
        return False
    for stump in candidate.components:
        value = getattr(row, stump.feature)
        if value is None or (stump.direction == "le" and value > stump.threshold) or (
            stump.direction == "ge" and value < stump.threshold
        ):
            return False
    return True


def _metrics(candidate: Candidate, features: Sequence[FeatureRow], outcomes: Sequence[OutcomeRow], split: str) -> CandidateMetrics:
    rows = _validate_rows(features, split)
    outcomes = tuple(outcomes)
    if len(outcomes) != 40 or any(row.split != split for row in outcomes):
        raise ValueError(f"{split} outcomes must be exact fixed 40")
    outcome_map = {(row.condition_id, row.unit_id): row for row in outcomes}
    if tuple(outcome_map) != tuple((row.condition_id, row.unit_id) for row in rows):
        raise ValueError(f"{split} feature/outcome identities differ")
    accepted_pairs = [(row, outcome_map[(row.condition_id, row.unit_id)]) for row in rows if _accept(candidate, row)]
    accepted = len(accepted_pairs)
    unsafe = sum(not outcome.safe for _, outcome in accepted_pairs)
    rescue = sum(outcome.safe_rescue for _, outcome in accepted_pairs)
    fp = sum(outcome.observed_negative_false_positive is True for _, outcome in accepted_pairs)
    fp_known = sum(isinstance(outcome.observed_negative_false_positive, bool) for _, outcome in accepted_pairs)
    net = sum(outcome.post_positive is True for _, outcome in accepted_pairs) - sum(
        outcome.baseline_positive for _, outcome in accepted_pairs
    )
    per_attack = []
    for condition in dict.fromkeys(row.condition_id for row in rows):
        selected = [outcome for row, outcome in accepted_pairs if row.condition_id == condition]
        per_attack.append(AttackMetrics(condition, len(selected), sum(not o.safe for o in selected),
                          sum(o.safe_rescue for o in selected),
                          sum(o.observed_negative_false_positive is True for o in selected), len(selected)/4.0))
    covered = sum(item.accepted_count >= 1 for item in per_attack)
    passed = accepted >= 10 and unsafe * 10 <= accepted and fp == 0 and rescue >= 8 and covered >= 5
    return CandidateMetrics(candidate.candidate_id, split, accepted, unsafe, rescue, net, fp,
                            accepted, fp_known, 40-accepted, covered, accepted/40.0,
                            None if accepted == 0 else unsafe/accepted, unsafe/40.0,
                            (40-accepted)/40.0, tuple(per_attack), passed)


def select_candidate(dev_features: Sequence[FeatureRow], dev_outcomes: Sequence[OutcomeRow]) -> Selection:
    candidates = generate_candidates(dev_features)
    table = tuple(_metrics(candidate, dev_features, dev_outcomes, "dev") for candidate in candidates)
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    feasible = [item for item in table if item.gates_passed]
    if not feasible:
        return Selection(R2_DEV_NO_FEASIBLE, None, None, candidates, table)
    selected_metrics = min(feasible, key=lambda item: (-item.safe_rescue_count, -item.accepted_count,
                           item.unsafe_accept_count, -item.covered_attack_count,
                           by_id[item.candidate_id].complexity, item.candidate_id))
    return Selection("R2_DEV_CANDIDATE_FROZEN", by_id[selected_metrics.candidate_id],
                     selected_metrics, candidates, table)


def evaluate_frozen_candidate(candidate: Candidate, test_features: Sequence[FeatureRow],
                              test_outcomes: Sequence[OutcomeRow]) -> tuple[str, CandidateMetrics]:
    metrics = _metrics(candidate, test_features, test_outcomes, "test")
    if metrics.gates_passed and metrics.covered_attack_count == 10:
        return R2_PASSED_ALL, metrics
    if metrics.gates_passed and 5 <= metrics.covered_attack_count <= 9:
        return R2_PASSED_PARTIAL, metrics
    return R2_FAILED, metrics


__all__ = [name for name in globals() if name.startswith("R2_") or name in {
    "FeatureRow", "OutcomeRow", "Stump", "Candidate", "AttackMetrics", "CandidateMetrics",
    "Selection", "feature_row_from_geometry", "outcome_row_from_repair", "generate_candidates",
    "select_candidate", "evaluate_frozen_candidate",
}]
