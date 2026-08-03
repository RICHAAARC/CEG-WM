"""Frozen blind bounded similarity-transform estimator for Q/K relations."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from itertools import product
from math import inf, isclose, isfinite, log, pi, sqrt
from struct import pack
from typing import Iterable, Sequence

import torch

from main.shared.key_schedule import (
    DerivedWrongKeyMaterial,
    derive_wrong_key_material,
    identify_root_key,
    stable_json_utf8,
)

from .qk_sync import (
    QkGeometrySyncError,
    QkGeometrySyncResult,
    projection_for_detection_key,
    row_normalized_relation_score,
    validate_qk_geometry_sync_result,
)

RECTIFICATION_CANDIDATE_ID = "rectification_similarity"
QK_CANDIDATE_ID = "qk_relation_similarity"
MINIMUM_COVERAGE = 0.45
ROTATION_LIMIT_DEGREES = 32.0
LOG_SCALE_LIMIT = log(sqrt(2.0))
TRANSLATION_LIMIT = 0.28

DIHEDRAL_MATRICES: tuple[tuple[str, tuple[tuple[float, float], tuple[float, float]]], ...] = (
    ("identity", ((1.0, 0.0), (0.0, 1.0))),
    ("x_flip", ((-1.0, 0.0), (0.0, 1.0))),
    ("y_flip", ((1.0, 0.0), (0.0, -1.0))),
    ("xy_flip", ((-1.0, 0.0), (0.0, -1.0))),
    ("rot90", ((0.0, -1.0), (1.0, 0.0))),
    ("rot_minus90", ((0.0, 1.0), (-1.0, 0.0))),
    ("diag", ((0.0, 1.0), (1.0, 0.0))),
    ("anti_diag", ((0.0, -1.0), (-1.0, 0.0))),
)
COARSE_ROTATIONS = (0.0, -32.0, -16.0, 16.0, 32.0)
COARSE_LOG_SCALES = (0.0, -LOG_SCALE_LIMIT, LOG_SCALE_LIMIT)
COARSE_TRANSLATIONS = (0.0, -TRANSLATION_LIMIT, TRANSLATION_LIMIT)
ANCHORS = (
    (-1.0, -1.0),
    (-1.0, 1.0),
    (1.0, -1.0),
    (1.0, 1.0),
    (-1.0, 0.0),
    (1.0, 0.0),
    (0.0, -1.0),
    (0.0, 1.0),
    (-0.5, -0.5),
    (-0.5, 0.5),
    (0.5, -0.5),
    (0.5, 0.5),
)


class GeometricTransformEstimatorError(ValueError):
    """Estimator input or frozen search behavior is invalid."""


@dataclass(frozen=True, slots=True)
class SimilarityTransform:
    """Canonical-to-observed affine and its frozen search coordinates."""

    dihedral: str
    residual_rotation_degrees: float
    log_scale: float
    translation_x: float
    translation_y: float
    matrix: tuple[tuple[float, float, float], tuple[float, float, float]]
    is_exact_identity: bool
    continuous_parameter_on_search_boundary: bool

    def tensor(self) -> torch.Tensor:
        return torch.tensor(self.matrix, dtype=torch.float32)


@dataclass(frozen=True, slots=True)
class GeometricTransformEstimation:
    """Highest-objective transform and unthresholded raw geometry metrics."""

    candidate_ids: tuple[str, str, str]
    transform: SimilarityTransform
    registered_objective: float
    second_registered_objective: float
    exact_identity_objective: float
    wrong_key_objectives: tuple[float, ...]
    canonical_score: float
    observation_score: float
    coverage_forward: float
    coverage_backward: float
    uniqueness_forward: float
    uniqueness_backward: float
    coverage: float
    uniqueness: float
    gap: float
    identity_margin: float
    key_margin: float
    inlier_ratio: float | None
    mean_residual: float
    epsilon_inlier: float | None
    anchor_residuals: tuple[float, ...]
    registered_root_key_public_digest: str
    observation_descriptor_digest: str
    observation_projection_digest: str
    observation_geometry_config_digest: str
    search_config_digest: str
    estimation_identity_digest: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "estimation_identity_digest",
            _estimation_identity_digest(
                candidate_ids=self.candidate_ids,
                transform=self.transform,
                registered_objective=self.registered_objective,
                second_registered_objective=self.second_registered_objective,
                exact_identity_objective=self.exact_identity_objective,
                wrong_objectives=self.wrong_key_objectives,
                canonical_score=self.canonical_score,
                observation_score=self.observation_score,
                coverage_forward=self.coverage_forward,
                coverage_backward=self.coverage_backward,
                uniqueness_forward=self.uniqueness_forward,
                uniqueness_backward=self.uniqueness_backward,
                coverage=self.coverage,
                uniqueness=self.uniqueness,
                gap=self.gap,
                identity_margin=self.identity_margin,
                key_margin=self.key_margin,
                inlier_ratio=self.inlier_ratio,
                mean_residual=self.mean_residual,
                epsilon_inlier=self.epsilon_inlier,
                anchor_residuals=self.anchor_residuals,
                root_key_public_digest=self.registered_root_key_public_digest,
                observation_descriptor_digest=self.observation_descriptor_digest,
                observation_projection_digest=self.observation_projection_digest,
                observation_geometry_config_digest=(
                    self.observation_geometry_config_digest
                ),
                search_config_digest=self.search_config_digest,
            ),
        )


@dataclass(frozen=True, slots=True)
class _SearchCandidate:
    dihedral: str
    rotation_degrees: float
    log_scale: float
    translation_x: float
    translation_y: float
    matrix: torch.Tensor


@dataclass(frozen=True, slots=True)
class _CandidateEvaluation:
    candidate: _SearchCandidate
    objective: float
    canonical_score: float
    observation_score: float
    coverage_forward: float
    coverage_backward: float
    uniqueness_forward: float
    uniqueness_backward: float


def _matrix_key(matrix: torch.Tensor) -> bytes:
    return b"".join(
        pack(">f", float(value))
        for value in matrix.to(dtype=torch.float32).reshape(-1)
    )


def _candidate(
    dihedral: str,
    rotation_degrees: float,
    log_scale_value: float,
    translation_x: float,
    translation_y: float,
) -> _SearchCandidate:
    dihedral_lookup = dict(DIHEDRAL_MATRICES)
    if dihedral not in dihedral_lookup:
        raise GeometricTransformEstimatorError("unknown dihedral identity")
    angle = rotation_degrees * pi / 180.0
    cosine = torch.cos(torch.tensor(angle, dtype=torch.float64))
    sine = torch.sin(torch.tensor(angle, dtype=torch.float64))
    rotation = torch.tensor(
        ((cosine, -sine), (sine, cosine)), dtype=torch.float64
    )
    scale = torch.exp(torch.tensor(log_scale_value, dtype=torch.float64))
    dihedral_matrix = torch.tensor(dihedral_lookup[dihedral], dtype=torch.float64)
    linear = scale * (rotation @ dihedral_matrix)
    matrix = torch.cat(
        (
            linear,
            torch.tensor(
                ((translation_x,), (translation_y,)), dtype=torch.float64
            ),
        ),
        dim=1,
    ).to(dtype=torch.float32)
    return _SearchCandidate(
        dihedral=dihedral,
        rotation_degrees=float(rotation_degrees),
        log_scale=float(log_scale_value),
        translation_x=float(translation_x),
        translation_y=float(translation_y),
        matrix=matrix,
    )


def _within_search_bounds(candidate: _SearchCandidate) -> bool:
    tolerance = 1e-12
    return (
        abs(candidate.rotation_degrees) <= ROTATION_LIMIT_DEGREES + tolerance
        and abs(candidate.log_scale) <= LOG_SCALE_LIMIT + tolerance
        and abs(candidate.translation_x) <= TRANSLATION_LIMIT + tolerance
        and abs(candidate.translation_y) <= TRANSLATION_LIMIT + tolerance
    )


def _token_coordinates(
    original_grid_side: int,
    token_indices: Sequence[int],
) -> tuple[torch.Tensor, tuple[int, ...]]:
    if type(original_grid_side) is not int or original_grid_side < 2:
        raise GeometricTransformEstimatorError(
            "original grid side must be an integer greater than one"
        )
    normalized_indices = tuple(token_indices)
    if (
        not normalized_indices
        or any(
            type(index) is not int
            or index < 0
            or index >= original_grid_side**2
            for index in normalized_indices
        )
    ):
        raise GeometricTransformEstimatorError("sampled token indices are invalid")
    axis_positions = tuple(
        sorted({index % original_grid_side for index in normalized_indices})
    )
    row_positions = tuple(
        sorted({index // original_grid_side for index in normalized_indices})
    )
    if axis_positions != row_positions or len(axis_positions) < 2:
        raise GeometricTransformEstimatorError(
            "sampled tokens must use one shared square-grid axis"
        )
    expected_indices = tuple(
        row * original_grid_side + column
        for row in axis_positions
        for column in axis_positions
    )
    if normalized_indices != expected_indices:
        raise GeometricTransformEstimatorError(
            "sampled token order must be row-major over the sampled axis"
        )
    coordinates = torch.tensor(
        [
            (
                -1.0
                + 2.0 * (index % original_grid_side)
                / (original_grid_side - 1),
                -1.0
                + 2.0 * (index // original_grid_side)
                / (original_grid_side - 1),
            )
            for index in normalized_indices
        ],
        dtype=torch.float32,
    )
    return coordinates, axis_positions


def sampling_matrix(
    canonical_to_observed: torch.Tensor,
    *,
    original_grid_side: int,
    token_indices: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the exact align-corners bilinear W matrix and valid rows."""

    if not isinstance(canonical_to_observed, torch.Tensor):
        raise GeometricTransformEstimatorError("transform matrix must be a tensor")
    matrix = canonical_to_observed.to(device="cpu", dtype=torch.float32)
    if matrix.shape != (2, 3) or not bool(torch.isfinite(matrix).all()):
        raise GeometricTransformEstimatorError(
            "transform matrix must be finite with [2,3] shape"
        )
    coordinates, axis_positions = _token_coordinates(
        original_grid_side,
        token_indices,
    )
    token_count = len(token_indices)
    mapped = coordinates @ matrix[:, :2].transpose(0, 1) + matrix[:, 2]
    weights = torch.zeros((token_count, token_count), dtype=torch.float32)
    valid = (
        (mapped[:, 0] >= -1.0)
        & (mapped[:, 0] <= 1.0)
        & (mapped[:, 1] >= -1.0)
        & (mapped[:, 1] <= 1.0)
    )
    for row_index in range(token_count):
        if not bool(valid[row_index]):
            continue
        x_original = float(
            (mapped[row_index, 0] + 1.0) * (original_grid_side - 1) / 2.0
        )
        y_original = float(
            (mapped[row_index, 1] + 1.0) * (original_grid_side - 1) / 2.0
        )

        def bracket(value: float) -> tuple[int, int, float]:
            if value <= axis_positions[0]:
                return 0, 0, 0.0
            if value >= axis_positions[-1]:
                last = len(axis_positions) - 1
                return last, last, 0.0
            upper_index = next(
                index
                for index, position in enumerate(axis_positions)
                if position >= value
            )
            lower_index = upper_index - 1
            lower_value = axis_positions[lower_index]
            upper_value = axis_positions[upper_index]
            fraction = (value - lower_value) / (upper_value - lower_value)
            return lower_index, upper_index, fraction

        x_lower, x_upper, x_fraction = bracket(x_original)
        y_lower, y_upper, y_fraction = bracket(y_original)
        sampled_side = len(axis_positions)
        contributions = (
            (y_lower, x_lower, (1.0 - x_fraction) * (1.0 - y_fraction)),
            (y_lower, x_upper, x_fraction * (1.0 - y_fraction)),
            (y_upper, x_lower, (1.0 - x_fraction) * y_fraction),
            (y_upper, x_upper, x_fraction * y_fraction),
        )
        for y_index, x_index, value in contributions:
            weights[row_index, y_index * sampled_side + x_index] += float(value)
    return weights, valid


def _inverse_affine(matrix: torch.Tensor) -> torch.Tensor:
    linear = matrix[:, :2].to(dtype=torch.float64)
    determinant = torch.linalg.det(linear)
    if not isfinite(float(determinant)) or abs(float(determinant)) <= 1e-12:
        raise GeometricTransformEstimatorError("transform linear part is singular")
    inverse_linear = torch.linalg.inv(linear)
    inverse_translation = -(inverse_linear @ matrix[:, 2].to(dtype=torch.float64))
    return torch.cat(
        (inverse_linear, inverse_translation.unsqueeze(1)), dim=1
    ).to(dtype=torch.float32)


def _coverage_and_uniqueness(
    weights: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[float, float]:
    valid_count = int(valid.sum())
    coverage = valid_count / len(valid)
    if valid_count == 0:
        return coverage, 0.0
    selected = torch.argmax(weights[valid], dim=1)
    uniqueness = len(torch.unique(selected)) / valid_count
    return float(coverage), float(uniqueness)


def _warp_relation(
    sampling: torch.Tensor,
    relation: torch.Tensor,
) -> torch.Tensor:
    channels = [
        sampling @ relation[:, :, channel] @ sampling.transpose(0, 1)
        for channel in range(4)
    ]
    return torch.stack(channels, dim=-1)


def _evaluate_candidate(
    candidate: _SearchCandidate,
    observation: QkGeometrySyncResult,
    key_projections: Sequence[torch.Tensor],
) -> _CandidateEvaluation | None:
    geometry_layer = observation.layers[0]
    token_count = geometry_layer.token_count
    sampling_metadata = {
        "original_grid_side": geometry_layer.original_grid_side,
        "token_indices": geometry_layer.token_indices,
    }
    forward, valid_forward = sampling_matrix(
        candidate.matrix,
        **sampling_metadata,
    )
    backward, valid_backward = sampling_matrix(
        _inverse_affine(candidate.matrix),
        **sampling_metadata,
    )
    coverage_forward, uniqueness_forward = _coverage_and_uniqueness(
        forward, valid_forward
    )
    coverage_backward, uniqueness_backward = _coverage_and_uniqueness(
        backward, valid_backward
    )
    if min(coverage_forward, coverage_backward) < MINIMUM_COVERAGE:
        return None

    token_weights = forward @ torch.ones(token_count, dtype=torch.float32)
    canonical_pair_weights = token_weights.unsqueeze(1) * token_weights.unsqueeze(0)
    canonical_layer_scores: list[float] = []
    observation_layer_scores: list[float] = []
    try:
        for layer, key_projection in zip(
            observation.layers, key_projections, strict=True
        ):
            relation = layer.relation_tensor()
            canonical_relation = _warp_relation(forward, relation)
            expected_observation = _warp_relation(backward, key_projection)
            canonical_layer_scores.append(
                row_normalized_relation_score(
                    canonical_relation,
                    key_projection,
                    valid_rows=valid_forward,
                    pair_weights=canonical_pair_weights,
                )
            )
            observation_layer_scores.append(
                row_normalized_relation_score(
                    relation,
                    expected_observation,
                    valid_rows=valid_backward,
                )
            )
    except QkGeometrySyncError:
        return None

    canonical_score = sum(canonical_layer_scores) / len(canonical_layer_scores)
    observation_score = sum(observation_layer_scores) / len(
        observation_layer_scores
    )
    deficits = (
        (1.0 - coverage_forward)
        + (1.0 - coverage_backward)
        + (1.0 - uniqueness_forward)
        + (1.0 - uniqueness_backward)
    )
    objective = float(
        torch.tensor(
            0.10 * canonical_score + 0.90 * observation_score - 0.01 * deficits,
            dtype=torch.float32,
        )
    )
    if not isfinite(objective):
        return None
    return _CandidateEvaluation(
        candidate=candidate,
        objective=objective,
        canonical_score=float(canonical_score),
        observation_score=float(observation_score),
        coverage_forward=coverage_forward,
        coverage_backward=coverage_backward,
        uniqueness_forward=uniqueness_forward,
        uniqueness_backward=uniqueness_backward,
    )


def _coarse_candidates() -> Iterable[_SearchCandidate]:
    for (dihedral, _), rotation, log_scale_value, translation_x, translation_y in product(
        DIHEDRAL_MATRICES,
        COARSE_ROTATIONS,
        COARSE_LOG_SCALES,
        COARSE_TRANSLATIONS,
        COARSE_TRANSLATIONS,
    ):
        yield _candidate(
            dihedral,
            rotation,
            log_scale_value,
            translation_x,
            translation_y,
        )


def _highest_objective_evaluation(
    evaluations: Sequence[_CandidateEvaluation],
) -> _CandidateEvaluation:
    if not evaluations:
        raise GeometricTransformEstimatorError("no finite transform candidate was scored")
    selected_evaluation = evaluations[0]
    for evaluation in evaluations[1:]:
        if evaluation.objective > selected_evaluation.objective:
            selected_evaluation = evaluation
    return selected_evaluation


def _run_search(
    observation: QkGeometrySyncResult,
    key_projections: Sequence[torch.Tensor],
) -> tuple[_CandidateEvaluation, _CandidateEvaluation, _CandidateEvaluation]:
    seen: set[bytes] = set()
    evaluations: list[_CandidateEvaluation] = []
    for candidate in _coarse_candidates():
        key = _matrix_key(candidate.matrix)
        if key in seen:
            continue
        seen.add(key)
        evaluation = _evaluate_candidate(candidate, observation, key_projections)
        if evaluation is not None:
            evaluations.append(evaluation)
    coarse_selected_evaluation = _highest_objective_evaluation(evaluations)
    exact_identity = next(
        evaluation
        for evaluation in evaluations
        if evaluation.candidate.dihedral == "identity"
        and evaluation.candidate.rotation_degrees == 0.0
        and evaluation.candidate.log_scale == 0.0
        and evaluation.candidate.translation_x == 0.0
        and evaluation.candidate.translation_y == 0.0
    )

    current = coarse_selected_evaluation
    deltas = [
        (8.0, LOG_SCALE_LIMIT / 2.0, 0.14, 0.14),
        (8.0 / 3.0, LOG_SCALE_LIMIT / 6.0, 0.14 / 3.0, 0.14 / 3.0),
        (8.0 / 9.0, LOG_SCALE_LIMIT / 18.0, 0.14 / 9.0, 0.14 / 9.0),
    ]
    for rotation_delta, scale_delta, x_delta, y_delta in deltas:
        # The all-zero offset is the current candidate.  Its matrix was already
        # retained at its first occurrence, but it still participates in this
        # round so a worse neighborhood cannot move the next-round center.
        round_evaluations: list[_CandidateEvaluation] = [current]
        for rotation_offset, scale_offset, x_offset, y_offset in product(
            (0.0, -rotation_delta, rotation_delta),
            (0.0, -scale_delta, scale_delta),
            (0.0, -x_delta, x_delta),
            (0.0, -y_delta, y_delta),
        ):
            candidate = _candidate(
                current.candidate.dihedral,
                current.candidate.rotation_degrees + rotation_offset,
                current.candidate.log_scale + scale_offset,
                current.candidate.translation_x + x_offset,
                current.candidate.translation_y + y_offset,
            )
            if not _within_search_bounds(candidate):
                continue
            key = _matrix_key(candidate.matrix)
            if key in seen:
                continue
            seen.add(key)
            evaluation = _evaluate_candidate(candidate, observation, key_projections)
            if evaluation is not None:
                evaluations.append(evaluation)
                round_evaluations.append(evaluation)
        current = _highest_objective_evaluation(round_evaluations)

    selected_evaluation = _highest_objective_evaluation(evaluations)
    second_candidates = [
        evaluation
        for evaluation in evaluations
        if _matrix_key(evaluation.candidate.matrix)
        != _matrix_key(selected_evaluation.candidate.matrix)
    ]
    second = _highest_objective_evaluation(second_candidates)
    return selected_evaluation, second, exact_identity


def _boundary(candidate: _SearchCandidate) -> bool:
    tolerance = 1e-6
    return (
        abs(abs(candidate.rotation_degrees) - ROTATION_LIMIT_DEGREES) <= tolerance
        or abs(abs(candidate.log_scale) - LOG_SCALE_LIMIT) <= tolerance
        or abs(abs(candidate.translation_x) - TRANSLATION_LIMIT) <= tolerance
        or abs(abs(candidate.translation_y) - TRANSLATION_LIMIT) <= tolerance
    )


def _exact_identity(candidate: _SearchCandidate) -> bool:
    return (
        candidate.dihedral == "identity"
        and candidate.rotation_degrees == 0.0
        and candidate.log_scale == 0.0
        and candidate.translation_x == 0.0
        and candidate.translation_y == 0.0
        and bool(torch.equal(candidate.matrix, torch.eye(2, 3, dtype=torch.float32)))
    )


def _anchor_metrics(
    candidate: _SearchCandidate,
    original_grid_side: int,
    token_indices: Sequence[int],
    epsilon_inlier: float | None,
) -> tuple[tuple[float, ...], float | None, float]:
    coordinates, _ = _token_coordinates(original_grid_side, token_indices)
    matrix = candidate.matrix
    occupied: set[int] = set()
    residuals: list[float] = []
    inlier_count = 0
    for anchor_x, anchor_y in ANCHORS:
        anchor = torch.tensor((anchor_x, anchor_y), dtype=torch.float32)
        mapped = matrix[:, :2] @ anchor + matrix[:, 2]
        if (
            float(mapped[0]) < -1.0
            or float(mapped[0]) > 1.0
            or float(mapped[1]) < -1.0
            or float(mapped[1]) > 1.0
        ):
            residuals.append(inf)
            continue
        distances = torch.linalg.vector_norm(coordinates - mapped, dim=1)
        nearest = int(torch.argmin(distances))
        residual = float(distances[nearest])
        residuals.append(residual)
        if (
            epsilon_inlier is not None
            and nearest not in occupied
            and residual <= epsilon_inlier
        ):
            inlier_count += 1
        occupied.add(nearest)
    mean_residual = (
        sum(residuals) / len(residuals)
        if all(isfinite(value) for value in residuals)
        else inf
    )
    inlier_ratio = (
        None if epsilon_inlier is None else inlier_count / len(ANCHORS)
    )
    return tuple(residuals), inlier_ratio, mean_residual


def _search_config_digest(epsilon_inlier: float | None) -> str:
    identity = {
        "candidate_id": RECTIFICATION_CANDIDATE_ID,
        "coarse_log_scale": ["0", "-log_sqrt2", "+log_sqrt2"],
        "coarse_rotation_degrees": [0, -32, -16, 16, 32],
        "coarse_translation": ["0", "-0.28", "+0.28"],
        "dihedral_order": [name for name, _ in DIHEDRAL_MATRICES],
        "epsilon_inlier_decimal": (
            None
            if epsilon_inlier is None
            else format(epsilon_inlier, ".17g")
        ),
        "objective_weights": ["0.10", "0.90", "-0.01_deficits"],
        "refinement_rounds": 3,
        "wrong_key_indices": list(range(8)),
    }
    return sha256(stable_json_utf8(identity)).hexdigest()


def _estimation_identity_digest(
    *,
    candidate_ids: Sequence[str],
    transform: SimilarityTransform,
    registered_objective: float,
    second_registered_objective: float,
    exact_identity_objective: float,
    wrong_objectives: Sequence[float],
    canonical_score: float,
    observation_score: float,
    coverage_forward: float,
    coverage_backward: float,
    uniqueness_forward: float,
    uniqueness_backward: float,
    coverage: float,
    uniqueness: float,
    gap: float,
    identity_margin: float,
    key_margin: float,
    inlier_ratio: float | None,
    mean_residual: float,
    epsilon_inlier: float | None,
    anchor_residuals: Sequence[float],
    root_key_public_digest: str,
    observation_descriptor_digest: str,
    observation_projection_digest: str,
    observation_geometry_config_digest: str,
    search_config_digest: str,
) -> str:
    identity = {
        "candidate_ids": list(candidate_ids),
        "registered_objective": format(registered_objective, ".17g"),
        "wrong_key_objectives": [
            format(value, ".17g") for value in wrong_objectives
        ],
        "canonical_score": format(canonical_score, ".17g"),
        "coverage": format(coverage, ".17g"),
        "coverage_backward": format(coverage_backward, ".17g"),
        "coverage_forward": format(coverage_forward, ".17g"),
        "epsilon_inlier": (
            None
            if epsilon_inlier is None
            else format(epsilon_inlier, ".17g")
        ),
        "exact_identity_objective": format(exact_identity_objective, ".17g"),
        "gap": format(gap, ".17g"),
        "identity_margin": format(identity_margin, ".17g"),
        "inlier_ratio": (
            None
            if inlier_ratio is None
            else format(inlier_ratio, ".17g")
        ),
        "key_margin": format(key_margin, ".17g"),
        "mean_residual": format(mean_residual, ".17g"),
        "anchor_residuals": [
            format(value, ".17g") for value in anchor_residuals
        ],
        "observation_descriptor_digest": observation_descriptor_digest,
        "observation_geometry_config_digest": (
            observation_geometry_config_digest
        ),
        "observation_projection_digest": observation_projection_digest,
        "observation_score": format(observation_score, ".17g"),
        "registered_root_key_public_digest": root_key_public_digest,
        "search_config_digest": search_config_digest,
        "second_registered_objective": format(
            second_registered_objective, ".17g"
        ),
        "transform": {
            "is_exact_identity": transform.is_exact_identity,
            "continuous_parameter_on_search_boundary": (
                transform.continuous_parameter_on_search_boundary
            ),
            "dihedral": transform.dihedral,
            "log_scale": format(transform.log_scale, ".17g"),
            "matrix": [
                [format(value, ".17g") for value in row]
                for row in transform.matrix
            ],
            "residual_rotation_degrees": format(
                transform.residual_rotation_degrees,
                ".17g",
            ),
            "translation_x": format(transform.translation_x, ".17g"),
            "translation_y": format(transform.translation_y, ".17g"),
        },
        "uniqueness": format(uniqueness, ".17g"),
        "uniqueness_backward": format(uniqueness_backward, ".17g"),
        "uniqueness_forward": format(uniqueness_forward, ".17g"),
    }
    return sha256(stable_json_utf8(identity)).hexdigest()


def _validate_observation(
    observation: object,
    detection_key: str | DerivedWrongKeyMaterial,
) -> QkGeometrySyncResult:
    if type(observation) is not QkGeometrySyncResult:
        raise GeometricTransformEstimatorError(
            "observation must be QkGeometrySyncResult"
        )
    try:
        validate_qk_geometry_sync_result(observation, detection_key)
    except QkGeometrySyncError as exc:
        raise GeometricTransformEstimatorError(
            "Q/K geometry observation validation failed"
        ) from exc
    expected_role = (
        "wrong"
        if type(detection_key) is DerivedWrongKeyMaterial
        else "registered"
    )
    expected_wrong_index = (
        detection_key.wrong_key_index
        if type(detection_key) is DerivedWrongKeyMaterial
        else None
    )
    if (
        observation.key_role != expected_role
        or observation.wrong_key_index != expected_wrong_index
    ):
        raise GeometricTransformEstimatorError(
            "estimator observation must match the supplied geometry key role"
        )
    expected_public_digest = (
        detection_key.registered_root_key_public_digest
        if type(detection_key) is DerivedWrongKeyMaterial
        else identify_root_key(detection_key).root_key_public_digest
    )
    if expected_public_digest != observation.root_key_public_digest:
        raise GeometricTransformEstimatorError(
            "detection key does not match Q/K observation key family"
        )
    if len(observation.layers) != 2:
        raise GeometricTransformEstimatorError("two relation layers are required")
    token_counts = {layer.token_count for layer in observation.layers}
    if len(token_counts) != 1:
        raise GeometricTransformEstimatorError(
            "both relation layers must share one token grid"
        )
    return observation


def geometric_transform_estimator(
    observation: QkGeometrySyncResult,
    detection_key: str | DerivedWrongKeyMaterial,
    *,
    epsilon_inlier: float | None,
) -> GeometricTransformEstimation:
    """Search the frozen transform family and return raw metrics, never reliability."""

    if type(detection_key) not in {str, DerivedWrongKeyMaterial}:
        raise GeometricTransformEstimatorError(
            "detection_key must be root key text or derived wrong-key material"
        )
    if epsilon_inlier is not None and (
        isinstance(epsilon_inlier, bool)
        or not isinstance(epsilon_inlier, (int, float))
        or not isfinite(float(epsilon_inlier))
        or float(epsilon_inlier) <= 0.0
    ):
        raise GeometricTransformEstimatorError(
            "epsilon_inlier must be a fitted positive finite value"
        )
    epsilon = (
        None if epsilon_inlier is None else float(epsilon_inlier)
    )
    observation = _validate_observation(observation, detection_key)

    registered_projections = projection_for_detection_key(
        observation, detection_key
    )
    registered_selected, second, identity = _run_search(
        observation,
        registered_projections,
    )
    wrong_objectives: list[float] = []
    for wrong_index in range(8):
        wrong_material = derive_wrong_key_material(
            observation.root_key_public_digest,
            wrong_index,
        )
        wrong_projections = projection_for_detection_key(
            observation, wrong_material
        )
        wrong_key_selected, _, _ = _run_search(
            observation,
            wrong_projections,
        )
        wrong_objectives.append(wrong_key_selected.objective)

    anchor_residuals, inlier_ratio, mean_residual = _anchor_metrics(
        registered_selected.candidate,
        observation.layers[0].original_grid_side,
        observation.layers[0].token_indices,
        epsilon,
    )
    coverage = min(
        registered_selected.coverage_forward,
        registered_selected.coverage_backward,
    )
    uniqueness = min(
        registered_selected.uniqueness_forward,
        registered_selected.uniqueness_backward,
    )
    transform = SimilarityTransform(
        dihedral=registered_selected.candidate.dihedral,
        residual_rotation_degrees=(
            registered_selected.candidate.rotation_degrees
        ),
        log_scale=registered_selected.candidate.log_scale,
        translation_x=registered_selected.candidate.translation_x,
        translation_y=registered_selected.candidate.translation_y,
        matrix=tuple(
            tuple(float(value) for value in row)
            for row in registered_selected.candidate.matrix
        ),
        is_exact_identity=_exact_identity(registered_selected.candidate),
        continuous_parameter_on_search_boundary=_boundary(
            registered_selected.candidate
        ),
    )
    wrong_objective_values = tuple(wrong_objectives)
    search_config_digest = _search_config_digest(epsilon)
    return GeometricTransformEstimation(
        candidate_ids=(
            "key_schedule_sha256_counter",
            QK_CANDIDATE_ID,
            RECTIFICATION_CANDIDATE_ID,
        ),
        transform=transform,
        registered_objective=registered_selected.objective,
        second_registered_objective=second.objective,
        exact_identity_objective=identity.objective,
        wrong_key_objectives=wrong_objective_values,
        canonical_score=registered_selected.canonical_score,
        observation_score=registered_selected.observation_score,
        coverage_forward=registered_selected.coverage_forward,
        coverage_backward=registered_selected.coverage_backward,
        uniqueness_forward=registered_selected.uniqueness_forward,
        uniqueness_backward=registered_selected.uniqueness_backward,
        coverage=coverage,
        uniqueness=uniqueness,
        gap=registered_selected.objective - second.objective,
        identity_margin=registered_selected.objective - identity.objective,
        key_margin=registered_selected.objective - max(wrong_objectives),
        inlier_ratio=inlier_ratio,
        mean_residual=mean_residual,
        epsilon_inlier=epsilon,
        anchor_residuals=anchor_residuals,
        registered_root_key_public_digest=observation.root_key_public_digest,
        observation_descriptor_digest=observation.descriptor_digest,
        observation_projection_digest=observation.projection_digest,
        observation_geometry_config_digest=observation.geometry_config_digest,
        search_config_digest=search_config_digest,
    )


def validate_geometric_transform_estimation(
    estimation: GeometricTransformEstimation,
) -> None:
    """Independently recheck transform structure, raw-metric algebra, and identity."""

    if type(estimation) is not GeometricTransformEstimation:
        raise GeometricTransformEstimatorError(
            "estimation must be GeometricTransformEstimation"
        )
    if estimation.candidate_ids != (
        "key_schedule_sha256_counter",
        QK_CANDIDATE_ID,
        RECTIFICATION_CANDIDATE_ID,
    ):
        raise GeometricTransformEstimatorError(
            "transform estimation candidate identity mismatch"
        )
    if len(estimation.wrong_key_objectives) != 8:
        raise GeometricTransformEstimatorError(
            "transform estimation must contain eight wrong-key objectives"
        )
    if len(estimation.anchor_residuals) != len(ANCHORS):
        raise GeometricTransformEstimatorError(
            "transform estimation must contain twelve anchor residuals"
        )
    if (estimation.epsilon_inlier is None) != (
        estimation.inlier_ratio is None
    ):
        raise GeometricTransformEstimatorError(
            "epsilon and inlier ratio must share one fitted state"
        )
    if estimation.epsilon_inlier is not None:
        if (
            isinstance(estimation.epsilon_inlier, bool)
            or not isinstance(estimation.epsilon_inlier, (int, float))
            or not isfinite(float(estimation.epsilon_inlier))
            or float(estimation.epsilon_inlier) <= 0.0
            or isinstance(estimation.inlier_ratio, bool)
            or not isinstance(estimation.inlier_ratio, (int, float))
            or not isfinite(float(estimation.inlier_ratio))
            or not 0.0 <= float(estimation.inlier_ratio) <= 1.0
        ):
            raise GeometricTransformEstimatorError(
                "fitted epsilon or inlier ratio is invalid"
            )
    derived_metrics = (
        (
            estimation.gap,
            estimation.registered_objective
            - estimation.second_registered_objective,
            "gap",
        ),
        (
            estimation.identity_margin,
            estimation.registered_objective
            - estimation.exact_identity_objective,
            "identity_margin",
        ),
        (
            estimation.key_margin,
            estimation.registered_objective
            - max(estimation.wrong_key_objectives),
            "key_margin",
        ),
        (
            estimation.coverage,
            min(estimation.coverage_forward, estimation.coverage_backward),
            "coverage",
        ),
        (
            estimation.uniqueness,
            min(
                estimation.uniqueness_forward,
                estimation.uniqueness_backward,
            ),
            "uniqueness",
        ),
    )
    for actual, derived, metric_name in derived_metrics:
        if not isclose(
            float(actual),
            float(derived),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise GeometricTransformEstimatorError(
                f"transform estimation {metric_name} algebra mismatch"
            )
    rebuilt = _candidate(
        estimation.transform.dihedral,
        estimation.transform.residual_rotation_degrees,
        estimation.transform.log_scale,
        estimation.transform.translation_x,
        estimation.transform.translation_y,
    )
    if _matrix_key(rebuilt.matrix) != _matrix_key(estimation.transform.tensor()):
        raise GeometricTransformEstimatorError(
            "transform parameters do not reconstruct the stored matrix"
        )
    if (
        estimation.transform.is_exact_identity
        != _exact_identity(rebuilt)
    ):
        raise GeometricTransformEstimatorError(
            "exact-identity flag does not match transform parameters"
        )
    if (
        estimation.transform.continuous_parameter_on_search_boundary
        != _boundary(rebuilt)
    ):
        raise GeometricTransformEstimatorError(
            "search-boundary flag does not match transform parameters"
        )
    expected = _estimation_identity_digest(
        candidate_ids=estimation.candidate_ids,
        transform=estimation.transform,
        registered_objective=estimation.registered_objective,
        second_registered_objective=estimation.second_registered_objective,
        exact_identity_objective=estimation.exact_identity_objective,
        wrong_objectives=estimation.wrong_key_objectives,
        canonical_score=estimation.canonical_score,
        observation_score=estimation.observation_score,
        coverage_forward=estimation.coverage_forward,
        coverage_backward=estimation.coverage_backward,
        uniqueness_forward=estimation.uniqueness_forward,
        uniqueness_backward=estimation.uniqueness_backward,
        coverage=estimation.coverage,
        uniqueness=estimation.uniqueness,
        gap=estimation.gap,
        identity_margin=estimation.identity_margin,
        key_margin=estimation.key_margin,
        inlier_ratio=estimation.inlier_ratio,
        mean_residual=estimation.mean_residual,
        epsilon_inlier=estimation.epsilon_inlier,
        anchor_residuals=estimation.anchor_residuals,
        root_key_public_digest=estimation.registered_root_key_public_digest,
        observation_descriptor_digest=estimation.observation_descriptor_digest,
        observation_projection_digest=estimation.observation_projection_digest,
        observation_geometry_config_digest=(
            estimation.observation_geometry_config_digest
        ),
        search_config_digest=estimation.search_config_digest,
    )
    if expected != estimation.estimation_identity_digest:
        raise GeometricTransformEstimatorError(
            "transform estimation identity digest mismatch"
        )


__all__ = [
    "GeometricTransformEstimation",
    "GeometricTransformEstimatorError",
    "SimilarityTransform",
    "geometric_transform_estimator",
    "sampling_matrix",
    "validate_geometric_transform_estimation",
]
