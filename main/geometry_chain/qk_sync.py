"""Keyed Q/K relation observation and geometry-synchronization write.

The public observation consumes actual attention query/key tensors.  It does
not accept hidden-state proxies, digests in place of tensors, generation
caches, or content scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import isclose, isfinite, isqrt, sqrt
from struct import pack
from typing import Callable, Mapping, Sequence

import torch

from main.shared.key_schedule import (
    DEFAULT_CONFIG as DEFAULT_KEY_SCHEDULE_CONFIG,
    DerivedWrongKeyMaterial,
    KeyScheduleError,
    derive_wrong_key_stream,
    identify_root_key,
    key_schedule_sha256_counter,
    stable_json_utf8,
)

QK_CANDIDATE_ID = "qk_relation_similarity"
RUNTIME_CANDIDATE_ID = "runtime_sd35_flowmatch"
MODEL_REVISION = "b940f670f0eda2d07fbb75229e779da1ad11eb80"
REGISTERED_LAYERS = (
    "transformer_blocks.0.attn",
    "transformer_blocks.23.attn",
)
CHANNEL_POLARITY = (1.0, -1.0, 1.0, 1.0)
RANK_TEMPERATURE = 0.25
MAX_GRID_SIDE = 8
CONTENT_RELATIVE_L2 = 0.012
GEOMETRY_RATIOS = (1.0 / 16.0, 1.0 / 8.0, 1.0 / 4.0)
LINE_SEARCH_FACTORS = tuple(0.5**index for index in range(8))


class QkGeometrySyncError(ValueError):
    """Q/K observation or synchronization write violates the frozen candidate."""


@dataclass(frozen=True, slots=True)
class QkLayerObservation:
    """One registered layer's real projected and normalized Q/K tensors."""

    layer_name: str
    query: torch.Tensor
    attention_key: torch.Tensor
    operator_identity: str


@dataclass(frozen=True, slots=True)
class QkLayerRelation:
    """Immutable values and identities for one layer's four-channel relation."""

    layer_name: str
    head_count: int
    head_width: int
    original_grid_side: int
    token_indices: tuple[int, ...]
    token_count: int
    relation_shape: tuple[int, int, int]
    relation_values: tuple[float, ...]
    projection_values: tuple[float, ...]
    relation_score: float
    key_domain_digest: str
    descriptor_digest: str
    projection_digest: str
    operator_identity: str

    def relation_tensor(self) -> torch.Tensor:
        """Reconstruct the frozen CPU float32 relation tensor."""

        return torch.tensor(self.relation_values, dtype=torch.float32).reshape(
            self.relation_shape
        )

    def projection_tensor(self) -> torch.Tensor:
        """Reconstruct the frozen CPU float32 keyed projection tensor."""

        return torch.tensor(self.projection_values, dtype=torch.float32).reshape(
            self.relation_shape
        )


@dataclass(frozen=True, slots=True)
class QkGeometrySyncResult:
    """Two-layer blind Q/K relation observation bound to one geometry key."""

    candidate_ids: tuple[str, str, str]
    model_revision: str
    layers: tuple[QkLayerRelation, ...]
    relation_score: float
    root_key_public_digest: str
    key_role: str
    wrong_key_index: int | None
    descriptor_digest: str
    projection_digest: str
    geometry_config_digest: str


@dataclass(frozen=True, slots=True)
class GeometrySynchronizationWriteResult:
    """Actual-dtype line-search outcome for the geometry synchronization write."""

    accepted: bool
    status: str
    geometry_ratio: float
    line_search_factor: float | None
    baseline_score: float
    accepted_score: float | None
    geometry_relative_l2_actual: float | None
    total_relative_l2_actual: float | None
    content_projection_relative: float | None
    written_latent: torch.Tensor | None


def _finite_float32_tensor(value: object, role: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise QkGeometrySyncError(f"{role} must be a torch.Tensor")
    if value.ndim != 3:
        raise QkGeometrySyncError(f"{role} must have [heads,tokens,head_width] shape")
    if value.shape[0] <= 0 or value.shape[1] <= 1 or value.shape[2] <= 0:
        raise QkGeometrySyncError(f"{role} dimensions must be positive")
    converted = value.to(device="cpu", dtype=torch.float32)
    if not bool(torch.isfinite(converted).all()):
        raise QkGeometrySyncError(f"{role} must contain only finite values")
    return converted


def _sampled_grid_indices(token_count: int) -> tuple[int, tuple[int, ...]]:
    side = isqrt(token_count)
    if side * side != token_count or side < 2:
        raise QkGeometrySyncError("Q/K image tokens must form a square grid of side >= 2")
    if side <= MAX_GRID_SIDE:
        positions = tuple(range(side))
    else:
        positions = tuple(
            int(round(index * (side - 1) / (MAX_GRID_SIDE - 1)))
            for index in range(MAX_GRID_SIDE)
        )
    indices = tuple(row * side + column for row in positions for column in positions)
    return side, indices


def _relation_tensor(
    query: torch.Tensor,
    attention_key: torch.Tensor,
) -> tuple[torch.Tensor, int, tuple[int, ...]]:
    query_cpu = _finite_float32_tensor(query, "query")
    key_cpu = _finite_float32_tensor(attention_key, "attention_key")
    if query_cpu.shape != key_cpu.shape:
        raise QkGeometrySyncError("query and attention_key shapes must match exactly")

    head_count, original_token_count, head_width = query_cpu.shape
    original_grid_side, token_indices = _sampled_grid_indices(original_token_count)
    index_tensor = torch.tensor(token_indices, dtype=torch.long)
    sampled_query = query_cpu.index_select(1, index_tensor)
    sampled_key = key_cpu.index_select(1, index_tensor)
    attention = torch.matmul(sampled_query, sampled_key.transpose(-1, -2)) / sqrt(
        head_width
    )
    centered_by_head = attention - attention.mean(dim=-1, keepdim=True)
    centered_logits = centered_by_head.mean(dim=0)
    probability = torch.softmax(attention, dim=-1).mean(dim=0)
    probability_row = probability / probability.sum(dim=-1, keepdim=True).clamp_min(
        1e-12
    )

    token_count = len(token_indices)
    pairwise_rank_logits = (
        centered_logits.unsqueeze(1) - centered_logits.unsqueeze(2)
    ) / RANK_TEMPERATURE
    rank_terms = torch.sigmoid(pairwise_rank_logits)
    off_diagonal = ~torch.eye(token_count, dtype=torch.bool)
    descending_rank = (
        1.0 + (rank_terms * off_diagonal.unsqueeze(0)).sum(dim=-1)
    ) / token_count

    coordinates = torch.tensor(
        [
            (
                -1.0
                + 2.0 * (token_index % original_grid_side) / (original_grid_side - 1),
                -1.0
                + 2.0 * (token_index // original_grid_side) / (original_grid_side - 1),
            )
            for token_index in token_indices
        ],
        dtype=torch.float32,
    )
    normalized_distance = torch.cdist(coordinates, coordinates) / (2.0 * sqrt(2.0))
    probability_centered = probability_row - probability_row.mean(
        dim=-1, keepdim=True
    )
    distance_centered = normalized_distance - normalized_distance.mean(
        dim=-1, keepdim=True
    )
    relation = torch.stack(
        (
            centered_logits,
            descending_rank,
            probability_row,
            probability_centered * distance_centered,
        ),
        dim=-1,
    ).to(dtype=torch.float32)
    if relation.shape != (token_count, token_count, 4):
        raise QkGeometrySyncError("four-channel relation construction failed")
    if not bool(torch.isfinite(relation).all()):
        raise QkGeometrySyncError("four-channel relation contains non-finite values")
    return relation, original_grid_side, token_indices


def qk_relation_tensor(
    query: torch.Tensor,
    attention_key: torch.Tensor,
) -> torch.Tensor:
    """Return the differentiable four-channel relation for actual Q/K tensors."""

    relation, _, _ = _relation_tensor(query, attention_key)
    return relation


def _geometry_projection(
    detection_key: str | DerivedWrongKeyMaterial,
    *,
    layer_name: str,
    token_count: int,
    model_revision: str,
) -> tuple[torch.Tensor, str, str, str, int | None]:
    domain_fields = {
        "candidate_id": QK_CANDIDATE_ID,
        "operator": "attention_relation_signs",
        "responsibility_domain": "geometry_sync",
        "model_revision": model_revision,
        "layer_name": layer_name,
        "token_count": token_count,
        "tensor_role": "pair_uniform",
    }
    try:
        if type(detection_key) is str:
            stream = key_schedule_sha256_counter(
                detection_key,
                domain_fields,
                (token_count, token_count),
                distribution="uniform",
            )
            public_digest = identify_root_key(
                detection_key
            ).root_key_public_digest
            key_role = "registered"
            wrong_key_index = None
        elif type(detection_key) is DerivedWrongKeyMaterial:
            stream = derive_wrong_key_stream(
                detection_key,
                domain_fields,
                (token_count, token_count),
                distribution="uniform",
            )
            public_digest = detection_key.registered_root_key_public_digest
            key_role = "wrong"
            wrong_key_index = detection_key.wrong_key_index
        else:
            raise QkGeometrySyncError(
                "detection_key must be root text or DerivedWrongKeyMaterial"
            )
    except KeyScheduleError as exc:
        raise QkGeometrySyncError("geometry key projection derivation failed") from exc

    uniform = torch.tensor(stream.values, dtype=torch.float32).reshape(
        token_count, token_count
    )
    signs = torch.where(uniform >= 0.5, 1.0, -1.0)
    upper = torch.triu(signs, diagonal=1)
    symmetric = upper + upper.transpose(0, 1)
    polarity = torch.tensor(CHANNEL_POLARITY, dtype=torch.float32)
    projection = symmetric.unsqueeze(-1) * polarity
    return projection, stream.domain_digest, public_digest, key_role, wrong_key_index


def _row_normalized_channel_scores(
    relation: torch.Tensor,
    projection: torch.Tensor,
    *,
    valid_rows: torch.Tensor | None = None,
    pair_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if relation.shape != projection.shape or relation.ndim != 3 or relation.shape[-1] != 4:
        raise QkGeometrySyncError("relation and projection must share [n,n,4] shape")
    token_count = relation.shape[0]
    if relation.shape[1] != token_count or token_count <= 1:
        raise QkGeometrySyncError("relation token axes must be square")
    if valid_rows is None:
        valid_rows = torch.ones(token_count, dtype=torch.bool)
    if valid_rows.shape != (token_count,) or valid_rows.dtype != torch.bool:
        raise QkGeometrySyncError("valid_rows must be a boolean token vector")
    if pair_weights is None:
        weights = torch.ones((token_count, token_count), dtype=torch.float32)
    else:
        if pair_weights.shape != (token_count, token_count):
            raise QkGeometrySyncError("pair_weights must have [n,n] shape")
        weights = pair_weights.to(dtype=torch.float32)
        if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
            raise QkGeometrySyncError("pair_weights must be finite and non-negative")
    weights = weights.clone()
    weights.fill_diagonal_(0.0)

    expanded_weights = weights.unsqueeze(-1)
    weight_sums = weights.sum(dim=1, keepdim=True).unsqueeze(-1)
    safe_weight_sums = weight_sums.clamp_min(torch.finfo(torch.float32).tiny)
    relation_means = (expanded_weights * relation).sum(
        dim=1, keepdim=True
    ) / safe_weight_sums
    projection_means = (expanded_weights * projection).sum(
        dim=1, keepdim=True
    ) / safe_weight_sums
    relation_centered = relation - relation_means
    projection_centered = projection - projection_means
    relation_energy = (
        expanded_weights * relation_centered.square()
    ).sum(dim=1)
    projection_energy = (
        expanded_weights * projection_centered.square()
    ).sum(dim=1)
    numerator = (
        expanded_weights * relation_centered * projection_centered
    ).sum(dim=1)
    usable = (
        valid_rows.unsqueeze(1)
        & (weight_sums.squeeze(1) > 0.0)
        & (relation_energy > 1e-24)
        & (projection_energy > 1e-24)
    )
    row_scores = numerator / torch.sqrt(
        (relation_energy * projection_energy).clamp_min(
            torch.finfo(torch.float32).tiny
        )
    )
    channel_scores: list[torch.Tensor] = []
    for channel_index in range(4):
        channel_usable = usable[:, channel_index]
        if not bool(channel_usable.any()):
            raise QkGeometrySyncError(
                "each relation layer/channel must contain at least one valid row"
            )
        channel_scores.append(row_scores[channel_usable, channel_index].mean())
    return torch.stack(channel_scores)


def row_normalized_relation_score(
    relation: torch.Tensor,
    projection: torch.Tensor,
    *,
    valid_rows: torch.Tensor | None = None,
    pair_weights: torch.Tensor | None = None,
) -> float:
    """Frozen equal-weight four-channel row correlation."""

    scores = _row_normalized_channel_scores(
        relation,
        projection,
        valid_rows=valid_rows,
        pair_weights=pair_weights,
    )
    value = float(scores.mean())
    if not isfinite(value):
        raise QkGeometrySyncError("relation score must be finite")
    return value


def differentiable_qk_relation_objective(
    observations: Sequence[QkLayerObservation],
    detection_key: str | DerivedWrongKeyMaterial,
    *,
    model_revision: str = MODEL_REVISION,
) -> torch.Tensor:
    """Return the frozen two-layer keyed relation score as a scalar tensor."""

    if type(model_revision) is not str or model_revision != MODEL_REVISION:
        raise QkGeometrySyncError("model_revision must match runtime_sd35_flowmatch")
    if isinstance(observations, (str, bytes)) or not isinstance(
        observations, Sequence
    ):
        raise QkGeometrySyncError(
            "observations must be the two registered Q/K layers"
        )
    if len(observations) != len(REGISTERED_LAYERS):
        raise QkGeometrySyncError("exactly two registered Q/K layers are required")

    layer_scores: list[torch.Tensor] = []
    public_digest: str | None = None
    key_role: str | None = None
    wrong_key_index: int | None = None
    for expected_layer, observation in zip(
        REGISTERED_LAYERS,
        observations,
        strict=True,
    ):
        if type(observation) is not QkLayerObservation:
            raise QkGeometrySyncError("each observation must be QkLayerObservation")
        if observation.layer_name != expected_layer:
            raise QkGeometrySyncError("Q/K layer order or identity mismatch")
        if (
            type(observation.operator_identity) is not str
            or not observation.operator_identity
        ):
            raise QkGeometrySyncError(
                "operator_identity must be a non-empty string"
            )
        relation, _, _ = _relation_tensor(
            observation.query,
            observation.attention_key,
        )
        projection, _, current_digest, current_role, current_wrong_index = (
            _geometry_projection(
                detection_key,
                layer_name=observation.layer_name,
                token_count=relation.shape[0],
                model_revision=model_revision,
            )
        )
        if public_digest is None:
            public_digest = current_digest
            key_role = current_role
            wrong_key_index = current_wrong_index
        elif (
            current_digest != public_digest
            or current_role != key_role
            or current_wrong_index != wrong_key_index
        ):
            raise QkGeometrySyncError("geometry key identity changed across layers")
        layer_scores.append(
            _row_normalized_channel_scores(
                relation,
                projection.to(device=relation.device, dtype=relation.dtype),
            ).mean()
        )

    objective = torch.stack(layer_scores).mean()
    if objective.ndim != 0 or objective.dtype is not torch.float32:
        raise QkGeometrySyncError(
            "differentiable relation objective must be one float32 scalar"
        )
    if not bool(torch.isfinite(objective)):
        raise QkGeometrySyncError("differentiable relation objective must be finite")
    return objective


def _tensor_float32_digest(value: torch.Tensor) -> str:
    flattened = value.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
    return sha256(b"".join(pack(">f", float(item)) for item in flattened)).hexdigest()


def _result_config_digest(
    layers: Sequence[QkLayerRelation],
    model_revision: str,
) -> str:
    identity = {
        "candidate_ids": [
            "key_schedule_sha256_counter",
            RUNTIME_CANDIDATE_ID,
            QK_CANDIDATE_ID,
        ],
        "channel_polarity": [1, -1, 1, 1],
        "key_schedule_config_digest": DEFAULT_KEY_SCHEDULE_CONFIG.config_digest,
        "layers": [
            {
                "head_count": layer.head_count,
                "head_width": layer.head_width,
                "layer_name": layer.layer_name,
                "operator_identity": layer.operator_identity,
                "original_grid_side": layer.original_grid_side,
                "token_indices": list(layer.token_indices),
            }
            for layer in layers
        ],
        "max_grid_side": MAX_GRID_SIDE,
        "model_revision": model_revision,
        "rank_temperature_ratio": "1/4",
        "row_correlation_weights": "uniform_off_diagonal",
    }
    return sha256(stable_json_utf8(identity)).hexdigest()


def _aggregate_descriptor_digest(layers: Sequence[QkLayerRelation]) -> str:
    return sha256(
        stable_json_utf8(
            {
                "layer_order": list(REGISTERED_LAYERS),
                "layer_digests": [layer.descriptor_digest for layer in layers],
            }
        )
    ).hexdigest()


def _aggregate_projection_digest(layers: Sequence[QkLayerRelation]) -> str:
    return sha256(
        stable_json_utf8(
            {
                "layer_order": list(REGISTERED_LAYERS),
                "layer_digests": [layer.projection_digest for layer in layers],
                "polarity": [1, -1, 1, 1],
            }
        )
    ).hexdigest()


def qk_geometry_sync(
    observations: Sequence[QkLayerObservation],
    detection_key: str | DerivedWrongKeyMaterial,
    *,
    model_revision: str = MODEL_REVISION,
) -> QkGeometrySyncResult:
    """Build the frozen two-layer relation and keyed synchronization objective."""

    if type(model_revision) is not str or model_revision != MODEL_REVISION:
        raise QkGeometrySyncError("model_revision must match runtime_sd35_flowmatch")
    if isinstance(observations, (str, bytes)) or not isinstance(observations, Sequence):
        raise QkGeometrySyncError("observations must be the two registered Q/K layers")
    if len(observations) != len(REGISTERED_LAYERS):
        raise QkGeometrySyncError("exactly two registered Q/K layers are required")

    layer_results: list[QkLayerRelation] = []
    public_digest: str | None = None
    key_role: str | None = None
    wrong_key_index: int | None = None
    for expected_layer, observation in zip(REGISTERED_LAYERS, observations, strict=True):
        if type(observation) is not QkLayerObservation:
            raise QkGeometrySyncError("each observation must be QkLayerObservation")
        if observation.layer_name != expected_layer:
            raise QkGeometrySyncError("Q/K layer order or identity mismatch")
        if type(observation.operator_identity) is not str or not observation.operator_identity:
            raise QkGeometrySyncError("operator_identity must be a non-empty string")
        relation, original_grid_side, token_indices = _relation_tensor(
            observation.query,
            observation.attention_key,
        )
        token_count = relation.shape[0]
        projection, domain_digest, current_digest, current_role, current_wrong_index = (
            _geometry_projection(
                detection_key,
                layer_name=observation.layer_name,
                token_count=token_count,
                model_revision=model_revision,
            )
        )
        if public_digest is None:
            public_digest = current_digest
            key_role = current_role
            wrong_key_index = current_wrong_index
        elif (
            current_digest != public_digest
            or current_role != key_role
            or current_wrong_index != wrong_key_index
        ):
            raise QkGeometrySyncError("geometry key identity changed across layers")
        layer_results.append(
            QkLayerRelation(
                layer_name=observation.layer_name,
                head_count=int(observation.query.shape[0]),
                head_width=int(observation.query.shape[2]),
                original_grid_side=original_grid_side,
                token_indices=token_indices,
                token_count=token_count,
                relation_shape=(token_count, token_count, 4),
                relation_values=tuple(float(item) for item in relation.reshape(-1)),
                projection_values=tuple(float(item) for item in projection.reshape(-1)),
                relation_score=row_normalized_relation_score(relation, projection),
                key_domain_digest=domain_digest,
                descriptor_digest=_tensor_float32_digest(relation),
                projection_digest=_tensor_float32_digest(projection),
                operator_identity=observation.operator_identity,
            )
        )

    layers = tuple(layer_results)
    result = QkGeometrySyncResult(
        candidate_ids=(
            "key_schedule_sha256_counter",
            RUNTIME_CANDIDATE_ID,
            QK_CANDIDATE_ID,
        ),
        model_revision=model_revision,
        layers=layers,
        relation_score=sum(layer.relation_score for layer in layers) / len(layers),
        root_key_public_digest=public_digest or "",
        key_role=key_role or "",
        wrong_key_index=wrong_key_index,
        descriptor_digest=_aggregate_descriptor_digest(layers),
        projection_digest=_aggregate_projection_digest(layers),
        geometry_config_digest=_result_config_digest(layers, model_revision),
    )
    validate_qk_geometry_sync_result(result, detection_key)
    return result


def validate_qk_geometry_sync_result(
    result: QkGeometrySyncResult,
    detection_key: str | DerivedWrongKeyMaterial,
) -> None:
    """Recheck the complete two-layer relation, projection, and digest structure."""

    if type(result) is not QkGeometrySyncResult:
        raise QkGeometrySyncError("result must be QkGeometrySyncResult")
    if result.candidate_ids != (
        "key_schedule_sha256_counter",
        RUNTIME_CANDIDATE_ID,
        QK_CANDIDATE_ID,
    ):
        raise QkGeometrySyncError("Q/K result candidate identity mismatch")
    if result.model_revision != MODEL_REVISION:
        raise QkGeometrySyncError("Q/K result model revision mismatch")
    if len(result.layers) != len(REGISTERED_LAYERS):
        raise QkGeometrySyncError("Q/K result must contain two registered layers")

    expected_public_digest: str | None = None
    expected_key_role: str | None = None
    expected_wrong_key_index: int | None = None
    for expected_layer_name, layer in zip(
        REGISTERED_LAYERS,
        result.layers,
        strict=True,
    ):
        if type(layer) is not QkLayerRelation or layer.layer_name != expected_layer_name:
            raise QkGeometrySyncError("Q/K layer order or type mismatch")
        if (
            type(layer.head_count) is not int
            or layer.head_count <= 0
            or type(layer.head_width) is not int
            or layer.head_width <= 0
        ):
            raise QkGeometrySyncError("Q/K head metadata must be positive integers")
        if type(layer.original_grid_side) is not int or layer.original_grid_side < 2:
            raise QkGeometrySyncError("Q/K original grid side is invalid")
        original_token_count = layer.original_grid_side**2
        _, expected_indices = _sampled_grid_indices(original_token_count)
        if layer.token_indices != expected_indices:
            raise QkGeometrySyncError("Q/K sampled token metadata mismatch")
        if layer.token_count != len(layer.token_indices):
            raise QkGeometrySyncError("Q/K token count does not match sampled indices")
        expected_shape = (layer.token_count, layer.token_count, 4)
        if layer.relation_shape != expected_shape:
            raise QkGeometrySyncError("Q/K relation shape metadata mismatch")
        expected_value_count = layer.token_count * layer.token_count * 4
        if (
            len(layer.relation_values) != expected_value_count
            or len(layer.projection_values) != expected_value_count
        ):
            raise QkGeometrySyncError("Q/K relation or projection value count mismatch")
        relation = layer.relation_tensor()
        projection = layer.projection_tensor()
        if not bool(torch.isfinite(relation).all()) or not bool(
            torch.isfinite(projection).all()
        ):
            raise QkGeometrySyncError("Q/K relation and projection must be finite")
        if layer.descriptor_digest != _tensor_float32_digest(relation):
            raise QkGeometrySyncError("Q/K layer descriptor digest mismatch")
        if layer.projection_digest != _tensor_float32_digest(projection):
            raise QkGeometrySyncError("Q/K layer projection digest mismatch")
        (
            expected_projection,
            expected_domain_digest,
            current_public_digest,
            current_key_role,
            current_wrong_key_index,
        ) = _geometry_projection(
            detection_key,
            layer_name=layer.layer_name,
            token_count=layer.token_count,
            model_revision=result.model_revision,
        )
        if (
            not torch.equal(projection, expected_projection)
            or layer.key_domain_digest != expected_domain_digest
        ):
            raise QkGeometrySyncError(
                "Q/K projection or key-domain identity mismatch"
            )
        if expected_public_digest is None:
            expected_public_digest = current_public_digest
            expected_key_role = current_key_role
            expected_wrong_key_index = current_wrong_key_index
        elif (
            current_public_digest != expected_public_digest
            or current_key_role != expected_key_role
            or current_wrong_key_index != expected_wrong_key_index
        ):
            raise QkGeometrySyncError("Q/K key identity changed across layers")
        expected_score = row_normalized_relation_score(relation, projection)
        if not isclose(
            layer.relation_score,
            expected_score,
            rel_tol=1e-7,
            abs_tol=1e-7,
        ):
            raise QkGeometrySyncError("Q/K layer relation score mismatch")
        if type(layer.operator_identity) is not str or not layer.operator_identity:
            raise QkGeometrySyncError("Q/K operator identity is invalid")

    if (
        result.root_key_public_digest != expected_public_digest
        or result.key_role != expected_key_role
        or result.wrong_key_index != expected_wrong_key_index
    ):
        raise QkGeometrySyncError("Q/K result key identity mismatch")
    expected_relation_score = sum(
        layer.relation_score for layer in result.layers
    ) / len(result.layers)
    if not isclose(
        result.relation_score,
        expected_relation_score,
        rel_tol=1e-7,
        abs_tol=1e-7,
    ):
        raise QkGeometrySyncError("Q/K aggregate relation score mismatch")
    if result.descriptor_digest != _aggregate_descriptor_digest(result.layers):
        raise QkGeometrySyncError("Q/K aggregate descriptor digest mismatch")
    if result.projection_digest != _aggregate_projection_digest(result.layers):
        raise QkGeometrySyncError("Q/K aggregate projection digest mismatch")
    if result.geometry_config_digest != _result_config_digest(
        result.layers,
        result.model_revision,
    ):
        raise QkGeometrySyncError("Q/K geometry configuration digest mismatch")


def projection_for_detection_key(
    observation: QkGeometrySyncResult,
    detection_key: str | DerivedWrongKeyMaterial,
) -> tuple[torch.Tensor, ...]:
    """Rebuild the per-layer geometry projection for estimator key comparisons."""

    if type(observation) is not QkGeometrySyncResult:
        raise QkGeometrySyncError("observation must be QkGeometrySyncResult")
    projections = []
    for layer in observation.layers:
        projection, _, public_digest, _, _ = _geometry_projection(
            detection_key,
            layer_name=layer.layer_name,
            token_count=layer.token_count,
            model_revision=observation.model_revision,
        )
        if public_digest != observation.root_key_public_digest:
            raise QkGeometrySyncError(
                "projection key does not belong to the registered key family"
            )
        projections.append(projection)
    return tuple(projections)


def _content_projection(
    vector: torch.Tensor,
    content_directions: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    flattened = vector.to(device="cpu", dtype=torch.float64).reshape(-1)
    if not bool(torch.isfinite(flattened).all()) or float(torch.linalg.vector_norm(flattened)) == 0:
        raise QkGeometrySyncError("geometry gradient must be finite and non-zero")
    if isinstance(content_directions, (str, bytes)) or not isinstance(
        content_directions, Sequence
    ):
        raise QkGeometrySyncError("content_directions must be an ordered sequence")
    if not content_directions:
        return torch.zeros_like(flattened), flattened
    columns = []
    for direction in content_directions:
        if not isinstance(direction, torch.Tensor) or direction.numel() != flattened.numel():
            raise QkGeometrySyncError("content direction shape must match geometry gradient")
        column = direction.to(device="cpu", dtype=torch.float64).reshape(-1)
        if not bool(torch.isfinite(column).all()):
            raise QkGeometrySyncError("content directions must be finite")
        norm = torch.linalg.vector_norm(column)
        if float(norm) == 0.0:
            raise QkGeometrySyncError("active content directions must be non-zero")
        columns.append(column / norm)
    matrix = torch.stack(columns, dim=1)
    left, singular_values, right_transpose = torch.linalg.svd(
        matrix,
        full_matrices=False,
    )
    sigma_max = singular_values.max()
    if float(sigma_max) <= 0.0:
        raise QkGeometrySyncError("content direction span is numerically empty")
    retained = singular_values > 1e-6 * sigma_max
    if not bool(retained.any()):
        raise QkGeometrySyncError("content direction span has no retained singular direction")
    left_retained = left[:, retained]
    right_retained = right_transpose[retained, :]
    reciprocal = torch.diag(1.0 / singular_values[retained])
    moore_penrose = (
        right_retained.transpose(0, 1)
        @ reciprocal
        @ left_retained.transpose(0, 1)
    )
    projected = matrix @ moore_penrose @ flattened
    return projected, flattened - projected


def geometry_direction_outside_content_span(
    geometry_gradient: torch.Tensor,
    content_directions: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Apply the frozen float64 Moore-Penrose content-span projection."""

    _, perpendicular = _content_projection(geometry_gradient, content_directions)
    gradient_norm = torch.linalg.vector_norm(
        geometry_gradient.to(device="cpu", dtype=torch.float64).reshape(-1)
    )
    perpendicular_norm = torch.linalg.vector_norm(perpendicular)
    if (
        not bool(torch.isfinite(perpendicular).all())
        or float(perpendicular_norm) <= 1e-12 * float(gradient_norm)
    ):
        raise QkGeometrySyncError("geometry gradient has no usable perpendicular direction")
    normalized = perpendicular / perpendicular_norm
    return normalized.to(dtype=torch.float32).reshape(geometry_gradient.shape)


def geometry_synchronization_write(
    baseline_latent: torch.Tensor,
    content_written_latent: torch.Tensor,
    geometry_gradient: torch.Tensor,
    content_directions: Sequence[torch.Tensor],
    *,
    geometry_ratio: float,
    baseline_score: float,
    materialize: Callable[[torch.Tensor], torch.Tensor],
    replay_score: Callable[[torch.Tensor], float],
) -> GeometrySynchronizationWriteResult:
    """Run the frozen actual-dtype geometry line search after content writing."""

    if geometry_ratio not in GEOMETRY_RATIOS:
        raise QkGeometrySyncError("geometry_ratio is outside the frozen finite set")
    if not isfinite(baseline_score):
        raise QkGeometrySyncError("baseline_score must be finite")
    if not isinstance(baseline_latent, torch.Tensor) or not isinstance(
        content_written_latent, torch.Tensor
    ):
        raise QkGeometrySyncError("baseline and content-written latents must be tensors")
    if baseline_latent.shape != content_written_latent.shape:
        raise QkGeometrySyncError("baseline and content-written latent shapes must match")
    baseline_float = baseline_latent.to(device="cpu", dtype=torch.float32)
    content_float = content_written_latent.to(device="cpu", dtype=torch.float32)
    if not bool(torch.isfinite(baseline_float).all()) or not bool(
        torch.isfinite(content_float).all()
    ):
        raise QkGeometrySyncError("latent inputs must be finite")
    baseline_norm = torch.linalg.vector_norm(baseline_float)
    if float(baseline_norm) == 0.0:
        raise QkGeometrySyncError("baseline latent must have non-zero L2 norm")

    direction = geometry_direction_outside_content_span(
        geometry_gradient,
        content_directions,
    )
    rho_geometry = CONTENT_RELATIVE_L2 * geometry_ratio
    total_limit = CONTENT_RELATIVE_L2 * sqrt(1.0 + geometry_ratio**2)
    full_update = rho_geometry * baseline_norm * direction
    baseline_actual = materialize(baseline_latent).to(
        device="cpu", dtype=torch.float32
    )
    if baseline_actual.shape != baseline_float.shape:
        raise QkGeometrySyncError("materialize changed the latent shape")

    for factor in LINE_SEARCH_FACTORS:
        candidate = materialize(
            content_written_latent + factor * full_update.to(content_written_latent)
        )
        if not isinstance(candidate, torch.Tensor) or candidate.shape != baseline_latent.shape:
            raise QkGeometrySyncError("materialize must return the same tensor shape")
        candidate_float = candidate.to(device="cpu", dtype=torch.float32)
        if not bool(torch.isfinite(candidate_float).all()):
            continue
        geometry_actual = candidate_float - content_float
        total_actual = candidate_float - baseline_actual
        geometry_norm = torch.linalg.vector_norm(geometry_actual)
        total_norm = torch.linalg.vector_norm(total_actual)
        if float(geometry_norm) == 0.0:
            continue
        projected, _ = _content_projection(geometry_actual, content_directions)
        projected_norm = torch.linalg.vector_norm(projected)
        score = replay_score(candidate)
        if not isfinite(score):
            continue
        geometry_limit = factor * rho_geometry * float(baseline_norm)
        projection_relative = (
            float(projected_norm / geometry_norm)
            if float(geometry_norm) > 0.0
            else float("inf")
        )
        if (
            score > baseline_score
            and 0.0 < float(geometry_norm) <= geometry_limit + 1e-12
            and float(total_norm) <= total_limit * float(baseline_norm) + 1e-12
            and projection_relative <= 1e-4
        ):
            return GeometrySynchronizationWriteResult(
                accepted=True,
                status="accepted",
                geometry_ratio=geometry_ratio,
                line_search_factor=factor,
                baseline_score=baseline_score,
                accepted_score=float(score),
                geometry_relative_l2_actual=float(geometry_norm / baseline_norm),
                total_relative_l2_actual=float(total_norm / baseline_norm),
                content_projection_relative=projection_relative,
                written_latent=candidate,
            )
    return GeometrySynchronizationWriteResult(
        accepted=False,
        status="geometry_synchronization_failed",
        geometry_ratio=geometry_ratio,
        line_search_factor=None,
        baseline_score=baseline_score,
        accepted_score=None,
        geometry_relative_l2_actual=None,
        total_relative_l2_actual=None,
        content_projection_relative=None,
        written_latent=None,
    )


__all__ = [
    "differentiable_qk_relation_objective",
    "GeometrySynchronizationWriteResult",
    "QkGeometrySyncError",
    "QkGeometrySyncResult",
    "QkLayerObservation",
    "QkLayerRelation",
    "geometry_direction_outside_content_span",
    "geometry_synchronization_write",
    "projection_for_detection_key",
    "qk_geometry_sync",
    "qk_relation_tensor",
    "row_normalized_relation_score",
    "validate_qk_geometry_sync_result",
]
