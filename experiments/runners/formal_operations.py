"""Formal public-image operations consumed by the governed experiment runner."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import json
from math import isfinite

import torch

from experiments.methods import (
    CegWmExperimentAdapter,
    CegWmExperimentAdapterConfiguration,
)
from main import (
    ContentDetectionResult,
    ContentDetectorBinding,
    ContentDetectorError,
    GeometricTransformEstimation,
    HfDetectionObservation,
    rgb8_image_digest,
    validate_content_detection_result,
)
from runtime import RuntimeAdapterState, Sd35RuntimeAdapter


CONTENT_OPERATION_SEMANTIC_VERSION = (
    "ceg_wm_formal_hf_content_detection_operation_v1"
)
GEOMETRY_OPERATION_SEMANTIC_VERSION = (
    "ceg_wm_formal_runtime_geometry_estimation_operation_v1"
)
PUBLIC_IMAGE_ENCODING = "rgb8_public_image_float32_unit_interval"
FORMAL_CONTENT_MODE = "hf_only"


class FormalOperationError(ValueError):
    """A formal experiment operation or its declared semantics drifted."""


def _canonical_primitive(value: object) -> object:
    return json.loads(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )


@dataclass(frozen=True, slots=True)
class FormalHfContentDetectionOperation:
    """Run the implemented HF detector on an ordinary RGB8 public image."""

    adapter: CegWmExperimentAdapter
    preprocessing_identity: str = PUBLIC_IMAGE_ENCODING

    def __post_init__(self) -> None:
        if type(self.adapter) is not CegWmExperimentAdapter:
            raise FormalOperationError(
                "content operation requires CegWmExperimentAdapter"
            )
        if (
            type(self.preprocessing_identity) is not str
            or not self.preprocessing_identity
        ):
            raise FormalOperationError(
                "content preprocessing identity is required"
            )

    def formal_runner_semantic_declaration(self) -> dict[str, object]:
        """Declare every configurable input that can change this operation."""

        return self._semantic_declaration()

    def __call__(
        self,
        image: torch.Tensor,
        detection_key: str,
    ) -> ContentDetectionResult:
        return self._detect(image, detection_key)

    def replay_validate_content_result(
        self,
        result: ContentDetectionResult,
        input_image: object,
        detection_key: str,
    ) -> ContentDetectionResult:
        if not isinstance(input_image, torch.Tensor):
            raise ContentDetectorError(
                "replay input must be an RGB8 tensor"
            )
        expected = self._detect(input_image, detection_key)
        validate_content_detection_result(expected)
        if result != expected:
            raise ContentDetectorError("content replay mismatch")
        return result

    def _detect(
        self,
        image: torch.Tensor,
        detection_key: str,
    ) -> ContentDetectionResult:
        observation = HfDetectionObservation.from_public_image_encoding(
            tuple(
                (image.to(dtype=torch.float32) / 255.0)
                .reshape(-1)
                .tolist()
            ),
            tuple(image.shape),
        )
        hf_result = self.adapter.detect_hf(
            observation,
            detection_key,
        ).result
        content_result = self.adapter.detect_content(hf_result).result
        return replace(
            content_result,
            content_input_image_digest=rgb8_image_digest(image),
            content_replay_operation=self,
        )

    def _semantic_declaration(self) -> dict[str, object]:
        configuration = self.adapter.configuration
        if type(configuration) is not CegWmExperimentAdapterConfiguration:
            raise FormalOperationError(
                "content adapter configuration exact type is required"
            )
        return {
            "adapter_configuration": _canonical_primitive(
                asdict(configuration)
            ),
            "adapter_config_digest": configuration.config_digest,
            "content_detector_public_callable": "main.content_detector",
            "formal_mode": FORMAL_CONTENT_MODE,
            "hf_detector_public_callable": "main.hf_detector",
            "image_encoding": self.preprocessing_identity,
            "pixel_conversion": "uint8_to_float32_divide_255",
            "semantic_version": CONTENT_OPERATION_SEMANTIC_VERSION,
        }

def create_formal_content_detector_binding(
    operation: FormalHfContentDetectionOperation,
    *,
    prototype_image: torch.Tensor,
    detection_key: str,
) -> tuple[ContentDetectorBinding, float]:
    """Construct one binding from an actual call through the formal operation."""

    if type(operation) is not FormalHfContentDetectionOperation:
        raise FormalOperationError(
            "formal content operation exact type is required"
        )
    prototype = operation(prototype_image, detection_key)
    binding = ContentDetectorBinding(
        content_detection_operation=operation,
        detector_identity=prototype.detector_identity,
        content_config_digest=prototype.content_config_digest,
        hf_detector_identity=prototype.hf_result.detector_identity,
        hf_detector_config_digest=(
            prototype.hf_result.detector_config_digest
        ),
        hf_template_digest=prototype.hf_result.template_digest,
        preprocessing_identity=operation.preprocessing_identity,
        formal_mode=prototype.formal_mode,
        root_key_public_digest=(
            prototype.hf_result.root_key_public_digest
        ),
        key_role=prototype.hf_result.key_role,
        wrong_key_index=prototype.hf_result.wrong_key_index,
    )
    return binding, prototype.content_score


@dataclass(frozen=True, slots=True)
class FormalRuntimeGeometryEstimationOperation:
    """Connect public-image runtime Q/K observation to the real estimator."""

    runtime_adapter: Sd35RuntimeAdapter
    adapter_configuration: CegWmExperimentAdapterConfiguration
    epsilon_inlier: float
    execution_scope: str
    _method_adapter: CegWmExperimentAdapter = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if type(self.runtime_adapter) is not Sd35RuntimeAdapter:
            raise FormalOperationError(
                "geometry operation requires Sd35RuntimeAdapter"
            )
        if (
            type(self.adapter_configuration)
            is not CegWmExperimentAdapterConfiguration
        ):
            raise FormalOperationError(
                "geometry adapter configuration exact type is required"
            )
        if (
            isinstance(self.epsilon_inlier, bool)
            or not isinstance(self.epsilon_inlier, (int, float))
            or not isfinite(float(self.epsilon_inlier))
            or float(self.epsilon_inlier) <= 0.0
        ):
            raise FormalOperationError(
                "geometry epsilon_inlier must be finite and positive"
            )
        if (
            type(self.execution_scope) is not str
            or not self.execution_scope
        ):
            raise FormalOperationError(
                "geometry execution scope is required"
            )
        object.__setattr__(self, "epsilon_inlier", float(self.epsilon_inlier))
        object.__setattr__(
            self,
            "_method_adapter",
            CegWmExperimentAdapter(self.adapter_configuration),
        )

    def formal_runner_semantic_declaration(self) -> dict[str, object]:
        """Declare method, runtime, estimator, and current session identity."""

        return self._semantic_declaration()

    def __call__(
        self,
        image: torch.Tensor,
        registered_key: str,
    ) -> GeometricTransformEstimation:
        if self.runtime_adapter.state is not RuntimeAdapterState.READY:
            raise FormalOperationError(
                "geometry runtime adapter must be ready before execution"
            )
        self._method_adapter.require_no_runtime_binding()
        runtime_observation = self.runtime_adapter.observe_detection_qk(image)
        observation = self._method_adapter.synchronize_qk_observation(
            runtime_observation,
            registered_key,
        ).result
        estimation = self._method_adapter.estimate_geometric_transform(
            observation,
            registered_key,
            epsilon_inlier=self.epsilon_inlier,
        ).result
        self._method_adapter.require_no_runtime_binding()
        return estimation

    def _semantic_declaration(self) -> dict[str, object]:
        runtime_configuration = self.runtime_adapter.configuration
        self._method_adapter.require_no_runtime_binding()
        if self.runtime_adapter.state is RuntimeAdapterState.READY:
            runtime_session: object | None = _canonical_primitive(
                asdict(self.runtime_adapter.session)
            )
        else:
            runtime_session = None
        return {
            "adapter_configuration": _canonical_primitive(
                asdict(self.adapter_configuration)
            ),
            "adapter_config_digest": (
                self.adapter_configuration.config_digest
            ),
            "epsilon_inlier": self.epsilon_inlier,
            "estimator_public_callable": (
                "main.geometric_transform_estimator"
            ),
            "execution_scope": self.execution_scope,
            "qk_public_callable": (
                "runtime.Sd35RuntimeAdapter.observe_detection_qk"
                " -> main.qk_geometry_sync"
            ),
            "runtime_candidate_id": runtime_configuration.candidate_id,
            "runtime_config_digest": (
                runtime_configuration.runtime_config_digest
            ),
            "runtime_configuration": _canonical_primitive(
                runtime_configuration.identity_mapping()
            ),
            "runtime_model_id": runtime_configuration.model_id,
            "runtime_model_revision": runtime_configuration.model_revision,
            "runtime_qk_layer_names": list(
                runtime_configuration.qk_layer_names
            ),
            "runtime_session": runtime_session,
            "runtime_state": self.runtime_adapter.state.value,
            "semantic_version": GEOMETRY_OPERATION_SEMANTIC_VERSION,
        }
