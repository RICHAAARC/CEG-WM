"""CPU-only tests for the two-unit semantic-texture operational boundary."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.runners import semantic_texture_operational_preflight as preflight


pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    ROOT / "configs/experiments/semantic_texture_operational_preflight.json"
)
REVISION = "1" * 40


class _PublicAdapter:
    def __init__(
        self,
        write_error: BaseException | None = None,
        *,
        public_result_identity: str = "4" * 64,
        witness_identity: str = "3" * 64,
        detector_error: BaseException | None = None,
        detector_result_identity: str = "5" * 64,
    ) -> None:
        self.write_error = write_error
        self.public_result_identity = public_result_identity
        self.witness_identity = witness_identity
        self.detector_error = detector_error
        self.detector_result_identity = detector_result_identity
        self.write_calls = 0
        self.rgb8_calls = 0
        self.detector_calls = 0
        self.call_sequence: list[str] = []

    def execute_semantic_texture_content_write_and_vae(
        self,
        base_latent: object,
        detection_key: str,
        semantic_runtime: object,
    ) -> object:
        self.write_calls += 1
        self.call_sequence.append("write")
        assert base_latent is _BASE_LATENT
        assert detection_key == "memory-only-detection-key"
        assert semantic_runtime is _SEMANTIC_RUNTIME
        if self.write_error is not None:
            raise self.write_error
        witness = SimpleNamespace(witness_identity=self.witness_identity)
        self.write_observation = SimpleNamespace(
            result_identity=self.public_result_identity,
            result=SimpleNamespace(witness=witness),
        )
        return self.write_observation

    def materialize_semantic_texture_written_rgb8(
        self, write_observation: object
    ) -> object:
        self.rgb8_calls += 1
        self.call_sequence.append("rgb8")
        assert write_observation is self.write_observation
        return _DETECTION_IMAGE_RGB8

    def detect_semantic_texture_candidate(
        self,
        detection_image_rgb8: object,
        detection_key: str,
        semantic_runtime: object,
        whitening_asset: object,
        *,
        hf_null: object,
        lf_null: object,
    ) -> object:
        self.detector_calls += 1
        self.call_sequence.append("detector")
        assert detection_image_rgb8 is _DETECTION_IMAGE_RGB8
        assert detection_key == "memory-only-detection-key"
        assert semantic_runtime is _SEMANTIC_RUNTIME
        assert whitening_asset is _WHITENING_ASSET
        assert hf_null is _HF_NULL
        assert lf_null is _LF_NULL
        if self.detector_error is not None:
            raise self.detector_error
        return SimpleNamespace(result_identity=self.detector_result_identity)


_BASE_LATENT = object()
_SEMANTIC_RUNTIME = object()
_DETECTION_IMAGE_RGB8 = object()
_WHITENING_ASSET = object()
_HF_NULL = object()
_LF_NULL = object()


def _execute(
    adapter: _PublicAdapter,
    configuration: preflight.SemanticTextureOperationalConfiguration | None = None,
):
    if configuration is None:
        configuration = preflight.load_semantic_texture_operational_configuration(
            CONFIG_PATH
        )
    ticks = iter((10.0, 10.5, 11.0, 11.25))
    return preflight.execute_semantic_texture_operational_preflight(
        adapter,
        configuration,
        observed_repository_revision=REVISION,
        run_id="semantic-texture-phase-a",
        base_latent=_BASE_LATENT,
        detection_key="memory-only-detection-key",
        semantic_runtime=_SEMANTIC_RUNTIME,
        whitening_asset=_WHITENING_ASSET,
        hf_null=_HF_NULL,
        lf_null=_LF_NULL,
        monotonic_clock=lambda: next(ticks),
    )


def test_semantic_texture_operational_preflight_runs_exact_two_units_without_science() -> None:
    adapter = _PublicAdapter()
    configuration = preflight.load_semantic_texture_operational_configuration(
        CONFIG_PATH
    )
    result = _execute(adapter, configuration)
    assert tuple(item.unit_id for item in result.unit_outcomes) == preflight.UNIT_ROSTER
    write, detector = result.unit_outcomes
    assert write.started is True and write.status == "passed"
    assert write.witness_identity == "3" * 64
    assert detector.started is True and detector.status == "passed"
    assert detector.public_result_identity == "5" * 64
    assert detector.witness_identity is None
    assert result.status == "passed"
    assert result.blocked_class is None
    assert result.diagnostic_only is True
    assert result.asset_authority_status == "diagnostic_bundle_authenticated"
    assert result.asset_bundle_digest == (
        "f9dd6df410cb4f7895376c65c5f6d3e764f6cfddabd0d64d525fdaaefd93de3d"
    )
    assert configuration.schema_version == 4
    assert configuration.detector_asset_bundle_sha256 == (
        "126f73150584d5c5a1e5b5e2dbffa9bb0379a9375c202ab49a87b56f99c41ea7"
    )
    assert configuration.detector_asset_bundle_relative_path == (
        "semantic_texture_soft_detector_assets/"
        "f9dd6df410cb4f7895376c65c5f6d3e764f6cfddabd0d64d525fdaaefd93de3d/"
        "semantic_texture_soft_detector_asset_bundle.json"
    )
    assert result.aggregate is None
    assert result.scientific_unit_count == 0
    assert result.science_started is False
    assert result.formal_tau_created is False
    assert result.candidate_promoted is False
    assert result.scientific_claims_supported is False
    assert adapter.call_sequence == ["write", "rgb8", "detector"]
    assert (adapter.write_calls, adapter.rgb8_calls, adapter.detector_calls) == (1, 1, 1)


def test_semantic_texture_operational_preflight_uses_public_adapter_only() -> None:
    tree = ast.parse(Path(preflight.__file__).read_text(encoding="utf-8"))
    called_adapter_attributes = {
        call.func.attr
        for call in ast.walk(tree)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "adapter"
    }
    assert called_adapter_attributes == {
        "detect_semantic_texture_candidate",
        "execute_semantic_texture_content_write_and_vae",
        "materialize_semantic_texture_written_rgb8",
    }
    source = Path(preflight.__file__).read_text(encoding="utf-8")
    assert "Sd35RuntimeAdapter" not in source
    assert "runtime.content_write" not in source
    assert "from main" not in source
    assert "semantic_texture_lf_detector" not in source
    assert "importlib" not in source
    adapter = _PublicAdapter()
    _execute(adapter)
    assert adapter.call_sequence == ["write", "rgb8", "detector"]
    assert (adapter.write_calls, adapter.rgb8_calls, adapter.detector_calls) == (1, 1, 1)


@pytest.mark.parametrize(
    "blocked_class",
    sorted(preflight.ALLOWED_BLOCKED_CLASSES),
)
def test_semantic_texture_operational_preflight_faults_are_blocked_and_aggregate_null(
    blocked_class: str,
) -> None:
    private_error_text = (
        "Drive credential token=secret-token account=private-account "
        "root_key=private-root-key prompt=private-prompt "
        "private_state=private-state /home/private/checkpoint-secret.bin"
    )
    chained_error = RuntimeError(f"chained failure: {private_error_text}")
    blocked_error = preflight.SemanticTextureOperationalBlockedError(
        blocked_class,
        private_error_text,
    )
    blocked_error.__cause__ = chained_error
    adapter = _PublicAdapter(
        blocked_error
    )
    result = _execute(adapter)
    write, detector = result.unit_outcomes
    assert write.started is True and write.status == "blocked"
    assert write.blocked_class == blocked_class
    assert write.sanitized_error_category == blocked_class
    assert write.sanitized_error_message is None
    assert write.sanitized_trace_tail == ()
    assert detector.started is False
    assert detector.blocked_class == blocked_class
    assert detector.sanitized_error_category == blocked_class
    assert detector.sanitized_error_message is None
    assert detector.sanitized_trace_tail == ()
    assert result.blocked_class == blocked_class
    assert result.aggregate is None
    assert result.science_started is False
    assert result.scientific_unit_count == 0
    assert result.diagnostic_only is True
    assert result.status == "blocked"
    assert adapter.detector_calls == 0
    assert adapter.rgb8_calls == 0
    persisted_result = json.dumps(result.as_dict(), sort_keys=True)
    for private_fragment in (
        private_error_text,
        "chained failure",
        "secret-token",
        "private-account",
        "private-root-key",
        "private-prompt",
        "private-state",
        "/home/private/checkpoint-secret.bin",
    ):
        assert private_fragment not in persisted_result


def test_semantic_texture_preflight_rejects_asset_override_private_state_and_live_detector_call(
    tmp_path: Path,
) -> None:
    configuration = preflight.load_semantic_texture_operational_configuration(
        CONFIG_PATH
    )
    assert configuration.schema_version == 4
    assert configuration.asset_authority_status == "diagnostic_bundle_authenticated"
    assert configuration.detector_asset_bundle_digest == (
        "f9dd6df410cb4f7895376c65c5f6d3e764f6cfddabd0d64d525fdaaefd93de3d"
    )
    assert configuration.detector_asset_bundle_sha256 == (
        "126f73150584d5c5a1e5b5e2dbffa9bb0379a9375c202ab49a87b56f99c41ea7"
    )
    assert configuration.detector_asset_bundle_relative_path == (
        "semantic_texture_soft_detector_assets/"
        "f9dd6df410cb4f7895376c65c5f6d3e764f6cfddabd0d64d525fdaaefd93de3d/"
        "semantic_texture_soft_detector_asset_bundle.json"
    )
    raw = CONFIG_PATH.read_text(encoding="utf-8").rstrip()
    mutated = raw[:-1] + ',\n  "whitening_asset_override": "forbidden"\n}\n'
    path = tmp_path / "configuration.json"
    path.write_text(mutated, encoding="utf-8")
    with pytest.raises(
        preflight.SemanticTextureOperationalPreflightError,
        match="fields drifted",
    ):
        preflight.load_semantic_texture_operational_configuration(path)
    observed_checkpoint = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    observed_checkpoint["inspyrenet_checkpoint_filename"] = "observed-input.bin"
    observed_path = tmp_path / "observed-configuration.json"
    observed_path.write_text(
        json.dumps(observed_checkpoint),
        encoding="utf-8",
    )
    observed_configuration = (
        preflight.load_semantic_texture_operational_configuration(observed_path)
    )
    assert observed_configuration.inspyrenet_checkpoint_filename == (
        "observed-input.bin"
    )
    assert observed_configuration.configuration_digest == (
        configuration.configuration_digest
    )
    assert observed_configuration == configuration
    observed_requirements = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    observed_requirements["requirements_lock_sha256"] = "5" * 64
    observed_requirements_path = tmp_path / "observed-requirements.json"
    observed_requirements_path.write_text(
        json.dumps(observed_requirements),
        encoding="utf-8",
    )
    observed_requirements_configuration = (
        preflight.load_semantic_texture_operational_configuration(
            observed_requirements_path
        )
    )
    assert observed_requirements_configuration.requirements_lock_sha256 == "5" * 64
    assert observed_requirements_configuration == configuration
    assert observed_requirements_configuration.configuration_digest == (
        configuration.configuration_digest
    )
    baseline_result = _execute(_PublicAdapter(), configuration)
    observed_requirements_result = _execute(
        _PublicAdapter(),
        observed_requirements_configuration,
    )
    assert observed_requirements_result.result_identity == baseline_result.result_identity
    adapter = _PublicAdapter()
    with pytest.raises(TypeError):
        preflight.execute_semantic_texture_operational_preflight(
            adapter,
            configuration,
            observed_repository_revision=REVISION,
            run_id="semantic-texture-phase-a",
            base_latent=_BASE_LATENT,
            detection_key="memory-only-detection-key",
            semantic_runtime=_SEMANTIC_RUNTIME,
            whitening_asset=_WHITENING_ASSET,
            hf_null=_HF_NULL,
            lf_null=_LF_NULL,
            whitening_asset_override=object(),
        )
    invalid_identity_adapters = (
        _PublicAdapter(public_result_identity="A" * 64),
        _PublicAdapter(witness_identity="3" * 63),
    )
    for invalid_identity_adapter in invalid_identity_adapters:
        invalid_result = _execute(invalid_identity_adapter)
        invalid_write, invalid_detector = invalid_result.unit_outcomes
        assert invalid_write.blocked_class == "integrity_blocked"
        assert invalid_write.public_result_identity is None
        assert invalid_write.witness_identity is None
        assert invalid_detector.started is False
        assert invalid_detector.blocked_class == "integrity_blocked"
        assert invalid_identity_adapter.detector_calls == 0
    result = _execute(adapter)
    write, detector = result.unit_outcomes
    assert write.status == "passed"
    assert detector.status == "passed"
    assert result.status == "passed"
    assert adapter.detector_calls == 1
    private_error_text = "detector token=private-token /home/private/detector.bin"
    blocked_result = _execute(
        _PublicAdapter(
            detector_error=preflight.SemanticTextureOperationalBlockedError(
                "implementation_blocked", private_error_text
            )
        )
    )
    blocked_write, blocked_detector = blocked_result.unit_outcomes
    assert blocked_write.status == "passed"
    assert blocked_detector.started is True and blocked_detector.status == "blocked"
    assert blocked_detector.blocked_class == "implementation_blocked"
    assert blocked_detector.sanitized_error_category == "implementation_blocked"
    assert blocked_detector.sanitized_error_message is None
    assert blocked_detector.sanitized_trace_tail == ()
    assert blocked_result.status == "blocked"
    assert blocked_result.blocked_class == "implementation_blocked"
    persisted_blocked_result = json.dumps(
        blocked_result.as_dict(), sort_keys=True
    )
    for private_fragment in (
        private_error_text,
        "private-token",
        "/home/private/detector.bin",
        "SemanticTextureOperationalBlockedError",
        "detector token",
    ):
        assert private_fragment not in persisted_blocked_result
