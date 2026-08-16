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
PACKAGE_IDENTITY = "2" * 64


class _PublicAdapter:
    def __init__(
        self,
        write_error: BaseException | None = None,
        *,
        public_result_identity: str = "4" * 64,
        witness_identity: str = "3" * 64,
    ) -> None:
        self.write_error = write_error
        self.public_result_identity = public_result_identity
        self.witness_identity = witness_identity
        self.write_calls = 0
        self.detector_calls = 0

    def execute_semantic_texture_content_write_and_vae(
        self,
        base_latent: object,
        detection_key: str,
        semantic_runtime: object,
    ) -> object:
        self.write_calls += 1
        assert base_latent is _BASE_LATENT
        assert detection_key == "memory-only-detection-key"
        assert semantic_runtime is _SEMANTIC_RUNTIME
        if self.write_error is not None:
            raise self.write_error
        witness = SimpleNamespace(witness_identity=self.witness_identity)
        return SimpleNamespace(
            result_identity=self.public_result_identity,
            result=SimpleNamespace(witness=witness),
        )

    def detect_semantic_texture_candidate(self, *args, **kwargs):
        self.detector_calls += 1
        raise AssertionError("Phase A must block before public detector call")


_BASE_LATENT = object()
_SEMANTIC_RUNTIME = object()


def _execute(adapter: _PublicAdapter):
    configuration = preflight.load_semantic_texture_operational_configuration(
        CONFIG_PATH
    )
    ticks = iter((10.0, 10.5, 11.0, 11.25))
    return preflight.execute_semantic_texture_operational_preflight(
        adapter,
        configuration,
        source_revision=REVISION,
        run_id="semantic-texture-phase-a",
        package_identity=PACKAGE_IDENTITY,
        base_latent=_BASE_LATENT,
        detection_key="memory-only-detection-key",
        semantic_runtime=_SEMANTIC_RUNTIME,
        monotonic_clock=lambda: next(ticks),
    )


def test_semantic_texture_operational_preflight_runs_exact_two_units_without_science() -> None:
    adapter = _PublicAdapter()
    result = _execute(adapter)
    assert tuple(item.unit_id for item in result.unit_outcomes) == preflight.UNIT_ROSTER
    assert result.unit_outcomes[0].status == "passed"
    assert result.unit_outcomes[0].witness_identity == "3" * 64
    assert result.unit_outcomes[1].started is True
    assert result.unit_outcomes[1].blocked_class == "identity_blocked"
    assert result.status == "blocked"
    assert result.aggregate is None
    assert result.scientific_unit_count == 0
    assert result.science_started is False
    assert result.formal_tau_created is False
    assert result.candidate_promoted is False
    assert result.scientific_claims_supported is False
    assert adapter.write_calls == 1
    assert adapter.detector_calls == 0


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
    }
    source = Path(preflight.__file__).read_text(encoding="utf-8")
    assert "Sd35RuntimeAdapter" not in source
    assert "runtime.content_write" not in source
    assert "from main" not in source
    adapter = _PublicAdapter()
    _execute(adapter)
    assert (adapter.write_calls, adapter.detector_calls) == (1, 0)


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
    assert adapter.detector_calls == 0
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
    assert configuration.asset_authority_status == "identity_blocked"
    raw = CONFIG_PATH.read_text(encoding="utf-8").rstrip()
    mutated = raw[:-1] + ',\n  "whitening_asset_override": "forbidden"\n}\n'
    path = tmp_path / "configuration.json"
    path.write_text(mutated, encoding="utf-8")
    with pytest.raises(
        preflight.SemanticTextureOperationalPreflightError,
        match="fields drifted",
    ):
        preflight.load_semantic_texture_operational_configuration(path)
    adapter = _PublicAdapter()
    with pytest.raises(TypeError):
        preflight.execute_semantic_texture_operational_preflight(
            adapter,
            configuration,
            source_revision=REVISION,
            run_id="semantic-texture-phase-a",
            package_identity=PACKAGE_IDENTITY,
            base_latent=_BASE_LATENT,
            detection_key="memory-only-detection-key",
            semantic_runtime=_SEMANTIC_RUNTIME,
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
    assert result.unit_outcomes[1].blocked_class == "identity_blocked"
    assert adapter.detector_calls == 0
