from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from runtime import (
    INSPYRENET_CHECKPOINT_ASSET_BASENAME,
    INSPYRENET_CHECKPOINT_ASSET_IDENTITY,
    INSPYRENET_CHECKPOINT_SHA256,
    INSPYRENET_CHECKPOINT_SIZE,
    InspyrenetSaliencyRuntime,
    InspyrenetSaliencyRuntimeError,
    RuntimeAdapterError,
    RuntimeAdapterState,
    RuntimeBackendIdentity,
    RuntimeConfigurationError,
    RuntimeDeviceCapabilities,
    RuntimeExecutionIdentity,
    create_runtime_adapter,
    load_runtime_configuration,
    parse_runtime_configuration,
    select_runtime_device,
)


@pytest.mark.unit
def test_public_runtime_surface_exposes_frozen_inspyrenet_saliency_owner() -> None:
    assert InspyrenetSaliencyRuntime.__module__ == "runtime.inspyrenet_saliency"
    assert InspyrenetSaliencyRuntimeError.__module__ == "runtime.inspyrenet_saliency"
    assert INSPYRENET_CHECKPOINT_ASSET_IDENTITY == "inspyrenet_saliency_checkpoint"
    assert INSPYRENET_CHECKPOINT_ASSET_BASENAME == "ckpt_base.pth"
    assert INSPYRENET_CHECKPOINT_SIZE == 367_520_613
    assert (
        INSPYRENET_CHECKPOINT_SHA256
        == "0a6fe2a73ab0532d6d0b8d82849a9760a226df719e3063d09b4149ece6f80fcd"
    )


def _config_mapping() -> dict[str, object]:
    path = Path("configs/runtime/runtime_sd35_flowmatch.json")
    return json.loads(path.read_text(encoding="utf-8"))


class MockBackend:
    def __init__(
        self,
        capabilities: RuntimeDeviceCapabilities,
        *,
        drift_field: str | None = None,
        drift_value: object = "drifted",
        fail_prepare: bool = False,
        reject_repeated_close: bool = False,
    ) -> None:
        self.capabilities = capabilities
        self.drift_field = drift_field
        self.drift_value = drift_value
        self.fail_prepare = fail_prepare
        self.reject_repeated_close = reject_repeated_close
        self.prepare_calls: list[tuple[str, str]] = []
        self.close_calls = 0

    def probe_devices(self) -> RuntimeDeviceCapabilities:
        return self.capabilities

    def prepare(self, configuration, selected_device):
        self.prepare_calls.append(
            (configuration.runtime_config_digest, selected_device)
        )
        if self.fail_prepare:
            raise RuntimeError("mock prepare failure")
        values = {
            "candidate_id": configuration.candidate_id,
            "runtime_config_digest": configuration.runtime_config_digest,
            "runtime_backend_name": "synthetic_sd35_backend",
            "selected_device": selected_device,
            "model_id": configuration.model_id,
            "model_revision": configuration.model_revision,
            "pipeline_class": configuration.pipeline_class,
            "scheduler_class": configuration.scheduler_class,
            "inference_steps": configuration.inference_steps,
            "guidance_scale": configuration.guidance_scale,
            "image_height": configuration.image_height,
            "image_width": configuration.image_width,
            "generation_seed_device": configuration.generation_seed_device,
            "latent_dtype": configuration.latent_dtype,
            "template_dtype": configuration.template_dtype,
            "score_dtype": configuration.score_dtype,
            "callback_index": configuration.callback_index,
            "callback_hold_scheduler_intervals": (
                configuration.callback_hold_scheduler_intervals
            ),
            "vae_decode_protocol": configuration.vae_decode_protocol,
            "vae_encode_protocol": configuration.vae_encode_protocol,
            "vae_scaling_factor_source": (
                configuration.vae_scaling_factor_source
            ),
            "vae_shift_factor_source": configuration.vae_shift_factor_source,
            "detection_schedule_index": configuration.detection_schedule_index,
            "detection_conditioning_protocol": (
                configuration.detection_conditioning_protocol
            ),
            "qk_layer_names": configuration.qk_layer_names,
            "dependency_lock": configuration.dependency_lock,
        }
        if self.drift_field is not None:
            values[self.drift_field] = self.drift_value
        return RuntimeBackendIdentity(**values)

    def close(self) -> None:
        if self.reject_repeated_close and self.close_calls:
            raise RuntimeError("mock backend close is not idempotent")
        self.close_calls += 1


@pytest.mark.unit
def test_runtime_configuration_freezes_sd35_identity() -> None:
    configuration = load_runtime_configuration()

    assert configuration.candidate_id == "runtime_sd35_flowmatch"
    assert configuration.model_id == "stabilityai/stable-diffusion-3.5-medium"
    assert (
        configuration.model_revision
        == "b940f670f0eda2d07fbb75229e779da1ad11eb80"
    )
    assert configuration.pipeline_class == "diffusers.StableDiffusion3Pipeline"
    assert (
        configuration.scheduler_class
        == "diffusers.FlowMatchEulerDiscreteScheduler"
    )
    assert (configuration.image_height, configuration.image_width) == (512, 512)
    assert configuration.inference_steps == 20
    assert configuration.guidance_scale == 4.5
    assert configuration.callback_index == 18
    assert configuration.callback_hold_scheduler_intervals == 1
    assert configuration.latent_dtype == "float16"
    assert configuration.template_dtype == configuration.score_dtype == "float32"
    assert configuration.detection_schedule_index == 7
    assert configuration.qk_layer_names == (
        "transformer_blocks.0.attn",
        "transformer_blocks.23.attn",
    )
    assert configuration.dependency_lock.diffusers == "0.38.0"
    assert configuration.dependency_lock.torch == "2.11.0"
    assert len(configuration.runtime_config_digest) == 64


@pytest.mark.unit
def test_runtime_configuration_digest_is_order_independent() -> None:
    source = _config_mapping()
    reordered = {key: source[key] for key in reversed(tuple(source))}

    first = parse_runtime_configuration(source)
    second = parse_runtime_configuration(reordered)

    assert first == second
    assert first.runtime_config_digest == second.runtime_config_digest


@pytest.mark.unit
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_revision", "main"),
        ("pipeline_class", "diffusers.AutoPipelineForText2Image"),
        ("scheduler_class", "diffusers.DDIMScheduler"),
        ("inference_steps", 21),
        ("guidance_scale", 0.0),
        ("image_height", 1024),
        ("latent_dtype", "float32"),
        ("callback_index", 17),
        ("detection_schedule_index", 8),
    ],
)
def test_runtime_configuration_rejects_candidate_drift(
    field: str,
    value: object,
) -> None:
    source = _config_mapping()
    source[field] = value

    with pytest.raises(RuntimeConfigurationError, match=field):
        parse_runtime_configuration(source)


@pytest.mark.unit
def test_runtime_configuration_rejects_extra_fields_and_dependency_drift() -> None:
    with_extra = _config_mapping()
    with_extra["fallback_model_id"] = "forbidden"
    with pytest.raises(RuntimeConfigurationError, match="extra"):
        parse_runtime_configuration(with_extra)

    dependency_drift = deepcopy(_config_mapping())
    dependency_lock = dependency_drift["dependency_lock"]
    assert isinstance(dependency_lock, list)
    diffusers_entry = dependency_lock[1]
    assert isinstance(diffusers_entry, dict)
    diffusers_entry["version_specifier"] = "0.39.0"
    with pytest.raises(RuntimeConfigurationError, match="dependency_lock"):
        parse_runtime_configuration(dependency_drift)


@pytest.mark.unit
def test_device_selection_is_deterministic_and_fail_closed() -> None:
    cpu_only = RuntimeDeviceCapabilities(
        cpu_available=True,
        cuda_device_count=0,
    )
    cuda = RuntimeDeviceCapabilities(
        cpu_available=True,
        cuda_device_count=2,
    )

    assert select_runtime_device(cpu_only, "auto") == "cpu"
    assert select_runtime_device(cuda, "auto") == "cuda:0"
    assert select_runtime_device(cuda, "cpu") == "cpu"
    assert select_runtime_device(cuda, "cuda") == "cuda:0"
    with pytest.raises(RuntimeAdapterError, match="no CUDA"):
        select_runtime_device(cpu_only, "cuda")
    with pytest.raises(RuntimeAdapterError, match="requested_device"):
        select_runtime_device(cpu_only, "mps")  # type: ignore[arg-type]


@pytest.mark.unit
def test_mock_backend_initialization_preserves_frozen_identity() -> None:
    backend = MockBackend(
        RuntimeDeviceCapabilities(
            cpu_available=True,
            cuda_device_count=0,
        )
    )
    adapter = create_runtime_adapter(backend)

    session = adapter.initialize("auto")
    execution_identity = adapter.revalidate_execution_identity()

    assert adapter.state is RuntimeAdapterState.READY
    assert type(execution_identity) is RuntimeExecutionIdentity
    assert execution_identity.runtime_state == "ready"
    assert execution_identity.backend_resources_owned is True
    assert execution_identity.runtime_session_identity_digest
    assert json.loads(
        json.dumps(execution_identity.identity_mapping())
    ) == execution_identity.identity_mapping()
    assert adapter.session is session
    assert session.runtime_config_digest == adapter.configuration.runtime_config_digest
    assert session.selected_device == "cpu"
    assert session.runtime_backend_name == "synthetic_sd35_backend"
    assert session.model_revision == adapter.configuration.model_revision
    assert session.inference_steps == adapter.configuration.inference_steps
    assert session.callback_index == adapter.configuration.callback_index
    assert (
        session.detection_conditioning_protocol
        == adapter.configuration.detection_conditioning_protocol
    )
    assert session.qk_layer_names == adapter.configuration.qk_layer_names
    assert backend.prepare_calls == [
        (adapter.configuration.runtime_config_digest, "cpu")
    ]
    with pytest.raises(RuntimeAdapterError, match="cannot initialize"):
        adapter.initialize("cpu")

    adapter.close()
    assert adapter.state is RuntimeAdapterState.CLOSED
    assert backend.close_calls == 1
    adapter.close()
    assert backend.close_calls == 1


@pytest.mark.unit
@pytest.mark.parametrize(
    "mutation",
    (
        "session_content",
        "configuration",
        "state",
        "resource_ownership",
    ),
)
def test_runtime_execution_identity_rejects_lifecycle_drift(
    mutation: str,
) -> None:
    backend = MockBackend(
        RuntimeDeviceCapabilities(
            cpu_available=True,
            cuda_device_count=0,
        )
    )
    adapter = create_runtime_adapter(backend)
    adapter.initialize("cpu")
    if mutation == "session_content":
        object.__setattr__(
            adapter.session,
            "model_revision",
            "drifted-model-revision",
        )
    elif mutation == "configuration":
        object.__setattr__(
            adapter.configuration,
            "model_id",
            "drifted/model",
        )
    elif mutation == "state":
        adapter._state = RuntimeAdapterState.CREATED
    else:
        adapter._owns_backend_resources = False

    with pytest.raises(RuntimeAdapterError, match="drifted|lost|differs"):
        adapter.revalidate_execution_identity()


@pytest.mark.unit
def test_failed_runtime_residual_state_is_rejected_and_close_cleans_it(
) -> None:
    backend = MockBackend(
        RuntimeDeviceCapabilities(
            cpu_available=True,
            cuda_device_count=0,
        )
    )
    adapter = create_runtime_adapter(backend)
    session = adapter.initialize("cpu")
    adapter._state = RuntimeAdapterState.FAILED
    adapter._owns_backend_resources = False
    adapter._session = session

    with pytest.raises(
        RuntimeAdapterError,
        match="residual execution state",
    ):
        adapter.revalidate_execution_identity()

    adapter.close()
    identity = adapter.revalidate_execution_identity()
    assert identity.runtime_state == "failed"
    assert identity.backend_resources_owned is False
    assert identity.runtime_session_identity_digest is None


@pytest.mark.unit
@pytest.mark.parametrize(
    ("field", "drift_value"),
    [
        ("candidate_id", "other_runtime_candidate"),
        ("runtime_config_digest", "0" * 64),
        ("runtime_backend_name", ""),
        ("selected_device", "cuda:1"),
        ("model_id", "different/model"),
        ("model_revision", "main"),
        ("pipeline_class", "diffusers.AutoPipelineForText2Image"),
        ("scheduler_class", "diffusers.DDIMScheduler"),
        ("inference_steps", 21),
        ("guidance_scale", 0.0),
        ("image_height", 1024),
        ("image_width", 1024),
        ("generation_seed_device", "cuda"),
        ("latent_dtype", "float32"),
        ("template_dtype", "float16"),
        ("score_dtype", "float16"),
        ("callback_index", 17),
        ("callback_hold_scheduler_intervals", 0),
        ("vae_decode_protocol", "different_decode"),
        ("vae_encode_protocol", "different_encode"),
        ("vae_scaling_factor_source", "hard_coded_scaling"),
        ("vae_shift_factor_source", "hard_coded_shift"),
        ("detection_schedule_index", 8),
        ("detection_conditioning_protocol", "cfg_enabled"),
        ("qk_layer_names", ("transformer_blocks.0.attn",)),
        ("dependency_lock", None),
    ],
)
def test_mock_backend_identity_drift_fails_and_releases_resources(
    field: str,
    drift_value: object,
) -> None:
    backend = MockBackend(
        RuntimeDeviceCapabilities(
            cpu_available=True,
            cuda_device_count=1,
        ),
        drift_field=field,
        drift_value=drift_value,
    )
    adapter = create_runtime_adapter(backend)

    with pytest.raises(RuntimeAdapterError, match="failed closed") as exc_info:
        adapter.initialize("cuda")

    assert isinstance(exc_info.value.__cause__, Exception)
    assert adapter.state is RuntimeAdapterState.FAILED
    assert backend.close_calls == 1
    with pytest.raises(RuntimeAdapterError, match="not ready"):
        _ = adapter.session


@pytest.mark.unit
def test_failed_initialization_cleanup_is_not_closed_twice() -> None:
    backend = MockBackend(
        RuntimeDeviceCapabilities(
            cpu_available=True,
            cuda_device_count=1,
        ),
        drift_field="callback_index",
        drift_value=17,
        reject_repeated_close=True,
    )
    adapter = create_runtime_adapter(backend)

    with pytest.raises(RuntimeAdapterError, match="failed closed"):
        adapter.initialize("cuda")

    assert adapter.state is RuntimeAdapterState.FAILED
    assert backend.close_calls == 1
    adapter.close()
    adapter.close()
    assert adapter.state is RuntimeAdapterState.FAILED
    assert backend.close_calls == 1


@pytest.mark.unit
def test_mock_backend_unexpected_failure_is_explicit() -> None:
    backend = MockBackend(
        RuntimeDeviceCapabilities(
            cpu_available=True,
            cuda_device_count=0,
        ),
        fail_prepare=True,
        reject_repeated_close=True,
    )
    adapter = create_runtime_adapter(backend)

    with pytest.raises(RuntimeAdapterError, match="unexpected") as exc_info:
        adapter.initialize()

    assert "mock prepare failure" in str(exc_info.value.__cause__)
    assert adapter.state is RuntimeAdapterState.FAILED
    assert backend.close_calls == 1
    adapter.close()
    adapter.close()
    assert adapter.state is RuntimeAdapterState.FAILED
    assert backend.close_calls == 1
