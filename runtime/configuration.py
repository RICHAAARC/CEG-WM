"""Frozen configuration parsing for the SD3.5 FlowMatch runtime candidate."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from typing import Mapping


RUNTIME_CANDIDATE_ID = "runtime_sd35_flowmatch"
DEFAULT_RUNTIME_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "runtime"
    / "runtime_sd35_flowmatch.json"
)

_RUNTIME_KEYS = frozenset(
    {
        "runtime_schema_version",
        "candidate_id",
        "model_id",
        "model_revision",
        "pipeline_class",
        "scheduler_class",
        "inference_steps",
        "guidance_scale",
        "image_height",
        "image_width",
        "generation_seed_device",
        "latent_dtype",
        "template_dtype",
        "score_dtype",
        "callback_index",
        "callback_hold_scheduler_intervals",
        "vae_decode_protocol",
        "vae_encode_protocol",
        "vae_scaling_factor_source",
        "vae_shift_factor_source",
        "detection_schedule_index",
        "detection_conditioning_protocol",
        "qk_layer_names",
        "dependency_lock",
    }
)
_DEPENDENCY_KEYS = (
    "python",
    "diffusers",
    "torch",
    "transformers",
    "accelerate",
    "numpy",
    "Pillow",
    "safetensors",
    "huggingface-hub",
)
_FROZEN_VALUES = {
    "runtime_schema_version": 1,
    "candidate_id": RUNTIME_CANDIDATE_ID,
    "pipeline_class": "diffusers.StableDiffusion3Pipeline",
    "scheduler_class": "diffusers.FlowMatchEulerDiscreteScheduler",
    "inference_steps": 20,
    "guidance_scale": 4.5,
    "image_height": 512,
    "image_width": 512,
    "generation_seed_device": "cpu",
    "latent_dtype": "float16",
    "template_dtype": "float32",
    "score_dtype": "float32",
    "callback_index": 18,
    "callback_hold_scheduler_intervals": 1,
    "vae_decode_protocol": "latent_div_scaling_factor_add_shift_factor",
    "vae_encode_protocol": "posterior_mode_sub_shift_factor_mul_scaling_factor",
    "vae_scaling_factor_source": "pipeline.vae.config.scaling_factor",
    "vae_shift_factor_source": "pipeline.vae.config.shift_factor",
    "detection_schedule_index": 7,
    "detection_conditioning_protocol": "sd3_empty_text_triplet_without_cfg",
}
_FROZEN_QK_LAYERS = (
    "transformer_blocks.0.attn",
    "transformer_blocks.23.attn",
)
class RuntimeConfigurationError(ValueError):
    """The frozen runtime configuration is missing, malformed, or drifted."""


def _canonical_json_bytes(value: Mapping[str, object]) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8", errors="strict")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise RuntimeConfigurationError(
            "runtime configuration is not canonical JSON"
        ) from exc


def _mapping(value: object, role: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise RuntimeConfigurationError(f"{role} must be a string-keyed mapping")
    return value


def _exact_keys(
    value: Mapping[str, object],
    expected: set[str] | frozenset[str],
    role: str,
) -> None:
    actual = set(value)
    if actual != set(expected):
        missing = sorted(set(expected) - actual)
        extra = sorted(actual - set(expected))
        raise RuntimeConfigurationError(
            f"{role} fields mismatch: missing={missing}, extra={extra}"
        )


def _text(value: object, field: str) -> str:
    if type(value) is not str or not value:
        raise RuntimeConfigurationError(f"{field} must be a non-empty string")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise RuntimeConfigurationError(
            f"{field} must be strict UTF-8 text"
        ) from exc
    return value


def _integer(value: object, field: str) -> int:
    if type(value) is not int:
        raise RuntimeConfigurationError(f"{field} must be an integer")
    return value


def _number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeConfigurationError(f"{field} must be numeric")
    converted = float(value)
    if not isfinite(converted):
        raise RuntimeConfigurationError(f"{field} must be finite")
    return converted


@dataclass(frozen=True, slots=True)
class RuntimeDependencyLock:
    """Ordered dependency-version metadata observed for runtime selection."""

    python: str
    diffusers: str
    torch: str
    transformers: str
    accelerate: str
    numpy: str
    pillow: str
    safetensors: str
    huggingface_hub: str

    def as_mapping(self) -> dict[str, str]:
        return {
            "python": self.python,
            "diffusers": self.diffusers,
            "torch": self.torch,
            "transformers": self.transformers,
            "accelerate": self.accelerate,
            "numpy": self.numpy,
            "Pillow": self.pillow,
            "safetensors": self.safetensors,
            "huggingface-hub": self.huggingface_hub,
        }

    def as_config_entries(self) -> list[dict[str, str]]:
        return [
            {
                "package_name": package_name,
                "version_specifier": version_specifier,
            }
            for package_name, version_specifier in self.as_mapping().items()
        ]


@dataclass(frozen=True, slots=True)
class Sd35RuntimeConfiguration:
    """Validated behavior identity with observed model selection metadata."""

    runtime_schema_version: int
    candidate_id: str
    model_id: str
    model_revision: str
    pipeline_class: str
    scheduler_class: str
    inference_steps: int
    guidance_scale: float
    image_height: int
    image_width: int
    generation_seed_device: str
    latent_dtype: str
    template_dtype: str
    score_dtype: str
    callback_index: int
    callback_hold_scheduler_intervals: int
    vae_decode_protocol: str
    vae_encode_protocol: str
    vae_scaling_factor_source: str
    vae_shift_factor_source: str
    detection_schedule_index: int
    detection_conditioning_protocol: str
    qk_layer_names: tuple[str, str]
    dependency_lock: RuntimeDependencyLock
    runtime_config_digest: str

    def identity_mapping(self) -> dict[str, object]:
        return {
            "runtime_schema_version": self.runtime_schema_version,
            "candidate_id": self.candidate_id,
            "pipeline_class": self.pipeline_class,
            "scheduler_class": self.scheduler_class,
            "inference_steps": self.inference_steps,
            "guidance_scale": self.guidance_scale,
            "image_height": self.image_height,
            "image_width": self.image_width,
            "generation_seed_device": self.generation_seed_device,
            "latent_dtype": self.latent_dtype,
            "template_dtype": self.template_dtype,
            "score_dtype": self.score_dtype,
            "callback_index": self.callback_index,
            "callback_hold_scheduler_intervals": (
                self.callback_hold_scheduler_intervals
            ),
            "vae_decode_protocol": self.vae_decode_protocol,
            "vae_encode_protocol": self.vae_encode_protocol,
            "vae_scaling_factor_source": self.vae_scaling_factor_source,
            "vae_shift_factor_source": self.vae_shift_factor_source,
            "detection_schedule_index": self.detection_schedule_index,
            "detection_conditioning_protocol": (
                self.detection_conditioning_protocol
            ),
            "qk_layer_names": list(self.qk_layer_names),
        }

    def configuration_mapping(self) -> dict[str, object]:
        """Return the complete runtime selection and behavior configuration."""

        return {
            **self.identity_mapping(),
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "dependency_lock": self.dependency_lock.as_config_entries(),
        }


def parse_runtime_configuration(value: object) -> Sd35RuntimeConfiguration:
    """Parse registered behavior identity and observed selection metadata."""

    root = _mapping(value, "runtime configuration")
    _exact_keys(root, _RUNTIME_KEYS, "runtime configuration")

    parsed: dict[str, object] = {
        "model_id": _text(root["model_id"], "model_id"),
        "model_revision": _text(root["model_revision"], "model_revision"),
    }
    for field, frozen in _FROZEN_VALUES.items():
        raw = root[field]
        if type(frozen) is int:
            parsed[field] = _integer(raw, field)
        elif type(frozen) is float:
            parsed[field] = _number(raw, field)
        else:
            parsed[field] = _text(raw, field)
        if parsed[field] != frozen:
            raise RuntimeConfigurationError(
                f"{field} does not match the registered runtime candidate"
            )

    raw_layers = root["qk_layer_names"]
    if not isinstance(raw_layers, list):
        raise RuntimeConfigurationError("qk_layer_names must be a JSON array")
    qk_layer_names = tuple(
        _text(layer, "qk_layer_names item") for layer in raw_layers
    )
    if qk_layer_names != _FROZEN_QK_LAYERS:
        raise RuntimeConfigurationError(
            "qk_layer_names do not match the registered runtime candidate"
        )

    raw_dependencies = root["dependency_lock"]
    if not isinstance(raw_dependencies, list):
        raise RuntimeConfigurationError(
            "dependency_lock must be an ordered JSON array"
        )
    if len(raw_dependencies) != len(_DEPENDENCY_KEYS):
        raise RuntimeConfigurationError(
            "dependency_lock package count does not match the registered candidate"
        )
    dependencies: dict[str, str] = {}
    for index, (entry, expected_package_name) in enumerate(
        zip(raw_dependencies, _DEPENDENCY_KEYS, strict=True)
    ):
        dependency = _mapping(
            entry,
            f"dependency_lock[{index}]",
        )
        _exact_keys(
            dependency,
            {"package_name", "version_specifier"},
            f"dependency_lock[{index}]",
        )
        package_name = _text(
            dependency["package_name"],
            f"dependency_lock[{index}].package_name",
        )
        if package_name != expected_package_name:
            raise RuntimeConfigurationError(
                "dependency_lock package order does not match the registered candidate"
            )
        dependencies[package_name] = _text(
            dependency["version_specifier"],
            f"dependency_lock[{index}].version_specifier",
        )
    dependency_lock = RuntimeDependencyLock(
        python=dependencies["python"],
        diffusers=dependencies["diffusers"],
        torch=dependencies["torch"],
        transformers=dependencies["transformers"],
        accelerate=dependencies["accelerate"],
        numpy=dependencies["numpy"],
        pillow=dependencies["Pillow"],
        safetensors=dependencies["safetensors"],
        huggingface_hub=dependencies["huggingface-hub"],
    )

    canonical_mapping = {
        key: value
        for key, value in root.items()
        if key not in {"model_id", "model_revision", "dependency_lock"}
    }
    runtime_config_digest = sha256(
        _canonical_json_bytes(canonical_mapping)
    ).hexdigest()
    configuration = Sd35RuntimeConfiguration(
        **parsed,
        qk_layer_names=qk_layer_names,
        dependency_lock=dependency_lock,
        runtime_config_digest=runtime_config_digest,
    )
    if sha256(
        _canonical_json_bytes(configuration.identity_mapping())
    ).hexdigest() != runtime_config_digest:
        raise RuntimeConfigurationError(
            "runtime configuration identity reconstruction mismatch"
        )
    return configuration


def load_runtime_configuration(
    path: str | Path = DEFAULT_RUNTIME_CONFIG_PATH,
) -> Sd35RuntimeConfiguration:
    """Load a UTF-8 JSON configuration without importing model dependencies."""

    config_path = Path(path)
    try:
        source = config_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise RuntimeConfigurationError(
            f"unable to read runtime configuration: {config_path}"
        ) from exc
    try:
        value = json.loads(source)
    except json.JSONDecodeError as exc:
        raise RuntimeConfigurationError(
            f"runtime configuration is not valid JSON: {config_path}"
        ) from exc
    return parse_runtime_configuration(value)
