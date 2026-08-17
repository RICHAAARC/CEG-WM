"""Runtime-owned generation observations for the frozen content router."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import importlib
import inspect
import json
from math import isfinite
import os
from pathlib import Path
import stat
from typing import Mapping

import torch
from torch.nn import functional as functional
from torch.nn.functional import conv2d as spatial_convolution

from main import (
    HfDetectionObservation,
    LfDetectionObservation,
    SemanticTextureRoutingObservations,
    SpatialRoutingObservation,
    derive_public_noise_stream,
    rgb8_image_digest,
)

from .adapter import RuntimeSession
from .backend import RuntimeContentBackend, RuntimeVaeFactors
from .configuration import Sd35RuntimeConfiguration
from .content_write import (
    RuntimeContentExecutionError,
    _decode_generation_latent,
    _encode_detection_image,
    _tensor,
)


ROUTING_OBSERVATION_CANDIDATE_ID = "routing_stqr"
ROUTING_PROBE_RELATIVE_STEP = 1.0e-3
_PREVIOUS_WRITE_CALLBACK_INDEX = 17
_ROUTING_WRITE_CALLBACK_INDEX = 18
SEMANTIC_TEXTURE_ROUTING_CANDIDATE_ID = "routing_semantic_texture_soft"
INSPYRENET_CHECKPOINT_BASENAME = "ckpt_base.pth"
INSPYRENET_CLASS_MODULE = "transparent_background.InSPyReNet"
INSPYRENET_CLASS_NAME = "InSPyReNet"
INSPYRENET_FACTORY_NAME = "InSPyReNet_SwinB"


class RuntimeRoutingObservationError(RuntimeError):
    """A generation-time routing observation violated its frozen boundary."""


@dataclass(frozen=True, slots=True)
class RuntimeSemanticTextureObservationResult:
    """Current ordinary RGB8-derived public M/T observations only."""

    candidate_id: str
    input_image_digest: str
    observations: SemanticTextureRoutingObservations
    source_class: str
    forward_api: str
    execution_evidence: str
    observation_identity: str


@dataclass(frozen=True, slots=True)
class RuntimeSemanticTextureDetectionObservationResult:
    """One-image runtime output for blind route plus public VAE observation."""

    semantic_texture: RuntimeSemanticTextureObservationResult
    hf_observation: HfDetectionObservation
    lf_observation: LfDetectionObservation
    runtime_config_digest: str
    observation_identity: str


def _load_registered_inspyrenet_factory() -> object:
    try:
        module = importlib.import_module(INSPYRENET_CLASS_MODULE)
    except Exception as exc:
        raise RuntimeRoutingObservationError(
            "registered transparent-background source dependency is unavailable"
        ) from exc
    candidates: list[object] = []
    for name, candidate in vars(module).items():
        if name.startswith("_") or not callable(candidate):
            continue
        try:
            parameters = inspect.signature(candidate).parameters
        except (TypeError, ValueError):
            continue
        if all(
            parameter_name in parameters
            for parameter_name in ("depth", "pretrained", "base_size")
        ):
            candidates.append(candidate)
    if len(candidates) != 1:
        raise RuntimeRoutingObservationError(
            "exactly one callable InSPyReNet public factory API is required"
        )
    return candidates[0]


def _load_registered_checkpoint(path: Path) -> Mapping[str, torch.Tensor]:
    if not isinstance(path, Path):
        raise RuntimeRoutingObservationError("InSPyReNet checkpoint path is invalid")
    try:
        checkpoint_stat = path.lstat()
    except OSError:
        raise RuntimeRoutingObservationError("InSPyReNet checkpoint is unavailable") from None
    if (
        not stat.S_ISREG(checkpoint_stat.st_mode)
    ):
        raise RuntimeRoutingObservationError("InSPyReNet checkpoint must be regular")
    descriptor: int | None = None
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_dev != checkpoint_stat.st_dev
            or opened.st_ino != checkpoint_stat.st_ino
            or opened.st_size != checkpoint_stat.st_size
        ):
            raise RuntimeRoutingObservationError(
                "InSPyReNet checkpoint changed during verification"
            )
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = None
            state_dict = torch.load(stream, map_location="cpu", weights_only=True)
    except RuntimeRoutingObservationError:
        raise
    except Exception as exc:
        raise RuntimeRoutingObservationError("InSPyReNet checkpoint load failed") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if (
        not isinstance(state_dict, Mapping)
        or not state_dict
        or any(type(key) is not str for key in state_dict)
        or any(type(value) is not torch.Tensor for value in state_dict.values())
    ):
        raise RuntimeRoutingObservationError(
            "InSPyReNet checkpoint is not a strict tensor state_dict"
        )
    return state_dict


def _validate_inspyrenet_output(output: object) -> torch.Tensor:
    if type(output) is not dict or set(output) != {"saliency", "laplacian"}:
        raise RuntimeRoutingObservationError("forward_inspyre output identity drifted")
    saliency = output["saliency"]
    laplacian = output["laplacian"]
    if (
        type(saliency) is not list
        or len(saliency) != 4
        or type(laplacian) is not list
        or len(laplacian) != 3
    ):
        raise RuntimeRoutingObservationError("forward_inspyre saliency pyramid drifted")
    for tensor in (*saliency, *laplacian):
        if (
            type(tensor) is not torch.Tensor
            or tensor.ndim != 4
            or tuple(tensor.shape[:2]) != (1, 1)
            or tensor.shape[2] <= 0
            or tensor.shape[3] <= 0
            or not tensor.is_floating_point()
            or not bool(torch.isfinite(tensor).all().item())
        ):
            raise RuntimeRoutingObservationError(
                "forward_inspyre pyramid tensor identity drifted"
            )
    raw_finest = saliency[-1]
    if tuple(raw_finest.shape[2:]) != (1024, 1024):
        raise RuntimeRoutingObservationError("raw finest d0 saliency output is invalid")
    return raw_finest


class InspyrenetSemanticRuntime:
    """Strict production InSPyReNet loader; injection is explicitly test-only."""

    __slots__ = ("_device", "_execution_evidence", "_model")

    def __init__(self, checkpoint_path: Path, *, selected_device: str = "cpu") -> None:
        try:
            device = torch.device(selected_device)
        except (RuntimeError, TypeError, ValueError):
            raise RuntimeRoutingObservationError("InSPyReNet device is invalid") from None
        factory = _load_registered_inspyrenet_factory()
        state_dict = _load_registered_checkpoint(checkpoint_path)
        try:
            model = factory(depth=64, pretrained=False, base_size=[384, 384])
            if not isinstance(model, torch.nn.Module) or not callable(
                getattr(model, "forward_inspyre", None)
            ):
                raise RuntimeRoutingObservationError(
                    "InSPyReNet factory must return a torch module with forward_inspyre"
                )
            incompatible = model.load_state_dict(state_dict, strict=True)
            if incompatible.missing_keys or incompatible.unexpected_keys:
                raise RuntimeRoutingObservationError(
                    "strict state_dict returned incompatibilities"
                )
            model.to(device)
            model.eval()
        except RuntimeRoutingObservationError:
            raise
        except Exception as exc:
            raise RuntimeRoutingObservationError(
                "strict InSPyReNet model initialization failed"
            ) from exc
        self._device = device
        self._model = model
        self._execution_evidence = "registered_source_checkpoint_strict_production"

    @classmethod
    def from_injected_model_for_test(
        cls,
        model: torch.nn.Module,
        *,
        selected_device: str = "cpu",
    ) -> InspyrenetSemanticRuntime:
        """Create explicit non-production evidence for lightweight boundary tests."""

        if not isinstance(model, torch.nn.Module) or not callable(
            getattr(model, "forward_inspyre", None)
        ):
            raise RuntimeRoutingObservationError(
                "injected test model must be a torch module with forward_inspyre"
            )
        instance = object.__new__(cls)
        instance._device = torch.device(selected_device)
        instance._model = model.to(instance._device).eval()
        instance._execution_evidence = "injected_minimal_model_test_only_not_production"
        return instance

    def observe(self, image_rgb8: torch.Tensor) -> RuntimeSemanticTextureObservationResult:
        image = image_rgb8.detach().to(device="cpu").contiguous().clone()
        try:
            input_digest = rgb8_image_digest(image)
        except ValueError:
            raise RuntimeRoutingObservationError(
                "semantic-texture input must be ordinary RGB8 [1,3,H,W]"
            ) from None
        normalized = image.to(dtype=torch.float32) / 255.0
        normalized = functional.interpolate(
            normalized,
            size=(1024, 1024),
            mode="bilinear",
            align_corners=False,
        )
        mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(1, 3, 1, 1)
        model_input = ((normalized - mean) / std).to(self._device)
        try:
            with torch.no_grad():
                raw_finest = _validate_inspyrenet_output(
                    self._model.forward_inspyre(model_input)
                )
                semantic = torch.sigmoid(raw_finest)
        except RuntimeRoutingObservationError:
            raise
        except Exception as exc:
            raise RuntimeRoutingObservationError(
                "direct forward_inspyre raw-d0 execution failed"
            ) from exc
        semantic = functional.interpolate(
            semantic.to(dtype=torch.float32),
            size=(64, 64),
            mode="bilinear",
            align_corners=False,
        ).to(device="cpu")
        if not bool(torch.isfinite(semantic).all().item()) or bool(
            ((semantic < 0.0) | (semantic > 1.0)).any().item()
        ):
            raise RuntimeRoutingObservationError("semantic probability M is invalid")

        rgb = image.to(dtype=torch.float32) / 255.0
        weights = torch.tensor((0.299, 0.587, 0.114), dtype=torch.float32).view(1, 3, 1, 1)
        grayscale = (rgb * weights).sum(dim=1, keepdim=True)
        padded = functional.pad(grayscale, (1, 1, 1, 1), mode="replicate")
        sobel_x = torch.tensor(
            ((-1.0, 0.0, 1.0), (-2.0, 0.0, 2.0), (-1.0, 0.0, 1.0)),
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        sobel_y = sobel_x.transpose(2, 3).contiguous()
        magnitude = torch.sqrt(
            spatial_convolution(padded, sobel_x).square()
            + spatial_convolution(padded, sobel_y).square()
        )
        downsampled = functional.interpolate(magnitude, size=(64, 64), mode="area")
        flattened = tuple(float(value) for value in downsampled.reshape(-1).tolist())
        positive = sorted(
            ((value, index) for index, value in enumerate(flattened) if value > 0.0),
            key=lambda item: (item[0], item[1]),
        )
        if positive:
            rank = (95 * len(positive) + 99) // 100
            positive_gradient_nearest_rank_p95 = positive[rank - 1][0]
            texture = torch.clamp(
                downsampled / positive_gradient_nearest_rank_p95,
                0.0,
                1.0,
            )
        else:
            texture = torch.zeros_like(downsampled)
        source_base = {
            "candidate_id": SEMANTIC_TEXTURE_ROUTING_CANDIDATE_ID,
            "forward_api": "forward_inspyre",
            "forward_output_selector": "out['saliency'][-1]_raw_finest_d0",
            "input_image_digest": input_digest,
            "model_preprocess": (
                "rgb8_to_float32_div255_static_1024_bilinear_align_corners_false_"
                "imagenet_mean_0.485_0.456_0.406_std_0.229_0.224_0.225"
            ),
            "probability_operator": "torch.sigmoid_exactly_once",
            "probability_resize": "bilinear_align_corners_false_to_64x64",
            "texture_operator": (
                "rgb8_float32_grayscale_0.299_0.587_0.114_replicate_pad1_"
                "sobel3x3_magnitude_area_64x64_positive_nearest_rank_p95_clamp"
            ),
        }
        semantic_observation = _spatial_observation(
            semantic.squeeze(1),
            source_identity={**source_base, "observation_role": "semantic_probability_M"},
            role="semantic_probability_M",
        )
        texture_observation = _spatial_observation(
            texture.squeeze(1),
            source_identity={**source_base, "observation_role": "texture_complexity_T"},
            role="texture_complexity_T",
        )
        observations = SemanticTextureRoutingObservations(
            semantic_probability=semantic_observation,
            texture_complexity=texture_observation,
        )
        identity = _canonical_digest(
            {
                **source_base,
                "semantic_probability_digest": semantic_observation.values_digest,
                "texture_complexity_digest": texture_observation.values_digest,
            }
        )
        return RuntimeSemanticTextureObservationResult(
            candidate_id=SEMANTIC_TEXTURE_ROUTING_CANDIDATE_ID,
            input_image_digest=input_digest,
            observations=observations,
            source_class=(
                f"{type(self._model).__module__}.{type(self._model).__qualname__}"
            ),
            forward_api="forward_inspyre",
            execution_evidence=self._execution_evidence,
            observation_identity=identity,
        )


@dataclass(frozen=True, slots=True)
class RuntimeRoutingReferenceMeasurement:
    """One transient generation measurement used to fit routing references.

    Callback tensors and semantic RGB stay in memory so the same measurement
    can be normalized after a cross-fit reference is frozen.  This type has no
    record serialization; only its numeric values and digests may cross into
    governed experiment records.
    """

    candidate_id: str
    runtime_config_digest: str
    model_id: str
    model_revision: str
    sample_index: int
    callback_indices: tuple[int, ...]
    previous_write_latent: torch.Tensor
    routing_write_latent: torch.Tensor
    semantic_rgb: torch.Tensor
    texture_gradient_values: tuple[float, ...]
    texture_spatial_shape: tuple[int, int]
    response_ratio_values: tuple[float, ...]
    response_spatial_shape: tuple[int, int]
    sensitivity_ratio_values: tuple[float, ...]
    sensitivity_spatial_shape: tuple[int, int]
    public_probe_domain_digest: str
    public_probe_values_float32_be_sha256: str
    nominal_relative_probe_step: float
    actual_probe_step: float

    def __post_init__(self) -> None:
        _validate_reference_measurement(self)


@dataclass(frozen=True, slots=True)
class RuntimeRoutingObservationResult:
    """Transient routing tensors and public numeric observations.

    The callback latents and decoded RGB image are runtime-owned inputs for the
    content embedder and the upper-layer CLIP semantic producer.  They are not
    record fields and this type intentionally exposes no record serialization.
    """

    candidate_id: str
    runtime_config_digest: str
    model_id: str
    model_revision: str
    sample_index: int
    callback_indices: tuple[int, ...]
    previous_write_latent: torch.Tensor
    routing_write_latent: torch.Tensor
    routing_rgb: torch.Tensor
    texture: SpatialRoutingObservation
    response: SpatialRoutingObservation
    sensitivity: SpatialRoutingObservation
    public_probe_domain_digest: str
    public_probe_values_float32_be_sha256: str
    nominal_relative_probe_step: float
    actual_probe_step: float
    reference_gradient: float
    reference_response: float
    reference_sensitivity: float


def _canonical_digest(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _float32_values(value: torch.Tensor, role: str) -> tuple[float, ...]:
    checked = _tensor(value, role=role)
    return tuple(
        float(item)
        for item in checked.detach().to(
            device="cpu",
            dtype=torch.float32,
        ).reshape(-1)
    )


def _is_sha256_digest(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_raw_spatial_values(
    values: object,
    spatial_shape: object,
    role: str,
) -> tuple[float, ...]:
    if (
        type(spatial_shape) is not tuple
        or len(spatial_shape) != 2
        or any(type(size) is not int or size <= 0 for size in spatial_shape)
    ):
        raise RuntimeRoutingObservationError(
            f"{role} spatial shape must be positive [H,W]"
        )
    if type(values) is not tuple or len(values) != spatial_shape[0] * spatial_shape[1]:
        raise RuntimeRoutingObservationError(
            f"{role} values do not match the spatial shape"
        )
    if any(
        type(value) is not float or not isfinite(value) or value < 0.0
        for value in values
    ):
        raise RuntimeRoutingObservationError(
            f"{role} values must be finite non-negative float32 values"
        )
    return values


def _validate_reference_measurement(
    measurement: RuntimeRoutingReferenceMeasurement,
) -> None:
    if measurement.candidate_id != ROUTING_OBSERVATION_CANDIDATE_ID:
        raise RuntimeRoutingObservationError(
            "routing measurement candidate identity drifted"
        )
    if (
        not measurement.model_id
        or not measurement.model_revision
        or not _is_sha256_digest(measurement.runtime_config_digest)
        or type(measurement.sample_index) is not int
        or measurement.sample_index < 0
    ):
        raise RuntimeRoutingObservationError(
            "routing measurement runtime or sample identity is invalid"
        )
    if measurement.callback_indices != tuple(
        range(len(measurement.callback_indices))
    ) or len(measurement.callback_indices) <= _ROUTING_WRITE_CALLBACK_INDEX:
        raise RuntimeRoutingObservationError(
            "routing measurement callback sequence drifted"
        )
    previous = _tensor(
        measurement.previous_write_latent,
        role="previous_write_latent",
        dtype=torch.float16,
    )
    routing = _tensor(
        measurement.routing_write_latent,
        role="routing_write_latent",
        shape=previous.shape,
        dtype=torch.float16,
        device=previous.device,
    )
    semantic_rgb = _tensor(measurement.semantic_rgb, role="semantic_rgb")
    if (
        semantic_rgb.ndim != 4
        or semantic_rgb.shape[0] != 1
        or semantic_rgb.shape[1] != 3
        or bool((semantic_rgb < 0.0).any().item())
        or bool((semantic_rgb > 1.0).any().item())
    ):
        raise RuntimeRoutingObservationError(
            "routing measurement semantic RGB is invalid"
        )
    _validate_raw_spatial_values(
        measurement.texture_gradient_values,
        measurement.texture_spatial_shape,
        "texture gradient",
    )
    _validate_raw_spatial_values(
        measurement.response_ratio_values,
        measurement.response_spatial_shape,
        "response ratio",
    )
    _validate_raw_spatial_values(
        measurement.sensitivity_ratio_values,
        measurement.sensitivity_spatial_shape,
        "sensitivity ratio",
    )
    if (
        not _is_sha256_digest(measurement.public_probe_domain_digest)
        or not _is_sha256_digest(
            measurement.public_probe_values_float32_be_sha256
        )
        or measurement.nominal_relative_probe_step
        != ROUTING_PROBE_RELATIVE_STEP
        or not isfinite(measurement.actual_probe_step)
        or measurement.actual_probe_step <= 0.0
    ):
        raise RuntimeRoutingObservationError(
            "routing measurement probe identity is invalid"
        )


def _positive_reference(value: object, role: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise RuntimeRoutingObservationError(
            f"{role} must be finite and positive"
        )
    return float(value)


def _validate_session(
    configuration: Sd35RuntimeConfiguration,
    session: RuntimeSession,
) -> None:
    if type(configuration) is not Sd35RuntimeConfiguration:
        raise RuntimeRoutingObservationError(
            "configuration must be Sd35RuntimeConfiguration"
        )
    if type(session) is not RuntimeSession:
        raise RuntimeRoutingObservationError("runtime session is invalid")
    expected = {
        "candidate_id": configuration.candidate_id,
        "runtime_config_digest": configuration.runtime_config_digest,
        "inference_steps": configuration.inference_steps,
        "latent_dtype": configuration.latent_dtype,
        "callback_index": configuration.callback_index,
        "vae_decode_protocol": configuration.vae_decode_protocol,
    }
    for field, expected_value in expected.items():
        if getattr(session, field) != expected_value:
            raise RuntimeRoutingObservationError(
                f"runtime session {field} drifted before routing observation"
            )
    if configuration.callback_index != _ROUTING_WRITE_CALLBACK_INDEX:
        raise RuntimeRoutingObservationError(
            "routing observation requires the registered write callback"
        )
    if configuration.inference_steps <= _ROUTING_WRITE_CALLBACK_INDEX:
        raise RuntimeRoutingObservationError(
            "runtime schedule does not expose the routing callbacks"
        )


def _spatial_observation(
    values: torch.Tensor,
    *,
    source_identity: object,
    role: str,
) -> SpatialRoutingObservation:
    checked = _tensor(values, role=role)
    if checked.ndim != 3 or checked.shape[0] != 1:
        raise RuntimeRoutingObservationError(
            f"{role} must have shape [1,H,W]"
        )
    if bool((checked < 0.0).any().item()) or bool((checked > 1.0).any().item()):
        raise RuntimeRoutingObservationError(
            f"{role} must remain in the closed unit interval"
        )
    return SpatialRoutingObservation(
        values=tuple(
            float(item)
            for item in checked.detach().to(
                device="cpu",
                dtype=torch.float32,
            ).reshape(-1)
        ),
        spatial_shape=(int(checked.shape[1]), int(checked.shape[2])),
        source_identity_digest=_canonical_digest(source_identity),
    )


def _texture_gradient(routing_rgb: torch.Tensor) -> torch.Tensor:
    rgb = _tensor(routing_rgb, role="routing_rgb")
    if rgb.ndim != 4 or rgb.shape[0] != 1 or rgb.shape[1] != 3:
        raise RuntimeRoutingObservationError(
            "routing RGB must have shape [1,3,H,W]"
        )
    if bool((rgb < 0.0).any().item()) or bool((rgb > 1.0).any().item()):
        raise RuntimeRoutingObservationError(
            "routing RGB must use the frozen closed unit interval"
        )
    rgb_float = rgb.detach().to(dtype=torch.float32)
    luminance_weights = torch.tensor(
        (0.299, 0.587, 0.114),
        dtype=torch.float32,
        device=rgb_float.device,
    ).reshape(1, 3, 1, 1)
    grayscale = (rgb_float * luminance_weights).sum(dim=1, keepdim=True)
    padded = functional.pad(grayscale, (1, 1, 1, 1), mode="replicate")
    sobel_horizontal = torch.tensor(
        ((-1.0, 0.0, 1.0), (-2.0, 0.0, 2.0), (-1.0, 0.0, 1.0)),
        dtype=torch.float32,
        device=rgb_float.device,
    ).reshape(1, 1, 3, 3)
    sobel_vertical = sobel_horizontal.transpose(2, 3).contiguous()
    gradient_horizontal = spatial_convolution(padded, sobel_horizontal)
    gradient_vertical = spatial_convolution(padded, sobel_vertical)
    gradient = torch.sqrt(
        gradient_horizontal.square() + gradient_vertical.square()
    ).squeeze(1)
    return _tensor(gradient, role="texture_gradient").detach().clone()


def _response_ratio(
    previous_latent: torch.Tensor,
    routing_latent: torch.Tensor,
) -> torch.Tensor:
    previous = previous_latent.detach().to(dtype=torch.float32)
    current = routing_latent.detach().to(dtype=torch.float32)
    difference_rms = torch.sqrt((current - previous).square().mean(dim=1))
    previous_rms = torch.sqrt(previous.square().mean(dim=1))
    current_rms = torch.sqrt(current.square().mean(dim=1))
    raw_response = difference_rms / (previous_rms + current_rms + 1.0e-12)
    return _tensor(raw_response, role="response_ratio").detach().clone()


def _sensitivity_ratio(
    backend: RuntimeContentBackend,
    routing_latent: torch.Tensor,
    routing_rgb: torch.Tensor,
    factors: RuntimeVaeFactors,
) -> tuple[torch.Tensor, str, str, float]:
    shape = tuple(int(size) for size in routing_latent.shape)
    stream = derive_public_noise_stream(
        {
            "candidate_id": ROUTING_OBSERVATION_CANDIDATE_ID,
            "conditioning_protocol": "generation_callback18_vae_local_sensitivity",
            "operator": "local_sensitivity_public_probe",
            "responsibility_domain": "public_noise",
            "schedule_index": _ROUTING_WRITE_CALLBACK_INDEX,
            "tensor_role": "latent_probe",
        },
        shape,
    )
    probe = torch.tensor(
        stream.values,
        dtype=torch.float32,
        device=routing_latent.device,
    ).reshape(shape)
    centered_probe = probe - probe.mean()
    probe_rms = torch.sqrt(centered_probe.square().mean())
    if not bool(torch.isfinite(probe_rms).item()) or float(probe_rms.item()) <= 0.0:
        raise RuntimeRoutingObservationError(
            "routing public probe has zero or non-finite RMS"
        )
    unit_probe = centered_probe / probe_rms
    latent_float = routing_latent.detach().to(dtype=torch.float32)
    latent_rms = torch.sqrt(latent_float.square().mean())
    if not bool(torch.isfinite(latent_rms).item()) or float(latent_rms.item()) <= 0.0:
        raise RuntimeRoutingObservationError(
            "routing callback latent has zero or non-finite RMS"
        )
    nominal_step = latent_rms * ROUTING_PROBE_RELATIVE_STEP
    perturbed_latent = (latent_float + nominal_step * unit_probe).to(
        dtype=routing_latent.dtype
    )
    actual_delta = perturbed_latent.to(dtype=torch.float32) - latent_float
    actual_step_tensor = torch.sqrt(actual_delta.square().mean())
    if (
        not bool(torch.isfinite(actual_step_tensor).item())
        or float(actual_step_tensor.item()) <= 0.0
    ):
        raise RuntimeRoutingObservationError(
            "routing public probe disappeared at the actual latent dtype"
        )
    actual_step = float(actual_step_tensor.item())
    try:
        perturbed_rgb = _decode_generation_latent(
            backend,
            perturbed_latent,
            factors,
            "routing_probe",
        )
    except RuntimeContentExecutionError as exc:
        raise RuntimeRoutingObservationError(
            "routing perturbed latent VAE decode failed"
        ) from exc
    if perturbed_rgb.shape != routing_rgb.shape:
        raise RuntimeRoutingObservationError(
            "routing probe VAE image shape drifted"
        )
    perturbed_rgb = _tensor(
        perturbed_rgb,
        role="routing_probe_rgb",
        shape=routing_rgb.shape,
    )
    if bool((perturbed_rgb < 0.0).any().item()) or bool(
        (perturbed_rgb > 1.0).any().item()
    ):
        raise RuntimeRoutingObservationError(
            "routing probe VAE decode must return RGB in [0,1]"
        )
    sensitivity = torch.sqrt(
        (
            perturbed_rgb.detach().to(dtype=torch.float32)
            - routing_rgb.detach().to(dtype=torch.float32)
        ).square().mean(dim=1)
    )
    raw_sensitivity = sensitivity / actual_step
    return (
        _tensor(
            raw_sensitivity,
            role="sensitivity_ratio",
        ).detach().clone(),
        stream.domain_digest,
        stream.values_float32_be_sha256,
        actual_step,
    )


def measure_generation_routing_reference_inputs(
    backend: RuntimeContentBackend,
    configuration: Sd35RuntimeConfiguration,
    session: RuntimeSession,
    base_latent: torch.Tensor,
    *,
    sample_index: int,
) -> RuntimeRoutingReferenceMeasurement:
    """Run one generation and retain raw T/R/Q_sens reference inputs."""

    if not isinstance(backend, RuntimeContentBackend):
        raise RuntimeRoutingObservationError(
            "prepared backend lacks the routing observation execution protocol"
        )
    _validate_session(configuration, session)
    if type(sample_index) is not int or sample_index < 0:
        raise RuntimeRoutingObservationError(
            "sample_index must be a non-negative integer"
        )
    try:
        latent = _tensor(
            base_latent,
            role="routing_base_latent",
            dtype=torch.float16,
        )
    except RuntimeContentExecutionError as exc:
        raise RuntimeRoutingObservationError(
            "routing base latent is invalid"
        ) from exc
    if latent.ndim != 4 or latent.shape[0] != 1:
        raise RuntimeRoutingObservationError(
            "routing base latent must have shape [1,C,H,W]"
        )
    if str(latent.device) != session.selected_device:
        raise RuntimeRoutingObservationError(
            "routing base latent device does not match the runtime session"
        )

    callback_indices: list[int] = []
    previous_latent: torch.Tensor | None = None
    routing_latent: torch.Tensor | None = None

    def capture_callback(index: int, callback_latent: torch.Tensor) -> torch.Tensor:
        nonlocal previous_latent, routing_latent
        if type(index) is not int or not 0 <= index < configuration.inference_steps:
            raise RuntimeRoutingObservationError(
                "routing generation reported a wrong callback index"
            )
        if index in callback_indices:
            raise RuntimeRoutingObservationError(
                "routing generation duplicated a callback index"
            )
        try:
            current = _tensor(
                callback_latent,
                role="routing_callback_latent",
                shape=latent.shape,
                dtype=torch.float16,
                device=latent.device,
            )
        except RuntimeContentExecutionError as exc:
            raise RuntimeRoutingObservationError(
                "routing generation callback latent is invalid"
            ) from exc
        callback_indices.append(index)
        if index == _PREVIOUS_WRITE_CALLBACK_INDEX:
            previous_latent = current.detach().clone()
        elif index == _ROUTING_WRITE_CALLBACK_INDEX:
            routing_latent = current.detach().clone()
        return current

    try:
        backend.run_generation(latent.detach().clone(), capture_callback)
    except RuntimeRoutingObservationError:
        raise
    except Exception as exc:
        raise RuntimeRoutingObservationError(
            "routing generation backend failed"
        ) from exc
    expected_indices = tuple(range(configuration.inference_steps))
    if tuple(callback_indices) != expected_indices:
        raise RuntimeRoutingObservationError(
            "routing generation callback sequence is missing or out of order"
        )
    if previous_latent is None or routing_latent is None:
        raise RuntimeRoutingObservationError(
            "routing generation did not expose both registered callbacks"
        )

    factors = backend.vae_factors()
    if type(factors) is not RuntimeVaeFactors:
        raise RuntimeRoutingObservationError(
            "backend VAE factors do not match the routing protocol"
        )
    try:
        routing_rgb = _decode_generation_latent(
            backend,
            routing_latent,
            factors,
            "routing",
        )
    except RuntimeContentExecutionError as exc:
        raise RuntimeRoutingObservationError(
            "routing callback latent VAE decode failed"
        ) from exc
    routing_rgb = _tensor(routing_rgb, role="routing_rgb").detach().clone()
    if (
        routing_rgb.ndim != 4
        or routing_rgb.shape[0] != 1
        or routing_rgb.shape[1] != 3
    ):
        raise RuntimeRoutingObservationError(
            "routing callback VAE decode must return [1,3,H,W] RGB"
        )
    if bool((routing_rgb < 0.0).any().item()) or bool(
        (routing_rgb > 1.0).any().item()
    ):
        raise RuntimeRoutingObservationError(
            "routing callback VAE decode must return RGB in [0,1]"
        )

    texture_gradient = _texture_gradient(routing_rgb)
    response_ratio = _response_ratio(previous_latent, routing_latent)
    (
        sensitivity_ratio,
        public_probe_domain_digest,
        public_probe_values_digest,
        actual_probe_step,
    ) = _sensitivity_ratio(
        backend,
        routing_latent,
        routing_rgb,
        factors,
    )
    texture_values = _float32_values(texture_gradient, "texture_gradient")
    response_values = _float32_values(response_ratio, "response_ratio")
    sensitivity_values = _float32_values(
        sensitivity_ratio,
        "sensitivity_ratio",
    )
    return RuntimeRoutingReferenceMeasurement(
        candidate_id=ROUTING_OBSERVATION_CANDIDATE_ID,
        runtime_config_digest=configuration.runtime_config_digest,
        model_id=configuration.model_id,
        model_revision=configuration.model_revision,
        sample_index=sample_index,
        callback_indices=tuple(callback_indices),
        previous_write_latent=previous_latent.detach().clone(),
        routing_write_latent=routing_latent.detach().clone(),
        semantic_rgb=routing_rgb.detach().clone(),
        texture_gradient_values=texture_values,
        texture_spatial_shape=(
            int(texture_gradient.shape[1]),
            int(texture_gradient.shape[2]),
        ),
        response_ratio_values=response_values,
        response_spatial_shape=(
            int(response_ratio.shape[1]),
            int(response_ratio.shape[2]),
        ),
        sensitivity_ratio_values=sensitivity_values,
        sensitivity_spatial_shape=(
            int(sensitivity_ratio.shape[1]),
            int(sensitivity_ratio.shape[2]),
        ),
        public_probe_domain_digest=public_probe_domain_digest,
        public_probe_values_float32_be_sha256=public_probe_values_digest,
        nominal_relative_probe_step=ROUTING_PROBE_RELATIVE_STEP,
        actual_probe_step=actual_probe_step,
    )


def normalize_generation_routing_measurement(
    measurement: RuntimeRoutingReferenceMeasurement,
    *,
    reference_gradient: float,
    reference_response: float,
    reference_sensitivity: float,
) -> RuntimeRoutingObservationResult:
    """Normalize one already measured generation without rerunning the model."""

    if type(measurement) is not RuntimeRoutingReferenceMeasurement:
        raise RuntimeRoutingObservationError(
            "routing normalization requires an exact reference measurement"
        )
    _validate_reference_measurement(measurement)
    gradient_reference = _positive_reference(
        reference_gradient,
        "reference_gradient",
    )
    response_reference = _positive_reference(
        reference_response,
        "reference_response",
    )
    sensitivity_reference = _positive_reference(
        reference_sensitivity,
        "reference_sensitivity",
    )
    source_identity = {
        "candidate_id": measurement.candidate_id,
        "runtime_config_digest": measurement.runtime_config_digest,
        "sample_index": measurement.sample_index,
        "callback_indices": measurement.callback_indices,
        "public_probe_domain_digest": measurement.public_probe_domain_digest,
    }
    texture_tensor = torch.tensor(
        measurement.texture_gradient_values,
        dtype=torch.float32,
    ).reshape((1, *measurement.texture_spatial_shape))
    response_tensor = torch.tensor(
        measurement.response_ratio_values,
        dtype=torch.float32,
    ).reshape((1, *measurement.response_spatial_shape))
    sensitivity_tensor = torch.tensor(
        measurement.sensitivity_ratio_values,
        dtype=torch.float32,
    ).reshape((1, *measurement.sensitivity_spatial_shape))
    texture = _spatial_observation(
        torch.clamp(texture_tensor / gradient_reference, 0.0, 1.0),
        source_identity={
            **source_identity,
            "observation_role": "texture_sobel_gradient",
            "reference_gradient": gradient_reference,
        },
        role="texture_observation",
    )
    response = _spatial_observation(
        torch.clamp(response_tensor / response_reference, 0.0, 1.0),
        source_identity={
            **source_identity,
            "observation_role": "scheduler_response_channel_rms",
            "reference_response": response_reference,
        },
        role="response_observation",
    )
    sensitivity = _spatial_observation(
        torch.clamp(sensitivity_tensor / sensitivity_reference, 0.0, 1.0),
        source_identity={
            **source_identity,
            "observation_role": "vae_local_sensitivity_rgb_rms",
            "reference_sensitivity": sensitivity_reference,
        },
        role="sensitivity_observation",
    )
    return RuntimeRoutingObservationResult(
        candidate_id=measurement.candidate_id,
        runtime_config_digest=measurement.runtime_config_digest,
        model_id=measurement.model_id,
        model_revision=measurement.model_revision,
        sample_index=measurement.sample_index,
        callback_indices=measurement.callback_indices,
        previous_write_latent=measurement.previous_write_latent.detach().clone(),
        routing_write_latent=measurement.routing_write_latent.detach().clone(),
        routing_rgb=measurement.semantic_rgb.detach().clone(),
        texture=texture,
        response=response,
        sensitivity=sensitivity,
        public_probe_domain_digest=measurement.public_probe_domain_digest,
        public_probe_values_float32_be_sha256=(
            measurement.public_probe_values_float32_be_sha256
        ),
        nominal_relative_probe_step=measurement.nominal_relative_probe_step,
        actual_probe_step=measurement.actual_probe_step,
        reference_gradient=gradient_reference,
        reference_response=response_reference,
        reference_sensitivity=sensitivity_reference,
    )


def observe_generation_routing(
    backend: RuntimeContentBackend,
    configuration: Sd35RuntimeConfiguration,
    session: RuntimeSession,
    base_latent: torch.Tensor,
    *,
    sample_index: int,
    reference_gradient: float,
    reference_response: float,
    reference_sensitivity: float,
) -> RuntimeRoutingObservationResult:
    """Measure once, then normalize T, R, and Q_sens with frozen references."""

    measurement = measure_generation_routing_reference_inputs(
        backend,
        configuration,
        session,
        base_latent,
        sample_index=sample_index,
    )
    return normalize_generation_routing_measurement(
        measurement,
        reference_gradient=reference_gradient,
        reference_response=reference_response,
        reference_sensitivity=reference_sensitivity,
    )


def observe_semantic_texture_detection(
    backend: RuntimeContentBackend,
    configuration: Sd35RuntimeConfiguration,
    session: RuntimeSession,
    detection_image_rgb8: torch.Tensor,
    semantic_runtime: InspyrenetSemanticRuntime,
) -> RuntimeSemanticTextureDetectionObservationResult:
    """Rebuild M/T and the VAE observation from the same current RGB8 image."""

    if not isinstance(backend, RuntimeContentBackend):
        raise RuntimeRoutingObservationError(
            "semantic-texture detection requires the public VAE backend"
        )
    _validate_session(configuration, session)
    if type(semantic_runtime) is not InspyrenetSemanticRuntime:
        raise RuntimeRoutingObservationError(
            "semantic-texture detection requires InspyrenetSemanticRuntime"
        )
    current_rgb8 = detection_image_rgb8.detach().to(device="cpu").contiguous().clone()
    semantic_texture = semantic_runtime.observe(current_rgb8)
    if semantic_texture.input_image_digest != rgb8_image_digest(current_rgb8):
        raise RuntimeRoutingObservationError(
            "semantic-texture current RGB8 snapshot identity drifted"
        )
    image = current_rgb8.to(
        device=session.selected_device,
        dtype=torch.float32,
    ) / 255.0
    factors = backend.vae_factors()
    if type(factors) is not RuntimeVaeFactors:
        raise RuntimeRoutingObservationError(
            "semantic-texture detection VAE factors drifted"
        )
    try:
        latent = _encode_detection_image(
            backend,
            image,
            factors,
            "semantic_texture_detection",
        )
    except RuntimeContentExecutionError as exc:
        raise RuntimeRoutingObservationError(
            "semantic-texture public RGB-to-VAE observation failed"
        ) from exc
    values = _float32_values(latent, "semantic_texture_detection_latent")
    shape = tuple(int(size) for size in latent.shape)
    hf_observation = HfDetectionObservation.from_public_image_encoding(values, shape)
    lf_observation = LfDetectionObservation.from_public_image_encoding(values, shape)
    if hf_observation.observation_digest != lf_observation.observation_digest:
        raise RuntimeRoutingObservationError(
            "semantic-texture LF/HF public VAE observations diverged"
        )
    identity = _canonical_digest(
        {
            "input_image_digest": semantic_texture.input_image_digest,
            "runtime_config_digest": configuration.runtime_config_digest,
            "semantic_texture_observation_identity": semantic_texture.observation_identity,
            "vae_observation_digest": hf_observation.observation_digest,
        }
    )
    return RuntimeSemanticTextureDetectionObservationResult(
        semantic_texture=semantic_texture,
        hf_observation=hf_observation,
        lf_observation=lf_observation,
        runtime_config_digest=configuration.runtime_config_digest,
        observation_identity=identity,
    )
