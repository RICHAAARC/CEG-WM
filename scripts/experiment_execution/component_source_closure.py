"""Deterministic source closures for reviewed HF and LF science identities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
from typing import Mapping, Sequence


LF_DIRECTIONAL_COMPONENT_IDS = (
    "key_schedule",
    "lf_carrier",
    "content_embedder",
    "lf_detector",
)
HF_REFERENCE_COMPONENT_IDS = (
    "key_schedule",
    "hf_carrier",
    "content_embedder",
    "hf_detector",
    "content_detector",
)
_LF_CANDIDATE_SOURCE_DEPENDENCIES = {
    "lf_detector": (
        (
            "main/content_chain/lf_whitening.py",
            "LfNullWhiteningAsset",
            "candidate_public_asset_contract",
        ),
    ),
}


class ComponentSourceClosureError(ValueError):
    """The reviewed component mapping cannot form a frozen source closure."""


def canonical_digest(value: object) -> str:
    return sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class ComponentSourceBinding:
    component_id: str
    source_role: str
    implementation_path: str
    implementation_symbol: str
    source_sha256: str


@dataclass(frozen=True, slots=True)
class ComponentSourceClosure:
    ordered_component_ids: tuple[str, ...]
    source_bindings: tuple[ComponentSourceBinding, ...]
    component_implementation_digest: str

    def canonical_payload(self) -> dict[str, object]:
        return {
            "ordered_component_ids": list(self.ordered_component_ids),
            "source_bindings": [asdict(binding) for binding in self.source_bindings],
        }


def _validated_source_binding(
    *,
    component_id: str,
    source_role: str,
    implementation_path: str,
    implementation_symbol: str,
    repository_root: Path,
) -> ComponentSourceBinding:
    relative = PurePosixPath(implementation_path)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or not relative.parts
        or relative.parts[0] != "main"
        or relative.name == "__init__.py"
        or relative.suffix != ".py"
    ):
        raise ComponentSourceClosureError(
            "component source must be a concrete Python implementation under main"
        )
    if type(implementation_symbol) is not str or not implementation_symbol:
        raise ComponentSourceClosureError("component implementation symbol is invalid")
    source_path = repository_root.joinpath(*relative.parts)
    try:
        source_bytes = source_path.read_bytes()
    except OSError as exc:
        raise ComponentSourceClosureError(
            "component source is missing or unreadable"
        ) from exc
    if not source_bytes:
        raise ComponentSourceClosureError("component source is empty")
    return ComponentSourceBinding(
        component_id=component_id,
        source_role=source_role,
        implementation_path=implementation_path,
        implementation_symbol=implementation_symbol,
        source_sha256=sha256(source_bytes).hexdigest(),
    )


def build_component_source_closure(
    ordered_component_ids: Sequence[str],
    reviewed_components: Mapping[str, Mapping[str, object]],
    repository_root: str | Path,
) -> ComponentSourceClosure:
    """Bind one registered component set; exclude delivery/import closure."""

    component_ids = tuple(ordered_component_ids)
    if component_ids == LF_DIRECTIONAL_COMPONENT_IDS:
        candidate_dependencies = _LF_CANDIDATE_SOURCE_DEPENDENCIES
    elif component_ids == HF_REFERENCE_COMPONENT_IDS:
        candidate_dependencies = {}
    else:
        raise ComponentSourceClosureError(
            "component closure order or membership drifted"
        )
    if len(set(component_ids)) != len(component_ids):
        raise ComponentSourceClosureError("component identities must be unique")
    root = Path(repository_root)
    bindings: list[ComponentSourceBinding] = []
    for component_id in component_ids:
        raw = reviewed_components.get(component_id)
        if not isinstance(raw, Mapping):
            raise ComponentSourceClosureError(
                "reviewed component mapping is incomplete"
            )
        implementation_path = raw.get("implementation_path")
        implementation_symbol = raw.get("implementation_symbol")
        if type(implementation_path) is not str:
            raise ComponentSourceClosureError("component implementation path is invalid")
        if type(implementation_symbol) is not str:
            raise ComponentSourceClosureError("component implementation symbol is invalid")
        bindings.append(
            _validated_source_binding(
                component_id=component_id,
                source_role="component_implementation",
                implementation_path=implementation_path,
                implementation_symbol=implementation_symbol,
                repository_root=root,
            )
        )
        for dependency_path, dependency_symbol, source_role in (
            candidate_dependencies.get(component_id, ())
        ):
            bindings.append(
                _validated_source_binding(
                    component_id=component_id,
                    source_role=source_role,
                    implementation_path=dependency_path,
                    implementation_symbol=dependency_symbol,
                    repository_root=root,
                )
            )
    payload = {
        "ordered_component_ids": list(component_ids),
        "source_bindings": [asdict(binding) for binding in bindings],
    }
    return ComponentSourceClosure(
        ordered_component_ids=component_ids,
        source_bindings=tuple(bindings),
        component_implementation_digest=canonical_digest(payload),
    )


def build_lf_directional_component_source_closure(
    reviewed_components: Mapping[str, Mapping[str, object]],
    repository_root: str | Path,
) -> ComponentSourceClosure:
    """Bind the reviewed LF directional implementations and public asset."""

    return build_component_source_closure(
        LF_DIRECTIONAL_COMPONENT_IDS,
        reviewed_components,
        repository_root,
    )


def build_hf_reference_component_source_closure(
    reviewed_components: Mapping[str, Mapping[str, object]],
    repository_root: str | Path,
) -> ComponentSourceClosure:
    """Bind only the five reviewed HF scientific component implementations."""

    return build_component_source_closure(
        HF_REFERENCE_COMPONENT_IDS,
        reviewed_components,
        repository_root,
    )


__all__ = [
    "HF_REFERENCE_COMPONENT_IDS",
    "LF_DIRECTIONAL_COMPONENT_IDS",
    "ComponentSourceBinding",
    "ComponentSourceClosure",
    "ComponentSourceClosureError",
    "build_component_source_closure",
    "build_hf_reference_component_source_closure",
    "build_lf_directional_component_source_closure",
    "canonical_digest",
]
