"""定义受治理论文产物 manifest 的最小通用结构。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ArtifactManifest:
    """记录一个论文候选产物的 provenance。"""

    artifact_id: str
    artifact_type: str
    input_paths: tuple[str, ...]
    output_paths: tuple[str, ...]
    config_digest: str
    code_version: str
    rebuild_command: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """转为普通字典, 便于写入 manifest JSON 文件。"""
        return asdict(self)


REQUIRED_MANIFEST_FIELDS = (
    "artifact_id",
    "artifact_type",
    "input_paths",
    "output_paths",
    "config_digest",
    "code_version",
    "rebuild_command",
)


def validate_manifest(manifest: dict[str, Any]) -> list[str]:
    """返回缺失的最小 manifest 字段列表。"""
    return [field_name for field_name in REQUIRED_MANIFEST_FIELDS if field_name not in manifest]
