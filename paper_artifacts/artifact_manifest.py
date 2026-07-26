"""构造论文产物 manifest, 该模块属于产物生成层而非核心方法层。"""

from __future__ import annotations

from typing import Any

from paper_artifacts.digest import build_stable_digest
from paper_artifacts.manifests import ArtifactManifest


def build_artifact_manifest(
    artifact_id: str,
    artifact_type: str,
    input_paths: tuple[str, ...],
    output_paths: tuple[str, ...],
    config: dict[str, Any],
    code_version: str,
    rebuild_command: str,
    metadata: dict[str, Any] | None = None,
) -> ArtifactManifest:
    """根据输入、输出和配置构造产物 manifest。"""
    return ArtifactManifest(
        artifact_id=artifact_id,
        artifact_type=artifact_type,
        input_paths=input_paths,
        output_paths=output_paths,
        config_digest=build_stable_digest(config),
        code_version=code_version,
        rebuild_command=rebuild_command,
        metadata=metadata or {},
    )
