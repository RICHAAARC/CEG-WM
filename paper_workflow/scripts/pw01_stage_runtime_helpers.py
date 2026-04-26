"""
File purpose: Internal PW01 runtime helpers for source-event execution.
Module type: Semi-general module
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, cast

from main.cli.run_common import build_seed_audit
from main.diffusion.sd3 import infer_runtime, pipeline_factory
from scripts.notebook_runtime_common import (
    REPO_ROOT,
    compute_file_sha256,
    copy_file,
    ensure_directory,
    load_yaml_mapping,
    normalize_path_value,
    run_command_with_logs,
    write_json_atomic,
)


CANONICAL_SOURCE_POOL_ATTESTATION_RELATIVE_ROOT = "artifacts/pw01_canonical_source_pool/attestation"
CANONICAL_SOURCE_POOL_SOURCE_IMAGES_RELATIVE_ROOT = "artifacts/pw01_canonical_source_pool/source_images"
CANONICAL_SOURCE_POOL_PREVIEW_RECORDS_RELATIVE_ROOT = "artifacts/pw01_canonical_source_pool/preview_generation_records"
PREVIEW_GENERATION_RECORD_FILE_NAME = "preview_generation_record.json"
PW01_STAGE_NAME = "PW01_Source_Event_Shards"


def _format_override_arg(arg_name: str, value: Any) -> str:
    """
    Format one CLI override token.

    Args:
        arg_name: Override name.
        value: Override value.

    Returns:
        Formatted override token.
    """
    if not isinstance(arg_name, str) or not arg_name:
        raise TypeError("arg_name must be non-empty str")
    return f"{arg_name}={json.dumps(value, ensure_ascii=False)}"


def _build_stage_override_items(
    stage_name: str,
    extra_overrides: Optional[Sequence[str]] = None,
) -> list[str]:
    """
    Build shared raw override items for one PW01 substage.

    Args:
        stage_name: Workflow stage name.
        extra_overrides: Optional extra raw override items.

    Returns:
        Flat raw override item list containing only key=value entries.
    """
    if not isinstance(stage_name, str) or not stage_name:
        raise TypeError("stage_name must be non-empty str")

    override_items = [
        _format_override_arg("run_root_reuse_allowed", True),
        _format_override_arg("run_root_reuse_reason", f"paper_workflow_pw01_{stage_name}"),
    ]
    if extra_overrides is not None:
        override_items.extend(str(item) for item in extra_overrides)
    return override_items


def _build_stage_overrides(stage_name: str, extra_overrides: Optional[Sequence[str]] = None) -> list[str]:
    """
    Build shared CLI overrides for one PW01 substage.

    Args:
        stage_name: Workflow stage name.
        extra_overrides: Optional extra override tokens.

    Returns:
        Flat CLI override argument list.
    """
    if not isinstance(stage_name, str) or not stage_name:
        raise TypeError("stage_name must be non-empty str")

    override_items = _build_stage_override_items(stage_name, extra_overrides)

    command_args: list[str] = []
    for override_arg in override_items:
        command_args.extend(["--override", override_arg])
    return command_args


def _build_stage_command(
    stage_name: str,
    config_path: Path,
    run_root: Path,
    extra_overrides: Optional[Sequence[str]] = None,
) -> list[str]:
    """
    Build the command for one workflow stage.

    Args:
        stage_name: Workflow stage name.
        config_path: Runtime config path.
        run_root: Workflow run root.
        extra_overrides: Optional extra overrides.

    Returns:
        Command token list.
    """
    if stage_name not in {"embed", "detect", "calibrate", "evaluate"}:
        raise ValueError(f"unsupported stage_name: {stage_name}")
    command = [
        sys.executable,
        "-m",
        f"main.cli.run_{stage_name}",
        "--out",
        str(run_root),
        "--config",
        str(config_path),
    ]
    if stage_name == "detect":
        command.extend(["--input", str(run_root / "records" / "embed_record.json")])
    command.extend(_build_stage_overrides(stage_name, extra_overrides))
    return command


def build_preview_runtime_session(config_path: Path | str) -> Dict[str, Any]:
    """
    Build reusable static preview runtime state for multiple PW01 preview generations.

    Args:
        config_path: Notebook-bound config snapshot path.

    Returns:
        Mapping containing reusable preview pipeline state.
    """
    if isinstance(config_path, Path):
        resolved_config_path = config_path.expanduser().resolve()
    elif isinstance(config_path, str) and config_path.strip():
        resolved_config_path = Path(config_path).expanduser().resolve()
    else:
        raise TypeError("config_path must be Path or non-empty str")

    cfg_obj = load_yaml_mapping(resolved_config_path)
    if not isinstance(cfg_obj, dict):
        raise TypeError("preview config must be dict")

    _resolve_source_pool_preview_generation_cfg(cfg_obj)
    existing_input_image_path = _resolve_source_pool_existing_input_image_path(cfg_obj)
    pipeline_result: Dict[str, Any] | None = None
    if existing_input_image_path is None:
        pipeline_result = pipeline_factory.build_pipeline_shell(cfg_obj)

    return {
        "config_path": normalize_path_value(resolved_config_path),
        "existing_input_image_path": existing_input_image_path,
        "pipeline_result": pipeline_result,
    }


def _build_prompt_sha256(prompt_text: str) -> str:
    """
    Compute the SHA256 digest of one prompt text.

    Args:
        prompt_text: Prompt text.

    Returns:
        Lowercase hexadecimal SHA256 digest.
    """
    if not isinstance(prompt_text, str) or not prompt_text:
        raise TypeError("prompt_text must be non-empty str")
    return hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()


def _normalize_direct_detect_payload(
    payload: Dict[str, Any],
    *,
    prompt_text: str,
    prompt_index: int,
    prompt_file_path: str,
    record_usage: str,
) -> Dict[str, Any]:
    """
    Normalize one direct detect record emitted by PW01.

    Args:
        payload: Original detect record payload.
        prompt_text: Prompt text.
        prompt_index: Prompt index.
        prompt_file_path: Normalized prompt file path.
        record_usage: Direct source usage marker.

    Returns:
        Normalized detect record payload.
    """
    if not isinstance(prompt_text, str) or not prompt_text:
        raise TypeError("prompt_text must be non-empty str")
    if not isinstance(prompt_index, int) or isinstance(prompt_index, bool) or prompt_index < 0:
        raise TypeError("prompt_index must be non-negative int")
    if not isinstance(prompt_file_path, str) or not prompt_file_path:
        raise TypeError("prompt_file_path must be non-empty str")
    if not isinstance(record_usage, str) or not record_usage:
        raise TypeError("record_usage must be non-empty str")

    normalized_payload = copy.deepcopy(payload)
    normalized_payload["label"] = True
    normalized_payload["ground_truth"] = True
    normalized_payload["is_watermarked"] = True
    normalized_payload["ground_truth_source"] = "paper_workflow_source_event_shard"
    normalized_payload["inference_prompt"] = prompt_text
    normalized_payload["pw01_source_pool"] = {
        "record_origin": "direct_source_record",
        "record_usage": record_usage,
        "prompt_index": prompt_index,
        "prompt_text": prompt_text,
        "prompt_sha256": _build_prompt_sha256(prompt_text),
        "prompt_file": prompt_file_path,
    }
    return normalized_payload


def _resolve_source_pool_preview_generation_cfg(cfg_obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Resolve the preview-generation config required by one PW01 subrun.

    Args:
        cfg_obj: Prompt-bound runtime config mapping.

    Returns:
        Canonical preview-generation config mapping.
    """
    if not isinstance(cfg_obj, dict):
        raise TypeError("cfg_obj must be dict")

    embed_cfg = cfg_obj.get("embed")
    if not isinstance(embed_cfg, dict):
        raise ValueError("PW01 preview requires cfg.embed mapping")

    preview_cfg = embed_cfg.get("preview_generation")
    if not isinstance(preview_cfg, dict):
        raise ValueError("PW01 preview requires embed.preview_generation mapping")

    if preview_cfg.get("enabled") is not True:
        raise ValueError("PW01 preview requires embed.preview_generation.enabled=true")

    artifact_rel_path = preview_cfg.get("artifact_rel_path")
    if not isinstance(artifact_rel_path, str) or not artifact_rel_path.strip():
        raise ValueError("PW01 preview requires non-empty embed.preview_generation.artifact_rel_path")

    return {"artifact_rel_path": artifact_rel_path.strip().replace("\\", "/")}


def _resolve_source_pool_preview_generation_record_rel_path(artifact_rel_path: str) -> str:
    """
    Derive the preview-generation record path from the configured preview artifact path.

    Args:
        artifact_rel_path: Artifact path relative to prompt_run_root/artifacts.

    Returns:
        Record path relative to prompt_run_root/artifacts.
    """
    if not isinstance(artifact_rel_path, str) or not artifact_rel_path:
        raise TypeError("artifact_rel_path must be non-empty str")

    preview_rel_path = Path(artifact_rel_path)
    preview_parent = preview_rel_path.parent.as_posix()
    if preview_parent in {"", "."}:
        return PREVIEW_GENERATION_RECORD_FILE_NAME
    return f"{preview_parent}/{PREVIEW_GENERATION_RECORD_FILE_NAME}"


def _resolve_source_pool_existing_input_image_path(cfg_obj: Dict[str, Any]) -> Optional[str]:
    """
    Resolve an explicit embed input image path from one prompt-bound runtime config.

    Args:
        cfg_obj: Prompt-bound runtime config mapping.

    Returns:
        Input image path string when configured; otherwise None.
    """
    if not isinstance(cfg_obj, dict):
        raise TypeError("cfg_obj must be dict")

    candidates = [cfg_obj.get("input_image_path")]
    embed_cfg = cfg_obj.get("embed")
    if isinstance(embed_cfg, dict):
        candidates.append(embed_cfg.get("input_image_path"))

    for value in candidates:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _inject_source_pool_input_image_path(
    cfg_obj: Dict[str, Any],
    *,
    input_image_path: str,
    preview_record_path: str,
    preview_record_rel_path: str,
    artifact_rel_path: str,
    creation_mode: str,
) -> Dict[str, Any]:
    """
    Inject the authoritative input image path and preview lineage references into the runtime config.

    Args:
        cfg_obj: Prompt-bound runtime config mapping.
        input_image_path: Persisted authoritative input image path.
        preview_record_path: Preview-generation record path.
        preview_record_rel_path: Preview-generation record path relative to artifacts.
        artifact_rel_path: Preview artifact path relative to artifacts.
        creation_mode: Preview creation mode token.

    Returns:
        Updated runtime config mapping.
    """
    if not isinstance(cfg_obj, dict):
        raise TypeError("cfg_obj must be dict")
    if not isinstance(input_image_path, str) or not input_image_path:
        raise TypeError("input_image_path must be non-empty str")
    if not isinstance(preview_record_path, str) or not preview_record_path:
        raise TypeError("preview_record_path must be non-empty str")
    if not isinstance(preview_record_rel_path, str) or not preview_record_rel_path:
        raise TypeError("preview_record_rel_path must be non-empty str")
    if not isinstance(artifact_rel_path, str) or not artifact_rel_path:
        raise TypeError("artifact_rel_path must be non-empty str")
    if not isinstance(creation_mode, str) or not creation_mode:
        raise TypeError("creation_mode must be non-empty str")

    cfg_copy = json.loads(json.dumps(cfg_obj))
    embed_cfg = dict(cfg_copy.get("embed")) if isinstance(cfg_copy.get("embed"), dict) else {}
    embed_cfg["input_image_path"] = input_image_path
    cfg_copy["embed"] = embed_cfg
    cfg_copy["pw01_source_pool_preview"] = {
        "input_image_path": input_image_path,
        "preview_generation_record_path": preview_record_path,
        "preview_generation_record_rel_path": preview_record_rel_path,
        "requested_artifact_rel_path": artifact_rel_path,
        "creation_mode": creation_mode,
    }
    return cfg_copy


def _persist_source_pool_preview_artifact(*, source_path: Path, target_path: Path) -> None:
    """
    Copy an existing authoritative input image into the prompt-scoped preview artifact path.

    Args:
        source_path: Existing source image path.
        target_path: Target artifact path under prompt_run_root/artifacts.

    Returns:
        None.
    """
    if not isinstance(source_path, Path):
        raise TypeError("source_path must be Path")
    if not isinstance(target_path, Path):
        raise TypeError("target_path must be Path")
    if not source_path.exists() or not source_path.is_file():
        raise FileNotFoundError(f"source_path not found: {source_path}")

    ensure_directory(target_path.parent)
    if source_path.resolve() != target_path.resolve():
        copy_file(source_path, target_path)


def _prepare_source_pool_preview_artifact_impl(
    *,
    cfg_obj: Dict[str, Any],
    prompt_run_root: Path,
    prompt_text: str,
    prompt_index: int,
    prompt_file_path: str,
    pipeline_result: Mapping[str, Any] | None,
    allow_pipeline_build: bool,
) -> Dict[str, Any]:
    """
    Prepare the authoritative prompt-conditioned preview artifact consumed by PW01.

    Args:
        cfg_obj: Prompt-bound runtime config mapping.
        prompt_run_root: Prompt-bound subrun root.
        prompt_text: Prompt text.
        prompt_index: Prompt index.
        prompt_file_path: Normalized prompt file path.
        pipeline_result: Optional reusable pipeline result mapping.
        allow_pipeline_build: Whether per-call pipeline construction is allowed.

    Returns:
        Mapping carrying the updated runtime config and preview-generation record payload.
    """
    if not isinstance(cfg_obj, dict):
        raise TypeError("cfg_obj must be dict")
    if not isinstance(prompt_run_root, Path):
        raise TypeError("prompt_run_root must be Path")
    if not isinstance(prompt_text, str) or not prompt_text:
        raise TypeError("prompt_text must be non-empty str")
    if not isinstance(prompt_index, int) or isinstance(prompt_index, bool) or prompt_index < 0:
        raise TypeError("prompt_index must be non-negative int")
    if not isinstance(prompt_file_path, str) or not prompt_file_path:
        raise TypeError("prompt_file_path must be non-empty str")
    if not isinstance(allow_pipeline_build, bool):
        raise TypeError("allow_pipeline_build must be bool")
    if pipeline_result is not None and not isinstance(pipeline_result, Mapping):
        raise TypeError("pipeline_result must be Mapping or None")

    preview_cfg = _resolve_source_pool_preview_generation_cfg(cfg_obj)
    artifact_rel_path = preview_cfg["artifact_rel_path"]
    record_rel_path = _resolve_source_pool_preview_generation_record_rel_path(artifact_rel_path)
    artifact_path = prompt_run_root / "artifacts" / Path(artifact_rel_path)
    record_path = prompt_run_root / "artifacts" / Path(record_rel_path)
    _, seed_digest, seed_value, seed_rule_id = build_seed_audit(cfg_obj, "embed")

    preview_record: Dict[str, Any] = {
        "artifact_type": "pw01_source_pool_preview_generation_record",
        "stage_name": PW01_STAGE_NAME,
        "prompt_index": prompt_index,
        "prompt_text": prompt_text,
        "prompt_file": prompt_file_path,
        "prompt_sha256": _build_prompt_sha256(prompt_text),
        "status": "failed",
        "reason": None,
        "exception_type": None,
        "exception_message": None,
        "pipeline_status": "not_required",
        "pipeline_error": None,
        "pipeline_runtime_meta": None,
        "pipeline_provenance_canon_sha256": None,
        "model_provenance_canon_sha256": None,
        "inference_status": "not_required",
        "inference_error": None,
        "inference_runtime_meta": None,
        "output_image_present": False,
        "requested_artifact_rel_path": artifact_rel_path,
        "requested_artifact_path": normalize_path_value(artifact_path),
        "persisted_artifact_path": None,
        "persisted_artifact_sha256": None,
        "record_path": normalize_path_value(record_path),
        "record_rel_path": record_rel_path,
        "seed": seed_value,
        "seed_digest": seed_digest,
        "seed_rule_id": seed_rule_id,
        "prompt": prompt_text,
        "device": cfg_obj.get("device", "cpu"),
        "model_id": cfg_obj.get("model_id"),
        "inference_args": {
            "num_inference_steps": cfg_obj.get("inference_num_steps"),
            "guidance_scale": cfg_obj.get("inference_guidance_scale"),
            "height": cfg_obj.get("inference_height"),
            "width": cfg_obj.get("inference_width"),
        },
        "creation_mode": "prompt_conditioned_preview",
    }

    existing_input_path = _resolve_source_pool_existing_input_image_path(cfg_obj)
    if isinstance(existing_input_path, str) and existing_input_path:
        resolved_existing_input = Path(existing_input_path).expanduser()
        if not resolved_existing_input.is_absolute():
            resolved_existing_input = (REPO_ROOT / resolved_existing_input).resolve()
        else:
            resolved_existing_input = resolved_existing_input.resolve()
        if not resolved_existing_input.exists() or not resolved_existing_input.is_file():
            raise FileNotFoundError(
                "PW01 preconfigured input image not found: "
                f"{resolved_existing_input}"
            )

        _persist_source_pool_preview_artifact(source_path=resolved_existing_input, target_path=artifact_path)
        preview_record["status"] = "ok"
        preview_record["reason"] = None
        preview_record["output_image_present"] = True
        preview_record["persisted_artifact_path"] = normalize_path_value(artifact_path)
        preview_record["persisted_artifact_sha256"] = compute_file_sha256(artifact_path)
        preview_record["creation_mode"] = "preconfigured_input_image"
        preview_record["source_input_image_path"] = normalize_path_value(resolved_existing_input)
        write_json_atomic(record_path, preview_record)
        return {
            "runtime_cfg": _inject_source_pool_input_image_path(
                cfg_obj,
                input_image_path=normalize_path_value(artifact_path),
                preview_record_path=normalize_path_value(record_path),
                preview_record_rel_path=record_rel_path,
                artifact_rel_path=artifact_rel_path,
                creation_mode=cast(str, preview_record["creation_mode"]),
            ),
            "preview_record": preview_record,
        }

    resolved_pipeline_result: Dict[str, Any]
    if pipeline_result is None:
        if not allow_pipeline_build:
            raise ValueError("preview runtime_session missing reusable pipeline_result")
        resolved_pipeline_result = pipeline_factory.build_pipeline_shell(cfg_obj)
    else:
        resolved_pipeline_result = dict(cast(Mapping[str, Any], pipeline_result))

    preview_record["pipeline_status"] = resolved_pipeline_result.get("pipeline_status")
    preview_record["pipeline_error"] = resolved_pipeline_result.get("pipeline_error")
    preview_record["pipeline_runtime_meta"] = resolved_pipeline_result.get("pipeline_runtime_meta")
    preview_record["pipeline_provenance_canon_sha256"] = resolved_pipeline_result.get("pipeline_provenance_canon_sha256")
    preview_record["model_provenance_canon_sha256"] = resolved_pipeline_result.get("model_provenance_canon_sha256")
    preview_pipeline_obj = resolved_pipeline_result.get("pipeline_obj")
    preview_device = cfg_obj.get("device", "cpu")

    try:
        infer_result = infer_runtime.run_sd3_inference(
            cfg_obj,
            preview_pipeline_obj,
            preview_device,
            seed_value,
            runtime_phase_label="preview_generation",
            injection_context=None,
            injection_modifier=None,
            capture_final_latents=False,
        )
        preview_status = infer_result.get("inference_status")
        if not isinstance(preview_status, str) or not preview_status:
            preview_status = infer_result.get("status")
        if not isinstance(preview_status, str) or not preview_status:
            preview_status = infer_runtime.INFERENCE_STATUS_FAILED

        preview_record["inference_status"] = preview_status
        preview_record["inference_error"] = infer_result.get("inference_error")
        preview_record["inference_runtime_meta"] = infer_result.get("inference_runtime_meta")

        preview_image = infer_result.get("output_image")
        preview_record["output_image_present"] = preview_image is not None
        if preview_status == infer_runtime.INFERENCE_STATUS_OK and preview_image is not None:
            ensure_directory(artifact_path.parent)
            preview_image.save(artifact_path)
            preview_record["status"] = "ok"
            preview_record["persisted_artifact_path"] = normalize_path_value(artifact_path)
            preview_record["persisted_artifact_sha256"] = compute_file_sha256(artifact_path)
        elif preview_status == infer_runtime.INFERENCE_STATUS_OK:
            preview_record["reason"] = "preview_inference_no_output_image"
        else:
            inference_error = infer_result.get("inference_error")
            if isinstance(inference_error, str) and inference_error:
                preview_record["reason"] = inference_error
            else:
                preview_record["reason"] = f"preview_inference_status={preview_status}"
    except Exception as exc:
        preview_record["reason"] = "source_pool_preview_generation_exception"
        preview_record["exception_type"] = type(exc).__name__
        preview_record["exception_message"] = str(exc)

    write_json_atomic(record_path, preview_record)
    if preview_record["status"] != "ok":
        raise RuntimeError(
            "PW01 source pool preview generation failed: "
            f"{json.dumps(preview_record, ensure_ascii=False, sort_keys=True)}"
        )

    persisted_artifact_path = cast(str, preview_record["persisted_artifact_path"])
    return {
        "runtime_cfg": _inject_source_pool_input_image_path(
            cfg_obj,
            input_image_path=persisted_artifact_path,
            preview_record_path=normalize_path_value(record_path),
            preview_record_rel_path=record_rel_path,
            artifact_rel_path=artifact_rel_path,
            creation_mode=cast(str, preview_record["creation_mode"]),
        ),
        "preview_record": preview_record,
    }


def _prepare_source_pool_preview_artifact(
    *,
    cfg_obj: Dict[str, Any],
    prompt_run_root: Path,
    prompt_text: str,
    prompt_index: int,
    prompt_file_path: str,
) -> Dict[str, Any]:
    """
    Prepare the authoritative prompt-conditioned preview artifact consumed by PW01.

    Args:
        cfg_obj: Prompt-bound runtime config mapping.
        prompt_run_root: Prompt-bound subrun root.
        prompt_text: Prompt text.
        prompt_index: Prompt index.
        prompt_file_path: Normalized prompt file path.

    Returns:
        Mapping carrying the updated runtime config and preview-generation record payload.
    """
    return _prepare_source_pool_preview_artifact_impl(
        cfg_obj=cfg_obj,
        prompt_run_root=prompt_run_root,
        prompt_text=prompt_text,
        prompt_index=prompt_index,
        prompt_file_path=prompt_file_path,
        pipeline_result=None,
        allow_pipeline_build=True,
    )


def prepare_source_pool_preview_artifact_with_runtime(
    *,
    cfg_obj: Dict[str, Any],
    prompt_run_root: Path,
    prompt_text: str,
    prompt_index: int,
    prompt_file_path: str,
    runtime_session: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    Prepare one PW01 preview artifact using a reusable worker-local preview runtime.

    Args:
        cfg_obj: Prompt-bound runtime config mapping.
        prompt_run_root: Prompt-bound subrun root.
        prompt_text: Prompt text.
        prompt_index: Prompt index.
        prompt_file_path: Normalized prompt file path.
        runtime_session: Preview runtime session mapping.

    Returns:
        Mapping carrying the updated runtime config and preview-generation record payload.
    """
    if not isinstance(runtime_session, Mapping):
        raise TypeError("runtime_session must be Mapping")

    return _prepare_source_pool_preview_artifact_impl(
        cfg_obj=cfg_obj,
        prompt_run_root=prompt_run_root,
        prompt_text=prompt_text,
        prompt_index=prompt_index,
        prompt_file_path=prompt_file_path,
        pipeline_result=cast(Mapping[str, Any] | None, runtime_session.get("pipeline_result")),
        allow_pipeline_build=False,
    )


def _build_prompt_scoped_package_relative_path(relative_root: str, index: int, file_name: str) -> str:
    """
    Build the package-relative path for one prompt-scoped canonical source artifact.

    Args:
        relative_root: Root package-relative directory.
        index: Prompt index.
        file_name: Artifact file name.

    Returns:
        Prompt-scoped package-relative path.
    """
    if not isinstance(relative_root, str) or not relative_root:
        raise TypeError("relative_root must be non-empty str")
    if not isinstance(index, int) or isinstance(index, bool) or index < 0:
        raise TypeError("index must be non-negative int")
    if not isinstance(file_name, str) or not file_name:
        raise TypeError("file_name must be non-empty str")
    return f"{relative_root}/prompt_{index:03d}/{file_name}"


def _build_optional_canonical_artifact_view(
    *,
    run_root: Path,
    source_path: Optional[Path],
    package_relative_path: Optional[str],
    missing_reason: str,
) -> Dict[str, Any]:
    """
    Build an explicit artifact view for an optional canonical source artifact.

    Args:
        run_root: Stage run root.
        source_path: Source artifact path when discoverable.
        package_relative_path: Canonical package-relative target path.
        missing_reason: Stable missing-state reason.

    Returns:
        Artifact view carrying existence, path, and package-relative metadata.
    """
    if not isinstance(run_root, Path):
        raise TypeError("run_root must be Path")
    if source_path is not None and not isinstance(source_path, Path):
        raise TypeError("source_path must be Path or None")
    if package_relative_path is not None and (
        not isinstance(package_relative_path, str) or not package_relative_path
    ):
        raise TypeError("package_relative_path must be non-empty str or None")
    if not isinstance(missing_reason, str) or not missing_reason:
        raise TypeError("missing_reason must be non-empty str")

    staged_path: Optional[Path] = None
    if isinstance(package_relative_path, str):
        staged_path = run_root / package_relative_path

    if isinstance(source_path, Path) and source_path.exists() and source_path.is_file():
        if staged_path is None:
            raise ValueError("package_relative_path is required when source artifact exists")
        copy_file(source_path, staged_path)
        return {
            "exists": True,
            "path": normalize_path_value(staged_path),
            "package_relative_path": package_relative_path,
            "missing_reason": None,
        }

    return {
        "exists": False,
        "path": normalize_path_value(staged_path) if isinstance(staged_path, Path) else None,
        "package_relative_path": package_relative_path,
        "missing_reason": missing_reason,
    }


def _resolve_source_pool_attestation_views(
    *,
    cfg_obj: Dict[str, Any],
    run_root: Path,
    prompt_run_root: Path,
    prompt_index: int,
) -> Dict[str, Dict[str, Any]]:
    """
    Resolve prompt-scoped attestation artifact views for PW01.

    Args:
        cfg_obj: Base runtime config mapping.
        run_root: Stage run root.
        prompt_run_root: Prompt-bound subrun root.
        prompt_index: Prompt index.

    Returns:
        Mapping from canonical attestation artifact name to artifact view.
    """
    if not isinstance(cfg_obj, dict):
        raise TypeError("cfg_obj must be dict")
    if not isinstance(run_root, Path):
        raise TypeError("run_root must be Path")
    if not isinstance(prompt_run_root, Path):
        raise TypeError("prompt_run_root must be Path")
    if not isinstance(prompt_index, int) or isinstance(prompt_index, bool) or prompt_index < 0:
        raise TypeError("prompt_index must be non-negative int")

    attestation_cfg = cfg_obj.get("attestation")
    attestation_enabled = (
        isinstance(attestation_cfg, dict)
        and isinstance(attestation_cfg.get("enabled"), bool)
        and attestation_cfg.get("enabled") is True
    )
    attestation_dir = prompt_run_root / "artifacts" / "attestation"
    artifact_specs = {
        "attestation_statement": ("attestation_statement.json", "attestation_statement_not_emitted"),
        "attestation_bundle": ("attestation_bundle.json", "attestation_bundle_not_emitted"),
        "attestation_result": ("attestation_result.json", "attestation_result_not_emitted"),
    }

    artifact_views: Dict[str, Dict[str, Any]] = {}
    for artifact_key, (file_name, missing_reason) in artifact_specs.items():
        package_relative_path = _build_prompt_scoped_package_relative_path(
            CANONICAL_SOURCE_POOL_ATTESTATION_RELATIVE_ROOT,
            prompt_index,
            file_name,
        )
        artifact_views[artifact_key] = _build_optional_canonical_artifact_view(
            run_root=run_root,
            source_path=(attestation_dir / file_name) if attestation_enabled else None,
            package_relative_path=package_relative_path,
            missing_reason="attestation_disabled" if not attestation_enabled else missing_reason,
        )
    return artifact_views


def _resolve_source_pool_source_image_view(
    *,
    cfg_obj: Dict[str, Any],
    run_root: Path,
    prompt_run_root: Path,
    prompt_index: int,
) -> Dict[str, Any]:
    """
    Resolve the source-image view for one prompt-bound PW01 subrun.

    Args:
        cfg_obj: Base runtime config mapping.
        run_root: Stage run root.
        prompt_run_root: Prompt-bound subrun root.
        prompt_index: Prompt index.

    Returns:
        Source-image artifact view with explicit missing-state semantics.
    """
    if not isinstance(cfg_obj, dict):
        raise TypeError("cfg_obj must be dict")
    if not isinstance(run_root, Path):
        raise TypeError("run_root must be Path")
    if not isinstance(prompt_run_root, Path):
        raise TypeError("prompt_run_root must be Path")
    if not isinstance(prompt_index, int) or isinstance(prompt_index, bool) or prompt_index < 0:
        raise TypeError("prompt_index must be non-negative int")

    embed_cfg = cfg_obj.get("embed")
    if not isinstance(embed_cfg, dict):
        return _build_optional_canonical_artifact_view(
            run_root=run_root,
            source_path=None,
            package_relative_path=None,
            missing_reason="preview_generation_config_missing",
        )
    preview_cfg = embed_cfg.get("preview_generation")
    if not isinstance(preview_cfg, dict):
        return _build_optional_canonical_artifact_view(
            run_root=run_root,
            source_path=None,
            package_relative_path=None,
            missing_reason="preview_generation_config_missing",
        )

    preview_enabled = isinstance(preview_cfg.get("enabled"), bool) and preview_cfg.get("enabled") is True
    if not preview_enabled:
        return _build_optional_canonical_artifact_view(
            run_root=run_root,
            source_path=None,
            package_relative_path=None,
            missing_reason="preview_generation_disabled",
        )

    artifact_rel_path = preview_cfg.get("artifact_rel_path")
    if not isinstance(artifact_rel_path, str) or not artifact_rel_path.strip():
        return _build_optional_canonical_artifact_view(
            run_root=run_root,
            source_path=None,
            package_relative_path=None,
            missing_reason="preview_generation_artifact_rel_path_missing",
        )

    preview_rel_path = Path(artifact_rel_path.strip().replace("\\", "/"))
    package_relative_path = _build_prompt_scoped_package_relative_path(
        CANONICAL_SOURCE_POOL_SOURCE_IMAGES_RELATIVE_ROOT,
        prompt_index,
        preview_rel_path.name,
    )
    return _build_optional_canonical_artifact_view(
        run_root=run_root,
        source_path=prompt_run_root / "artifacts" / preview_rel_path,
        package_relative_path=package_relative_path,
        missing_reason="source_image_not_emitted",
    )


def _resolve_source_pool_preview_generation_record_view(
    *,
    cfg_obj: Dict[str, Any],
    run_root: Path,
    prompt_run_root: Path,
    prompt_index: int,
) -> Dict[str, Any]:
    """
    Resolve the prompt-scoped preview-generation record view for one PW01 subrun.

    Args:
        cfg_obj: Prompt-bound runtime config mapping.
        run_root: Stage run root.
        prompt_run_root: Prompt-bound subrun root.
        prompt_index: Prompt index.

    Returns:
        Preview-generation record artifact view.
    """
    if not isinstance(cfg_obj, dict):
        raise TypeError("cfg_obj must be dict")
    if not isinstance(run_root, Path):
        raise TypeError("run_root must be Path")
    if not isinstance(prompt_run_root, Path):
        raise TypeError("prompt_run_root must be Path")
    if not isinstance(prompt_index, int) or isinstance(prompt_index, bool) or prompt_index < 0:
        raise TypeError("prompt_index must be non-negative int")

    preview_cfg = _resolve_source_pool_preview_generation_cfg(cfg_obj)
    record_rel_path = _resolve_source_pool_preview_generation_record_rel_path(preview_cfg["artifact_rel_path"])
    package_relative_path = _build_prompt_scoped_package_relative_path(
        CANONICAL_SOURCE_POOL_PREVIEW_RECORDS_RELATIVE_ROOT,
        prompt_index,
        PREVIEW_GENERATION_RECORD_FILE_NAME,
    )
    return _build_optional_canonical_artifact_view(
        run_root=run_root,
        source_path=prompt_run_root / "artifacts" / Path(record_rel_path),
        package_relative_path=package_relative_path,
        missing_reason="preview_generation_record_not_emitted",
    )


def _run_stage(stage_name: str, command: Sequence[str], run_root: Path) -> Dict[str, Any]:
    """
    Run one workflow stage and persist stdout/stderr logs.

    Args:
        stage_name: Workflow stage name.
        command: Command token list.
        run_root: Stage run root.

    Returns:
        Stage execution summary.
    """
    if not isinstance(stage_name, str) or not stage_name:
        raise TypeError("stage_name must be non-empty str")
    if not isinstance(run_root, Path):
        raise TypeError("run_root must be Path")

    logs_dir = ensure_directory(run_root / "logs")
    result = run_command_with_logs(
        command=command,
        cwd=REPO_ROOT,
        stdout_log_path=logs_dir / f"{stage_name}_stdout.log",
        stderr_log_path=logs_dir / f"{stage_name}_stderr.log",
    )
    result["status"] = "ok" if result.get("return_code") == 0 else "failed"
    return result