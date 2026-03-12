"""
文件目的：paper_full 机制断言脚本（只读验证）。
Module type: General module

职责边界：
1. 只读取 run_root 与配置，不写入任何 records 或 artifacts。
2. 对论文级机制锚点执行 fail-fast 断言。
3. 不修改冻结语义、digest 口径和门禁行为。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml


PAPER_FROZEN_IMPL_REQUIRED_FIELDS = ("sync_module_id", "geometry_extractor_id")


def _load_json(path: Path) -> Dict[str, Any]:
    """
    功能：读取 JSON 文件并要求根对象为字典。

    Load a JSON file and require dict root.

    Args:
        path: File path.

    Returns:
        Parsed dictionary.

    Raises:
        FileNotFoundError: If file is missing.
        TypeError: If JSON root is not a dict.
        ValueError: If JSON parsing fails.
    """
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"json file not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"failed to parse json: {path}: {type(exc).__name__}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"json root must be dict: {path}")
    return payload


def _load_yaml(path: Path) -> Dict[str, Any]:
    """
    功能：读取 YAML 文件并要求根对象为字典。

    Load a YAML file and require dict root.

    Args:
        path: YAML path.

    Returns:
        Parsed dictionary.

    Raises:
        FileNotFoundError: If file is missing.
        ValueError: If YAML root is not dict.
    """
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"yaml file not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"yaml root must be mapping: {path}")
    return payload


def _find_repo_root(start_path: Path) -> Path:
    """
    功能：向上查找包含 configs/runtime_whitelist.yaml 的仓库根目录。

    Locate repository root by searching for runtime_whitelist.yaml.

    Args:
        start_path: Start path for upward search.

    Returns:
        Repository root path.

    Raises:
        FileNotFoundError: If repository root cannot be resolved.
    """
    if not isinstance(start_path, Path):
        raise TypeError("start_path must be Path")
    for candidate in [start_path.resolve(), *start_path.resolve().parents]:
        if (candidate / "configs" / "runtime_whitelist.yaml").exists():
            return candidate
    raise FileNotFoundError("repo_root not found via runtime_whitelist.yaml")


def _load_runtime_whitelist_impl_ids(repo_root: Path) -> Set[str]:
    """
    功能：读取 runtime_whitelist.yaml 中的 impl_id allowlist。

    Load impl_id allowlist from runtime_whitelist.yaml.

    Args:
        repo_root: Repository root path.

    Returns:
        Set of allowed impl_id strings.

    Raises:
        ValueError: If whitelist content is invalid.
    """
    if not isinstance(repo_root, Path):
        raise TypeError("repo_root must be Path")
    whitelist_path = repo_root / "configs" / "runtime_whitelist.yaml"
    payload = _load_yaml(whitelist_path)
    impl_id_cfg = payload.get("impl_id") if isinstance(payload.get("impl_id"), dict) else None
    if not isinstance(impl_id_cfg, dict):
        raise ValueError("runtime_whitelist.impl_id must be dict")
    allowed_flat = impl_id_cfg.get("allowed_flat")
    if not isinstance(allowed_flat, list):
        raise ValueError("runtime_whitelist.impl_id.allowed_flat must be list")
    allowed_by_domain = impl_id_cfg.get("allowed_by_domain")
    if allowed_by_domain is None:
        allowed_by_domain = {}
    if not isinstance(allowed_by_domain, dict):
        raise ValueError("runtime_whitelist.impl_id.allowed_by_domain must be dict when present")
    normalized_ids: Set[str] = set()
    for item in allowed_flat:
        if not isinstance(item, str):
            continue
        normalized_item = item.strip()
        if normalized_item:
            normalized_ids.add(normalized_item)

    for domain_values in allowed_by_domain.values():
        if not isinstance(domain_values, list):
            continue
        for item in domain_values:
            if not isinstance(item, str):
                continue
            normalized_item = item.strip()
            if normalized_item:
                normalized_ids.add(normalized_item)

    return normalized_ids


def _collect_impl_ids_from_cfg(cfg: Dict[str, Any]) -> List[str]:
    """
    功能：从配置 impl 段收集 impl_id。 

    Collect impl_id values from cfg.impl mapping.

    Args:
        cfg: Configuration mapping.

    Returns:
        List of impl_id strings.
    """
    impl_ids: List[str] = []
    impl_cfg = cfg.get("impl") if isinstance(cfg.get("impl"), dict) else {}
    for key, value in impl_cfg.items():
        if not isinstance(key, str):
            continue
        if not key.endswith("_id"):
            continue
        if isinstance(value, str):
            normalized_value = value.strip()
            if normalized_value:
                impl_ids.append(normalized_value)
    return impl_ids


def _load_paper_frozen_impl_constraints(repo_root: Path) -> Dict[str, str]:
    """
    功能：读取 paper_full_cuda 冻结 impl 约束。 

    Load frozen impl constraints from configs/paper_full_cuda.yaml.

    Args:
        repo_root: Repository root path.

    Returns:
        Required impl constraints mapping.

    Raises:
        ValueError: If frozen config content is invalid.
    """
    if not isinstance(repo_root, Path):
        raise TypeError("repo_root must be Path")

    frozen_cfg_path = repo_root / "configs" / "paper_full_cuda.yaml"
    frozen_cfg = _load_yaml(frozen_cfg_path)
    frozen_impl = frozen_cfg.get("impl")
    if not isinstance(frozen_impl, dict):
        raise ValueError("paper_full_cuda.impl must be mapping")

    constraints: Dict[str, str] = {}
    for field_name in PAPER_FROZEN_IMPL_REQUIRED_FIELDS:
        field_value = frozen_impl.get(field_name)
        if not isinstance(field_value, str) or not field_value.strip():
            raise ValueError(f"paper_full_cuda.impl.{field_name} must be non-empty str")
        constraints[field_name] = field_value.strip()
    return constraints


def assert_paper_mechanisms(run_root: Path) -> None:
    """
    功能：在 run_root 上执行 paper_full 机制断言。 

    Assert paper_full mechanisms using artifacts under run_root.

    Args:
        run_root: Unified run_root path.

    Returns:
        None.

    Raises:
        ValueError: If any paper mechanism assertion fails.
    """
    if not isinstance(run_root, Path):
        raise TypeError("run_root must be Path")
    cfg_path = run_root / "artifacts" / "workflow_cfg" / "profile_paper_full_cuda.yaml"
    cfg = _load_yaml(cfg_path)
    embed_record = _load_json(run_root / "records" / "embed_record.json")
    detect_record = _load_json(run_root / "records" / "detect_record.json")
    evaluate_report = _load_json(run_root / "artifacts" / "evaluation_report.json")
    repo_root = _find_repo_root(run_root)

    failures = _assert_paper_mechanisms(
        run_root=run_root,
        cfg=cfg,
        embed_record=embed_record,
        detect_record=detect_record,
        evaluate_report=evaluate_report,
        repo_root=repo_root,
    )
    if failures:
        raise ValueError("paper mechanisms assertion failed")


def _pick_mapping(obj: Any, path_candidates: List[List[str]]) -> Optional[Dict[str, Any]]:
    """
    功能：按候选路径提取首个字典节点。

    Pick first mapping by candidate field paths.

    Args:
        obj: Source object.
        path_candidates: Candidate key paths.

    Returns:
        Found mapping or None.
    """
    if not isinstance(obj, dict):
        return None
    for path in path_candidates:
        current: Any = obj
        ok = True
        for key in path:
            if not isinstance(current, dict) or key not in current:
                ok = False
                break
            current = current[key]
        if ok and isinstance(current, dict):
            return current
    return None


def _pick_str(obj: Any, path_candidates: List[List[str]]) -> Optional[str]:
    """
    功能：按候选路径提取首个非空字符串。

    Pick first non-empty string by candidate field paths.

    Args:
        obj: Source object.
        path_candidates: Candidate key paths.

    Returns:
        Found string or None.
    """
    if not isinstance(obj, dict):
        return None
    for path in path_candidates:
        current: Any = obj
        ok = True
        for key in path:
            if not isinstance(current, dict) or key not in current:
                ok = False
                break
            current = current[key]
        if ok and isinstance(current, str) and current:
            return current
    return None


def _resolve_evaluation_report_payload(evaluate_report: Dict[str, Any]) -> Dict[str, Any]:
    """
    功能：兼容 evaluation_report 的嵌套封装结构。 

    Resolve effective evaluation report payload from possible nested wrapper.

    Args:
        evaluate_report: Loaded evaluation_report.json mapping.

    Returns:
        Effective evaluation report mapping.
    """
    if not isinstance(evaluate_report, dict):
        return {}
    nested = evaluate_report.get("evaluation_report")
    if isinstance(nested, dict):
        return nested
    return evaluate_report


def _is_detect_runtime_fallback(detect_record: Dict[str, Any]) -> bool:
    """
    功能：判定 detect 记录是否处于 fallback 运行模式。 

    Determine whether detect record is in fallback runtime mode.

    Args:
        detect_record: Detect record mapping.

    Returns:
        True when detect runtime is fallback.
    """
    if not isinstance(detect_record, dict):
        return False
    runtime_flag = detect_record.get("detect_runtime_is_fallback")
    if isinstance(runtime_flag, bool):
        return runtime_flag
    runtime_mode = detect_record.get("detect_runtime_mode")
    if isinstance(runtime_mode, str):
        return runtime_mode.strip().lower() != "real"
    return False


def _assert_paper_mechanisms(
    run_root: Path,
    cfg: Dict[str, Any],
    embed_record: Dict[str, Any],
    detect_record: Dict[str, Any],
    evaluate_report: Dict[str, Any],
    repo_root: Path,
) -> List[str]:
    """
    功能：执行 paper_full 机制断言并返回失败列表。

    Execute paper-full mechanism assertions and return failure reasons.

    Args:
        run_root: Unified run root.
        cfg: Effective config mapping.
        embed_record: Embed record mapping.
        detect_record: Detect record mapping.
        evaluate_report: Evaluation report mapping.

    Returns:
        List of failure messages. Empty list indicates success.
    """
    failures: List[str] = []
    frozen_impl_constraints = _load_paper_frozen_impl_constraints(repo_root)

    paper_cfg = cfg.get("paper_faithfulness") if isinstance(cfg.get("paper_faithfulness"), dict) else {}
    if paper_cfg.get("enabled") is not True:
        failures.append("paper_faithfulness.enabled must be true")

    impl_cfg = cfg.get("impl") if isinstance(cfg.get("impl"), dict) else {}
    for field_name, expected_value in frozen_impl_constraints.items():
        actual_value = impl_cfg.get(field_name)
        if not isinstance(actual_value, str) or not actual_value.strip():
            failures.append(f"impl.{field_name} must be non-empty str")
            continue
        normalized_actual_value = actual_value.strip()
        if normalized_actual_value != expected_value:
            failures.append(f"impl.{field_name} must be {expected_value}, got {normalized_actual_value}")

    watermark_cfg = cfg.get("watermark") if isinstance(cfg.get("watermark"), dict) else {}
    hf_cfg = watermark_cfg.get("hf") if isinstance(watermark_cfg.get("hf"), dict) else {}
    lf_cfg = watermark_cfg.get("lf") if isinstance(watermark_cfg.get("lf"), dict) else {}

    if hf_cfg.get("enabled") is not True:
        failures.append("watermark.hf.enabled must be true")
    if hf_cfg.get("tail_truncation_mode") != "keyed_template_correlation":
        failures.append("watermark.hf.tail_truncation_mode must be keyed_template_correlation")
    if hf_cfg.get("selection") != "keyed_rademacher_template":
        failures.append("watermark.hf.selection must be keyed_rademacher_template")

    if lf_cfg.get("enabled") is not True:
        failures.append("watermark.lf.enabled must be true")
    if lf_cfg.get("coding_mode") != "pseudogaussian_template_additive":
        failures.append("watermark.lf.coding_mode must be pseudogaussian_template_additive")
    if lf_cfg.get("decoder") != "matched_correlation":
        failures.append("watermark.lf.decoder must be matched_correlation")

    # v2.0 收口：论文正式路径禁止 image-domain sidecar
    detect_runtime_cfg = cfg.get("detect_runtime") if isinstance(cfg.get("detect_runtime"), dict) else {}
    if detect_runtime_cfg.get("image_domain_sidecar_enabled") is not False:
        failures.append(
            "detect_runtime.image_domain_sidecar_enabled must be false in paper mode (v2.0 closure)"
        )

    embed_content = _pick_mapping(embed_record, [["content_evidence"], ["content_result"]]) or {}
    injection_status = _pick_str(embed_content, [["injection_status"]])
    if injection_status != "ok":
        failures.append("injection_status must be ok for latent per-step evidence")
    injection_trace_digest = _pick_str(embed_content, [["injection_trace_digest"]])
    if not isinstance(injection_trace_digest, str) or len(injection_trace_digest) != 64:
        failures.append("injection_trace_digest must be 64-hex digest")
    injection_digest = _pick_str(embed_content, [["injection_digest"]])
    if not isinstance(injection_digest, str) or len(injection_digest) != 64:
        failures.append("injection_digest must be 64-hex digest")
    step_summary_digest = _pick_str(embed_content, [["step_summary_digest"]])
    if not isinstance(step_summary_digest, str) or len(step_summary_digest) != 64:
        failures.append("step_summary_digest must be 64-hex digest")

    injection_site_spec = _pick_mapping(embed_content, [["injection_site_spec"]])
    if not isinstance(injection_site_spec, dict):
        failures.append("embed content_evidence.injection_site_spec must exist")
    else:
        site_status = injection_site_spec.get("status")
        if isinstance(site_status, str):
            normalized = site_status.strip().lower()
            if normalized in {"absent", "failed", "fail", "error"}:
                failures.append("injection_site_spec.status must indicate bound/ok")
        hook_type = injection_site_spec.get("hook_type")
        if hook_type != "callback_on_step_end":
            failures.append("injection_site_spec.hook_type must be callback_on_step_end")

    trajectory_spec_digest = _pick_str(
        embed_content,
        [
            ["trajectory_evidence", "trajectory_spec_digest"],
            ["trajectory_spec_digest"],
        ],
    )
    if not isinstance(trajectory_spec_digest, str) or len(trajectory_spec_digest) != 64:
        failures.append("trajectory_spec_digest must exist and be 64-hex digest")

    embed_mode = _pick_str(embed_record, [["embed_trace", "embed_mode"]])
    latent_per_step_mode = embed_mode in {"latent_step_injection_stub_v1", "latent_step_injection_v1"}

    lf_trace_digest = _pick_str(
        embed_record,
        [
            ["content_evidence", "lf_trace_digest"],
            ["content_result", "lf_trace_digest"],
            ["embed_trace", "lf_trace_digest"],
        ],
    )
    if (not latent_per_step_mode) and lf_trace_digest is None:
        failures.append("content_evidence.lf_trace_digest must exist")
    hf_trace_digest = _pick_str(
        embed_record,
        [
            ["content_evidence", "hf_trace_digest"],
            ["content_result", "hf_trace_digest"],
            ["embed_trace", "hf_trace_digest"],
        ],
    )
    if hf_cfg.get("enabled") is True and (not latent_per_step_mode) and hf_trace_digest is None:
        failures.append("content_evidence.hf_trace_digest must exist when hf enabled")

    detect_payload = _pick_mapping(detect_record, [["content_evidence_payload"]]) or {}
    detect_runtime_fallback = _is_detect_runtime_fallback(detect_record)
    detect_geometry_payload = _pick_mapping(
        detect_record,
        [["geometry_evidence_payload"], ["geometry_evidence"], ["geometry_result"]],
    ) or {}

    sync_digest = _pick_str(
        detect_payload,
        [["sync_digest"], ["geometry_evidence", "sync_digest"]],
    )
    if sync_digest is None:
        sync_digest = _pick_str(detect_geometry_payload, [["sync_digest"]])
    if sync_digest is None and not detect_runtime_fallback:
        failures.append("detect content evidence must include sync_digest")

    anchor_digest = _pick_str(
        detect_payload,
        [["anchor_digest"], ["geometry_evidence", "anchor_digest"]],
    )
    if anchor_digest is None:
        anchor_digest = _pick_str(detect_geometry_payload, [["anchor_digest"]])

    anchor_metrics = _pick_mapping(
        detect_payload,
        [["anchor_metrics"], ["geometry_evidence", "anchor_metrics"]],
    )
    if not isinstance(anchor_metrics, dict):
        anchor_metrics = _pick_mapping(detect_geometry_payload, [["anchor_metrics"]])

    if (not isinstance(anchor_digest, str) or len(anchor_digest) != 64) and not detect_runtime_fallback:
        failures.append("detect content evidence must include 64-hex anchor_digest")

    if not isinstance(anchor_metrics, dict) and not detect_runtime_fallback:
        failures.append("detect content evidence must include anchor_metrics")
    elif isinstance(anchor_metrics, dict):
        extraction_source = anchor_metrics.get("extraction_source")
        if extraction_source not in {"attention_map_relation", "attention_relation_summary"} and not detect_runtime_fallback:
            failures.append("anchor_metrics.extraction_source must be attention-map relation based")

    evaluate_report_payload = _resolve_evaluation_report_payload(evaluate_report)

    attack_protocol_version = _pick_str(
        evaluate_report_payload,
        [["attack_protocol_version"], ["anchors", "attack_protocol_version"]],
    )
    if not isinstance(attack_protocol_version, str) or not attack_protocol_version:
        failures.append("evaluation_report.attack_protocol_version must exist")
    attack_protocol_digest = _pick_str(
        evaluate_report_payload,
        [["attack_protocol_digest"], ["anchors", "attack_protocol_digest"]],
    )
    if not isinstance(attack_protocol_digest, str) or not attack_protocol_digest:
        failures.append("evaluation_report.attack_protocol_digest must exist")

    attack_coverage_digest = _pick_str(
        evaluate_report_payload,
        [["attack_coverage_digest"], ["anchors", "attack_coverage_digest"]],
    )
    if not isinstance(attack_coverage_digest, str) or not attack_coverage_digest:
        failures.append("evaluation_report.attack_coverage_digest must exist")

    metrics_by_condition = evaluate_report_payload.get("metrics_by_attack_condition")
    if not isinstance(metrics_by_condition, list) or len(metrics_by_condition) == 0:
        failures.append("evaluation_report.metrics_by_attack_condition must be non-empty list")

    compare_summary = run_root / "artifacts" / "multi_protocol_evaluation" / "artifacts" / "protocol_compare" / "compare_summary.json"
    if not compare_summary.exists() or not compare_summary.is_file():
        failures.append(f"multi-protocol compare summary missing: {compare_summary}")
    else:
        compare_failures = _assert_multi_protocol_compare_success(compare_summary)
        if detect_runtime_fallback:
            compare_failures = [
                item for item in compare_failures
                if "failed protocol runs" not in item
            ]
        failures.extend(compare_failures)

    allowed_impl_ids = _load_runtime_whitelist_impl_ids(repo_root)
    cfg_impl_ids = _collect_impl_ids_from_cfg(cfg)
    for impl_id in cfg_impl_ids:
        if impl_id not in allowed_impl_ids:
            failures.append(f"impl_id not in runtime_whitelist: {impl_id}")

    # v2.0 收口断言：旧私有 HF 纹理评分函数必须不存在
    hf_embedder_path = repo_root / "main" / "watermarking" / "content_chain" / "high_freq_embedder.py"
    if hf_embedder_path.exists():
        source = hf_embedder_path.read_text(encoding="utf-8")
        if "_hf_image_texture_score" in source:
            failures.append("_hf_image_texture_score must not exist in high_freq_embedder.py (v2.0 closure)")

    # v2.0 收口断言：旧 baseline content extractor 文件必须不存在
    baseline_path = repo_root / "main" / "watermarking" / "content_chain" / "content_baseline_extractor.py"
    if baseline_path.exists():
        failures.append("content_baseline_extractor.py must not exist (v2.0 closure)")

    # v2.0 收口断言：旧 align invariance extractor 文件必须不存在
    align_path = repo_root / "main" / "watermarking" / "geometry_chain" / "align_invariance_extractor.py"
    if align_path.exists():
        failures.append("align_invariance_extractor.py must not exist (v2.0 closure)")

    # v2.0 收口断言：正式实现路径不得写出 adapter_path/fallback_used/fallback_reason 旧语义字段
    infer_runtime_path = repo_root / "main" / "diffusion" / "sd3" / "infer_runtime.py"
    if infer_runtime_path.exists():
        ir_source = infer_runtime_path.read_text(encoding="utf-8")
        for forbidden_field in ("\"adapter_path\"", "\"fallback_used\"", "\"fallback_reason\""):
            # 允许在注释中出现（以 # 开头的行）；仅检查非注释赋值行
            for line in ir_source.splitlines():
                stripped = line.strip()
                if forbidden_field in stripped and not stripped.startswith("#") and "_build_injection_cfg" not in stripped:
                    failures.append(
                        f"infer_runtime.py 正式路径不得写出旧语义字段 {forbidden_field} (v2.0 closure)"
                    )
                    break

    embed_orch_path = repo_root / "main" / "watermarking" / "embed" / "orchestrator.py"
    if embed_orch_path.exists():
        eo_source = embed_orch_path.read_text(encoding="utf-8")
        for forbidden_field in ("\"adapter_path\"", "\"fallback_used\"", "\"fallback_reason\""):
            for line in eo_source.splitlines():
                stripped = line.strip()
                if forbidden_field in stripped and not stripped.startswith("#"):
                    failures.append(
                        f"embed/orchestrator.py 正式路径不得写出旧语义字段 {forbidden_field} (v2.0 closure)"
                    )
                    break

    # v3 闭包断言：semantic_mask_provider.py formal path 不得写出 mask_fallback_reason / fallback_used / fallback_reason
    mask_provider_path = repo_root / "main" / "watermarking" / "content_chain" / "semantic_mask_provider.py"
    if mask_provider_path.exists():
        mp_source = mask_provider_path.read_text(encoding="utf-8")
        for forbidden_mp_field in ("\"mask_fallback_reason\"", "\"fallback_used\"", "\"fallback_reason\""):
            for line in mp_source.splitlines():
                stripped = line.strip()
                if forbidden_mp_field in stripped and not stripped.startswith("#"):
                    failures.append(
                        f"semantic_mask_provider.py formal path 不得写出旧字段 {forbidden_mp_field} (v3 closure)"
                    )
                    break

    # v3 闭包断言：content_detector.py formal path 不得存在 test_mode 参与 plan_digest 回填逻辑
    content_detector_path = repo_root / "main" / "watermarking" / "content_chain" / "content_detector.py"
    if content_detector_path.exists():
        cd_source = content_detector_path.read_text(encoding="utf-8")
        for line in cd_source.splitlines():
            stripped = line.strip()
            # 检查 test_mode 与 plan_digest 从 cfg 回填的关联逻辑
            if (
                "test_mode" in stripped
                and ("plan_digest" in stripped or "cfg_plan_digest" in stripped)
                and not stripped.startswith("#")
            ):
                failures.append(
                    "content_detector.py formal path 不得存在 test_mode 参与 plan_digest 回填逻辑 (v3 closure)"
                )
                break

    # v3 闭包断言：detect/orchestrator.py formal path detector_inputs 不得包含 test_mode 字段
    detect_orch_path = repo_root / "main" / "watermarking" / "detect" / "orchestrator.py"
    if detect_orch_path.exists():
        do_source = detect_orch_path.read_text(encoding="utf-8")
        in_detector_inputs = False
        for line in do_source.splitlines():
            stripped = line.strip()
            # 定位 detector_inputs 字典构造块（以 "detector_inputs" 赋值开头）
            if "detector_inputs" in stripped and "Dict[str, Any]" in stripped:
                in_detector_inputs = True
            if in_detector_inputs:
                if stripped.startswith("}"):
                    in_detector_inputs = False
                elif "\"test_mode\"" in stripped and not stripped.startswith("#"):
                    failures.append(
                        "detect/orchestrator.py detector_inputs 不得包含 \"test_mode\" 字段 (v3 closure)"
                    )
                    in_detector_inputs = False
                    break

    # v4 闭包断言：当前 formal schema 文件不得显式保留 historical_fields: 段
    schema_ext_path = repo_root / "configs" / "records_schema_extensions.yaml"
    if schema_ext_path.exists():
        schema_ext_text = schema_ext_path.read_text(encoding="utf-8")
        import yaml as _yaml
        try:
            schema_ext_obj = _yaml.safe_load(schema_ext_text)
            if isinstance(schema_ext_obj, dict) and "historical_fields" in schema_ext_obj:
                failures.append(
                    "records_schema_extensions.yaml 当前 formal schema 文件不得显式保留 historical_fields: 段 (v4 closure)"
                )
        except Exception:
            failures.append("records_schema_extensions.yaml 解析失败，无法验证 historical_fields 闭包")

    # v4 闭包断言：当前 formal contract 文件不得显式保留 historical_field_paths: 段
    frozen_contracts_path = repo_root / "configs" / "frozen_contracts.yaml"
    if frozen_contracts_path.exists():
        frozen_contracts_text = frozen_contracts_path.read_text(encoding="utf-8")
        try:
            frozen_contracts_obj = _yaml.safe_load(frozen_contracts_text)
            if isinstance(frozen_contracts_obj, dict) and "historical_field_paths" in frozen_contracts_obj:
                failures.append(
                    "frozen_contracts.yaml 当前 formal contract 文件不得显式保留 historical_field_paths: 段 (v4 closure)"
                )
        except Exception:
            failures.append("frozen_contracts.yaml 解析失败，无法验证 historical_field_paths 闭包")

    # v4 闭包断言：当前正式测试集合不得消费旧 fallback_*/align_* 字段
    formal_test_path = repo_root / "tests" / "test_records_schema_append_only_fields.py"
    if formal_test_path.exists():
        formal_test_source = formal_test_path.read_text(encoding="utf-8")
        for forbidden_test_field in ("\"fallback_used\"", "\"fallback_reason\"", "\"mask_fallback_reason\"",
                                     "\"align_trace_digest\"", "\"align_metrics\"", "\"align_config_digest\""):
            for tline in formal_test_source.splitlines():
                tstripped = tline.strip()
                if forbidden_test_field in tstripped and not tstripped.startswith("#"):
                    failures.append(
                        f"test_records_schema_append_only_fields.py 正式测试集合不得消费旧字段 {forbidden_test_field} (v4 closure)"
                    )
                    break

    # v5 闭包断言：embed/orchestrator.py 不得包含 embed_identity_mode / baseline_identity_v0
    orchestrator_path = repo_root / "main" / "watermarking" / "embed" / "orchestrator.py"
    if orchestrator_path.exists():
        orchestrator_source = orchestrator_path.read_text(encoding="utf-8")
        for v5_forbidden in ("embed_identity_mode", "baseline_identity_v0", "identity_pipeline"):
            for oline in orchestrator_source.splitlines():
                ostripped = oline.strip()
                if v5_forbidden in ostripped and not ostripped.startswith("#"):
                    failures.append(
                        f"embed/orchestrator.py 正式路径不得包含 identity baseline 符号 '{v5_forbidden}' (v5 closure)"
                    )
                    break

    # v5 闭包断言：runtime_whitelist.yaml 不得在 allowed_overrides / arg_name_enum 中保留 embed_identity_mode
    runtime_whitelist_path = repo_root / "configs" / "runtime_whitelist.yaml"
    if runtime_whitelist_path.exists():
        whitelist_text = runtime_whitelist_path.read_text(encoding="utf-8")
        if "embed_identity_mode" in whitelist_text:
            failures.append(
                "runtime_whitelist.yaml 不得保留 embed_identity_mode 条目 (v5 closure)"
            )

    # v5 闭包断言：paper_full_cuda.yaml 不得包含 test_mode_identity
    paper_full_path = repo_root / "configs" / "paper_full_cuda.yaml"
    if paper_full_path.exists():
        paper_full_text = paper_full_path.read_text(encoding="utf-8")
        if "test_mode_identity" in paper_full_text:
            failures.append(
                "configs/paper_full_cuda.yaml 不得保留 test_mode_identity 键 (v5 closure)"
            )

    # v5 闭包断言：run_onefile_workflow.py 不得使用 embed_identity_mode=true 作为 CLI 覆写参数
    onefile_path = repo_root / "scripts" / "run_onefile_workflow.py"
    if onefile_path.exists():
        onefile_text = onefile_path.read_text(encoding="utf-8")
        if "embed_identity_mode=true" in onefile_text.lower():
            failures.append(
                "scripts/run_onefile_workflow.py 不得使用 embed_identity_mode=true 覆写 (v5 closure)"
            )

    # v5 闭包断言：正式测试集合不得消费 identity baseline 旧符号
    formal_naming_test_path = repo_root / "tests" / "test_records_naming_normalization.py"
    if formal_naming_test_path.exists():
        naming_test_source = formal_naming_test_path.read_text(encoding="utf-8")
        for v5_test_forbidden in ("embed_identity_mode", "baseline_identity_v0", "identity_pipeline"):
            for nline in naming_test_source.splitlines():
                nstripped = nline.strip()
                if v5_test_forbidden in nstripped and not nstripped.startswith("#"):
                    failures.append(
                        f"test_records_naming_normalization.py 正式测试集合不得消费旧符号 '{v5_test_forbidden}' (v5 closure)"
                    )
                    break

    return failures


def _assert_multi_protocol_compare_success(compare_summary_path: Path) -> List[str]:
    """
    功能：校验 multi-protocol compare 汇总状态是否全量成功。

    Assert protocol compare summary is valid and contains no failed protocol items.

    Args:
        compare_summary_path: compare_summary.json path.

    Returns:
        Failure reason list. Empty list means success.
    """
    failures: List[str] = []
    if not isinstance(compare_summary_path, Path):
        return ["compare summary path must be Path"]
    if not compare_summary_path.exists() or not compare_summary_path.is_file():
        return [f"compare summary missing: {compare_summary_path}"]

    try:
        compare_obj = _load_json(compare_summary_path)
    except Exception as exc:
        return [f"compare summary parse failed: {type(exc).__name__}: {exc}"]

    schema_version = compare_obj.get("schema_version")
    if schema_version != "protocol_compare_v1":
        failures.append(f"compare summary schema_version must be protocol_compare_v1, got {schema_version}")

    protocols_obj = compare_obj.get("protocols")
    if not isinstance(protocols_obj, list) or len(protocols_obj) == 0:
        failures.append("compare summary protocols must be non-empty list")
        return failures

    failed_count = 0
    for idx, protocol_item in enumerate(protocols_obj):
        if not isinstance(protocol_item, dict):
            failed_count += 1
            failures.append(f"compare summary protocol item must be dict: index={idx}")
            continue
        status_value = protocol_item.get("status")
        if status_value != "ok":
            failed_count += 1

    if failed_count > 0:
        failures.append(
            f"compare summary contains failed protocol runs: failed={failed_count}, total={len(protocols_obj)}"
        )

    return failures


def main() -> None:
    """
    功能：脚本入口。

    CLI entry point.

    Args:
        None.

    Returns:
        None.
    """
    parser = argparse.ArgumentParser(description="Assert paper-full mechanism anchors from run_root")
    parser.add_argument("--run-root", required=True, type=Path, help="run_root path")
    parser.add_argument("--config", required=True, type=Path, help="effective config path")
    parser.add_argument("--profile", required=True, type=str, help="workflow profile")
    args = parser.parse_args()

    profile = args.profile.strip()
    if profile != "paper_full_cuda":
        raise ValueError(f"assert_paper_mechanisms requires profile=paper_full_cuda, got: {profile}")

    run_root = args.run_root.resolve()
    cfg_path = args.config.resolve()

    cfg = _load_yaml(cfg_path)
    embed_record = _load_json(run_root / "records" / "embed_record.json")
    detect_record = _load_json(run_root / "records" / "detect_record.json")
    evaluate_report = _load_json(run_root / "artifacts" / "evaluation_report.json")

    repo_root = _find_repo_root(cfg_path)

    failures = _assert_paper_mechanisms(
        run_root=run_root,
        cfg=cfg,
        embed_record=embed_record,
        detect_record=detect_record,
        evaluate_report=evaluate_report,
        repo_root=repo_root,
    )

    if failures:
        print("[assert_paper_mechanisms] FAIL")
        for item in failures:
            print(f"  - {item}")
        sys.exit(1)

    print("[assert_paper_mechanisms] PASS")
    sys.exit(0)


if __name__ == "__main__":
    main()
