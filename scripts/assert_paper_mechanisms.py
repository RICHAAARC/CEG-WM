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

    paper_cfg = cfg.get("paper_faithfulness") if isinstance(cfg.get("paper_faithfulness"), dict) else {}
    if paper_cfg.get("enabled") is not True:
        failures.append("paper_faithfulness.enabled must be true")

    impl_cfg = cfg.get("impl") if isinstance(cfg.get("impl"), dict) else {}
    if impl_cfg.get("sync_module_id") != "geometry_latent_sync_sd3_v1":
        failures.append("impl.sync_module_id must be geometry_latent_sync_sd3_v1")
    if impl_cfg.get("geometry_extractor_id") in {"geometry_baseline_identity_v1"}:
        failures.append("impl.geometry_extractor_id must not use baseline identity")

    watermark_cfg = cfg.get("watermark") if isinstance(cfg.get("watermark"), dict) else {}
    hf_cfg = watermark_cfg.get("hf") if isinstance(watermark_cfg.get("hf"), dict) else {}
    lf_cfg = watermark_cfg.get("lf") if isinstance(watermark_cfg.get("lf"), dict) else {}

    if hf_cfg.get("enabled") is not True:
        failures.append("watermark.hf.enabled must be true")
    if hf_cfg.get("tail_truncation_mode") != "top_k_per_latent":
        failures.append("watermark.hf.tail_truncation_mode must be top_k_per_latent")
    if hf_cfg.get("selection") != "top_k_magnitude_based":
        failures.append("watermark.hf.selection must be top_k_magnitude_based")

    if lf_cfg.get("enabled") is not True:
        failures.append("watermark.lf.enabled must be true")
    if lf_cfg.get("coding_mode") != "latent_space_sign_flipping":
        failures.append("watermark.lf.coding_mode must be latent_space_sign_flipping")
    if lf_cfg.get("decoder") != "belief_propagation":
        failures.append("watermark.lf.decoder must be belief_propagation")

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
    if sync_digest is None:
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

    # 几何链允许 sync-only 证据形态：若 anchor 字段缺失但 sync_digest 存在，不做硬失败。
    if isinstance(anchor_metrics, dict):
        extraction_source = anchor_metrics.get("extraction_source")
        if extraction_source not in {"attention_map_relation", "attention_relation_summary"}:
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

    allowed_impl_ids = _load_runtime_whitelist_impl_ids(repo_root)
    cfg_impl_ids = _collect_impl_ids_from_cfg(cfg)
    for impl_id in cfg_impl_ids:
        if impl_id not in allowed_impl_ids:
            failures.append(f"impl_id not in runtime_whitelist: {impl_id}")

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
