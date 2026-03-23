"""
阈值校准 CLI 入口

功能说明：
- 本模块提供一个 CLI 入口，用于执行阈值校准流程。
- 当前实现执行真实 NP 校准流程，支持阈值工件与元数据工件写盘。
- 流程包含事实源绑定、配置校验、实现解析、校准执行与记录闭包。
"""

import sys
import argparse
import glob
from pathlib import Path
from typing import Dict, Any, cast
import uuid

from main.cli import assert_module_execution


assert_module_execution("run_calibrate")

from main.core import time_utils
from main.core.contracts import (
    load_frozen_contracts,
    bind_contract_to_record,
    get_contract_interpretation
)
from main.policy.runtime_whitelist import (
    load_runtime_whitelist,
    load_policy_path_semantics,
    assert_consistent_with_semantics,
    bind_whitelist_to_record,
    bind_semantics_to_record
)
from main.core import records_io
from main.core import digests
from main.core import config_loader
from main.core import schema
from main.core import status
from main.policy import path_policy
from main.registries import runtime_resolver
from main.watermarking.detect.orchestrator import run_calibrate_orchestrator
from main.core.errors import RunFailureReason
from main.evaluation.workflow_inputs import ensure_minimal_ground_truth_records
from main.cli.run_common import (
    bind_impl_identity_fields,
    set_failure_status,
    format_fact_sources_mismatch,
    build_seed_audit,
    build_determinism_controls,
    normalize_nondeterminism_notes,
    build_cli_config_migration_hint
)


def _resolve_label_from_detect_record(record: Any) -> bool | None:
    """
    功能：从 detect record 解析布尔标签。 

    Resolve boolean label from detect record candidates.

    Args:
        record: Detect record mapping.

    Returns:
        True/False label, or None when missing.
    """
    if not isinstance(record, dict):
        raise TypeError("record must be dict")
    record_dict = cast(Dict[str, Any], record)
    for key_name in ["label", "ground_truth", "is_watermarked"]:
        value = record_dict.get(key_name)
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)) and value in (0, 1):
            return bool(value)
    return None


def _validate_detect_record_label_balance_for_calibration(cfg: Any) -> tuple[int, int]:
    """
    功能：校验校准阶段 detect_records 的正负样本计数。 

    Validate detect record label balance before calibration execution.

    Args:
        cfg: Runtime config mapping.

    Returns:
        Tuple of (n_pos, n_neg) label counts from matched detect records.

    Raises:
        ValueError: If records glob is missing or n_pos/n_neg is zero.
    """
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be dict")
    cfg_dict = cast(Dict[str, Any], cfg)

    calibration_node = cfg_dict.get("calibration")
    calibration_cfg: Dict[str, Any] = cast(Dict[str, Any], calibration_node) if isinstance(calibration_node, dict) else {}
    records_glob = calibration_cfg.get("detect_records_glob")
    if not isinstance(records_glob, str) or not records_glob:
        raise ValueError("calibration.detect_records_glob is required")

    matched_paths = sorted(glob.glob(records_glob, recursive=True))
    if len(matched_paths) == 0:
        raise ValueError(f"no detect records matched: {records_glob}")

    n_pos = 0
    n_neg = 0
    for path_str in matched_paths:
        path_obj = Path(path_str)
        if not path_obj.is_file():
            continue
        payload = records_io.read_json(str(path_obj))
        if not isinstance(payload, dict):
            continue
        payload_dict = cast(Dict[str, Any], payload)
        label_value = _resolve_label_from_detect_record(payload_dict)
        if label_value is True:
            n_pos += 1
        elif label_value is False:
            n_neg += 1

    if n_pos <= 0 or n_neg <= 0:
        # 样本空集风险，必须 fail-fast 阻断校准。
        raise ValueError(
            "calibration requires both positive and negative labeled detect records "
            f"(n_pos={n_pos}, n_neg={n_neg}, detect_records_glob={records_glob})"
        )
    return n_pos, n_neg


def _autofill_calibration_detect_records_glob(cfg: Any, run_root: Any) -> None:
    """
    功能：为 paper/onefile 标准路径自动补全 calibration.detect_records_glob。

    Autofill calibration.detect_records_glob from the current run_root when the
    config leaves it unset. This keeps paper_full profiles immediately runnable
    without expanding the formal override surface.

    Args:
        cfg: Mutable runtime config mapping.
        run_root: Current run root path.

    Returns:
        None.
    """
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be dict")
    if not isinstance(run_root, Path):
        raise TypeError("run_root must be Path")
    cfg_dict = cast(Dict[str, Any], cfg)

    calibration_node = cfg_dict.get("calibration")
    calibration_cfg: Dict[str, Any] = cast(Dict[str, Any], calibration_node) if isinstance(calibration_node, dict) else {}
    existing_glob = calibration_cfg.get("detect_records_glob")
    if isinstance(existing_glob, str) and existing_glob:
        return

    candidate_glob = str((run_root / "records" / "*detect*.json").resolve())
    calibration_cfg = dict(calibration_cfg)
    calibration_cfg["detect_records_glob"] = candidate_glob
    cfg_dict["calibration"] = calibration_cfg


def _ensure_calibration_detect_records_ready(cfg: Any, run_root: Any) -> Dict[str, Any]:
    """
    功能：为 calibration 阶段补齐主代码内生的最小标签输入。

    Ensure calibration has a labelled detect-record set prepared by main code.

    Args:
        cfg: Mutable runtime config mapping.
        run_root: Current run root path.

    Returns:
        Preparation summary mapping.
    """
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be dict")
    if not isinstance(run_root, Path):
        raise TypeError("run_root must be Path")
    return ensure_minimal_ground_truth_records(cfg, run_root, "calibrate")


def run_calibrate(output_dir: Any, config_path: Any, overrides: Any = None) -> None:
    """
    功能：执行校准流程。

    Execute calibration workflow.

    Args:
        output_dir: Run root directory for records/artifacts.
        config_path: YAML config path.
        overrides: Optional CLI override args list.
    
    Returns:
        None.
    
    Raises:
        ValueError: If input arguments are invalid.
    """
    if not output_dir:
        # output_dir 输入不合法，必须 fail-fast。
        raise ValueError("output_dir must be non-empty str")
    if not config_path:
        # config_path 输入不合法，必须 fail-fast。
        raise ValueError("config_path must be non-empty str")
    if overrides is not None and not isinstance(overrides, list):
        # overrides 输入不合法，必须 fail-fast。
        raise ValueError("overrides must be list or None")
    validated_overrides: list[str] | None = None
    if isinstance(overrides, list):
        override_candidates = cast(list[Any], overrides)
        if any(not isinstance(item, str) or not item for item in override_candidates):
            # overrides 列表项不合法，必须 fail-fast。
            raise ValueError("overrides items must be non-empty str")
        validated_overrides = cast(list[str], override_candidates)

    # 创建 run_root 与最小 run_meta。
    run_root = path_policy.derive_run_root(Path(output_dir))
    records_dir = run_root / "records"
    artifacts_dir = run_root / "artifacts"
    logs_dir = run_root / "logs"
    layout_initialized = False
    allow_nonempty_run_root = False
    allow_nonempty_run_root_reason = None
    override_applied_for_layout = None
    
    # 初始化最小 run_meta。
    started_at = time_utils.now_utc_iso_z()
    run_meta: Dict[str, Any] = {
        "run_id": f"run-{uuid.uuid4().hex}",
        "command": "calibrate",
        "created_at_utc": started_at,
        "started_at": started_at,
        "output_dir_input": output_dir,
        "cfg_digest": "unknown",
        "policy_path": "unknown",
        "impl_id": "unknown",
        "impl_version": "unknown",
        "impl_identity": None,
        "impl_identity_digest": None,
        "status_ok": True,
        "status_reason": RunFailureReason.OK,
        "status_details": None,
        "manifest_rel_path": "unknown",
        "path_policy": None,
        "run_root_reuse_allowed": False,
        "run_root_reuse_reason": None
    }
    
    error = None
    record: Dict[str, Any] | None = None
    try:
        # 加载事实源。
        print("[Calibrate] Loading fact sources...")
        contracts = load_frozen_contracts(config_loader.FROZEN_CONTRACTS_PATH)
        whitelist = load_runtime_whitelist(config_loader.RUNTIME_WHITELIST_PATH)
        semantics = load_policy_path_semantics(config_loader.POLICY_PATH_SEMANTICS_PATH)
        injection_scope_manifest = config_loader.load_injection_scope_manifest()

        # 绑定冻结锚点到 run_meta。
        print("[Calibrate] Binding freeze anchors to run_meta...")
        status.bind_freeze_anchors_to_run_meta(
            run_meta,
            contracts,
            whitelist,
            semantics,
            injection_scope_manifest
        )

        # 生成事实源快照用于运行期一致性校验。
        snapshot = records_io.build_fact_sources_snapshot(
            contracts,
            whitelist,
            semantics,
            injection_scope_manifest
        )
        run_meta["bound_fact_sources"] = snapshot

        # 验证 whitelist 与 semantics 一致性。
        print("[Calibrate] Validating whitelist-semantics consistency...")
        assert_consistent_with_semantics(whitelist, semantics)

        interpretation = get_contract_interpretation(contracts)
        try:
            cfg, cfg_digest, cfg_audit_metadata = config_loader.load_and_validate_config(
                config_path,
                whitelist,
                semantics,
                contracts,
                interpretation,
                overrides=validated_overrides
            )
        except Exception as exc:
            set_failure_status(run_meta, RunFailureReason.CONFIG_INVALID, exc)
            raise
        run_meta["cfg_digest"] = cfg_digest
        run_meta["policy_path"] = cfg["policy_path"]

        _autofill_calibration_detect_records_glob(cfg, run_root)
        run_meta["workflow_input_preparation"] = _ensure_calibration_detect_records_ready(cfg, run_root)

        # 样本有效性前置门禁（n_pos/n_neg 不能为0），同时获取样本计数用于落盘。
        n_pos_labeled, n_neg_labeled = _validate_detect_record_label_balance_for_calibration(cfg)

        seed_parts, seed_digest, seed_value, seed_rule_id = build_seed_audit(cfg, "calibrate")
        cfg["seed"] = seed_value
        run_meta["seed_parts"] = seed_parts
        run_meta["seed_digest"] = seed_digest
        run_meta["seed_rule_id"] = seed_rule_id
        run_meta["seed_value"] = seed_value
        run_meta["cfg"] = cfg

        determinism_controls = build_determinism_controls(cfg)
        if determinism_controls is not None:
            run_meta["determinism_controls"] = determinism_controls
        nondeterminism_notes = normalize_nondeterminism_notes(cfg.get("nondeterminism_notes"))
        if nondeterminism_notes is not None:
            run_meta["nondeterminism_notes"] = nondeterminism_notes

        allow_nonempty_run_root = cfg.get("allow_nonempty_run_root", False)
        allow_nonempty_run_root_reason = cfg.get("allow_nonempty_run_root_reason")
        override_applied_for_layout = cfg.get("override_applied")

        layout = path_policy.ensure_output_layout(
            run_root,
            allow_nonempty_run_root=bool(allow_nonempty_run_root),
            allow_nonempty_run_root_reason=allow_nonempty_run_root_reason,
            override_applied=override_applied_for_layout
        )
        layout_initialized = True
        # 锚定依赖环境到 requirements.txt。
        path_policy.anchor_requirements(run_root)
        records_dir = layout["records_dir"]
        artifacts_dir = layout["artifacts_dir"]
        logs_dir = layout["logs_dir"]

        run_meta["path_policy"] = {
            "allow_nonempty_run_root": bool(allow_nonempty_run_root),
            "allow_nonempty_run_root_reason": allow_nonempty_run_root_reason
        }
        run_meta["run_root_reuse_allowed"] = bool(allow_nonempty_run_root)
        run_meta["run_root_reuse_reason"] = allow_nonempty_run_root_reason

        # 写入 cfg_audit 工件到 artifacts/cfg_audit/cfg_audit.json。
        cfg_audit_dir = artifacts_dir / "cfg_audit"
        cfg_audit_dir.mkdir(parents=True, exist_ok=True)
        cfg_audit_path = cfg_audit_dir / "cfg_audit.json"
        try:
            records_io.write_artifact_json(str(cfg_audit_path), cfg_audit_metadata)
        except records_io.FactSourcesNotInitializedError:
            records_io.write_artifact_json_unbound(run_root, artifacts_dir, str(cfg_audit_path), cfg_audit_metadata)
        
        # 将 cfg_audit digest 锚定字段写入 run_meta。
        run_meta["cfg_pruned_for_digest_canon_sha256"] = cfg_audit_metadata["cfg_pruned_for_digest_canon_sha256"]
        run_meta["cfg_audit_canon_sha256"] = cfg_audit_metadata["cfg_audit_canon_sha256"]

        with records_io.bound_fact_sources(
            contracts,
            whitelist,
            semantics,
            run_root,
            records_dir,
            artifacts_dir,
            logs_dir,
            injection_scope_manifest=injection_scope_manifest
        ):
            bound_fact_sources = records_io.get_bound_fact_sources()
            run_meta["bound_fact_sources"] = bound_fact_sources
            if bound_fact_sources != snapshot:
                # 事实源快照不一致，必须 fail-fast。
                raise ValueError(format_fact_sources_mismatch(snapshot, bound_fact_sources))

            try:
                impl_identity, impl_set, impl_set_capabilities_digest = runtime_resolver.build_runtime_impl_set_from_cfg(
                    cfg,
                    whitelist,
                )
            except Exception as exc:
                set_failure_status(run_meta, RunFailureReason.IMPL_RESOLVE_FAILED, exc)
                raise
            run_meta["impl_id"] = impl_identity.content_extractor_id
            run_meta["impl_version"] = impl_set.content_extractor.impl_version
            run_meta["impl_identity"] = impl_identity.as_dict()
            run_meta["impl_identity_digest"] = runtime_resolver.compute_impl_identity_digest(impl_identity)
            run_meta["impl_set_capabilities_digest"] = impl_set_capabilities_digest
            impl_set_capabilities_extended_digest = cfg.get("impl_set_capabilities_extended_digest")
            if isinstance(impl_set_capabilities_extended_digest, str) and impl_set_capabilities_extended_digest:
                run_meta["impl_set_capabilities_extended_digest"] = impl_set_capabilities_extended_digest

            # 构造 calibration record。
            print("[Calibrate] Generating calibration record...")
            record = run_calibrate_orchestrator(cfg, impl_set)
            record["cfg_digest"] = cfg_digest
            record["policy_path"] = cfg["policy_path"]
            # 写入正负样本计数，用于审计校准数据集组成。
            record["n_pos"] = n_pos_labeled
            record["n_neg"] = n_neg_labeled
            override_applied = cfg.get("override_applied")
            if override_applied is not None:
                record["override_applied"] = override_applied
            # 写入 impl_set_capabilities_digest。
            record["impl_set_capabilities_digest"] = impl_set_capabilities_digest
            if isinstance(impl_set_capabilities_extended_digest, str) and impl_set_capabilities_extended_digest:
                record["impl_set_capabilities_extended_digest"] = impl_set_capabilities_extended_digest

            thresholds_artifact = record.get("thresholds_artifact")
            threshold_metadata_artifact = record.get("threshold_metadata_artifact")
            if not isinstance(thresholds_artifact, dict):
                # thresholds_artifact 缺失或类型不合法，必须 fail-fast。
                raise TypeError("thresholds_artifact must be dict")
            if not isinstance(threshold_metadata_artifact, dict):
                # threshold_metadata_artifact 缺失或类型不合法，必须 fail-fast。
                raise TypeError("threshold_metadata_artifact must be dict")
            thresholds_artifact_dict = cast(Dict[str, Any], thresholds_artifact)
            threshold_metadata_artifact_dict = cast(Dict[str, Any], threshold_metadata_artifact)

            thresholds_dir = artifacts_dir / "thresholds"
            thresholds_dir.mkdir(parents=True, exist_ok=True)
            thresholds_path = thresholds_dir / "thresholds_artifact.json"
            threshold_metadata_path = thresholds_dir / "threshold_metadata_artifact.json"

            path_policy.validate_output_target(thresholds_path, "artifact", run_root)
            path_policy.validate_output_target(threshold_metadata_path, "artifact", run_root)
            records_io.write_artifact_json(str(thresholds_path), thresholds_artifact_dict)
            records_io.write_artifact_json(str(threshold_metadata_path), threshold_metadata_artifact_dict)

            record["thresholds_artifact_path"] = str(thresholds_path)
            record["threshold_metadata_artifact_path"] = str(threshold_metadata_path)
            record["thresholds_artifact_digest"] = digests.canonical_sha256(thresholds_artifact_dict)
            record["threshold_metadata_artifact_digest"] = digests.canonical_sha256(threshold_metadata_artifact_dict)

            schema.ensure_required_fields(record, cfg, interpretation)

            # 口径版本标识写入 run_meta。
            run_meta["thresholds_rule_id"] = record.get("thresholds_rule_id", "<absent>")
            run_meta["thresholds_rule_version"] = record.get("thresholds_rule_version", "<absent>")

            # 绑定 impl_identity 字段族，由 cfg 提供。
            bind_impl_identity_fields(record, impl_identity, impl_set, contracts)

            # 绑定事实源字段。
            print("[Calibrate] Binding fact source fields...")
            bind_contract_to_record(record, contracts)
            bind_whitelist_to_record(record, whitelist)
            bind_semantics_to_record(record, semantics)

            try:
                schema.validate_record(record, interpretation=interpretation)
            except Exception as exc:
                set_failure_status(run_meta, RunFailureReason.RUNTIME_ERROR, exc)
                raise

            # 写盘，触发 freeze_gate.assert_prewrite。
            record_path = records_dir / "calibration_record.json"
            path_policy.validate_output_target(record_path, "record", run_root)
            print(f"[Calibrate] Writing record to {record_path}...")
            try:
                records_io.write_json(str(record_path), record)
            except Exception as exc:
                set_failure_status(run_meta, RunFailureReason.GATE_FAILED, exc)
                raise
    except Exception as exc:
        if run_meta.get("status_ok", True):
            set_failure_status(run_meta, RunFailureReason.RUNTIME_ERROR, exc)
        error = exc
    finally:
        # 绑定闭包最小合同字段，并产出 run_closure.json。
        if not layout_initialized:
            try:
                layout = path_policy.ensure_output_layout(
                    run_root,
                    allow_nonempty_run_root=bool(allow_nonempty_run_root),
                    allow_nonempty_run_root_reason=allow_nonempty_run_root_reason,
                    override_applied=override_applied_for_layout
                )
                records_dir = layout["records_dir"]
                artifacts_dir = layout["artifacts_dir"]
                logs_dir = layout["logs_dir"]
            except Exception as exc:
                if run_meta.get("status_ok", True):
                    set_failure_status(run_meta, RunFailureReason.RUNTIME_ERROR, exc)
                run_meta["status_details"] = f"layout_init_failed: {exc}"
        run_meta["ended_at"] = time_utils.now_utc_iso_z()
        try:
            status.finalize_run(run_root, records_dir, artifacts_dir, run_meta)
        except Exception:
            # finalize_run 失败必须 fail-fast。
            raise
        if error is not None:
            raise error

    if record is not None:
        print(f"[Calibrate] [OK] Calibration record written successfully")
        print(f"[Calibrate] Record contains {len(record)} fields (15 fact source fields + {len(record) - 15} business fields)")



def main():
    """主流程。"""
    parser = argparse.ArgumentParser(
        description="Calibrate thresholds (real NP implementation)"
    )
    parser.add_argument(
        "--out",
        default="tmp/cli_smoke/calibrate_run",
        help="Output run root directory"
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Config YAML path"
    )
    parser.add_argument(
        "--override",
        "--overrides",
        action="append",
        default=None,
        help="Override config key=value (JSON value). Can be repeated."
    )
    
    args = parser.parse_args()
    
    try:
        run_calibrate(args.out, args.config, args.override)
        sys.exit(0)
    except Exception as e:
        print(f"[Calibrate] [ERROR] Error: {e}", file=sys.stderr)
        hint = build_cli_config_migration_hint(e)
        if isinstance(hint, str) and hint:
            print(f"[Calibrate] [HINT] {hint}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
