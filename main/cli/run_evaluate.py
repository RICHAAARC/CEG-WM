"""
评估性能 CLI 入口

功能说明：
- 规范化输出目录路径，确保输出布局，加载合同与白名单，验证配置，解析实现，构造记录，绑定字段，写盘，并产出闭包。
- 包含详细的输入验证、错误处理与状态更新机制，确保健壮性与可维护性。
- 当前实现执行只读阈值评测流程。
"""

import sys
import argparse
import glob
from pathlib import Path
import uuid
from typing import Any, Dict, cast

from main.cli import assert_module_execution


assert_module_execution("run_evaluate")

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
from main.core import config_loader
from main.core import schema
from main.core import status
from main.policy import path_policy
from main.registries import runtime_resolver
from main.evaluation import report_builder as eval_report_builder
from main.watermarking.detect.orchestrator import run_evaluate_orchestrator
from main.watermarking.fusion import decision_writer
from main.core.errors import RunFailureReason
from main.cli.run_common import (
    bind_impl_identity_fields,
    set_failure_status,
    format_fact_sources_mismatch,
    build_seed_audit,
    build_determinism_controls,
    normalize_nondeterminism_notes,
    build_cli_config_migration_hint
)


def _resolve_label_from_detect_record(record: Dict[str, Any]) -> bool | None:
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
    for key_name in ["label", "ground_truth", "is_watermarked"]:
        value = record.get(key_name)
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)) and value in (0, 1):
            return bool(value)
    return None


def _validate_detect_record_label_balance_for_evaluate(cfg: Dict[str, Any]) -> None:
    """
    功能：校验评估阶段 detect_records 的正负样本计数。 

    Validate detect record label balance before evaluate execution.

    Args:
        cfg: Runtime config mapping.

    Returns:
        None.

    Raises:
        ValueError: If records glob is missing or n_pos/n_neg is zero.
    """
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be dict")

    evaluate_node = cfg.get("evaluate")
    evaluate_cfg = evaluate_node if isinstance(evaluate_node, dict) else {}
    records_glob = evaluate_cfg.get("detect_records_glob")
    if not isinstance(records_glob, str) or not records_glob:
        raise ValueError("evaluate.detect_records_glob is required")

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
        label_value = _resolve_label_from_detect_record(payload)
        if label_value is True:
            n_pos += 1
        elif label_value is False:
            n_neg += 1

    if n_pos <= 0 or n_neg <= 0:
        # 样本空集风险，必须 fail-fast 阻断评估。
        raise ValueError(
            "evaluate requires both positive and negative labeled detect records "
            f"(n_pos={n_pos}, n_neg={n_neg}, detect_records_glob={records_glob})"
        )


def _autofill_evaluate_inputs(cfg: Dict[str, Any], run_root: Path) -> None:
    """
    功能：为 paper/onefile 标准路径自动补全 evaluate 输入工件位置。

    Autofill evaluate.detect_records_glob and evaluate.thresholds_path from the
    current run_root when the config leaves them unset.

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

    evaluate_node = cfg.get("evaluate")
    evaluate_cfg = evaluate_node if isinstance(evaluate_node, dict) else {}
    evaluate_cfg = dict(evaluate_cfg)
    existing_glob = evaluate_cfg.get("detect_records_glob")
    if not isinstance(existing_glob, str) or not existing_glob:
        evaluate_cfg["detect_records_glob"] = str((run_root / "records" / "*detect*.json").resolve())

    existing_thresholds_path = evaluate_cfg.get("thresholds_path")
    if not isinstance(existing_thresholds_path, str) or not existing_thresholds_path:
        thresholds_path = (run_root / "artifacts" / "thresholds" / "thresholds_artifact.json").resolve()
        evaluate_cfg["thresholds_path"] = str(thresholds_path)

    cfg["evaluate"] = evaluate_cfg


def run_evaluate(output_dir: str, config_path: str, overrides: list[str] | None = None) -> None:
    """
    功能：执行评估流程（只读阈值模式）。

    Execute evaluation workflow in readonly-threshold mode.

    Args:
        output_dir: Run root directory for records/artifacts.
        config_path: YAML config path.
        overrides: Optional CLI override args list.
    
    Returns:
        None.
    """
    if not isinstance(output_dir, str) or not output_dir:
        # output_dir 输入不合法，必须 fail-fast。
        raise ValueError("output_dir must be non-empty str")
    if not isinstance(config_path, str) or not config_path:
        # config_path 输入不合法，必须 fail-fast。
        raise ValueError("config_path must be non-empty str")
    if overrides is not None and not isinstance(overrides, list):
        # overrides 输入不合法，必须 fail-fast。
        raise ValueError("overrides must be list or None")

    # 创建 layout 和最小 run_meta。
    run_root = path_policy.derive_run_root(Path(output_dir))
    records_dir = run_root / "records"
    artifacts_dir = run_root / "artifacts"
    logs_dir = run_root / "logs"
    layout_initialized = False
    allow_nonempty_run_root = False
    allow_nonempty_run_root_reason = None
    override_applied_for_layout = None
    
    # 初始化最小 run_meta，待完善。
    started_at = time_utils.now_utc_iso_z()
    run_meta = {
        "run_id": f"run-{uuid.uuid4().hex}",
        "command": "evaluate",
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
        "path_policy": {
            "allow_nonempty_run_root": False
        }
    }
    
    error = None
    try:
        # 加载事实源。
        print("[Evaluate] Loading fact sources...")
        contracts = load_frozen_contracts(config_loader.FROZEN_CONTRACTS_PATH)
        whitelist = load_runtime_whitelist(config_loader.RUNTIME_WHITELIST_PATH)
        semantics = load_policy_path_semantics(config_loader.POLICY_PATH_SEMANTICS_PATH)
        injection_scope_manifest = config_loader.load_injection_scope_manifest()

        # 绑定冻结锚点到 run_meta。
        print("[Evaluate] Binding freeze anchors to run_meta...")
        status.bind_freeze_anchors_to_run_meta(
            run_meta,
            contracts,
            whitelist,
            semantics,
            injection_scope_manifest
        )

        # 生成事实源快照用于后续一致性校验。
        snapshot = records_io.build_fact_sources_snapshot(
            contracts,
            whitelist,
            semantics,
            injection_scope_manifest
        )
        run_meta["bound_fact_sources"] = snapshot

        # 验证 whitelist 与 semantics 一致性。
        print("[Evaluate] Validating whitelist-semantics consistency...")
        assert_consistent_with_semantics(whitelist, semantics)

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
            interpretation = get_contract_interpretation(contracts)
            try:
                cfg, cfg_digest, cfg_audit_metadata = config_loader.load_and_validate_config(
                    config_path,
                    whitelist,
                    semantics,
                    contracts,
                    interpretation,
                    overrides=overrides
                )
            except Exception as exc:
                set_failure_status(run_meta, RunFailureReason.CONFIG_INVALID, exc)
                raise
            run_meta["cfg_digest"] = cfg_digest
            run_meta["policy_path"] = cfg["policy_path"]

            _autofill_evaluate_inputs(cfg, run_root)

            # 样本有效性前置门禁（n_pos/n_neg 不能为0）。
            _validate_detect_record_label_balance_for_evaluate(cfg)

            seed_parts, seed_digest, seed_value, seed_rule_id = build_seed_audit(cfg, "evaluate")
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
            
            # 提取 run_root 复用参数。
            allow_nonempty_run_root = cfg.get("allow_nonempty_run_root", False)
            allow_nonempty_run_root_reason = cfg.get("allow_nonempty_run_root_reason")
            override_applied_for_layout = cfg.get("override_applied")
            
            # 创建输出布局（显式参数模式）。
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
            
            # 记录路径策略配置与审计字段。
            run_meta["path_policy"] = {
                "allow_nonempty_run_root": bool(allow_nonempty_run_root),
                "allow_nonempty_run_root_reason": allow_nonempty_run_root_reason
            }
            run_meta["run_root_reuse_allowed"] = bool(allow_nonempty_run_root)
            run_meta["run_root_reuse_reason"] = allow_nonempty_run_root_reason

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

            # 绑定 evaluate 报告锚点（优先使用 CLI 已解析事实源）。
            cfg["__evaluate_cfg_digest__"] = cfg_digest
            cfg["__policy_path__"] = cfg["policy_path"]
            cfg["__impl_digest__"] = impl_set_capabilities_extended_digest if isinstance(impl_set_capabilities_extended_digest, str) and impl_set_capabilities_extended_digest else impl_set_capabilities_digest

            # 构造 evaluation record。
            print("[Evaluate] Generating evaluation record...")
            record = run_evaluate_orchestrator(cfg, impl_set)
            if not isinstance(record, dict):
                # record 类型不符合预期，必须 fail-fast。
                raise TypeError("orchestrator output must be dict")
            record["cfg_digest"] = cfg_digest
            record["policy_path"] = cfg["policy_path"]
            override_applied = cfg.get("override_applied")
            if override_applied is not None:
                record["override_applied"] = override_applied
            # 写入 impl_set_capabilities_digest。
            record["impl_set_capabilities_digest"] = impl_set_capabilities_digest
            if isinstance(impl_set_capabilities_extended_digest, str) and impl_set_capabilities_extended_digest:
                record["impl_set_capabilities_extended_digest"] = impl_set_capabilities_extended_digest

            schema.ensure_required_fields(record, cfg, interpretation)

            # 口径版本标识写入 run_meta。
            run_meta["thresholds_rule_id"] = record.get("thresholds_rule_id", "<absent>")
            run_meta["thresholds_rule_version"] = record.get("thresholds_rule_version", "<absent>")

            # 绑定 impl_identity 字段族。
            bind_impl_identity_fields(record, impl_identity, impl_set, contracts)

            # 绑定事实源字段。
            print("[Evaluate] Binding fact source fields...")
            bind_contract_to_record(record, contracts)
            bind_whitelist_to_record(record, whitelist)
            bind_semantics_to_record(record, semantics)

            # CLI 单点收口：融合判决唯一 decision 写入与序列化。
            # 位置：ensure_required_fields 之后、validate_record 之前。
            decision_writer.assert_decision_write_bypass_blocked(record, interpretation)
            fusion_result = record.get("fusion_result")
            if fusion_result is None:
                # fusion_result 缺失是致命错误，必须 fail-fast。
                raise ValueError("fusion_result is required in record but was None")
            # 验证 fusion_result 是 FusionDecision 对象。
            from main.watermarking.fusion.interfaces import FusionDecision
            if not isinstance(fusion_result, FusionDecision):
                raise ValueError(
                    f"fusion_result must be FusionDecision instance, got {type(fusion_result)}"
                )
            # 调用 decision_writer 写入 decision 字段。
            decision_writer.apply_fusion_decision_to_record(
                record,
                fusion_result,
                interpretation
            )
            # 序列化 fusion_result 为可 JSON 化的 dict，覆盖原值。
            record["fusion_result"] = fusion_result.to_dict()

            metrics_node = record.get("metrics")
            metrics_payload = cast(Dict[str, Any], metrics_node) if isinstance(metrics_node, dict) else {}
            for metric_key in [
                "tpr_at_fpr_primary",
                "fpr_empirical",
                "fnr_empirical",
                "n_total",
                "n_pos",
                "n_neg",
            ]:
                if metric_key in metrics_payload and metric_key not in record:
                    # append-only：保留 metrics.*，镜像顶层关键指标供历史读取口径使用。
                    record[metric_key] = metrics_payload.get(metric_key)
            n_pos = metrics_payload.get("n_pos")
            n_neg = metrics_payload.get("n_neg")
            is_degenerate_evaluate = (
                isinstance(n_pos, int) and n_pos <= 0
            ) or (
                isinstance(n_neg, int) and n_neg <= 0
            )
            if is_degenerate_evaluate:
                # append-only: 仅在退化分支标记 evaluate_status。
                record["evaluate_status"] = "degenerate"

            try:
                schema.validate_record(record, interpretation=interpretation)
            except Exception as exc:
                set_failure_status(run_meta, RunFailureReason.RUNTIME_ERROR, exc)
                raise

            # 评测报告工件统一收口写盘（records_io artifact 路径）。
            evaluation_report_payload = record.get("evaluation_report")
            if isinstance(evaluation_report_payload, dict):
                evaluation_report_path = artifacts_dir / "evaluation_report.json"
                path_policy.validate_output_target(evaluation_report_path, "artifact", run_root)
                eval_report_builder.write_eval_report_via_records_io(
                    evaluation_report_payload,
                    str(evaluation_report_path),
                )

                # 兼容历史读取路径：保留 artifacts/eval_report.json（append-only）。
                eval_report_legacy_path = artifacts_dir / "eval_report.json"
                path_policy.validate_output_target(eval_report_legacy_path, "artifact", run_root)
                eval_report_builder.write_eval_report_via_records_io(
                    evaluation_report_payload,
                    str(eval_report_legacy_path),
                )
            
            # 写盘，触发 freeze_gate.assert_prewrite。
            record_path = records_dir / "evaluate_record.json"
            path_policy.validate_output_target(record_path, "record", run_root)
            print(f"[Evaluate] Writing record to {record_path}...")
            try:
                records_io.write_json(str(record_path), record)
            except Exception as exc:
                set_failure_status(run_meta, RunFailureReason.GATE_FAILED, exc)
                raise

            if is_degenerate_evaluate:
                # 退化评估必须在落盘后强制失败，防止静默通过下游门禁。
                raise ValueError("Degenerate evaluation: n_pos or n_neg must be positive")
    except Exception as exc:
        if run_meta.get("status_ok", True):
            set_failure_status(run_meta, RunFailureReason.RUNTIME_ERROR, exc)
        error = exc
    finally:
        # Fallback: 若布局未成功初始化，则强制创建默认布局。
        if not layout_initialized:
            try:
                layout = path_policy.ensure_output_layout(
                    run_root,
                    allow_nonempty_run_root=False,
                    allow_nonempty_run_root_reason=None,
                    override_applied=None
                )
                records_dir = layout["records_dir"]
                artifacts_dir = layout["artifacts_dir"]
                logs_dir = layout["logs_dir"]
            except Exception:
                pass
        # 绑定闭包最小合同字段，并产出 run_closure.json。
        run_meta["ended_at"] = time_utils.now_utc_iso_z()
        try:
            status.finalize_run(run_root, records_dir, artifacts_dir, run_meta)
        except Exception:
            # finalize_run 失败必须 fail-fast。
            raise
        if error is not None:
            raise error

    print(f"[Evaluate] [OK] Evaluation record written successfully")
    print(f"[Evaluate]   Record contains {len(record)} fields (15 fact source fields + {len(record) - 15} business fields)")



def main():
    """主流程。"""
    parser = argparse.ArgumentParser(
        description="Evaluate performance (readonly-threshold implementation)"
    )
    parser.add_argument(
        "--out",
        default="tmp/cli_smoke/evaluate_run",
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
        run_evaluate(args.out, args.config, args.override)
        sys.exit(0)
    except Exception as e:
        print(f"[Evaluate] [ERROR] Error: {e}", file=sys.stderr)
        hint = build_cli_config_migration_hint(e)
        if isinstance(hint, str) and hint:
            print(f"[Evaluate] [HINT] {hint}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
