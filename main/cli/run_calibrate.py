"""
阈值校准 CLI 入口

功能说明：
- 本模块提供一个 CLI 入口，用于执行阈值校准流程。
- 当前实现为 placeholder，主要演示如何加载事实源、验证一致性、构造记录并写盘。
- 未来将替换为实际的校准逻辑，当前重点在于展示整体流程和接口设计。
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Any
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
from main.core import config_loader
from main.core import schema
from main.core import status
from main.policy import path_policy
from main.registries import runtime_resolver
from main.watermarking.detect.orchestrator import run_calibrate_orchestrator
from main.core.errors import RunFailureReason
from main.cli.run_common import (
    bind_impl_identity_fields,
    set_failure_status,
    format_fact_sources_mismatch
)


def run_calibrate(output_dir: str, config_path: str, overrides: list[str] | None = None) -> None:
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

    # 创建 run_root 与最小 run_meta。
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
    record = None
    try:
        # 加载事实源。
        print("[Calibrate] Loading fact sources...")
        contracts = load_frozen_contracts(config_loader.FROZEN_CONTRACTS_PATH)
        whitelist = load_runtime_whitelist(config_loader.RUNTIME_WHITELIST_PATH)
        semantics = load_policy_path_semantics(config_loader.POLICY_PATH_SEMANTICS_PATH)

        # 绑定冻结锚点到 run_meta（A1/A7 修复）。
        print("[Calibrate] Binding freeze anchors to run_meta...")
        status.bind_freeze_anchors_to_run_meta(run_meta, contracts, whitelist, semantics)

        # 生成事实源快照用于后续一致性校验。
        snapshot = records_io.build_fact_sources_snapshot(
            contracts,
            whitelist,
            semantics
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
                overrides=overrides
            )
        except Exception as exc:
            set_failure_status(run_meta, RunFailureReason.CONFIG_INVALID, exc)
            raise
        run_meta["cfg_digest"] = cfg_digest
        run_meta["policy_path"] = cfg["policy_path"]

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
            logs_dir
        ):
            bound_fact_sources = records_io.get_bound_fact_sources()
            run_meta["bound_fact_sources"] = bound_fact_sources
            if bound_fact_sources != snapshot:
                # 事实源快照不一致，必须 fail-fast。
                raise ValueError(format_fact_sources_mismatch(snapshot, bound_fact_sources))

            try:
                impl_identity, impl_set, impl_set_capabilities_digest = runtime_resolver.build_runtime_impl_set_from_cfg(cfg)
            except Exception as exc:
                set_failure_status(run_meta, RunFailureReason.IMPL_RESOLVE_FAILED, exc)
                raise
            run_meta["impl_id"] = impl_identity.content_extractor_id
            run_meta["impl_version"] = impl_set.content_extractor.impl_version
            run_meta["impl_identity"] = impl_identity.as_dict()
            run_meta["impl_identity_digest"] = runtime_resolver.compute_impl_identity_digest(impl_identity)
            run_meta["impl_set_capabilities_digest"] = impl_set_capabilities_digest

            # 构造 calibration record，本阶段为 placeholder。
            print("[Calibrate] Generating calibration record (placeholder)...")
            record = run_calibrate_orchestrator(cfg, impl_set)
            record["cfg_digest"] = cfg_digest
            record["policy_path"] = cfg["policy_path"]
            override_applied = cfg.get("override_applied")
            if override_applied is not None:
                record["override_applied"] = override_applied
            # 写入 impl_set_capabilities_digest。
            record["impl_set_capabilities_digest"] = impl_set_capabilities_digest

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
        description="Calibrate thresholds (placeholder implementation)"
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
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
