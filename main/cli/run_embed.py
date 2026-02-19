"""
嵌入水印 CLI 入口

功能说明：
- 规范化输出目录路径，确保输出布局，加载合同与白名单，验证配置，解析实现，构造记录，绑定字段，写盘，并产出闭包。
- 包含详细的输入验证、错误处理与状态更新机制，确保健壮性与可维护性。
- 目前实现为 placeholder，未来会逐步完善业务逻辑与字段定义。
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime, timezone
import uuid

from main.cli import assert_module_execution


assert_module_execution("run_embed")

from main.core import time_utils
from main.core.contracts import (
    load_frozen_contracts,
    bind_contract_to_record,
    get_contract_interpretation,
    FrozenContracts
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
from main.core.errors import MissingRequiredFieldError, RunFailureReason
from main.registries import runtime_resolver
from main.diffusion.sd3 import pipeline_factory
from main.diffusion.sd3 import infer_runtime
from main.diffusion.sd3 import infer_trace
from main.watermarking.embed.orchestrator import run_embed_orchestrator
from main.cli.run_common import (
    set_value_by_field_path,
    set_failure_status,
    format_fact_sources_mismatch,
    bind_impl_identity_fields as _bind_impl_identity_fields,
    build_seed_audit,
    build_determinism_controls,
    normalize_nondeterminism_notes
)


def bind_impl_identity_fields(
    record: dict,
    identity: runtime_resolver.ImplIdentity,
    impl_set: runtime_resolver.BuiltImplSet,
    contracts
) -> None:
    """
    功能：绑定 impl_identity 相关字段。

    Bind impl identity, version, and digest fields into record.

    Args:
        record: Record dict to mutate.
        identity: Impl identity mapping.
        impl_set: Built implementation set.
        contracts: Loaded FrozenContracts.

    Returns:
        None.

    Raises:
        MissingRequiredFieldError: If required impl fields are missing.
        TypeError: If inputs are invalid.
    """
    return _bind_impl_identity_fields(record, identity, impl_set, contracts)


def run_embed(output_dir: str, config_path: str, overrides: list[str] | None = None) -> None:
    """
    功能：执行嵌入流程，本阶段为 placeholder。

    Execute embed workflow (placeholder implementation).

    Args:
        output_dir: Run root directory for records/artifacts.
        config_path: YAML config path.
        overrides: Optional CLI override args list.
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
        "command": "embed",
        "created_at_utc": started_at,
        "started_at": started_at,
        "output_dir_input": output_dir,
        "cfg_digest": "unknown",
        "policy_path": "unknown",
        "impl_id": "unknown",
        "impl_version": "unknown",
        "impl_identity": None,
        "impl_identity_digest": None,
        "pipeline_provenance_canon_sha256": "<absent>",
        "pipeline_status": "unbuilt",
        "pipeline_error": "<absent>",
        "pipeline_runtime_meta": None,
        "env_fingerprint_canon_sha256": "<absent>",
        "diffusers_version": "<absent>",
        "transformers_version": "<absent>",
        "safetensors_version": "<absent>",
        "model_provenance_canon_sha256": "<absent>",
        "status_ok": True,
        "status_reason": RunFailureReason.OK,
        "status_details": None,
        "manifest_rel_path": "unknown",
        "path_policy": {
            "allow_nonempty_run_root": False
        }
    }
    
    error = None
    pipeline_result = None
    try:
        # 加载事实源。
        print("[Embed] Loading fact sources...")
        contracts = load_frozen_contracts(config_loader.FROZEN_CONTRACTS_PATH)
        whitelist = load_runtime_whitelist(config_loader.RUNTIME_WHITELIST_PATH)
        semantics = load_policy_path_semantics(config_loader.POLICY_PATH_SEMANTICS_PATH)
        injection_scope_manifest = config_loader.load_injection_scope_manifest()

        # 绑定冻结锚点到 run_meta。
        print("[Embed] Binding freeze anchors to run_meta...")
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
        print("[Embed] Validating whitelist-semantics consistency...")
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

            seed_parts, seed_digest, seed_value, seed_rule_id = build_seed_audit(cfg, "embed")
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

            pipeline_result = pipeline_factory.build_pipeline_shell(cfg)
            run_meta["pipeline_provenance_canon_sha256"] = pipeline_result.get("pipeline_provenance_canon_sha256")
            run_meta["pipeline_status"] = pipeline_result.get("pipeline_status")
            run_meta["pipeline_error"] = pipeline_result.get("pipeline_error")
            run_meta["pipeline_runtime_meta"] = pipeline_result.get("pipeline_runtime_meta")
            run_meta["env_fingerprint_canon_sha256"] = pipeline_result.get("env_fingerprint_canon_sha256")
            run_meta["diffusers_version"] = pipeline_result.get("diffusers_version")
            run_meta["transformers_version"] = pipeline_result.get("transformers_version")
            run_meta["safetensors_version"] = pipeline_result.get("safetensors_version")
            run_meta["model_provenance_canon_sha256"] = pipeline_result.get("model_provenance_canon_sha256")
            
            # (7.7) Real Dataflow Smoke: 在 pipeline_result 之后调用 inference
            pipeline_obj = pipeline_result.get("pipeline_obj")
            device = cfg.get("device", "cpu")
            seed = seed_value
            
            inference_result = infer_runtime.run_sd3_inference(cfg, pipeline_obj, device, seed)
            inference_status = inference_result.get("inference_status")
            inference_error = inference_result.get("inference_error")
            inference_runtime_meta = inference_result.get("inference_runtime_meta")
            trajectory_evidence = inference_result.get("trajectory_evidence")
            
            # 构造 infer_trace 并计算 digest
            infer_trace_obj = infer_trace.build_infer_trace(
                cfg,
                inference_status,
                inference_error,
                inference_runtime_meta,
                trajectory_evidence
            )
            infer_trace_canon_sha256 = infer_trace.compute_infer_trace_canon_sha256(infer_trace_obj)
            
            # 将 inference 相关字段写入 run_meta（append-only）
            run_meta["infer_trace"] = infer_trace_obj
            run_meta["infer_trace_canon_sha256"] = infer_trace_canon_sha256
            run_meta["inference_status"] = inference_status
            run_meta["inference_error"] = inference_error
            run_meta["inference_runtime_meta"] = inference_runtime_meta
            run_meta["trajectory_evidence"] = trajectory_evidence
            
            # 提取 run_root 复用参数。
            allow_nonempty_run_root = cfg.get("allow_nonempty_run_root", False)
            allow_nonempty_run_root_reason = cfg.get("allow_nonempty_run_root_reason")
            override_applied_for_layout = cfg.get("override_applied")
            
            # 创建输出布局。
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
                impl_identity, impl_set, impl_set_capabilities_digest = runtime_resolver.build_runtime_impl_set_from_cfg(cfg)
            except Exception as exc:
                set_failure_status(run_meta, RunFailureReason.IMPL_RESOLVE_FAILED, exc)
                raise
            run_meta["impl_id"] = impl_identity.content_extractor_id
            run_meta["impl_version"] = impl_set.content_extractor.impl_version
            run_meta["impl_identity"] = impl_identity.as_dict()
            run_meta["impl_identity_digest"] = runtime_resolver.compute_impl_identity_digest(impl_identity)
            run_meta["impl_set_capabilities_digest"] = impl_set_capabilities_digest

            # 构造 embed record，本阶段为 placeholder。
            print("[Embed] Generating embed record (placeholder)...")
            record = run_embed_orchestrator(cfg, impl_set, cfg_digest, trajectory_evidence=trajectory_evidence)
            if not isinstance(record, dict):
                # record 类型不符合预期，必须 fail-fast。
                raise TypeError("orchestrator output must be dict")
            record["cfg_digest"] = cfg_digest
            record["policy_path"] = cfg["policy_path"]
            if isinstance(pipeline_result, dict):
                record["pipeline_impl_id"] = pipeline_result.get("pipeline_impl_id")
                record["pipeline_provenance"] = pipeline_result.get("pipeline_provenance")
                record["pipeline_provenance_canon_sha256"] = pipeline_result.get("pipeline_provenance_canon_sha256")
                record["pipeline_runtime_meta"] = pipeline_result.get("pipeline_runtime_meta")
                record["env_fingerprint_canon_sha256"] = pipeline_result.get("env_fingerprint_canon_sha256")
                record["diffusers_version"] = pipeline_result.get("diffusers_version")
                record["transformers_version"] = pipeline_result.get("transformers_version")
                record["safetensors_version"] = pipeline_result.get("safetensors_version")
                record["model_provenance_canon_sha256"] = pipeline_result.get("model_provenance_canon_sha256")
            
            # 将 inference 相关字段写入 record（append-only）
            record["infer_trace"] = run_meta.get("infer_trace")
            record["infer_trace_canon_sha256"] = run_meta.get("infer_trace_canon_sha256")
            record["inference_status"] = run_meta.get("inference_status")
            record["inference_error"] = run_meta.get("inference_error")
            record["inference_runtime_meta"] = run_meta.get("inference_runtime_meta")
            
            override_applied = cfg.get("override_applied")
            if override_applied is not None:
                record["override_applied"] = override_applied
            # 写入 impl_set_capabilities_digest。
            record["impl_set_capabilities_digest"] = impl_set_capabilities_digest

            schema.ensure_required_fields(record, cfg, interpretation)

            # 口径版本标识写入 run_meta。
            run_meta["thresholds_rule_id"] = record.get("thresholds_rule_id", "<absent>")
            run_meta["thresholds_rule_version"] = record.get("thresholds_rule_version", "<absent>")

            # 绑定 impl_identity 字段族。
            bind_impl_identity_fields(record, impl_identity, impl_set, contracts)

            # 绑定事实源字段。
            print("[Embed] Binding fact source fields...")
            bind_contract_to_record(record, contracts)
            bind_whitelist_to_record(record, whitelist)
            bind_semantics_to_record(record, semantics)

            try:
                schema.validate_record(record, interpretation=interpretation)
            except Exception as exc:
                set_failure_status(run_meta, RunFailureReason.RUNTIME_ERROR, exc)
                raise

            # 写盘，触发 freeze_gate.assert_prewrite。
            record_path = records_dir / "embed_record.json"
            path_policy.validate_output_target(record_path, "record", run_root)
            print(f"[Embed] Writing record to {record_path}...")
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

    print(f"[Embed] [OK] Embed record written successfully")
    print(f"[Embed]   Record contains {len(record)} fields (15 fact source fields + {len(record) - 15} business fields)")



def main():
    """主流程。"""
    parser = argparse.ArgumentParser(
        description="Embed watermark (placeholder implementation)"
    )
    parser.add_argument(
        "--out",
        default="tmp/cli_smoke/embed_run",
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
        run_embed(args.out, args.config, args.override)
        sys.exit(0)
    except Exception as e:
        print(f"[Embed] [ERROR] Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
