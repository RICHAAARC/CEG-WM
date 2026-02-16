"""
冻结签署入口点

功能说明：
- 以 run_root 产物证据包为输入执行 freeze sign-off。
- 执行最小静态审计集合作为补充证据。
- 校验 run_closure / manifest / cfg_audit / env_audit / path_audit 一致性。

Module type: Core innovation module
"""

import argparse
import hashlib
import json
import subprocess
import sys
import locale
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

# 添加 repo 到 sys.path 以支持 main 包导入
_scripts_dir = Path(__file__).resolve().parent
_repo_root = _scripts_dir.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from main.core import time_utils

SIGNOFF_REPORT_SCHEMA_VERSION = "v1"

MINIMUM_AUDIT_SCRIPTS = [
    "audits/audit_write_bypass_scan.py",
    "audits/audit_registry_injection_surface.py",
    "audits/audit_policy_path_semantics_binding.py",
    "audits/audit_path_policy_escape_rejection.py",
    "audits/audit_freeze_surface_integrity.py",
    "audits/audit_dangerous_exec_and_pickle_scan.py",
    "audits/audit_network_access_scan.py",
]


def execute_audit_script(script_path: Path, repo_root: Path) -> Dict[str, Any]:
    """
    功能：执行单个静态审计脚本并返回结构化结果。

    Execute one audit script and return a normalized result record.

    Args:
        script_path: Path to audit script.
        repo_root: Repository root directory.

    Returns:
        Structured audit result dictionary.
    """
    try:
        result = subprocess.run(
            [sys.executable, str(script_path), str(repo_root)],
            capture_output=True,
            text=False,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        return {
            "audit_id": f"ERROR.{script_path.stem}",
            "severity": "BLOCK",
            "result": "FAIL",
            "rule": f"audit script timeout: {script_path.name}",
            "impact": "audit timeout",
            "fix": "optimize script runtime",
            "evidence": {"timeout_seconds": 60},
        }
    except Exception as exc:
        return {
            "audit_id": f"ERROR.{script_path.stem}",
            "severity": "BLOCK",
            "result": "FAIL",
            "rule": f"audit script execution exception: {script_path.name}",
            "impact": "audit execution exception",
            "fix": "fix script runtime exception",
            "evidence": {"error": f"{type(exc).__name__}: {exc}"},
        }

    stdout_text = _decode_bytes(result.stdout)
    stderr_text = _decode_bytes(result.stderr)

    if result.returncode != 0:
        return {
            "audit_id": f"ERROR.{script_path.stem}",
            "severity": "BLOCK",
            "result": "FAIL",
            "rule": f"audit script failed: {script_path.name}",
            "impact": "audit script returned non-zero",
            "fix": "fix audit script exit behavior",
            "evidence": {
                "exit_code": result.returncode,
                "stdout_text": stdout_text[:1000],
                "stderr_text": stderr_text[:1000],
            },
        }

    output = stdout_text.strip()
    if not output:
        return {
            "audit_id": f"ERROR.{script_path.stem}",
            "severity": "BLOCK",
            "result": "FAIL",
            "rule": f"audit script produced empty output: {script_path.name}",
            "impact": "audit output missing",
            "fix": "ensure JSON output",
            "evidence": {
                "stdout_text": stdout_text[:1000],
                "stderr_text": stderr_text[:1000],
                "exit_code": result.returncode,
            },
        }

    try:
        parsed_any = json.loads(output)
    except json.JSONDecodeError as exc:
        return {
            "audit_id": f"ERROR.{script_path.stem}",
            "severity": "BLOCK",
            "result": "FAIL",
            "rule": f"audit output is not valid JSON: {script_path.name}",
            "impact": "audit output parse failed",
            "fix": "output valid JSON",
            "evidence": {"error": f"{type(exc).__name__}: {exc}"},
        }

    if not isinstance(parsed_any, dict):
        return {
            "audit_id": f"ERROR.{script_path.stem}",
            "severity": "BLOCK",
            "result": "FAIL",
            "rule": f"audit output root must be dict: {script_path.name}",
            "impact": "audit schema invalid",
            "fix": "return dict result",
            "evidence": {"root_type": str(type(parsed_any).__name__)},
        }

    parsed = cast(Dict[str, Any], parsed_any)
    return parsed


def _decode_bytes(data: Optional[bytes]) -> str:
    """
    功能：解码子进程字节输出。

    Decode subprocess bytes with stable fallback.

    Args:
        data: Bytes or None.

    Returns:
        Decoded text with replacement for invalid bytes.

    Raises:
        TypeError: If data is not bytes or None.
    """
    if data is None:
        return ""
    if not isinstance(data, (bytes, bytearray)):
        # data 类型不符合预期，必须 fail-fast。
        raise TypeError("data must be bytes or None")
    try:
        return bytes(data).decode("utf-8", errors="replace")
    except Exception:
        encoding = locale.getpreferredencoding(False)
        return bytes(data).decode(encoding, errors="replace")


def _load_json_file(path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    功能：加载 JSON 文件并返回结构化错误。

    Load JSON file and return parsed dict or error string.

    Args:
        path: JSON file path.

    Returns:
        Tuple of (parsed_dict_or_none, error_or_none).
    """
    if not path.exists() or not path.is_file():
        return None, f"missing_file: {path}"

    try:
        content_any = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, f"json_parse_failed: {type(exc).__name__}: {exc}"

    if not isinstance(content_any, dict):
        return None, f"json_root_not_dict: {path}"

    content = cast(Dict[str, Any], content_any)
    return content, None


def validate_run_root_evidence(run_root: Path) -> Dict[str, Any]:
    """
    功能：校验 run_root 产物证据包完整性与一致性。

    Validate required signoff evidence from run_root artifacts.

    Args:
        run_root: Target run root directory.

    Returns:
        Evidence validation report with status, errors and key summaries.
    """
    artifacts_dir = run_root / "artifacts"
    records_dir = run_root / "records"

    errors: List[str] = []
    warnings: List[str] = []

    run_closure_path = artifacts_dir / "run_closure.json"
    run_closure, run_closure_error = _load_json_file(run_closure_path)
    if run_closure_error is not None:
        errors.append(run_closure_error)
        return {
            "status": "failed",
            "errors": errors,
            "warnings": warnings,
            "paths": {
                "run_closure": str(run_closure_path),
            },
        }

    if run_closure is None:
        errors.append("run_closure_unexpected_none")
        return {
            "status": "failed",
            "errors": errors,
            "warnings": warnings,
            "paths": {
                "run_closure": str(run_closure_path),
            },
        }

    status_obj = run_closure.get("status")
    if not isinstance(status_obj, dict):
        errors.append("run_closure.status must be dict")
    else:
        status_obj_typed = cast(Dict[str, Any], status_obj)
        status_ok = status_obj_typed.get("ok")
        if not isinstance(status_ok, bool):
            errors.append("run_closure.status.ok must be bool")

    # cfg_audit 必须存在
    cfg_audit_path = artifacts_dir / "cfg_audit" / "cfg_audit.json"
    _, cfg_audit_error = _load_json_file(cfg_audit_path)
    if cfg_audit_error is not None:
        errors.append(cfg_audit_error)

    # env_audit 目录至少一份
    env_audits_dir = artifacts_dir / "env_audits"
    env_audit_files = sorted(env_audits_dir.glob("env_audit_*.json")) if env_audits_dir.exists() else []
    if len(env_audit_files) == 0:
        errors.append(f"missing_env_audit_files: {env_audits_dir}")

    # path_audit 在 run_closure 标记为 ok 时必须存在
    path_audit_status = run_closure.get("path_audit_status")
    path_audits_dir = artifacts_dir / "path_audits"
    path_audit_files = sorted(path_audits_dir.glob("path_audit_*.json")) if path_audits_dir.exists() else []
    if path_audit_status == "ok" and len(path_audit_files) == 0:
        errors.append(f"missing_path_audit_files: {path_audits_dir}")

    # records_bundle + anchors 一致性校验（若存在）
    records_bundle_obj = run_closure.get("records_bundle")
    facts_anchor_obj = run_closure.get("facts_anchor")

    manifest_path: Optional[Path] = None
    manifest: Optional[Dict[str, Any]] = None
    if isinstance(records_bundle_obj, dict):
        records_bundle_typed = cast(Dict[str, Any], records_bundle_obj)
        manifest_rel_path = records_bundle_typed.get("manifest_rel_path")
        if not isinstance(manifest_rel_path, str) or not manifest_rel_path:
            errors.append("records_bundle.manifest_rel_path must be non-empty str")
        else:
            manifest_path = run_root / manifest_rel_path
            manifest, manifest_error = _load_json_file(manifest_path)
            if manifest_error is not None:
                errors.append(manifest_error)
            else:
                manifest_typed = cast(Dict[str, Any], manifest)
                manifest_anchors = manifest_typed.get("anchors")
                if not isinstance(manifest_anchors, dict):
                    errors.append("records manifest anchors must be dict")
                elif isinstance(facts_anchor_obj, dict):
                    manifest_anchors_typed = cast(Dict[str, Any], manifest_anchors)
                    facts_anchor_typed = cast(Dict[str, Any], facts_anchor_obj)
                    for key in (
                        "contract_bound_digest",
                        "whitelist_bound_digest",
                        "policy_path_semantics_bound_digest",
                    ):
                        if manifest_anchors_typed.get(key) != facts_anchor_typed.get(key):
                            errors.append(f"anchor_mismatch: {key}")
                else:
                    errors.append("run_closure.facts_anchor must be dict when records_bundle exists")
    else:
        # 若 records 目录存在 record 文件但 records_bundle 缺失，记为错误。
        record_files = sorted(records_dir.glob("*.json")) if records_dir.exists() else []
        if len(record_files) > 0:
            errors.append("records_bundle missing while records files exist")

    result_status = "ok" if len(errors) == 0 else "failed"
    return {
        "status": result_status,
        "errors": errors,
        "warnings": warnings,
        "paths": {
            "run_closure": str(run_closure_path),
            "cfg_audit": str(cfg_audit_path),
            "env_audits_dir": str(env_audits_dir),
            "path_audits_dir": str(path_audits_dir),
            "manifest": str(manifest_path) if manifest_path is not None else "<absent>",
        },
    }


def compute_signoff_decision(
    static_audits: List[Dict[str, Any]],
    evidence_report: Dict[str, Any]
) -> Dict[str, Any]:
    """
    功能：综合静态审计与 run_root 证据校验结果生成签署决策。

    Compute final signoff decision from static and artifact evidence.

    Args:
        static_audits: Static audit result list.
        evidence_report: Artifact evidence validation report.

    Returns:
        Decision dictionary.
    """
    block_fails = [
        item for item in static_audits
        if item.get("result") == "FAIL" and item.get("severity") == "BLOCK"
    ]

    evidence_failed = evidence_report.get("status") != "ok"

    if block_fails or evidence_failed:
        decision = "BLOCK_FREEZE"
    else:
        decision = "ALLOW_FREEZE"

    reasons: List[Dict[str, Any]] = []
    for item in block_fails:
        reasons.append({
            "source": "static_audit",
            "audit_id": item.get("audit_id", "<absent>"),
            "rule": item.get("rule", "<absent>"),
            "impact": item.get("impact", "<absent>"),
            "fix": item.get("fix", "<absent>"),
        })

    if evidence_failed:
        reasons.append({
            "source": "run_root_evidence",
            "audit_id": "F1_F3_D2.evidence_bundle",
            "rule": "run_root evidence package must be complete and anchor-consistent",
            "impact": "freeze sign-off evidence is incomplete or inconsistent",
            "fix": "regenerate run outputs and ensure run_closure/manifest/audits are complete",
            "errors": evidence_report.get("errors", []),
        })

    return {
        "decision": decision,
        "blocking_reasons": reasons,
        "static_audit_counts": {
            "total": len(static_audits),
            "pass": sum(1 for item in static_audits if item.get("result") == "PASS"),
            "fail": sum(1 for item in static_audits if item.get("result") == "FAIL"),
            "na": sum(1 for item in static_audits if item.get("result") == "N.A."),
            "block_fail": len(block_fails),
        },
        "evidence_status": evidence_report.get("status"),
    }


def run_pytest(repo_root: Path) -> Dict[str, Any]:
    """
    功能：执行 pytest 并收集测试结果。

    Run pytest and collect test results for non-blocking audit evidence.

    Args:
        repo_root: Repository root directory.

    Returns:
        Test result dictionary with status, exit_code, and summary.
    """
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q"],
            cwd=str(repo_root),
            capture_output=True,
            text=False,
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        return {
            "status": "fail",
            "exit_code": -1,
            "command": "pytest -q",
            "error": "pytest timeout",
            "stdout_tail": "",
            "stderr_tail": "",
            "test_summary": "timeout",
        }
    except Exception as exc:
        return {
            "status": "fail",
            "exit_code": -1,
            "command": "pytest -q",
            "error": f"{type(exc).__name__}: {exc}",
            "stdout_tail": "",
            "stderr_tail": "",
            "test_summary": "exception",
        }

    stdout_text = _decode_bytes(result.stdout)
    stderr_text = _decode_bytes(result.stderr)

    status = "pass" if result.returncode == 0 else "fail"
    return {
        "status": status,
        "exit_code": result.returncode,
        "command": "pytest -q",
        "stdout_tail": stdout_text[-500:] if len(stdout_text) > 500 else stdout_text,
        "stderr_tail": stderr_text[-500:] if len(stderr_text) > 500 else stderr_text,
        "test_summary": "pytest executed" if status == "pass" else "pytest failed (see NON_BLOCK audit evidence)",
    }


def snapshot_frozen_constraints(repo_root: Path, run_root: Path) -> Dict[str, Any]:
    """
    功能：快照冻结约束文件至 artifacts/signoff。

    Snapshot frozen constraint files for external verification.
    Copies configs/{frozen_contracts,runtime_whitelist,policy_path_semantics}.yaml to artifacts/signoff.

    Args:
        repo_root: Repository root directory.
        run_root: Run root directory.

    Returns:
        Snapshot status dictionary with files manifest.
    """
    snapshot_dir = run_root / "artifacts" / "signoff" / "frozen_constraints_snapshot"
    try:
        snapshot_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        return {
            "status": "failed",
            "error": f"failed to create snapshot_dir: {exc}",
            "files": [],
        }

    constraint_files = [
        "frozen_contracts.yaml",
        "runtime_whitelist.yaml",
        "policy_path_semantics.yaml",
    ]

    files_manifest: List[Dict[str, Any]] = []
    errors: List[str] = []

    for fname in constraint_files:
        src_path = repo_root / "configs" / fname
        if not src_path.exists():
            errors.append(f"missing: {fname}")
            continue
        try:
            dst_path = snapshot_dir / fname
            dst_path.write_bytes(src_path.read_bytes())
            file_sha256 = hashlib.sha256(src_path.read_bytes()).hexdigest()
            file_size = src_path.stat().st_size
            files_manifest.append({
                "path": fname,
                "sha256": file_sha256,
                "size_bytes": file_size,
            })
        except Exception as exc:
            errors.append(f"copy failed: {fname}: {exc}")

    status = "ok" if not errors else "partial"
    result = {
        "status": status,
        "snapshot_dir": str(snapshot_dir),
        "files": files_manifest,
    }
    if errors:
        result["errors"] = errors

    return result


def extract_anchors_from_run_closure(run_root: Path) -> Dict[str, Any]:
    """
    功能：从 run_closure.json 抽取关键冻结锚点。

    Extract frozen contract anchors from run_closure for signoff verification.

    Args:
        run_root: Run root directory.

    Returns:
        Dictionary with extracted anchor fields.
    """
    run_closure_path = run_root / "artifacts" / "run_closure.json"
    anchors: Dict[str, Any] = {}

    try:
        if run_closure_path.exists():
            with open(run_closure_path, 'r', encoding='utf-8') as f:
                run_closure = json.load(f)
            if isinstance(run_closure, dict):
                anchor_fields = [
                    "contract_version", "contract_bound_digest",
                    "schema_version",
                    "whitelist_version", "whitelist_bound_digest",
                    "policy_path_semantics_version", "policy_path_semantics_bound_digest",
                ]
                for field in anchor_fields:
                    if field in run_closure:
                        anchors[field] = run_closure[field]
    except Exception:
        pass

    return anchors if anchors else {"status": "<absent>"}


def main() -> None:
    """
    功能：冻结签署主入口。

    Main entry for run_root-based freeze signoff.
    """
    parser = argparse.ArgumentParser(description="运行 freeze sign-off（基于 run_root 证据包）")
    parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
        help="运行输出根目录（必须包含 records/ 与 artifacts/）",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="仓库根目录（用于执行静态审计）",
    )
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    repo_root = args.repo_root.resolve()

    if not run_root.is_dir():
        print(f"错误：run_root 不存在或不是目录: {run_root}", file=sys.stderr)
        sys.exit(2)
    if not repo_root.is_dir():
        print(f"错误：repo_root 不存在或不是目录: {repo_root}", file=sys.stderr)
        sys.exit(2)

    scripts_dir = repo_root / "scripts"
    if not scripts_dir.is_dir():
        print(f"错误：scripts 目录不存在: {scripts_dir}", file=sys.stderr)
        sys.exit(2)

    print(f"[Freeze Signoff] run_root: {run_root}")
    print(f"[Freeze Signoff] repo_root: {repo_root}")

    # 执行 pytest（NON_BLOCK，仅记录）
    print(f"[Freeze Signoff] 执行 pytest -q...")
    pytest_result = run_pytest(repo_root)

    # 执行静态审计
    static_results: List[Dict[str, Any]] = []
    for relative_script in MINIMUM_AUDIT_SCRIPTS:
        script_path = scripts_dir / relative_script
        if not script_path.exists():
            static_results.append({
                "audit_id": f"ERROR.missing_{script_path.stem}",
                "severity": "BLOCK",
                "result": "FAIL",
                "rule": f"audit script missing: {relative_script}",
                "impact": "required audit script is missing",
                "fix": "restore required script",
                "evidence": {"path": str(script_path)},
            })
            continue
        result = execute_audit_script(script_path, repo_root)
        static_results.append(result)

    # 校验 run_root 证据包
    evidence_report = validate_run_root_evidence(run_root)

    # 快照冻结约束文件（增强证据）
    print(f"[Freeze Signoff] 快照冻结约束文件...")
    snapshot_result = snapshot_frozen_constraints(repo_root, run_root)

    # 计算签署决策
    decision = compute_signoff_decision(static_results, evidence_report)

    # 抽取冻结锚点
    anchors = extract_anchors_from_run_closure(run_root)

    # 生成新 schema 的 signoff_report
    signoff_payload: Dict[str, Any] = {
        "schema_version": SIGNOFF_REPORT_SCHEMA_VERSION,
        "freeze_signoff_decision": decision.get("decision"),
        "generated_at_utc": time_utils.now_utc_iso_z(),
        "run_root": str(run_root),
        "repo_root": str(repo_root),
        "tests": pytest_result,  # NON_BLOCK 段，不影响决策
        "audits": static_results,
        "blocking_reasons": decision.get("blocking_reasons", []),
        "run_root_evidence": evidence_report,
        "frozen_constraints_snapshot": snapshot_result,  # 增强证据（失败不阻断）
        "anchors": anchors,
        "minimum_audit_set": MINIMUM_AUDIT_SCRIPTS,
        "summary": {
            "decision": decision.get("decision"),
            "static_audit_counts": decision.get("static_audit_counts"),
            "evidence_status": decision.get("evidence_status"),
        },
    }

    signoff_dir = run_root / "artifacts" / "signoff"
    signoff_dir.mkdir(parents=True, exist_ok=True)
    report_path = signoff_dir / "signoff_report.json"
    report_path.write_text(json.dumps(signoff_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[Freeze Signoff] 报告路径: {report_path}")
    print(f"[Freeze Signoff] 决策: {decision.get('decision')}")
    print(f"[Freeze Signoff] pytest 状态: {pytest_result.get('status')} (NON_BLOCK)")

    if decision.get("decision") == "ALLOW_FREEZE":
        sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":
    main()
