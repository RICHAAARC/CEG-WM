"""
文件目的：执行最小论文复现流水线并生成 repro bundle。
Module type: Core innovation module

职责边界：
1. 仅编排既有 CLI 阶段（embed/detect/calibrate/evaluate）与 table 导出、signoff 复用。
2. 不修改算法、NP 阈值规则、digest 口径与冻结语义。
3. 所有 artifacts 写盘通过受控写盘接口并受 path_policy 约束。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


# 添加 repo 到 sys.path 以支持 main 包导入。
_scripts_dir = Path(__file__).resolve().parent
_repo_root = _scripts_dir.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from main.core import config_loader, time_utils
from main.core.records_io import (
    copy_file_controlled_unbound,
    write_artifact_json_unbound,
    write_artifact_text_unbound,
)
from main.evaluation import table_export
from main.policy import path_policy


REPRO_MANIFEST_REL_PATH = "artifacts/repro/run_manifest.json"
REPRO_REPORT_REL_PATH = "artifacts/repro/evaluation_report.json"
REPRO_TABLE_REL_PATH = "artifacts/repro/tables/metrics.csv"

REPRO_BUNDLE_DIR_REL = "artifacts/repro_bundle"
REPRO_BUNDLE_MANIFEST_REL_PATH = "artifacts/repro_bundle/manifest.json"
REPRO_BUNDLE_POINTERS_REL_PATH = "artifacts/repro_bundle/pointers.json"


def _compute_file_sha256(file_path: Path) -> str:
    """
    功能：计算文件 SHA256。 

    Compute SHA256 digest for a file.

    Args:
        file_path: Path to target file.

    Returns:
        SHA256 hex digest string.

    Raises:
        TypeError: If file_path type is invalid.
        FileNotFoundError: If file does not exist.
    """
    if not isinstance(file_path, Path):
        raise TypeError("file_path must be Path")
    if not file_path.exists() or not file_path.is_file():
        raise FileNotFoundError(f"file not found: {file_path}")

    hasher = hashlib.sha256()
    with file_path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _to_rel_path(target_path: Path, run_root: Path) -> str:
    """
    功能：将路径转换为 run_root 相对路径。 

    Convert an absolute path into run_root-relative POSIX path.

    Args:
        target_path: Absolute target path.
        run_root: Run root path.

    Returns:
        Relative path string in POSIX format.

    Raises:
        ValueError: If target_path is outside run_root.
    """
    relative_path = target_path.resolve().relative_to(run_root.resolve())
    return relative_path.as_posix()


def _load_json_dict(file_path: Path) -> Dict[str, Any]:
    """
    功能：读取并校验 JSON 根对象为 dict。 

    Load a JSON file and enforce dict root type.

    Args:
        file_path: Path to JSON file.

    Returns:
        Parsed JSON dictionary.

    Raises:
        TypeError: If root object is not a dict.
        FileNotFoundError: If file does not exist.
        ValueError: If JSON parse fails.
    """
    if not file_path.exists() or not file_path.is_file():
        raise FileNotFoundError(f"json file not found: {file_path}")
    try:
        parsed_obj = json.loads(file_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"failed to parse json: {file_path}: {type(exc).__name__}: {exc}") from exc
    if not isinstance(parsed_obj, dict):
        raise TypeError(f"json root must be dict: {file_path}")
    return parsed_obj


def _build_stage_command(
    stage_name: str,
    run_root: Path,
    config_path: Path,
    attack_protocol_path: Path,
    seeds: Optional[str],
    max_samples: Optional[int],
) -> List[str]:
    """
    功能：构建阶段 CLI 命令。 

    Build CLI command for a pipeline stage using existing command entry points.

    Args:
        stage_name: Stage name in {embed, detect, calibrate, evaluate}.
        run_root: Run root directory.
        config_path: Config YAML path.
        attack_protocol_path: Attack protocol YAML path.
        seeds: Optional seed spec.
        max_samples: Optional max samples for minimal run.

    Returns:
        CLI command argument list.

    Raises:
        ValueError: If stage_name is unsupported.
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

    if seeds is not None:
        command.extend(["--override", f"seed={json.dumps(seeds)}"])
    # Note: max_samples is not a valid override parameter and is handled by the caller
    if stage_name == "evaluate":
        command.extend(
            [
                "--override",
                f"evaluate.attack_protocol_path={json.dumps(str(attack_protocol_path))}",
            ]
        )

    return command


def _run_stage_command(
    stage_name: str,
    run_root: Path,
    config_path: Path,
    attack_protocol_path: Path,
    seeds: Optional[str],
    max_samples: Optional[int],
    repo_root: Path,
) -> str:
    """
    功能：执行单个流水线阶段命令。 

    Execute one stage command with fail-fast diagnostics.

    Args:
        stage_name: Stage name.
        run_root: Run root directory.
        config_path: Config YAML path.
        attack_protocol_path: Attack protocol YAML path.
        seeds: Optional seed spec.
        max_samples: Optional max sample count.
        repo_root: Repository root for subprocess cwd.

    Returns:
        Joined command string.

    Raises:
        RuntimeError: If subprocess returns non-zero.
    """
    command = _build_stage_command(
        stage_name=stage_name,
        run_root=run_root,
        config_path=config_path,
        attack_protocol_path=attack_protocol_path,
        seeds=seeds,
        max_samples=max_samples,
    )
    result = subprocess.run(
        command,
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "repro pipeline stage failed\n"
            f"  - stage: {stage_name}\n"
            f"  - command: {' '.join(command)}\n"
            f"  - run_root: {run_root}\n"
            f"  - stdout_tail: {result.stdout[-1000:]}\n"
            f"  - stderr_tail: {result.stderr[-1000:]}"
        )
    return " ".join(command)


def _run_signoff(run_root: Path, repo_root: Path) -> str:
    """
    功能：执行 freeze signoff 以复用审计与快照产物。 

    Run freeze signoff script before repro bundle generation.

    Args:
        run_root: Run root directory.
        repo_root: Repository root.

    Returns:
        Signoff command string.

    Raises:
        RuntimeError: If signoff fails.
    """
    command = [
        sys.executable,
        str(repo_root / "scripts" / "run_freeze_signoff.py"),
        "--run-root",
        str(run_root),
        "--repo-root",
        str(repo_root),
    ]
    result = subprocess.run(
        command,
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "freeze signoff failed\n"
            f"  - command: {' '.join(command)}\n"
            f"  - run_root: {run_root}\n"
            f"  - stdout_tail: {result.stdout[-1000:]}\n"
            f"  - stderr_tail: {result.stderr[-1000:]}"
        )
    return " ".join(command)


def _copy_with_digest_verification(
    run_root: Path,
    artifacts_dir: Path,
    src_path: Path,
    dst_path: Path,
) -> Dict[str, Any]:
    """
    功能：受控复制文件并验证源/目标 SHA256 一致。 

    Copy file via controlled interface and verify source/destination hashes are equal.

    Args:
        run_root: Run root directory.
        artifacts_dir: Artifacts directory.
        src_path: Source file path.
        dst_path: Destination file path.

    Returns:
        Copy metadata with relative paths and hashes.

    Raises:
        RuntimeError: If hash mismatch occurs.
    """
    copy_file_controlled_unbound(
        run_root=run_root,
        artifacts_dir=artifacts_dir,
        src_path=src_path,
        dst_path=dst_path,
        kind="artifact",
    )

    source_sha256 = _compute_file_sha256(src_path)
    dest_sha256 = _compute_file_sha256(dst_path)
    if source_sha256 != dest_sha256:
        raise RuntimeError(
            "copy hash mismatch\n"
            f"  - source: {src_path}\n"
            f"  - destination: {dst_path}\n"
            f"  - source_sha256: {source_sha256}\n"
            f"  - dest_sha256: {dest_sha256}"
        )

    return {
        "source_path": _to_rel_path(src_path, run_root),
        "dest_path": _to_rel_path(dst_path, run_root),
        "source_sha256": source_sha256,
        "dest_sha256": dest_sha256,
    }


def _build_source_pointers(run_root: Path) -> Dict[str, Any]:
    """
    功能：生成源工件指针表（相对路径 + sha256）。 

    Build source artifact pointers without changing source payloads.

    Args:
        run_root: Run root directory.

    Returns:
        Pointers dictionary.

    Raises:
        FileNotFoundError: If required source artifact is missing.
    """
    required_paths = [
        run_root / "artifacts" / "run_closure.json",
        run_root / "artifacts" / "records_manifest.json",
        run_root / "artifacts" / "thresholds" / "thresholds_artifact.json",
        run_root / "artifacts" / "thresholds" / "threshold_metadata_artifact.json",
        run_root / "records" / "evaluate_record.json",
        run_root / "artifacts" / "signoff" / "signoff_report.json",
    ]

    files: List[Dict[str, Any]] = []
    for file_path in required_paths:
        if not file_path.exists() or not file_path.is_file():
            raise FileNotFoundError(f"required source artifact missing: {file_path}")
        files.append(
            {
                "path": _to_rel_path(file_path, run_root),
                "sha256": _compute_file_sha256(file_path),
            }
        )

    return {
        "schema_version": "v1",
        "files": files,
    }


def _build_repro_bundle(
    run_root: Path,
    report_obj: Dict[str, Any],
    stage_commands: List[str],
    signoff_command: str,
) -> Tuple[Path, Dict[str, Any]]:
    """
    功能：构建论文复现包目录与 manifest/pointers。 

    Build repro_bundle directory from signoff outputs and evaluate/table artifacts.

    Args:
        run_root: Run root directory.
        report_obj: Evaluation report dictionary.
        stage_commands: Executed stage command list.
        signoff_command: Executed signoff command.

    Returns:
        Tuple of (bundle_manifest_path, bundle_manifest_dict).

    Raises:
        RuntimeError: If required artifacts are missing or inconsistent.
    """
    artifacts_dir = run_root / "artifacts"

    # 源文件路径。
    signoff_report_src = artifacts_dir / "signoff" / "signoff_report.json"
    signoff_bundle_src_dir = artifacts_dir / "signoff" / "signoff_bundle"
    snapshot_src_dir = artifacts_dir / "signoff" / "frozen_constraints_snapshot"
    repro_report_src = artifacts_dir / "repro" / "evaluation_report.json"
    repro_table_src = artifacts_dir / "repro" / "tables" / "metrics.csv"

    if not signoff_report_src.exists():
        raise RuntimeError(f"required signoff report missing: {signoff_report_src}")
    if not snapshot_src_dir.exists() or not snapshot_src_dir.is_dir():
        raise RuntimeError(f"required signoff snapshot dir missing: {snapshot_src_dir}")
    if not repro_report_src.exists() or not repro_report_src.is_file():
        raise RuntimeError(f"required repro report missing: {repro_report_src}")
    if not repro_table_src.exists() or not repro_table_src.is_file():
        raise RuntimeError(f"required repro table missing: {repro_table_src}")

    copy_records: List[Dict[str, Any]] = []

    # 复制 audit_report。
    copy_records.append(
        _copy_with_digest_verification(
            run_root=run_root,
            artifacts_dir=artifacts_dir,
            src_path=signoff_report_src,
            dst_path=artifacts_dir / "repro_bundle" / "audit_report.json",
        )
    )

    # 复制 configs snapshot。
    snapshot_files = sorted([path for path in snapshot_src_dir.rglob("*") if path.is_file()])
    if not snapshot_files:
        raise RuntimeError(f"signoff snapshot is empty: {snapshot_src_dir}")
    for src_path in snapshot_files:
        relative_part = src_path.relative_to(snapshot_src_dir)
        dst_path = artifacts_dir / "repro_bundle" / "configs_snapshot" / relative_part
        copy_records.append(
            _copy_with_digest_verification(
                run_root=run_root,
                artifacts_dir=artifacts_dir,
                src_path=src_path,
                dst_path=dst_path,
            )
        )

    # 复制 signoff_bundle（若存在）。
    signoff_bundle_present = signoff_bundle_src_dir.exists() and signoff_bundle_src_dir.is_dir()
    if signoff_bundle_present:
        signoff_bundle_files = sorted([path for path in signoff_bundle_src_dir.rglob("*") if path.is_file()])
        for src_path in signoff_bundle_files:
            relative_part = src_path.relative_to(signoff_bundle_src_dir)
            dst_path = artifacts_dir / "repro_bundle" / "signoff_bundle" / relative_part
            copy_records.append(
                _copy_with_digest_verification(
                    run_root=run_root,
                    artifacts_dir=artifacts_dir,
                    src_path=src_path,
                    dst_path=dst_path,
                )
            )

    # 复制 evaluation report 与 table。
    copy_records.append(
        _copy_with_digest_verification(
            run_root=run_root,
            artifacts_dir=artifacts_dir,
            src_path=repro_report_src,
            dst_path=artifacts_dir / "repro_bundle" / "evaluation" / "report.json",
        )
    )
    copy_records.append(
        _copy_with_digest_verification(
            run_root=run_root,
            artifacts_dir=artifacts_dir,
            src_path=repro_table_src,
            dst_path=artifacts_dir / "repro_bundle" / "evaluation" / "tables" / "metrics.csv",
        )
    )

    # 构造 pointers（源工件只读引用）。
    pointers_obj = _build_source_pointers(run_root)
    pointers_obj["copy_records"] = copy_records
    pointers_path = artifacts_dir / "repro_bundle" / "pointers.json"
    write_artifact_json_unbound(
        run_root=run_root,
        artifacts_dir=artifacts_dir,
        path=str(pointers_path),
        obj=pointers_obj,
        indent=2,
        ensure_ascii=False,
    )

    # 构造 bundle manifest。
    run_root_resolved = run_root.resolve()
    try:
        run_root_relative = run_root_resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        run_root_relative = run_root_resolved.as_posix()

    manifest_obj = {
        "schema_version": "v1",
        "created_at_utc": time_utils.now_utc_iso_z(),
        "run_root": {
            "absolute": str(run_root_resolved),
            "relative_to_cwd": run_root_relative,
        },
        "cfg_digest": report_obj.get("cfg_digest", "<absent>"),
        "plan_digest": report_obj.get("plan_digest", "<absent>"),
        "thresholds_digest": report_obj.get("thresholds_digest", "<absent>"),
        "threshold_metadata_digest": report_obj.get("threshold_metadata_digest", "<absent>"),
        "impl_digest": report_obj.get("impl_digest", "<absent>"),
        "fusion_rule_version": report_obj.get("fusion_rule_version", "<absent>"),
        "attack_protocol_version": report_obj.get("attack_protocol_version", "<absent>"),
        "attack_protocol_digest": report_obj.get("attack_protocol_digest", "<absent>"),
        "policy_path": report_obj.get("policy_path", "<absent>"),
        "commands": {
            "stages": stage_commands,
            "signoff": signoff_command,
        },
        "signoff_bundle_present": signoff_bundle_present,
        "pointers_rel_path": _to_rel_path(pointers_path, run_root),
    }

    manifest_path = artifacts_dir / "repro_bundle" / "manifest.json"
    write_artifact_json_unbound(
        run_root=run_root,
        artifacts_dir=artifacts_dir,
        path=str(manifest_path),
        obj=manifest_obj,
        indent=2,
        ensure_ascii=False,
    )

    return manifest_path, manifest_obj


def run_repro_pipeline(
    run_root: Path,
    config_path: Path,
    attack_protocol_path: Path,
    seeds: Optional[str],
    max_samples: Optional[int],
    repo_root: Path,
    stage_runner: Optional[
        Callable[[str, Path, Path, Path, Optional[str], Optional[int], Path], str]
    ] = None,
    signoff_runner: Optional[Callable[[Path, Path], str]] = None,
) -> Dict[str, Any]:
    """
    功能：执行最小复现流水线并生成 repro 与 repro_bundle 工件。 

    Run protocol-driven minimal pipeline: embed -> detect -> calibrate -> evaluate -> table export
    and generate reproducibility artifacts under run_root/artifacts.

    Args:
        run_root: Target run root directory.
        config_path: Main config YAML path.
        attack_protocol_path: Attack protocol YAML path.
        seeds: Optional seed specification.
        max_samples: Optional small-sample cap for minimal runs.
        repo_root: Repository root path.
        stage_runner: Optional stage execution callback for testing.
        signoff_runner: Optional signoff execution callback for testing.

    Returns:
        Summary dictionary containing manifest paths and key anchors.

    Raises:
        RuntimeError: If any required stage/artifact check fails.
    """
    if not isinstance(run_root, Path):
        raise TypeError("run_root must be Path")
    if not isinstance(config_path, Path):
        raise TypeError("config_path must be Path")
    if not isinstance(attack_protocol_path, Path):
        raise TypeError("attack_protocol_path must be Path")
    if not isinstance(repo_root, Path):
        raise TypeError("repo_root must be Path")

    run_root = path_policy.derive_run_root(run_root)
    layout = path_policy.ensure_output_layout(
        run_root,
        allow_nonempty_run_root=True,
        allow_nonempty_run_root_reason="repro_pipeline",
        override_applied={"allow_nonempty_run_root": True},
    )
    artifacts_dir = layout["artifacts_dir"]

    # 使用唯一 YAML 加载入口加载 cfg 与 attack protocol（用于溯源记录）。
    cfg_obj, cfg_provenance = config_loader.load_yaml_with_provenance(config_path)
    attack_protocol_obj, attack_protocol_provenance = config_loader.load_yaml_with_provenance(attack_protocol_path)

    runner = stage_runner if stage_runner is not None else _run_stage_command
    signoff_exec = signoff_runner if signoff_runner is not None else _run_signoff

    stage_commands: List[str] = []
    for stage_name in ["embed", "detect", "calibrate", "evaluate"]:
        command_str = runner(
            stage_name,
            run_root,
            config_path,
            attack_protocol_path,
            seeds,
            max_samples,
            repo_root,
        )
        stage_commands.append(command_str)

    # table_export：从 evaluate_record 读取 evaluation_report 导出 CSV。
    evaluate_record_path = run_root / "records" / "evaluate_record.json"
    evaluate_record = _load_json_dict(evaluate_record_path)
    report_obj = evaluate_record.get("evaluation_report")
    if not isinstance(report_obj, dict):
        raise RuntimeError(
            "evaluation_report missing in evaluate_record\n"
            f"  - path: {evaluate_record_path}"
        )

    csv_content = table_export.export_metrics_to_csv(report_obj)

    # 写入 repro 工件。
    repro_report_path = run_root / REPRO_REPORT_REL_PATH
    repro_table_path = run_root / REPRO_TABLE_REL_PATH

    write_artifact_json_unbound(
        run_root=run_root,
        artifacts_dir=artifacts_dir,
        path=str(repro_report_path),
        obj=report_obj,
        indent=2,
        ensure_ascii=False,
    )
    write_artifact_text_unbound(
        run_root=run_root,
        artifacts_dir=artifacts_dir,
        path=str(repro_table_path),
        content=csv_content,
    )

    # 运行 signoff（复用 signoff 产物并执行一致性校验）。
    signoff_command = signoff_exec(run_root, repo_root)

    # 构建 repro bundle。
    bundle_manifest_path, bundle_manifest = _build_repro_bundle(
        run_root=run_root,
        report_obj=report_obj,
        stage_commands=stage_commands,
        signoff_command=signoff_command,
    )

    run_manifest_obj = {
        "schema_version": "v1",
        "created_at_utc": time_utils.now_utc_iso_z(),
        "run_root": {
            "absolute": str(run_root.resolve()),
            "relative_to_cwd": _to_rel_path(run_root.resolve(), Path.cwd().resolve())
            if run_root.resolve().is_relative_to(Path.cwd().resolve())
            else str(run_root.resolve()),
        },
        "inputs": {
            "config_path": str(config_path.resolve()),
            "attack_protocol_path": str(attack_protocol_path.resolve()),
            "seeds": seeds,
            "max_samples": max_samples,
        },
        "config_provenance": {
            "path": cfg_provenance.path,
            "file_sha256": cfg_provenance.file_sha256,
            "canon_sha256": cfg_provenance.canon_sha256,
        },
        "attack_protocol_provenance": {
            "path": attack_protocol_provenance.path,
            "file_sha256": attack_protocol_provenance.file_sha256,
            "canon_sha256": attack_protocol_provenance.canon_sha256,
            "version": attack_protocol_obj.get("version", "<absent>") if isinstance(attack_protocol_obj, dict) else "<absent>",
        },
        "stage_commands": stage_commands,
        "signoff_command": signoff_command,
        "anchors": {
            "cfg_digest": report_obj.get("cfg_digest", "<absent>"),
            "plan_digest": report_obj.get("plan_digest", "<absent>"),
            "thresholds_digest": report_obj.get("thresholds_digest", "<absent>"),
            "threshold_metadata_digest": report_obj.get("threshold_metadata_digest", "<absent>"),
            "impl_digest": report_obj.get("impl_digest", "<absent>"),
            "fusion_rule_version": report_obj.get("fusion_rule_version", "<absent>"),
            "attack_protocol_version": report_obj.get("attack_protocol_version", "<absent>"),
            "attack_protocol_digest": report_obj.get("attack_protocol_digest", "<absent>"),
            "policy_path": report_obj.get("policy_path", "<absent>"),
        },
        "outputs": {
            "repro_manifest": REPRO_MANIFEST_REL_PATH,
            "repro_report": REPRO_REPORT_REL_PATH,
            "repro_table": REPRO_TABLE_REL_PATH,
            "repro_bundle_manifest": _to_rel_path(bundle_manifest_path, run_root),
        },
    }

    run_manifest_path = run_root / REPRO_MANIFEST_REL_PATH
    write_artifact_json_unbound(
        run_root=run_root,
        artifacts_dir=artifacts_dir,
        path=str(run_manifest_path),
        obj=run_manifest_obj,
        indent=2,
        ensure_ascii=False,
    )

    return {
        "run_manifest_path": str(run_manifest_path),
        "bundle_manifest_path": str(bundle_manifest_path),
        "cfg_digest": report_obj.get("cfg_digest", "<absent>"),
        "thresholds_digest": report_obj.get("thresholds_digest", "<absent>"),
        "threshold_metadata_digest": report_obj.get("threshold_metadata_digest", "<absent>"),
        "attack_protocol_version": report_obj.get("attack_protocol_version", "<absent>"),
        "policy_path": report_obj.get("policy_path", "<absent>"),
        "bundle_manifest": bundle_manifest,
    }


def main() -> None:
    """
    功能：命令行入口。 

    CLI entry for repro pipeline execution.

    Args:
        None.

    Returns:
        None.
    """
    parser = argparse.ArgumentParser(description="Run minimal repro pipeline and generate repro bundle")
    parser.add_argument("--run-root", required=True, help="Run root directory")
    parser.add_argument("--config", required=True, help="Main config path")
    parser.add_argument(
        "--attack-protocol",
        default=config_loader.ATTACK_PROTOCOL_PATH,
        help="Attack protocol YAML path",
    )
    parser.add_argument("--seeds", default=None, help="Optional seeds spec")
    parser.add_argument("--max-samples", type=int, default=16, help="Optional max sample cap")
    parser.add_argument(
        "--repo-root",
        default=str(_repo_root),
        help="Repository root directory",
    )
    args = parser.parse_args()

    try:
        result = run_repro_pipeline(
            run_root=Path(args.run_root),
            config_path=Path(args.config),
            attack_protocol_path=Path(args.attack_protocol),
            seeds=args.seeds,
            max_samples=args.max_samples,
            repo_root=Path(args.repo_root),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        sys.exit(0)
    except Exception as exc:
        print(
            f"[ReproPipeline][ERROR] {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
