"""
文件目的：验证 00_main notebook 的 stage package 解析与最小运行合同。
Module type: General module

职责边界：
1. 仅覆盖 03 notebook 的 SOURCE_PACKAGE_PATH 初始化，以及 04 notebook 的手工优先 / 缺省自动发现闭环。
2. 通过直接解析 notebook JSON 与执行 precheck cell，验证 notebook 层语义，不修改 scripts/04_Release_And_Signoff.py 的 formal contract。
3. 不重跑主链，不触发真实 Colab 或 Google Drive 依赖。
"""

from __future__ import annotations

import json
import time
import zipfile
from pathlib import Path
from typing import Any, Dict

from scripts.notebook_runtime_common import (
    build_failure_diagnostics_filename,
    compute_file_sha256,
    finalize_stage_package,
    resolve_export_package_manifest_path,
    resolve_stage_package_input_or_discover,
    write_json_atomic,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_03_PATH = REPO_ROOT / "notebook" / "00_main" / "03_Experiment_Matrix_Full.ipynb"
NOTEBOOK_04_PATH = REPO_ROOT / "notebook" / "00_main" / "04_Release_And_Signoff.ipynb"


def _load_notebook(notebook_path: Path) -> Dict[str, Any]:
    """
    功能：读取 notebook JSON 文档。 

    Load one notebook JSON document.

    Args:
        notebook_path: Notebook path.

    Returns:
        Parsed notebook mapping.
    """
    notebook_obj = json.loads(notebook_path.read_text(encoding="utf-8"))
    if not isinstance(notebook_obj, dict):
        raise AssertionError(f"notebook root must be dict: {notebook_path}")
    return notebook_obj


def _find_code_cell_source(notebook_path: Path, marker: str) -> str:
    """
    功能：按标记文本定位 notebook code cell。 

    Locate one code-cell source block by a stable marker string.

    Args:
        notebook_path: Notebook path.
        marker: Marker text expected inside one code cell.

    Returns:
        Joined code-cell source string.
    """
    if not isinstance(marker, str) or not marker:
        raise TypeError("marker must be non-empty str")

    notebook_obj = _load_notebook(notebook_path)
    cells = notebook_obj.get("cells")
    if not isinstance(cells, list):
        raise AssertionError(f"cells must be list: {notebook_path}")
    for cell in cells:
        if not isinstance(cell, dict) or cell.get("cell_type") != "code":
            continue
        source = cell.get("source", [])
        if not isinstance(source, list):
            continue
        source_text = "\n".join(str(line) for line in source)
        if marker in source_text:
            return source_text
    raise AssertionError(f"code cell marker not found: {marker}")


def _create_formal_stage_package(base_dir: Path, *, stage_name: str, stage_run_id: str) -> Path:
    """
    功能：构造最小可发现 formal stage package。 

    Create one minimal discoverable formal stage package for notebook tests.

    Args:
        base_dir: Base test directory.
        stage_name: Stage name.
        stage_run_id: Stage run identifier.

    Returns:
        Formal package ZIP path.
    """
    package_root = base_dir / "package_root" / stage_name / stage_run_id
    export_root = base_dir / "exports" / stage_name / stage_run_id
    run_root = base_dir / "run_root" / stage_name / stage_run_id
    package_root.mkdir(parents=True, exist_ok=True)
    export_root.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)

    write_json_atomic(
        package_root / "artifacts" / "stage_manifest.json",
        {
            "stage_name": stage_name,
            "stage_run_id": stage_run_id,
            "stage_status": "ok",
            "workflow_summary_path": f"/drive/runs/{stage_name}/{stage_run_id}/artifacts/workflow_summary.json",
        },
    )
    write_json_atomic(
        package_root / "artifacts" / "workflow_summary.json",
        {
            "stage_name": stage_name,
            "stage_run_id": stage_run_id,
            "status": "ok",
        },
    )

    package_manifest = finalize_stage_package(
        stage_name=stage_name,
        stage_run_id=stage_run_id,
        package_root=package_root,
        export_root=export_root,
        source_stage_run_id=None,
        source_stage_package_path=None,
        package_manifest_path=run_root / "artifacts" / "package_manifest.json",
    )
    return Path(str(package_manifest["package_path"]))


def _create_failure_diagnostics_package(base_dir: Path, *, stage_name: str, stage_run_id: str) -> Path:
    """
    功能：构造最小 diagnostics package。 

    Create one minimal diagnostics package that must stay excluded from formal
    discovery.

    Args:
        base_dir: Base test directory.
        stage_name: Stage name.
        stage_run_id: Stage run identifier.

    Returns:
        Diagnostics package ZIP path.
    """
    export_root = base_dir / "exports" / stage_name / stage_run_id
    export_root.mkdir(parents=True, exist_ok=True)
    package_path = export_root / build_failure_diagnostics_filename(stage_name, stage_run_id)
    with zipfile.ZipFile(package_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "failure_diagnostics_summary.json",
            json.dumps(
                {
                    "stage_name": stage_name,
                    "stage_run_id": stage_run_id,
                    "stage_status": "failed",
                },
                ensure_ascii=False,
                indent=2,
            ),
        )
        archive.writestr(
            "artifacts/stage_manifest.json",
            json.dumps(
                {
                    "stage_name": stage_name,
                    "stage_run_id": stage_run_id,
                    "stage_status": "failed",
                },
                ensure_ascii=False,
                indent=2,
            ),
        )
    write_json_atomic(
        resolve_export_package_manifest_path(export_root),
        {
            "stage_name": stage_name,
            "stage_run_id": stage_run_id,
            "package_filename": package_path.name,
            "package_path": package_path.as_posix(),
            "package_sha256": compute_file_sha256(package_path),
            "package_role": "failure_diagnostics_package",
            "package_discovery_scope": "excluded_from_formal_discovery",
            "diagnostics_package_path": package_path.as_posix(),
        },
    )
    return package_path


def _execute_stage_04_precheck_cell(
    tmp_path: Path,
    *,
    drive_project_root: Path,
    stage_01_package_path: str | None,
    stage_02_package_path: str | None,
    stage_03_package_path: str | None,
    require_stage_02: bool,
    require_stage_03: bool,
) -> Dict[str, Any]:
    """
    功能：在测试命名空间中执行 stage 04 的 precheck code cell。 

    Execute the stage-04 notebook precheck code cell inside a test namespace.

    Args:
        tmp_path: Temporary pytest directory.
        drive_project_root: Fake drive project root.
        stage_01_package_path: Optional manual stage-01 package path.
        stage_02_package_path: Optional manual stage-02 package path.
        stage_03_package_path: Optional manual stage-03 package path.
        require_stage_02: Whether stage 02 is required.
        require_stage_03: Whether stage 03 is required.

    Returns:
        Execution namespace populated by the cell.
    """
    config_path = tmp_path / "default.yaml"
    script_path = tmp_path / "04_Release_And_Signoff.py"
    config_path.write_text("policy_path: content_np_geo_rescue\n", encoding="utf-8")
    script_path.write_text("pass\n", encoding="utf-8")

    precheck_source = _find_code_cell_source(NOTEBOOK_04_PATH, "PRECHECK_RESULTS: list[dict[str, Any]] = []")
    namespace: Dict[str, Any] = {
        "__builtins__": __builtins__,
        "Any": Any,
        "Path": Path,
        "json": json,
        "CONFIG_PATH": config_path,
        "SCRIPT_PATH": script_path,
        "DRIVE_PROJECT_ROOT": drive_project_root,
        "STAGE_01_PACKAGE_PATH": stage_01_package_path,
        "STAGE_02_PACKAGE_PATH": stage_02_package_path,
        "STAGE_03_PACKAGE_PATH": stage_03_package_path,
        "REQUIRE_STAGE_02": require_stage_02,
        "REQUIRE_STAGE_03": require_stage_03,
        "print_json": lambda *_args, **_kwargs: None,
    }
    exec(precheck_source, namespace)
    return namespace


def test_stage_03_notebook_defines_source_package_path_as_none() -> None:
    """
    功能：验证 stage 03 notebook 显式初始化 SOURCE_PACKAGE_PATH。 

    Verify the stage-03 notebook explicitly initializes SOURCE_PACKAGE_PATH to
    None so later cells do not raise NameError in the default flow.

    Args:
        None.

    Returns:
        None.
    """
    config_source = _find_code_cell_source(NOTEBOOK_03_PATH, 'NOTEBOOK_NAME = "03_Experiment_Matrix_Full"')
    namespace: Dict[str, Any] = {"__builtins__": __builtins__}
    exec(config_source, namespace)

    assert "SOURCE_PACKAGE_PATH" in namespace
    assert namespace["SOURCE_PACKAGE_PATH"] is None


def test_stage_04_precheck_auto_discovers_stage_01_package_when_manual_input_absent(tmp_path: Path) -> None:
    """
    功能：验证 stage 04 precheck 能自动发现 stage 01 formal package。 

    Verify the stage-04 notebook precheck discovers the latest valid stage-01
    package when no manual input is provided.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.
    """
    drive_project_root = tmp_path / "drive_project_root"
    stage_01_package_path = _create_formal_stage_package(
        drive_project_root,
        stage_name="01_Paper_Full_Cuda",
        stage_run_id="stage01_auto",
    )

    namespace = _execute_stage_04_precheck_cell(
        tmp_path,
        drive_project_root=drive_project_root,
        stage_01_package_path=None,
        stage_02_package_path=None,
        stage_03_package_path=None,
        require_stage_02=False,
        require_stage_03=False,
    )

    stage_01_summary = namespace["PRECHECK_STAGE_PACKAGE_SUMMARY"]["stage_01"]
    assert namespace["hard_fail"] == []
    assert stage_01_summary["manual_input_used"] is False
    assert stage_01_summary["required"] is True
    assert stage_01_summary["formal_package_valid"] is True
    assert stage_01_summary["selected_package_path"] == stage_01_package_path.as_posix()
    assert stage_01_summary["auto_discovered_package_path"] == stage_01_package_path.as_posix()
    assert namespace["RESOLVED_STAGE_01_PACKAGE_PATH"] == stage_01_package_path


def test_stage_04_precheck_allows_missing_optional_stage_02_package(tmp_path: Path) -> None:
    """
    功能：验证 stage 02 非必需时，缺失不应触发 hard fail。 

    Verify the stage-04 notebook precheck does not hard fail when stage 02 is
    optional and absent while the required stages resolve successfully.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.
    """
    drive_project_root = tmp_path / "drive_project_root"
    _create_formal_stage_package(
        drive_project_root,
        stage_name="01_Paper_Full_Cuda",
        stage_run_id="stage01_optional_stage02",
    )
    stage_03_package_path = _create_formal_stage_package(
        drive_project_root,
        stage_name="03_Experiment_Matrix_Full",
        stage_run_id="stage03_required",
    )

    namespace = _execute_stage_04_precheck_cell(
        tmp_path,
        drive_project_root=drive_project_root,
        stage_01_package_path=None,
        stage_02_package_path=None,
        stage_03_package_path=None,
        require_stage_02=False,
        require_stage_03=True,
    )

    stage_02_summary = namespace["PRECHECK_STAGE_PACKAGE_SUMMARY"]["stage_02"]
    stage_03_summary = namespace["PRECHECK_STAGE_PACKAGE_SUMMARY"]["stage_03"]
    assert namespace["hard_fail"] == []
    assert stage_02_summary["required"] is False
    assert stage_02_summary["formal_package_valid"] is False
    assert namespace["RESOLVED_STAGE_02_PACKAGE_PATH"] is None
    assert stage_03_summary["formal_package_valid"] is True
    assert namespace["RESOLVED_STAGE_03_PACKAGE_PATH"] == stage_03_package_path


def test_stage_04_precheck_allows_missing_optional_stage_03_package(tmp_path: Path) -> None:
    """
    功能：验证 stage 03 非必需时，缺失不应触发 hard fail。 

    Verify the stage-04 notebook precheck does not hard fail when stage 03 is
    optional and absent while the required stages resolve successfully.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.
    """
    drive_project_root = tmp_path / "drive_project_root"
    _create_formal_stage_package(
        drive_project_root,
        stage_name="01_Paper_Full_Cuda",
        stage_run_id="stage01_optional_stage03",
    )
    stage_02_package_path = _create_formal_stage_package(
        drive_project_root,
        stage_name="02_Parallel_Attestation_Statistics",
        stage_run_id="stage02_required",
    )

    namespace = _execute_stage_04_precheck_cell(
        tmp_path,
        drive_project_root=drive_project_root,
        stage_01_package_path=None,
        stage_02_package_path=None,
        stage_03_package_path=None,
        require_stage_02=True,
        require_stage_03=False,
    )

    stage_02_summary = namespace["PRECHECK_STAGE_PACKAGE_SUMMARY"]["stage_02"]
    stage_03_summary = namespace["PRECHECK_STAGE_PACKAGE_SUMMARY"]["stage_03"]
    assert namespace["hard_fail"] == []
    assert stage_02_summary["formal_package_valid"] is True
    assert namespace["RESOLVED_STAGE_02_PACKAGE_PATH"] == stage_02_package_path
    assert stage_03_summary["required"] is False
    assert stage_03_summary["formal_package_valid"] is False
    assert namespace["RESOLVED_STAGE_03_PACKAGE_PATH"] is None


def test_stage_04_helper_does_not_discover_diagnostics_package(tmp_path: Path) -> None:
    """
    功能：验证 diagnostics package 不会被误识别为 formal package。 

    Verify diagnostics packages remain excluded from notebook auto discovery
    when no formal package exists.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.
    """
    drive_project_root = tmp_path / "drive_project_root"
    _create_failure_diagnostics_package(
        drive_project_root,
        stage_name="01_Paper_Full_Cuda",
        stage_run_id="stage01_diagnostics_only",
    )

    resolution = resolve_stage_package_input_or_discover(
        None,
        drive_project_root / "exports" / "01_Paper_Full_Cuda",
        expected_stage_name="01_Paper_Full_Cuda",
    )

    assert resolution["selected_package_valid"] is False
    assert resolution["selected_package_path"] == "<absent>"
    assert resolution["candidates"] == []


def test_stage_04_helper_prefers_manual_package_input_over_auto_discovery(tmp_path: Path) -> None:
    """
    功能：验证手工指定 package 时优先于自动发现结果。 

    Verify manual package input wins over automatic discovery when both are
    available.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.
    """
    drive_project_root = tmp_path / "drive_project_root"
    manual_package_path = _create_formal_stage_package(
        drive_project_root,
        stage_name="01_Paper_Full_Cuda",
        stage_run_id="stage01_manual",
    )
    time.sleep(0.02)
    _create_formal_stage_package(
        drive_project_root,
        stage_name="01_Paper_Full_Cuda",
        stage_run_id="stage01_latest_auto",
    )

    resolution = resolve_stage_package_input_or_discover(
        str(manual_package_path),
        drive_project_root / "exports" / "01_Paper_Full_Cuda",
        expected_stage_name="01_Paper_Full_Cuda",
    )

    assert resolution["manual_input_used"] is True
    assert resolution["selected_package_valid"] is True
    assert resolution["selected_package_path"] == manual_package_path.as_posix()
    assert resolution["auto_discovered_package_path"] == "<skipped_manual_override>"
    assert resolution["selection_reason"] == "manual package path override"


def test_stage_04_execute_cell_uses_resolved_package_paths() -> None:
    """
    功能：验证 stage 04 执行 cell 复用 precheck 已解析的 package 路径。 

    Verify the stage-04 execute cell uses the pre-resolved package variables
    instead of re-reading the raw manual-input variables.

    Args:
        None.

    Returns:
        None.
    """
    execute_source = _find_code_cell_source(NOTEBOOK_04_PATH, "STAGE_RUN_ID = make_stage_run_id(NOTEBOOK_NAME)")

    assert "RESOLVED_STAGE_01_PACKAGE_PATH" in execute_source
    assert "str(RESOLVED_STAGE_01_PACKAGE_PATH)" in execute_source
    assert "str(Path(STAGE_01_PACKAGE_PATH))" not in execute_source
    assert "if RESOLVED_STAGE_02_PACKAGE_PATH is not None:" in execute_source
    assert "if RESOLVED_STAGE_03_PACKAGE_PATH is not None:" in execute_source
