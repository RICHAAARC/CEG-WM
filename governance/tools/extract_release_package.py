"""按外层 profile 生成研究交付候选目录。"""

from __future__ import annotations

import argparse
import ast
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class ExtractionProfile:
    """表示一个可执行的论文附件抽离 profile。"""

    profile_name: str
    include_paths: tuple[str, ...]
    exclude_parts: tuple[str, ...]
    package_readme_template: str
    optional_include_paths: tuple[str, ...] = ()


SENSITIVE_NAME_PARTS = (
    ".env",
    "credential",
    "secret",
    "private_key",
    "id_rsa",
    "id_ed25519",
)
SENSITIVE_CONFIG_KEY = re.compile(
    r"(?im)^\s*[\"']?(?:password|secret|token|api_key|access_key)[\"']?\s*[:=]"
)
ABSOLUTE_LOCAL_PATH = re.compile(
    r"(?<![A-Za-z0-9_])(?:/(?:home|Users|mnt|content|tmp|var|opt|root)/|[A-Za-z]:[\\/])"
)
CONFIG_SUFFIXES = {".json", ".yaml", ".yml", ".toml"}
PORTABILITY_SCAN_SUFFIXES = CONFIG_SUFFIXES | {".py", ".sh"}


PROFILES = {
    "minimal_method_package": ExtractionProfile(
        profile_name="minimal_method_package",
        include_paths=(
            "main",
            "pyproject.toml",
        ),
        optional_include_paths=(
            "configs/methods",
            "tests/unit/method",
            "tests/functional/method",
        ),
        package_readme_template="templates/release_readmes/minimal_method_package.md",
        exclude_parts=(
            ".agents",
            ".codex",
            "governance",
            "experiments",
            "scripts",
            "configs/baselines",
            "notebooks",
            "third_party",
            "audit_reports",
            "outputs",
            "__pycache__",
            ".pytest_cache",
        ),
    ),
    "paper_artifact_rebuild_package": ExtractionProfile(
        profile_name="paper_artifact_rebuild_package",
        include_paths=(
            "configs",
            "experiments/protocol",
            "paper_artifacts",
            "docs/guides/artifact_rebuild.md",
            "docs/reference/field_registry.md",
            "docs/reference/artifact_evidence.md",
            "tests/functional",
            "pyproject.toml",
        ),
        optional_include_paths=("scripts/artifact_rebuild",),
        package_readme_template="templates/release_readmes/paper_artifact_rebuild_package.md",
        exclude_parts=(
            ".agents",
            ".codex",
            "governance",
            "tests/integration",
            "tests/helpers",
            "notebooks",
            "third_party",
            "runtime",
            "main",
            "audit_reports",
            "outputs",
            "__pycache__",
            ".pytest_cache",
        ),
    ),
    "experiment_execution_package": ExtractionProfile(
        profile_name="experiment_execution_package",
        include_paths=(
            "main",
            "runtime",
            "experiments",
            "configs",
            "infrastructure",
            "tests/integration",
            "tests/smoke",
            "pyproject.toml",
        ),
        optional_include_paths=("scripts/experiment_execution",),
        package_readme_template="templates/release_readmes/experiment_execution_package.md",
        exclude_parts=(
            ".agents",
            ".codex",
            "governance",
            "docs",
            "notebooks",
            "paper_artifacts",
            "tests/unit",
            "tests/functional",
            "tests/formal",
            "tests/helpers",
            "tests/fixtures",
            "audit_reports",
            "outputs",
            "__pycache__",
            ".pytest_cache",
        ),
    ),
}


def should_skip(relative_path: Path, exclude_parts: Iterable[str]) -> bool:
    """按仓库相对前缀排除路径，并在任意层排除缓存目录。"""
    normalized = relative_path.as_posix()
    for excluded in exclude_parts:
        excluded_normalized = excluded.strip("/").replace("\\", "/")
        if excluded_normalized in {"__pycache__", ".pytest_cache"} and excluded_normalized in relative_path.parts:
            return True
        if normalized == excluded_normalized or normalized.startswith(f"{excluded_normalized}/"):
            return True
    return False


def iter_copy_candidates(root_path: Path, include_path: str, profile: ExtractionProfile) -> Iterable[Path]:
    """遍历某个 include path 下允许复制的文件。"""
    source = root_path / include_path
    if not source.exists():
        return
    if source.is_file():
        relative = source.relative_to(root_path)
        if not should_skip(relative, profile.exclude_parts):
            yield source
        return
    for path in source.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root_path)
        if should_skip(relative, profile.exclude_parts):
            continue
        yield path


def _safety_violations(root_path: Path, files: Iterable[Path]) -> list[dict[str, str]]:
    """检查候选文件名、敏感配置键和不可移植的本机绝对路径。"""
    violations: list[dict[str, str]] = []
    for source_file in files:
        relative = source_file.relative_to(root_path).as_posix()
        lowered_parts = [part.lower() for part in source_file.relative_to(root_path).parts]
        if any(marker in part for part in lowered_parts for marker in SENSITIVE_NAME_PARTS):
            violations.append({"path": relative, "reason": "sensitive_filename"})
            continue
        try:
            text = source_file.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if source_file.suffix.lower() in CONFIG_SUFFIXES and SENSITIVE_CONFIG_KEY.search(text):
            violations.append({"path": relative, "reason": "sensitive_config_key"})
        if source_file.suffix.lower() in PORTABILITY_SCAN_SUFFIXES and ABSOLUTE_LOCAL_PATH.search(text):
            violations.append({"path": relative, "reason": "absolute_local_path"})
    return violations


def _has_registered_baseline(root_path: Path) -> bool:
    registry = root_path / "docs" / "reference" / "baseline_registry.md"
    if not registry.exists():
        return False
    rows = [line.strip() for line in registry.read_text(encoding="utf-8").splitlines() if line.strip().startswith("|")]
    return any(
        len(cells := [cell.strip() for cell in row.strip("|").split("|")]) >= 7
        and cells[0] not in {"baseline_name", "---"}
        and not all(set(cell) <= {"-", ":"} for cell in cells)
        for row in rows
    )


def _has_substantive_definition(path: Path) -> bool:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, SyntaxError):
        return False
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        meaningful = []
        for child in node.body:
            if isinstance(child, ast.Pass):
                continue
            if isinstance(child, ast.Expr) and isinstance(child.value, ast.Constant) and isinstance(child.value.value, str):
                continue
            if (
                isinstance(child, ast.Raise)
                and isinstance(child.exc, ast.Call)
                and isinstance(child.exc.func, ast.Name)
                and child.exc.func.id == "NotImplementedError"
            ):
                continue
            meaningful.append(child)
        if meaningful:
            return True
    return False


def _has_test_function(path: Path) -> bool:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, SyntaxError):
        return False
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
        and any(isinstance(child, (ast.Assert, ast.Call)) for child in ast.walk(node))
        for node in ast.walk(tree)
    )


def _validation_violations(root_path: Path, profile_name: str, copied_files: list[str]) -> list[dict[str, str]]:
    """报告候选包距离可独立验证还缺少的实质实现或测试。"""
    violations: list[dict[str, str]] = []
    pyproject = root_path / "pyproject.toml"
    pyproject_text = pyproject.read_text(encoding="utf-8") if pyproject.exists() else ""
    project_section = re.search(r"(?ms)^\[project\]\s*$\n(?P<body>.*?)(?=^\[|\Z)", pyproject_text)
    build_section = re.search(r"(?ms)^\[build-system\]\s*$\n(?P<body>.*?)(?=^\[|\Z)", pyproject_text)
    if (
        project_section is None
        or re.search(r"(?m)^\s*name\s*=", project_section.group("body")) is None
        or build_section is None
        or re.search(r"(?m)^\s*requires\s*=", build_section.group("body")) is None
        or re.search(r"(?m)^\s*build-backend\s*=", build_section.group("body")) is None
    ):
        violations.append({"path": "pyproject.toml", "reason": "package_metadata_missing"})

    if profile_name == "minimal_method_package":
        implementation = [
            path
            for path in copied_files
            if path.startswith("main/")
            and path.endswith(".py")
            and path != "main/__init__.py"
            and _has_substantive_definition(root_path / path)
        ]
        method_tests = [
            path
            for path in copied_files
            if path.startswith("tests/")
            and Path(path).name.startswith("test_")
            and _has_test_function(root_path / path)
        ]
        if not implementation:
            violations.append({"path": "main", "reason": "method_implementation_missing"})
        if not method_tests:
            violations.append({"path": "tests", "reason": "method_package_tests_missing"})
    elif profile_name == "experiment_execution_package":
        execution_code = [
            path
            for path in copied_files
            if path.endswith(".py")
            and (path.startswith("runtime/") or path.startswith("experiments/runners/"))
            and not path.endswith("/__init__.py")
            and _has_substantive_definition(root_path / path)
        ]
        execution_tests = [
            path
            for path in copied_files
            if Path(path).name.startswith("test_")
            and (path.startswith("tests/integration/") or path.startswith("tests/smoke/"))
            and _has_test_function(root_path / path)
        ]
        if not execution_code:
            violations.append({"path": "runtime", "reason": "experiment_execution_implementation_missing"})
        if not execution_tests:
            violations.append({"path": "tests", "reason": "experiment_execution_tests_missing"})
    return violations


def extract_profile(
    root: str | Path,
    output: str | Path,
    profile_name: str,
    dry_run: bool = False,
    include_third_party: bool = False,
) -> dict:
    """按指定 profile 复制文件, 并返回抽离清单。"""
    root_path = Path(root).resolve()
    output_path = Path(output).resolve()
    if profile_name not in PROFILES:
        raise ValueError(f"不支持的抽离 profile: {profile_name}")
    profile = PROFILES[profile_name]

    copied_files: list[str] = []
    source_files: list[Path] = []
    missing_paths: list[str] = []
    include_paths = profile.include_paths + profile.optional_include_paths
    if include_third_party:
        if profile_name != "experiment_execution_package":
            raise ValueError("third_party 只允许加入实验执行候选包")
        include_paths += ("third_party",)
    for include_path in include_paths:
        source = root_path / include_path
        if not source.exists():
            if include_path not in profile.optional_include_paths:
                missing_paths.append(include_path)
            continue
        for source_file in iter_copy_candidates(root_path, include_path, profile):
            relative = source_file.relative_to(root_path)
            copied_files.append(relative.as_posix())
            source_files.append(source_file)

    readme_template = root_path / profile.package_readme_template
    if readme_template.exists():
        copied_files.append("README.md")
        source_files.append(readme_template)
    else:
        missing_paths.append(profile.package_readme_template)

    safety_violations = _safety_violations(root_path, source_files)
    if include_third_party and not _has_registered_baseline(root_path):
        safety_violations.append({"path": "third_party", "reason": "baseline_provenance_missing"})
    validation_violations = _validation_violations(root_path, profile_name, copied_files)
    release_candidate_ready = not missing_paths and not safety_violations and not validation_violations

    manifest = {
        "profile_name": profile.profile_name,
        "copied_files": sorted(copied_files),
        "missing_paths": missing_paths,
        "safety_violations": safety_violations,
        "validation_violations": validation_violations,
        "third_party_included": include_third_party,
        "release_candidate_ready": release_candidate_ready,
        "excluded_parts": list(profile.exclude_parts),
        "dry_run": dry_run,
    }
    if not dry_run:
        if safety_violations:
            reasons = ", ".join(f"{item['path']}:{item['reason']}" for item in safety_violations)
            raise ValueError(f"候选包安全检查失败: {reasons}")
        for source_file in source_files:
            relative = (
                Path("README.md")
                if source_file == readme_template
                else source_file.relative_to(root_path)
            )
            target_file = output_path / relative
            target_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_file, target_file)
        output_path.mkdir(parents=True, exist_ok=True)
        manifest_path = output_path / "extraction_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(description="按治理 profile 抽离发布或论文附件候选目录。")
    parser.add_argument(
        "--include-third-party",
        action="store_true",
        help="仅对实验执行候选包显式纳入 third_party；同时要求 baseline registry 已登记来源。",
    )
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILES),
        default="minimal_method_package",
        help="选择抽离 profile。",
    )
    parser.add_argument(
        "--root",
        default=".",
        help="仓库根目录。",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="输出目录。建议使用 release_packages/ 下的未提交目录。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只输出将要复制的文件清单, 不写入文件。",
    )
    return parser


def main() -> None:
    """命令行入口。"""
    parser = build_parser()
    args = parser.parse_args()
    manifest = extract_profile(
        args.root,
        args.output,
        args.profile,
        args.dry_run,
        include_third_party=args.include_third_party,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
