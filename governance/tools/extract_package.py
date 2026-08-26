"""Extract minimal, governance-free research packages."""

from __future__ import annotations

import argparse
import ast
import json
import re
import shutil
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True, slots=True)
class ExtractionProfile:
    name: str
    required_paths: tuple[str, ...]
    optional_paths: tuple[str, ...]
    excluded_prefixes: tuple[str, ...]
    readme_template: str


COMMON_EXCLUSIONS = (
    ".agents",
    ".codex",
    ".git",
    "governance",
    "notebooks",
    "models",
    "data",
    "outputs",
    "audit_reports",
    "release_packages",
    "__pycache__",
    ".pytest_cache",
    ".egg-info",
)

PROFILES = {
    "content_chain_execution": ExtractionProfile(
        name="content_chain_execution",
        required_paths=(
            "src",
            "experiments",
            "configs/content_chain",
            "tests/unit",
            "tests/integration",
            "pyproject.toml",
        ),
        optional_paths=(),
        excluded_prefixes=COMMON_EXCLUSIONS,
        readme_template="templates/packages/content_chain_execution.md",
    ),
}

SENSITIVE_NAME_MARKERS = (".env", "credential", "secret", "private_key", "id_rsa", "id_ed25519")
SENSITIVE_CONFIG_KEY = re.compile(
    r"(?im)^\s*[\"']?(?:password|secret|token|api_key|access_key)[\"']?\s*[:=]"
)
ABSOLUTE_LOCAL_PATH = re.compile(
    r"(?<![A-Za-z0-9_])(?:/(?:home|Users|mnt|content|tmp|var|opt|root)/|[A-Za-z]:[\\/])"
)
TEXT_SCAN_SUFFIXES = {".json", ".yaml", ".yml", ".toml", ".py", ".sh"}


def _is_excluded(relative: Path, excluded_prefixes: Iterable[str]) -> bool:
    normalized = relative.as_posix()
    for raw_prefix in excluded_prefixes:
        prefix = raw_prefix.strip("/").replace("\\", "/")
        if prefix in {"__pycache__", ".pytest_cache"} and prefix in relative.parts:
            return True
        if prefix == ".egg-info" and any(part.endswith(prefix) for part in relative.parts):
            return True
        if normalized == prefix or normalized.startswith(f"{prefix}/"):
            return True
    return False


def _iter_files(root: Path, include_path: str, profile: ExtractionProfile) -> Iterable[Path]:
    source = root / include_path
    if source.is_file():
        relative = source.relative_to(root)
        if not _is_excluded(relative, profile.excluded_prefixes):
            yield source
        return
    for path in sorted(source.rglob("*")):
        if path.is_file() and not _is_excluded(path.relative_to(root), profile.excluded_prefixes):
            yield path


def _safety_violations(root: Path, files: Iterable[Path]) -> list[dict[str, str]]:
    violations: list[dict[str, str]] = []
    for path in files:
        relative = path.relative_to(root)
        if any(marker in part.lower() for part in relative.parts for marker in SENSITIVE_NAME_MARKERS):
            violations.append({"path": relative.as_posix(), "reason": "sensitive_filename"})
            continue
        if path.suffix.lower() not in TEXT_SCAN_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        if path.suffix.lower() in {".json", ".yaml", ".yml", ".toml"} and SENSITIVE_CONFIG_KEY.search(text):
            violations.append({"path": relative.as_posix(), "reason": "sensitive_config_key"})
        if ABSOLUTE_LOCAL_PATH.search(text):
            violations.append({"path": relative.as_posix(), "reason": "absolute_local_path"})
    return violations


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
            if isinstance(child, ast.Raise) and isinstance(child.exc, ast.Call):
                if isinstance(child.exc.func, ast.Name) and child.exc.func.id == "NotImplementedError":
                    continue
            meaningful.append(child)
        if meaningful:
            return True
    return False


def _has_test(path: Path) -> bool:
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


def _package_metadata_violations(root: Path) -> list[dict[str, str]]:
    path = root / "pyproject.toml"
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError):
        return [{"path": "pyproject.toml", "reason": "package_metadata_unreadable"}]
    project = data.get("project", {})
    build_system = data.get("build-system", {})
    if not project.get("name") or not build_system.get("requires") or not build_system.get("build-backend"):
        return [{"path": "pyproject.toml", "reason": "package_metadata_missing"}]
    return []


def _readiness_violations(root: Path, profile_name: str) -> list[dict[str, str]]:
    violations: list[dict[str, str]] = []
    method_root = root / "src" / "cegwm" / "method"
    method_sources = [] if not method_root.exists() else [
        path for path in method_root.rglob("*.py")
        if path.name != "__init__.py" and _has_substantive_definition(path)
    ]
    method_test_root = root / "tests" / "unit"
    method_tests = [] if not method_test_root.exists() else [
        path for path in method_test_root.glob("test_*method*.py") if _has_test(path)
    ]
    if not method_sources:
        violations.append({"path": "src/cegwm/method", "reason": "method_implementation_missing"})
    if not method_tests:
        violations.append({"path": "tests/unit/test_*method*.py", "reason": "method_tests_missing"})

    runner_root = root / "experiments"
    runners = [
        path for path in runner_root.glob("run_content_*.py")
        if _has_substantive_definition(path)
    ]
    integration_root = root / "tests" / "integration"
    integration_tests = [
        path for path in integration_root.glob("test_content_*.py") if _has_test(path)
    ]
    if not runners:
        violations.append({"path": "experiments/run_content_*.py", "reason": "content_runner_missing"})
    if not integration_tests:
        violations.append({"path": "tests/integration/test_content_*.py", "reason": "content_integration_tests_missing"})
    return violations


def extract_profile(
    root: str | Path,
    output: str | Path,
    profile_name: str,
    *,
    dry_run: bool = False,
) -> dict[str, object]:
    root_path = Path(root).resolve()
    output_path = Path(output).resolve()
    if profile_name not in PROFILES:
        raise ValueError(f"unsupported extraction profile: {profile_name}")
    profile = PROFILES[profile_name]

    source_files: list[Path] = []
    copied_files: list[str] = []
    missing_paths: list[str] = []
    for include_path in profile.required_paths + profile.optional_paths:
        source = root_path / include_path
        if not source.exists():
            if include_path in profile.required_paths:
                missing_paths.append(include_path)
            continue
        for source_file in _iter_files(root_path, include_path, profile):
            source_files.append(source_file)
            copied_files.append(source_file.relative_to(root_path).as_posix())

    readme_template = root_path / profile.readme_template
    if readme_template.exists():
        source_files.append(readme_template)
        copied_files.append("README.md")
    else:
        missing_paths.append(profile.readme_template)

    safety_violations = _safety_violations(root_path, source_files)
    structural_violations = _package_metadata_violations(root_path)
    readiness_violations = _readiness_violations(root_path, profile_name)
    structurally_valid = not missing_paths and not safety_violations and not structural_violations
    release_candidate_ready = structurally_valid and not readiness_violations

    manifest: dict[str, object] = {
        "profile_name": profile_name,
        "copied_files": sorted(set(copied_files)),
        "missing_paths": missing_paths,
        "safety_violations": safety_violations,
        "structural_violations": structural_violations,
        "readiness_violations": readiness_violations,
        "excluded_prefixes": list(profile.excluded_prefixes),
        "structurally_valid": structurally_valid,
        "release_candidate_ready": release_candidate_ready,
        "dry_run": dry_run,
    }

    if dry_run:
        return manifest
    if not structurally_valid:
        raise ValueError("extraction profile is structurally invalid")
    if output_path.exists() and any(output_path.iterdir()):
        raise ValueError("output directory must be absent or empty")

    for source_file in source_files:
        relative = Path("README.md") if source_file == readme_template else source_file.relative_to(root_path)
        target = output_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, target)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "extraction_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract a minimal CEG-WM research package.")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="method_core")
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    manifest = extract_profile(args.root, args.output, args.profile, dry_run=args.dry_run)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
