"""审计正式文件和目录命名。"""

from __future__ import annotations

import ast
import io
import json
from pathlib import Path
import re
import sys
import tokenize

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from governance.harness.lib.file_scanner import iter_governed_paths
from governance.harness.lib.json_report import build_report, exit_with_report
from governance.harness.lib.naming_rules import (
    has_weak_semantic_text,
    has_weak_semantic_token,
    is_allowed_directory_name,
    is_allowed_file_name,
)


CONFIG_KEY_PATTERN = re.compile(r"^\s*[\"']?(?P<key>[A-Za-z][A-Za-z0-9_-]*)[\"']?\s*[:=]")


def _python_semantic_violations(path: Path, relative: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    violations = []
    try:
        tree = ast.parse(text)
    except SyntaxError as error:
        return [{"path": str(relative), "reason": "python_ast_unreadable", "line": error.lineno or 0}]

    names: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.append((node.name, node.lineno))
        elif isinstance(node, ast.Name):
            names.append((node.id, node.lineno))
        elif isinstance(node, ast.Attribute):
            names.append((node.attr, node.lineno))
        elif isinstance(node, ast.arg):
            names.append((node.arg, node.lineno))
        elif isinstance(node, ast.keyword) and node.arg:
            names.append((node.arg, node.lineno))
    for name, line in sorted(set(names), key=lambda item: (item[1], item[0])):
        if has_weak_semantic_token(name):
            violations.append({"path": str(relative), "reason": "weak_semantic_identifier", "identifier": name, "line": line})

    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            docstring = ast.get_docstring(node, clean=False)
            if docstring and has_weak_semantic_text(docstring):
                violations.append({"path": str(relative), "reason": "weak_semantic_docstring", "line": getattr(node, "lineno", 1)})

    try:
        tokens = tokenize.generate_tokens(io.StringIO(text).readline)
        for token in tokens:
            if token.type == tokenize.COMMENT and has_weak_semantic_text(token.string):
                violations.append({"path": str(relative), "reason": "weak_semantic_comment", "line": token.start[0]})
    except tokenize.TokenError as error:
        violations.append({"path": str(relative), "reason": "python_tokens_unreadable", "detail": str(error)})
    return violations


def _nested_mapping_keys(value: object) -> list[str]:
    keys: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            keys.append(str(key))
            keys.extend(_nested_mapping_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.extend(_nested_mapping_keys(child))
    return keys


def _config_semantic_violations(path: Path, relative: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    keyed_lines: list[tuple[str, int]] = []
    try:
        document = json.loads(text)
    except json.JSONDecodeError:
        for line_number, line in enumerate(text.splitlines(), start=1):
            match = CONFIG_KEY_PATTERN.match(line)
            if match:
                keyed_lines.append((match.group("key"), line_number))
    else:
        keyed_lines.extend((key, 1) for key in _nested_mapping_keys(document))
    return [
        {"path": str(relative), "reason": "weak_semantic_config_key", "key": key, "line": line}
        for key, line in keyed_lines
        if has_weak_semantic_token(key)
    ]


def run_audit(root: str | Path) -> dict:
    root_path = Path(root)
    violations = []
    checked_paths = []
    for path in iter_governed_paths(root_path):
        relative = path.relative_to(root_path)
        checked_paths.append(str(relative))
        if path.is_dir():
            if not is_allowed_directory_name(path.name):
                violations.append({"path": str(relative), "reason": "directory_name_not_snake_case"})
        elif path.is_file():
            if not is_allowed_file_name(path.name):
                violations.append({"path": str(relative), "reason": "file_name_not_snake_case"})
        if has_weak_semantic_token(path.stem if path.is_file() else path.name):
            violations.append({"path": str(relative), "reason": "weak_semantic_token"})
        if path.is_file() and path.suffix == ".py":
            violations.extend(_python_semantic_violations(path, relative))
        if path.is_file() and path.suffix.lower() in {".json", ".yaml", ".yml", ".toml"}:
            violations.extend(_config_semantic_violations(path, relative))
    return build_report("audit_naming_conventions", "fail" if violations else "pass", violations, checked_paths)


def main() -> None:
    exit_with_report(run_audit(Path.cwd()))


if __name__ == "__main__":
    main()
