"""审计正式文件和目录命名。"""

from __future__ import annotations

import ast
import io
import json
from pathlib import Path
import re
import sys
import tokenize
import tomllib
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from governance.harness.lib.file_scanner import iter_governed_paths
from governance.harness.lib.json_report import build_report, exit_with_report
from governance.harness.lib.naming_rules import (
    has_malformed_semantic_numeric_suffix,
    has_ordinal_identity_text,
    has_ordinal_identity_polysemy,
    has_weak_semantic_identity_value,
    has_weak_semantic_text,
    has_weak_semantic_token,
    is_allowed_directory_name,
    is_allowed_file_name,
)


CONFIG_KEY_PATTERN = re.compile(r"^\s*[\"']?(?P<key>[A-Za-z][A-Za-z0-9_-]*)[\"']?\s*[:=]")
IDENTITY_CONTEXT_PATTERN = re.compile(
    r"(?:^|[_-])(?:run_phase|schema|protocol|metric|case|gate|artifact|path|"
    r"identity|id|name|label|function|combination|phase|specification|digest|"
    r"record|result|controls?)(?:$|[_-])",
    re.IGNORECASE,
)


def _python_assignment_target_names(target: ast.expr) -> set[str]:
    """Collect formal identity context names from one assignment target."""
    names: set[str] = set()
    for child in ast.walk(target):
        if isinstance(child, ast.Name):
            names.add(child.id)
        elif isinstance(child, ast.Attribute):
            names.add(child.attr)
        elif (
            isinstance(child, ast.Subscript)
            and isinstance(child.slice, ast.Constant)
            and isinstance(child.slice.value, str)
        ):
            names.add(child.slice.value)
    return names


def _is_test_function_ordinal_exception(
    relative: Path,
    *,
    node_kind: str,
    name: str,
) -> bool:
    """Allow ordinal-looking pytest node names only below a tests directory."""
    return (
        "tests" in relative.parts[:-1]
        and node_kind in {"FunctionDef", "AsyncFunctionDef"}
        and name.startswith("test_")
    )


def _python_string_literals(value: ast.expr) -> list[tuple[str, int]]:
    """Collect string literals from a formal value and nested tuple/list values."""
    if isinstance(value, ast.Constant) and isinstance(value.value, str):
        return [(value.value, value.lineno)]
    if isinstance(value, (ast.Tuple, ast.List)):
        return [
            literal
            for element in value.elts
            for literal in _python_string_literals(element)
        ]
    return []


def _python_local_class_fields(tree: ast.AST) -> dict[str, tuple[str, ...]]:
    """Return unambiguous ordered fields declared by local classes.

    The binding is intentionally syntax-only: it neither imports modules nor
    executes decorators or constructors.  Duplicate local class names are
    omitted because their call target cannot be resolved unambiguously.
    """
    declarations: dict[str, list[tuple[str, ...]]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        fields: list[str] = []
        for statement in node.body:
            if (
                isinstance(statement, ast.AnnAssign)
                and isinstance(statement.target, ast.Name)
            ):
                fields.append(statement.target.id)
        if fields:
            declarations.setdefault(node.name, []).append(tuple(fields))
    return {
        name: definitions[0]
        for name, definitions in declarations.items()
        if len(definitions) == 1
    }


def _python_semantic_violations(path: Path, relative: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    violations = []
    try:
        tree = ast.parse(text)
    except SyntaxError as error:
        return [{"path": str(relative), "reason": "python_ast_unreadable", "line": error.lineno or 0}]

    names: list[tuple[str, int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.append((node.name, node.lineno, type(node).__name__))
        elif isinstance(node, ast.Name):
            names.append((node.id, node.lineno, "Name"))
        elif isinstance(node, ast.Attribute):
            names.append((node.attr, node.lineno, "Attribute"))
        elif isinstance(node, ast.arg):
            names.append((node.arg, node.lineno, "arg"))
        elif isinstance(node, ast.keyword) and node.arg:
            names.append((node.arg, node.lineno, "keyword"))
    for name, line, node_kind in sorted(
        set(names),
        key=lambda item: (item[1], item[0], item[2]),
    ):
        if has_weak_semantic_token(name):
            violations.append({"path": str(relative), "reason": "weak_semantic_identifier", "identifier": name, "line": line})
        if has_ordinal_identity_text(name) and not _is_test_function_ordinal_exception(
            relative,
            node_kind=node_kind,
            name=name,
        ):
            violations.append(
                {
                    "path": str(relative),
                    "reason": "ordinal_identity_identifier",
                    "identifier": name,
                    "line": line,
                }
            )

    identity_strings: list[tuple[str, int]] = []
    local_class_fields = _python_local_class_fields(tree)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names_in_target = {
                name
                for target in targets
                for name in _python_assignment_target_names(target)
            }
            value = node.value
            if any(IDENTITY_CONTEXT_PATTERN.search(name) for name in names_in_target):
                identity_strings.extend(_python_string_literals(value))
        elif isinstance(node, ast.keyword) and node.arg and IDENTITY_CONTEXT_PATTERN.search(node.arg):
            identity_strings.extend(_python_string_literals(node.value))
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in local_class_fields
        ):
            for field_name, argument in zip(
                local_class_fields[node.func.id],
                node.args,
                strict=False,
            ):
                if IDENTITY_CONTEXT_PATTERN.search(field_name):
                    identity_strings.extend(_python_string_literals(argument))
        elif isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values, strict=True):
                if (
                    isinstance(key, ast.Constant)
                    and isinstance(key.value, str)
                    and IDENTITY_CONTEXT_PATTERN.search(key.value)
                ):
                    identity_strings.extend(_python_string_literals(value))
        elif isinstance(node, ast.Compare):
            left_names = {
                child.id if isinstance(child, ast.Name) else child.attr
                for child in ast.walk(node.left)
                if isinstance(child, (ast.Name, ast.Attribute))
            }
            if any(IDENTITY_CONTEXT_PATTERN.search(name) for name in left_names):
                for comparator in node.comparators:
                    identity_strings.extend(_python_string_literals(comparator))
    for value, line in sorted(set(identity_strings), key=lambda item: (item[1], item[0])):
        if has_weak_semantic_identity_value(value):
            violations.append(
                {
                    "path": str(relative),
                    "reason": "weak_semantic_python_string",
                    "line": line,
                }
            )
        if has_ordinal_identity_text(value):
            violations.append(
                {
                    "path": str(relative),
                    "reason": "ordinal_identity_python_string",
                    "line": line,
                }
            )

    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            docstring = ast.get_docstring(node, clean=False)
            if docstring and has_weak_semantic_text(docstring):
                violations.append({"path": str(relative), "reason": "weak_semantic_docstring", "line": getattr(node, "lineno", 1)})
            if docstring and has_ordinal_identity_text(docstring):
                violations.append(
                    {
                        "path": str(relative),
                        "reason": "ordinal_identity_docstring",
                        "line": getattr(node, "lineno", 1),
                    }
                )

    try:
        tokens = tokenize.generate_tokens(io.StringIO(text).readline)
        for token in tokens:
            if token.type == tokenize.COMMENT and has_weak_semantic_text(token.string):
                violations.append({"path": str(relative), "reason": "weak_semantic_comment", "line": token.start[0]})
            if token.type == tokenize.COMMENT and has_ordinal_identity_text(token.string):
                violations.append(
                    {
                        "path": str(relative),
                        "reason": "ordinal_identity_comment",
                        "line": token.start[0],
                    }
                )
    except tokenize.TokenError as error:
        violations.append({"path": str(relative), "reason": "python_tokens_unreadable", "detail": str(error)})
    for line_number, line in enumerate(text.splitlines(), start=1):
        if has_malformed_semantic_numeric_suffix(line):
            violations.append(
                {
                    "path": str(relative),
                    "reason": "malformed_semantic_numeric_suffix",
                    "line": line_number,
                }
            )
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


def _identity_and_path_values(
    value: object,
    inherited_context: bool = False,
) -> list[str]:
    values: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            child_context = inherited_context or bool(
                IDENTITY_CONTEXT_PATTERN.search(key_text)
                or key_text.endswith(("_paths", "paths"))
            )
            if isinstance(child, str) and child_context:
                values.append(child)
            values.extend(_identity_and_path_values(child, child_context))
    elif isinstance(value, list):
        for child in value:
            if isinstance(child, str) and inherited_context:
                values.append(child)
            values.extend(_identity_and_path_values(child, inherited_context))
    return values


def _ordinal_bindings(
    value: object,
    path: tuple[str, ...] = (),
    inherited_context: bool = False,
) -> list[tuple[str, str]]:
    bindings: list[tuple[str, str]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            context = inherited_context or bool(
                IDENTITY_CONTEXT_PATTERN.search(key_text)
                or key_text.endswith(("_paths", "paths"))
            )
            semantic_context = ".".join((*path, key_text))
            if has_ordinal_identity_text(key_text):
                bindings.append((key_text, semantic_context))
            if isinstance(child, str) and context and has_ordinal_identity_text(child):
                bindings.append((child, semantic_context))
            bindings.extend(
                _ordinal_bindings(child, (*path, key_text), context)
            )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            semantic_context = ".".join((*path, str(index)))
            if isinstance(child, str) and inherited_context and has_ordinal_identity_text(child):
                bindings.append((child, semantic_context))
            bindings.extend(
                _ordinal_bindings(child, (*path, str(index)), inherited_context)
            )
    return bindings


def _config_semantic_violations(path: Path, relative: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    keyed_lines: list[tuple[str, int]] = []
    document: object | None = None
    try:
        if path.suffix.lower() == ".toml":
            document = tomllib.loads(text)
        elif path.suffix.lower() in {".yaml", ".yml"}:
            from governance.harness.lib.project_policy import load_json_compatible_yaml

            document = load_json_compatible_yaml(path)
        else:
            document = json.loads(text)
    except (json.JSONDecodeError, ValueError, tomllib.TOMLDecodeError):
        for line_number, line in enumerate(text.splitlines(), start=1):
            match = CONFIG_KEY_PATTERN.match(line)
            if match:
                keyed_lines.append((match.group("key"), line_number))
    else:
        keyed_lines.extend((key, 1) for key in _nested_mapping_keys(document))
    violations = [
        {"path": str(relative), "reason": "weak_semantic_config_key", "key": key, "line": line}
        for key, line in keyed_lines
        if has_weak_semantic_token(key)
    ]
    for key, line in keyed_lines:
        if has_ordinal_identity_text(key):
            violations.append(
                {
                    "path": str(relative),
                    "reason": "ordinal_identity_config_key",
                    "key": key,
                    "line": line,
                }
            )
    if document is not None:
        for value in _identity_and_path_values(document):
            if has_weak_semantic_identity_value(value):
                violations.append(
                    {
                        "path": str(relative),
                        "reason": "weak_semantic_config_value",
                        "value": value,
                        "line": 1,
                    }
                )
            if has_ordinal_identity_text(value):
                violations.append(
                    {
                        "path": str(relative),
                        "reason": "ordinal_identity_config_value",
                        "value": value,
                        "line": 1,
                    }
                )
        if has_ordinal_identity_polysemy(_ordinal_bindings(document)):
            violations.append(
                {
                    "path": str(relative),
                    "reason": "ordinal_identity_polysemy",
                    "line": 1,
                }
            )
    for line_number, line in enumerate(text.splitlines(), start=1):
        if has_malformed_semantic_numeric_suffix(line):
            violations.append(
                {
                    "path": str(relative),
                    "reason": "malformed_semantic_numeric_suffix",
                    "line": line_number,
                }
            )
    return violations


def _text_semantic_violations(path: Path, relative: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".ipynb":
        try:
            notebook = json.loads(text)
            violations = []
            for cell in notebook.get("cells", ()):
                source = "".join(cell.get("source", ()))
                if has_weak_semantic_text(source):
                    violations.append(
                        {
                            "path": str(relative),
                            "reason": (
                                "weak_semantic_notebook_code"
                                if cell.get("cell_type") == "code"
                                else "weak_semantic_notebook_markdown"
                            ),
                            "line": 1,
                        }
                    )
                if has_ordinal_identity_text(source):
                    violations.append(
                        {
                            "path": str(relative),
                            "reason": (
                                "ordinal_identity_notebook_code"
                                if cell.get("cell_type") == "code"
                                else "ordinal_identity_notebook_markdown"
                            ),
                            "line": 1,
                        }
                    )
                if has_malformed_semantic_numeric_suffix(source):
                    violations.append(
                        {
                            "path": str(relative),
                            "reason": "malformed_semantic_numeric_suffix",
                            "line": 1,
                        }
                    )
            return violations
        except (json.JSONDecodeError, TypeError):
            return [
                {
                    "path": str(relative),
                    "reason": "notebook_unreadable",
                    "line": 0,
                }
            ]
    violations = []
    weak_identity_found = False
    if path.suffix.lower() == ".md":
        for line in text.splitlines():
            candidate = line.strip().strip("`#>*- ")
            if has_weak_semantic_identity_value(candidate):
                weak_identity_found = True
                break
    elif path.suffix.lower() in {".svg", ".drawio"}:
        try:
            root = ET.fromstring(text)
        except ET.ParseError:
            root = None
        if root is not None:
            values = []
            for element in root.iter():
                if element.text and element.text.strip():
                    values.append(element.text.strip())
                values.extend(
                    value.strip()
                    for value in element.attrib.values()
                    if value.strip()
                )
            weak_identity_found = any(
                has_weak_semantic_identity_value(value) for value in values
            )
    if weak_identity_found:
        violations.append(
            {
                "path": str(relative),
                "reason": "weak_semantic_markdown",
                "line": 1,
            }
        )
    if has_ordinal_identity_text(text):
        violations.append(
            {
                "path": str(relative),
                "reason": "ordinal_identity_markdown",
                "line": 1,
            }
        )
    for line_number, line in enumerate(text.splitlines(), start=1):
        if has_malformed_semantic_numeric_suffix(line):
            violations.append(
                {
                    "path": str(relative),
                    "reason": "malformed_semantic_numeric_suffix",
                    "line": line_number,
                }
            )
    return violations


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
        if has_ordinal_identity_text(path.stem if path.is_file() else path.name):
            violations.append(
                {
                    "path": str(relative),
                    "reason": "ordinal_identity_path_component",
                }
            )
        if path.is_file() and path.suffix == ".py":
            violations.extend(_python_semantic_violations(path, relative))
        if path.is_file() and path.suffix.lower() in {".json", ".yaml", ".yml", ".toml"}:
            violations.extend(_config_semantic_violations(path, relative))
        if path.is_file() and path.suffix.lower() in {".md", ".ipynb", ".svg", ".drawio"}:
            violations.extend(_text_semantic_violations(path, relative))
    return build_report("audit_naming_conventions", "fail" if violations else "pass", violations, checked_paths)


def main() -> None:
    exit_with_report(run_audit(Path.cwd()))


if __name__ == "__main__":
    main()
