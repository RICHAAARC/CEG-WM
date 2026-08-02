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
import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from governance.harness.lib.file_scanner import iter_governed_paths
from governance.harness.lib.field_rules import inspect_field_registry
from governance.harness.lib.json_report import build_report, exit_with_report
from governance.harness.lib.naming_rules import (
    LOCAL_MATH_BINDING_NAMES,
    NUMBERED_RESPONSIBILITY_WORDS,
    canonical_version_context,
    has_generic_mechanical_numeric_suffix,
    has_mechanical_identity_token_in_text,
    has_malformed_semantic_numeric_suffix,
    has_ordinal_identity_text,
    has_ordinal_identity_polysemy,
    has_weak_semantic_identity_value,
    has_weak_semantic_path_name,
    has_weak_semantic_text,
    has_weak_semantic_token,
    is_allowed_directory_name,
    is_allowed_file_name,
    is_allowed_registered_numeric_field_role,
    is_scientific_l2_identifier,
    is_explicit_version_context,
    is_noncanonical_version_context,
)


CONFIG_KEY_PATTERN = re.compile(r"^\s*[\"']?(?P<key>[A-Za-z][A-Za-z0-9_-]*)[\"']?\s*[:=]")
IDENTITY_CONTEXT_PATTERN = re.compile(
    r"(?:^|[_-])(?:run_phase|schema|protocol|metric|case|gate|artifact|path|"
    r"identity|id|name|label|function|combination|phase|specification|digest|"
    r"controls?)(?:$|[_-])",
    re.IGNORECASE,
)
BUSINESS_PATH_ROOTS = frozenset(
    {
        "main",
        "runtime",
        "experiments",
        "configs",
        "notebooks",
        "paper_artifacts",
        "scripts",
    }
)
RESPONSIBILITY_CONTEXT_PATTERN = re.compile(
    rf"(?:^|[_-])(?:{'|'.join(sorted(NUMBERED_RESPONSIBILITY_WORDS))})(?:$|[_-])",
    re.IGNORECASE,
)


def _is_business_production_path(relative: Path) -> bool:
    return bool(relative.parts and relative.parts[0] in BUSINESS_PATH_ROOTS)


TEST_BEHAVIOR_SEMANTIC_PATTERN = re.compile(
    r"(?:^|_)(?:rejects?|preserves?|fails?|failure|raises?|returns?|accepts?|"
    r"prevents?|detects?|reports?|produces?|releases?|explicit)(?:_|$)",
    re.IGNORECASE,
)


def _is_weak_test_node_without_behavior(name: str) -> bool:
    """Reject weak test identities that state no observable behavioral outcome."""
    if not name.startswith("test_") or not has_weak_semantic_identity_value(name):
        return False
    tokens = tuple(part for part in name.split("_") if part)
    return len(tokens) < 5 or not TEST_BEHAVIOR_SEMANTIC_PATTERN.search(name)


def _is_formal_identity_context(
    name: str,
    registered_identity_fields: frozenset[str],
    registered_fields: frozenset[str],
) -> bool:
    """Prefer registered identity fields, with explicit identity tokens as fallback."""
    if is_explicit_version_context(name) or is_noncanonical_version_context(name):
        return True
    if name in registered_fields:
        return name in registered_identity_fields
    return (
        bool(IDENTITY_CONTEXT_PATTERN.search(name))
        or bool(RESPONSIBILITY_CONTEXT_PATTERN.search(name))
        or name.endswith(("_paths", "paths"))
    )


def _is_python_identity_name(
    name: str,
    node_kind: str,
    relative: Path,
    registered_identity_fields: frozenset[str],
    registered_fields: frozenset[str],
) -> bool:
    """Reject ordinal identifiers while preserving local coordinate variables."""
    if _is_formal_identity_context(
        name,
        registered_identity_fields,
        registered_fields,
    ):
        return True
    # Non-formal test-fixture variables may describe fixture dimensions or
    # synthetic objects. This is not a test-node exemption: functions/classes
    # and every registered/fallback formal binding remain governed above.
    if (
        "tests" in relative.parts[:-1]
        and node_kind not in {"FunctionDef", "AsyncFunctionDef", "ClassDef"}
    ):
        return False
    # These lowercase names are local interpolation coordinates, not public or
    # persisted identities. Uppercase X1/Y2 work labels remain forbidden.
    if (
        node_kind not in {"FunctionDef", "AsyncFunctionDef", "ClassDef"}
        and re.fullmatch(r"[xy][01](?:_unclamped)?", name)
    ):
        return False
    # L2 is allowed only as an explicitly named scientific norm token outside
    # a registered/fallback formal-identity binding.
    return not is_scientific_l2_identifier(name)


def _is_display_identity(text: str) -> bool:
    """Return whether one visible label is shaped like an ordinal identity."""
    candidate = text.strip().strip("`#>*- ；。:：")
    if not candidate or not has_ordinal_identity_text(candidate):
        return False
    if "\n" in candidate:
        return False
    if "`" in candidate:
        return False
    if re.search(
        rf"(?:^|\s)(?:{'|'.join(sorted(NUMBERED_RESPONSIBILITY_WORDS))})"
        r"[-_ ]*\d+(?:$|\s)",
        candidate,
        re.IGNORECASE,
    ):
        return len(candidate.split()) <= 4
    return bool(re.fullmatch(r"[A-Za-z0-9_-]+", candidate)) and (
        candidate.lower() != "l2"
    )


def _is_display_weak_identity(text: str) -> bool:
    """Limit Markdown weak-word checks to one displayed identity, not prose."""
    candidate = text.strip().strip("`#>*- ；。:：")
    return bool(
        candidate
        and "`" not in candidate
        and re.fullmatch(r"[A-Za-z0-9_-]+", candidate)
        and has_weak_semantic_identity_value(candidate)
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


def _python_local_math_name_node_ids(tree: ast.AST) -> frozenset[int]:
    """Prove narrow local math bindings without executing the inspected module."""
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent

    lexical_scopes = (
        ast.Module,
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        ast.Lambda,
        ast.ClassDef,
    )

    def lexical_scope(node: ast.AST) -> ast.AST | None:
        current = parents.get(node)
        while current is not None and not isinstance(current, lexical_scopes):
            current = parents.get(current)
        return current

    binding_completion_positions: dict[tuple[int, str], list[tuple[int, int]]] = {}
    allowed_node_ids: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        if isinstance(node, ast.AnnAssign) and node.value is None:
            # A bare annotation declares a name but does not bind a value.
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if not isinstance(target, ast.Name) or target.id not in LOCAL_MATH_BINDING_NAMES:
                continue
            scope = lexical_scope(target)
            # A class-body assignment creates an attribute-like public identity,
            # not a local mathematical variable.
            if scope is None or isinstance(scope, ast.ClassDef):
                continue
            key = (id(scope), target.id)
            binding_completion_positions.setdefault(key, []).append(
                (
                    getattr(node, "end_lineno", node.lineno),
                    getattr(node, "end_col_offset", target.col_offset),
                )
            )
            allowed_node_ids.add(id(target))

    for node in ast.walk(tree):
        if (
            not isinstance(node, ast.Name)
            or not isinstance(node.ctx, ast.Load)
            or node.id not in LOCAL_MATH_BINDING_NAMES
        ):
            continue
        scope = lexical_scope(node)
        if scope is None or isinstance(scope, ast.ClassDef):
            continue
        binding_positions = binding_completion_positions.get((id(scope), node.id), ())
        # A read is local only after an earlier source position completed the
        # exact binding in this same lexical scope.  Using the assignment end
        # also prevents its own RHS from being mistaken for a subsequent read.
        if any(
            binding_position < (node.lineno, node.col_offset)
            for binding_position in binding_positions
        ):
            allowed_node_ids.add(id(node))
    return frozenset(allowed_node_ids)


def _python_semantic_violations(
    path: Path,
    relative: Path,
    registered_identity_fields: frozenset[str],
    registered_fields: frozenset[str],
) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    violations = []
    try:
        tree = ast.parse(text)
    except SyntaxError as error:
        return [{"path": str(relative), "reason": "python_ast_unreadable", "line": error.lineno or 0}]

    local_math_name_node_ids = _python_local_math_name_node_ids(tree)
    names: list[tuple[str, int, str, bool]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.append((node.name, node.lineno, type(node).__name__, False))
        elif isinstance(node, ast.Name):
            names.append(
                (node.id, node.lineno, "Name", id(node) in local_math_name_node_ids)
            )
        elif isinstance(node, ast.Attribute):
            names.append((node.attr, node.lineno, "Attribute", False))
        elif isinstance(node, ast.arg):
            names.append((node.arg, node.lineno, "arg", False))
        elif isinstance(node, ast.keyword) and node.arg:
            names.append((node.arg, node.lineno, "keyword", False))
    for name, line, node_kind, is_local_math_name in sorted(
        set(names),
        key=lambda item: (item[1], item[0], item[2]),
    ):
        if is_local_math_name:
            continue
        is_callable_or_class = node_kind in {"FunctionDef", "AsyncFunctionDef", "ClassDef"}
        if (
            has_weak_semantic_token(name)
            or (
                is_callable_or_class
                and (
                    (_is_business_production_path(relative) and has_weak_semantic_identity_value(name))
                    or _is_weak_test_node_without_behavior(name)
                )
            )
            or (
                has_generic_mechanical_numeric_suffix(name)
                and not is_allowed_registered_numeric_field_role(name)
                and (
                    _is_business_production_path(relative)
                    or is_callable_or_class
                    or _is_formal_identity_context(
                        name,
                        registered_identity_fields,
                        registered_fields,
                    )
                )
            )
        ):
            violations.append({"path": str(relative), "reason": "weak_semantic_identifier", "identifier": name, "line": line})
        if has_ordinal_identity_text(name) and _is_python_identity_name(
            name,
            node_kind,
            relative,
            registered_identity_fields,
            registered_fields,
        ):
            violations.append(
                {
                    "path": str(relative),
                    "reason": "ordinal_identity_identifier",
                    "identifier": name,
                    "line": line,
                }
            )

    identity_strings: list[tuple[str, int, str]] = []
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
            formal_names = {
                name
                for name in names_in_target
                if _is_formal_identity_context(
                    name,
                    registered_identity_fields,
                    registered_fields,
                )
            }
            for formal_name in formal_names:
                identity_strings.extend(
                    (literal, line, formal_name)
                    for literal, line in _python_string_literals(value)
                )
        elif (
            isinstance(node, ast.keyword)
            and node.arg
            and _is_formal_identity_context(
                node.arg,
                registered_identity_fields,
                registered_fields,
            )
        ):
            identity_strings.extend(
                (literal, line, node.arg)
                for literal, line in _python_string_literals(node.value)
            )
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
                if _is_formal_identity_context(
                    field_name,
                    registered_identity_fields,
                    registered_fields,
                ):
                    identity_strings.extend(
                        (literal, line, field_name)
                        for literal, line in _python_string_literals(argument)
                    )
        elif isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values, strict=True):
                if (
                    isinstance(key, ast.Constant)
                    and isinstance(key.value, str)
                    and _is_formal_identity_context(
                        key.value,
                        registered_identity_fields,
                        registered_fields,
                    )
                ):
                    identity_strings.extend(
                        (literal, line, key.value)
                        for literal, line in _python_string_literals(value)
                    )
        elif isinstance(node, ast.Compare):
            left_names = {
                child.id if isinstance(child, ast.Name) else child.attr
                for child in ast.walk(node.left)
                if isinstance(child, (ast.Name, ast.Attribute))
            }
            formal_names = {
                name
                for name in left_names
                if _is_formal_identity_context(
                    name,
                    registered_identity_fields,
                    registered_fields,
                )
            }
            if formal_names:
                for comparator in node.comparators:
                    for formal_name in formal_names:
                        identity_strings.extend(
                            (literal, line, formal_name)
                            for literal, line in _python_string_literals(comparator)
                        )
    for value, line, context in sorted(
        set(identity_strings),
        key=lambda item: (item[1], item[0], item[2]),
    ):
        if is_noncanonical_version_context(context):
            violations.append(
                {
                    "path": str(relative),
                    "reason": "version_context_identifier_not_canonical",
                    "identifier": context,
                    "canonical_context": canonical_version_context(context),
                    "line": line,
                }
            )
        if has_weak_semantic_identity_value(value, version_context=context):
            violations.append(
                {
                    "path": str(relative),
                    "reason": "weak_semantic_python_string",
                    "context": context,
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
    if has_ordinal_identity_polysemy(
        [(value, context) for value, _, context in identity_strings]
    ):
        violations.append(
            {
                "path": str(relative),
                "reason": "ordinal_identity_polysemy",
                "line": 1,
            }
        )

    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            docstring = ast.get_docstring(node, clean=False)
            if docstring and (
                has_weak_semantic_text(docstring)
                or (
                    _is_business_production_path(relative)
                    and has_mechanical_identity_token_in_text(docstring)
                )
            ):
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
            if token.type == tokenize.COMMENT and (
                has_weak_semantic_text(token.string)
                or (
                    _is_business_production_path(relative)
                    and has_mechanical_identity_token_in_text(token.string)
                )
            ):
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
    registered_identity_fields: frozenset[str],
    registered_fields: frozenset[str],
    inherited_context: str | None = None,
) -> list[tuple[str, str]]:
    values: list[tuple[str, str]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            child_context = bool(
                _is_formal_identity_context(
                    key_text,
                    registered_identity_fields,
                    registered_fields,
                )
            )
            if isinstance(child, str) and child_context:
                values.append((child, key_text))
            values.extend(
                _identity_and_path_values(
                    child,
                    registered_identity_fields,
                    registered_fields,
                    key_text if child_context and isinstance(child, list) else None,
                )
            )
    elif isinstance(value, list):
        for child in value:
            if isinstance(child, str) and inherited_context is not None:
                values.append((child, inherited_context))
            values.extend(
                _identity_and_path_values(
                child,
                registered_identity_fields,
                registered_fields,
                    inherited_context,
                )
            )
    return values


def _is_narrow_historical_source_literal(value: str, context: str) -> bool:
    """Preserve only the already registered external historical source spelling."""
    if context not in {"source_id", "project_name", "read_only_path"}:
        return False
    return value in {
        "ceg_wm_old_main",
        "CEG-WM-OLD-main",
        "/home/richar/projects/CEG-WM-OLD-main/CEG-WM-OLD-main",
    }


def _ordinal_bindings(
    value: object,
    registered_identity_fields: frozenset[str],
    registered_fields: frozenset[str],
    path: tuple[str, ...] = (),
    inherited_context: bool = False,
) -> list[tuple[str, str]]:
    bindings: list[tuple[str, str]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            context = bool(
                _is_formal_identity_context(
                    key_text,
                    registered_identity_fields,
                    registered_fields,
                )
            )
            semantic_context = ".".join((*path, key_text))
            if has_ordinal_identity_text(key_text):
                bindings.append((key_text, semantic_context))
            if isinstance(child, str) and context and has_ordinal_identity_text(child):
                bindings.append((child, semantic_context))
            bindings.extend(
                _ordinal_bindings(
                    child,
                    registered_identity_fields,
                    registered_fields,
                    (*path, key_text),
                    context if isinstance(child, list) else False,
                )
            )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            semantic_context = ".".join((*path, str(index)))
            if isinstance(child, str) and inherited_context and has_ordinal_identity_text(child):
                bindings.append((child, semantic_context))
            bindings.extend(
                _ordinal_bindings(
                child,
                registered_identity_fields,
                registered_fields,
                    (*path, str(index)),
                    inherited_context,
                )
            )
    return bindings


def _config_semantic_violations(
    path: Path,
    relative: Path,
    registered_identity_fields: frozenset[str],
    registered_fields: frozenset[str],
) -> list[dict]:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return [{"path": str(relative), "reason": "config_unreadable"}]
    keyed_lines: list[tuple[str, int]] = []
    document: object | None = None
    try:
        if path.suffix.lower() == ".toml":
            document = tomllib.loads(text)
        elif path.suffix.lower() in {".yaml", ".yml"}:
            document = yaml.safe_load(text)
        else:
            document = json.loads(text)
    except (json.JSONDecodeError, ValueError, tomllib.TOMLDecodeError, yaml.YAMLError):
        violations = [
            {
                "path": str(relative),
                "reason": "config_unreadable",
            }
        ]
        for line_number, line in enumerate(text.splitlines(), start=1):
            match = CONFIG_KEY_PATTERN.match(line)
            if match:
                keyed_lines.append((match.group("key"), line_number))
    else:
        keyed_lines.extend((key, 1) for key in _nested_mapping_keys(document))
        violations = []
    violations.extend(
        [
            {"path": str(relative), "reason": "weak_semantic_config_key", "key": key, "line": line}
            for key, line in keyed_lines
            if has_weak_semantic_token(key)
            or (
                _is_formal_identity_context(
                    key,
                    registered_identity_fields,
                    registered_fields,
                )
                and has_weak_semantic_identity_value(key)
            )
        ]
    )
    for key, line in keyed_lines:
        if is_noncanonical_version_context(key, surface="config"):
            violations.append(
                {
                    "path": str(relative),
                    "reason": "version_context_key_not_canonical",
                    "key": key,
                    "canonical_context": canonical_version_context(key),
                    "line": line,
                }
            )
        if (
            _is_formal_identity_context(
                key,
                registered_identity_fields,
                registered_fields,
            )
            and has_ordinal_identity_text(key)
        ):
            violations.append(
                {
                    "path": str(relative),
                    "reason": "ordinal_identity_config_key",
                    "key": key,
                    "line": line,
                }
            )
    if document is not None:
        for value, context in _identity_and_path_values(
            document,
            registered_identity_fields,
            registered_fields,
        ):
            if has_weak_semantic_identity_value(
                value,
                version_context=context,
            ) and not _is_narrow_historical_source_literal(
                value, context
            ):
                violations.append(
                    {
                        "path": str(relative),
                        "reason": "weak_semantic_config_value",
                        "value": value,
                        "context": context,
                        "line": 1,
                    }
                )
            if has_ordinal_identity_text(value) and not is_scientific_l2_identifier(value):
                violations.append(
                    {
                        "path": str(relative),
                        "reason": "ordinal_identity_config_value",
                        "value": value,
                        "context": context,
                        "line": 1,
                    }
                )
        if has_ordinal_identity_polysemy(
            _ordinal_bindings(
                document,
                registered_identity_fields,
                registered_fields,
            )
        ):
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
                if (
                    has_ordinal_identity_text(source)
                    if cell.get("cell_type") == "code"
                    else _is_display_identity(source)
                ):
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
        weak_identity_found = any(
            _is_display_weak_identity(line) for line in text.splitlines()
        )
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
                for attribute in ("value", "label"):
                    value = element.attrib.get(attribute)
                    if value and value.strip():
                        values.append(value.strip())
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
    if path.suffix.lower() == ".md":
        ordinal_identity_found = any(
            _is_display_identity(line) for line in text.splitlines()
        )
    elif path.suffix.lower() in {".svg", ".drawio"}:
        ordinal_identity_found = root is not None and any(
            _is_display_identity(value) for value in values
        )
    else:
        ordinal_identity_found = has_ordinal_identity_text(text)
    if ordinal_identity_found:
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
    registry_inspection = inspect_field_registry(root_path)
    field_registry = registry_inspection.rows
    registered_fields = frozenset(field_registry)
    registered_identity_fields = frozenset(
        row.field_name
        for row in field_registry.values()
        if row.category in {"method_identity", "runtime_identity"}
    )
    violations = list(registry_inspection.violations)
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
        path_name = path.stem if path.is_file() else path.name
        if has_weak_semantic_token(path_name) or (
            relative.parts
            and relative.parts[0] in BUSINESS_PATH_ROOTS
            and has_weak_semantic_path_name(path_name)
        ):
            violations.append({"path": str(relative), "reason": "weak_semantic_token"})
        if has_ordinal_identity_text(path_name):
            violations.append(
                {
                    "path": str(relative),
                    "reason": "ordinal_identity_path_component",
                }
            )
        if path.is_file() and path.suffix == ".py":
            violations.extend(
                _python_semantic_violations(
                    path,
                    relative,
                    registered_identity_fields,
                    registered_fields,
                )
            )
        if path.is_file() and path.suffix.lower() in {".json", ".yaml", ".yml", ".toml"}:
            violations.extend(
                _config_semantic_violations(
                    path,
                    relative,
                    registered_identity_fields,
                    registered_fields,
                )
            )
        if path.is_file() and path.suffix.lower() in {".md", ".ipynb", ".svg", ".drawio"}:
            violations.extend(_text_semantic_violations(path, relative))
    return build_report("audit_naming_conventions", "fail" if violations else "pass", violations, checked_paths)


def main() -> None:
    exit_with_report(run_audit(Path.cwd()))


if __name__ == "__main__":
    main()
