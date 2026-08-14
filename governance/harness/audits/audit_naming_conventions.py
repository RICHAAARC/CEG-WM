"""审计正式文件和目录命名。"""

from __future__ import annotations

import ast
from hashlib import sha256
import io
import json
from pathlib import Path, PurePosixPath
import re
import stat
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

_UPSTREAM_SOURCE_ROOT = Path("runtime/_vendor/transparent_background")
_UPSTREAM_SOURCE_MANIFEST = _UPSTREAM_SOURCE_ROOT / "SOURCE.json"
_UPSTREAM_SOURCE_REPOSITORY = (
    "https://github.com/plemeri/transparent-background"
)
_UPSTREAM_SOURCE_REVISION = "f0fa91701a98cfc8e955c554e84522f365ec6da3"
_UPSTREAM_SOURCE_TREE = "19c4aae7fe5ca6d77ddbd8cc4a4e0be662bfcb5c"
_UPSTREAM_SOURCE_NAMESPACE = "runtime._vendor.transparent_background"
_UPSTREAM_SOURCE_FILE_ATTESTATIONS = {
    "LICENSE": (
        "LICENSE",
        "a08a7c43ff8fe90648f889d4f937b178c29ab9be1f92244f685bf7f97cb53f91",
    ),
    "__init__.py": (
        None,
        "01ba4719c80b6fe911b091a7c05124b64eeece964e09c058ef8f9805daca546b",
    ),
    "InSPyReNet.py": (
        "transparent_background/InSPyReNet.py",
        "e2f7d66c37b778ab1fce10553604075a54d93691b4612b952d7d44a8388cf42b",
    ),
    "modules/__init__.py": (
        None,
        "01ba4719c80b6fe911b091a7c05124b64eeece964e09c058ef8f9805daca546b",
    ),
    "modules/layers.py": (
        "transparent_background/modules/layers.py",
        "7f5c6ad133af2234b74ff6d067e95f09022f598abc0d987b8a2d99a1044d66d7",
    ),
    "modules/context_module.py": (
        "transparent_background/modules/context_module.py",
        "b5b612e4d86848a3e69b66d89effcc8698e434d6f50270595605c1d42cb844d4",
    ),
    "modules/attention_module.py": (
        "transparent_background/modules/attention_module.py",
        "30e05975d0e8a9ff9f3dddaf0fa278556d16d9f40b4df8f76d193f4de8c8dcae",
    ),
    "modules/decoder_module.py": (
        "transparent_background/modules/decoder_module.py",
        "1a0b8d23cace8f68ceee14f76802af8d8762ce4dff9327a97538d26b7e7f936d",
    ),
    "backbones/__init__.py": (
        None,
        "01ba4719c80b6fe911b091a7c05124b64eeece964e09c058ef8f9805daca546b",
    ),
    "backbones/SwinTransformer.py": (
        "transparent_background/backbones/SwinTransformer.py",
        "6f76d560fec382c8526a7230f4bbd95d122b97bdea44de452586a79f8a5ac41d",
    ),
}
_UPSTREAM_SOURCE_ORIGINAL_SHA256 = {
    "LICENSE": "a08a7c43ff8fe90648f889d4f937b178c29ab9be1f92244f685bf7f97cb53f91",
    "__init__.py": None,
    "InSPyReNet.py": "9bf8c73a361200888e48677c1df55b81bb1bdb669cfd91d73a01c01d24efbef4",
    "modules/__init__.py": None,
    "modules/layers.py": "e57eedd05bece9f14cf6b2798e0c2ed09382e60d200ed6352895a979f80ed5e8",
    "modules/context_module.py": "b5b612e4d86848a3e69b66d89effcc8698e434d6f50270595605c1d42cb844d4",
    "modules/attention_module.py": "7f34d941393fb9dfc69f14ff02f731e5e1487f55cde9e79a7195d328922db2fb",
    "modules/decoder_module.py": "a6c99bfdfed9cefd4184662b4a093d179e6a0c805d92ad21122ebaf95e05ee20",
    "backbones/__init__.py": None,
    "backbones/SwinTransformer.py": "78c53d0cbd05f9a0d3cbd1dfbf86f6b989f8708281b6915e5267b03850cd8d82",
}
_UPSTREAM_SOURCE_TRANSFORMATIONS = {
    "LICENSE": [],
    "__init__.py": ["add_empty_namespace_initializer"],
    "InSPyReNet.py": [
        "remove_os_sys_imports_and_sys_path_mutation",
        "rewrite_transparent_background_imports_to_vendored_relative_namespace",
        "normalize_terminal_newline",
        "strip_ascii_trailing_whitespace",
    ],
    "modules/__init__.py": ["add_empty_namespace_initializer"],
    "modules/layers.py": ["strip_ascii_trailing_whitespace"],
    "modules/context_module.py": [],
    "modules/attention_module.py": [
        "rewrite_transparent_background_import_to_vendored_relative_namespace",
        "normalize_terminal_newline",
        "strip_ascii_trailing_whitespace",
    ],
    "modules/decoder_module.py": [
        "normalize_terminal_newline",
        "strip_ascii_trailing_whitespace",
    ],
    "backbones/__init__.py": ["add_empty_namespace_initializer"],
    "backbones/SwinTransformer.py": [
        "strip_ascii_trailing_whitespace",
        "normalize_terminal_newline",
    ],
}
_UPSTREAM_SOURCE_STRUCTURAL_PATHS = frozenset(
    {
        Path("runtime/_vendor"),
        _UPSTREAM_SOURCE_MANIFEST,
    }
)


def _attested_upstream_source_paths(root_path: Path) -> frozenset[Path]:
    """Return exact upstream files only when the complete closure is authentic."""

    manifest_path = root_path / _UPSTREAM_SOURCE_MANIFEST
    try:
        manifest_stat = manifest_path.lstat()
        if manifest_path.is_symlink() or not stat.S_ISREG(manifest_stat.st_mode):
            return frozenset()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return frozenset()
    if type(manifest) is not dict or set(manifest) != {
        "source_repository",
        "upstream_commit",
        "upstream_tree",
        "source_license",
        "vendored_namespace",
        "files",
    }:
        return frozenset()
    if (
        manifest["source_repository"] != _UPSTREAM_SOURCE_REPOSITORY
        or manifest["upstream_commit"] != _UPSTREAM_SOURCE_REVISION
        or manifest["upstream_tree"] != _UPSTREAM_SOURCE_TREE
        or manifest["source_license"] != "MIT"
        or manifest["vendored_namespace"] != _UPSTREAM_SOURCE_NAMESPACE
        or type(manifest["files"]) is not list
    ):
        return frozenset()
    entries: dict[str, dict] = {}
    for entry in manifest["files"]:
        if type(entry) is not dict or set(entry) != {
            "upstream_path",
            "local_path",
            "upstream_sha256",
            "local_sha256",
            "transformations",
        }:
            return frozenset()
        local_path = entry["local_path"]
        if type(local_path) is not str:
            return frozenset()
        pure_path = PurePosixPath(local_path)
        if (
            pure_path.is_absolute()
            or pure_path.as_posix() != local_path
            or not pure_path.parts
            or any(part in {"", ".", ".."} for part in pure_path.parts)
            or local_path in entries
        ):
            return frozenset()
        entries[local_path] = entry
    if set(entries) != set(_UPSTREAM_SOURCE_FILE_ATTESTATIONS):
        return frozenset()
    attested: set[Path] = set()
    for local_path, (upstream_path, local_sha256) in (
        _UPSTREAM_SOURCE_FILE_ATTESTATIONS.items()
    ):
        entry = entries[local_path]
        if (
            entry["upstream_path"] != upstream_path
            or entry["upstream_sha256"]
            != _UPSTREAM_SOURCE_ORIGINAL_SHA256[local_path]
            or entry["local_sha256"] != local_sha256
            or entry["transformations"]
            != _UPSTREAM_SOURCE_TRANSFORMATIONS[local_path]
        ):
            return frozenset()
        source_path = root_path / _UPSTREAM_SOURCE_ROOT / local_path
        try:
            source_stat = source_path.lstat()
            if source_path.is_symlink() or not stat.S_ISREG(source_stat.st_mode):
                return frozenset()
            payload = source_path.read_bytes()
        except OSError:
            return frozenset()
        if sha256(payload).hexdigest() != local_sha256:
            return frozenset()
        if upstream_path is not None:
            attested.add(_UPSTREAM_SOURCE_ROOT / local_path)
    return frozenset(attested)


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

    def is_identity_shaped_name(name: str) -> bool:
        return bool(
            has_weak_semantic_identity_value(name)
            or has_ordinal_identity_text(name)
            or _is_formal_identity_context(name, frozenset(), frozenset())
        )

    def is_math_annotation(annotation: ast.expr) -> bool:
        if isinstance(annotation, ast.Name):
            return annotation.id in {"int", "float", "complex", "Tensor", "NDArray"}
        if isinstance(annotation, ast.Attribute):
            return annotation.attr in {
                "Tensor",
                "ndarray",
                "float16",
                "float32",
                "float64",
            }
        return False

    value_events: dict[
        tuple[int, str],
        list[tuple[tuple[int, int], bool]],
    ] = {}
    attribute_events: dict[
        tuple[int, tuple[str, ...]],
        list[tuple[tuple[int, int], bool]],
    ] = {}

    mathematical_parameter_role = re.compile(
        r"(?:^|_)(?:array|axis|coefficient|delta|dim|dimension|factor|index|"
        r"latent|matrix|scalar|scale|score|scores|signal|tensor|value|values|"
        r"vector|weight)(?:_|$)",
        re.IGNORECASE,
    )
    scope_parameters: dict[int, dict[str, bool]] = {}
    for scope in ast.walk(tree):
        if not isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        arguments = scope.args
        parameter_nodes = [
            *arguments.posonlyargs,
            *arguments.args,
            *arguments.kwonlyargs,
        ]
        if arguments.vararg is not None:
            parameter_nodes.append(arguments.vararg)
        if arguments.kwarg is not None:
            parameter_nodes.append(arguments.kwarg)
        scope_parameters[id(scope)] = {
            argument.arg: bool(
                (
                    argument.annotation is not None
                    and is_math_annotation(argument.annotation)
                )
                or mathematical_parameter_role.search(argument.arg)
            )
            for argument in parameter_nodes
        }

    local_functions: dict[tuple[int, str], list[ast.FunctionDef | ast.AsyncFunctionDef]] = {}
    local_classes: dict[tuple[int, str], list[ast.ClassDef]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        scope = lexical_scope(node)
        if scope is not None and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            local_functions.setdefault((id(scope), node.name), []).append(node)
        elif scope is not None:
            local_classes.setdefault((id(scope), node.name), []).append(node)

    import_bindings: dict[
        tuple[int, str],
        list[tuple[tuple[int, int], str | None]],
    ] = {}
    assignment_name_positions: dict[
        tuple[int, str],
        list[tuple[int, int]],
    ] = {}
    assignment_sources: dict[
        tuple[int, str],
        list[tuple[tuple[int, int], ast.Assign | ast.AnnAssign]],
    ] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        position = (
            getattr(node, "end_lineno", node.lineno),
            getattr(node, "end_col_offset", node.col_offset),
        )
        for target in targets:
            scope = lexical_scope(target)
            if scope is None:
                continue
            for name in _python_assignment_target_names(target):
                assignment_name_positions.setdefault((id(scope), name), []).append(
                    position
                )
                assignment_sources.setdefault((id(scope), name), []).append(
                    (position, node)
                )
    namespace_modules = frozenset({"math", "numpy", "torch"})
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        scope = lexical_scope(node)
        if scope is None:
            continue
        position = (
            getattr(node, "end_lineno", node.lineno),
            getattr(node, "end_col_offset", node.col_offset),
        )
        if isinstance(node, ast.Import):
            for alias in node.names:
                bound_name = alias.asname or alias.name.split(".", maxsplit=1)[0]
                canonical_namespace = (
                    alias.name if alias.name in namespace_modules else None
                )
                import_bindings.setdefault((id(scope), bound_name), []).append(
                    (position, canonical_namespace)
                )
        else:
            for alias in node.names:
                bound_name = alias.asname or alias.name
                import_bindings.setdefault((id(scope), bound_name), []).append(
                    (position, None)
                )

    future_annotations = any(
        isinstance(node, ast.ImportFrom)
        and node.module == "__future__"
        and any(alias.name == "annotations" for alias in node.names)
        for node in getattr(tree, "body", ())
    )

    def is_annotation_reference(node: ast.AST) -> bool:
        child = node
        current = parents.get(child)
        while current is not None:
            if isinstance(current, ast.arg) and current.annotation is child:
                return True
            if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return current.returns is child
            if isinstance(current, ast.AnnAssign) and current.annotation is child:
                return True
            if isinstance(current, (ast.stmt, ast.Lambda)):
                return False
            child = current
            current = parents.get(current)
        return False

    def expression_evaluation_scope(node: ast.AST) -> ast.AST | None:
        scope = lexical_scope(node)
        if not isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            return scope
        child = node
        while parents.get(child) is not scope:
            parent = parents.get(child)
            if parent is None:
                return scope
            child = parent
        body = (
            scope.body
            if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef))
            else scope.body
        )
        if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)) and child in body:
            return scope
        if isinstance(scope, ast.Lambda) and child is body:
            return scope
        return lexical_scope(scope)

    direct_calls: dict[tuple[int, str], list[tuple[int, int]]] = {}
    direct_call_nodes: dict[tuple[int, str], list[ast.Call]] = {}
    immediate_result_invocations: dict[
        tuple[int, str],
        list[tuple[int, int]],
    ] = {}
    callable_invocation_depths: dict[
        tuple[int, str, tuple[int, int]],
        int,
    ] = {}

    def possible_call_targets(
        called_name: str,
        scope: ast.AST,
        reference_position: tuple[int, int],
        seen: frozenset[str] = frozenset(),
    ) -> frozenset[str]:
        if called_name in seen:
            return frozenset()

        def is_conditional_binding(node: ast.AST) -> bool:
            current = parents.get(node)
            while current is not None and current is not scope:
                if isinstance(
                    current,
                    (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.Match),
                ):
                    return True
                current = parents.get(current)
            return False

        binding_events: list[
            tuple[tuple[int, int], bool, frozenset[str]]
        ] = []
        if called_name in scope_parameters.get(id(scope), {}):
            binding_events.append(((-1, -1), False, frozenset()))
        for definition in local_functions.get((id(scope), called_name), ()):
            position = (
                getattr(definition, "end_lineno", definition.lineno),
                getattr(definition, "end_col_offset", definition.col_offset),
            )
            if position < reference_position:
                binding_events.append(
                    (
                        position,
                        is_conditional_binding(definition),
                        frozenset({called_name}),
                    )
                )
        for definition in local_classes.get((id(scope), called_name), ()):
            position = (
                getattr(definition, "end_lineno", definition.lineno),
                getattr(definition, "end_col_offset", definition.col_offset),
            )
            if position < reference_position:
                binding_events.append(
                    (position, is_conditional_binding(definition), frozenset())
                )
        for position, _ in import_bindings.get((id(scope), called_name), ()):
            if position < reference_position:
                binding_events.append((position, False, frozenset()))
        for position, assignment in assignment_sources.get(
            (id(scope), called_name), ()
        ):
            if position >= reference_position:
                continue
            targets = frozenset()
            if (
                isinstance(assignment, ast.Assign)
                and len(assignment.targets) == 1
                and isinstance(assignment.targets[0], ast.Name)
                and assignment.targets[0].id == called_name
                and isinstance(assignment.value, ast.Name)
            ):
                targets = possible_call_targets(
                    assignment.value.id,
                    scope,
                    (assignment.lineno, assignment.col_offset),
                    seen | {called_name},
                )
            binding_events.append(
                (position, is_conditional_binding(assignment), targets)
            )

        def reduced_targets(before: tuple[int, int]) -> frozenset[str]:
            targets = frozenset()
            for _, conditional, event_targets in sorted(
                (
                    event
                    for event in binding_events
                    if event[0] < before
                ),
                key=lambda event: event[0],
            ):
                targets = targets | event_targets if conditional else event_targets
            return targets

        def assignment_targets(
            assignment: ast.Assign | ast.AnnAssign,
        ) -> frozenset[str] | None:
            targets = (
                assignment.targets
                if isinstance(assignment, ast.Assign)
                else [assignment.target]
            )
            if not any(
                isinstance(target, ast.Name) and target.id == called_name
                for target in targets
            ):
                return None
            if not isinstance(assignment.value, ast.Name):
                return frozenset()
            return possible_call_targets(
                assignment.value.id,
                scope,
                (assignment.lineno, assignment.col_offset),
                seen | {called_name},
            )

        def alias_branch_targets(
            statements: list[ast.stmt],
            incoming: frozenset[str],
        ) -> frozenset[str]:
            targets = incoming
            for statement in statements:
                if isinstance(statement, (ast.Assign, ast.AnnAssign)):
                    assigned_targets = assignment_targets(statement)
                    if assigned_targets is not None:
                        targets = assigned_targets
                    continue
                if isinstance(statement, ast.If):
                    if (
                        isinstance(statement.test, ast.Constant)
                        and type(statement.test.value) is bool
                    ):
                        selected = (
                            statement.body
                            if statement.test.value
                            else statement.orelse
                        )
                        targets = alias_branch_targets(selected, targets)
                    else:
                        body_targets = alias_branch_targets(statement.body, targets)
                        else_targets = (
                            alias_branch_targets(statement.orelse, targets)
                            if statement.orelse
                            else targets
                        )
                        targets = body_targets | else_targets
                    continue
                if isinstance(statement, ast.Match):
                    outcomes = [
                        alias_branch_targets(case.body, targets)
                        for case in statement.cases
                        if not (
                            isinstance(case.guard, ast.Constant)
                            and type(case.guard.value) is bool
                            and not case.guard.value
                        )
                    ]
                    last_case = statement.cases[-1] if statement.cases else None
                    exhaustive = bool(
                        last_case is not None
                        and (
                            last_case.guard is None
                            or (
                                isinstance(last_case.guard, ast.Constant)
                                and type(last_case.guard.value) is bool
                                and last_case.guard.value
                            )
                        )
                        and isinstance(last_case.pattern, ast.MatchAs)
                        and last_case.pattern.pattern is None
                        and last_case.pattern.name is None
                    )
                    if not exhaustive:
                        outcomes.append(targets)
                    targets = frozenset().union(*outcomes)
                    continue
                if isinstance(statement, ast.Try):
                    body_targets = alias_branch_targets(statement.body, targets)
                    normal_targets = alias_branch_targets(
                        statement.orelse,
                        body_targets,
                    )
                    outcomes = [normal_targets]
                    outcomes.extend(
                        alias_branch_targets(handler.body, incoming)
                        for incoming in (targets, body_targets)
                        for handler in statement.handlers
                    )
                    targets = frozenset().union(*outcomes)
                    if statement.finalbody:
                        targets = alias_branch_targets(statement.finalbody, targets)
                    continue
                if isinstance(
                    statement,
                    (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
                ):
                    continue
                if any(
                    lexical_scope(target) is scope
                    and called_name in _python_assignment_target_names(target)
                    for node in ast.walk(statement)
                    if isinstance(node, (ast.Assign, ast.AnnAssign))
                    for target in (
                        node.targets if isinstance(node, ast.Assign) else [node.target]
                    )
                ):
                    targets = targets | frozenset()
            return targets

        conditional_merges = sorted(
            (
                node
                for node in ast.walk(scope)
                if isinstance(node, (ast.If, ast.Match, ast.Try))
                and lexical_scope(node) is scope
                and (
                    getattr(node, "end_lineno", node.lineno),
                    getattr(node, "end_col_offset", node.col_offset),
                )
                < reference_position
            ),
            key=lambda node: (
                getattr(node, "end_lineno", node.lineno),
                getattr(node, "end_col_offset", node.col_offset),
            ),
        )
        for conditional in conditional_merges:
            assigned_here = any(
                lexical_scope(target) is scope
                and called_name in _python_assignment_target_names(target)
                for node in ast.walk(conditional)
                if isinstance(node, (ast.Assign, ast.AnnAssign))
                for target in (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                )
            )
            if not assigned_here:
                continue
            start = (conditional.lineno, conditional.col_offset)
            incoming = reduced_targets(start)
            if isinstance(conditional, ast.Match):
                outcomes = [
                    alias_branch_targets(case.body, incoming)
                    for case in conditional.cases
                    if not (
                        isinstance(case.guard, ast.Constant)
                        and type(case.guard.value) is bool
                        and not case.guard.value
                    )
                ]
                last_case = conditional.cases[-1] if conditional.cases else None
                exhaustive = bool(
                    last_case is not None
                    and (
                        last_case.guard is None
                        or (
                            isinstance(last_case.guard, ast.Constant)
                            and type(last_case.guard.value) is bool
                            and last_case.guard.value
                        )
                    )
                    and isinstance(last_case.pattern, ast.MatchAs)
                    and last_case.pattern.pattern is None
                    and last_case.pattern.name is None
                )
                if not exhaustive:
                    outcomes.append(incoming)
                merged_targets = frozenset().union(*outcomes)
            elif isinstance(conditional, ast.Try):
                body_targets = alias_branch_targets(conditional.body, incoming)
                normal_targets = alias_branch_targets(
                    conditional.orelse,
                    body_targets,
                )
                outcomes = [normal_targets]
                outcomes.extend(
                    alias_branch_targets(handler.body, handler_incoming)
                    for handler_incoming in (incoming, body_targets)
                    for handler in conditional.handlers
                )
                merged_targets = frozenset().union(*outcomes)
                if conditional.finalbody:
                    merged_targets = alias_branch_targets(
                        conditional.finalbody,
                        merged_targets,
                    )
            elif (
                isinstance(conditional.test, ast.Constant)
                and type(conditional.test.value) is bool
            ):
                selected = (
                    conditional.body
                    if conditional.test.value
                    else conditional.orelse
                )
                merged_targets = alias_branch_targets(selected, incoming)
            else:
                body_targets = alias_branch_targets(conditional.body, incoming)
                else_targets = (
                    alias_branch_targets(conditional.orelse, incoming)
                    if conditional.orelse
                    else incoming
                )
                merged_targets = body_targets | else_targets
            end = (
                getattr(conditional, "end_lineno", conditional.lineno),
                getattr(conditional, "end_col_offset", conditional.col_offset),
            )
            binding_events.append((end, False, merged_targets))
        if not binding_events:
            return frozenset()
        targets = frozenset()
        for _, conditional, event_targets in sorted(
            binding_events,
            key=lambda event: event[0],
        ):
            targets = targets | event_targets if conditional else event_targets
        return targets

    def historical_call_targets(
        called_name: str,
        scope: ast.AST,
        seen: frozenset[str] = frozenset(),
    ) -> frozenset[str]:
        if called_name in seen:
            return frozenset()
        targets = {
            called_name
            for _ in local_functions.get((id(scope), called_name), ())
        }
        for _, assignment in assignment_sources.get((id(scope), called_name), ()):
            if isinstance(assignment.value, ast.Name):
                targets.update(
                    historical_call_targets(
                        assignment.value.id,
                        scope,
                        seen | {called_name},
                    )
                )
        return frozenset(targets)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if future_annotations and is_annotation_reference(node):
            continue
        scope = expression_evaluation_scope(node)
        if scope is not None:
            call_position = (node.lineno, node.col_offset)
            direct_calls.setdefault((id(scope), node.func.id), []).append(call_position)
            direct_call_nodes.setdefault((id(scope), node.func.id), []).append(node)
            possible_targets = possible_call_targets(
                node.func.id,
                scope,
                call_position,
            )
            if not possible_targets:
                possible_targets = historical_call_targets(node.func.id, scope)
            for possible_target in possible_targets:
                if possible_target != node.func.id:
                    direct_calls.setdefault(
                        (id(scope), possible_target), []
                    ).append(call_position)
                    direct_call_nodes.setdefault(
                        (id(scope), possible_target), []
                    ).append(node)
            parent = parents.get(node)
            invocation_depth = 0
            call_ancestor: ast.AST = node
            while isinstance(parent, ast.Call) and parent.func is call_ancestor:
                invocation_depth += 1
                call_ancestor = parent
                parent = parents.get(parent)
            for target in {node.func.id, *possible_targets}:
                callable_invocation_depths[
                    (id(scope), target, call_position)
                ] = max(
                    invocation_depth,
                    callable_invocation_depths.get(
                        (id(scope), target, call_position),
                        0,
                    ),
                )
            parent = parents.get(node)
            if isinstance(parent, ast.Call) and parent.func is node:
                for target in {node.func.id, *possible_targets}:
                    immediate_result_invocations.setdefault(
                        (id(scope), target), []
                    ).append(call_position)

    indirect_function_references: dict[
        tuple[int, str],
        list[tuple[int, int]],
    ] = {}

    def is_simple_alias_reference(node: ast.Name) -> bool:
        parent = parents.get(node)
        if not isinstance(parent, ast.Assign) or parent.value is not node:
            return False
        return bool(
            len(parent.targets) == 1
            and isinstance(parent.targets[0], ast.Name)
        )

    for node in ast.walk(tree):
        if not isinstance(node, ast.Name) or not isinstance(node.ctx, ast.Load):
            continue
        parent = parents.get(node)
        if isinstance(parent, ast.Call) and parent.func is node:
            continue
        if future_annotations and is_annotation_reference(node):
            continue
        if is_simple_alias_reference(node):
            continue
        scope = expression_evaluation_scope(node)
        if scope is None:
            continue
        reference_position = (node.lineno, node.col_offset)
        targets = possible_call_targets(node.id, scope, reference_position)
        for target in targets:
            indirect_function_references.setdefault((id(scope), target), []).append(
                reference_position
            )

    def lambda_free_call_names(lambda_node: ast.Lambda) -> frozenset[str]:
        local_parameters = {
            argument.arg
            for argument in (
                *lambda_node.args.posonlyargs,
                *lambda_node.args.args,
                *lambda_node.args.kwonlyargs,
            )
        }
        if lambda_node.args.vararg is not None:
            local_parameters.add(lambda_node.args.vararg.arg)
        if lambda_node.args.kwarg is not None:
            local_parameters.add(lambda_node.args.kwarg.arg)
        return frozenset(
            node.func.id
            for node in ast.walk(lambda_node.body)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and expression_evaluation_scope(node) is lambda_node
            and node.func.id not in local_parameters
        )

    def active_lambda_nodes(
        name: str,
        scope: ast.AST,
        position: tuple[int, int],
        seen: frozenset[str] = frozenset(),
    ) -> tuple[ast.Lambda, ...]:
        if name in seen:
            return ()
        active: tuple[ast.Lambda, ...] = ()
        for assignment_position, assignment in sorted(
            assignment_sources.get((id(scope), name), ()),
            key=lambda item: item[0],
        ):
            if assignment_position >= position:
                continue
            value = assignment.value
            if isinstance(value, ast.Lambda):
                active = (value,)
            elif isinstance(value, ast.Name):
                active = active_lambda_nodes(
                    value.id,
                    scope,
                    (assignment.lineno, assignment.col_offset),
                    seen | {name},
                )
            else:
                active = ()
        return active

    for (scope_id, called_name), call_nodes in tuple(direct_call_nodes.items()):
        scope = next(
            (candidate for candidate in ast.walk(tree) if id(candidate) == scope_id),
            None,
        )
        if scope is None:
            continue
        for call_node in call_nodes:
            call_position = (call_node.lineno, call_node.col_offset)
            for lambda_node in active_lambda_nodes(
                called_name,
                scope,
                call_position,
            ):
                for free_name in lambda_free_call_names(lambda_node):
                    for target in possible_call_targets(
                        free_name,
                        scope,
                        call_position,
                    ):
                        direct_calls.setdefault((scope_id, target), []).append(
                            call_position
                        )

    def active_free_callable_names(
        name: str,
        scope: ast.AST,
        position: tuple[int, int],
        seen: frozenset[str] = frozenset(),
    ) -> frozenset[str]:
        if name in seen:
            return frozenset()
        has_local_nonassignment = bool(
            name in scope_parameters.get(id(scope), {})
            or local_functions.get((id(scope), name))
            or local_classes.get((id(scope), name))
            or import_bindings.get((id(scope), name))
        )
        sources = assignment_sources.get((id(scope), name), ())
        if not sources and not has_local_nonassignment:
            return frozenset({name})

        def source_names(assignment: ast.Assign | ast.AnnAssign) -> frozenset[str]:
            if not isinstance(assignment.value, ast.Name):
                return frozenset()
            return active_free_callable_names(
                assignment.value.id,
                scope,
                (assignment.lineno, assignment.col_offset),
                seen | {name},
            )

        events: list[tuple[tuple[int, int], bool, frozenset[str]]] = []
        for source_position, assignment in sources:
            if source_position >= position:
                continue
            conditional = False
            current = parents.get(assignment)
            while current is not None and current is not scope:
                if isinstance(current, ast.If):
                    conditional = True
                    break
                current = parents.get(current)
            events.append((source_position, conditional, source_names(assignment)))

        def reduce_events(before: tuple[int, int]) -> frozenset[str]:
            result = frozenset()
            for _, conditional, event_names in sorted(
                (event for event in events if event[0] < before),
                key=lambda event: event[0],
            ):
                result = result | event_names if conditional else event_names
            return result

        def branch_names(
            statements: list[ast.stmt],
            incoming: frozenset[str],
        ) -> frozenset[str]:
            result = incoming
            for statement in statements:
                if isinstance(statement, (ast.Assign, ast.AnnAssign)):
                    targets = (
                        statement.targets
                        if isinstance(statement, ast.Assign)
                        else [statement.target]
                    )
                    if any(
                        isinstance(target, ast.Name) and target.id == name
                        for target in targets
                    ):
                        result = source_names(statement)
                    continue
                if isinstance(statement, ast.If):
                    if (
                        isinstance(statement.test, ast.Constant)
                        and type(statement.test.value) is bool
                    ):
                        selected = (
                            statement.body
                            if statement.test.value
                            else statement.orelse
                        )
                        result = branch_names(selected, result)
                    else:
                        body_names = branch_names(statement.body, result)
                        else_names = (
                            branch_names(statement.orelse, result)
                            if statement.orelse
                            else result
                        )
                        result = body_names | else_names
            return result

        for conditional in sorted(
            (
                node
                for node in ast.walk(scope)
                if isinstance(node, ast.If)
                and lexical_scope(node) is scope
                and (
                    getattr(node, "end_lineno", node.lineno),
                    getattr(node, "end_col_offset", node.col_offset),
                )
                < position
            ),
            key=lambda node: (
                getattr(node, "end_lineno", node.lineno),
                getattr(node, "end_col_offset", node.col_offset),
            ),
        ):
            incoming = reduce_events((conditional.lineno, conditional.col_offset))
            if (
                isinstance(conditional.test, ast.Constant)
                and type(conditional.test.value) is bool
            ):
                selected = (
                    conditional.body if conditional.test.value else conditional.orelse
                )
                merged = branch_names(selected, incoming)
            else:
                body_names = branch_names(conditional.body, incoming)
                else_names = (
                    branch_names(conditional.orelse, incoming)
                    if conditional.orelse
                    else incoming
                )
                merged = body_names | else_names
            end = (
                getattr(conditional, "end_lineno", conditional.lineno),
                getattr(conditional, "end_col_offset", conditional.col_offset),
            )
            events.append((end, False, merged))
        return reduce_events(position)

    function_free_call_names: dict[int, frozenset[str]] = {}
    function_returned_free_names: dict[int, frozenset[str]] = {}
    function_returned_lambdas: dict[int, tuple[ast.Lambda, ...]] = {}
    function_default_lambda_calls: dict[
        int,
        tuple[tuple[str, ast.Lambda], ...],
    ] = {}
    for function_group in local_functions.values():
        for function in function_group:
            local_bindings = {
                *scope_parameters.get(id(function), {}),
                *(
                    name
                    for scope_id, name in assignment_name_positions
                    if scope_id == id(function)
                ),
                *(
                    name
                    for scope_id, name in local_functions
                    if scope_id == id(function)
                ),
                *(
                    name
                    for scope_id, name in local_classes
                    if scope_id == id(function)
                ),
                *(
                    name
                    for scope_id, name in import_bindings
                    if scope_id == id(function)
                ),
            }
            free_call_names = {
                node.func.id
                for node in ast.walk(function)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and expression_evaluation_scope(node) is function
                and node.func.id not in local_bindings
            }
            function_free_call_names[id(function)] = frozenset(free_call_names)
            returned_names: set[str] = set()
            returned_lambdas: list[ast.Lambda] = []
            for node in ast.walk(function):
                if not isinstance(node, ast.Return) or lexical_scope(node) is not function:
                    continue
                if isinstance(node.value, ast.IfExp):
                    if (
                        isinstance(node.value.test, ast.Constant)
                        and type(node.value.test.value) is bool
                    ):
                        expressions = (
                            node.value.body
                            if node.value.test.value
                            else node.value.orelse,
                        )
                    else:
                        expressions = (node.value.body, node.value.orelse)
                else:
                    expressions = (node.value,)
                for expression in expressions:
                    if isinstance(expression, ast.Name):
                        returned_names.update(
                            active_free_callable_names(
                                expression.id,
                                function,
                                (expression.lineno, expression.col_offset),
                            )
                        )
                    elif isinstance(expression, ast.Lambda):
                        returned_lambdas.append(expression)
            function_returned_free_names[id(function)] = frozenset(returned_names)
            function_returned_lambdas[id(function)] = tuple(returned_lambdas)
            positional_parameters = [
                *function.args.posonlyargs,
                *function.args.args,
            ]
            positional_default_start = len(positional_parameters) - len(
                function.args.defaults
            )
            defaults: list[tuple[str, ast.Lambda]] = []
            for index, parameter in enumerate(positional_parameters):
                default_index = index - positional_default_start
                if default_index < 0:
                    continue
                default = function.args.defaults[default_index]
                if isinstance(default, ast.Lambda):
                    defaults.append((parameter.arg, default))
            for parameter, default in zip(
                function.args.kwonlyargs,
                function.args.kw_defaults,
                strict=True,
            ):
                if isinstance(default, ast.Lambda):
                    defaults.append((parameter.arg, default))
            called_parameter_names = {
                node.func.id
                for node in ast.walk(function)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and expression_evaluation_scope(node) is function
            }
            function_default_lambda_calls[id(function)] = tuple(
                (name, default)
                for name, default in defaults
                if name in called_parameter_names
            )

    scopes_by_id = {id(scope): scope for scope in ast.walk(tree)}

    def nested_invocation_depth(call: ast.Call) -> int:
        depth = 0
        child: ast.AST = call
        parent = parents.get(child)
        while isinstance(parent, ast.Call) and parent.func is child:
            depth += 1
            child = parent
            parent = parents.get(parent)
        return depth

    def execute_lambda_value(
        lambda_node: ast.Lambda,
        resolution_scope: ast.AST,
        position: tuple[int, int],
        remaining_depth: int,
        seen: frozenset[int] = frozenset(),
    ) -> None:
        if id(lambda_node) in seen:
            return
        for free_name in lambda_free_call_names(lambda_node):
            for target in possible_call_targets(
                free_name,
                resolution_scope,
                position,
            ):
                direct_calls.setdefault((id(resolution_scope), target), []).append(
                    position
                )
        for nested_call in (
            node
            for node in ast.walk(lambda_node.body)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Lambda)
            and expression_evaluation_scope(node) is lambda_node
        ):
            execute_lambda_value(
                nested_call.func,
                resolution_scope,
                position,
                nested_invocation_depth(nested_call),
                seen | {id(lambda_node)},
            )
        if remaining_depth <= 0:
            return

        def execute_returned(expression: ast.expr) -> None:
            if isinstance(expression, ast.Lambda):
                execute_lambda_value(
                    expression,
                    resolution_scope,
                    position,
                    remaining_depth - 1,
                    seen | {id(lambda_node)},
                )
                return
            if isinstance(expression, ast.Name):
                for nested_lambda in active_lambda_nodes(
                    expression.id,
                    resolution_scope,
                    position,
                ):
                    execute_lambda_value(
                        nested_lambda,
                        resolution_scope,
                        position,
                        remaining_depth - 1,
                        seen | {id(lambda_node)},
                    )
                for target in possible_call_targets(
                    expression.id,
                    resolution_scope,
                    position,
                ):
                    direct_calls.setdefault(
                        (id(resolution_scope), target), []
                    ).append(position)
                    callable_invocation_depths[
                        (id(resolution_scope), target, position)
                    ] = max(
                        remaining_depth - 1,
                        callable_invocation_depths.get(
                            (id(resolution_scope), target, position),
                            -1,
                        ),
                    )
                return
            if isinstance(expression, ast.IfExp):
                execute_returned(expression.body)
                execute_returned(expression.orelse)
                return
            if isinstance(expression, ast.Call) and isinstance(
                expression.func,
                ast.Name,
            ):
                for target in possible_call_targets(
                    expression.func.id,
                    resolution_scope,
                    position,
                ):
                    direct_calls.setdefault(
                        (id(resolution_scope), target), []
                    ).append(position)
                    callable_invocation_depths[
                        (id(resolution_scope), target, position)
                    ] = max(
                        remaining_depth,
                        callable_invocation_depths.get(
                            (id(resolution_scope), target, position),
                            -1,
                        ),
                    )

        execute_returned(lambda_node.body)

    for call in (
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and not (future_annotations and is_annotation_reference(node))
    ):
        scope = expression_evaluation_scope(call)
        if scope is None:
            continue
        lambdas = (
            (call.func,)
            if isinstance(call.func, ast.Lambda)
            else (
                active_lambda_nodes(
                    call.func.id,
                    scope,
                    (call.lineno, call.col_offset),
                )
                if isinstance(call.func, ast.Name)
                else ()
            )
        )
        for lambda_node in lambdas:
            execute_lambda_value(
                lambda_node,
                scope,
                (call.lineno, call.col_offset),
                nested_invocation_depth(call),
            )

    def call_uses_default(
        function: ast.FunctionDef | ast.AsyncFunctionDef,
        call: ast.Call,
        parameter_name: str,
    ) -> bool:
        positional_parameters = [
            *function.args.posonlyargs,
            *function.args.args,
        ]
        parameter_index = next(
            (
                index
                for index, parameter in enumerate(positional_parameters)
                if parameter.arg == parameter_name
            ),
            None,
        )
        if parameter_index is not None and parameter_index < len(call.args):
            return False
        return not any(
            keyword.arg == parameter_name
            for keyword in call.keywords
            if keyword.arg is not None
        )

    for (scope_id, function_name), function_group in local_functions.items():
        candidate_scope = scopes_by_id.get(scope_id)
        if candidate_scope is None:
            continue
        immediate_positions = set(
            immediate_result_invocations.get((scope_id, function_name), ())
        )
        for function in function_group:
            for position in immediate_positions:
                for returned_lambda in function_returned_lambdas.get(
                    id(function), ()
                ):
                    execute_lambda_value(
                        returned_lambda,
                        candidate_scope,
                        position,
                        0,
                    )
                for free_name in function_returned_free_names.get(id(function), ()):
                    for target in possible_call_targets(
                        free_name,
                        candidate_scope,
                        position,
                    ):
                        direct_calls.setdefault((scope_id, target), []).append(position)
            for call in direct_call_nodes.get((scope_id, function_name), ()):
                position = (call.lineno, call.col_offset)
                for parameter_name, default in function_default_lambda_calls.get(
                    id(function), ()
                ):
                    if not call_uses_default(function, call, parameter_name):
                        continue
                    for free_name in lambda_free_call_names(default):
                        for target in possible_call_targets(
                            free_name,
                            candidate_scope,
                            position,
                        ):
                            direct_calls.setdefault((scope_id, target), []).append(
                                position
                            )

    processed_callable_depths: dict[
        tuple[int, str, tuple[int, int]],
        int,
    ] = {}
    pending_callable_depths = dict(callable_invocation_depths)
    while pending_callable_depths:
        key, depth = pending_callable_depths.popitem()
        if depth <= processed_callable_depths.get(key, -1):
            continue
        processed_callable_depths[key] = depth
        if depth <= 0:
            continue
        scope_id, function_name, position = key
        candidate_scope = scopes_by_id.get(scope_id)
        if candidate_scope is None:
            continue
        for function in local_functions.get((scope_id, function_name), ()):
            for free_name in function_returned_free_names.get(id(function), ()):
                for target in possible_call_targets(
                    free_name,
                    candidate_scope,
                    position,
                ):
                    direct_calls.setdefault((scope_id, target), []).append(position)
                    target_key = (scope_id, target, position)
                    target_depth = depth - 1
                    if target_depth > processed_callable_depths.get(target_key, -1):
                        pending_callable_depths[target_key] = max(
                            target_depth,
                            pending_callable_depths.get(target_key, -1),
                        )

    changed = True
    while changed:
        changed = False
        for (scope_id, function_name), function_group in local_functions.items():
            exposure_positions = {
                *direct_calls.get((scope_id, function_name), ()),
                *indirect_function_references.get((scope_id, function_name), ()),
            }
            if not exposure_positions:
                continue
            candidate_scope = scopes_by_id.get(scope_id)
            if candidate_scope is None:
                continue
            for function in function_group:
                for free_name in function_free_call_names.get(id(function), ()):
                    for exposure_position in exposure_positions:
                        for target in possible_call_targets(
                            free_name,
                            candidate_scope,
                            exposure_position,
                        ):
                            positions = direct_calls.setdefault(
                                (scope_id, target), []
                            )
                            if exposure_position not in positions:
                                positions.append(exposure_position)
                                changed = True

    def latest_event(
        events: list[tuple[tuple[int, int], bool]],
        position: tuple[int, int],
    ) -> bool | None:
        prior_events = [event for event in events if event[0] < position]
        if not prior_events:
            return None
        return max(
            enumerate(prior_events),
            key=lambda indexed_event: (indexed_event[1][0], indexed_event[0]),
        )[1][1]

    def attribute_key(node: ast.Attribute) -> tuple[str, ...] | None:
        parts = [node.attr]
        value = node.value
        while isinstance(value, ast.Attribute):
            parts.append(value.attr)
            value = value.value
        if not isinstance(value, ast.Name):
            return None
        parts.append(value.id)
        return tuple(reversed(parts))

    def has_prior_math_binding(node: ast.Name, scope: ast.AST) -> bool:
        return bool(
            latest_event(
                value_events.get((id(scope), node.id), []),
                (node.lineno, node.col_offset),
            )
        )

    resolving_functions: set[int] = set()
    validated_function_parameters: dict[int, frozenset[str]] = {}
    validated_keyword_containers: dict[int, str] = {}
    keyword_container_invalidations: dict[
        tuple[int, str],
        list[tuple[int, int]],
    ] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.NamedExpr) and isinstance(node.target, ast.Name):
            scope = lexical_scope(node.target)
            if scope is not None:
                keyword_container_invalidations.setdefault(
                    (id(scope), node.target.id), []
                ).append((node.lineno, node.col_offset))
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.attr != "values"
        ):
            scope = expression_evaluation_scope(node)
            if scope is not None:
                keyword_container_invalidations.setdefault(
                    (id(scope), node.func.value.id), []
                ).append((node.lineno, node.col_offset))

    def statement_block_return_proof(
        statements: list[ast.stmt],
    ) -> tuple[bool, bool]:
        """Return ``(guaranteed_return, supported_structure)`` for one block."""
        for statement in statements:
            if isinstance(statement, ast.Return):
                return statement.value is not None, True
            if isinstance(statement, ast.If):
                body_returns, body_supported = statement_block_return_proof(
                    statement.body
                )
                else_returns, else_supported = statement_block_return_proof(
                    statement.orelse
                )
                if not body_supported or not else_supported:
                    return False, False
                if (
                    statement.orelse
                    and body_returns
                    and else_returns
                ):
                    return True, True
                continue
            if isinstance(statement, (ast.AnnAssign, ast.Assign, ast.Expr, ast.Pass)):
                continue
            return False, False
        return False, True

    def function_returns_math(
        function: ast.FunctionDef | ast.AsyncFunctionDef,
        call: ast.Call,
    ) -> bool:
        if id(function) in resolving_functions:
            return False
        returns: list[ast.Return] = []

        class ReturnCollector(ast.NodeVisitor):
            def visit_Return(self, node: ast.Return) -> None:
                returns.append(node)

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                if node is function:
                    self.generic_visit(node)

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
                if node is function:
                    self.generic_visit(node)

            def visit_ClassDef(self, node: ast.ClassDef) -> None:
                return

            def visit_Lambda(self, node: ast.Lambda) -> None:
                return

        ReturnCollector().visit(function)
        guaranteed_return, supported_structure = statement_block_return_proof(
            function.body
        )
        if (
            not returns
            or any(node.value is None for node in returns)
            or not supported_structure
            or not guaranteed_return
        ):
            return False
        positional_parameters = [
            *function.args.posonlyargs,
            *function.args.args,
        ]
        if (
            len(call.args) > len(positional_parameters)
            and function.args.vararg is None
        ):
            return False
        validated_parameters = {
            parameter.arg
            for parameter, _ in zip(positional_parameters, call.args, strict=False)
        }
        keyword_parameters = {
            parameter.arg
            for parameter in (
                *function.args.args,
                *function.args.kwonlyargs,
            )
        }
        for keyword in call.keywords:
            if keyword.arg is None:
                return False
            if keyword.arg in validated_parameters:
                return False
            if keyword.arg not in keyword_parameters:
                if function.args.kwarg is None:
                    return False
            else:
                validated_parameters.add(keyword.arg)
        default_scope = lexical_scope(function) or function
        positional_default_start = len(positional_parameters) - len(
            function.args.defaults
        )
        for index, parameter in enumerate(positional_parameters):
            if parameter.arg in validated_parameters:
                continue
            default_index = index - positional_default_start
            if default_index < 0 or not is_math_expression(
                function.args.defaults[default_index],
                default_scope,
            ):
                return False
            validated_parameters.add(parameter.arg)
        for parameter, default in zip(
            function.args.kwonlyargs,
            function.args.kw_defaults,
            strict=True,
        ):
            if parameter.arg in validated_parameters:
                continue
            if default is None or not is_math_expression(default, default_scope):
                return False
            validated_parameters.add(parameter.arg)
        if function.args.vararg is not None:
            validated_parameters.add(function.args.vararg.arg)
        resolving_functions.add(id(function))
        previous_parameters = validated_function_parameters.get(id(function))
        validated_function_parameters[id(function)] = frozenset(validated_parameters)
        previous_keyword_container = validated_keyword_containers.get(id(function))
        if function.args.kwarg is not None:
            validated_keyword_containers[id(function)] = function.args.kwarg.arg
        try:
            return all(
                is_math_expression(node.value, function)
                for node in returns
                if node.value is not None
            )
        finally:
            resolving_functions.remove(id(function))
            if previous_parameters is None:
                validated_function_parameters.pop(id(function), None)
            else:
                validated_function_parameters[id(function)] = previous_parameters
            if previous_keyword_container is None:
                validated_keyword_containers.pop(id(function), None)
            else:
                validated_keyword_containers[id(function)] = previous_keyword_container

    def resolved_local_function(
        name: str,
        scope: ast.AST,
        position: tuple[int, int],
    ) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
        for candidate_scope in scope_chain(scope):
            definitions = local_functions.get((id(candidate_scope), name), ())
            has_competing_binding = bool(
                name in scope_parameters.get(id(candidate_scope), {})
                or assignment_name_positions.get((id(candidate_scope), name))
                or import_bindings.get((id(candidate_scope), name))
                or local_classes.get((id(candidate_scope), name))
            )
            if definitions or has_competing_binding:
                if (
                    len(definitions) == 1
                    and type(definitions[0]) is ast.FunctionDef
                    and not has_competing_binding
                ):
                    definition = definitions[0]
                    definition_position = (
                        getattr(definition, "end_lineno", definition.lineno),
                        getattr(
                            definition,
                            "end_col_offset",
                            definition.col_offset,
                        ),
                    )
                    if candidate_scope is scope:
                        return definition if definition_position < position else None
                    caller_name = (
                        scope.name
                        if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef))
                        else None
                    )
                    caller_positions = (
                        direct_calls.get((id(candidate_scope), caller_name), ())
                        if caller_name is not None
                        else ()
                    )
                    if any(
                        caller_position < definition_position
                        for caller_position in caller_positions
                    ):
                        return None
                    if caller_name is not None and any(
                        reference_position < definition_position
                        for reference_position in indirect_function_references.get(
                            (id(candidate_scope), caller_name), ()
                        )
                    ):
                        return None
                    return definition
                return None
        return None

    external_math_functions = frozenset(
        {
            "abs",
            "complex",
            "float",
            "int",
            "len",
            "max",
            "min",
            "pow",
            "round",
            "sum",
        }
    )
    external_math_call_members = {
        "math": frozenset(
            {
                "acos",
                "asin",
                "atan",
                "ceil",
                "cos",
                "exp",
                "fabs",
                "floor",
                "isfinite",
                "log",
                "log1p",
                "pow",
                "sin",
                "sqrt",
                "tan",
            }
        ),
        "np": frozenset(
            {
                "abs",
                "array",
                "asarray",
                "clip",
                "concatenate",
                "dot",
                "matmul",
                "maximum",
                "mean",
                "minimum",
                "sqrt",
                "stack",
                "sum",
            }
        ),
        "numpy": frozenset(
            {
                "abs",
                "array",
                "asarray",
                "clip",
                "concatenate",
                "dot",
                "matmul",
                "maximum",
                "mean",
                "minimum",
                "sqrt",
                "stack",
                "sum",
            }
        ),
        "torch": frozenset(
            {
                "abs",
                "arange",
                "as_tensor",
                "cat",
                "clamp",
                "cos",
                "dot",
                "exp",
                "full",
                "linspace",
                "log",
                "matmul",
                "mean",
                "norm",
                "ones",
                "sin",
                "sqrt",
                "stack",
                "sum",
                "tensor",
                "zeros",
            }
        ),
    }
    external_math_value_members = {
        "math": frozenset({"e", "inf", "nan", "pi", "tau"}),
        "np": frozenset({"e", "float16", "float32", "float64", "inf", "nan", "pi"}),
        "numpy": frozenset({"e", "float16", "float32", "float64", "inf", "nan", "pi"}),
        "torch": frozenset(
            {"bfloat16", "float16", "float32", "float64", "inf", "nan", "pi"}
        ),
    }
    tensor_math_methods = frozenset(
        {
            "abs",
            "clamp",
            "clip",
            "cos",
            "dot",
            "double",
            "exp",
            "flatten",
            "float",
            "log",
            "matmul",
            "mean",
            "norm",
            "permute",
            "pow",
            "reshape",
            "sin",
            "sqrt",
            "square",
            "squeeze",
            "sum",
            "to",
            "transpose",
            "unsqueeze",
            "view",
        }
    )

    def scope_chain(scope: ast.AST) -> tuple[ast.AST, ...]:
        scopes = []
        current: ast.AST | None = scope
        while current is not None:
            scopes.append(current)
            current = lexical_scope(current)
        return tuple(scopes)

    def visible_name_binding(
        name: str,
        scope: ast.AST,
        position: tuple[int, int],
    ) -> tuple[str, str | None] | None:
        for scope_index, candidate_scope in enumerate(scope_chain(scope)):
            bindings: list[tuple[tuple[int, int], str, str | None]] = []
            if name in scope_parameters.get(id(candidate_scope), {}):
                bindings.append(((-1, -1), "shadow", None))
            bindings.extend(
                (event_position, "shadow", None)
                for event_position in assignment_name_positions.get(
                    (id(candidate_scope), name), []
                )
            )
            bindings.extend(
                (
                    (
                        getattr(node, "end_lineno", node.lineno),
                        getattr(node, "end_col_offset", node.col_offset),
                    ),
                    "shadow",
                    None,
                )
                for node in (
                    *local_functions.get((id(candidate_scope), name), ()),
                    *local_classes.get((id(candidate_scope), name), ()),
                )
            )
            bindings.extend(
                (event_position, "namespace" if source else "shadow", source)
                for event_position, source in import_bindings.get(
                    (id(candidate_scope), name), []
                )
            )
            if scope_index == 0:
                bindings = [
                    binding
                    for binding in bindings
                    if binding[0] < position or binding[0] == (-1, -1)
                ]
                if bindings:
                    _, kind, source = max(bindings, key=lambda binding: binding[0])
                    return kind, source
                continue
            if not bindings:
                continue
            namespace_sources = {
                source
                for _, kind, source in bindings
                if kind == "namespace" and source is not None
            }
            if len(namespace_sources) == 1 and all(
                kind == "namespace" for _, kind, _ in bindings
            ):
                return "namespace", next(iter(namespace_sources))
            return "shadow", None
        return None

    def name_is_shadowed(
        name: str,
        scope: ast.AST,
        position: tuple[int, int],
    ) -> bool:
        return visible_name_binding(name, scope, position) is not None

    def resolved_namespace(
        name: str,
        scope: ast.AST,
        position: tuple[int, int],
    ) -> str | None:
        binding = visible_name_binding(name, scope, position)
        if binding is None or binding[0] != "namespace":
            return None
        return binding[1]

    def resolved_plain_name_math(
        name: str,
        scope: ast.AST,
        position: tuple[int, int],
    ) -> bool:
        for scope_index, candidate_scope in enumerate(scope_chain(scope)):
            assignments = value_events.get((id(candidate_scope), name), [])
            if id(candidate_scope) in validated_function_parameters:
                parameter = name in validated_function_parameters[id(candidate_scope)]
            else:
                parameter = scope_parameters.get(id(candidate_scope), {}).get(name)
            has_other_binding = bool(
                local_functions.get((id(candidate_scope), name))
                or local_classes.get((id(candidate_scope), name))
                or import_bindings.get((id(candidate_scope), name))
            )
            if scope_index == 0:
                event = latest_event(assignments, position)
                if event is not None:
                    return event
                if parameter is not None:
                    return parameter and not is_identity_shaped_name(name)
                if has_other_binding:
                    return False
                continue
            source_assignments = assignment_sources.get(
                (id(candidate_scope), name), []
            )
            if source_assignments:
                caller_name = (
                    scope.name
                    if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef))
                    else None
                )
                caller_positions = (
                    direct_calls.get((id(candidate_scope), caller_name), ())
                    if caller_name is not None
                    else ()
                )
                if len(source_assignments) != 1:
                    return False
                assignment_position, assignment = source_assignments[0]
                if any(
                    caller_position < assignment_position
                    for caller_position in caller_positions
                ):
                    return False
                assignment_value = assignment.value
                assignment_math = bool(
                    assignment_value is not None
                    and is_math_expression(assignment_value, candidate_scope)
                    and (
                        not isinstance(assignment, ast.AnnAssign)
                        or is_math_annotation(assignment.annotation)
                    )
                )
                sources = [assignment_math]
            else:
                sources = []
            if parameter is not None:
                sources.append(parameter and not is_identity_shaped_name(name))
            if has_other_binding:
                return False
            if not sources:
                continue
            return len(sources) == 1 and sources[0]
        return False

    def readonly_len_parameter(
        function: ast.FunctionDef | ast.AsyncFunctionDef | None,
    ) -> str | None:
        if (
            type(function) is not ast.FunctionDef
            or function.decorator_list
            or function.args.posonlyargs
            or len(function.args.args) != 1
            or function.args.vararg is not None
            or function.args.kwonlyargs
            or function.args.kwarg is not None
            or function.args.defaults
            or len(function.body) != 1
            or not isinstance(function.body[0], ast.Return)
            or not isinstance(function.body[0].value, ast.Call)
        ):
            return None
        call = function.body[0].value
        parameter = function.args.args[0].arg
        if (
            not isinstance(call.func, ast.Name)
            or call.func.id != "len"
            or len(call.args) != 1
            or call.keywords
            or not isinstance(call.args[0], ast.Name)
            or call.args[0].id != parameter
            or name_is_shadowed(
                "len",
                function,
                (call.lineno, call.col_offset),
            )
        ):
            return None
        return parameter

    def readonly_len_argument(
        call: ast.Call,
        function: ast.FunctionDef | ast.AsyncFunctionDef | None,
    ) -> ast.expr | None:
        parameter = readonly_len_parameter(function)
        if parameter is None:
            return None
        if len(call.args) == 1 and not call.keywords:
            return call.args[0]
        if (
            not call.args
            and len(call.keywords) == 1
            and call.keywords[0].arg == parameter
        ):
            return call.keywords[0].value
        return None

    def is_math_call(call: ast.Call, scope: ast.AST) -> bool:
        position = (call.lineno, call.col_offset)
        function = (
            resolved_local_function(call.func.id, scope, position)
            if isinstance(call.func, ast.Name)
            else None
        )
        builtin_name = (
            call.func.id
            if isinstance(call.func, ast.Name)
            and function is None
            and call.func.id in external_math_functions
            and not name_is_shadowed(call.func.id, scope, position)
            else None
        )
        if builtin_name == "len" and (
            len(call.args) != 1 or call.keywords
        ):
            return False
        if builtin_name == "sum" and (
            not 1 <= len(call.args) <= 2 or call.keywords
        ):
            return False

        def is_validated_keyword_container(node: ast.expr) -> bool:
            if not isinstance(node, ast.Name):
                return False
            original_name = validated_keyword_containers.get(id(scope))
            if original_name != node.id:
                return False
            aliases = {original_name}
            changed = True
            while changed:
                changed = False
                for (scope_id, alias_name), sources in assignment_sources.items():
                    if scope_id != id(scope) or alias_name in aliases:
                        continue
                    if any(
                        source_position < (node.lineno, node.col_offset)
                        and isinstance(assignment.value, ast.Name)
                        and assignment.value.id in aliases
                        for source_position, assignment in sources
                    ):
                        aliases.add(alias_name)
                        changed = True
            if any(
                invalidation_position < (node.lineno, node.col_offset)
                for alias_name in aliases
                for invalidation_position in keyword_container_invalidations.get(
                    (id(scope), alias_name), ()
                )
            ):
                return False
            for call in ast.walk(scope):
                if not isinstance(call, ast.Call):
                    continue
                call_position = (call.lineno, call.col_offset)
                if call_position >= (node.lineno, node.col_offset):
                    continue
                passed_names = {
                    argument.id
                    for argument in call.args
                    if isinstance(argument, ast.Name)
                }
                passed_names.update(
                    keyword.value.id
                    for keyword in call.keywords
                    if isinstance(keyword.value, ast.Name)
                )
                if not (passed_names & aliases):
                    continue
                is_current_len = bool(
                    isinstance(call.func, ast.Name)
                    and call.func.id == "len"
                    and len(call.args) == 1
                    and call.args[0] is node
                    and not call.keywords
                )
                helper = (
                    resolved_local_function(
                        call.func.id,
                        scope,
                        call_position,
                    )
                    if isinstance(call.func, ast.Name)
                    else None
                )
                is_current_readonly_len_helper = (
                    readonly_len_argument(call, helper) is node
                )
                if not (is_current_len or is_current_readonly_len_helper):
                    return False
            return bool(
                not any(
                    assignment_position < (node.lineno, node.col_offset)
                    for assignment_position in assignment_name_positions.get(
                        (id(scope), node.id), ()
                    )
                )
                and not local_functions.get((id(scope), node.id))
                and not local_classes.get((id(scope), node.id))
                and not import_bindings.get((id(scope), node.id))
            )

        def is_keyword_values_extraction(node: ast.expr) -> bool:
            return bool(
                isinstance(node, ast.Call)
                and not node.args
                and not node.keywords
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "values"
                and is_validated_keyword_container(node.func.value)
            )

        readonly_argument = readonly_len_argument(call, function)
        arguments_are_math = all(
            (
                builtin_name == "len"
                and is_validated_keyword_container(argument)
            )
            or (
                readonly_argument is argument
                and is_validated_keyword_container(argument)
            )
            or (
                builtin_name == "sum"
                and is_keyword_values_extraction(argument)
            )
            or is_math_expression(argument, scope)
            for argument in call.args
        )
        if not (
            arguments_are_math
            and all(
                keyword.arg is not None
                and (
                    (
                        readonly_argument is keyword.value
                        and is_validated_keyword_container(keyword.value)
                    )
                    or (
                        not is_identity_shaped_name(keyword.arg)
                        and is_math_expression(keyword.value, scope)
                    )
                )
                for keyword in call.keywords
            )
        ):
            return False
        if isinstance(call.func, ast.Name):
            if function is not None:
                return function_returns_math(function, call)
            return builtin_name is not None
        if not isinstance(call.func, ast.Attribute):
            return False
        key = attribute_key(call.func)
        namespace = (
            resolved_namespace(key[0], scope, position)
            if key is not None
            else None
        )
        if key and namespace is not None:
            return bool(
                len(key) == 2
                and key[1] in external_math_call_members[namespace]
            )
        return (
            call.func.attr in tensor_math_methods
            and is_math_expression(call.func.value, scope)
        )

    def is_math_expression(expression: ast.expr, scope: ast.AST) -> bool:
        if isinstance(expression, ast.Constant):
            return type(expression.value) in {int, float, complex}
        if isinstance(expression, ast.Name):
            if expression.id in LOCAL_MATH_BINDING_NAMES:
                return has_prior_math_binding(expression, scope)
            return resolved_plain_name_math(
                expression.id,
                scope,
                (expression.lineno, expression.col_offset),
            )
        if isinstance(expression, ast.Attribute):
            key = attribute_key(expression)
            if key is not None:
                event = latest_event(
                    attribute_events.get((id(scope), key), []),
                    (expression.lineno, expression.col_offset),
                )
                if event is not None:
                    return event
                namespace = resolved_namespace(
                    key[0],
                    scope,
                    (expression.lineno, expression.col_offset),
                )
                if namespace is not None:
                    return bool(
                        len(key) == 2
                        and key[1] in external_math_value_members[namespace]
                    )
            return (
                expression.attr in {"dtype", "ndim", "shape"}
                and is_math_expression(expression.value, scope)
            )
        if isinstance(expression, ast.Subscript):
            return is_math_expression(expression.value, scope) and is_math_expression(
                expression.slice, scope
            )
        if isinstance(expression, ast.Slice):
            return all(
                part is None or is_math_expression(part, scope)
                for part in (expression.lower, expression.upper, expression.step)
            )
        if isinstance(expression, ast.UnaryOp):
            return is_math_expression(expression.operand, scope)
        if isinstance(expression, ast.BinOp):
            return is_math_expression(expression.left, scope) and is_math_expression(
                expression.right, scope
            )
        if isinstance(expression, ast.BoolOp):
            return all(
                is_math_expression(value, scope) for value in expression.values
            )
        if isinstance(expression, ast.Compare):
            return is_math_expression(expression.left, scope) and all(
                is_math_expression(comparator, scope)
                for comparator in expression.comparators
            )
        if isinstance(expression, ast.IfExp):
            if (
                isinstance(expression.test, ast.Constant)
                and type(expression.test.value) is bool
            ):
                selected = expression.body if expression.test.value else expression.orelse
                return is_math_expression(selected, scope)
            return is_math_expression(
                expression.body,
                scope,
            ) and is_math_expression(expression.orelse, scope)
        if isinstance(expression, (ast.Tuple, ast.List, ast.Set)):
            return all(
                is_math_expression(element, scope) for element in expression.elts
            )
        if isinstance(expression, ast.Call):
            return is_math_call(expression, scope)
        return False

    allowed_node_ids: set[int] = set()
    assignment_nodes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
    ]
    control_nodes = [
        node
        for node in ast.walk(tree)
        if isinstance(
            node,
            (ast.If, ast.Try, ast.Match, ast.For, ast.AsyncFor, ast.While),
        )
    ]
    assignment_math: dict[int, bool] = {}
    unreachable_assignments: set[int] = set()

    def mark_dead_block_assignments(statements: list[ast.stmt]) -> None:
        terminated = False
        for statement in statements:
            if terminated:
                unreachable_assignments.update(
                    id(node)
                    for node in ast.walk(statement)
                    if isinstance(node, (ast.Assign, ast.AnnAssign))
                )
                continue
            nested_blocks: list[list[ast.stmt]] = []
            if isinstance(statement, ast.If):
                nested_blocks.extend((statement.body, statement.orelse))
            elif isinstance(statement, ast.Try):
                nested_blocks.extend(
                    (
                        statement.body,
                        statement.orelse,
                        statement.finalbody,
                        *(handler.body for handler in statement.handlers),
                    )
                )
            elif isinstance(statement, ast.Match):
                nested_blocks.extend(case.body for case in statement.cases)
            elif isinstance(statement, (ast.For, ast.AsyncFor, ast.While)):
                nested_blocks.extend((statement.body, statement.orelse))
            for block in nested_blocks:
                mark_dead_block_assignments(block)
            if isinstance(
                statement,
                (ast.Return, ast.Raise, ast.Break, ast.Continue),
            ):
                terminated = True

    for scope in ast.walk(tree):
        if isinstance(scope, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef)):
            mark_dead_block_assignments(scope.body)

    def is_flow_stop(value: object) -> bool:
        return bool(
            isinstance(value, tuple)
            and len(value) == 3
            and value[0] == "flow_stop"
        )

    def flow_stop(kind: str, status: bool | None) -> tuple[str, str, bool | None]:
        return "flow_stop", kind, status

    def flow_status(value: object) -> bool | None:
        return value[2] if is_flow_stop(value) else value  # type: ignore[index,return-value]

    def merge_flow_outcomes(outcomes: list[object]) -> object:
        continuing = [outcome for outcome in outcomes if not is_flow_stop(outcome)]
        if continuing:
            if any(
                is_flow_stop(outcome)
                and outcome[1] in {"break", "continue"}  # type: ignore[index]
                for outcome in outcomes
            ):
                return False
            return bool(all(continuing))
        if not outcomes:
            return False
        stop_kinds = {outcome[1] for outcome in outcomes if is_flow_stop(outcome)}  # type: ignore[index]
        if len(stop_kinds) == 1:
            kind = next(iter(stop_kinds))
            statuses = [flow_status(outcome) for outcome in outcomes]
            return flow_stop(kind, bool(statuses and all(statuses)))
        return flow_stop("terminated", False)

    def branch_assigned_names(statements: list[ast.stmt], scope: ast.AST) -> set[str]:
        names: set[str] = set()
        for statement in statements:
            for node in ast.walk(statement):
                if isinstance(
                    node,
                    (
                        ast.FunctionDef,
                        ast.AsyncFunctionDef,
                        ast.ClassDef,
                        ast.Lambda,
                    ),
                ) and node is not statement:
                    continue
                if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                    continue
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if lexical_scope(target) is scope:
                        names.update(_python_assignment_target_names(target))
        return names

    def branch_final_status(
        statements: list[ast.stmt],
        scope: ast.AST,
        name: str,
        incoming: bool | None,
    ) -> object:
        status: object = incoming
        for statement in statements:
            if isinstance(statement, (ast.Return, ast.Raise, ast.Break, ast.Continue)):
                return flow_stop(type(statement).__name__.lower(), flow_status(status))
            if isinstance(statement, (ast.Assign, ast.AnnAssign)):
                targets = (
                    statement.targets
                    if isinstance(statement, ast.Assign)
                    else [statement.target]
                )
                if any(
                    lexical_scope(target) is scope
                    and name in _python_assignment_target_names(target)
                    for target in targets
                ):
                    status = assignment_math.get(id(statement), False)
                continue
            if isinstance(statement, ast.If):
                status = control_final_status(
                    statement,
                    scope,
                    name,
                    flow_status(status),
                )
                if is_flow_stop(status):
                    return status
                continue
            if isinstance(
                statement,
                (ast.Try, ast.Match, ast.For, ast.AsyncFor, ast.While),
            ):
                status = control_final_status(
                    statement,
                    scope,
                    name,
                    flow_status(status),
                )
                if is_flow_stop(status):
                    return status
                continue
            if isinstance(
                statement,
                (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
            ):
                continue
            if name in branch_assigned_names([statement], scope):
                status = False
        return status

    def control_final_status(
        statement: ast.If | ast.Try | ast.Match | ast.For | ast.AsyncFor | ast.While,
        scope: ast.AST,
        name: str,
        incoming: bool | None,
    ) -> object:
        if isinstance(statement, ast.If):
            if (
                isinstance(statement.test, ast.Constant)
                and type(statement.test.value) is bool
            ):
                selected = statement.body if statement.test.value else statement.orelse
                return branch_final_status(selected, scope, name, incoming)
            body_status = branch_final_status(
                statement.body,
                scope,
                name,
                incoming,
            )
            else_status = (
                branch_final_status(
                    statement.orelse,
                    scope,
                    name,
                    incoming,
                )
                if statement.orelse
                else incoming
            )
            return merge_flow_outcomes([body_status, else_status])
        if isinstance(statement, ast.Try):
            body_status = branch_final_status(
                statement.body,
                scope,
                name,
                incoming,
            )
            normal_status = (
                body_status
                if is_flow_stop(body_status)
                else branch_final_status(
                    statement.orelse,
                    scope,
                    name,
                    flow_status(body_status),
                )
            )
            outcomes: list[object] = [normal_status]
            for handler in statement.handlers:
                outcomes.extend(
                    (
                        branch_final_status(
                            handler.body,
                            scope,
                            name,
                            incoming,
                        ),
                        branch_final_status(
                            handler.body,
                            scope,
                            name,
                            flow_status(body_status),
                        ),
                    )
                )
            if statement.finalbody:
                final_outcomes: list[object] = []
                for outcome in outcomes:
                    final_status = branch_final_status(
                        statement.finalbody,
                        scope,
                        name,
                        flow_status(outcome),
                    )
                    if is_flow_stop(final_status):
                        final_outcomes.append(final_status)
                    elif is_flow_stop(outcome):
                        final_outcomes.append(
                            flow_stop(outcome[1], flow_status(final_status))  # type: ignore[index]
                        )
                    else:
                        final_outcomes.append(final_status)
                outcomes = final_outcomes
            return merge_flow_outcomes(outcomes)
        if isinstance(statement, ast.Match):
            outcomes = [
                branch_final_status(
                    case.body,
                    scope,
                    name,
                    incoming,
                )
                for case in statement.cases
                if not (
                    isinstance(case.guard, ast.Constant)
                    and type(case.guard.value) is bool
                    and not case.guard.value
                )
            ]
            last_case = statement.cases[-1] if statement.cases else None
            exhaustive = bool(
                last_case is not None
                and (
                    last_case.guard is None
                    or (
                        isinstance(last_case.guard, ast.Constant)
                        and type(last_case.guard.value) is bool
                        and last_case.guard.value
                    )
                )
                and isinstance(last_case.pattern, ast.MatchAs)
                and last_case.pattern.pattern is None
                and last_case.pattern.name is None
            )
            if not exhaustive:
                outcomes.append(incoming)
            return merge_flow_outcomes(outcomes)
        zero_iteration_status = branch_final_status(
            statement.orelse,
            scope,
            name,
            incoming,
        )
        if (
            isinstance(statement, ast.While)
            and isinstance(statement.test, ast.Constant)
            and statement.test.value is False
        ):
            return zero_iteration_status
        body_incoming = incoming
        if isinstance(statement, (ast.For, ast.AsyncFor)) and name in set(
            _python_assignment_target_names(statement.target)
        ):
            body_incoming = False
        iteration_status = branch_final_status(
            statement.body,
            scope,
            name,
            body_incoming,
        )
        if is_flow_stop(iteration_status):
            stop_kind = iteration_status[1]
            if stop_kind == "break":
                statically_nonempty = bool(
                    isinstance(statement, (ast.For, ast.AsyncFor))
                    and isinstance(statement.iter, (ast.Tuple, ast.List, ast.Set))
                    and statement.iter.elts
                )
                if statically_nonempty:
                    return flow_status(iteration_status)
                return merge_flow_outcomes(
                    [zero_iteration_status, flow_status(iteration_status)]
                )
            if stop_kind in {"return", "raise"}:
                return merge_flow_outcomes(
                    [zero_iteration_status, iteration_status]
                )
            iteration_status = flow_status(iteration_status)
        iteration_status = branch_final_status(
            statement.orelse,
            scope,
            name,
            flow_status(iteration_status),
        )
        return merge_flow_outcomes([zero_iteration_status, iteration_status])

    flow_events = [
        (
            (
                getattr(node, "end_lineno", node.lineno),
                getattr(node, "end_col_offset", node.col_offset),
            ),
            0,
            "assignment",
            node,
        )
        for node in assignment_nodes
    ]
    flow_events.extend(
        (
            (
                getattr(node, "end_lineno", node.lineno),
                getattr(node, "end_col_offset", node.col_offset),
            ),
            1,
            "control",
            node,
        )
        for node in control_nodes
    )
    for completion_position, _, event_kind, node in sorted(
        flow_events,
        key=lambda event: (event[0], event[1]),
    ):
        if event_kind == "control":
            scope = lexical_scope(node)
            if scope is None or not isinstance(
                node,
                (ast.If, ast.Try, ast.Match, ast.For, ast.AsyncFor, ast.While),
            ):
                continue
            assigned_names = branch_assigned_names(
                [node],
                scope,
            )
            for name in assigned_names:
                incoming = latest_event(
                    value_events.get((id(scope), name), []),
                    (node.lineno, node.col_offset),
                )
                control_status = control_final_status(node, scope, name, incoming)
                value_events.setdefault((id(scope), name), []).append(
                    (
                        completion_position,
                        False if is_flow_stop(control_status) else bool(control_status),
                    )
                )
            continue

        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        value = node.value
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        target_scopes = {
            lexical_scope(target)
            for target in targets
            if lexical_scope(target) is not None
        }
        is_mathematical_binding = bool(
            id(node) not in unreachable_assignments
            and len(target_scopes) == 1
            and value is not None
            and is_math_expression(value, next(iter(target_scopes)))
            and (
                not isinstance(node, ast.AnnAssign)
                or is_math_annotation(node.annotation)
            )
        )
        assignment_math[id(node)] = is_mathematical_binding
        for target in targets:
            scope = lexical_scope(target)
            if scope is None:
                continue
            if isinstance(target, ast.Name):
                value_events.setdefault((id(scope), target.id), []).append(
                    (completion_position, is_mathematical_binding)
                )
                if (
                    target.id in LOCAL_MATH_BINDING_NAMES
                    and not isinstance(scope, ast.ClassDef)
                    and is_mathematical_binding
                ):
                    allowed_node_ids.add(id(target))
            elif isinstance(target, ast.Attribute):
                key = attribute_key(target)
                if key is not None:
                    attribute_events.setdefault((id(scope), key), []).append(
                        (completion_position, is_mathematical_binding)
                    )

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
        prior_bindings = [
            event
            for event in value_events.get((id(scope), node.id), ())
            if event[0] < (node.lineno, node.col_offset)
        ]
        # A read is local only after an earlier source position completed the
        # latest mathematical binding in this same lexical scope.  A later
        # non-mathematical reassignment revokes the exception.
        if prior_bindings and max(prior_bindings, key=lambda event: event[0])[1]:
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
    attested_upstream_paths = _attested_upstream_source_paths(root_path)
    structural_upstream_paths = (
        _UPSTREAM_SOURCE_STRUCTURAL_PATHS
        if attested_upstream_paths
        else frozenset()
    )
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
        upstream_semantic_path = relative in attested_upstream_paths
        upstream_structural_path = relative in structural_upstream_paths
        if path.is_dir() and not upstream_structural_path:
            if not is_allowed_directory_name(path.name):
                violations.append({"path": str(relative), "reason": "directory_name_not_snake_case"})
        elif path.is_file() and not (
            upstream_semantic_path or upstream_structural_path
        ):
            if not is_allowed_file_name(path.name):
                violations.append({"path": str(relative), "reason": "file_name_not_snake_case"})
        path_name = path.stem if path.is_file() else path.name
        if not (upstream_semantic_path or upstream_structural_path):
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
        if path.is_file() and path.suffix == ".py" and not upstream_semantic_path:
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
