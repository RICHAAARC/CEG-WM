"""审计 13 项方法职责的候选绑定、架构接线和独立语义复核记录。

本审计中的 AST 检查只是必要的结构门，不能单独证明实现不是代理算法，
也不能替代独立方法语义复核、runtime 验证或科学效果证据。
"""

from __future__ import annotations

import ast
import copy
import hashlib
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from governance.harness.lib.json_report import build_report, exit_with_report
from governance.harness.lib.naming_rules import has_weak_semantic_token
from governance.harness.lib.project_policy import load_json_compatible_yaml


PROJECT_STAGE_PATTERN = re.compile(
    r"`project_stage`\s*:\s*`(?P<stage>[a-z][a-z0-9_]*)`"
)
REQUIRED_MANIFEST_FIELDS = (
    "method_name",
    "design_path",
    "candidate_specification_path",
    "candidate_specification_sha256",
    "components",
    "test_paths",
    "behavioral_checks",
    "independent_semantic_review",
)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
REVISION_PATTERN = re.compile(r"[0-9a-f]{40,64}")
HISTORICAL_HF_NAME_TOKENS = ("direct" + "_hf", "direct" + "hf")


def _is_within(relative: Path, root: Path) -> bool:
    return relative == root or root in relative.parents


def _module_name(relative: Path) -> str:
    return ".".join(relative.with_suffix("").parts)


def _git(
    root: Path,
    *arguments: str,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )


def _git_blob(
    root: Path,
    revision: str,
    relative: Path,
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "cat-file",
            "blob",
            f"{revision}:{relative.as_posix()}",
        ],
        check=False,
        capture_output=True,
    )


def _current_identity_uses_historical_hf_name(policy: dict) -> list[str]:
    current_identity_fields = {
        "required_method_invariants": policy.get("required_method_invariants", []),
        "required_method_components": policy.get("required_method_components", []),
        "required_component_responsibilities": policy.get(
            "required_component_responsibilities",
            {},
        ),
        "required_component_paths": policy.get("required_component_paths", {}),
        "required_component_candidate_ids": policy.get(
            "required_component_candidate_ids",
            {},
        ),
        "required_behavioral_checks": policy.get("required_behavioral_checks", []),
        "required_behavior_component_bindings": policy.get(
            "required_behavior_component_bindings",
            {},
        ),
    }
    return sorted(
        field
        for field, value in current_identity_fields.items()
        if any(token in str(value).lower() for token in HISTORICAL_HF_NAME_TOKENS)
    )


def _top_level_functions(
    path: Path,
) -> tuple[ast.Module | None, dict[str, ast.FunctionDef | ast.AsyncFunctionDef]]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, SyntaxError):
        return None, {}
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    return tree, functions


def _function_has_input_dependent_result(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> bool:
    parameter_names = {
        argument.arg
        for argument in (
            list(node.args.posonlyargs)
            + list(node.args.args)
            + list(node.args.kwonlyargs)
        )
        if argument.arg not in {"self", "cls"}
    }
    if node.args.vararg is not None:
        parameter_names.add(node.args.vararg.arg)
    if node.args.kwarg is not None:
        parameter_names.add(node.args.kwarg.arg)
    if not parameter_names:
        return False

    assignments: list[tuple[set[str], ast.AST]] = []
    for child in ast.walk(node):
        if isinstance(child, ast.Assign):
            targets = {
                target.id
                for target in child.targets
                if isinstance(target, ast.Name)
            }
            assignments.append((targets, child.value))
        elif isinstance(child, ast.AnnAssign) and child.value is not None:
            targets = {child.target.id} if isinstance(child.target, ast.Name) else set()
            assignments.append((targets, child.value))

    dependent_names = set(parameter_names)
    changed = True
    while changed:
        changed = False
        for targets, value in assignments:
            if not targets or not any(
                isinstance(child, ast.Name) and child.id in dependent_names
                for child in ast.walk(value)
            ):
                continue
            additions = targets - dependent_names
            if additions:
                dependent_names.update(additions)
                changed = True

    meaningful_nodes = [
        child
        for child in node.body
        if not isinstance(child, ast.Pass)
        and not (
            isinstance(child, ast.Expr)
            and isinstance(child.value, ast.Constant)
            and isinstance(child.value.value, str)
        )
        and not (
            isinstance(child, ast.Raise)
            and isinstance(child.exc, ast.Call)
            and isinstance(child.exc.func, ast.Name)
            and child.exc.func.id == "NotImplementedError"
        )
    ]
    if not meaningful_nodes:
        return False

    for child in ast.walk(node):
        if not isinstance(child, ast.Return) or child.value is None:
            continue
        if any(
            isinstance(value, ast.Name) and value.id in dependent_names
            for value in ast.walk(child.value)
        ):
            return True
    return False


def _function_is_alias_only(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> bool:
    """Reject a responsibility symbol that only forwards to another symbol/value."""
    statements = [
        child
        for child in node.body
        if not (
            isinstance(child, ast.Expr)
            and isinstance(child.value, ast.Constant)
            and isinstance(child.value.value, str)
        )
    ]
    if len(statements) != 1 or not isinstance(statements[0], ast.Return):
        return False
    value = statements[0].value
    return isinstance(value, (ast.Call, ast.Name, ast.Attribute))


def _validate_policy_component_surface(policy: dict) -> list[dict]:
    """Validate the policy's component/path/responsibility/binding tables as one surface."""
    violations: list[dict] = []
    components = policy.get("required_method_components")
    if not isinstance(components, list):
        return [{"reason": "method_component_policy_surface_invalid"}]

    declared_count = policy.get("required_method_component_count")
    if (
        declared_count != 13
        or len(components) != declared_count
        or len(set(components)) != declared_count
    ):
        violations.append(
            {
                "reason": "method_component_policy_count_mismatch",
                "expected_count": 13,
                "declared_count": declared_count,
                "listed_count": len(components),
            }
        )

    component_set = set(components)
    for field in (
        "required_component_responsibilities",
        "required_component_paths",
        "required_component_candidate_ids",
    ):
        value = policy.get(field)
        if not isinstance(value, dict) or set(value) != component_set:
            violations.append(
                {
                    "reason": "method_component_policy_surface_mismatch",
                    "field": field,
                }
            )

    paths = policy.get("required_component_paths")
    if isinstance(paths, dict) and len(set(paths.values())) != len(paths):
        violations.append(
            {
                "reason": "method_component_policy_path_reused",
            }
        )

    checks = policy.get("required_behavioral_checks")
    bindings = policy.get("required_behavior_component_bindings")
    if (
        not isinstance(checks, list)
        or len(checks) != len(set(checks))
        or not isinstance(bindings, dict)
        or set(bindings) != set(checks)
    ):
        violations.append(
            {
                "reason": "method_behavior_policy_surface_mismatch",
            }
        )
    elif any(
        not isinstance(names, list)
        or not names
        or not set(names).issubset(component_set)
        for names in bindings.values()
    ):
        violations.append(
            {
                "reason": "method_behavior_policy_component_unknown",
            }
        )
    return violations


def _test_functions(
    path: Path,
) -> tuple[ast.Module | None, dict[str, ast.FunctionDef | ast.AsyncFunctionDef]]:
    tree, functions = _top_level_functions(path)
    return (
        tree,
        {
            name: node
            for name, node in functions.items()
            if name.startswith("test_")
        },
    )


def _has_default_marker(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    for decorator in node.decorator_list:
        if (
            not isinstance(decorator, ast.Attribute)
            or decorator.attr not in {"unit", "quick"}
        ):
            continue
        mark = decorator.value
        if (
            isinstance(mark, ast.Attribute)
            and mark.attr == "mark"
            and isinstance(mark.value, ast.Name)
            and mark.value.id == "pytest"
        ):
            return True
    return False


def _registered_symbol_bindings(
    tree: ast.Module,
    registered_symbols: set[str],
) -> tuple[dict[str, str], dict[str, str]]:
    direct: dict[str, str] = {}
    modules: dict[str, str] = {}
    registered_modules = {
        symbol.rsplit(".", 1)[0]
        for symbol in registered_symbols
    }
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in registered_modules:
                    modules[alias.asname or alias.name.split(".", 1)[0]] = alias.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                imported = f"{node.module}.{alias.name}"
                if imported in registered_symbols:
                    direct[alias.asname or alias.name] = imported
    return direct, modules


def _called_symbol(
    call: ast.Call,
    direct_bindings: dict[str, str],
    module_bindings: dict[str, str],
) -> str | None:
    if isinstance(call.func, ast.Name):
        return direct_bindings.get(call.func.id)
    if (
        isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id in module_bindings
    ):
        return f"{module_bindings[call.func.value.id]}.{call.func.attr}"
    return None


def _called_registered_symbols(
    node: ast.AST,
    direct_bindings: dict[str, str],
    module_bindings: dict[str, str],
) -> set[str]:
    return {
        symbol
        for child in ast.walk(node)
        if isinstance(child, ast.Call)
        for symbol in [
            _called_symbol(child, direct_bindings, module_bindings)
        ]
        if symbol is not None
    }


def _asserted_registered_symbols(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    direct_bindings: dict[str, str],
    module_bindings: dict[str, str],
) -> set[str]:
    assignments: list[tuple[set[str], ast.AST]] = []
    for child in ast.walk(node):
        if isinstance(child, ast.Assign):
            targets = {
                target.id
                for target in child.targets
                if isinstance(target, ast.Name)
            }
            assignments.append((targets, child.value))
        elif isinstance(child, ast.AnnAssign) and child.value is not None:
            targets = {child.target.id} if isinstance(child.target, ast.Name) else set()
            assignments.append((targets, child.value))

    result_dependencies: dict[str, set[str]] = {}
    changed = True
    while changed:
        changed = False
        for targets, value in assignments:
            dependencies = _called_registered_symbols(
                value,
                direct_bindings,
                module_bindings,
            )
            for child in ast.walk(value):
                if isinstance(child, ast.Name):
                    dependencies.update(result_dependencies.get(child.id, set()))
            for target in targets:
                previous = result_dependencies.get(target, set())
                updated = previous | dependencies
                if updated != previous:
                    result_dependencies[target] = updated
                    changed = True

    asserted: set[str] = set()
    for assertion in ast.walk(node):
        if not isinstance(assertion, ast.Assert):
            continue
        if isinstance(assertion.test, ast.Constant):
            continue
        asserted.update(
            _called_registered_symbols(
                assertion.test,
                direct_bindings,
                module_bindings,
            )
        )
        for child in ast.walk(assertion.test):
            if isinstance(child, ast.Name):
                asserted.update(result_dependencies.get(child.id, set()))
    return asserted


class _TestShapeNormalizer(ast.NodeTransformer):
    def visit_Name(self, node: ast.Name) -> ast.AST:
        return ast.copy_location(ast.Name(id="_name", ctx=node.ctx), node)

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        normalized = self.generic_visit(node)
        assert isinstance(normalized, ast.Attribute)
        normalized.attr = "_attribute"
        return normalized

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        return ast.copy_location(
            ast.Constant(value=type(node.value).__name__),
            node,
        )


def _normalized_test_shape(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> str:
    body = copy.deepcopy(node.body)
    normalized = _TestShapeNormalizer().visit(ast.Module(body=body, type_ignores=[]))
    ast.fix_missing_locations(normalized)
    return ast.dump(normalized, include_attributes=False)


def run_audit(root: str | Path) -> dict:
    root_path = Path(root)
    policy_path = (
        root_path / "governance" / "policies" / "method_readiness_rules.yaml"
    )
    contract_path = root_path / ".codex" / "project_contract.md"
    checked_paths = [
        str(policy_path.relative_to(root_path)),
        str(contract_path.relative_to(root_path)),
    ]
    violations: list[dict] = []

    try:
        policy = load_json_compatible_yaml(policy_path)
        contract_text = contract_path.read_text(encoding="utf-8")
    except (OSError, ValueError, UnicodeError) as error:
        violations.append(
            {
                "path": str(policy_path.relative_to(root_path)),
                "reason": "method_readiness_authority_unreadable",
                "detail": str(error),
            }
        )
        return build_report(
            "audit_method_readiness",
            "fail",
            violations,
            checked_paths,
        )

    stage_match = PROJECT_STAGE_PATTERN.search(contract_text)
    if not stage_match:
        violations.append(
            {
                "path": str(contract_path.relative_to(root_path)),
                "reason": "project_stage_missing",
            }
        )
        return build_report(
            "audit_method_readiness",
            "fail",
            violations,
            checked_paths,
        )
    project_stage = stage_match.group("stage")
    if project_stage not in set(policy["stage_order"]):
        violations.append(
            {
                "path": str(contract_path.relative_to(root_path)),
                "reason": "project_stage_not_registered",
                "stage": project_stage,
            }
        )
        return build_report(
            "audit_method_readiness",
            "fail",
            violations,
            checked_paths,
        )
    historical_name_fields = _current_identity_uses_historical_hf_name(policy)
    if historical_name_fields:
        violations.append(
            {
                "path": str(policy_path.relative_to(root_path)),
                "reason": "historical_hf_name_used_for_current_method_identity",
                "fields": historical_name_fields,
            }
        )
    for policy_violation in _validate_policy_component_surface(policy):
        violations.append(
            {
                "path": str(policy_path.relative_to(root_path)),
                **policy_violation,
            }
        )
    if project_stage not in set(policy["method_readiness_stages"]):
        return build_report(
            "audit_method_readiness",
            "fail" if violations else "pass",
            violations,
            checked_paths,
        )

    manifest_path = root_path / policy["manifest_path"]
    manifest_relative = str(manifest_path.relative_to(root_path))
    checked_paths.append(manifest_relative)
    try:
        manifest = load_json_compatible_yaml(manifest_path)
    except (OSError, ValueError, UnicodeError) as error:
        violations.append(
            {
                "path": manifest_relative,
                "reason": "method_readiness_manifest_unreadable",
                "detail": str(error),
            }
        )
        return build_report(
            "audit_method_readiness",
            "fail",
            violations,
            checked_paths,
        )

    missing_fields = [
        field
        for field in REQUIRED_MANIFEST_FIELDS
        if field not in manifest
    ]
    if missing_fields:
        violations.append(
            {
                "path": manifest_relative,
                "reason": "method_readiness_field_missing",
                "fields": missing_fields,
            }
        )
        return build_report(
            "audit_method_readiness",
            "fail",
            violations,
            checked_paths,
        )

    method_name = str(manifest["method_name"])
    if not method_name or has_weak_semantic_token(method_name):
        violations.append(
            {
                "path": manifest_relative,
                "reason": "method_name_not_semantic",
            }
        )

    candidate_relative = Path(str(manifest["candidate_specification_path"]))
    expected_candidate_relative = Path(policy["candidate_specification_path"])
    candidate_path = root_path / candidate_relative
    checked_paths.append(candidate_relative.as_posix())
    candidate_digest = str(manifest["candidate_specification_sha256"])
    if (
        candidate_relative != expected_candidate_relative
        or not candidate_path.is_file()
    ):
        violations.append(
            {
                "path": candidate_relative.as_posix(),
                "reason": "method_candidate_specification_path_mismatch",
                "expected_path": expected_candidate_relative.as_posix(),
            }
        )
    elif not SHA256_PATTERN.fullmatch(candidate_digest):
        violations.append(
            {
                "path": candidate_relative.as_posix(),
                "reason": "method_candidate_specification_digest_mismatch",
            }
        )
    else:
        candidate_text = candidate_path.read_text(encoding="utf-8")
        manifest_components = manifest.get("components")
        readiness_candidate_ids = {
            str(candidate_id)
            for component in (
                manifest_components.values()
                if isinstance(manifest_components, dict)
                else ()
            )
            if isinstance(component, dict)
            for candidate_id in (
                component.get("candidate_ids")
                if isinstance(component.get("candidate_ids"), list)
                else ()
            )
        }
        expected_candidate_ids = sorted(
            {
                candidate_id
                for values in policy["required_component_candidate_ids"].values()
                for candidate_id in values
            }
            | readiness_candidate_ids
        )
        missing_candidate_ids = [
            candidate_id
            for candidate_id in expected_candidate_ids
            if f"`{candidate_id}`" not in candidate_text
        ]
        if missing_candidate_ids:
            violations.append(
                {
                    "path": candidate_relative.as_posix(),
                    "reason": "method_candidate_specification_id_missing",
                    "candidate_ids": missing_candidate_ids,
                }
            )

    design_relative = Path(str(manifest["design_path"]))
    design_path = root_path / design_relative
    checked_paths.append(design_relative.as_posix())
    if (
        not _is_within(design_relative, Path(policy["design_root"]))
        or design_path.name == "README.md"
        or not design_path.is_file()
    ):
        violations.append(
            {
                "path": design_relative.as_posix(),
                "reason": "method_design_evidence_invalid",
            }
        )

    components = (
        manifest["components"]
        if isinstance(manifest["components"], dict)
        else {}
    )
    required_components = set(policy["required_method_components"])
    required_responsibilities = policy["required_component_responsibilities"]
    required_paths = policy["required_component_paths"]
    required_candidate_ids = policy["required_component_candidate_ids"]
    missing_components = sorted(required_components - set(components))
    if missing_components:
        violations.append(
            {
                "path": manifest_relative,
                "reason": "method_component_missing",
                "components": missing_components,
            }
        )
    unexpected_components = sorted(set(components) - required_components)
    if unexpected_components:
        violations.append(
            {
                "path": manifest_relative,
                "reason": "method_component_unexpected",
                "components": unexpected_components,
            }
        )

    registered_components: dict[str, dict[str, str]] = {}
    claimed_symbols: dict[str, str] = {}
    for component_name in sorted(required_components & set(components)):
        component = components[component_name]
        if not isinstance(component, dict):
            violations.append(
                {
                    "path": manifest_relative,
                    "reason": "method_component_definition_invalid",
                    "component": component_name,
                }
            )
            continue
        responsibility = component.get("responsibility")
        if responsibility != required_responsibilities.get(component_name):
            violations.append(
                {
                    "path": manifest_relative,
                    "reason": "method_component_responsibility_mismatch",
                    "component": component_name,
                }
            )
        candidate_values = component.get("candidate_ids")
        declared_candidate_ids = (
            [str(value) for value in candidate_values]
            if isinstance(candidate_values, list)
            else []
        )
        expected_candidate_ids = required_candidate_ids.get(component_name, [])
        if declared_candidate_ids != expected_candidate_ids:
            violations.append(
                {
                    "path": manifest_relative,
                    "reason": "method_component_candidate_binding_mismatch",
                    "component": component_name,
                    "expected_candidate_ids": expected_candidate_ids,
                    "declared_candidate_ids": declared_candidate_ids,
                }
            )
        relative_value = component.get("implementation_path")
        symbol_value = component.get("implementation_symbol")
        if (
            not isinstance(relative_value, str)
            or not relative_value
            or not isinstance(symbol_value, str)
            or not symbol_value.isidentifier()
            or has_weak_semantic_token(symbol_value)
        ):
            violations.append(
                {
                    "path": manifest_relative,
                    "reason": "method_component_symbol_binding_invalid",
                    "component": component_name,
                }
            )
            continue

        relative = Path(relative_value)
        expected_relative = Path(required_paths[component_name])
        if relative != expected_relative:
            violations.append(
                {
                    "path": relative.as_posix(),
                    "reason": "method_component_implementation_path_mismatch",
                    "component": component_name,
                    "expected_path": expected_relative.as_posix(),
                }
            )
            continue
        path = root_path / relative
        checked_paths.append(relative.as_posix())
        if (
            not _is_within(relative, Path(policy["implementation_root"]))
            or path.suffix != ".py"
            or not path.is_file()
        ):
            violations.append(
                {
                    "path": relative.as_posix(),
                    "reason": "method_component_implementation_path_invalid",
                    "component": component_name,
                }
            )
            continue
        _, functions = _top_level_functions(path)
        function = functions.get(symbol_value)
        if function is None:
            violations.append(
                {
                    "path": relative.as_posix(),
                    "reason": "method_component_implementation_symbol_missing",
                    "component": component_name,
                    "symbol": symbol_value,
                }
            )
            continue
        if not _function_has_input_dependent_result(function):
            violations.append(
                {
                    "path": relative.as_posix(),
                    "reason": "method_component_implementation_input_independent",
                    "component": component_name,
                    "symbol": symbol_value,
                }
            )
        if _function_is_alias_only(function):
            violations.append(
                {
                    "path": relative.as_posix(),
                    "reason": "method_component_implementation_alias_only",
                    "component": component_name,
                    "symbol": symbol_value,
                }
            )

        full_symbol = f"{_module_name(relative)}.{symbol_value}"
        previous_component = claimed_symbols.get(full_symbol)
        if previous_component is not None:
            violations.append(
                {
                    "path": manifest_relative,
                    "reason": "method_component_implementation_symbol_reused",
                    "component": component_name,
                    "other_component": previous_component,
                    "symbol": full_symbol,
                }
            )
        else:
            claimed_symbols[full_symbol] = component_name
        registered_components[component_name] = {
            "full_symbol": full_symbol,
            "implementation_path": relative.as_posix(),
            "implementation_symbol": symbol_value,
        }

    test_roots = tuple(Path(value) for value in policy["test_roots"])
    test_values = manifest["test_paths"]
    test_paths = (
        [Path(str(value)) for value in test_values]
        if isinstance(test_values, list)
        else []
    )
    test_function_index: dict[
        Path,
        tuple[
            ast.Module,
            dict[str, ast.FunctionDef | ast.AsyncFunctionDef],
        ],
    ] = {}
    if not test_paths:
        violations.append(
            {
                "path": manifest_relative,
                "reason": "method_test_missing",
            }
        )
    for relative in test_paths:
        path = root_path / relative
        checked_paths.append(relative.as_posix())
        if (
            not any(_is_within(relative, root) for root in test_roots)
            or not path.is_file()
        ):
            violations.append(
                {
                    "path": relative.as_posix(),
                    "reason": "method_test_path_invalid",
                }
            )
            continue
        tree, functions = _test_functions(path)
        if tree is None or not functions:
            violations.append(
                {
                    "path": relative.as_posix(),
                    "reason": "method_test_not_collectable",
                }
            )
        else:
            test_function_index[relative] = (tree, functions)

    declared_checks = (
        manifest["behavioral_checks"]
        if isinstance(manifest["behavioral_checks"], dict)
        else {}
    )
    required_checks = tuple(policy["required_behavioral_checks"])
    expected_bindings = policy["required_behavior_component_bindings"]
    missing_checks = sorted(set(required_checks) - set(declared_checks))
    if missing_checks:
        violations.append(
            {
                "path": manifest_relative,
                "reason": "method_behavioral_check_missing",
                "checks": missing_checks,
            }
        )
    unexpected_checks = sorted(set(declared_checks) - set(required_checks))
    if unexpected_checks:
        violations.append(
            {
                "path": manifest_relative,
                "reason": "method_behavioral_check_unexpected",
                "checks": unexpected_checks,
            }
        )

    registered_symbols = {
        component["full_symbol"]
        for component in registered_components.values()
    }
    declared_nodes: set[str] = set()
    structural_shapes: dict[str, str] = {}
    for check_name in required_checks:
        check = declared_checks.get(check_name)
        if not isinstance(check, dict):
            violations.append(
                {
                    "path": manifest_relative,
                    "reason": "method_behavioral_check_definition_invalid",
                    "check": check_name,
                }
            )
            continue
        node_id = check.get("test_node")
        component_values = check.get("components")
        declared_component_names = (
            set(str(value) for value in component_values)
            if isinstance(component_values, list)
            else set()
        )
        expected_component_names = set(expected_bindings.get(check_name, []))
        if declared_component_names != expected_component_names:
            violations.append(
                {
                    "path": manifest_relative,
                    "reason": "method_behavioral_component_binding_mismatch",
                    "check": check_name,
                    "expected_components": sorted(expected_component_names),
                    "declared_components": sorted(declared_component_names),
                }
            )
        if not isinstance(node_id, str) or "::" not in node_id:
            violations.append(
                {
                    "path": manifest_relative,
                    "reason": "method_behavioral_test_node_invalid",
                    "check": check_name,
                }
            )
            continue
        relative_text, function_name = node_id.rsplit("::", 1)
        relative = Path(relative_text)
        if node_id in declared_nodes:
            violations.append(
                {
                    "path": manifest_relative,
                    "reason": "method_behavioral_test_node_reused",
                    "check": check_name,
                }
            )
        declared_nodes.add(node_id)
        indexed = test_function_index.get(relative)
        function = indexed[1].get(function_name) if indexed else None
        if (
            relative not in test_paths
            or function is None
            or function_name != f"test_{check_name}"
        ):
            violations.append(
                {
                    "path": node_id,
                    "reason": "method_behavioral_test_not_bound",
                    "check": check_name,
                }
            )
            continue

        if not _has_default_marker(function):
            violations.append(
                {
                    "path": node_id,
                    "reason": "method_behavioral_test_not_in_default_suite",
                    "check": check_name,
                }
            )

        tree = indexed[0]
        direct_bindings, module_bindings = _registered_symbol_bindings(
            tree,
            registered_symbols,
        )
        expected_symbols = {
            registered_components[name]["full_symbol"]
            for name in expected_component_names
            if name in registered_components
        }
        called_symbols = _called_registered_symbols(
            function,
            direct_bindings,
            module_bindings,
        )
        missing_calls = sorted(expected_symbols - called_symbols)
        if missing_calls:
            violations.append(
                {
                    "path": node_id,
                    "reason": "method_behavioral_test_does_not_call_component_symbols",
                    "check": check_name,
                    "symbols": missing_calls,
                }
            )
        asserted_symbols = _asserted_registered_symbols(
            function,
            direct_bindings,
            module_bindings,
        )
        missing_assertions = sorted(expected_symbols - asserted_symbols)
        if missing_assertions:
            violations.append(
                {
                    "path": node_id,
                    "reason": "method_behavioral_test_assertion_not_data_dependent",
                    "check": check_name,
                    "symbols": missing_assertions,
                }
            )

        shape = _normalized_test_shape(function)
        previous_check = structural_shapes.get(shape)
        if previous_check is not None:
            violations.append(
                {
                    "path": node_id,
                    "reason": "method_behavioral_test_structure_reused",
                    "check": check_name,
                    "other_check": previous_check,
                }
            )
        else:
            structural_shapes[shape] = check_name

    semantic_review = manifest["independent_semantic_review"]
    if not isinstance(semantic_review, dict):
        violations.append(
            {
                "path": manifest_relative,
                "reason": "method_independent_semantic_review_invalid",
            }
        )
    else:
        review_decision = semantic_review.get("decision")
        review_reference = semantic_review.get("review_reference")
        reviewed_revision = semantic_review.get("reviewed_repository_revision")
        reviewed_candidate_digest = semantic_review.get(
            "candidate_specification_sha256"
        )
        if (
            policy.get("independent_semantic_review_required") is not True
            or review_decision != "approve"
            or not isinstance(review_reference, str)
            or len(review_reference.strip()) < 8
            or not isinstance(reviewed_revision, str)
            or not REVISION_PATTERN.fullmatch(reviewed_revision)
            or reviewed_candidate_digest != candidate_digest
        ):
            violations.append(
                {
                    "path": manifest_relative,
                    "reason": "method_independent_semantic_review_invalid",
                }
            )
        else:
            repository_check = _git(
                root_path,
                "rev-parse",
                "--is-inside-work-tree",
            )
            revision_check = _git(
                root_path,
                "rev-parse",
                "--verify",
                f"{reviewed_revision}^{{commit}}",
            )
            if (
                repository_check.returncode != 0
                or repository_check.stdout.strip() != "true"
                or revision_check.returncode != 0
            ):
                violations.append(
                    {
                        "path": manifest_relative,
                        "reason": "method_independent_review_revision_unverifiable",
                    }
                )
            else:
                reviewed_candidate_blob = _git_blob(
                    root_path,
                    reviewed_revision,
                    candidate_relative,
                )
                if reviewed_candidate_blob.returncode != 0:
                    violations.append(
                        {
                            "path": manifest_relative,
                            "reason": "method_independent_review_revision_unverifiable",
                        }
                    )
                elif (
                    hashlib.sha256(reviewed_candidate_blob.stdout).hexdigest()
                    != candidate_digest
                ):
                    violations.append(
                        {
                            "path": candidate_relative.as_posix(),
                            "reason": "method_candidate_specification_digest_mismatch",
                        }
                    )
                protected_paths = sorted(
                    {
                        *[
                            component["implementation_path"]
                            for component in registered_components.values()
                        ],
                        *[relative.as_posix() for relative in test_paths],
                    }
                )
                changed_after_review = _git(
                    root_path,
                    "diff",
                    "--name-only",
                    f"{reviewed_revision}..HEAD",
                    "--",
                    *protected_paths,
                )
                uncommitted_protected = _git(
                    root_path,
                    "status",
                    "--porcelain",
                    "--",
                    *protected_paths,
                )
                if (
                    changed_after_review.returncode != 0
                    or changed_after_review.stdout.strip()
                    or uncommitted_protected.returncode != 0
                    or uncommitted_protected.stdout.strip()
                ):
                    violations.append(
                        {
                            "path": manifest_relative,
                            "reason": "method_independent_review_binding_stale",
                        }
                    )

    return build_report(
        "audit_method_readiness",
        "fail" if violations else "pass",
        violations,
        checked_paths,
    )


def main() -> None:
    exit_with_report(run_audit(Path.cwd()))


if __name__ == "__main__":
    main()
