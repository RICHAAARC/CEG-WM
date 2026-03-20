"""
功能：测试 freeze surface 审计对 interpretation fail-fast 的静态识别。

Module type: General module

Verify that the freeze surface audit recognizes direct interpretation checks,
local aliases derived from interpretation, and rejects unrelated none checks.
"""

import ast
import sys
import textwrap
from pathlib import Path


_TEST_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _TEST_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.audits.audit_freeze_surface_integrity import (
    _has_interpretation_fail_fast,
    _is_interpretation_none_check,
    check_schema_authority,
)


def _parse_function(source_text: str) -> ast.FunctionDef:
    module = ast.parse(textwrap.dedent(source_text))
    function_node = module.body[0]
    assert isinstance(function_node, ast.FunctionDef)
    return function_node


def test_interpretation_none_check_accepts_direct_name():
    """
    功能：直接使用 interpretation 变量名时应识别为 fail-fast 条件。

    Verify that direct interpretation is None checks are accepted.

    Args:
        None.

    Returns:
        None.
    """
    function_node = _parse_function(
        """
        def sample(*, interpretation=None):
            if interpretation is None:
                raise RuntimeError("missing interpretation")
        """
    )
    if_node = function_node.body[0]

    assert isinstance(if_node, ast.If)
    assert _is_interpretation_none_check(if_node.test)
    assert _has_interpretation_fail_fast(function_node) is True


def test_interpretation_fail_fast_accepts_local_alias():
    """
    功能：interpretation 的局部别名应被识别为等价 fail-fast。

    Verify that a local alias derived from interpretation is recognized.

    Args:
        None.

    Returns:
        None.
    """
    function_node = _parse_function(
        """
        def sample(*, interpretation=None):
            interpretation_obj = interpretation
            if interpretation_obj is None:
                raise RuntimeError("missing interpretation")
        """
    )

    assert _has_interpretation_fail_fast(function_node) is True


def test_interpretation_fail_fast_rejects_unrelated_none_check():
    """
    功能：无关变量的 none 检查不得被误识别为 interpretation fail-fast。

    Verify that unrelated none checks are not treated as interpretation fail-fast.

    Args:
        None.

    Returns:
        None.
    """
    function_node = _parse_function(
        """
        def sample(*, interpretation=None, unrelated=None):
            other_obj = interpretation
            if unrelated is None:
                raise RuntimeError("unrelated none")
        """
    )

    assert _has_interpretation_fail_fast(function_node) is False


def test_schema_authority_accepts_current_validate_record():
    """
    功能：当前 schema.validate_record 的局部别名 fail-fast 应通过静态审计。

    Verify that the current schema module passes the schema authority audit.

    Args:
        None.

    Returns:
        None.
    """
    result = check_schema_authority(_REPO_ROOT)

    assert result["validate_record_found"] is True
    assert result["interpretation_required"] is True
    assert result["pass"] is True