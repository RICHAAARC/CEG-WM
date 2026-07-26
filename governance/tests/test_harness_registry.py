"""验证完整 harness 不会遗漏已存在的审计模块。"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from governance.harness.run_all_audits import AUDIT_MODULE_NAMES


@pytest.mark.constraint
def test_all_audit_modules_are_registered_and_importable() -> None:
    audit_root = Path("governance/harness/audits")
    discovered_modules = {
        f"governance.harness.audits.{path.stem}"
        for path in audit_root.glob("audit_*.py")
    }
    registered_modules = set(AUDIT_MODULE_NAMES)

    assert len(AUDIT_MODULE_NAMES) == len(registered_modules)
    assert registered_modules == discovered_modules
    for module_name in sorted(registered_modules):
        module = importlib.import_module(module_name)
        assert callable(getattr(module, "run_audit", None))
