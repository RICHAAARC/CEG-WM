"""提供仓库 intake 检查。"""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from governance.harness.lib.project_policy import load_root_policy


def inspect_repository(root: str | Path) -> dict:
    """返回当前仓库的基础目录状态。"""
    root_path = Path(root)
    policy = load_root_policy(root_path)
    directory_status = {
        name: {
            "exists": (root_path / name).exists(),
            "path": str(root_path / name),
            "kind": metadata["kind"],
            "audited": metadata["audited"],
        }
        for name, metadata in policy["root_registry"].items()
    }
    contract_path = root_path / ".codex" / "project_contract.md"
    return {
        "repository_mode": "governed_repository" if contract_path.exists() else "uninitialized_repository",
        "project_contract_exists": contract_path.exists(),
        "directory_status": directory_status,
        "governed_files": policy.get("governed_files", []),
    }


def main(argv: list[str] | None = None) -> None:
    """命令行入口。"""
    arguments = argv or sys.argv
    root = Path(arguments[1]) if len(arguments) > 1 else Path.cwd()
    print(json.dumps(inspect_repository(root), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv)
