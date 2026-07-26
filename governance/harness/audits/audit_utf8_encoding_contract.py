"""审计受治理文本文件是否可用 UTF-8 解码。"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from governance.harness.lib.file_scanner import iter_governed_text_files
from governance.harness.lib.json_report import build_report, exit_with_report


def run_audit(root: str | Path) -> dict:
    root_path = Path(root)
    violations = []
    checked_paths = []
    for path in iter_governed_text_files(root_path):
        checked_paths.append(str(path.relative_to(root_path)))
        try:
            path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            violations.append({"path": str(path.relative_to(root_path)), "reason": "not_utf8"})
    return build_report("audit_utf8_encoding_contract", "fail" if violations else "pass", violations, checked_paths)


def main() -> None:
    exit_with_report(run_audit(Path.cwd()))


if __name__ == "__main__":
    main()
