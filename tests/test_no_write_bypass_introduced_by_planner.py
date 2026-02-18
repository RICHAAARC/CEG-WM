"""
File purpose: Static guard test for planner write bypass.
Module type: General module
"""

from pathlib import Path


FORBIDDEN_WRITE_PATTERNS = [
    "open(",
    ".write_text(",
    ".write_bytes(",
    "json.dump(",
    "yaml.dump(",
    "pickle.dump(",
    "Path.write_text(",
    "Path.write_bytes("
]


def test_no_write_bypass_introduced_by_planner() -> None:
    planner_path = Path("main/watermarking/content_chain/subspace/placeholder_planner.py")
    source = planner_path.read_text(encoding="utf-8")

    hits = [pattern for pattern in FORBIDDEN_WRITE_PATTERNS if pattern in source]
    assert hits == [], f"planner file contains potential write bypass patterns: {hits}"
