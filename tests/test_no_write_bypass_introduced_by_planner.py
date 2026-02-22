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

FORBIDDEN_NETWORK_PATTERNS = [
    "requests.",
    "httpx.",
    "urllib.",
    "urllib3.",
    "socket.",
    "hf_hub_download",
    "snapshot_download"
]

FORBIDDEN_DYNAMIC_EXECUTION_PATTERNS = [
    "eval(",
    "exec(",
    "compile("
]

FORBIDDEN_UNSAFE_DESERIALIZATION_PATTERNS = [
    "pickle.load(",
    "pickle.loads(",
    "dill.load(",
    "dill.loads("
]


def test_no_write_bypass_introduced_by_planner() -> None:
    planner_path = Path("main/watermarking/content_chain/subspace/subspace_planner_impl.py")
    source = planner_path.read_text(encoding="utf-8")

    hits = [pattern for pattern in FORBIDDEN_WRITE_PATTERNS if pattern in source]
    assert hits == [], f"planner file contains potential write bypass patterns: {hits}"


def test_no_network_access_introduced_by_planner() -> None:
    planner_path = Path("main/watermarking/content_chain/subspace/subspace_planner_impl.py")
    source = planner_path.read_text(encoding="utf-8")

    hits = [pattern for pattern in FORBIDDEN_NETWORK_PATTERNS if pattern in source]
    assert hits == [], f"planner file contains potential network access patterns: {hits}"


def test_no_dynamic_execution_introduced_by_planner() -> None:
    planner_path = Path("main/watermarking/content_chain/subspace/subspace_planner_impl.py")
    source = planner_path.read_text(encoding="utf-8")

    hits = [pattern for pattern in FORBIDDEN_DYNAMIC_EXECUTION_PATTERNS if pattern in source]
    assert hits == [], f"planner file contains dynamic execution patterns: {hits}"


def test_no_unsafe_deserialization_introduced_by_planner() -> None:
    planner_path = Path("main/watermarking/content_chain/subspace/subspace_planner_impl.py")
    source = planner_path.read_text(encoding="utf-8")

    hits = [pattern for pattern in FORBIDDEN_UNSAFE_DESERIALIZATION_PATTERNS if pattern in source]
    assert hits == [], f"planner file contains unsafe deserialization patterns: {hits}"
