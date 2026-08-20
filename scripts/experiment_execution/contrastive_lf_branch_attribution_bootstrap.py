"""Git-less authenticated Stage-A bootstrap derived only from its own path."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Sequence


EMBEDDED_MANIFEST = "contrastive_lf_branch_attribution_package_manifest.json"


class ContrastiveLfBootstrapError(RuntimeError):
    pass


def _authenticate(
    root: Path,
    *,
    expected_revision: str,
    expected_package_identity: str,
    expected_embedded_manifest_sha256: str,
) -> dict[str, object]:
    manifest_path = root / EMBEDDED_MANIFEST
    blob = manifest_path.read_bytes()
    if sha256(blob).hexdigest() != expected_embedded_manifest_sha256:
        raise ContrastiveLfBootstrapError("embedded manifest digest mismatch")
    manifest = json.loads(blob)
    if (
        manifest.get("package_ready") is not True
        or manifest.get("source_revision") != expected_revision
        or manifest.get("package_identity") != expected_package_identity
        or manifest.get("stage_a_actions") != ["null_fit", "candidate_selection"]
    ):
        raise ContrastiveLfBootstrapError("package authority mismatch")
    declared = {item["path"]: item for item in manifest.get("copied_files", [])}
    expected = set(declared) | {EMBEDDED_MANIFEST}
    observed = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix not in {".pyc", ".pyo"}
    }
    if observed != expected:
        raise ContrastiveLfBootstrapError("package persistent member set mismatch")
    for relative, item in declared.items():
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise ContrastiveLfBootstrapError("package member type mismatch")
        member = path.read_bytes()
        if len(member) != item["size_bytes"] or sha256(member).hexdigest() != item["sha256"]:
            raise ContrastiveLfBootstrapError("package member digest mismatch")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-package-identity", required=True)
    parser.add_argument("--expected-embedded-manifest-sha256", required=True)
    parser.add_argument("--authenticate-only", action="store_true")
    parser.add_argument("--new-run-id")
    parser.add_argument("--session-id")
    parser.add_argument("--runs-root")
    parser.add_argument("--package-sha256")
    arguments = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    try:
        manifest = _authenticate(
            root,
            expected_revision=arguments.expected_revision,
            expected_package_identity=arguments.expected_package_identity,
            expected_embedded_manifest_sha256=arguments.expected_embedded_manifest_sha256,
        )
    except Exception as exc:
        print(json.dumps({"failure_reason": type(exc).__name__, "science_started": False, "scientific_unit_count": 0, "stage": "package_authentication", "status": "blocked"}, sort_keys=True))
        return 2
    if arguments.authenticate_only:
        print(json.dumps({"package_identity": manifest["package_identity"], "science_started": False, "scientific_unit_count": 0, "status": "authenticated"}, sort_keys=True))
        return 0
    if not all(
        (
            arguments.new_run_id,
            arguments.session_id,
            arguments.runs_root,
            arguments.package_sha256,
        )
    ):
        parser.error(
            "--new-run-id, --session-id, --runs-root and --package-sha256 are required"
        )
    sys.dont_write_bytecode = True
    root_text = str(root)
    if not sys.path or sys.path[0] != root_text:
        sys.path.insert(0, root_text)
    from scripts.experiment_execution.contrastive_lf_branch_attribution_entrypoint import main as entrypoint_main
    return entrypoint_main(
        (
            "--execute",
            "--observed-repository-revision",
            arguments.expected_revision,
            "--new-run-id",
            arguments.new_run_id,
            "--session-id",
            arguments.session_id,
            "--runs-root",
            arguments.runs_root,
            "--package-sha256",
            arguments.package_sha256,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
