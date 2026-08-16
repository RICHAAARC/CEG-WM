"""Git-less and persistence tests for semantic-texture Phase A delivery."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
from typing import Mapping
import zipfile

import pytest

from scripts.experiment_execution import (
    build_semantic_texture_operational_preflight_package as builder,
)
from scripts.experiment_execution import (
    semantic_texture_operational_preflight_bootstrap as bootstrap,
)
from scripts.experiment_execution import (
    semantic_texture_operational_preflight_server as server,
)


pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _committed_repository(tmp_path: Path) -> tuple[Path, str]:
    repository = tmp_path / "repository"
    repository.mkdir()
    for relative in sorted(builder.EXACT_SOURCE_FILES):
        source = ROOT / relative
        target = repository / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    _git(repository, "init", "--quiet")
    _git(repository, "config", "user.name", "CEG-WM Phase A Test")
    _git(repository, "config", "user.email", "phase-a@example.invalid")
    _git(repository, "add", ".")
    _git(repository, "commit", "--quiet", "-m", "phase-a fixture")
    revision = _git(repository, "rev-parse", "HEAD")
    assert len(revision) == 40
    assert _git(repository, "status", "--porcelain=v1") == ""
    return repository, revision


def _built_package(tmp_path: Path) -> dict[str, object]:
    repository, revision = _committed_repository(tmp_path)
    output = tmp_path / "delivery" / "semantic-texture-phase-a.zip"
    result = builder.build_semantic_texture_operational_preflight_package(
        repository_root=repository,
        source_revision=revision,
        output=output,
    )
    return {
        "build": result,
        "manifest": Path(result["delivery_manifest_path"]),
        "output": output,
        "repository": repository,
        "revision": revision,
    }


def test_semantic_texture_preflight_package_is_exact_gitless_and_excludes_outer_layers(
    tmp_path: Path,
) -> None:
    fixture = _built_package(tmp_path)
    with zipfile.ZipFile(fixture["output"]) as archive:
        names = set(archive.namelist())
        embedded = json.loads(archive.read(builder.EMBEDDED_MANIFEST_PATH))
    assert names == {
        *(builder.SOURCE_TO_ARCHIVE_PATH.get(path, path) for path in builder.EXACT_SOURCE_FILES),
        builder.EMBEDDED_MANIFEST_PATH,
    }
    assert builder.ENTRYPOINT_PATH in names
    assert builder.SERVER_PATH in names
    assert "scripts/experiment_execution/build_semantic_texture_operational_preflight_package.py" not in names
    assert "scripts/experiment_execution/semantic_texture_operational_preflight_bootstrap.py" not in names
    assert not any(
        forbidden in PurePosixPath(name).parts
        for name in names
        for forbidden in (
            ".agents",
            ".codex",
            "governance",
            "notebooks",
            "outputs",
            "tests",
        )
    )
    assert embedded["package_ready"] is True
    extract_root = tmp_path / "gitless-package"
    code, result = bootstrap.run_semantic_texture_operational_preflight_bootstrap(
        archive=fixture["output"],
        manifest=fixture["manifest"],
        expected_sha256=fixture["build"]["archive_sha256"],
        expected_size=fixture["build"]["archive_size_bytes"],
        extract_root=extract_root,
        entrypoint_args=("--help",),
        environment={
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
        },
    )
    assert code == 0
    assert result["status"] == "passed"
    assert not (extract_root / ".git").exists()


def test_semantic_texture_preflight_package_rebuild_is_deterministic(
    tmp_path: Path,
) -> None:
    fixture = _built_package(tmp_path)
    second = tmp_path / "second" / "semantic-texture-phase-a.zip"
    rebuilt = builder.build_semantic_texture_operational_preflight_package(
        repository_root=fixture["repository"],
        source_revision=fixture["revision"],
        output=second,
    )
    assert second.read_bytes() == Path(fixture["output"]).read_bytes()
    assert rebuilt["archive_sha256"] == fixture["build"]["archive_sha256"]
    assert Path(rebuilt["delivery_manifest_path"]).read_bytes() == Path(
        fixture["manifest"]
    ).read_bytes()


def test_semantic_texture_preflight_bootstrap_persists_result_before_nonzero(
    tmp_path: Path,
) -> None:
    fixture = _built_package(tmp_path)
    extract_root = tmp_path / "blocked-gitless-package"
    code, result = bootstrap.run_semantic_texture_operational_preflight_bootstrap(
        archive=fixture["output"],
        manifest=fixture["manifest"],
        expected_sha256=fixture["build"]["archive_sha256"],
        expected_size=fixture["build"]["archive_size_bytes"],
        extract_root=extract_root,
        entrypoint_args=("--unsupported-phase-a-argument",),
        environment={
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
        },
    )
    result_path = extract_root.with_name(
        extract_root.name + ".bootstrap-result.json"
    )
    assert code != 0
    assert result["status"] == "blocked"
    assert result_path.is_file()
    assert json.loads(result_path.read_text(encoding="utf-8"))["status"] == "blocked"


@dataclass(frozen=True)
class _BlockedResult:
    value: Mapping[str, object]

    def as_dict(self) -> dict[str, object]:
        return dict(self.value)


class _DeliveredBlocked(RuntimeError):
    pass


def test_semantic_texture_preflight_server_finalizes_result_zip_receipt_before_raise(
    tmp_path: Path,
) -> None:
    run_id = "semantic-texture-phase-a"
    value = {
        "aggregate": None,
        "blocked_class": "identity_blocked",
        "candidate_promoted": False,
        "configuration_digest": "1" * 64,
        "formal_tau_created": False,
        "package_identity": "2" * 64,
        "profile_id": "semantic_texture_operational_preflight",
        "result_identity": "3" * 64,
        "run_id": run_id,
        "science_started": False,
        "scientific_claims_supported": False,
        "scientific_unit_count": 0,
        "source_revision": "4" * 40,
        "status": "blocked",
        "unit_outcomes": [
            {"started": True, "status": "passed"},
            {
                "blocked_class": "identity_blocked",
                "started": True,
                "status": "blocked",
            },
        ],
    }
    output_root = tmp_path / "delivery"
    receipt: dict[str, object] = {}
    with pytest.raises(_DeliveredBlocked):
        exit_code, receipt = server.finalize_semantic_texture_operational_preflight_delivery(
            _BlockedResult(value),
            output_root=output_root,
            diagnostics={"diagnostics/asset_authority.json": {"status": "identity_blocked"}},
        )
        assert exit_code != 0
        raise _DeliveredBlocked("raise only after complete delivery")
    result_path = output_root / server.RESULT_FILENAME
    archive_path = output_root / receipt["archive_filename"]
    receipt_path = output_root / server.RECEIPT_FILENAME
    assert result_path.is_file() and archive_path.is_file() and receipt_path.is_file()
    assert receipt["archive_sha256"] == sha256(archive_path.read_bytes()).hexdigest()
    with zipfile.ZipFile(archive_path) as archive:
        assert server.RESULT_FILENAME in archive.namelist()
        assert server.RECEIPT_FILENAME not in archive.namelist()
        assert "diagnostics/asset_authority.json" in archive.namelist()
    persisted_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert persisted_receipt["archive_sha256"] == receipt["archive_sha256"]
    assert not any(
        str(tmp_path) in json.dumps(document)
        for document in (
            json.loads(result_path.read_text(encoding="utf-8")),
            persisted_receipt,
        )
    )
