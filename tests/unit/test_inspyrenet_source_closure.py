"""Verify the bounded InSPyReNet source and dependency closure."""

from __future__ import annotations

import ast
from hashlib import sha256
import json
from pathlib import Path

import pytest


pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
VENDOR_ROOT = ROOT / "runtime/_vendor/transparent_background"
SOURCE_MANIFEST = VENDOR_ROOT / "SOURCE.json"
GPU_REQUIREMENTS = ROOT / "requirements_inspyrenet_salient_local_lf_gpu_execution.txt"
BASE_GPU_REQUIREMENTS = ROOT / "requirements_development_exploration_gpu_execution.txt"

EXPECTED_UPSTREAM_COMMIT = "f0fa91701a98cfc8e955c554e84522f365ec6da3"
EXPECTED_UPSTREAM_TREE = "19c4aae7fe5ca6d77ddbd8cc4a4e0be662bfcb5c"
EXPECTED_FILE_HASHES = {
    "LICENSE": (
        "a08a7c43ff8fe90648f889d4f937b178c29ab9be1f92244f685bf7f97cb53f91",
        "a08a7c43ff8fe90648f889d4f937b178c29ab9be1f92244f685bf7f97cb53f91",
    ),
    "__init__.py": (
        None,
        "01ba4719c80b6fe911b091a7c05124b64eeece964e09c058ef8f9805daca546b",
    ),
    "InSPyReNet.py": (
        "9bf8c73a361200888e48677c1df55b81bb1bdb669cfd91d73a01c01d24efbef4",
        "e2f7d66c37b778ab1fce10553604075a54d93691b4612b952d7d44a8388cf42b",
    ),
    "modules/__init__.py": (
        None,
        "01ba4719c80b6fe911b091a7c05124b64eeece964e09c058ef8f9805daca546b",
    ),
    "modules/layers.py": (
        "e57eedd05bece9f14cf6b2798e0c2ed09382e60d200ed6352895a979f80ed5e8",
        "7f5c6ad133af2234b74ff6d067e95f09022f598abc0d987b8a2d99a1044d66d7",
    ),
    "modules/context_module.py": (
        "b5b612e4d86848a3e69b66d89effcc8698e434d6f50270595605c1d42cb844d4",
        "b5b612e4d86848a3e69b66d89effcc8698e434d6f50270595605c1d42cb844d4",
    ),
    "modules/attention_module.py": (
        "7f34d941393fb9dfc69f14ff02f731e5e1487f55cde9e79a7195d328922db2fb",
        "30e05975d0e8a9ff9f3dddaf0fa278556d16d9f40b4df8f76d193f4de8c8dcae",
    ),
    "modules/decoder_module.py": (
        "a6c99bfdfed9cefd4184662b4a093d179e6a0c805d92ad21122ebaf95e05ee20",
        "1a0b8d23cace8f68ceee14f76802af8d8762ce4dff9327a97538d26b7e7f936d",
    ),
    "backbones/__init__.py": (
        None,
        "01ba4719c80b6fe911b091a7c05124b64eeece964e09c058ef8f9805daca546b",
    ),
    "backbones/SwinTransformer.py": (
        "78c53d0cbd05f9a0d3cbd1dfbf86f6b989f8708281b6915e5267b03850cd8d82",
        "6f76d560fec382c8526a7230f4bbd95d122b97bdea44de452586a79f8a5ac41d",
    ),
}
EXPECTED_TRANSFORMATIONS = {
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
TRAILING_WHITESPACE_BY_LINE = {
    "InSPyReNet.py": {
        21: b"        ",
        37: b"        ",
        39: b"        ",
        43: b"        ",
        45: b"        ",
        53: b"    ",
        57: b"            ",
        65: b"    ",
        68: b"    ",
        70: b"        ",
        87: b"        ",
        91: b"        ",
        95: b"        ",
        97: b"    ",
        100: b"        ",
        105: b"            ",
        113: b"        ",
        119: b"                ",
        124: b"            ",
        126: b" ",
        127: b"            ",
        131: b"            ",
        135: b"            ",
        139: b"            ",
        144: b"    ",
    },
    "modules/layers.py": {
        21: b"        ",
        25: b"        ",
        29: b"            ",
        66: b"        ",
        70: b"        ",
        74: b"        ",
        79: b"        ",
        116: b"        ",
        146: b"        ",
    },
    "modules/attention_module.py": {
        21: b"        ",
        40: b"        ",
        47: b"        ",
        76: b"        ",
        77: b"        ",
        80: b"        ",
        97: b"        ",
    },
    "modules/decoder_module.py": {
        14: b"        ",
        17: b"        ",
    },
    "backbones/SwinTransformer.py": {
        603: b"            ",
        629: b"        ",
        636: b"        ",
        643: b"        ",
    },
}
ADDED_REQUIREMENT_PINS = {
    "kornia==0.8.3",
    "kornia-rs==0.1.14",
    "opencv-python-headless==4.12.0.88",
    "timm==1.0.28",
    "torchvision==0.26.0+cu128",
}


def _digest(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def _requirement_lines(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text("utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _restore_normalized_source_bytes(local_path: str, payload: bytes) -> bytes:
    if local_path not in TRAILING_WHITESPACE_BY_LINE:
        return payload
    assert payload.endswith(b"\n") and not payload.endswith(b"\n\n")
    lines = payload.splitlines(keepends=True)
    assert all(line.rstrip(b"\n") == line.rstrip(b"\n").rstrip(b" \t") for line in lines)
    for line_ordinal, suffix in TRAILING_WHITESPACE_BY_LINE[local_path].items():
        line = lines[line_ordinal - 1]
        assert line.endswith(b"\n")
        lines[line_ordinal - 1] = line[:-1] + suffix + b"\n"
    restored = b"".join(lines)
    if local_path == "backbones/SwinTransformer.py":
        restored += b"\n"
    return restored


def _restore_upstream_bytes(local_path: str, payload: bytes) -> bytes:
    payload = _restore_normalized_source_bytes(local_path, payload)
    if local_path == "InSPyReNet.py":
        local_header = (
            b"import torch\n"
            b"import torch.nn as nn\n"
            b"import torch.nn.functional as F\n"
            b"import numpy as np\n\n"
            b"from .modules.layers import *\n"
            b"from .modules.context_module import *\n"
            b"from .modules.attention_module import *\n"
            b"from .modules.decoder_module import *\n\n"
            b"from .backbones.SwinTransformer import SwinB"
        )
        upstream_header = (
            b"import os\n"
            b"import sys\n"
            b"import torch\n"
            b"import torch.nn as nn\n"
            b"import torch.nn.functional as F\n"
            b"import numpy as np\n\n"
            b"filepath = os.path.abspath(__file__)\n"
            b"repopath = os.path.split(filepath)[0]\n"
            b"sys.path.append(repopath)\n\n"
            b"from transparent_background.modules.layers import *\n"
            b"from transparent_background.modules.context_module import *\n"
            b"from transparent_background.modules.attention_module import *\n"
            b"from transparent_background.modules.decoder_module import *\n\n"
            b"from transparent_background.backbones.SwinTransformer import SwinB"
        )
        assert payload.count(local_header) == 1
        assert payload.endswith(b"\n")
        return payload.replace(local_header, upstream_header)[:-1]
    if local_path == "modules/attention_module.py":
        local_import = b"from .layers import *"
        upstream_import = b"from transparent_background.modules.layers import *"
        assert payload.count(local_import) == 1
        assert payload.endswith(b"\n")
        return payload.replace(local_import, upstream_import)[:-1]
    if local_path == "modules/decoder_module.py":
        assert payload.endswith(b"\n")
        return payload[:-1]
    return payload


def test_inspyrenet_source_manifest_binds_upstream_and_local_bytes() -> None:
    manifest = json.loads(SOURCE_MANIFEST.read_text("utf-8"))

    assert manifest["source_repository"] == (
        "https://github.com/plemeri/transparent-background"
    )
    assert manifest["upstream_commit"] == EXPECTED_UPSTREAM_COMMIT
    assert manifest["upstream_tree"] == EXPECTED_UPSTREAM_TREE
    assert manifest["source_license"] == "MIT"
    assert manifest["vendored_namespace"] == (
        "runtime._vendor.transparent_background"
    )
    entries = {entry["local_path"]: entry for entry in manifest["files"]}
    assert set(entries) == set(EXPECTED_FILE_HASHES)

    tracked_paths = {
        path.relative_to(VENDOR_ROOT).as_posix()
        for path in VENDOR_ROOT.rglob("*")
        if path.is_file() and path != SOURCE_MANIFEST
    }
    assert tracked_paths == set(EXPECTED_FILE_HASHES)

    for local_path, (upstream_digest, local_digest) in EXPECTED_FILE_HASHES.items():
        entry = entries[local_path]
        payload = (VENDOR_ROOT / local_path).read_bytes()
        assert entry["upstream_sha256"] == upstream_digest
        assert entry["local_sha256"] == local_digest
        assert entry["transformations"] == EXPECTED_TRANSFORMATIONS[local_path]
        assert _digest(payload) == local_digest
        if upstream_digest is None:
            assert entry["upstream_path"] is None
            assert payload.strip() == b""
        else:
            assert _digest(_restore_upstream_bytes(local_path, payload)) == (
                upstream_digest
            )


def test_inspyrenet_vendored_import_closure_is_explicit_and_side_effect_free() -> None:
    python_paths = sorted(VENDOR_ROOT.rglob("*.py"))
    external_roots: set[str] = set()
    relative_modules: set[str] = set()
    for path in python_paths:
        tree = ast.parse(path.read_text("utf-8"), filename=path.as_posix())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                external_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    relative_modules.add(node.module or "")
                elif node.module:
                    external_roots.add(node.module.split(".", 1)[0])

    assert external_roots == {
        "cv2",
        "kornia",
        "numpy",
        "operator",
        "timm",
        "torch",
        "typing",
    }
    assert relative_modules == {
        "backbones.SwinTransformer",
        "layers",
        "modules.attention_module",
        "modules.context_module",
        "modules.decoder_module",
        "modules.layers",
    }
    source = (VENDOR_ROOT / "InSPyReNet.py").read_text("utf-8")
    assert "sys.path" not in source
    assert "filepath" not in source
    assert "repopath" not in source
    assert "transparent_background." not in source
    assert not list(VENDOR_ROOT.rglob("*.pth"))


def test_inspyrenet_gpu_requirement_lock_only_extends_registered_runtime() -> None:
    base_lines = _requirement_lines(BASE_GPU_REQUIREMENTS)
    candidate_lines = _requirement_lines(GPU_REQUIREMENTS)

    assert set(candidate_lines) - set(base_lines) == ADDED_REQUIREMENT_PINS
    assert [
        line for line in candidate_lines if line not in ADDED_REQUIREMENT_PINS
    ] == base_lines
    assert "torch==2.11.0+cu128" in candidate_lines
    assert "numpy==2.0.2" in candidate_lines
    assert "pillow==11.3.0" in candidate_lines
    assert "opencv-python" not in {line.split("==", 1)[0] for line in candidate_lines}
    assert not {
        "albumentations",
        "gdown",
        "pymatting",
        "wget",
    }.intersection(line.split("==", 1)[0] for line in candidate_lines)
