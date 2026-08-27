"""Isolated adapter for one immutable Content source checkout.

The coordinator invokes this file by absolute path with the child working
directory set to the detached source checkout.  This module performs runtime
delegation and emits bounded public events; it does not compute T0 statistics.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Mapping

EVENT_PREFIX = "CEGWM_TEXTURE_EVENT"
PUBLIC_FAILURES = {"FileNotFoundError", "ImportError", "MemoryError", "OSError", "OutOfMemoryError", "RuntimeError", "TimeoutError", "TypeError", "ValueError"}
_ACTIVE_BINDINGS: Mapping[str, Any] = {}


def _stable(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _event(value: Mapping[str, Any]) -> None:
    print(f"{EVENT_PREFIX} {_stable(dict(value))}", flush=True)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _identity(root: Path, expected: str) -> None:
    if root.resolve() != Path.cwd().resolve():
        raise RuntimeError("adapter cwd must be the detached source checkout")
    exact = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True).stdout.strip()
    tree = subprocess.run(["git", "rev-parse", "HEAD^{tree}"], cwd=root, check=True, capture_output=True, text=True).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain"], cwd=root, check=True, capture_output=True, text=True).stdout
    if exact != expected or status:
        raise RuntimeError("adapter source checkout identity differs")
    for relative in ("src/cegwm/__init__.py", "experiments/__init__.py"):
        if not (root / relative).is_file():
            raise RuntimeError("adapter source module root differs")
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(root))
    _event({"event": "source_validated", "exact": exact, "tree": tree})


def _modules_inside(root: Path, *modules: Any) -> None:
    resolved = root.resolve()
    for module in modules:
        path = Path(getattr(module, "__file__", "")).resolve()
        if resolved not in path.parents:
            raise RuntimeError("imported production module escaped source checkout")


def _json(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _image_hash(image: Any) -> str:
    if getattr(image, "mode", None) != "RGB" or getattr(image, "size", None) != (512, 512):
        raise ValueError("method output must be ordinary RGB 512x512")
    raw = image.tobytes("raw", "RGB")
    if len(raw) != 512 * 512 * 3:
        raise ValueError("method RGB byte count differs")
    return _sha(raw)


def _write_plain(image: Any, output: Path, ordinal: int, unit_id: str) -> tuple[str, str, str]:
    raw = image.tobytes("raw", "RGB")
    if getattr(image, "mode", None) != "RGB" or getattr(image, "size", None) != (512, 512) or len(raw) != 512 * 512 * 3:
        raise ValueError("plain image must be RGB 512x512")
    ppm = b"P6\n512 512\n255\n" + raw
    relative = f"plain_rgb/{ordinal:02d}-{unit_id}.ppm"
    path = output / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(ppm)
    return relative, _sha(ppm), _sha(raw)


def _write_transient_candidate(image: Any, root: Path, ordinal: int) -> tuple[str, str, str]:
    """Write a coordinator-owned, non-artifact C3 PPM binding."""
    raw = image.tobytes("raw", "RGB")
    if getattr(image, "mode", None) != "RGB" or getattr(image, "size", None) != (512, 512) or len(raw) != 512 * 512 * 3:
        raise ValueError("candidate image must be RGB 512x512")
    relative = f"c3/{ordinal:03d}.ppm"
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    ppm = b"P6\n512 512\n255\n" + raw
    with path.open("xb") as handle:
        handle.write(ppm)
    return relative, _sha(ppm), _sha(raw)


def _generator(seed: int) -> Any:
    import torch
    return torch.Generator(device="cuda").manual_seed(seed)


def _secrets(require_key: bool = True) -> tuple[str, str]:
    root_key = os.environ.pop("CEG_WM_ROOT_KEY", "")
    token = os.environ.pop("HF_TOKEN", "")
    if require_key and not root_key.strip():
        token = ""
        raise RuntimeError("CEG_WM_ROOT_KEY is required")
    if not token.strip():
        root_key = ""
        raise RuntimeError("HF_TOKEN is required")
    return root_key, token


def _failure_class(error: Exception) -> str:
    name = type(error).__name__
    return name if name in PUBLIC_FAILURES else "OtherOperationalError"


def _cache_observation(root: Path) -> dict[str, Any]:
    """Return cache metadata only; cache contents never qualify execution."""

    try:
        return {"status": "available", "file_count": sum(1 for item in root.rglob("*") if item.is_file()), "record_only": True}
    except Exception as error:
        return {"status": "unavailable", "failure_class": _failure_class(error), "record_only": True}


def _hf_home_binding(expected: object) -> tuple[Path | None, dict[str, Any]]:
    try:
        actual = Path(os.environ.get("HF_HOME", "")).resolve()
        expected_path = Path(expected).resolve()
    except Exception as error:
        return None, {"status": "unavailable", "failure_class": _failure_class(error), "record_only": True}
    return actual, {"status": "matched" if expected_path == actual else "mismatched", "record_only": True}


def _verify_protocol(method: str, protocol: Any, binding: Mapping[str, Any]) -> None:
    expected = binding.get("source_bindings", {}).get(method)
    if not isinstance(expected, dict) or getattr(protocol, "protocol_id", None) != expected.get("protocol_id") or getattr(protocol, "protocol_digest", None) != expected.get("protocol_digest"):
        raise RuntimeError("source protocol identity differs")


def _load_v2(root: Path, token: str) -> tuple[Any, Any, Any]:
    from experiments import run_content_adaptive_dual_branch_v2_clean as engine
    _modules_inside(root, engine)
    protocol = engine._load_protocol(root)
    pipeline, assets = engine._load_pipeline_and_assets("stabilityai/stable-diffusion-3.5-medium", token)
    return engine, protocol, (pipeline, assets)


def _prefetch(root: Path, token: str, output: Path, expected_cache: Path) -> None:
    del root, token
    cache, hf_home_binding = _hf_home_binding(expected_cache)
    def version(name: str) -> tuple[str, str | None]:
        try:
            return importlib.metadata.version(name), None
        except Exception as error:
            return "unavailable", _failure_class(error)
    try:
        import torch
        cuda = str(torch.version.cuda)
        gpu = str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else "unavailable"
        environment_failure: str | None = None
    except Exception as error:
        cuda = gpu = "unavailable"
        environment_failure = _failure_class(error)
    versions = {name: version(name) for name in ("torch", "torchvision", "transformers")}
    if environment_failure is None:
        environment_failure = next((failure for _, failure in versions.values() if failure is not None), None)
    environment_status = "unavailable" if environment_failure is not None else "available"
    environment = {"status": environment_status, "python": sys.version.split()[0], "torch": versions["torch"][0], "torchvision": versions["torchvision"][0], "transformers": versions["transformers"][0], "cuda": cuda, "gpu": gpu, "record_only": True}
    if environment_failure is not None:
        environment["failure_class"] = environment_failure
    binding = {"hf_home": str(cache) if cache is not None else "", "hf_home_binding": hf_home_binding, "cache_observation": _cache_observation(cache) if cache is not None else dict(hf_home_binding), "environment_record": environment}
    path = output / "model_bindings.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write((_stable(binding) + "\n").encode("ascii"))
    _event({"event": "asset_prefetch", "hf_home_status": hf_home_binding["status"], "cache_status": binding["cache_observation"]["status"], "environment_status": environment_status})


def _unit_failure(method: str, unit: Mapping[str, Any], error: Exception, *, event_name: str = "unit") -> None:
    name = type(error).__name__
    _event({"event": event_name, "method": method, "global_ordinal": unit["global_ordinal"], "unit_id": unit["unit_id"], "status": "operational_failure", "failure_class": name if name in PUBLIC_FAILURES else "OtherOperationalError"})


def _emit_success(method: str, unit: Mapping[str, Any], image: Any, null: Any, scores: Mapping[str, float], null_scores: Mapping[str, float] | None, *, transient_candidate: tuple[str, str, str] | None = None) -> None:
    labels = ("registered", *(f"wrong_{index:02d}" for index in range(16)))
    domains = ("v4_lf", "hf") if method == "c6" else ("ordinary_lf", "hf")
    branch = {"ordinary_lf": "lf", "v4_lf": "lf", "hf": "hf"}
    def mapped(value: Mapping[str, float]) -> dict[str, dict[str, float]]:
        return {domain: {label: value[f"{branch[domain]}__{label}"] for label in labels} for domain in domains}
    event = {"event": "unit", "method": method, "global_ordinal": unit["global_ordinal"], "unit_id": unit["unit_id"], "status": "success", "candidate_rgb_sha256": _image_hash(image), "primary_null_rgb_sha256": _image_hash(null), "candidate_scores": mapped(scores), "null_scores": mapped(null_scores) if null_scores is not None else {}}
    if transient_candidate is not None:
        _relative, ppm_sha, rgb_sha = transient_candidate
        if rgb_sha != event["candidate_rgb_sha256"]:
            raise RuntimeError("transient candidate RGB binding differs")
        # Paths stay only in the coordinator's private binding file.
        event.update({"candidate_ppm_sha256": ppm_sha, "candidate_raw_rgb_sha256": rgb_sha})
    _event(event)


def _common_plain(root: Path, units: list[dict[str, Any]], token: str, output: Path) -> None:
    from cegwm.runtime.diffusers_sd35 import run_sd35_plain
    import torch
    import diffusers
    if not torch.cuda.is_available():
        raise RuntimeError("cuda_required_for_plain_generation")
    pipeline_class = getattr(diffusers, "StableDiffusion3Pipeline", None)
    if pipeline_class is None or not callable(getattr(pipeline_class, "from_pretrained", None)):
        raise RuntimeError("installed diffusers lacks StableDiffusion3Pipeline")
    pipeline = pipeline_class.from_pretrained("stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.float16, token=token)
    if not callable(pipeline):
        raise TypeError("SD3.5 pipeline must be callable")
    pipeline.to("cuda")
    for unit in units:
        try:
            image = run_sd35_plain(pipeline, unit["prompt"], height=512, width=512, generator=_generator(unit["seed"]))
            relative, ppm_sha, rgb_sha = _write_plain(image, output, unit["global_ordinal"], unit["unit_id"])
            _event({"event": "plain", "global_ordinal": unit["global_ordinal"], "unit_id": unit["unit_id"], "status": "success", "relative_path": relative, "plain_ppm_sha256": ppm_sha, "plain_rgb_sha256": rgb_sha})
        except Exception as error:  # fixed denominator
            name = type(error).__name__
            _event({"event": "plain", "global_ordinal": unit["global_ordinal"], "unit_id": unit["unit_id"], "status": "operational_failure", "failure_class": name if name in PUBLIC_FAILURES else "OtherOperationalError"})


def _v234(root: Path, method: str, units: list[dict[str, Any]], key_text: str, token: str, output_root: Path | None = None, *, event_method: str | None = None, transient_root: Path | None = None) -> None:
    from cegwm.shared.keys import normalize_detection_key
    if method == "v2":
        from experiments import run_content_adaptive_dual_branch_v2_clean as runner
        engine = runner
        _modules_inside(root, runner)
        protocol = runner._load_protocol(root)
        _verify_protocol("v2", protocol, _ACTIVE_BINDINGS)
        pipeline, assets = runner._load_pipeline_and_assets("stabilityai/stable-diffusion-3.5-medium", token)
        run_joint = runner.run_sd35_content_adaptive
        scorer = lambda image, key, wrong: engine._blind_scores(image, key, wrong, assets.hf_public_assets, assets.lf_public_assets)
    elif method == "v3":
        from experiments import run_content_v3_clean as runner
        from experiments import run_content_adaptive_dual_branch_v2_clean as engine
        _modules_inside(root, runner, engine)
        protocol = runner._load_protocol(root)
        _verify_protocol("v3", protocol, _ACTIVE_BINDINGS)
        pipeline, assets = runner._load_pipeline_and_assets("stabilityai/stable-diffusion-3.5-medium", token)
        run_joint = runner.CONTENT_V3_RUNNER_VARIANT.run_joint
        scorer = lambda image, key, wrong: engine._blind_scores(image, key, wrong, assets.hf_public_assets, assets.lf_public_assets)
    else:
        from experiments import run_content_v4_clean as runner
        from experiments import run_content_adaptive_dual_branch_v2_clean as engine
        _modules_inside(root, runner, engine)
        protocol = runner._load_protocol(root)
        _verify_protocol("v4", protocol, _ACTIVE_BINDINGS)
        pipeline, assets = runner._load_pipeline_and_assets("stabilityai/stable-diffusion-3.5-medium", token)
        run_joint = runner._run_joint
        scorer = lambda image, key, wrong: engine._blind_scores_with_lf_scorer(image, key, wrong, assets.hf_public_assets, assets.lf_public_assets, runner.score_content_v4_lf_image)
    key = normalize_detection_key(key_text)
    wrong = engine._wrong_keys(key, protocol)
    plain_bindings = {item["global_ordinal"]: item for item in _json(str(output_root / "plain_bindings.json")) if item.get("status") == "success"} if output_root is not None else {}
    for unit in units:
        try:
            output = run_joint(pipeline, unit["prompt"], key, assets, height=512, width=512, generator=_generator(unit["seed"]))
            if output_root is None:
                from cegwm.runtime.diffusers_sd35 import run_sd35_plain
                null = run_sd35_plain(pipeline, unit["prompt"], height=512, width=512, generator=_generator(unit["seed"]))
            else:
                binding = plain_bindings.get(unit["global_ordinal"])
                if not isinstance(binding, dict) or not isinstance(binding.get("relative_path"), str):
                    raise RuntimeError("common plain binding missing")
                from PIL import Image
                with Image.open(output_root / binding["relative_path"]) as opened:
                    null = opened.convert("RGB").copy()
            transient = _write_transient_candidate(output.image, transient_root, unit["global_ordinal"]) if event_method == "c3" and transient_root is not None else None
            emitted_scores = engine._flat_scores(scorer(output.image, key, wrong))
            emitted_null_scores = None if event_method == "c3" else engine._flat_scores(scorer(null, key, wrong))
            if transient is None:
                _emit_success(event_method or method, unit, output.image, null, emitted_scores, emitted_null_scores)
            else:
                _emit_success(event_method or method, unit, output.image, null, emitted_scores, emitted_null_scores, transient_candidate=transient)
        except Exception as error:
            _unit_failure(event_method or method, unit, error)
    key = b""


def _v5_validate(root: Path, bindings: Mapping[str, Any]) -> None:
    from experiments import run_content_v5_clean as v5
    _modules_inside(root, v5, v5.v4_runner)
    expected = bindings.get("v4_blobs")
    if not isinstance(expected, dict):
        raise ValueError("V5 validation requires V4 blob bindings")
    if v5.CONTENT_V5_CONTROL_RUNNER_VARIANT.load_pipeline_and_assets is not v5.v4_runner._load_pipeline_and_assets or v5.CONTENT_V5_CONTROL_RUNNER_VARIANT.run_joint is not v5.v4_runner._run_joint or v5.CONTENT_V5_CONTROL_RUNNER_VARIANT.lf_scorer is not v5.v4_runner.score_content_v4_lf_image:
        raise RuntimeError("V5 no longer delegates generation and scoring exactly to V4")
    for relative, digest in expected.items():
        if _sha((root / relative).read_bytes()) != digest:
            raise RuntimeError("V5 delegated V4 blob differs")
    _verify_protocol("v5", v5._load_protocol(root), bindings)
    _event({"event": "v5_validated", "reuse_source_method": "v4"})


def _paired_v6(root: Path, units: list[dict[str, Any]], key_text: str, token: str, output_root: Path | None = None, *, event_method: str = "v6") -> None:
    from experiments import run_content_v6_clean as runner
    from experiments import run_content_adaptive_dual_branch_v2_clean as engine
    from cegwm.shared.keys import normalize_detection_key
    _modules_inside(root, runner, engine)
    protocol = runner._load_protocol(root)
    _verify_protocol("v6", protocol, _ACTIVE_BINDINGS)
    pipeline, assets = runner._load_pipeline_and_assets("stabilityai/stable-diffusion-3.5-medium", token)
    key = normalize_detection_key(key_text)
    wrong = engine._wrong_keys(key, protocol)
    for unit in units:
        try:
            if output_root is None:
                output = runner._run_pair(pipeline, unit["prompt"], key, assets, height=512, width=512, seed=unit["seed"])
            else:
                from PIL import Image
                from cegwm.method.content_iss_v6 import content_v6_h, iss_beta
                from cegwm.runtime.content_iss_sd35_v6 import _generator, _run_content_v6_pass2
                bindings = {item["global_ordinal"]: item for item in _json(str(output_root / "plain_bindings.json")) if item.get("status") == "success"}
                binding = bindings.get(unit["global_ordinal"])
                if not isinstance(binding, dict) or not isinstance(binding.get("relative_path"), str): raise RuntimeError("common plain binding missing")
                with Image.open(output_root / binding["relative_path"]) as opened: null = opened.convert("RGB").copy()
                evaluation = assets.evaluation_assets
                beta = iss_beta(content_v6_h(null, key, evaluation.lf_public_assets), evaluation.iss_asset)
                image, _measurement = _run_content_v6_pass2(pipeline, unit["prompt"], key, evaluation, beta, height=512, width=512, generator=_generator(unit["seed"]))
                output = type("C6Output", (), {"image": image, "primary_null": null})()
            score = lambda image: engine._flat_scores(engine._blind_scores_with_lf_scorer(image, key, wrong, assets.hf_public_assets, assets.lf_public_assets, runner.score_content_v4_lf_image))
            _emit_success(event_method, unit, output.image, output.primary_null, score(output.image), score(output.primary_null) if event_method != "c6" else None)
        except Exception as error:
            _unit_failure(event_method, unit, error)
    key = b""


def _paired_v7(root: Path, units: list[dict[str, Any]], key_text: str, token: str, asset_root: Path) -> None:
    from experiments import run_content_v7_formal_initial as runner
    from experiments import run_content_adaptive_dual_branch_v2_clean as engine
    from cegwm.method.content_iss_v7 import ISS_ASSET_FILENAME, load_iss_asset
    from cegwm.runtime.content_iss_sd35_v7 import ContentV7EvaluationAssets, run_content_v7_evaluation_pair
    from cegwm.shared.keys import normalize_detection_key
    _modules_inside(root, runner, engine)
    formal = runner.load_content_v7_formal_protocol(root)
    _verify_protocol("v7", formal, _ACTIVE_BINDINGS)
    pipeline, embed = runner._load_pipeline_and_assets("stabilityai/stable-diffusion-3.5-medium", token)
    asset = load_iss_asset(asset_root / ISS_ASSET_FILENAME, asset_root / f"{ISS_ASSET_FILENAME}.sha256")
    evaluation_assets = ContentV7EvaluationAssets(embed, embed.lf_public_assets, asset)
    key = normalize_detection_key(key_text)
    for unit in units:
        protocol = formal.evaluations[0 if unit["roster_id"] == "content_v234_old" else 1]
        wrong = engine._wrong_keys(key, protocol)
        try:
            output = run_content_v7_evaluation_pair(pipeline, unit["prompt"], key, evaluation_assets, height=512, width=512, seed=unit["seed"])
            score = lambda image: engine._flat_scores(engine._blind_scores(image, key, wrong, evaluation_assets.hf_public_assets, evaluation_assets.lf_public_assets))
            _emit_success("v7", unit, output.image, output.primary_null, score(output.image), score(output.primary_null))
        except Exception as error:
            _unit_failure("v7", unit, error)
    key = b""


def _paired_v8(root: Path, units: list[dict[str, Any]], key_text: str, token: str, asset_root: Path) -> None:
    from experiments import run_content_v8_formal_initial as runner
    from cegwm.method.content_iss_v8 import ISS_ASSET_ROLE_ID, derive_wrong_keys, load_iss_asset
    from cegwm.runtime.content_iss_sd35_v8 import run_content_v8_evaluation_pair
    from cegwm.shared.keys import normalize_detection_key
    _modules_inside(root, runner)
    protocol = runner.load_content_v8_protocol(root)
    _verify_protocol("v8", protocol, _ACTIVE_BINDINGS)
    pipeline, assets = runner._load_pipeline_and_assets(token)
    asset_filename = f"{ISS_ASSET_ROLE_ID}.json"
    asset = load_iss_asset(asset_root / asset_filename, asset_root / f"{asset_filename}.sha256", expected_protocol_digest=protocol.protocol_digest)
    by_id = {unit.unit_id: unit for roster in protocol.evaluation_rosters for unit in roster.units}
    key = normalize_detection_key(key_text)
    wrong = derive_wrong_keys(key)
    for unit in units:
        try:
            output = run_content_v8_evaluation_pair(pipeline, by_id[unit["unit_id"]], key, assets, asset)
            score = lambda image: runner._blind_scores(image, key, wrong, assets)
            _emit_success("v8", unit, output.image, output.primary_null, score(output.image), score(output.primary_null))
        except Exception as error:
            _unit_failure("v8", unit, error)
    key = b""


def _safe_transient_ppm(root: Path, relative: str, ppm_sha: str, rgb_sha: str) -> Any:
    """Load one coordinator-owned P6 binding without publishing its path."""
    candidate = (root / relative).resolve()
    if root.resolve() not in candidate.parents or candidate.suffix != ".ppm" or not candidate.is_file():
        raise ValueError("transient PPM path differs")
    payload = candidate.read_bytes()
    if len(payload) != 15 + 512 * 512 * 3 or not payload.startswith(b"P6\n512 512\n255\n") or _sha(payload) != ppm_sha or _sha(payload[15:]) != rgb_sha:
        raise ValueError("transient PPM binding differs")
    from PIL import Image
    with Image.open(candidate) as opened:
        image = opened.convert("RGB").copy()
    if _image_hash(image) != rgb_sha:
        raise ValueError("transient RGB conversion differs")
    return image


def _c3_v4_lf_rescore(root: Path, units: list[dict[str, Any]], key_text: str, token: str, bindings_path: Path, transient_root: Path) -> None:
    """V4-only LF rescore of the already-produced C3 candidate and common plain."""
    from cegwm.shared.keys import normalize_detection_key
    from experiments import run_content_v4_clean as runner
    from experiments import run_content_adaptive_dual_branch_v2_clean as engine
    _modules_inside(root, runner, engine)
    protocol = runner._load_protocol(root)
    _verify_protocol("v4", protocol, _ACTIVE_BINDINGS)
    key = normalize_detection_key(key_text)
    wrong = engine._wrong_keys(key, protocol)
    if not isinstance(wrong, (tuple, list)) or len(wrong) != 16 or len(set(wrong)) != 16:
        raise RuntimeError("V4 wrong-key schedule differs")
    pipeline, assets = runner._load_pipeline_and_assets("stabilityai/stable-diffusion-3.5-medium", token)
    del pipeline  # scorer assets may load a pipeline, but this phase never invokes diffusion.
    bindings = _json(str(bindings_path))
    if not isinstance(bindings, Mapping):
        raise ValueError("transient C3 binding schema differs")
    labels = ("registered", *(f"wrong_{index:02d}" for index in range(16)))
    for unit in units:
        try:
            binding = bindings.get(str(unit["global_ordinal"]))
            if not isinstance(binding, Mapping) or set(binding) != {"candidate_relative", "candidate_ppm_sha256", "candidate_rgb_sha256", "plain_relative", "plain_ppm_sha256", "plain_rgb_sha256"}:
                raise ValueError("transient C3 unit binding differs")
            candidate = _safe_transient_ppm(transient_root, str(binding["candidate_relative"]), str(binding["candidate_ppm_sha256"]), str(binding["candidate_rgb_sha256"]))
            plain = _safe_transient_ppm(transient_root, str(binding["plain_relative"]), str(binding["plain_ppm_sha256"]), str(binding["plain_rgb_sha256"]))
            keys = (key, *wrong)
            candidate_scores = {label: float(runner.score_content_v4_lf_image(candidate, item, assets.lf_public_assets)) for label, item in zip(labels, keys)}
            null_scores = {label: float(runner.score_content_v4_lf_image(plain, item, assets.lf_public_assets)) for label, item in zip(labels, keys)}
            _event({"event": "v4_lf_rescore", "method": "c3", "global_ordinal": unit["global_ordinal"], "unit_id": unit["unit_id"], "status": "success", "candidate_rgb_sha256": binding["candidate_rgb_sha256"], "plain_rgb_sha256": binding["plain_rgb_sha256"], "candidate_ppm_sha256": binding["candidate_ppm_sha256"], "plain_ppm_sha256": binding["plain_ppm_sha256"], "candidate_scores": {"v4_lf": candidate_scores}, "null_scores": {"v4_lf": null_scores}})
        except Exception as error:
            _unit_failure("c3", unit, error, event_name="v4_lf_rescore")
    key = b""


def execute(args: argparse.Namespace) -> int:
    global _ACTIVE_BINDINGS
    root = Path(args.source_root).resolve()
    _identity(root, args.expected_exact)
    output = Path(args.local_output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    actual_hf_home, hf_home_binding = _hf_home_binding(Path(args.hf_cache_root))
    units = _json(args.units_json) if args.units_json else []
    bindings = _json(args.model_bindings_json) if args.model_bindings_json else {}
    _ACTIVE_BINDINGS = bindings
    bindings["hf_home"] = str(actual_hf_home) if actual_hf_home is not None else ""
    bindings["hf_home_binding"] = hf_home_binding
    key, token = _secrets(require_key=args.phase not in {"asset_prefetch", "common_plain_v2", "v5_validate"})
    try:
        if args.phase == "asset_prefetch":
            _prefetch(root, token, output, Path(args.hf_cache_root))
        elif args.phase == "common_plain_v2":
            _common_plain(root, units, token, output)
        elif args.phase in {"v2", "v3", "v4", "c2", "c3"}:
            _v234(root, {"c2": "v2", "c3": "v3"}.get(args.phase, args.phase), units, key, token, output, event_method=args.phase, transient_root=Path(args.transient_root).resolve() if args.transient_root else None)
        elif args.phase == "c3_v4_lf_rescore":
            if not args.transient_bindings_json or not args.transient_root:
                raise ValueError("V4 LF rescore transient binding is required")
            _c3_v4_lf_rescore(root, units, key, token, Path(args.transient_bindings_json), Path(args.transient_root).resolve())
        elif args.phase == "v5_validate":
            _v5_validate(root, bindings)
        elif args.phase in {"v6", "c6"}:
            _paired_v6(root, units, key, token, output, event_method=args.phase)
        elif args.phase == "v7":
            _paired_v7(root, units, key, token, Path(args.v7_asset_root))
        elif args.phase == "v8":
            _paired_v8(root, units, key, token, Path(args.v8_asset_root))
        else:
            raise ValueError("adapter phase differs")
    finally:
        key = token = ""
    _event({"event": "phase_complete", "phase": args.phase})
    return 0


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--expected-exact", required=True)
    parser.add_argument("--phase", required=True, choices=("asset_prefetch", "common_plain_v2", "v2", "v3", "v4", "v5_validate", "v6", "v7", "v8", "c2", "c3", "c3_v4_lf_rescore", "c6"))
    parser.add_argument("--units-json")
    parser.add_argument("--plain-bindings-json")
    parser.add_argument("--local-output-root", required=True)
    parser.add_argument("--hf-cache-root", required=True)
    parser.add_argument("--model-bindings-json")
    parser.add_argument("--v7-asset-root")
    parser.add_argument("--v8-asset-root")
    parser.add_argument("--transient-root")
    parser.add_argument("--transient-bindings-json")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        raise SystemExit(execute(_arguments()))
    except Exception as error:  # bounded child failure only
        name = type(error).__name__
        _event({"event": "fatal", "failure_class": name if name in PUBLIC_FAILURES else "OtherOperationalError"})
        raise SystemExit(2) from None
