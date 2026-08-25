"""One-shot fit-first formal runner for Content V7 ordinary-score ISS."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

from experiments import run_content_adaptive_dual_branch_v2_clean as engine
from experiments import run_content_v3_clean as v3_runner
from cegwm.method.content_iss_v7 import (
    ISS_ASSET_FILENAME,
    build_iss_asset,
    derive_development_key,
    fit_iss_gain_target,
    load_iss_asset,
    stable_json_bytes,
)
from cegwm.method.hf import FrozenHFPublicAssets
from cegwm.method.lf import FrozenLFPublicAssets
from cegwm.protocol.content_chain_v2 import ContentChainProtocol
from cegwm.protocol.content_chain_v7 import (
    CONTENT_V7_ARMS,
    CONTENT_V7_EXECUTION_SCOPE_ID,
    CONTENT_V7_RECORD_CONTRACT_ID,
    CONTENT_V7_RUN_PREFIX,
    ContentV7FormalProtocol,
    load_content_v7_formal_protocol,
)
from cegwm.runtime.content_adaptive_sd35_v3 import ContentV3EmbedAssets
from cegwm.runtime.content_iss_sd35_v7 import (
    ContentV7DevelopmentAssets,
    ContentV7EvaluationAssets,
    run_content_v7_development_pair,
    run_content_v7_evaluation_pair,
)
from cegwm.shared.keys import normalize_detection_key, public_key_digest

KEY_ENV = "CEG_WM_ROOT_KEY"
TOKEN_ENV = "HF_TOKEN"
TERMINAL_FILENAME = f"{CONTENT_V7_RUN_PREFIX}.zip"
FIT_RECEIPT_PREFIX = "CEGWM_CONTENT_V7_FORMAL_PROGRESS"
SUMMARY_PREFIX = "CEGWM_CONTENT_V7_FORMAL_SUMMARY"


@dataclass(frozen=True, slots=True)
class ContentV7RunnerAssets:
    evaluation_assets: ContentV7EvaluationAssets

    @property
    def embed_assets(self) -> ContentV3EmbedAssets:
        return self.evaluation_assets.embed_assets

    @property
    def hf_public_assets(self) -> FrozenHFPublicAssets:
        return self.evaluation_assets.hf_public_assets

    @property
    def lf_public_assets(self) -> FrozenLFPublicAssets:
        return self.evaluation_assets.lf_public_assets


def _forbidden_delegate(*args: Any, **kwargs: Any) -> Any:
    del args, kwargs
    raise RuntimeError("Content V7 uses only its integrated paired runtime")


CONTENT_V7_RUNNER_VARIANT = engine.ContentRunnerVariant(
    name="Content V7",
    execution_scope_id=CONTENT_V7_EXECUTION_SCOPE_ID,
    complete_execution="complete_for_content_v7_independent_evaluation_invocation",
    arms=CONTENT_V7_ARMS,
    record_contract_id=CONTENT_V7_RECORD_CONTRACT_ID,
    state_schema_id="content_v7_no_resume_state_forbidden",
    run_prefix=CONTENT_V7_RUN_PREFIX,
    load_protocol=_forbidden_delegate,
    load_pipeline_and_assets=_forbidden_delegate,
    run_joint=_forbidden_delegate,
)


def _paths(artifact_sink: Path, exact: str) -> tuple[Path, Path, Path, Path]:
    root = artifact_sink / exact / CONTENT_V7_RUN_PREFIX
    asset = root / ISS_ASSET_FILENAME
    terminal = root / TERMINAL_FILENAME
    return (
        asset,
        root / f"{ISS_ASSET_FILENAME}.sha256",
        terminal,
        root / f"{TERMINAL_FILENAME}.sha256",
    )


def _require_create_only(paths: tuple[Path, ...]) -> None:
    if any(path.exists() for path in paths):
        raise FileExistsError("create-only Content V7 formal destination exists")


def _copy_create_only(source: Path, destination: Path) -> None:
    opened = False
    try:
        with source.open("rb") as incoming, destination.open("xb") as outgoing:
            opened = True
            shutil.copyfileobj(incoming, outgoing)
    except BaseException:
        if opened:
            destination.unlink(missing_ok=True)
        raise


def _publish_asset_pair(asset_path: Path, sidecar_path: Path, payload: bytes) -> str:
    _require_create_only((asset_path, sidecar_path))
    asset_path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(payload).hexdigest()
    with tempfile.TemporaryDirectory(prefix=".content-v7-asset-", dir=asset_path.parent) as staging:
        staged_asset = Path(staging) / asset_path.name
        staged_sidecar = Path(staging) / sidecar_path.name
        staged_asset.write_bytes(payload)
        staged_sidecar.write_bytes(f"{digest}  {asset_path.name}\n".encode("ascii"))
        created: list[Path] = []
        try:
            _copy_create_only(staged_asset, asset_path)
            created.append(asset_path)
            _copy_create_only(staged_sidecar, sidecar_path)
            created.append(sidecar_path)
        except BaseException:
            for path in reversed(created):
                path.unlink(missing_ok=True)
            raise
    return digest


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _load_pipeline_and_assets(
    model_id: str,
    token: str,
) -> tuple[Any, ContentV3EmbedAssets]:
    return v3_runner._load_pipeline_and_assets(model_id, token)


def _progress(phase: str, completed: int, total: int) -> None:
    print(
        f"{FIT_RECEIPT_PREFIX} "
        + json.dumps(
            {"phase": phase, "completed": completed, "fixed_total": total},
            separators=(",", ":"),
        ),
        flush=True,
    )


def _failure_transaction(
    *,
    unit: Any,
    run_id: str,
    exact: str,
    protocol: ContentChainProtocol,
    key_digest: str,
    error: Exception,
) -> list[dict[str, Any]]:
    error_class = engine._public_operational_error_class(error)
    return [
        engine._content_v2_record(
            run_id=run_id,
            unit_id=unit.unit_id,
            source_cluster_id=unit.source_id,
            arm=arm,
            condition="clean",
            code_revision=exact,
            config_digest=protocol.protocol_digest,
            key_public_digest=key_digest,
            status="operational_failure",
            failure_reason=error_class,
            variant=CONTENT_V7_RUNNER_VARIANT,
        )
        for arm in CONTENT_V7_ARMS
    ]


def _unit_transaction(
    *,
    unit: Any,
    pipeline: Any,
    assets: ContentV7RunnerAssets,
    key: bytes,
    wrong_keys: tuple[bytes, ...],
    run_id: str,
    exact: str,
    protocol: ContentChainProtocol,
    key_digest: str,
) -> list[dict[str, Any]]:
    output = run_content_v7_evaluation_pair(
        pipeline,
        unit.prompt,
        key,
        assets.evaluation_assets,
        height=unit.height,
        width=unit.width,
        seed=unit.seed,
    )
    joint_scores = engine._blind_scores(
        output.image,
        key,
        wrong_keys,
        assets.hf_public_assets,
        assets.lf_public_assets,
    )
    null_scores = engine._blind_scores(
        output.primary_null,
        key,
        wrong_keys,
        assets.hf_public_assets,
        assets.lf_public_assets,
    )
    metrics = engine._candidate_aggregate_metrics(
        unit.unit_id,
        output.measurement,
        engine._psnr(output.image, output.primary_null),
        share_sum_absolute_tolerance=protocol.config["aggregate_measurement"][
            "branch_share_sum_absolute_tolerance"
        ],
    )
    common = {
        "run_id": run_id,
        "unit_id": unit.unit_id,
        "source_cluster_id": unit.source_id,
        "condition": "clean",
        "code_revision": exact,
        "config_digest": protocol.protocol_digest,
        "key_public_digest": key_digest,
        "status": "success",
        "variant": CONTENT_V7_RUNNER_VARIANT,
    }
    return [
        engine._content_v2_record(
            **common,
            arm=CONTENT_V7_ARMS[0],
            scores=engine._flat_scores(joint_scores),
            metrics={
                name: float(value) for name, value in metrics.items() if name != "unit_id"
            },
        ),
        engine._content_v2_record(
            **common,
            arm=CONTENT_V7_ARMS[1],
            scores=engine._flat_scores(null_scores),
            metrics={"paired_rgb_psnr_db": float(metrics["paired_rgb_psnr_db"])},
        ),
    ]


def _evaluate_invocation(
    *,
    invocation_index: int,
    protocol: ContentChainProtocol,
    pipeline: Any,
    assets: ContentV7RunnerAssets,
    key: bytes,
    exact: str,
    key_digest: str,
) -> dict[str, Any]:
    invocation_id = protocol.protocol_id.rsplit("/", 1)[-1]
    run_id = (
        f"{CONTENT_V7_RUN_PREFIX}-{invocation_index:02d}-"
        f"{protocol.protocol_digest[:12]}-{key_digest[:12]}"
    )
    wrong_keys = engine._wrong_keys(key, protocol)
    records: list[dict[str, Any]] = []
    for unit_index, unit in enumerate(protocol.roster, 1):
        try:
            transaction = _unit_transaction(
                unit=unit,
                pipeline=pipeline,
                assets=assets,
                key=key,
                wrong_keys=wrong_keys,
                run_id=run_id,
                exact=exact,
                protocol=protocol,
                key_digest=key_digest,
            )
        except Exception as error:  # noqa: BLE001 - fixed denominator records the attempt
            transaction = _failure_transaction(
                unit=unit,
                run_id=run_id,
                exact=exact,
                protocol=protocol,
                key_digest=key_digest,
                error=error,
            )
        records.extend(transaction)
        _progress(invocation_id, unit_index, 8)
    identity = engine._public_identity(
        protocol,
        exact=exact,
        key_digest=key_digest,
        run_id=run_id,
        variant=CONTENT_V7_RUNNER_VARIANT,
    )
    result = engine._derive_result(
        records,
        protocol,
        identity,
        variant=CONTENT_V7_RUNNER_VARIANT,
    )
    return {"invocation_id": invocation_id, **result}


def _terminal_payload(
    *,
    formal: ContentV7FormalProtocol,
    exact: str,
    key_digest: str,
    asset_sha256: str,
    asset_sidecar_sha256: str,
    evaluations: list[dict[str, Any]],
) -> dict[str, Any]:
    if len(evaluations) != 2:
        raise RuntimeError("Content V7 terminal requires exactly two evaluation results")
    return {
        "exact": exact,
        "execution_scope_id": CONTENT_V7_EXECUTION_SCOPE_ID,
        "formal_protocol_id": formal.protocol_id,
        "formal_protocol_digest": formal.protocol_digest,
        "public_key_digest": key_digest,
        "runtime_asset_binding": {
            "filename": ISS_ASSET_FILENAME,
            "sha256": asset_sha256,
            "sidecar_sha256": asset_sidecar_sha256,
        },
        "evaluation_result_count": 2,
        "evaluations": evaluations,
        "independent_fixed_denominators": [8, 8],
        "independent_fixed_record_counts": [16, 16],
        "pooling_applied": False,
        "cross_cohort_conjunction_applied": False,
        "combined_result_produced": False,
        "limitations": list(formal.config["limitations"]),
    }


def _publish_terminal(
    *,
    terminal_path: Path,
    formal: ContentV7FormalProtocol,
    asset_path: Path,
    sidecar_path: Path,
    result: dict[str, Any],
) -> None:
    receipt = {
        "artifact_kind": "terminal",
        "formal_protocol_id": formal.protocol_id,
        "formal_protocol_digest": formal.protocol_digest,
        "runtime_asset_member": asset_path.name,
        "runtime_asset_sidecar_member": sidecar_path.name,
        "result_member": "result.json",
        "evaluation_result_count": 2,
        "external_validation_required": True,
    }
    terminal_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".content-v7-terminal-") as local:
        engine._publish_pair(
            local_run_root=Path(local),
            sink_run_root=terminal_path.parent,
            archive_name=terminal_path.name,
            members=(
                ("receipt.json", _json_bytes(receipt)),
                (asset_path.name, asset_path.read_bytes()),
                (sidecar_path.name, sidecar_path.read_bytes()),
                ("result.json", _json_bytes(result)),
            ),
        )


def execute(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve()
    artifact_sink = Path(args.artifact_sink).resolve()
    exact = engine._git_exact(repo_root, args.expected_exact)
    formal = load_content_v7_formal_protocol(repo_root)
    asset_path, sidecar_path, terminal_path, terminal_sidecar = _paths(
        artifact_sink, exact
    )
    _require_create_only((asset_path, sidecar_path, terminal_path, terminal_sidecar))

    root_key_text = os.environ.pop(KEY_ENV, "")
    token = os.environ.pop(TOKEN_ENV, "")
    if not root_key_text.strip():
        token = ""
        raise RuntimeError("CEG_WM_ROOT_KEY_is_required_for_Content_V7_formal")
    key = normalize_detection_key(root_key_text)
    development_key = derive_development_key(root_key_text)
    root_key_text = ""
    if not token.strip():
        key = b""
        development_key = b""
        raise RuntimeError("HF_TOKEN_is_required_for_Content_V7_formal")
    try:
        pipeline, embed_assets = _load_pipeline_and_assets(
            formal.config["generation_runtime"]["model_id"], token
        )
    finally:
        token = ""

    development_assets = ContentV7DevelopmentAssets(
        embed_assets, embed_assets.lf_public_assets
    )
    measurements = []
    for unit_index, unit in enumerate(formal.data.development, 1):
        measurements.append(
            run_content_v7_development_pair(
                pipeline, unit, development_key, development_assets
            )
        )
        _progress("fit", unit_index, 32)
    if len(measurements) != 32:
        raise RuntimeError("Content V7 fit did not complete exactly 32 units")
    fit = fit_iss_gain_target(measurements)
    measurements.clear()
    asset = build_iss_asset(exact, development_key, fit)
    development_key = b""
    asset_sha256 = _publish_asset_pair(asset_path, sidecar_path, asset.json_bytes)
    published_asset = load_iss_asset(asset_path, sidecar_path)
    asset_sidecar_sha256 = hashlib.sha256(sidecar_path.read_bytes()).hexdigest()
    _progress("asset_published", 1, 1)

    evaluation_assets = ContentV7RunnerAssets(ContentV7EvaluationAssets(
        embed_assets, embed_assets.lf_public_assets, published_asset
    ))
    key_digest = public_key_digest(key)
    evaluations = [
        _evaluate_invocation(
            invocation_index=index,
            protocol=protocol,
            pipeline=pipeline,
            assets=evaluation_assets,
            key=key,
            exact=exact,
            key_digest=key_digest,
        )
        for index, protocol in enumerate(formal.evaluations, 1)
    ]
    key = b""
    result = _terminal_payload(
        formal=formal,
        exact=exact,
        key_digest=key_digest,
        asset_sha256=asset_sha256,
        asset_sidecar_sha256=asset_sidecar_sha256,
        evaluations=evaluations,
    )
    _publish_terminal(
        terminal_path=terminal_path,
        formal=formal,
        asset_path=asset_path,
        sidecar_path=sidecar_path,
        result=result,
    )
    print(
        f"{SUMMARY_PREFIX} "
        + stable_json_bytes({
            "asset_sha256": asset_sha256,
            "evaluation_result_count": 2,
            "evaluation_rcs": [int(item["rc"]) for item in evaluations],
            "formal_protocol_digest": formal.protocol_digest,
            "terminal_published": True,
        }).decode("ascii"),
        flush=True,
    )
    return 0


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--expected-exact", required=True)
    parser.add_argument("--artifact-sink", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(execute(_arguments()))
