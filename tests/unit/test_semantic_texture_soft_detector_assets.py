"""Unit coverage for Phase-B soft-detector asset protocol boundaries."""

from __future__ import annotations

import ast
from dataclasses import replace
import json
from pathlib import Path
from struct import pack, unpack
from types import SimpleNamespace

import pytest

from experiments.metrics.lf_whitened_score_screening import (
    fit_semantic_texture_lf_whitening_asset,
)
from experiments.metrics import lf_whitened_score_screening as lf_screening
from experiments.methods.ceg_wm import (
    materialize_semantic_texture_soft_detector_asset_bundle,
    serialize_semantic_texture_soft_detector_asset_bundle,
)
from experiments.protocol.semantic_texture_soft_detector_assets import (
    BRANCH_NULL_COUNT,
    BRANCH_NULL_ROLE,
    WHITENING_FIT_COUNT,
    WHITENING_FIT_ROLE,
    SemanticTextureSoftDetectorAssetProtocolError,
    SemanticTextureSoftDetectorAssetBundle,
    load_manifest,
    validate_partition_disjointness,
)
from experiments.runners import semantic_texture_soft_detector_asset_preparation as preparation
from main import NullScoreRecord, SemanticTextureBranchNullCalibration
from main import (
    LfDetectionObservation,
    SemanticTextureLfWhiteningAsset,
    SemanticTextureRoutingObservations,
    SpatialRoutingObservation,
    lf_carrier,
    semantic_texture_content_router,
    semantic_texture_lf_detector,
)
from scripts.experiment_execution import (
    semantic_texture_soft_detector_asset_preparation_server as delivery_server,
)


pytestmark = pytest.mark.unit


def test_semantic_texture_soft_detector_asset_manifests_and_bundle_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[2]
    protocol_tree = ast.parse(
        (root / "experiments/protocol/semantic_texture_soft_detector_assets.py").read_text(
            encoding="utf-8"
        )
    )
    assert all(
        not (
            isinstance(node, (ast.Import, ast.ImportFrom))
            and (
                (isinstance(node, ast.ImportFrom) and node.module and node.module.startswith(("main", "runtime", "experiments.methods", "experiments.metrics", "experiments.runners", "governance", "tests")))
                or (
                    isinstance(node, ast.Import)
                    and any(alias.name.startswith(("main", "runtime", "experiments.methods", "experiments.metrics", "experiments.runners", "governance", "tests")) for alias in node.names)
                )
            )
        )
        for node in ast.walk(protocol_tree)
    )
    whitening = load_manifest(
        root
        / "configs/experiments/"
        "semantic_texture_soft_detector_whitening_fit_manifest.json",
        expected_role=WHITENING_FIT_ROLE,
        count=WHITENING_FIT_COUNT,
    )
    branch = load_manifest(
        root
        / "configs/experiments/"
        "semantic_texture_soft_detector_branch_null_manifest.json",
        expected_role=BRANCH_NULL_ROLE,
        count=BRANCH_NULL_COUNT,
    )
    validate_partition_disjointness(
        whitening,
        branch,
        target_prompt="a red cube",
        target_seed=2026081701,
    )
    assert [item.source_row for item in whitening.entries] == list(range(1, 33))
    assert [item.source_row for item in branch.entries] == list(range(33, 65))

    source_roster = json.loads(
        (
            root / "configs/experiments/hf_only_reference_prompt_roster.json"
        ).read_text(encoding="utf-8")
    )
    used_prompt_digests: set[str] = set()
    used_seeds: set[int] = set()
    for path in sorted((root / "configs/experiments").glob("*.json")):
        if path.name.endswith("manifest.json") or path.name == (
            "hf_only_reference_prompt_roster.json"
        ):
            continue
        source = path.read_text(encoding="utf-8")
        for row in source_roster["rows"]:
            if row["prompt_digest"] in source:
                used_prompt_digests.add(row["prompt_digest"])
    for entry in (*whitening.entries, *branch.entries):
        assert entry.prompt_digest not in used_prompt_digests
        assert entry.generation_seed not in used_seeds
        used_seeds.add(entry.generation_seed)

    energies = tuple((1.0,) * 96 for _ in range(32))
    fit = fit_semantic_texture_lf_whitening_asset(
        energies,
        fit_manifest_sha256=whitening.digest(),
        lf_carrier_config_digest="a" * 64,
    )
    whitening_asset = SemanticTextureLfWhiteningAsset.from_canonical_payload(
        fit.canonical_payload,
        whitening_asset_digest=fit.whitening_asset_digest,
    )
    records = tuple(
        NullScoreRecord(
            score=float(index),
            source_cluster_id=entry.source_cluster_id,
            sample_id=entry.image_lineage_digest,
        )
        for index, entry in enumerate(branch.entries)
    )
    hf_null = SemanticTextureBranchNullCalibration(
        branch="hf",
        detector_identity="b" * 64,
        partition_identity=branch.digest(),
        records=records,
    )
    lf_null = SemanticTextureBranchNullCalibration(
        branch="lf",
        detector_identity="c" * 64,
        partition_identity=branch.digest(),
        records=records,
    )
    bundle = serialize_semantic_texture_soft_detector_asset_bundle(
        whitening_manifest_digest=whitening.digest(),
        branch_null_manifest_digest=branch.digest(),
        whitening_asset=whitening_asset,
        hf_null=hf_null,
        lf_null=lf_null,
    )
    bundle.validate()
    materialized_asset, materialized_hf, materialized_lf = (
        materialize_semantic_texture_soft_detector_asset_bundle(bundle)
    )
    assert materialized_asset.whitening_asset_digest == whitening_asset.whitening_asset_digest
    assert materialized_hf.calibration_identity == hf_null.calibration_identity
    assert materialized_lf.calibration_identity == lf_null.calibration_identity
    with pytest.raises(SemanticTextureSoftDetectorAssetProtocolError):
        replace(bundle, bundle_digest="0" * 64).validate()
    with pytest.raises(SemanticTextureSoftDetectorAssetProtocolError):
        validate_partition_disjointness(
            whitening,
            branch,
            target_prompt=whitening.entries[0].prompt_text,
            target_seed=2026081701,
        )

    public_key = "phase-b-public-detector-key"
    public_carrier_digest = lf_carrier(
        public_key, (1, 16, 64, 64)
    ).carrier_config_digest

    class SyntheticAdapter:
        def __init__(self) -> None:
            self.prepared_seeds: list[int] = []
            self.branch_calls = 0

        def prepare_semantic_texture_clean_primary_null(
            self,
            latent: int,
            _detection_key: str,
            _semantic_runtime: object,
        ) -> object:
            self.prepared_seeds.append(latent)
            return SimpleNamespace(
                lf_carrier_config_digest=public_carrier_digest,
                runtime_detection=SimpleNamespace(
                    lf_observation=SimpleNamespace(values=(0.0,) * (16 * 64 * 64))
                ),
                routing_result=SimpleNamespace(mask_lf=(1.0,) * (16 * 64 * 64)),
            )

        def observe_semantic_texture_primary_null_branches(
            self,
            _prepared: object,
            _detection_key: str,
            _whitening_asset: object,
        ) -> object:
            self.branch_calls += 1
            return SimpleNamespace(
                hf_result=SimpleNamespace(
                    detector_identity="b" * 64,
                    hf_score=float(self.branch_calls),
                ),
                lf_result=SimpleNamespace(
                    detector_identity="c" * 64,
                    lf_score=float(self.branch_calls),
                ),
            )

    prompts: list[str] = []
    adapter = SyntheticAdapter()
    monkeypatch.setattr(
        preparation,
        "semantic_texture_clean_null_band_energy_sums",
        lambda _values, _mask: (1.0,) * 96,
    )
    prepared_bundle = preparation.prepare_semantic_texture_soft_detector_assets(
        adapter,
        preparation.SemanticTextureSoftDetectorAssetPreparationConfiguration(
            whitening_fit_manifest_path=(
                "configs/experiments/"
                "semantic_texture_soft_detector_whitening_fit_manifest.json"
            ),
            branch_null_manifest_path=(
                "configs/experiments/"
                "semantic_texture_soft_detector_branch_null_manifest.json"
            ),
            target_prompt="a red cube",
            target_seed=2026081701,
        ),
        repository_root=root,
        detection_key="test-only-detection-key",
        semantic_runtime=object(),
        base_latent_for_seed=lambda seed: seed,
        set_generation_prompt=prompts.append,
    )
    prepared_bundle.validate()
    assert adapter.prepared_seeds == [
        *(range(202608190000, 202608190032)),
        *(range(202608190100, 202608190132)),
    ]
    assert adapter.branch_calls == 32
    assert len(prompts) == 64

    produced_asset, produced_hf, produced_lf = (
        materialize_semantic_texture_soft_detector_asset_bundle(prepared_bundle)
    )
    route = semantic_texture_content_router(
        (1, 16, 64, 64),
        mode="routing_semantic_texture_soft",
        observations=SemanticTextureRoutingObservations(
            semantic_probability=SpatialRoutingObservation(
                values=(0.25,) * (64 * 64),
                spatial_shape=(64, 64),
                source_identity_digest="c" * 64,
            ),
            texture_complexity=SpatialRoutingObservation(
                values=(0.75,) * (64 * 64),
                spatial_shape=(64, 64),
                source_identity_digest="d" * 64,
            ),
        ),
    )
    public_result = semantic_texture_lf_detector(
        LfDetectionObservation.from_public_image_encoding(
            tuple(((index % 31) - 15) / 31.0 for index in range(16 * 64 * 64)),
            (1, 16, 64, 64),
        ),
        public_key,
        route,
        produced_asset,
    )
    assert public_result.whitening_asset_digest == produced_asset.whitening_asset_digest

    delivery_parent = tmp_path / "asset-delivery"
    delivery_parent.mkdir()
    code, receipt = delivery_server.finalize_semantic_texture_soft_detector_asset_delivery(
        prepared_bundle,
        observed_repository_revision="b" * 40,
        run_id="semantic-texture-soft-detector-assets-test",
        output_root=delivery_parent,
    )
    assert code == 0
    root = delivery_parent / prepared_bundle.bundle_digest
    assert {path.name for path in root.iterdir()} == {
        delivery_server.BUNDLE_FILENAME,
        delivery_server.RESULT_FILENAME,
        delivery_server.RECEIPT_FILENAME,
        delivery_server.CHECKSUMS_FILENAME,
        f"semantic_texture_soft_detector_assets_{prepared_bundle.bundle_digest}.zip",
    }
    assert receipt["status"] == "passed"
    assert receipt["diagnostic_only"] is True
    assert receipt["science_started"] is False

    persisted = json.loads((root / delivery_server.BUNDLE_FILENAME).read_text("utf-8"))
    loaded_bundle = SemanticTextureSoftDetectorAssetBundle.from_mapping(persisted)
    loaded_asset, loaded_hf, loaded_lf = (
        materialize_semantic_texture_soft_detector_asset_bundle(loaded_bundle)
    )
    assert loaded_bundle.bundle_digest == prepared_bundle.bundle_digest
    assert loaded_asset.whitening_asset_digest == prepared_bundle.whitening_asset_digest
    assert loaded_hf.calibration_identity == produced_hf.calibration_identity
    assert loaded_lf.calibration_identity == produced_lf.calibration_identity
    tampered = json.loads(json.dumps(persisted))
    tampered["hf_null_payload"]["records"][0]["score_float64_hex"] = "0x1.8000000000000p+0"
    with pytest.raises(SemanticTextureSoftDetectorAssetProtocolError):
        SemanticTextureSoftDetectorAssetBundle.from_mapping(tampered)
    tampered = json.loads(json.dumps(persisted))
    tampered["unexpected"] = True
    with pytest.raises(SemanticTextureSoftDetectorAssetProtocolError):
        SemanticTextureSoftDetectorAssetBundle.from_mapping(tampered)

    captured: dict[str, tuple[float, ...]] = {}
    monkeypatch.setattr(
        lf_screening,
        "clean_null_band_energy_sums",
        lambda values: captured.setdefault("routed", tuple(values)) or (0.0,) * 96,
    )
    fractional_values = tuple((index % 17 - 8) / 13.0 for index in range(16 * 64 * 64))
    fractional_masks = tuple(0.125 + (index % 11) / 17.0 for index in range(16 * 64 * 64))
    lf_screening.semantic_texture_clean_null_band_energy_sums(
        fractional_values, fractional_masks
    )
    expected_routed = tuple(
        unpack(">f", pack(">f", value * mask))[0]
        for value, mask in zip(fractional_values, fractional_masks, strict=True)
    )
    assert captured["routed"] == expected_routed
