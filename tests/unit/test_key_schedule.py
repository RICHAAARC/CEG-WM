from base64 import b64encode
from dataclasses import FrozenInstanceError
from hashlib import sha256
from pathlib import Path
from struct import pack

import pytest

from main.shared import key_schedule as key_schedule_module
from main.shared.key_schedule import (
    CANDIDATE_ID,
    KEYED_PRG_VERSION,
    NORMAL_QUANTILE_TABLE_SHA256,
    KeyScheduleConfig,
    KeyScheduleError,
    derive_public_noise_stream,
    derive_wrong_key_material,
    derive_wrong_key_stream,
    identify_root_key,
    key_schedule_sha256_counter,
    stable_json_utf8,
)


GOLDEN_DOMAIN = {
    "candidate_id": "key_schedule_sha256_counter",
    "operator": "golden_vector",
    "responsibility_domain": "key_schedule_test",
    "tensor_role": "gaussian",
}

HF_DOMAIN = {
    "candidate_id": "hf_sparse_tail",
    "operator": "carrier_template",
    "responsibility_domain": "hf_carrier",
    "model_revision": "b940f670f0eda2d07fbb75229e779da1ad11eb80",
    "tensor_role": "base_gaussian",
}

LF_DOMAIN = {
    "candidate_id": "lf_low_pass",
    "operator": "carrier_template",
    "responsibility_domain": "lf_carrier",
    "model_revision": "b940f670f0eda2d07fbb75229e779da1ad11eb80",
    "tensor_role": "base_gaussian",
}

GEOMETRY_DOMAIN = {
    "candidate_id": "qk_relation_similarity",
    "operator": "attention_relation_signs",
    "responsibility_domain": "geometry_sync",
    "model_revision": "b940f670f0eda2d07fbb75229e779da1ad11eb80",
    "layer_name": "transformer_blocks.0.attn",
    "token_count": 64,
    "tensor_role": "pair_uniform",
}

PUBLIC_QK_DOMAIN = {
    "candidate_id": "qk_relation_similarity",
    "operator": "public_image_only_qk_detection_noise",
    "responsibility_domain": "public_noise",
    "model_revision": "b940f670f0eda2d07fbb75229e779da1ad11eb80",
    "schedule_index": 7,
    "conditioning_protocol": "sd3_empty_text_triplet_without_cfg",
    "tensor_role": "scheduler_noise",
}


@pytest.mark.unit
def test_key_schedule_root_and_domain_separation() -> None:
    assert stable_json_utf8({"π": [None, True, 7], "a": "原样"}) == (
        b'{"a":"\xe5\x8e\x9f\xe6\xa0\xb7","\xcf\x80":[null,true,7]}'
    )
    identity = identify_root_key("ceg-wm-golden-root-π")
    assert identity.root_key_public_digest == (
        "51ad81701f05213fbd7ee5cecc0987ffca7d8be76cff58394dc0da4fe8e1423d"
    )
    assert identify_root_key("\u00e9") != identify_root_key("e\u0301")

    hf = key_schedule_sha256_counter("registered-root", HF_DOMAIN, [2, 2])
    lf = key_schedule_sha256_counter("registered-root", LF_DOMAIN, [2, 2])
    geometry = key_schedule_sha256_counter(
        "registered-root",
        GEOMETRY_DOMAIN,
        [2, 2],
        distribution="uniform",
    )
    assert len({hf.domain_digest, lf.domain_digest, geometry.domain_digest}) == 3
    assert hf.values != lf.values
    assert hf.config_digest == lf.config_digest == geometry.config_digest


@pytest.mark.unit
def test_key_schedule_counter_quantile_golden() -> None:
    gaussian = key_schedule_sha256_counter(
        "ceg-wm-golden-root-π",
        GOLDEN_DOMAIN,
        [2, 3],
    )
    uniform = key_schedule_sha256_counter(
        "ceg-wm-golden-root-π",
        GOLDEN_DOMAIN,
        [2, 3],
        distribution="uniform",
    )

    assert gaussian.domain_digest == (
        "e5b8e35d13815c1d23a09286da0bfe661e0330e38eda19e239f19224f7b1998f"
    )
    assert gaussian.quantile_indices == (
        172059,
        964892,
        707530,
        322430,
        968250,
        915318,
    )
    assert [pack(">f", value).hex() for value in gaussian.values] == [
        "bf7a508b",
        "3fb40402",
        "3ee7f9d3",
        "bf00c274",
        "3fb6d22b",
        "3f91f4c9",
    ]
    assert gaussian.values_float32_be_sha256 == (
        "c82e2f254ab05f4502d397aa444d8facefaa64e0c4df4f1617e12948acecb8d0"
    )
    assert [pack(">f", value).hex() for value in uniform.values] == [
        "3e2806fb",
        "3f6b7eec",
        "3f1ca35d",
        "3e6af2b7",
        "3f213aef",
        "3ef25444",
    ]
    assert all(0.0 < value < 1.0 for value in uniform.values)

    cross_block = key_schedule_sha256_counter(
        "ceg-wm-golden-root-π",
        GOLDEN_DOMAIN,
        [1, 14],
    )
    assert cross_block.domain_digest == (
        "f70de8c70d23476c05d67457103c1aceecfd320ef512a7895479f5a113d7d170"
    )
    assert cross_block.quantile_indices == (
        666601,
        190935,
        927525,
        564118,
        976534,
        107375,
        656472,
        326102,
        1000891,
        898163,
        96925,
        355206,
        144019,
        470889,
    )


@pytest.mark.unit
def test_key_schedule_wrong_key_and_public_noise() -> None:
    registered = identify_root_key("ceg-wm-golden-root-π")
    wrong = derive_wrong_key_material(registered.root_key_public_digest, 0)
    assert wrong.material_text == (
        "ceg-wm-wrong-key:"
        "843d3aa0d4d81ed3b17c7d0bd970145ef912ed3188db3079237214da185c985f"
    )
    assert wrong == derive_wrong_key_material(registered.root_key_public_digest, 0)

    registered_stream = key_schedule_sha256_counter(
        "ceg-wm-golden-root-π",
        GEOMETRY_DOMAIN,
        [3, 3],
        distribution="uniform",
    )
    wrong_stream = derive_wrong_key_stream(
        wrong,
        GEOMETRY_DOMAIN,
        [3, 3],
        distribution="uniform",
    )
    public_stream = derive_public_noise_stream(
        PUBLIC_QK_DOMAIN,
        [3, 3],
        distribution="uniform",
    )
    assert registered_stream.values != wrong_stream.values
    assert public_stream == derive_public_noise_stream(
        PUBLIC_QK_DOMAIN,
        [3, 3],
        distribution="uniform",
    )
    assert public_stream.domain_digest not in {
        registered_stream.domain_digest,
        wrong_stream.domain_digest,
    }


@pytest.mark.unit
@pytest.mark.parametrize(
    "root_key_text",
    ["", b"bytes-are-not-keys", "\ud800"],
)
def test_key_schedule_rejects_invalid_root_keys(root_key_text: object) -> None:
    with pytest.raises(KeyScheduleError):
        identify_root_key(root_key_text)  # type: ignore[arg-type]


@pytest.mark.unit
@pytest.mark.parametrize(
    "value",
    [1.0, {"float": 1.0}, ("tuple",), {"set"}, {1: "non-string-key"}],
)
def test_key_schedule_stable_json_rejects_unregistered_types(value: object) -> None:
    with pytest.raises(KeyScheduleError):
        stable_json_utf8(value)


@pytest.mark.unit
def test_key_schedule_rejects_domain_shape_and_material_role_drift() -> None:
    extra_domain = dict(HF_DOMAIN, unexpected="drift")
    with pytest.raises(KeyScheduleError):
        key_schedule_sha256_counter("registered-root", extra_domain, [2, 2])
    with pytest.raises(KeyScheduleError):
        key_schedule_sha256_counter("registered-root", HF_DOMAIN, [2, 0])
    with pytest.raises(KeyScheduleError):
        key_schedule_sha256_counter(
            "registered-root",
            PUBLIC_QK_DOMAIN,
            [2, 2],
        )
    with pytest.raises(KeyScheduleError):
        derive_public_noise_stream(HF_DOMAIN, [2, 2])


@pytest.mark.unit
def test_key_schedule_rejects_counter_overflow_before_streaming() -> None:
    with pytest.raises(KeyScheduleError, match="counter"):
        key_schedule_sha256_counter(
            "registered-root",
            HF_DOMAIN,
            [(1 << 128) * 4 + 1],
            distribution="uniform",
        )


@pytest.mark.unit
def test_key_schedule_configuration_and_results_are_immutable() -> None:
    config = KeyScheduleConfig()
    result = key_schedule_sha256_counter(
        "registered-root",
        HF_DOMAIN,
        [1, 2],
        config=config,
    )
    assert config.candidate_id == result.candidate_id == CANDIDATE_ID
    assert config.keyed_prg_version == KEYED_PRG_VERSION
    assert len(config.config_digest) == 64
    with pytest.raises(FrozenInstanceError):
        config.candidate_id = "drift"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.domain_digest = "drift"  # type: ignore[misc]


@pytest.mark.unit
def test_key_schedule_quantile_asset_digest_is_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tampered_table = tmp_path / "normal-table.bin"
    tampered_table.write_bytes(b64encode(b"\x00" * (1 << 22)))
    monkeypatch.setattr(
        key_schedule_module,
        "_NORMAL_QUANTILE_TABLE_PATH",
        tampered_table,
    )
    key_schedule_module._load_normal_quantile_table.cache_clear()
    try:
        with pytest.raises(KeyScheduleError, match="digest"):
            key_schedule_sha256_counter(
                "ceg-wm-golden-root-π",
                GOLDEN_DOMAIN,
                [1],
            )
    finally:
        key_schedule_module._load_normal_quantile_table.cache_clear()

    original_table = (
        Path(key_schedule_module.__file__).with_name(
            "normal_quantile_table20_float32_be.txt"
        )
    ).read_text(encoding="ascii")
    decoded_table = key_schedule_module.b64decode(original_table, validate=True)
    assert len(decoded_table) == 1 << 22
    assert sha256(decoded_table).hexdigest() == NORMAL_QUANTILE_TABLE_SHA256
