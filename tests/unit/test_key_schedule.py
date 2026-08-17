from base64 import b64encode
from dataclasses import FrozenInstanceError
from fractions import Fraction
from hashlib import sha256
from pathlib import Path
from random import Random
from struct import pack

import pytest

from main.shared import key_schedule as key_schedule_module
from main.shared.key_schedule import (
    CANDIDATE_ID,
    DerivedWrongKeyMaterial,
    KEYED_PRG_VERSION,
    NORMAL_QUANTILE_TABLE_SHA256,
    KeyScheduleConfig,
    KeyScheduleError,
    derive_public_noise_stream,
    derive_wrong_key_material,
    derive_wrong_key_stream,
    identify_root_key,
    key_schedule_sha256_counter,
    normal_quantile_table_lookup,
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
    "tensor_role": "base_gaussian",
}

LF_DOMAIN = {
    "candidate_id": "lf_low_pass",
    "operator": "carrier_template",
    "responsibility_domain": "lf_carrier",
    "tensor_role": "base_gaussian",
}

GEOMETRY_DOMAIN = {
    "candidate_id": "qk_relation_similarity",
    "operator": "attention_relation_signs",
    "responsibility_domain": "geometry_sync",
    "layer_name": "transformer_blocks.0.attn",
    "token_count": 64,
    "tensor_role": "pair_uniform",
}

PUBLIC_QK_DOMAIN = {
    "candidate_id": "qk_relation_similarity",
    "operator": "public_image_only_qk_detection_noise",
    "responsibility_domain": "public_noise",
    "schedule_index": 7,
    "conditioning_protocol": "sd3_empty_text_triplet_without_cfg",
    "tensor_role": "scheduler_noise",
}


def _positive_binary32_fraction(binary32_bits: int) -> Fraction:
    exponent_bits = (binary32_bits >> 23) & 0xFF
    fraction_bits = binary32_bits & ((1 << 23) - 1)
    if exponent_bits == 0:
        significand = fraction_bits
        exponent = -149
    else:
        significand = (1 << 23) | fraction_bits
        exponent = exponent_bits - 127 - 23
    if exponent >= 0:
        return Fraction(significand << exponent, 1)
    return Fraction(significand, 1 << (-exponent))


def _oracle_uniform_binary32_bits(mantissa: int) -> int:
    target = Fraction(mantissa + 1, (1 << 53) + 2)
    lower_bits = 0
    upper_bits = 0x3F800000
    while lower_bits + 1 < upper_bits:
        middle_bits = (lower_bits + upper_bits) // 2
        if _positive_binary32_fraction(middle_bits) <= target:
            lower_bits = middle_bits
        else:
            upper_bits = middle_bits
    lower_value = _positive_binary32_fraction(lower_bits)
    if lower_value == target:
        return lower_bits
    upper_value = _positive_binary32_fraction(upper_bits)
    lower_distance = target - lower_value
    upper_distance = upper_value - target
    if lower_distance < upper_distance:
        return lower_bits
    if upper_distance < lower_distance:
        return upper_bits
    return lower_bits if lower_bits & 1 == 0 else upper_bits


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
    registered_identity = identify_root_key("registered-root")
    expected_hf_envelope = {
        "distribution": "gaussian",
        "key_role": "registered",
        "normal_quantile_table_sha256": NORMAL_QUANTILE_TABLE_SHA256,
        "root_key_public_digest": registered_identity.root_key_public_digest,
        "semantic_domain": HF_DOMAIN,
        "shape": [2, 2],
        "version": KEYED_PRG_VERSION,
        "wrong_key_index": None,
    }
    assert set(expected_hf_envelope) == {
        "distribution",
        "key_role",
        "normal_quantile_table_sha256",
        "root_key_public_digest",
        "semantic_domain",
        "shape",
        "version",
        "wrong_key_index",
    }
    assert hf.domain_digest == sha256(
        b"registered-root\x00" + stable_json_utf8(expected_hf_envelope)
    ).hexdigest()
    alternate_geometry_domain = {
        **GEOMETRY_DOMAIN,
        "layer_name": "transformer_blocks.23.attn",
        "token_count": 65,
    }
    alternate_geometry = key_schedule_sha256_counter(
        "registered-root",
        alternate_geometry_domain,
        [2, 2],
        distribution="uniform",
    )
    assert alternate_geometry.domain_digest != geometry.domain_digest


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
        "8a70d7d728e57077123b2df8092dc4a7608030af3105d6bf6d0f4ff376a20ecc"
    )
    assert gaussian.quantile_indices_random == (
        609925,
        291593,
        478679,
        1031076,
        435951,
        227847,
    )
    assert [pack(">f", value).hex() for value in gaussian.values] == [
        "3e531dca",
        "bf16aa80",
        "bddfbb80",
        "40082924",
        "be59dea7",
        "bf4807e6",
    ]
    assert gaussian.values_float32_be_sha256 == (
        "216538d38ea0453bf8c20f77f08c7a08f76cc69c23339a149da1e69151ff073a"
    )
    assert [pack(">f", value).hex() for value in uniform.values] == [
        "3eaec2aa",
        "3f5cbcf4",
        "3f130075",
        "3e70cd8e",
        "3e0a0b24",
        "3f6d4553",
    ]
    assert all(0.0 < value < 1.0 for value in uniform.values)

    cross_block = key_schedule_sha256_counter(
        "ceg-wm-golden-root-π",
        GOLDEN_DOMAIN,
        [1, 14],
    )
    assert cross_block.domain_digest == (
        "5b6fb9e957687b4ead04c587b6bb300c11687ccd872b5e7f2a1b42a4ef9ac155"
    )
    assert cross_block.quantile_indices_random == (
        989210,
        206028,
        844962,
        125147,
        357930,
        574565,
        499012,
        177947,
        11892,
        156803,
        676679,
        221429,
        60515,
        664296,
    )


@pytest.mark.unit
def test_key_schedule_uniform_exact_binary32_adversarial_rounding() -> None:
    adversarial_mantissa = 6755399172620288
    result = key_schedule_module._uniform_mantissa_to_float32(
        adversarial_mantissa
    )
    assert pack(">f", result).hex() == "3f3fffff"
    assert int.from_bytes(pack(">f", result), "big") == (
        _oracle_uniform_binary32_bits(adversarial_mantissa)
    )


@pytest.mark.unit
def test_key_schedule_uniform_exact_binary32_edges_and_random_cross_check() -> None:
    edge_mantissas = (
        0,
        1,
        (1 << 24) - 1,
        1 << 24,
        (1 << 52) - 1,
        1 << 52,
        6755399172620287,
        6755399172620288,
        6755399172620289,
        (1 << 53) - 2,
        (1 << 53) - 1,
    )
    generator = Random(20260727)
    sampled_mantissas = tuple(generator.getrandbits(53) for _ in range(256))
    for mantissa in edge_mantissas + sampled_mantissas:
        actual_value = key_schedule_module._uniform_mantissa_to_float32(mantissa)
        actual_bits = int.from_bytes(pack(">f", actual_value), "big")
        assert actual_bits == _oracle_uniform_binary32_bits(mantissa)


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
def test_key_schedule_wrong_key_stream_rederives_material_from_public_identity() -> None:
    registered = identify_root_key("ceg-wm-golden-root-π")
    canonical = derive_wrong_key_material(registered.root_key_public_digest, 3)
    baseline = derive_wrong_key_stream(
        canonical,
        GEOMETRY_DOMAIN,
        [2, 3],
        distribution="uniform",
    )

    directly_constructed = DerivedWrongKeyMaterial(
        wrong_key_index=3,
        registered_root_key_public_digest=registered.root_key_public_digest,
    )
    assert derive_wrong_key_stream(
        directly_constructed,
        GEOMETRY_DOMAIN,
        [2, 3],
        distribution="uniform",
    ) == baseline
    with pytest.raises(TypeError):
        DerivedWrongKeyMaterial(
            wrong_key_index=3,
            registered_root_key_public_digest=registered.root_key_public_digest,
            material_text="attacker-chosen-material",
        )


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
    missing_qk_field = dict(GEOMETRY_DOMAIN)
    del missing_qk_field["layer_name"]
    with pytest.raises(KeyScheduleError):
        key_schedule_sha256_counter(
            "registered-root",
            missing_qk_field,
            [2, 2],
            distribution="uniform",
        )
    nonapplicable_qk_field = dict(HF_DOMAIN, layer_name="not-applicable")
    with pytest.raises(KeyScheduleError):
        key_schedule_sha256_counter(
            "registered-root",
            nonapplicable_qk_field,
            [2, 2],
        )
    public_schedule_drift = dict(PUBLIC_QK_DOMAIN, schedule_index=8)
    with pytest.raises(KeyScheduleError):
        derive_public_noise_stream(
            public_schedule_drift,
            [2, 2],
            distribution="uniform",
        )


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
def test_key_schedule_public_quantile_lookup_reuses_frozen_asset() -> None:
    assert pack(">f", normal_quantile_table_lookup(0)).hex() == "c09cd4b3"
    assert pack(">f", normal_quantile_table_lookup(524288)).hex() == "35a06c99"
    assert pack(">f", normal_quantile_table_lookup((1 << 20) - 1)).hex() == (
        "409cd4b3"
    )
    with pytest.raises(KeyScheduleError, match="index"):
        normal_quantile_table_lookup(-1)
    with pytest.raises(KeyScheduleError, match="index"):
        normal_quantile_table_lookup(1 << 20)


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
