"""冻结的 CEG-WM SHA-256 counter 密钥与伪随机流协议。

本模块独占 root-key 编码、职责域验证、wrong-key/public-noise 派生以及
uniform/Gaussian float32 物化。它不保存原始 root key，也不调用设备 RNG。
"""

from __future__ import annotations

from base64 import b64decode
from binascii import Error as Base64Error
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
import json
from math import prod
from pathlib import Path
from struct import pack, unpack, unpack_from
from typing import Literal, Sequence

CANDIDATE_ID = "key_schedule_sha256_counter"
KEYED_PRG_VERSION = "sha256_counter_semantic_domain_v2_normal_icdf_table20_float32"
NORMAL_QUANTILE_TABLE_SHA256 = (
    "70abf440a7f3670147965ffa52f5aaa639dab97f6282b68f3a9a1b1ce5e6cf5a"
)
PUBLIC_NOISE_KEY_MATERIAL = "ceg-wm-public-noise:key-schedule-sha256-counter"

_NORMAL_TABLE_ENTRY_COUNT = 1 << 20
_NORMAL_TABLE_BYTE_COUNT = _NORMAL_TABLE_ENTRY_COUNT * 4
_MAX_COUNTER_BLOCKS = 1 << 128
_NORMAL_QUANTILE_TABLE_PATH = Path(__file__).with_name(
    "normal_quantile_table20_float32_be.txt"
)

Distribution = Literal["uniform", "gaussian"]


class KeyScheduleError(ValueError):
    """密钥协议输入、身份或冻结资产不符合候选规格。"""


@dataclass(frozen=True, slots=True)
class KeyScheduleConfig:
    """冻结候选的不可变配置身份。"""

    candidate_id: str = CANDIDATE_ID
    keyed_prg_version: str = KEYED_PRG_VERSION
    normal_quantile_table_sha256: str = NORMAL_QUANTILE_TABLE_SHA256
    root_key_encoding: str = "strict_utf8_no_normalization"
    stable_serialization: str = "stable_json_utf8_v1"
    uniform_protocol: str = "uint64_be_high53_open_interval_float32"
    gaussian_protocol: str = "msb_first_20bit_midpoint_table_float32"

    def __post_init__(self) -> None:
        expected = (
            CANDIDATE_ID,
            KEYED_PRG_VERSION,
            NORMAL_QUANTILE_TABLE_SHA256,
            "strict_utf8_no_normalization",
            "stable_json_utf8_v1",
            "uint64_be_high53_open_interval_float32",
            "msb_first_20bit_midpoint_table_float32",
        )
        if tuple(self.as_identity_value().values()) != expected:
            raise KeyScheduleError("key schedule configuration identity mismatch")

    def as_identity_value(self) -> dict[str, str]:
        """返回进入稳定配置摘要的唯一字段和值。"""

        return {
            "candidate_id": self.candidate_id,
            "keyed_prg_version": self.keyed_prg_version,
            "normal_quantile_table_sha256": self.normal_quantile_table_sha256,
            "root_key_encoding": self.root_key_encoding,
            "stable_serialization": self.stable_serialization,
            "uniform_protocol": self.uniform_protocol,
            "gaussian_protocol": self.gaussian_protocol,
        }

    @property
    def config_digest(self) -> str:
        """冻结密钥协议配置的 SHA-256 摘要。"""

        return sha256(stable_json_utf8(self.as_identity_value())).hexdigest()


@dataclass(frozen=True, slots=True)
class RootKeyIdentity:
    """可持久化的 root-key 公共身份，不含密钥材料。"""

    root_key_public_digest: str
    config_digest: str


@dataclass(frozen=True, slots=True)
class DerivedWrongKeyMaterial:
    """预登记 wrong-key 的公开派生身份。"""

    wrong_key_index: int
    registered_root_key_public_digest: str

    @property
    def material_text(self) -> str:
        """按冻结公式重建仅供内存调用的派生材料。"""

        return _derive_wrong_key_material_text(
            self.registered_root_key_public_digest,
            self.wrong_key_index,
        )


@dataclass(frozen=True, slots=True)
class DerivedInternalLfDecoyMaterial:
    """In-memory-only candidate-specific LF internal-decoy material."""

    lf_candidate_id: str
    internal_decoy_index: int
    registered_root_key_public_digest: str

    @property
    def material_text(self) -> str:
        return _derive_internal_lf_decoy_material_text(
            self.registered_root_key_public_digest,
            self.lf_candidate_id,
            self.internal_decoy_index,
        )


@dataclass(frozen=True, slots=True)
class KeyStreamResult:
    """不可变的 row-major CPU float32 流及其可审计身份。"""

    candidate_id: str
    distribution: Distribution
    shape: tuple[int, ...]
    values: tuple[float, ...]
    domain_digest: str
    config_digest: str
    values_float32_be_sha256: str
    quantile_indices_random: tuple[int, ...] | None


DEFAULT_CONFIG = KeyScheduleConfig()


def _validate_config(config: object) -> KeyScheduleConfig:
    if type(config) is not KeyScheduleConfig:
        raise KeyScheduleError("config must be the frozen KeyScheduleConfig")
    return config


def _validate_text(value: object, field_name: str, *, allow_empty: bool = False) -> str:
    if type(value) is not str or (not value and not allow_empty):
        raise KeyScheduleError(f"{field_name} must be a {'non-empty ' if not allow_empty else ''}str")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise KeyScheduleError(f"{field_name} must encode as strict UTF-8") from exc
    return value


def _validate_stable_json_value(value: object, active: set[int]) -> None:
    value_type = type(value)
    if value is None or value_type in {bool, int}:
        return
    if value_type is str:
        _validate_text(value, "stable JSON string", allow_empty=True)
        return
    if value_type is list:
        identity = id(value)
        if identity in active:
            raise KeyScheduleError("stable JSON value must not contain cycles")
        active.add(identity)
        try:
            for item in value:
                _validate_stable_json_value(item, active)
        finally:
            active.remove(identity)
        return
    if value_type is dict:
        identity = id(value)
        if identity in active:
            raise KeyScheduleError("stable JSON value must not contain cycles")
        active.add(identity)
        try:
            for key, item in value.items():
                if type(key) is not str:
                    raise KeyScheduleError("stable JSON map keys must be str")
                _validate_text(key, "stable JSON map key", allow_empty=True)
                _validate_stable_json_value(item, active)
        finally:
            active.remove(identity)
        return
    raise KeyScheduleError(
        "stable JSON accepts only null, bool, int, str, list, and str-keyed dict"
    )


def stable_json_utf8(value: object) -> bytes:
    """按冻结 JSON 规则序列化并执行严格 UTF-8 编码。"""

    _validate_stable_json_value(value, set())
    try:
        serialized = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return serialized.encode("utf-8", errors="strict")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise KeyScheduleError("stable JSON serialization failed") from exc


def _normalize_shape(shape: Sequence[int]) -> tuple[int, ...]:
    if isinstance(shape, (str, bytes)) or not isinstance(shape, Sequence):
        raise KeyScheduleError("shape must be a non-empty sequence of positive integers")
    normalized = tuple(shape)
    if not normalized or any(type(size) is not int or size <= 0 for size in normalized):
        raise KeyScheduleError("shape must be a non-empty sequence of positive integers")
    return normalized


def _expect_exact_fields(
    domain_fields: dict[str, object],
    expected_fields: set[str],
) -> None:
    actual_fields = set(domain_fields)
    if actual_fields != expected_fields:
        raise KeyScheduleError("domain fields do not match the frozen responsibility schema")


def _expect_literal(domain_fields: dict[str, object], field: str, value: object) -> None:
    if domain_fields[field] != value or type(domain_fields[field]) is not type(value):
        raise KeyScheduleError(f"domain field {field} does not match the frozen value")


def _expect_non_empty_text(domain_fields: dict[str, object], field: str) -> None:
    _validate_text(domain_fields[field], f"domain field {field}")


def _validate_domain_fields(domain_fields: object) -> tuple[dict[str, object], bool]:
    if type(domain_fields) is not dict:
        raise KeyScheduleError("domain_fields must be a plain dict")
    _validate_stable_json_value(domain_fields, set())

    candidate_id = domain_fields.get("candidate_id")
    operator = domain_fields.get("operator")
    responsibility = domain_fields.get("responsibility_domain")
    identity = (candidate_id, operator, responsibility)

    if identity == (CANDIDATE_ID, "golden_vector", "key_schedule_test"):
        _expect_exact_fields(
            domain_fields,
            {"candidate_id", "operator", "responsibility_domain", "tensor_role"},
        )
        _expect_literal(domain_fields, "tensor_role", "gaussian")
        return dict(domain_fields), False

    if identity == ("hf_sparse_tail", "carrier_template", "hf_carrier"):
        _expect_exact_fields(
            domain_fields,
            {
                "candidate_id",
                "operator",
                "responsibility_domain",
                "tensor_role",
            },
        )
        _expect_literal(domain_fields, "tensor_role", "base_gaussian")
        return dict(domain_fields), False

    if identity == ("lf_low_pass", "carrier_template", "lf_carrier"):
        _expect_exact_fields(
            domain_fields,
            {
                "candidate_id",
                "operator",
                "responsibility_domain",
                "tensor_role",
            },
        )
        _expect_literal(domain_fields, "tensor_role", "base_gaussian")
        return dict(domain_fields), False

    contrastive_carrier_operators = {
        "lf_multiscale_lowpass_contrastive": {
            "carrier_template_lowpass_five_by_five",
            "carrier_template_lowpass_nine_by_nine",
        },
        "lf_five_by_five_lowpass_contrastive": {
            "carrier_template_lowpass_five_by_five",
        },
    }
    if (
        candidate_id in contrastive_carrier_operators
        and operator in contrastive_carrier_operators[candidate_id]
        and responsibility == "lf_carrier"
    ):
        _expect_exact_fields(
            domain_fields,
            {"candidate_id", "operator", "responsibility_domain", "tensor_role"},
        )
        _expect_literal(domain_fields, "tensor_role", "base_gaussian")
        return dict(domain_fields), False

    internal_operators = {
        "internal_decoy_carrier_template_lowpass_five_by_five",
        "internal_decoy_carrier_template_lowpass_nine_by_nine",
    }
    if (
        candidate_id in contrastive_carrier_operators
        and operator in internal_operators
        and responsibility == "lf_detector_internal_decoy"
    ):
        allowed_scale = (
            operator.endswith("five_by_five")
            and "carrier_template_lowpass_five_by_five"
            in contrastive_carrier_operators[candidate_id]
        ) or (
            operator.endswith("nine_by_nine")
            and "carrier_template_lowpass_nine_by_nine"
            in contrastive_carrier_operators[candidate_id]
        )
        if not allowed_scale:
            raise KeyScheduleError("internal LF decoy scale is not registered")
        _expect_exact_fields(
            domain_fields,
            {"candidate_id", "operator", "responsibility_domain", "tensor_role"},
        )
        _expect_literal(domain_fields, "tensor_role", "base_gaussian")
        return dict(domain_fields), False

    if identity == (
        "qk_relation_similarity",
        "attention_relation_signs",
        "geometry_sync",
    ):
        _expect_exact_fields(
            domain_fields,
            {
                "candidate_id",
                "operator",
                "responsibility_domain",
                "layer_name",
                "token_count",
                "tensor_role",
            },
        )
        _expect_non_empty_text(domain_fields, "layer_name")
        token_count = domain_fields["token_count"]
        if type(token_count) is not int or token_count <= 1:
            raise KeyScheduleError("domain field token_count must be an integer greater than one")
        _expect_literal(domain_fields, "tensor_role", "pair_uniform")
        return dict(domain_fields), False

    if identity == (
        "qk_relation_similarity",
        "public_image_only_qk_detection_noise",
        "public_noise",
    ):
        _expect_exact_fields(
            domain_fields,
            {
                "candidate_id",
                "operator",
                "responsibility_domain",
                "schedule_index",
                "conditioning_protocol",
                "tensor_role",
            },
        )
        _expect_literal(domain_fields, "schedule_index", 7)
        _expect_literal(
            domain_fields,
            "conditioning_protocol",
            "sd3_empty_text_triplet_without_cfg",
        )
        _expect_literal(domain_fields, "tensor_role", "scheduler_noise")
        return dict(domain_fields), True

    if identity == (
        "routing_stqr",
        "local_sensitivity_public_probe",
        "public_noise",
    ):
        _expect_exact_fields(
            domain_fields,
            {
                "candidate_id",
                "conditioning_protocol",
                "operator",
                "responsibility_domain",
                "schedule_index",
                "tensor_role",
            },
        )
        _expect_literal(domain_fields, "schedule_index", 18)
        _expect_literal(
            domain_fields,
            "conditioning_protocol",
            "generation_callback18_vae_local_sensitivity",
        )
        _expect_literal(domain_fields, "tensor_role", "latent_probe")
        return dict(domain_fields), True

    raise KeyScheduleError("unregistered key schedule responsibility domain")


def identify_root_key(
    root_key_text: str,
    *,
    config: KeyScheduleConfig = DEFAULT_CONFIG,
) -> RootKeyIdentity:
    """计算唯一允许持久化的 root-key 公共摘要。"""

    config = _validate_config(config)
    root_key_text = _validate_text(root_key_text, "root_key_text")
    payload = {
        "candidate_id": CANDIDATE_ID,
        "record_role": "root_key_public_digest",
        "root_key_text": root_key_text,
    }
    return RootKeyIdentity(
        root_key_public_digest=sha256(stable_json_utf8(payload)).hexdigest(),
        config_digest=config.config_digest,
    )


def _validate_public_digest(value: object) -> str:
    digest = _validate_text(value, "registered_root_key_public_digest")
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise KeyScheduleError(
            "registered_root_key_public_digest must be 64 lowercase hexadecimal characters"
        )
    return digest


def derive_wrong_key_material(
    registered_root_key_public_digest: str,
    wrong_key_index: int,
) -> DerivedWrongKeyMaterial:
    """从公开注册摘要与预登记索引确定性派生 wrong-key 根材料。"""

    digest = _validate_public_digest(registered_root_key_public_digest)
    if type(wrong_key_index) is not int or wrong_key_index < 0:
        raise KeyScheduleError("wrong_key_index must be a non-negative integer")
    return DerivedWrongKeyMaterial(
        wrong_key_index=wrong_key_index,
        registered_root_key_public_digest=digest,
    )


def derive_internal_lf_decoy_material(
    registered_root_key_public_digest: str,
    lf_candidate_id: str,
    internal_decoy_index: int,
) -> DerivedInternalLfDecoyMaterial:
    """Create one authenticated in-memory LF internal-decoy capability."""

    digest = _validate_public_digest(registered_root_key_public_digest)
    if lf_candidate_id not in {
        "lf_multiscale_lowpass_contrastive",
        "lf_five_by_five_lowpass_contrastive",
    }:
        raise KeyScheduleError("LF internal decoy candidate is not registered")
    if type(internal_decoy_index) is not int or internal_decoy_index not in range(8):
        raise KeyScheduleError("internal_decoy_index must be in [0,7]")
    return DerivedInternalLfDecoyMaterial(
        lf_candidate_id=lf_candidate_id,
        internal_decoy_index=internal_decoy_index,
        registered_root_key_public_digest=digest,
    )


def _derive_internal_lf_decoy_material_text(
    registered_root_key_public_digest: str,
    lf_candidate_id: str,
    internal_decoy_index: int,
) -> str:
    digest = _validate_public_digest(registered_root_key_public_digest)
    if lf_candidate_id not in {
        "lf_multiscale_lowpass_contrastive",
        "lf_five_by_five_lowpass_contrastive",
    } or type(internal_decoy_index) is not int or internal_decoy_index not in range(8):
        raise KeyScheduleError("LF internal decoy identity is invalid")
    payload = {
        "candidate_id": CANDIDATE_ID,
        "derivation_role": "candidate_internal_lf_decoy",
        "lf_candidate_id": lf_candidate_id,
        "registered_root_key_public_digest": digest,
        "internal_decoy_index": internal_decoy_index,
    }
    return "ceg-wm-internal-lf-decoy:" + sha256(stable_json_utf8(payload)).hexdigest()


def _derive_wrong_key_material_text(
    registered_root_key_public_digest: str,
    wrong_key_index: int,
) -> str:
    digest = _validate_public_digest(registered_root_key_public_digest)
    if type(wrong_key_index) is not int or wrong_key_index < 0:
        raise KeyScheduleError("wrong_key_index must be a non-negative integer")
    payload = {
        "candidate_id": CANDIDATE_ID,
        "derivation_role": "geometry_and_content_wrong_key",
        "registered_root_key_public_digest": digest,
        "wrong_key_index": wrong_key_index,
    }
    return "ceg-wm-wrong-key:" + sha256(stable_json_utf8(payload)).hexdigest()


def _required_block_count(element_count: int, distribution: Distribution) -> int:
    if distribution == "uniform":
        block_count = (element_count + 3) // 4
    elif distribution == "gaussian":
        block_count = (element_count * 20 + 255) // 256
    else:
        raise KeyScheduleError("distribution must be uniform or gaussian")
    if block_count > _MAX_COUNTER_BLOCKS:
        raise KeyScheduleError("uint128 counter would overflow")
    return block_count


def _domain_digest(
    key_material: str,
    domain_fields: dict[str, object],
    shape: tuple[int, ...],
    distribution: Distribution,
    *,
    key_role: str,
    root_key_public_digest: str,
    wrong_key_index: int | None,
) -> bytes:
    seed_envelope = {
        "distribution": distribution,
        "key_role": key_role,
        "normal_quantile_table_sha256": NORMAL_QUANTILE_TABLE_SHA256,
        "root_key_public_digest": root_key_public_digest,
        "semantic_domain": domain_fields,
        "shape": list(shape),
        "version": KEYED_PRG_VERSION,
        "wrong_key_index": wrong_key_index,
    }
    return sha256(
        key_material.encode("utf-8")
        + b"\x00"
        + stable_json_utf8(seed_envelope)
    ).digest()


def _counter_blocks(domain_digest: bytes, block_count: int):
    for counter in range(block_count):
        yield sha256(domain_digest + counter.to_bytes(16, "big", signed=False)).digest()


def _uniform_mantissa_to_float32(mantissa: int) -> float:
    """把冻结的 53-bit 有理数直接按 RNE 物化为 IEEE-754 binary32。"""

    if type(mantissa) is not int or not 0 <= mantissa < (1 << 53):
        raise KeyScheduleError("uniform mantissa must be an unsigned 53-bit integer")

    numerator = mantissa + 1
    denominator = (1 << 53) + 2
    exponent = numerator.bit_length() - denominator.bit_length()
    if exponent >= 0:
        if numerator < (denominator << exponent):
            exponent -= 1
    elif numerator << (-exponent) < denominator:
        exponent -= 1

    scaled_numerator = numerator << (23 - exponent)
    significand, remainder = divmod(scaled_numerator, denominator)
    doubled_remainder = remainder << 1
    if doubled_remainder > denominator or (
        doubled_remainder == denominator and significand & 1
    ):
        significand += 1

    if significand == 1 << 24:
        significand >>= 1
        exponent += 1
    if not (1 << 23) <= significand < (1 << 24):
        raise KeyScheduleError("uniform binary32 significand is out of range")

    exponent_bits = exponent + 127
    if not 1 <= exponent_bits <= 127:
        raise KeyScheduleError("uniform binary32 exponent is out of range")
    binary32_bits = (exponent_bits << 23) | (significand - (1 << 23))
    return unpack(">f", pack(">I", binary32_bits))[0]


@lru_cache(maxsize=1)
def _load_normal_quantile_table() -> bytes:
    try:
        encoded_table = _NORMAL_QUANTILE_TABLE_PATH.read_text(encoding="ascii")
        table = b64decode(encoded_table, validate=True)
    except (OSError, UnicodeError, Base64Error) as exc:
        raise KeyScheduleError("frozen normal quantile table is unavailable") from exc
    if len(table) != _NORMAL_TABLE_BYTE_COUNT:
        raise KeyScheduleError("frozen normal quantile table has an invalid byte length")
    if sha256(table).hexdigest() != NORMAL_QUANTILE_TABLE_SHA256:
        raise KeyScheduleError("frozen normal quantile table digest mismatch")
    return table


def normal_quantile_table_lookup(
    index: int,
    *,
    config: KeyScheduleConfig = DEFAULT_CONFIG,
) -> float:
    """按冻结表的 IEEE-754 binary32 原值查询一个 midpoint normal quantile。"""

    _validate_config(config)
    if type(index) is not int or not 0 <= index < _NORMAL_TABLE_ENTRY_COUNT:
        raise KeyScheduleError(
            "normal quantile table index must be an integer in [0,1048575]"
        )
    return unpack_from(">f", _load_normal_quantile_table(), index * 4)[0]


def _uniform_values(
    domain_digest: bytes,
    element_count: int,
    block_count: int,
) -> tuple[float, ...]:
    values: list[float] = []
    for block in _counter_blocks(domain_digest, block_count):
        for offset in (0, 8, 16, 24):
            word = int.from_bytes(block[offset : offset + 8], "big", signed=False)
            mantissa = word >> 11
            values.append(_uniform_mantissa_to_float32(mantissa))
            if len(values) == element_count:
                return tuple(values)
    raise KeyScheduleError("uniform counter stream ended unexpectedly")


def _gaussian_indices(
    domain_digest: bytes,
    element_count: int,
    block_count: int,
) -> tuple[int, ...]:
    indices: list[int] = []
    buffer = 0
    available_bits = 0
    for block in _counter_blocks(domain_digest, block_count):
        buffer = (buffer << 256) | int.from_bytes(block, "big", signed=False)
        available_bits += 256
        while available_bits >= 20 and len(indices) < element_count:
            available_bits -= 20
            indices.append((buffer >> available_bits) & (_NORMAL_TABLE_ENTRY_COUNT - 1))
            buffer &= (1 << available_bits) - 1 if available_bits else 0
        if len(indices) == element_count:
            return tuple(indices)
    raise KeyScheduleError("Gaussian counter stream ended unexpectedly")


def _result(
    *,
    distribution: Distribution,
    shape: tuple[int, ...],
    values: tuple[float, ...],
    domain_digest: bytes,
    config: KeyScheduleConfig,
    quantile_indices_random: tuple[int, ...] | None,
) -> KeyStreamResult:
    encoded_values = b"".join(pack(">f", value) for value in values)
    return KeyStreamResult(
        candidate_id=CANDIDATE_ID,
        distribution=distribution,
        shape=shape,
        values=values,
        domain_digest=domain_digest.hex(),
        config_digest=config.config_digest,
        values_float32_be_sha256=sha256(encoded_values).hexdigest(),
        quantile_indices_random=quantile_indices_random,
    )


def _derive_stream(
    *,
    key_material: str,
    shape: Sequence[int],
    domain_fields: dict[str, object],
    distribution: Distribution,
    expected_public_domain: bool,
    key_role: str,
    root_key_public_digest: str,
    wrong_key_index: int | None,
    config: KeyScheduleConfig,
) -> KeyStreamResult:
    config = _validate_config(config)
    key_material = _validate_text(key_material, "key_material")
    normalized_shape = _normalize_shape(shape)
    validated_domain, is_public_domain = _validate_domain_fields(domain_fields)
    if is_public_domain is not expected_public_domain:
        raise KeyScheduleError("secret and public-noise responsibility domains cannot be mixed")
    element_count = prod(normalized_shape)
    block_count = _required_block_count(element_count, distribution)
    digest = _domain_digest(
        key_material,
        validated_domain,
        normalized_shape,
        distribution,
        key_role=key_role,
        root_key_public_digest=root_key_public_digest,
        wrong_key_index=wrong_key_index,
    )

    if distribution == "uniform":
        values = _uniform_values(digest, element_count, block_count)
        return _result(
            distribution=distribution,
            shape=normalized_shape,
            values=values,
            domain_digest=digest,
            config=config,
            quantile_indices_random=None,
        )

    indices = _gaussian_indices(digest, element_count, block_count)
    table = _load_normal_quantile_table()
    values = tuple(unpack_from(">f", table, index * 4)[0] for index in indices)
    return _result(
        distribution=distribution,
        shape=normalized_shape,
        values=values,
        domain_digest=digest,
        config=config,
        quantile_indices_random=indices,
    )


def key_schedule_sha256_counter(
    root_key_text: str,
    domain_fields: dict[str, object],
    shape: Sequence[int],
    *,
    distribution: Distribution = "gaussian",
    config: KeyScheduleConfig = DEFAULT_CONFIG,
) -> KeyStreamResult:
    """为注册 root key 派生冻结职责域的 CPU float32 流。"""

    root_key_text = _validate_text(root_key_text, "root_key_text")
    root_identity = identify_root_key(root_key_text, config=config)
    return _derive_stream(
        key_material=root_key_text,
        shape=shape,
        domain_fields=domain_fields,
        distribution=distribution,
        expected_public_domain=False,
        key_role="registered",
        root_key_public_digest=root_identity.root_key_public_digest,
        wrong_key_index=None,
        config=config,
    )


def derive_wrong_key_stream(
    wrong_key_material: DerivedWrongKeyMaterial,
    domain_fields: dict[str, object],
    shape: Sequence[int],
    *,
    distribution: Distribution = "gaussian",
    config: KeyScheduleConfig = DEFAULT_CONFIG,
) -> KeyStreamResult:
    """让预登记 wrong key 走与注册 key 完全相同的秘密职责域。"""

    if type(wrong_key_material) is not DerivedWrongKeyMaterial:
        raise KeyScheduleError("wrong_key_material must come from derive_wrong_key_material")
    material_text = _derive_wrong_key_material_text(
        wrong_key_material.registered_root_key_public_digest,
        wrong_key_material.wrong_key_index,
    )
    return _derive_stream(
        key_material=material_text,
        shape=shape,
        domain_fields=domain_fields,
        distribution=distribution,
        expected_public_domain=False,
        key_role="wrong",
        root_key_public_digest=(
            wrong_key_material.registered_root_key_public_digest
        ),
        wrong_key_index=wrong_key_material.wrong_key_index,
        config=config,
    )


def derive_internal_lf_decoy_stream(
    internal_decoy_material: DerivedInternalLfDecoyMaterial,
    domain_fields: dict[str, object],
    shape: Sequence[int],
    *,
    distribution: Distribution = "gaussian",
    config: KeyScheduleConfig = DEFAULT_CONFIG,
) -> KeyStreamResult:
    """Derive an LF internal decoy in its separate material/domain authority."""

    if type(internal_decoy_material) is not DerivedInternalLfDecoyMaterial:
        raise KeyScheduleError(
            "internal_decoy_material must come from derive_internal_lf_decoy_material"
        )
    if domain_fields.get("candidate_id") != internal_decoy_material.lf_candidate_id:
        raise KeyScheduleError("LF internal decoy candidate/domain mismatch")
    return _derive_stream(
        key_material=internal_decoy_material.material_text,
        shape=shape,
        domain_fields=domain_fields,
        distribution=distribution,
        expected_public_domain=False,
        key_role="internal_decoy",
        root_key_public_digest=internal_decoy_material.registered_root_key_public_digest,
        wrong_key_index=internal_decoy_material.internal_decoy_index,
        config=config,
    )


def derive_public_noise_stream(
    domain_fields: dict[str, object],
    shape: Sequence[int],
    *,
    distribution: Distribution = "gaussian",
    config: KeyScheduleConfig = DEFAULT_CONFIG,
) -> KeyStreamResult:
    """从固定公共材料派生与任何 secret root 无关的公开噪声。"""

    public_identity = identify_root_key(
        PUBLIC_NOISE_KEY_MATERIAL,
        config=config,
    )
    return _derive_stream(
        key_material=PUBLIC_NOISE_KEY_MATERIAL,
        shape=shape,
        domain_fields=domain_fields,
        distribution=distribution,
        expected_public_domain=True,
        key_role="public",
        root_key_public_digest=public_identity.root_key_public_digest,
        wrong_key_index=None,
        config=config,
    )
