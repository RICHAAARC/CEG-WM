"""冻结的 CEG-WM SHA-256 counter 密钥与伪随机流协议。

本模块独占 root-key 编码、职责域验证、wrong-key/public-noise 派生以及
uniform/Gaussian float32 物化。它不保存原始 root key，也不调用设备 RNG。
"""

from __future__ import annotations

from base64 import b64decode
from binascii import Error as Base64Error
from dataclasses import dataclass, field
from functools import lru_cache
from hashlib import sha256
import json
from math import prod
from pathlib import Path
from struct import pack, unpack, unpack_from
from typing import Literal, Sequence

CANDIDATE_ID = "key_schedule_sha256_counter"
KEYED_PRG_VERSION = "sha256_counter_normal_icdf_table20_float32"
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
    """仅供内存调用的预登记 wrong-key 派生材料。"""

    wrong_key_index: int
    registered_root_key_public_digest: str
    material_text: str = field(repr=False)


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
    quantile_indices: tuple[int, ...] | None


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
                "model_revision",
                "tensor_role",
            },
        )
        _expect_non_empty_text(domain_fields, "model_revision")
        _expect_literal(domain_fields, "tensor_role", "base_gaussian")
        return dict(domain_fields), False

    if identity == ("lf_low_pass", "carrier_template", "lf_carrier"):
        _expect_exact_fields(
            domain_fields,
            {
                "candidate_id",
                "operator",
                "responsibility_domain",
                "model_revision",
                "tensor_role",
            },
        )
        _expect_non_empty_text(domain_fields, "model_revision")
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
                "model_revision",
                "layer_name",
                "token_count",
                "tensor_role",
            },
        )
        _expect_non_empty_text(domain_fields, "model_revision")
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
                "model_revision",
                "schedule_index",
                "conditioning_protocol",
                "tensor_role",
            },
        )
        _expect_non_empty_text(domain_fields, "model_revision")
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
                "operator",
                "responsibility_domain",
                "model_revision",
                "sample_index",
                "tensor_role",
            },
        )
        _expect_non_empty_text(domain_fields, "model_revision")
        sample_index = domain_fields["sample_index"]
        if type(sample_index) is not int or sample_index < 0:
            raise KeyScheduleError("domain field sample_index must be a non-negative integer")
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
    payload = {
        "candidate_id": CANDIDATE_ID,
        "derivation_role": "geometry_and_content_wrong_key",
        "registered_root_key_public_digest": digest,
        "wrong_key_index": wrong_key_index,
    }
    material_text = "ceg-wm-wrong-key:" + sha256(stable_json_utf8(payload)).hexdigest()
    return DerivedWrongKeyMaterial(
        wrong_key_index=wrong_key_index,
        registered_root_key_public_digest=digest,
        material_text=material_text,
    )


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
) -> bytes:
    payload = {
        "keyed_prg_version": KEYED_PRG_VERSION,
        "key_material": key_material,
        "domain_fields": domain_fields,
        "shape": list(shape),
    }
    return sha256(stable_json_utf8(payload)).digest()


def _counter_blocks(domain_digest: bytes, block_count: int):
    for counter in range(block_count):
        yield sha256(domain_digest + counter.to_bytes(16, "big", signed=False)).digest()


def _float32(value: float) -> float:
    return unpack(">f", pack(">f", value))[0]


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


def _uniform_values(
    domain_digest: bytes,
    element_count: int,
    block_count: int,
) -> tuple[float, ...]:
    values: list[float] = []
    denominator = (1 << 53) + 2
    for block in _counter_blocks(domain_digest, block_count):
        for offset in (0, 8, 16, 24):
            word = int.from_bytes(block[offset : offset + 8], "big", signed=False)
            mantissa = word >> 11
            values.append(_float32((mantissa + 1) / denominator))
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
    quantile_indices: tuple[int, ...] | None,
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
        quantile_indices=quantile_indices,
    )


def _derive_stream(
    *,
    key_material: str,
    shape: Sequence[int],
    domain_fields: dict[str, object],
    distribution: Distribution,
    expected_public_domain: bool,
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
    digest = _domain_digest(key_material, validated_domain, normalized_shape)

    if distribution == "uniform":
        values = _uniform_values(digest, element_count, block_count)
        return _result(
            distribution=distribution,
            shape=normalized_shape,
            values=values,
            domain_digest=digest,
            config=config,
            quantile_indices=None,
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
        quantile_indices=indices,
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
    return _derive_stream(
        key_material=root_key_text,
        shape=shape,
        domain_fields=domain_fields,
        distribution=distribution,
        expected_public_domain=False,
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
    return _derive_stream(
        key_material=wrong_key_material.material_text,
        shape=shape,
        domain_fields=domain_fields,
        distribution=distribution,
        expected_public_domain=False,
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

    return _derive_stream(
        key_material=PUBLIC_NOISE_KEY_MATERIAL,
        shape=shape,
        domain_fields=domain_fields,
        distribution=distribution,
        expected_public_domain=True,
        config=config,
    )
