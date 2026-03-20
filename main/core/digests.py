"""
digest 规范化唯一入口

功能说明：
- 规范化所有 YAML 事实源的 digest 计算，禁止其他模块直接进行 digest 相关操作。
- 计算 canonical JSON 的 SHA256 作为对象的规范化 digest，确保同一对象无论输入格式如何变化都能得到相同的 digest。
- 计算文件的 SHA256 作为文件内容的 digest，确保文件内容的任何变化都能被检测到。
- 计算绑定 digest，结合版本和组件 digest 生成一个综合的 digest，用于记录和追踪。
- 包含详细的输入验证和错误处理，确保健壮性和可维护性。
- 额外 digest 能力需通过版本化追加接入，且不得改变既有计算口径与可复算性。
"""

import json
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Union, cast

from .errors import DigestCanonicalizationError


def normalize_for_digest(obj: Any, path: str = "<root>") -> Any:
    """
    功能：递归把对象归一为 JSON-like 类型并严格校验。
    
    Recursively normalize object to JSON-compatible types and validate strictly.
    Allowed types: None, bool, int, float, str, list, dict.
    Dict keys must be str; otherwise returns the object as-is after deep validation.
    
    Args:
        obj: Object to normalize.
        path: Current field path for error reporting.
    
    Returns:
        Normalized object (already JSON-compatible, no transformation needed).
    
    Raises:
        DigestCanonicalizationError: If unsupported type or non-str dict key found.
    """
    # 基本类型直接返回。
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    
    # 处理 dict：必须递归检查，且所有 key 必须是 str。
    if isinstance(obj, dict):
        mapping = cast(Dict[Any, Any], obj)
        for raw_key, raw_value in mapping.items():
            key_obj: Any = raw_key
            if not isinstance(key_obj, str):
                raise DigestCanonicalizationError(
                    f"Dict key at {path} must be str, got {type(key_obj).__name__}: {repr(key_obj)}",
                    field_path=path,
                    offending_type=f"non-str-dict-key({type(key_obj).__name__})"
                )
            normalize_for_digest(raw_value, f"{path}.{key_obj}")
        return cast(Dict[str, Any], obj)
    
    # 处理 list：保持顺序，递归检查元素。
    if isinstance(obj, list):
        items = cast(List[Any], obj)
        for idx, item in enumerate(items):
            normalize_for_digest(item, f"{path}[{idx}]")
        return items
    
    # 其他类型一律拒绝。
    raise DigestCanonicalizationError(
        f"Non-JSON-serializable type at {path}: {type(obj).__name__}",
        field_path=path,
        offending_type=type(obj).__name__
    )


def canonical_json_dumps(obj: Any) -> bytes:
    """
    功能：规范化 JSON 序列化，固定参数。
    
    Canonical JSON serialization with fixed parameters for digest computation.
    Parameters: sort_keys=True, separators=(",", ":"), ensure_ascii=False.
    First normalizes input to JSON-compatible types.
    
    Args:
        obj: JSON-serializable object (dict/list/str/int/float/bool/None).
    
    Returns:
        UTF-8 encoded JSON bytes.
    
    Raises:
        DigestCanonicalizationError: If obj contains non-JSON-serializable types.
    """
    # 归一化输入，检查类型和 dict key。
    normalize_for_digest(obj)
    
    # JSON 序列化。
    try:
        json_str = json.dumps(
            obj,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False
        )
        return json_str.encode("utf-8")
    except (TypeError, ValueError) as e:
        raise DigestCanonicalizationError(
            f"JSON serialization failed: {e}",
            offending_type=type(obj).__name__
        ) from e


def canonical_sha256(obj: Any) -> str:
    """
    功能：计算对象的规范化 sha256。
    
    Compute SHA256 of canonical JSON serialization.
    
    Args:
        obj: JSON-serializable object.
    
    Returns:
        Lowercase hex SHA256 digest string.
    """
    canonical_bytes = canonical_json_dumps(obj)
    return hashlib.sha256(canonical_bytes).hexdigest()


def file_sha256(path: Union[str, Path]) -> str:
    """
    功能：计算文件的 sha256。
    
    Compute SHA256 of raw file bytes.
    
    Args:
        path: File path.
    
    Returns:
        Lowercase hex SHA256 digest string.
    """
    path = Path(path)
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def semantic_digest(obj: Any) -> str:
    """
    功能：计算语义 digest，当前等于 canonical_sha256。
    
    Compute semantic digest (currently equivalent to canonical_sha256).
    Separate function for future extensibility.
    
    Args:
        obj: JSON-serializable object.
    
    Returns:
        Lowercase hex digest string.
    """
    return canonical_sha256(obj)


def bound_digest(
    version: str,
    semantic_digest_value: str,
    file_sha256_value: str,
    canon_sha256_value: str
) -> str:
    """
    功能：计算绑定 digest，固定公式确保可复算。
    
    Compute bound digest from version and component digests.
    Formula: SHA256 of canonical JSON of {version, semantic_digest, file_sha256, canon_sha256}.
    
    Args:
        version: Version string from YAML.
        semantic_digest_value: Semantic digest of object.
        file_sha256_value: SHA256 of raw file bytes.
        canon_sha256_value: SHA256 of canonical JSON.
    
    Returns:
        Lowercase hex bound digest string.
    """
    binding_obj = {
        "version": version,
        "semantic_digest": semantic_digest_value,
        "file_sha256": file_sha256_value,
        "canon_sha256": canon_sha256_value
    }
    return canonical_sha256(binding_obj)
