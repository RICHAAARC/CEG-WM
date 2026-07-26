"""提供稳定摘要能力, 用于记录可重建输入和配置。"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def stable_json_dumps(value: Any) -> str:
    """将 JSON 兼容对象转为稳定字符串, 便于跨平台生成一致摘要。"""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def build_stable_digest(value: Any) -> str:
    """根据 JSON 兼容对象生成 SHA-256 摘要。"""
    encoded = stable_json_dumps(value).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
