"""
File purpose: Cryptographic generation attestation watermark 模块入口。
Module type: General module

公开接口：
    AttestationStatement — 生成事件语句数据类
    build_attestation_statement — 构造生成事件语句
    compute_attestation_digest  — 计算 d_A（statement 的 canonical SHA256）
    derive_attestation_keys     — 从 d_A 派生四类密钥（LF/HF/GEO/TR）
    compute_prompt_commit       — HMAC 承诺 prompt
    compute_seed_commit         — HMAC 承诺 seed
    compute_lf_attestation_payload  — 生成 LF 主通道 attestation payload
    compute_trajectory_commit   — 计算生成轨迹承诺
"""

from .attestation_statement import (
    AttestationStatement,
    build_attestation_statement,
    compute_attestation_digest,
)
from .commitments import (
    compute_prompt_commit,
    compute_seed_commit,
)
from .key_derivation import derive_attestation_keys
from .trajectory_commit import compute_trajectory_commit

__all__ = [
    "AttestationStatement",
    "build_attestation_statement",
    "compute_attestation_digest",
    "derive_attestation_keys",
    "compute_prompt_commit",
    "compute_seed_commit",
    "compute_trajectory_commit",
]
