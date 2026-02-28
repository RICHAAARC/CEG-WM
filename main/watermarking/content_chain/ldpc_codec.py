"""
文件用途：为 LF coder 提供可复现的 LDPC 编码与软判决译码工具。
Module type: Semi-general module
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from main.core import digests


LDPC_CODEC_ID = "ldpc_codec_v1"


class SparseLDPC:
    """
    功能：提供稀疏 LDPC 的确定性构造、编码与 BP 解码。

    Sparse LDPC codec with deterministic parity-check construction,
    systematic-like encoding under fixed dimensions, and min-sum BP decoding.

    Args:
        n: Codeword length.
        k: Message length.
        col_weight: Target column weight for parity-check matrix.
        seed: Deterministic seed integer.

    Returns:
        None.

    Raises:
        ValueError: If dimensions or parameters are invalid.
    """

    def __init__(self, n: int, k: int, col_weight: int, seed: int) -> None:
        if not isinstance(n, int) or n <= 1:
            # 码长非法，无法构造稳定校验矩阵。
            raise ValueError("n must be int > 1")
        if not isinstance(k, int) or k <= 0:
            # 消息长度非法，无法编码。
            raise ValueError("k must be positive int")
        if k >= n:
            # 需要冗余位，k 必须严格小于 n。
            raise ValueError("k must be smaller than n")
        if not isinstance(col_weight, int) or col_weight <= 0:
            # 列重非法会导致 H 矩阵退化。
            raise ValueError("col_weight must be positive int")
        if not isinstance(seed, int):
            # seed 必须为整数，保证可重复性。
            raise ValueError("seed must be int")

        self.n = n
        self.k = k
        self.m = n - k
        self.col_weight = col_weight
        self.seed = seed
        self.h_matrix = self.build_parity_check_matrix(n=n, k=k, col_weight=col_weight, seed=seed)
        self._h_message = self.h_matrix[:, : self.k] % 2
        self._h_parity = self.h_matrix[:, self.k :] % 2
        self._h_parity_inverse = _invert_binary_matrix(self._h_parity)

    @staticmethod
    def build_parity_check_matrix(n: int, k: int, col_weight: int, seed: int) -> np.ndarray:
        """
        功能：构造确定性的稀疏校验矩阵 H。

        Build deterministic sparse parity-check matrix H with dimensions (n-k, n).

        Args:
            n: Codeword length.
            k: Message length.
            col_weight: Target column weight in message-part columns.
            seed: Deterministic RNG seed.

        Returns:
            Binary parity-check matrix with shape [n-k, n].

        Raises:
            ValueError: If inputs are invalid.
        """
        if not isinstance(n, int) or not isinstance(k, int) or not isinstance(col_weight, int):
            raise ValueError("n, k, col_weight must be int")
        if n <= 1 or k <= 0 or k >= n:
            raise ValueError("Require n > 1 and 0 < k < n")
        if col_weight <= 0:
            raise ValueError("col_weight must be positive")
        if not isinstance(seed, int):
            raise ValueError("seed must be int")

        row_count = n - k
        rng = np.random.default_rng(seed)

        h_left = np.zeros((row_count, k), dtype=np.int8)
        effective_weight = min(col_weight, row_count)
        for col_idx in range(k):
            row_indices = rng.choice(row_count, size=effective_weight, replace=False)
            h_left[row_indices, col_idx] = 1

        h_right = np.eye(row_count, dtype=np.int8)
        h_matrix = np.concatenate([h_left, h_right], axis=1)
        return h_matrix % 2

    def encode(self, message_bits: List[int]) -> List[int]:
        """
        功能：将消息位编码为合法 LDPC 码字。

        Encode message bits to a valid codeword that satisfies H * c^T = 0 over GF(2).

        Args:
            message_bits: Message bits in {0, 1} or {-1, +1}.

        Returns:
            Codeword bits in {-1, +1} with length n.

        Raises:
            ValueError: If input length or values are invalid.
        """
        message_binary = _normalize_bits_to_binary(message_bits)
        if len(message_binary) != self.k:
            # 消息长度与 k 不一致，无法编码。
            raise ValueError(f"message_bits length must be {self.k}")

        msg_vec = np.asarray(message_binary, dtype=np.int8)
        rhs = (self._h_message @ msg_vec) % 2
        parity = (self._h_parity_inverse @ rhs) % 2
        codeword_binary = np.concatenate([msg_vec, parity.astype(np.int8)], axis=0)
        return _binary_to_bipolar(codeword_binary.tolist())

    def decode_bp(self, llr_input: List[float], max_iter: int) -> Tuple[List[int], bool, int, int]:
        """
        功能：执行基于 min-sum 的 BP 软判决解码。

        Decode LLR evidence via min-sum belief propagation on Tanner graph.

        Args:
            llr_input: Channel LLR values, length n.
            max_iter: Maximum number of BP iterations.

        Returns:
            Tuple of (decoded_bits_bipolar, converged, iteration_count, syndrome_weight).

        Raises:
            ValueError: If inputs are invalid.
        """
        if not isinstance(llr_input, list) or len(llr_input) != self.n:
            raise ValueError(f"llr_input must be list with length {self.n}")
        if not isinstance(max_iter, int) or max_iter <= 0:
            raise ValueError("max_iter must be positive int")

        llr = np.asarray(llr_input, dtype=np.float64)
        h = self.h_matrix
        row_count, col_count = h.shape

        check_neighbors = [np.where(h[r] == 1)[0] for r in range(row_count)]
        var_neighbors = [np.where(h[:, c] == 1)[0] for c in range(col_count)]

        q = np.zeros((row_count, col_count), dtype=np.float64)
        r = np.zeros((row_count, col_count), dtype=np.float64)

        for row_idx in range(row_count):
            cols = check_neighbors[row_idx]
            q[row_idx, cols] = llr[cols]

        converged = False
        iteration_count = 0
        syndrome_weight = row_count
        posterior = llr.copy()

        for iteration_idx in range(max_iter):
            for row_idx in range(row_count):
                cols = check_neighbors[row_idx]
                if cols.size == 0:
                    continue
                incoming = q[row_idx, cols]
                signs = np.sign(incoming)
                signs[signs == 0.0] = 1.0
                abs_vals = np.abs(incoming)
                total_sign = np.prod(signs)

                if cols.size == 1:
                    r[row_idx, cols[0]] = total_sign * abs_vals[0]
                    continue

                min_pos = int(np.argmin(abs_vals))
                min_val = abs_vals[min_pos]
                second_min = np.partition(abs_vals, 1)[1]

                for local_idx, col_idx in enumerate(cols):
                    sign_excluding = total_sign * signs[local_idx]
                    magnitude = second_min if local_idx == min_pos else min_val
                    r[row_idx, col_idx] = sign_excluding * magnitude

            posterior = llr.copy()
            for col_idx in range(col_count):
                rows = var_neighbors[col_idx]
                if rows.size == 0:
                    continue
                posterior[col_idx] = llr[col_idx] + np.sum(r[rows, col_idx])
                for row_idx in rows:
                    q[row_idx, col_idx] = posterior[col_idx] - r[row_idx, col_idx]

            hard_binary = (posterior < 0.0).astype(np.int8)
            syndrome = (h @ hard_binary) % 2
            syndrome_weight = int(np.sum(syndrome))
            iteration_count = iteration_idx + 1
            if syndrome_weight == 0:
                converged = True
                break

        decoded_bits = _binary_to_bipolar((posterior < 0.0).astype(np.int8).tolist())
        return decoded_bits, converged, iteration_count, syndrome_weight


def build_ldpc_spec(message_length: int, ecc_sparsity: int, seed_key: str) -> Dict[str, Any]:
    """
    功能：构建确定性的 parity-check 规格。

    Build a deterministic LDPC parity-check specification.

    Args:
        message_length: Number of message bits.
        ecc_sparsity: Number of non-zero entries per check row.
        seed_key: Seed material used to derive deterministic structure.

    Returns:
        Dictionary with parity-check matrix and digest anchors.

    Raises:
        ValueError: If input arguments are invalid.
    """
    if not isinstance(message_length, int) or message_length <= 0:
        raise ValueError("message_length must be positive int")
    if not isinstance(ecc_sparsity, int) or ecc_sparsity <= 0:
        raise ValueError("ecc_sparsity must be positive int")
    if not isinstance(seed_key, str) or not seed_key:
        raise ValueError("seed_key must be non-empty str")

    block_length = max(message_length + 8, int(round(message_length * 1.5)))
    row_count = block_length - message_length
    row_weight = min(ecc_sparsity, row_count)
    seed_digest = hashlib.sha256(seed_key.encode("utf-8")).hexdigest()
    seed_int = int(seed_digest[:16], 16) % (2 ** 31 - 1)

    h_matrix = SparseLDPC.build_parity_check_matrix(
        n=block_length,
        k=message_length,
        col_weight=row_weight,
        seed=seed_int,
    )

    parity_list = h_matrix.tolist()
    parity_digest = digests.canonical_sha256(
        {
            "n": block_length,
            "k": message_length,
            "row_count": row_count,
            "row_weight": row_weight,
            "parity_matrix": parity_list,
        }
    )
    return {
        "codec_id": LDPC_CODEC_ID,
        "n": block_length,
        "k": message_length,
        "message_length": message_length,
        "row_count": row_count,
        "row_weight": row_weight,
        "parity_matrix": parity_list,
        "parity_check_digest": parity_digest,
        "seed_digest": seed_digest,
        "seed_int": seed_int,
    }


def encode_message_bits(message_bits: List[int], ldpc_spec: Dict[str, Any]) -> List[int]:
    """
    功能：在固定消息长度约束下返回 LDPC 绑定的编码位。

    Return code bits bound to LDPC spec under fixed message-length constraints.

    Args:
        message_bits: Message bits encoded as ±1 values.
        ldpc_spec: LDPC specification returned by build_ldpc_spec.

    Returns:
        Encoded codeword bits in ±1 representation.

    Raises:
        ValueError: If input shapes are inconsistent.
    """
    if not isinstance(message_bits, list) or not message_bits:
        raise ValueError("message_bits must be non-empty list")
    if not isinstance(ldpc_spec, dict):
        raise ValueError("ldpc_spec must be dict")
    expected_len = int(ldpc_spec.get("message_length", 0))
    if expected_len != len(message_bits):
        raise ValueError("message_bits length must match ldpc_spec.message_length")

    n = int(ldpc_spec.get("n", 0))
    k = int(ldpc_spec.get("k", expected_len))
    row_weight = int(ldpc_spec.get("row_weight", 1))
    seed_int = ldpc_spec.get("seed_int")
    if not isinstance(seed_int, int):
        seed_digest = str(ldpc_spec.get("seed_digest", ""))
        if not seed_digest:
            raise ValueError("ldpc_spec must include seed_int or seed_digest")
        seed_int = int(seed_digest[:16], 16) % (2 ** 31 - 1)

    codec = SparseLDPC(n=n, k=k, col_weight=row_weight, seed=seed_int)
    return codec.encode(message_bits)


def decode_soft_llr(llr_values: List[float], ldpc_spec: Dict[str, Any], max_iterations: int) -> Dict[str, Any]:
    """
    功能：基于 parity-check 的软判决迭代译码。

    Decode soft LLR values with an iterative parity-check constrained procedure.

    Args:
        llr_values: Soft evidence values where larger indicates stronger +1 evidence.
        ldpc_spec: LDPC specification returned by build_ldpc_spec.
        max_iterations: Maximum number of decoding iterations.

    Returns:
        Dict containing decoded_bits, bp_converged, bp_iteration_count, and syndrome_weight.

    Raises:
        ValueError: If arguments are invalid.
    """
    if not isinstance(llr_values, list) or not llr_values:
        raise ValueError("llr_values must be non-empty list")
    if not isinstance(ldpc_spec, dict):
        raise ValueError("ldpc_spec must be dict")
    if not isinstance(max_iterations, int) or max_iterations <= 0:
        raise ValueError("max_iterations must be positive int")

    n = int(ldpc_spec.get("n", len(llr_values)))
    k = int(ldpc_spec.get("k", int(ldpc_spec.get("message_length", 0))))
    row_weight = int(ldpc_spec.get("row_weight", 1))
    seed_int = ldpc_spec.get("seed_int")
    if not isinstance(seed_int, int):
        seed_digest = str(ldpc_spec.get("seed_digest", ""))
        if not seed_digest:
            raise ValueError("ldpc_spec must include seed_int or seed_digest")
        seed_int = int(seed_digest[:16], 16) % (2 ** 31 - 1)

    if len(llr_values) < n:
        # 输入维度不足，无法完成 BP。
        raise ValueError(f"llr_values length must be >= n ({n})")

    codec = SparseLDPC(n=n, k=k, col_weight=row_weight, seed=seed_int)
    decoded_bits, converged, iteration_count, syndrome_weight = codec.decode_bp(
        llr_input=[float(v) for v in llr_values[:n]],
        max_iter=max_iterations,
    )
    return {
        "decoded_bits": decoded_bits,
        "bp_converged": converged,
        "bp_iteration_count": iteration_count,
        "syndrome_weight": syndrome_weight,
    }


def _normalize_bits_to_binary(bits: List[int]) -> List[int]:
    """
    功能：将比特序列统一转换为 GF(2) 二值表示。

    Normalize input bits from {-1,+1}/{0,1} to {0,1}.

    Args:
        bits: Bit values in supported domains.

    Returns:
        Binary bits in {0, 1}.

    Raises:
        ValueError: If unsupported bit values appear.
    """
    if not isinstance(bits, list) or not bits:
        raise ValueError("bits must be non-empty list")
    normalized: List[int] = []
    for bit in bits:
        value = int(bit)
        if value in (0, 1):
            normalized.append(value)
            continue
        if value in (-1, 1):
            normalized.append(0 if value < 0 else 1)
            continue
        raise ValueError(f"Unsupported bit value: {bit}")
    return normalized


def _binary_to_bipolar(bits: List[int]) -> List[int]:
    """
    功能：将 GF(2) 比特转换为双极性表示。

    Convert binary bits in {0,1} to bipolar bits in {-1,+1}.

    Args:
        bits: Binary bits list.

    Returns:
        Bipolar bit list.

    Raises:
        ValueError: If input contains non-binary values.
    """
    result: List[int] = []
    for bit in bits:
        value = int(bit)
        if value == 0:
            result.append(-1)
            continue
        if value == 1:
            result.append(1)
            continue
        raise ValueError(f"binary bit must be 0 or 1, got {bit}")
    return result


def _invert_binary_matrix(matrix: np.ndarray) -> np.ndarray:
    """
    功能：在 GF(2) 上求逆矩阵。

    Compute matrix inverse over GF(2) using Gauss-Jordan elimination.

    Args:
        matrix: Square binary matrix.

    Returns:
        Inverse matrix in GF(2).

    Raises:
        ValueError: If matrix is non-square or singular.
    """
    if not isinstance(matrix, np.ndarray):
        raise ValueError("matrix must be numpy array")
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")

    n = matrix.shape[0]
    left = (matrix.astype(np.int8) % 2).copy()
    right = np.eye(n, dtype=np.int8)

    for col_idx in range(n):
        pivot_row: Optional[int] = None
        for row_idx in range(col_idx, n):
            if left[row_idx, col_idx] == 1:
                pivot_row = row_idx
                break
        if pivot_row is None:
            # 奇异矩阵在 GF(2) 上不可逆。
            raise ValueError("matrix is singular in GF(2)")
        if pivot_row != col_idx:
            left[[col_idx, pivot_row]] = left[[pivot_row, col_idx]]
            right[[col_idx, pivot_row]] = right[[pivot_row, col_idx]]

        for row_idx in range(n):
            if row_idx == col_idx:
                continue
            if left[row_idx, col_idx] == 1:
                left[row_idx, :] = (left[row_idx, :] + left[col_idx, :]) % 2
                right[row_idx, :] = (right[row_idx, :] + right[col_idx, :]) % 2

    return right % 2
