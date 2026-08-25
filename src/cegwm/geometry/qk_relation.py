"""Keyed numeric relation construction for image-derived Q/K tensors."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from cegwm.geometry.types import QKRelation
from cegwm.shared import prg_bytes, prg_normal

_SUBKEY_DOMAIN = "geometry-v1/subkey/qk-corner-sync"


def _geometry_subkey(detection_key: str | bytes | bytearray | memoryview) -> bytes:
    """Derive geometry-only key material without sharing another method domain."""

    return prg_bytes(detection_key, _SUBKEY_DOMAIN, 32)


def keyed_qk_relation(
    query: ArrayLike,
    key: ArrayLike,
    detection_key: str | bytes | bytearray | memoryview,
    *,
    comparison_key: str | bytes | bytearray | memoryview | None = None,
) -> QKRelation:
    """Construct a real Q/K Gram relation and a domain-separated projection.

    ``query`` and ``key`` are image-derived tensors flattened to token rows.
    No embedding-side tensors or attack parameters are accepted by this API.
    """

    q = np.asarray(query, dtype=np.float64)
    k = np.asarray(key, dtype=np.float64)
    if q.ndim != 2 or k.ndim != 2 or q.shape != k.shape or q.shape[0] < 2 or q.shape[1] < 1:
        raise ValueError("query and key must be equal rank-2 arrays with at least two tokens")
    if not np.isfinite(q).all() or not np.isfinite(k).all():
        raise ValueError("query and key must be finite")

    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), np.finfo(np.float64).eps)
    k = k / np.maximum(np.linalg.norm(k, axis=1, keepdims=True), np.finfo(np.float64).eps)
    relation = (q @ k.T) / np.sqrt(float(q.shape[1]))
    subkey = _geometry_subkey(detection_key)
    carrier = prg_normal(subkey, "geometry-v1/qk-relation-projection", relation.shape, dtype=np.float64)
    projection = float(np.mean(relation * carrier))
    diagonal = np.diag(relation)
    off_diagonal = relation[~np.eye(relation.shape[0], dtype=bool)]
    gap = float(np.mean(diagonal) - np.mean(off_diagonal))
    coverage = float(np.mean(np.linalg.norm(np.asarray(query), axis=1) > 0.0))
    wrong_key_margin = 0.0
    if comparison_key is not None:
        wrong = prg_normal(
            _geometry_subkey(comparison_key),
            "geometry-v1/qk-relation-projection",
            relation.shape,
            dtype=np.float64,
        )
        wrong_key_margin = float(abs(projection) - abs(np.mean(relation * wrong)))
    return QKRelation(relation, projection, coverage, gap, wrong_key_margin)
