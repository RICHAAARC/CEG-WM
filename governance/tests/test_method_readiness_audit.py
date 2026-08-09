"""验证 method-readiness 结构门和独立语义复核绑定。

这些临时 fixture 只验证审计机制，不代表 CEG-WM 方法或科学效果已经成立。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from governance.harness.audits.audit_method_readiness import run_audit


RESPONSIBILITIES = {
    "key_schedule": "root_key_derivation_and_prg",
    "content_router": "content_observation_and_adaptive_routing",
    "lf_carrier": "low_frequency_carrier_template_and_write_direction",
    "hf_carrier": "high_frequency_carrier_template_and_write_direction",
    "content_embedder": "lf_hf_combined_embedding_and_total_budget",
    "lf_detector": "low_frequency_blind_scoring",
    "hf_detector": "high_frequency_direct_scoring",
    "content_detector": "lf_hf_score_standardization_and_content_detection",
    "qk_geometry_sync": "keyed_qk_geometry_synchronization_and_relation_observation",
    "geometric_transform_estimator": "blind_bounded_geometric_transform_estimation",
    "geometry_reliability": "independent_geometry_reliability_conjunction",
    "image_rectifier": "image_coordinate_rectification",
    "conditional_recovery_decision": "conditional_same_detector_recovery",
}
COMPONENT_PATHS = {
    "key_schedule": "main/shared/key_schedule.py",
    "content_router": "main/content_chain/routing.py",
    "lf_carrier": "main/content_chain/lf_carrier.py",
    "hf_carrier": "main/content_chain/hf_carrier.py",
    "content_embedder": "main/content_chain/embedder.py",
    "lf_detector": "main/content_chain/lf_detector.py",
    "hf_detector": "main/content_chain/hf_detector.py",
    "content_detector": "main/content_chain/detector.py",
    "qk_geometry_sync": "main/geometry_chain/qk_sync.py",
    "geometric_transform_estimator": "main/geometry_chain/transform_estimator.py",
    "geometry_reliability": "main/geometry_chain/reliability.py",
    "image_rectifier": "main/geometry_chain/rectifier.py",
    "conditional_recovery_decision": "main/joint_decision/detector.py",
}
CANDIDATE_IDS = {
    "key_schedule": ["key_schedule_sha256_counter"],
    "content_router": [
        "key_schedule_sha256_counter",
        "routing_stqr",
        "routing_uniform_control",
    ],
    "lf_carrier": ["key_schedule_sha256_counter", "lf_low_pass"],
    "hf_carrier": [
        "key_schedule_sha256_counter",
        "runtime_sd35_flowmatch",
        "hf_sparse_tail",
    ],
    "content_embedder": [
        "runtime_sd35_flowmatch",
        "hf_sparse_tail",
        "lf_low_pass",
        "routing_stqr",
        "routing_uniform_control",
    ],
    "lf_detector": [
        "key_schedule_sha256_counter",
        "lf_low_pass",
        "lf_null_whitened_matched_score",
    ],
    "hf_detector": ["key_schedule_sha256_counter", "hf_sparse_tail"],
    "content_detector": [
        "hf_sparse_tail",
        "lf_low_pass",
        "content_combination_calibrated",
    ],
    "qk_geometry_sync": [
        "key_schedule_sha256_counter",
        "runtime_sd35_flowmatch",
        "qk_relation_similarity",
    ],
    "geometric_transform_estimator": [
        "key_schedule_sha256_counter",
        "qk_relation_similarity",
        "rectification_similarity",
    ],
    "geometry_reliability": [
        "key_schedule_sha256_counter",
        "qk_relation_similarity",
        "rectification_similarity",
    ],
    "image_rectifier": ["rectification_similarity"],
    "conditional_recovery_decision": ["joint_conditional_recovery"],
}
SYMBOLS = {
    "key_schedule": "key_schedule_sha256_counter",
    "content_router": "content_router",
    "lf_carrier": "lf_carrier",
    "hf_carrier": "hf_carrier",
    "content_embedder": "content_embedder",
    "lf_detector": "lf_detector",
    "hf_detector": "hf_detector",
    "content_detector": "content_detector",
    "qk_geometry_sync": "qk_geometry_sync",
    "geometric_transform_estimator": "geometric_transform_estimator",
    "geometry_reliability": "geometry_reliability",
    "image_rectifier": "image_rectifier",
    "conditional_recovery_decision": "conditional_recovery_decision",
}
BEHAVIOR_BINDINGS = {
    "key_schedule_root_and_domain_separation": ["key_schedule"],
    "key_schedule_counter_quantile_golden": ["key_schedule"],
    "key_schedule_wrong_key_and_public_noise": ["key_schedule"],
    "hf_sparse_support": ["hf_carrier"],
    "hf_template_normalization_order_and_unit_l2": ["hf_carrier"],
    "hf_direct_score_time_centering": ["hf_detector"],
    "lf_domain_and_independent_key": ["lf_carrier"],
    "lf_blind_score_time_centering": ["lf_detector"],
    "lf_wrong_key_rejection": ["lf_carrier", "lf_detector"],
    "lf_whitened_asset_and_detector_are_explicit_no_fallback_candidates": [
        "lf_detector"
    ],
    "routing_mask_partition_and_range": ["content_router"],
    "routing_disabled_uniform_control": ["content_router"],
    "content_embedding_branch_consumption": [
        "content_router",
        "lf_carrier",
        "hf_carrier",
        "content_embedder",
    ],
    "content_embedding_total_budget_and_frozen_allocation": ["content_embedder"],
    "content_embedding_active_zero_direction_fail_closed": ["content_embedder"],
    "content_wrong_key_rejection": [
        "hf_carrier",
        "hf_detector",
        "lf_carrier",
        "lf_detector",
    ],
    "content_scores_independently_observable": [
        "lf_detector",
        "hf_detector",
        "content_detector",
    ],
    "content_combination_branch_consumption": [
        "lf_detector",
        "hf_detector",
        "content_detector",
    ],
    "content_combination_frozen_formula_identity": ["content_detector"],
    "content_combination_wrong_key_not_masked": [
        "lf_detector",
        "hf_detector",
        "content_detector",
    ],
    "qk_relation_consumption": ["qk_geometry_sync"],
    "qk_similarity_transform_identifiability": [
        "qk_geometry_sync",
        "geometric_transform_estimator",
    ],
    "geometry_reliability_fail_closed": [
        "geometric_transform_estimator",
        "geometry_reliability",
    ],
    "geometry_reliability_wrong_key_and_raw_metrics": [
        "geometric_transform_estimator",
        "geometry_reliability",
    ],
    "rectification_coordinate_protocol": ["image_rectifier"],
    "near_threshold_recovery_gate": ["conditional_recovery_decision"],
    "geometry_no_direct_positive": [
        "geometry_reliability",
        "conditional_recovery_decision",
    ],
    "joint_same_detector_threshold": [
        "content_detector",
        "image_rectifier",
        "conditional_recovery_decision",
    ],
}


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_authority(root: Path, stage: str) -> None:
    policy = {
        "stage_order": [
            "project_constraint_framework",
            "research_defined",
            "method_construction_authorized",
            "method_implemented",
            "runtime_verified",
        ],
        "method_readiness_stages": ["method_implemented", "runtime_verified"],
        "manifest_path": ".codex/research_state/method_readiness.yaml",
        "candidate_specification_path": "docs/design/candidate_specifications.md",
        "design_root": "docs/design",
        "implementation_root": "main",
        "test_roots": ["tests/unit", "tests/functional"],
        "required_method_component_count": 13,
        "required_method_components": list(RESPONSIBILITIES),
        "required_component_responsibilities": RESPONSIBILITIES,
        "required_component_paths": COMPONENT_PATHS,
        "required_component_candidate_ids": CANDIDATE_IDS,
        "required_behavioral_checks": list(BEHAVIOR_BINDINGS),
        "required_behavior_component_bindings": BEHAVIOR_BINDINGS,
        "independent_semantic_review_required": True,
    }
    _write(
        root / "governance/policies/method_readiness_rules.yaml",
        json.dumps(policy),
    )
    _write(
        root / ".codex/project_contract.md",
        f"- `project_stage`: `{stage}`\n",
    )


def _method_sources() -> dict[str, str]:
    return {
        "main/shared/key_schedule.py": (
            "import hashlib\n"
            "import json\n"
            "def key_schedule_sha256_counter(root_key, domain_fields, shape, count):\n"
            "    if type(root_key) is not str or not root_key:\n"
            "        raise ValueError('non-empty text key required')\n"
            "    payload = {'keyed_prg_version': 'sha256_counter_normal_icdf_table20_float32', 'key_material': root_key, 'domain_fields': domain_fields, 'shape': list(shape)}\n"
            "    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')\n"
            "    domain = hashlib.sha256(encoded).digest()\n"
            "    stream = b''.join(hashlib.sha256(domain + counter.to_bytes(16, 'big')).digest() for counter in range((count * 20 + 255) // 256))\n"
            "    bits = int.from_bytes(stream, 'big')\n"
            "    width = len(stream) * 8\n"
            "    indices = tuple((bits >> (width - 20 * (index + 1))) & ((1 << 20) - 1) for index in range(count))\n"
            "    return domain.hex(), indices\n"
        ),
        "main/content_chain/routing.py": (
            "def content_router(observations, size, enabled=True):\n"
            "    if not enabled:\n"
            "        ones = (1.0,) * size\n"
            "        return {'A': ones, 'mask_lf': ones, 'mask_hf': ones, 'route_identity': 'routing_uniform_control'}\n"
            "    semantic = observations['semantic']\n"
            "    texture = observations['texture']\n"
            "    response = observations['response']\n"
            "    sensitivity = observations['sensitivity']\n"
            "    if not all(len(values) == size for values in (semantic, texture, response, sensitivity)):\n"
            "        raise ValueError('routing observation shape mismatch')\n"
            "    attention = tuple(((1.0 - s) * (1.0 - r) * (1.0 - q)) ** (1.0 / 3.0) for s, r, q in zip(semantic, response, sensitivity))\n"
            "    mask_lf = tuple(a * (1.0 - t) for a, t in zip(attention, texture))\n"
            "    mask_hf = tuple(a * t for a, t in zip(attention, texture))\n"
            "    return {'A': attention, 'mask_lf': mask_lf, 'mask_hf': mask_hf, 'route_identity': 'routing_stqr'}\n"
        ),
        "main/content_chain/lf_carrier.py": (
            "def lf_carrier(values, key_signs, mask=None):\n"
            "    low = [(value + sum(values) / len(values)) / 2 for value in values]\n"
            "    mean = sum(low) / len(low)\n"
            "    keyed = [(value - mean) * sign for value, sign in zip(low, key_signs)]\n"
            "    masked = keyed if mask is None else [value * weight for value, weight in zip(keyed, mask)]\n"
            "    norm = sum(value * value for value in masked) ** 0.5\n"
            "    if norm == 0:\n"
            "        raise ValueError('zero carrier direction')\n"
            "    return tuple(value / norm for value in masked)\n"
        ),
        "main/content_chain/lf_detector.py": (
            "def lf_detector(observed, template):\n"
            "    observed_mean = sum(observed) / len(observed)\n"
            "    template_mean = sum(template) / len(template)\n"
            "    left = [value - observed_mean for value in observed]\n"
            "    right = [value - template_mean for value in template]\n"
            "    numerator = sum(a * b for a, b in zip(left, right))\n"
            "    denominator = (sum(a * a for a in left) * sum(b * b for b in right)) ** 0.5\n"
            "    if denominator == 0:\n"
            "        raise ValueError('zero centered energy')\n"
            "    return numerator / denominator\n"
            "def lf_null_whitened_matched_detector(observed, template, weights):\n"
            "    if len(observed) != len(template) or len(weights) != len(observed):\n"
            "        raise ValueError('whitening shape mismatch')\n"
            "    whitened_observed = tuple(value * weight for value, weight in zip(observed, weights))\n"
            "    whitened_template = tuple(value * weight for value, weight in zip(template, weights))\n"
            "    return lf_detector(whitened_observed, whitened_template)\n"
        ),
        "main/content_chain/hf_carrier.py": (
            "def hf_carrier(tail_values, key_signs, keep, mask=None):\n"
            "    ranked = sorted(range(len(tail_values)), key=lambda i: (-abs(tail_values[i]), i))\n"
            "    support = set(ranked[:keep])\n"
            "    sparse = [tail_values[i] * key_signs[i] if i in support else 0.0 for i in range(len(tail_values))]\n"
            "    masked = sparse if mask is None else [value * weight for value, weight in zip(sparse, mask)]\n"
            "    norm = sum(value * value for value in masked) ** 0.5\n"
            "    if norm == 0:\n"
            "        raise ValueError('zero carrier direction')\n"
            "    return tuple(value / norm for value in masked)\n"
        ),
        "main/content_chain/embedder.py": (
            "def content_embedder(lf_direction, hf_direction, allocation, target_total_l2, mode='combined'):\n"
            "    if allocation not in {0.25, 0.50, 0.75}:\n"
            "        raise ValueError('unregistered allocation')\n"
            "    lf_norm = sum(value * value for value in lf_direction) ** 0.5\n"
            "    hf_norm = sum(value * value for value in hf_direction) ** 0.5\n"
            "    if mode not in {'lf_only', 'hf_only', 'combined'}:\n"
            "        raise ValueError('unregistered embedding mode')\n"
            "    if (mode in {'lf_only', 'combined'} and lf_norm == 0) or (mode in {'hf_only', 'combined'} and hf_norm == 0):\n"
            "        raise ValueError('active zero direction')\n"
            "    lf_unit = tuple(value / lf_norm for value in lf_direction) if lf_norm else tuple(0.0 for value in lf_direction)\n"
            "    hf_unit = tuple(value / hf_norm for value in hf_direction) if hf_norm else tuple(0.0 for value in hf_direction)\n"
            "    direction_cosine = sum(lf * hf for lf, hf in zip(lf_unit, hf_unit))\n"
            "    if mode == 'lf_only':\n"
            "        combined = lf_unit\n"
            "    elif mode == 'hf_only':\n"
            "        combined = hf_unit\n"
            "    else:\n"
            "        combined = tuple(allocation * lf + (1.0 - allocation) * hf for lf, hf in zip(lf_unit, hf_unit))\n"
            "    combined_norm = sum(value * value for value in combined) ** 0.5\n"
            "    if combined_norm == 0:\n"
            "        raise ValueError('combined zero direction')\n"
            "    delta = tuple(target_total_l2 * value / combined_norm for value in combined)\n"
            "    return {'delta': delta, 'allocation': allocation, 'mode': mode, 'target_total_l2': target_total_l2, 'combined_pre_norm': combined_norm, 'direction_cosine': direction_cosine}\n"
        ),
        "main/content_chain/hf_detector.py": (
            "def hf_detector(observed, template):\n"
            "    observed_mean = sum(observed) / len(observed)\n"
            "    template_mean = sum(template) / len(template)\n"
            "    left = [value - observed_mean for value in observed]\n"
            "    right = [value - template_mean for value in template]\n"
            "    numerator = sum(a * b for a, b in zip(left, right))\n"
            "    denominator = (sum(a * a for a in left) * sum(b * b for b in right)) ** 0.5\n"
            "    return numerator / denominator\n"
        ),
        "main/content_chain/detector.py": (
            "def _midrank_normal_score(score, null_scores, normal_table):\n"
            "    if len(null_scores) < 2:\n"
            "        raise ValueError('at least two null scores required')\n"
            "    less = sum(value < score for value in null_scores)\n"
            "    equal = sum(value == score for value in null_scores)\n"
            "    raw = (less + 0.5 * equal) / len(null_scores)\n"
            "    epsilon = 1.0 / (2.0 * len(null_scores))\n"
            "    clipped = min(1.0 - epsilon, max(epsilon, raw))\n"
            "    index = min(len(normal_table) - 1, int(clipped * len(normal_table)))\n"
            "    return float(normal_table[index])\n"
            "def content_detector(lf_score, hf_score, lf_null, hf_null, combination, normal_table):\n"
            "    z_lf = _midrank_normal_score(lf_score, lf_null, normal_table)\n"
            "    z_hf = _midrank_normal_score(hf_score, hf_null, normal_table)\n"
            "    if combination == 'hf_only_standardized_score':\n"
            "        combined = z_hf\n"
            "    elif combination == 'weighted_hf_lf_standardized_score':\n"
            "        weight = 0.50\n"
            "        combined = weight * z_hf + (1.0 - weight * weight) ** 0.5 * z_lf\n"
            "    elif combination == 'maximum_hf_lf_standardized_score':\n"
            "        combined = max(z_hf, z_lf)\n"
            "    else:\n"
            "        raise ValueError('unregistered combination')\n"
            "    return {'lf': lf_score, 'hf': hf_score, 'combined': combined, 'z_lf': z_lf, 'z_hf': z_hf}\n"
        ),
        "main/geometry_chain/qk_sync.py": (
            "def qk_geometry_sync(query, key):\n"
            "    logits = tuple(q * k for q, k in zip(query, key))\n"
            "    scale = sum(abs(value) for value in logits) or 1.0\n"
            "    probabilities = tuple(abs(value) / scale for value in logits)\n"
            "    return {'logits': logits, 'probabilities': probabilities}\n"
        ),
        "main/geometry_chain/transform_estimator.py": (
            "def geometric_transform_estimator(scored_candidates):\n"
            "    ordered = sorted(scored_candidates, key=lambda item: item[1], reverse=True)\n"
            "    best_transform, best_score = ordered[0]\n"
            "    second_score = ordered[1][1]\n"
            "    return {'transform': best_transform, 'score': best_score, 'gap': best_score - second_score, 'coverage': 0.9, 'uniqueness': 0.8, 'key_margin': 0.7, 'inlier_ratio': 0.85, 'residual': 0.02, 'boundary': False, 'identity_margin': 0.2}\n"
        ),
        "main/geometry_chain/reliability.py": (
            "import math\n"
            "def geometry_reliability(metrics, limits):\n"
            "    required = ('score', 'gap', 'coverage', 'uniqueness', 'key_margin', 'inlier_ratio', 'residual', 'identity_margin')\n"
            "    if any(name not in metrics or not math.isfinite(metrics[name]) for name in required):\n"
            "        return False, 'missing_or_nonfinite', metrics.get('score')\n"
            "    checks = (metrics['coverage'] >= limits['coverage'], metrics['uniqueness'] >= limits['uniqueness'], metrics['gap'] >= limits['gap'], metrics['key_margin'] >= limits['key_margin'], metrics['inlier_ratio'] >= limits['inlier_ratio'], metrics['residual'] <= limits['residual'], not metrics.get('boundary', True), metrics['identity_margin'] >= limits['identity_margin'])\n"
            "    if not all(checks):\n"
            "        return False, 'conjunction_failed', metrics['score']\n"
            "    return True, 'reliable', metrics['score']\n"
        ),
        "main/geometry_chain/rectifier.py": (
            "def image_rectifier(values, shift):\n"
            "    size = len(values)\n"
            "    return tuple(values[(index + shift) % size] for index in range(size))\n"
        ),
        "main/joint_decision/detector.py": (
            "def conditional_recovery_decision(raw_score, rectified_score, threshold, near_threshold, reliable):\n"
            "    if raw_score >= threshold:\n"
            "        return True, raw_score, False\n"
            "    if not near_threshold or not reliable:\n"
            "        return False, raw_score, False\n"
            "    return rectified_score >= threshold, rectified_score, True\n"
        ),
    }


def _behavior_test_source() -> str:
    return """import pytest
from main.shared.key_schedule import key_schedule_sha256_counter
from main.content_chain.detector import content_detector
from main.content_chain.embedder import content_embedder
from main.content_chain.hf_carrier import hf_carrier
from main.content_chain.hf_detector import hf_detector
from main.content_chain.lf_carrier import lf_carrier
from main.content_chain.lf_detector import lf_detector, lf_null_whitened_matched_detector
from main.content_chain.routing import content_router
from main.geometry_chain.qk_sync import qk_geometry_sync
from main.geometry_chain.reliability import geometry_reliability
from main.geometry_chain.rectifier import image_rectifier
from main.geometry_chain.transform_estimator import geometric_transform_estimator
from main.joint_decision.detector import conditional_recovery_decision

@pytest.mark.unit
def test_key_schedule_root_and_domain_separation():
    left = key_schedule_sha256_counter("registered", {"role": "hf"}, (2, 2), 4)
    right = key_schedule_sha256_counter("registered", {"role": "lf"}, (2, 2), 4)
    other = key_schedule_sha256_counter("wrong", {"role": "hf"}, (2, 2), 4)
    assert left != right
    assert left != other

@pytest.mark.unit
def test_key_schedule_counter_quantile_golden():
    fields = {"candidate_id": "key_schedule_sha256_counter", "operator": "golden_vector", "responsibility_domain": "key_schedule_test", "tensor_role": "gaussian"}
    result = key_schedule_sha256_counter("ceg-wm-golden-root-π", fields, (2, 3), 6)
    assert result[0] == "e5b8e35d13815c1d23a09286da0bfe661e0330e38eda19e239f19224f7b1998f"
    assert result[1] == (172059, 964892, 707530, 322430, 968250, 915318)

@pytest.mark.unit
def test_key_schedule_wrong_key_and_public_noise():
    registered = key_schedule_sha256_counter("registered", {"role": "geometry"}, (2, 2), 4)
    wrong = key_schedule_sha256_counter("ceg-wm-wrong-key:0", {"role": "geometry"}, (2, 2), 4)
    public = key_schedule_sha256_counter("ceg-wm-public-noise:key-schedule-sha256-counter", {"role": "public_noise"}, (2, 2), 4)
    assert registered != wrong
    assert public != registered and public != wrong

@pytest.mark.unit
def test_hf_sparse_support():
    template = hf_carrier((0.2, -4.0, 3.0, 0.1), (1, 1, 1, 1), 2)
    support = [index for index, value in enumerate(template) if value]
    assert support == [1, 2]

@pytest.mark.unit
def test_hf_template_normalization_order_and_unit_l2():
    template = hf_carrier((0.0, 4.0, -3.0, 0.0), (1, 1, 1, 1), 2)
    unit_l2 = sum(value * value for value in template)
    assert abs(unit_l2 - 1.0) < 1e-12 and template[0] == template[3] == 0.0

@pytest.mark.unit
def test_hf_direct_score_time_centering():
    base_score = hf_detector((2.0, 4.0, 6.0), (1.0, 2.0, 3.0))
    shifted_score = hf_detector((12.0, 14.0, 16.0), (6.0, 7.0, 8.0))
    assert base_score == pytest.approx(1.0)
    assert shifted_score == pytest.approx(base_score)

@pytest.mark.unit
def test_lf_domain_and_independent_key():
    registered = lf_carrier((1.0, 2.0, 4.0, 8.0), (1, -1, 1, -1))
    wrong = lf_carrier((1.0, 2.0, 4.0, 8.0), (-1, 1, -1, 1))
    assert len(registered) == 4
    assert registered != wrong

@pytest.mark.unit
def test_lf_blind_score_time_centering():
    base = lf_detector((2.0, 4.0, 6.0), (1.0, 2.0, 3.0))
    shifted = lf_detector((12.0, 14.0, 16.0), (6.0, 7.0, 8.0))
    assert base == pytest.approx(1.0)
    assert shifted == pytest.approx(base)
    assert isinstance(shifted, float)

@pytest.mark.unit
def test_lf_wrong_key_rejection():
    registered = lf_carrier((1.0, 2.0, 4.0, 8.0), (1, -1, 1, -1))
    wrong = lf_carrier((1.0, 2.0, 4.0, 8.0), (-1, 1, -1, 1))
    registered_score = lf_detector(registered, registered)
    wrong_score = lf_detector(registered, wrong)
    assert registered_score > wrong_score

@pytest.mark.unit
def test_lf_whitened_asset_and_detector_are_explicit_no_fallback_candidates():
    observed = (3.0, -2.0, 1.0, 4.0)
    template = (1.0, -1.0, 2.0, 0.5)
    raw = lf_detector(observed, template)
    whitened = lf_null_whitened_matched_detector(
        observed,
        template,
        (4.0, 0.5, 2.0, 0.25),
    )
    assert whitened != raw

@pytest.mark.unit
def test_routing_mask_partition_and_range():
    observations = {
        "semantic": (0.2, 0.4, 0.1, 0.3),
        "texture": (0.75, 0.25, 0.6, 0.4),
        "response": (0.1, 0.2, 0.3, 0.1),
        "sensitivity": (0.3, 0.2, 0.1, 0.4),
    }
    routed = content_router(observations, 4, True)
    partitions = tuple(
        lf + hf for lf, hf in zip(routed["mask_lf"], routed["mask_hf"])
    )
    assert partitions == pytest.approx(routed["A"])
    assert all(
        0.0 <= value <= 1.0
        for name in ("A", "mask_lf", "mask_hf")
        for value in routed[name]
    )
    assert set(routed) == {"A", "mask_lf", "mask_hf", "route_identity"}

@pytest.mark.unit
def test_routing_disabled_uniform_control():
    disabled = content_router(None, 4, False)
    assert disabled == {
        "A": (1.0, 1.0, 1.0, 1.0),
        "mask_lf": (1.0, 1.0, 1.0, 1.0),
        "mask_hf": (1.0, 1.0, 1.0, 1.0),
        "route_identity": "routing_uniform_control",
    }

@pytest.mark.unit
def test_content_embedding_branch_consumption():
    observations = {
        "semantic": (0.2, 0.4, 0.1, 0.3),
        "texture": (0.75, 0.25, 0.6, 0.4),
        "response": (0.1, 0.2, 0.3, 0.1),
        "sensitivity": (0.3, 0.2, 0.1, 0.4),
    }
    route = content_router(observations, 4, True)
    lf = lf_carrier(
        (1.0, 2.0, 4.0, 8.0),
        (1, -1, 1, -1),
        route["mask_lf"],
    )
    hf = hf_carrier(
        (4.0, -3.0, 0.2, 0.1),
        (1, 1, 1, 1),
        2,
        route["mask_hf"],
    )
    mixed = content_embedder(lf, hf, 0.50, 0.012)
    uniform = content_router(None, 4, False)
    uniform_lf = lf_carrier(
        (1.0, 2.0, 4.0, 8.0),
        (1, -1, 1, -1),
        uniform["mask_lf"],
    )
    changed_lf = lf_carrier(
        (1.0, 2.0, 4.0, 8.0),
        (-1, 1, -1, 1),
        route["mask_lf"],
    )
    changed = content_embedder(changed_lf, hf, 0.50, 0.012)
    assert route["mask_lf"] != uniform["mask_lf"] and lf != uniform_lf
    assert mixed["delta"] != changed["delta"]

@pytest.mark.unit
def test_content_embedding_total_budget_and_frozen_allocation():
    lf_direction = (1.0, 0.0)
    hf_direction = (0.6, 0.8)
    results = [
        content_embedder(
            lf_direction,
            hf_direction,
            allocation,
            0.012,
            mode,
        )
        for mode in ("hf_only", "lf_only", "combined")
        for allocation in (0.25, 0.50, 0.75)
    ]
    energies = [
        sum(value * value for value in result["delta"]) ** 0.5
        for result in results
    ]
    with pytest.raises(ValueError, match="unregistered allocation"):
        content_embedder(lf_direction, hf_direction, 0.40, 0.012)
    combined = results[6]
    expected_pre_norm = (
        0.25**2 + 0.75**2 + 2 * 0.25 * 0.75 * 0.6
    ) ** 0.5
    expected_delta = (
        0.012 * (0.25 + 0.75 * 0.6) / expected_pre_norm,
        0.012 * (0.75 * 0.8) / expected_pre_norm,
    )
    assert energies == pytest.approx([0.012] * 9)
    assert all(result["target_total_l2"] == pytest.approx(0.012) for result in results)
    assert all(result["allocation"] in {0.25, 0.50, 0.75} for result in results)
    assert {result["mode"] for result in results} == {
        "hf_only",
        "lf_only",
        "combined",
    }
    assert all(result["direction_cosine"] == pytest.approx(0.6) for result in results)
    assert combined["combined_pre_norm"] == pytest.approx(expected_pre_norm)
    assert combined["delta"] == pytest.approx(expected_delta)
    assert set(combined) == {
        "delta",
        "allocation",
        "mode",
        "target_total_l2",
        "combined_pre_norm",
        "direction_cosine",
    }

@pytest.mark.unit
def test_content_embedding_active_zero_direction_fail_closed():
    valid = content_embedder((1.0, -1.0), (1.0, 1.0), 0.50, 0.012)
    with pytest.raises(ValueError, match="active zero direction"):
        content_embedder((0.0, 0.0), (1.0, -1.0), 0.50, 0.012)
    assert valid["target_total_l2"] == pytest.approx(0.012)

@pytest.mark.unit
def test_content_wrong_key_rejection():
    registered_hf = hf_carrier((4.0, -3.0, 0.1), (1, 1, 1), 2)
    wrong_hf = hf_carrier((4.0, -3.0, 0.1), (-1, 1, -1), 2)
    registered_lf = lf_carrier((1.0, 2.0, 4.0), (1, -1, 1))
    wrong_lf = lf_carrier((1.0, 2.0, 4.0), (-1, 1, -1))
    hf_margin = hf_detector(registered_hf, registered_hf) - hf_detector(registered_hf, wrong_hf)
    lf_margin = lf_detector(registered_lf, registered_lf) - lf_detector(registered_lf, wrong_lf)
    assert hf_margin > 0 and lf_margin > 0

@pytest.mark.unit
def test_content_scores_independently_observable():
    lf_score = lf_detector((1.0, 2.0, 4.0), (1.0, 2.0, 4.0))
    hf_score = hf_detector((1.0, -1.0, 2.0), (1.0, -1.0, 2.0))
    result = content_detector(lf_score, hf_score, (-1.0, 0.0, 0.5), (-1.0, 0.0, 0.5), "hf_only_standardized_score", (-2.0, -1.0, 0.0, 1.0, 2.0))
    assert result["lf"] == lf_score and result["hf"] == hf_score
    assert set(result) == {"lf", "hf", "combined", "z_lf", "z_hf"}

@pytest.mark.unit
def test_content_combination_branch_consumption():
    lf_a = lf_detector((1.0, 2.0, 4.0), (1.0, 2.0, 4.0))
    lf_b = lf_detector((1.0, 2.0, 4.0), (4.0, 2.0, 1.0))
    hf_a = hf_detector((1.0, -1.0, 2.0), (1.0, -1.0, 2.0))
    hf_b = hf_detector((1.0, -1.0, 2.0), (2.0, -1.0, 1.0))
    table = (-2.0, -1.0, 0.0, 1.0, 2.0)
    first = content_detector(lf_a, hf_a, (-1.0, 0.0, 0.5), (-1.0, 0.0, 0.5), "weighted_hf_lf_standardized_score", table)
    second = content_detector(lf_b, hf_b, (-1.0, 0.0, 0.5), (-1.0, 0.0, 0.5), "weighted_hf_lf_standardized_score", table)
    assert first["combined"] != second["combined"]

@pytest.mark.unit
def test_content_combination_frozen_formula_identity():
    table = (-2.0, -1.0, 0.0, 1.0, 2.0)
    result = content_detector(0.0, 0.5, (-1.0, 0.0, 1.0), (-1.0, 0.0, 0.5), "weighted_hf_lf_standardized_score", table)
    expected = 0.50 * result["z_hf"] + (1.0 - 0.50 ** 2) ** 0.5 * result["z_lf"]
    assert result["combined"] == pytest.approx(expected)
    assert result["z_lf"] == 0.0 and result["z_hf"] == 2.0

@pytest.mark.unit
def test_content_combination_wrong_key_not_masked():
    lf_registered = lf_detector((1.0, 2.0, 4.0), (1.0, 2.0, 4.0))
    lf_wrong = lf_detector((1.0, 2.0, 4.0), (4.0, 2.0, 1.0))
    hf_registered = hf_detector((1.0, -1.0, 2.0), (1.0, -1.0, 2.0))
    hf_wrong = hf_detector((1.0, -1.0, 2.0), (-1.0, 1.0, -2.0))
    table = (-2.0, -1.0, 0.0, 1.0, 2.0)
    registered = content_detector(lf_registered, hf_registered, (-1.0, 0.0, 0.5), (-1.0, 0.0, 0.5), "weighted_hf_lf_standardized_score", table)
    wrong = content_detector(lf_wrong, hf_wrong, (-1.0, 0.0, 0.5), (-1.0, 0.0, 0.5), "weighted_hf_lf_standardized_score", table)
    assert registered["lf"] != wrong["lf"] and registered["hf"] != wrong["hf"]
    assert registered["combined"] > wrong["combined"]

@pytest.mark.unit
def test_qk_relation_consumption():
    relation = qk_geometry_sync((2.0, 1.0), (3.0, -4.0))
    assert set(relation) == {"logits", "probabilities"}
    assert relation["logits"] == (6.0, -4.0)

@pytest.mark.unit
def test_qk_similarity_transform_identifiability():
    relation = qk_geometry_sync((1.0, 2.0), (1.0, 2.0))
    estimate = geometric_transform_estimator([(("identity",), sum(relation["probabilities"])), (("rotated",), 0.1)])
    assert estimate["transform"] == ("identity",)
    assert estimate["gap"] > 0

@pytest.mark.unit
def test_geometry_reliability_fail_closed():
    estimate = geometric_transform_estimator([(("a",), 0.5), (("b",), 0.49)])
    limits = {"coverage": 0.5, "uniqueness": 0.5, "gap": 0.05, "key_margin": 0.5, "inlier_ratio": 0.5, "residual": 0.1, "identity_margin": 0.1}
    cases = [
        estimate,
        dict(estimate, coverage=0.1, gap=0.2),
        dict(estimate, uniqueness=0.1, gap=0.2),
        dict(estimate, residual=0.9, gap=0.2),
        dict(estimate, boundary=True, gap=0.2),
    ]
    decisions = [geometry_reliability(case, limits) for case in cases]
    assert estimate["transform"] == ("a",)
    assert all(decision[:2] == (False, "conjunction_failed") for decision in decisions)

@pytest.mark.unit
def test_geometry_reliability_wrong_key_and_raw_metrics():
    estimate = geometric_transform_estimator([(("identity",), 0.9), (("other",), 0.1)])
    wrong_key_metrics = dict(estimate, key_margin=-0.1)
    limits = {"coverage": 0.5, "uniqueness": 0.5, "gap": 0.05, "key_margin": 0.5, "inlier_ratio": 0.5, "residual": 0.1, "identity_margin": 0.1}
    wrong_key = geometry_reliability(wrong_key_metrics, limits)
    nonfinite = geometry_reliability(dict(estimate, residual=float("nan")), limits)
    assert wrong_key[:2] == (False, "conjunction_failed")
    assert nonfinite[:2] == (False, "missing_or_nonfinite")

@pytest.mark.unit
def test_rectification_coordinate_protocol():
    rectified = image_rectifier(("a", "b", "c", "d"), 1)
    expected = ("b", "c", "d", "a")
    assert tuple(rectified) == expected

@pytest.mark.unit
def test_near_threshold_recovery_gate():
    blocked = conditional_recovery_decision(0.4, 0.9, 0.8, False, True)
    rescued = conditional_recovery_decision(0.4, 0.9, 0.8, True, True)
    assert blocked[2] is False
    assert rescued == (True, 0.9, True)

@pytest.mark.unit
def test_geometry_no_direct_positive():
    metrics = {"score": 2.0, "gap": 1.0, "coverage": 1.0, "uniqueness": 1.0, "key_margin": -1.0, "inlier_ratio": 1.0, "residual": 0.0, "boundary": False, "identity_margin": 1.0}
    limits = {"coverage": 0.5, "uniqueness": 0.5, "gap": 0.05, "key_margin": 0.5, "inlier_ratio": 0.5, "residual": 0.1, "identity_margin": 0.1}
    reliable = geometry_reliability(metrics, limits)
    decision = conditional_recovery_decision(0.1, 2.0, 0.8, True, reliable[0])
    assert reliable[0] is False
    assert decision[0] is False
    assert decision[1:] == (0.1, False)

@pytest.mark.unit
def test_joint_same_detector_threshold():
    template = (1.0, -1.0, 1.0, -1.0)
    null = (-1.0, 0.0, 0.5)
    table = (-2.0, -1.0, 0.0, 1.0, 2.0)
    raw_hf = hf_detector((1.0, 1.0, -1.0, -1.0), template)
    raw_score = content_detector(-1.0, raw_hf, null, null, "hf_only_standardized_score", table)["combined"]
    rectified = image_rectifier((1.0, 1.0, -1.0, -1.0), 1)
    rectified_hf = hf_detector(rectified, template)
    rectified_score = content_detector(-1.0, rectified_hf, null, null, "hf_only_standardized_score", table)["combined"]
    final = conditional_recovery_decision(raw_score, rectified_score, 0.8, True, True)
    assert final[1] == rectified_score
    assert raw_score < 0.8
"""


def _commit_reviewed_surface(root: Path) -> str:
    commands = [
        ["git", "init", "-q"],
        ["git", "config", "user.email", "governance@example.invalid"],
        ["git", "config", "user.name", "Governance Fixture"],
        ["git", "add", "."],
        ["git", "commit", "-q", "-m", "reviewed method surface"],
    ]
    for command in commands:
        subprocess.run(command, cwd=root, check=True)
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write_schema_complete_method_fixture(root: Path) -> None:
    """写入仅供门禁自测使用的候选特异性、分层 fixture。"""
    candidate_path = root / "docs/design/candidate_specifications.md"
    candidate_names = sorted(
        {
            candidate_id
            for values in CANDIDATE_IDS.values()
            for candidate_id in values
        }
    )
    _write(
        candidate_path,
        "# Frozen Candidate Specifications\n\n"
        + "\n".join(f"- `{name}`" for name in candidate_names)
        + "\n",
    )
    _write(root / "docs/design/method_architecture.md", "# Method Architecture\n")
    for relative, source in _method_sources().items():
        _write(root / relative, source)
    test_relative = "tests/unit/test_candidate_specific_method.py"
    _write(root / test_relative, _behavior_test_source())
    reviewed_revision = _commit_reviewed_surface(root)
    candidate_digest = hashlib.sha256(candidate_path.read_bytes()).hexdigest()

    manifest = {
        "method_name": "ceg_wm_dual_chain",
        "design_path": "docs/design/method_architecture.md",
        "candidate_specification_path": "docs/design/candidate_specifications.md",
        "candidate_specification_sha256": candidate_digest,
        "components": {
            component: {
                "responsibility": RESPONSIBILITIES[component],
                "candidate_ids": CANDIDATE_IDS[component],
                "implementation_path": COMPONENT_PATHS[component],
                "implementation_symbol": SYMBOLS[component],
            }
            for component in RESPONSIBILITIES
        },
        "test_paths": [test_relative],
        "behavioral_checks": {
            name: {
                "test_node": f"{test_relative}::test_{name}",
                "components": components,
            }
            for name, components in BEHAVIOR_BINDINGS.items()
        },
        "independent_semantic_review": {
            "decision": "approve",
            "review_reference": "independent-review-fixture",
            "reviewed_repository_revision": reviewed_revision,
            "candidate_specification_sha256": candidate_digest,
        },
    }
    _write(
        root / ".codex/research_state/method_readiness.yaml",
        json.dumps(manifest),
    )


def _write_legacy_arithmetic_proxy_fixture(root: Path) -> None:
    """复现审计指出的单文件算术玩具，不把它命名为有效方法证据。"""
    _write_schema_complete_method_fixture(root)
    proxy_path = root / "main/dual_chain.py"
    _write(
        proxy_path,
        "def route_content(values, enabled):\n"
        "    return values if enabled else tuple(reversed(values))\n"
        "def write_lf_carrier(value, scale):\n"
        "    return value * scale\n"
        "def write_hf_carrier(value, offset):\n"
        "    return value + offset\n"
        "def score_hf_evidence(lf_score, hf_score):\n"
        "    return max(lf_score, hf_score)\n"
        "def synchronize_qk_geometry(query, key):\n"
        "    return query - key\n"
        "def estimate_geometric_transform(observation, reference):\n"
        "    return observation / reference\n"
        "def image_rectifier(values, trim):\n"
        "    return values[trim:]\n"
        "def conditional_recovery_decision(raw_score, rectified_score, eligible):\n"
        "    return rectified_score if eligible else raw_score\n",
    )
    legacy_test_relative = "tests/unit/test_dual_chain.py"
    _write(
        root / legacy_test_relative,
        """import pytest
from main.dual_chain import (
    conditional_recovery_decision,
    estimate_geometric_transform,
    image_rectifier,
    route_content,
    score_hf_evidence,
    synchronize_qk_geometry,
    write_hf_carrier,
    write_lf_carrier,
)

@pytest.mark.unit
def test_content_chain_effect():
    assert score_hf_evidence(write_lf_carrier(2, 3), write_hf_carrier(2, 5)) == 7

@pytest.mark.unit
def test_content_chain_disabled_control():
    assert route_content((1, 2, 3), True) != route_content((1, 2, 3), False)

@pytest.mark.unit
def test_wrong_key_rejection():
    assert write_hf_carrier(2, 5) > write_hf_carrier(2, -1)

@pytest.mark.unit
def test_geometry_transform_estimation():
    assert estimate_geometric_transform(synchronize_qk_geometry(9, 3), 2) == 3

@pytest.mark.unit
def test_geometry_unreliable_control():
    assert 0 < estimate_geometric_transform(1, 4) < 1

@pytest.mark.unit
def test_near_threshold_recovery_gate():
    assert conditional_recovery_decision(4, 9, False) == 4

@pytest.mark.unit
def test_geometry_no_direct_positive():
    assert conditional_recovery_decision(-2, 8, False) < 0

@pytest.mark.unit
def test_same_detector_same_threshold():
    assert score_hf_evidence(0, image_rectifier((1, 8), 1)[0]) == 8
""",
    )
    manifest_path, manifest = _manifest(root)
    legacy_symbols = {
        "content_router": "route_content",
        "lf_carrier": "write_lf_carrier",
        "hf_carrier": "write_hf_carrier",
        "hf_detector": "score_hf_evidence",
        "content_detector": "score_hf_evidence",
        "qk_geometry_sync": "synchronize_qk_geometry",
        "geometric_transform_estimator": "estimate_geometric_transform",
        "image_rectifier": "image_rectifier",
        "conditional_recovery_decision": "conditional_recovery_decision",
    }
    for component in manifest["components"].values():
        component["implementation_path"] = "main/dual_chain.py"
    for name, symbol in legacy_symbols.items():
        manifest["components"][name]["implementation_symbol"] = symbol
    manifest["test_paths"] = [legacy_test_relative]
    manifest["behavioral_checks"] = {
        "content_chain_effect": {
            "test_node": f"{legacy_test_relative}::test_content_chain_effect",
            "components": ["lf_carrier", "hf_carrier", "content_detector"],
        },
        "content_chain_disabled_control": {
            "test_node": (
                f"{legacy_test_relative}::test_content_chain_disabled_control"
            ),
            "components": ["content_router"],
        },
        "wrong_key_rejection": {
            "test_node": f"{legacy_test_relative}::test_wrong_key_rejection",
            "components": ["hf_carrier", "hf_detector"],
        },
        "geometry_transform_estimation": {
            "test_node": (
                f"{legacy_test_relative}::test_geometry_transform_estimation"
            ),
            "components": [
                "qk_geometry_sync",
                "geometric_transform_estimator",
            ],
        },
        "geometry_unreliable_control": {
            "test_node": (
                f"{legacy_test_relative}::test_geometry_unreliable_control"
            ),
            "components": ["geometric_transform_estimator"],
        },
        "near_threshold_recovery_gate": {
            "test_node": (
                f"{legacy_test_relative}::test_near_threshold_recovery_gate"
            ),
            "components": ["conditional_recovery_decision"],
        },
        "geometry_no_direct_positive": {
            "test_node": (
                f"{legacy_test_relative}::test_geometry_no_direct_positive"
            ),
            "components": ["conditional_recovery_decision"],
        },
        "same_detector_same_threshold": {
            "test_node": (
                f"{legacy_test_relative}::test_same_detector_same_threshold"
            ),
            "components": [
                "content_detector",
                "image_rectifier",
                "conditional_recovery_decision",
            ],
        },
    }
    _write(manifest_path, json.dumps(manifest))


def _manifest(root: Path) -> tuple[Path, dict]:
    path = root / ".codex/research_state/method_readiness.yaml"
    return path, json.loads(path.read_text(encoding="utf-8"))


def _commit_fixture_change_and_refresh_review(root: Path) -> None:
    subprocess.run(["git", "add", "main", "tests"], cwd=root, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "mutated reviewed surface"],
        cwd=root,
        check=True,
    )
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    manifest_path, manifest = _manifest(root)
    manifest["independent_semantic_review"]["reviewed_repository_revision"] = revision
    _write(manifest_path, json.dumps(manifest))


def _run_fixture_behavior(root: Path, check_name: str) -> subprocess.CompletedProcess[str]:
    _, manifest = _manifest(root)
    node = manifest["behavioral_checks"][check_name]["test_node"]
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-s", node],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )


@pytest.mark.unit
def test_research_stage_does_not_require_method_evidence(tmp_path: Path) -> None:
    _write_authority(tmp_path, "research_defined")
    assert run_audit(tmp_path)["decision"] == "pass"


@pytest.mark.unit
def test_research_stage_rejects_noncanonical_component_count(
    tmp_path: Path,
) -> None:
    _write_authority(tmp_path, "research_defined")
    policy_path = tmp_path / "governance/policies/method_readiness_rules.yaml"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    policy["required_method_component_count"] = 10
    _write(policy_path, json.dumps(policy))
    report = run_audit(tmp_path)
    assert report["decision"] == "fail"
    assert any(
        violation["reason"] == "method_component_policy_count_mismatch"
        for violation in report["violations"]
    )


@pytest.mark.unit
def test_current_identity_rejects_historical_hf_project_name(
    tmp_path: Path,
) -> None:
    _write_authority(tmp_path, "research_defined")
    policy_path = tmp_path / "governance/policies/method_readiness_rules.yaml"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    policy["required_method_components"].append("direct" + "_hf_carrier")
    _write(policy_path, json.dumps(policy))
    report = run_audit(tmp_path)
    assert report["decision"] == "fail"
    assert report["violations"][0]["reason"] == (
        "historical_hf_name_used_for_current_method_identity"
    )


@pytest.mark.unit
def test_construction_stage_allows_work_without_completion_claim(
    tmp_path: Path,
) -> None:
    _write_authority(tmp_path, "method_construction_authorized")
    assert run_audit(tmp_path)["decision"] == "pass"


@pytest.mark.unit
@pytest.mark.parametrize(
    "component",
    ["content_embedder", "lf_detector", "geometry_reliability"],
)
def test_split_responsibility_component_is_required(
    tmp_path: Path,
    component: str,
) -> None:
    _write_authority(tmp_path, "method_implemented")
    _write_schema_complete_method_fixture(tmp_path)
    manifest_path, manifest = _manifest(tmp_path)
    del manifest["components"][component]
    _write(manifest_path, json.dumps(manifest))
    report = run_audit(tmp_path)
    assert report["decision"] == "fail"
    assert any(
        violation["reason"] == "method_component_missing"
        and component in violation["components"]
        for violation in report["violations"]
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("component", "wrong_path"),
    [
        ("content_embedder", "main/content_chain/lf_carrier.py"),
        ("lf_detector", "main/content_chain/detector.py"),
        ("geometry_reliability", "main/geometry_chain/transform_estimator.py"),
    ],
)
def test_split_responsibility_cannot_be_folded_into_another_path(
    tmp_path: Path,
    component: str,
    wrong_path: str,
) -> None:
    _write_authority(tmp_path, "method_implemented")
    _write_schema_complete_method_fixture(tmp_path)
    manifest_path, manifest = _manifest(tmp_path)
    manifest["components"][component]["implementation_path"] = wrong_path
    _write(manifest_path, json.dumps(manifest))
    report = run_audit(tmp_path)
    assert report["decision"] == "fail"
    assert any(
        violation["reason"] == "method_component_implementation_path_mismatch"
        and violation["component"] == component
        for violation in report["violations"]
    )


@pytest.mark.unit
def test_method_stage_rejects_missing_manifest(tmp_path: Path) -> None:
    _write_authority(tmp_path, "method_implemented")
    report = run_audit(tmp_path)
    assert report["decision"] == "fail"
    assert report["violations"][0]["reason"] == "method_readiness_manifest_unreadable"


@pytest.mark.unit
def test_schema_complete_candidate_fixture_passes_structural_gate(
    tmp_path: Path,
) -> None:
    _write_authority(tmp_path, "method_implemented")
    _write_schema_complete_method_fixture(tmp_path)
    assert run_audit(tmp_path)["decision"] == "pass"


@pytest.mark.unit
def test_legacy_arithmetic_proxy_is_fail_closed(tmp_path: Path) -> None:
    _write_authority(tmp_path, "method_implemented")
    _write_legacy_arithmetic_proxy_fixture(tmp_path)
    toy_tests = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import runpy,sys; "
                "sys.path.insert(0, '.'); "
                "ns=runpy.run_path('tests/unit/test_dual_chain.py'); "
                "tests=[value for name,value in ns.items() "
                "if name.startswith('test_')]; "
                "[test() for test in tests]; "
                "print(f'{len(tests)} passed')"
            ),
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert toy_tests.returncode == 0
    assert "8 passed" in toy_tests.stdout
    report = run_audit(tmp_path)
    assert report["decision"] == "fail"
    reasons = {violation["reason"] for violation in report["violations"]}
    assert "method_component_implementation_path_mismatch" in reasons
    assert "method_behavioral_check_missing" in reasons


@pytest.mark.unit
def test_component_must_bind_exact_candidate_ids(tmp_path: Path) -> None:
    _write_authority(tmp_path, "method_implemented")
    _write_schema_complete_method_fixture(tmp_path)
    manifest_path, manifest = _manifest(tmp_path)
    manifest["components"]["lf_carrier"]["candidate_ids"] = ["historical_lf"]
    _write(manifest_path, json.dumps(manifest))
    report = run_audit(tmp_path)
    assert any(
        violation["reason"] == "method_component_candidate_binding_mismatch"
        for violation in report["violations"]
    )


@pytest.mark.unit
def test_component_cannot_move_into_central_proxy_module(tmp_path: Path) -> None:
    _write_authority(tmp_path, "method_implemented")
    _write_schema_complete_method_fixture(tmp_path)
    manifest_path, manifest = _manifest(tmp_path)
    manifest["components"]["lf_carrier"]["implementation_path"] = (
        "main/content_chain/hf_carrier.py"
    )
    _write(manifest_path, json.dumps(manifest))
    report = run_audit(tmp_path)
    assert any(
        violation["reason"] == "method_component_implementation_path_mismatch"
        for violation in report["violations"]
    )


@pytest.mark.unit
def test_component_alias_only_symbol_is_rejected(tmp_path: Path) -> None:
    _write_authority(tmp_path, "method_implemented")
    _write_schema_complete_method_fixture(tmp_path)
    path = tmp_path / COMPONENT_PATHS["content_embedder"]
    _write(
        path,
        "from main.content_chain.hf_carrier import hf_carrier\n"
        "def content_embedder(values, signs, keep, target_total_l2):\n"
        "    return hf_carrier(values, signs, keep)\n",
    )
    _commit_fixture_change_and_refresh_review(tmp_path)
    report = run_audit(tmp_path)
    assert any(
        violation["reason"] == "method_component_implementation_alias_only"
        and violation["component"] == "content_embedder"
        for violation in report["violations"]
    )


@pytest.mark.unit
def test_input_independent_component_is_rejected(tmp_path: Path) -> None:
    _write_authority(tmp_path, "method_implemented")
    _write_schema_complete_method_fixture(tmp_path)
    path = tmp_path / COMPONENT_PATHS["lf_carrier"]
    _write(path, "def lf_carrier(values, key_signs):\n    return (1.0,)\n")
    report = run_audit(tmp_path)
    assert any(
        violation["reason"] == "method_component_implementation_input_independent"
        for violation in report["violations"]
    )


@pytest.mark.unit
def test_embedder_that_ignores_lf_branch_fails_candidate_behavior(
    tmp_path: Path,
) -> None:
    _write_authority(tmp_path, "method_implemented")
    _write_schema_complete_method_fixture(tmp_path)
    path = tmp_path / COMPONENT_PATHS["content_embedder"]
    _write(
        path,
        "def content_embedder(lf_direction, hf_direction, allocation, target_total_l2):\n"
        "    hf_norm = sum(value * value for value in hf_direction) ** 0.5\n"
        "    if hf_norm == 0:\n"
        "        raise ValueError('active zero direction')\n"
        "    delta = tuple(target_total_l2 * value / hf_norm for value in hf_direction)\n"
        "    return {'delta': delta, 'allocation': allocation, 'mode': 'combined', 'target_total_l2': target_total_l2, 'combined_pre_norm': hf_norm, 'direction_cosine': 0.0}\n",
    )
    _commit_fixture_change_and_refresh_review(tmp_path)
    assert run_audit(tmp_path)["decision"] == "pass"
    behavior = _run_fixture_behavior(
        tmp_path,
        "content_embedding_branch_consumption",
    )
    assert behavior.returncode != 0
    assert "FAILED" in behavior.stdout


@pytest.mark.unit
def test_wrong_content_combination_formula_fails_candidate_behavior(
    tmp_path: Path,
) -> None:
    _write_authority(tmp_path, "method_implemented")
    _write_schema_complete_method_fixture(tmp_path)
    path = tmp_path / COMPONENT_PATHS["content_detector"]
    _write(
        path,
        "def content_detector(lf_score, hf_score, lf_null, hf_null, combination, normal_table):\n"
        "    combined = (lf_score + hf_score) / 2.0\n"
        "    observed = {'lf': lf_score, 'hf': hf_score, 'combined': combined}\n"
        "    return observed\n",
    )
    _commit_fixture_change_and_refresh_review(tmp_path)
    assert run_audit(tmp_path)["decision"] == "pass"
    behavior = _run_fixture_behavior(
        tmp_path,
        "content_combination_frozen_formula_identity",
    )
    assert behavior.returncode != 0
    assert "FAILED" in behavior.stdout


@pytest.mark.unit
def test_method_specific_behavior_node_is_required(tmp_path: Path) -> None:
    _write_authority(tmp_path, "method_implemented")
    _write_schema_complete_method_fixture(tmp_path)
    manifest_path, manifest = _manifest(tmp_path)
    del manifest["behavioral_checks"]["hf_sparse_support"]
    _write(manifest_path, json.dumps(manifest))
    report = run_audit(tmp_path)
    assert any(
        violation["reason"] == "method_behavioral_check_missing"
        for violation in report["violations"]
    )


@pytest.mark.unit
def test_key_schedule_golden_behavior_node_is_required(tmp_path: Path) -> None:
    _write_authority(tmp_path, "method_implemented")
    _write_schema_complete_method_fixture(tmp_path)
    manifest_path, manifest = _manifest(tmp_path)
    del manifest["behavioral_checks"]["key_schedule_counter_quantile_golden"]
    _write(manifest_path, json.dumps(manifest))
    report = run_audit(tmp_path)
    assert any(
        violation["reason"] == "method_behavioral_check_missing"
        for violation in report["violations"]
    )


@pytest.mark.unit
def test_trivial_assertion_cannot_satisfy_method_behavior(tmp_path: Path) -> None:
    _write_authority(tmp_path, "method_implemented")
    _write_schema_complete_method_fixture(tmp_path)
    path = tmp_path / "tests/unit/test_candidate_specific_method.py"
    source = path.read_text(encoding="utf-8")
    start = source.index("def test_hf_sparse_support():")
    end = source.index("@pytest.mark.unit", start)
    replacement = "def test_hf_sparse_support():\n    assert True\n\n"
    _write(path, source[:start] + replacement + source[end:])
    report = run_audit(tmp_path)
    reasons = {violation["reason"] for violation in report["violations"]}
    assert "method_behavioral_test_does_not_call_component_symbols" in reasons
    assert "method_behavioral_test_assertion_not_data_dependent" in reasons


@pytest.mark.unit
def test_independent_semantic_review_is_mandatory(tmp_path: Path) -> None:
    _write_authority(tmp_path, "method_implemented")
    _write_schema_complete_method_fixture(tmp_path)
    manifest_path, manifest = _manifest(tmp_path)
    manifest["independent_semantic_review"]["decision"] = "self_report"
    _write(manifest_path, json.dumps(manifest))
    report = run_audit(tmp_path)
    assert any(
        violation["reason"] == "method_independent_semantic_review_invalid"
        for violation in report["violations"]
    )


@pytest.mark.unit
def test_review_binding_fails_after_protected_change(tmp_path: Path) -> None:
    _write_authority(tmp_path, "method_implemented")
    _write_schema_complete_method_fixture(tmp_path)
    path = tmp_path / COMPONENT_PATHS["image_rectifier"]
    _write(
        path,
        path.read_text(encoding="utf-8") + "\n# changed after independent review\n",
    )
    report = run_audit(tmp_path)
    assert any(
        violation["reason"] == "method_independent_review_binding_stale"
        for violation in report["violations"]
    )
