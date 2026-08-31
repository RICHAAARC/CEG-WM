"""Frozen scope registry for the Baseline-V1 generative-watermark main table."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BaselineSpec:
    """A method identity, without asserting that its source or adapter is available."""

    baseline_id: str
    display_name: str
    official_repository_url: str
    sd35_path: str
    score_direction: str | None = None
    source_status: str = "not_available"
    adapter_status: str = "not_available"
    result_status: str = "not_available"
    paper_claim_support: bool = False


PRIMARY_BASELINES: tuple[BaselineSpec, ...] = (
    BaselineSpec(
        "tree_ring",
        "Tree-Ring",
        "https://github.com/YuxinWenRick/tree-ring-watermark",
        "method_faithful_sd35_adaptation",
    ),
    BaselineSpec(
        "gaussian_shading",
        "Gaussian Shading",
        "https://github.com/bsmhmmlf/Gaussian-Shading",
        "method_faithful_sd35_adaptation",
    ),
    BaselineSpec(
        "shallow_diffuse",
        "Shallow Diffuse",
        "https://github.com/liwd190019/Shallow-Diffuse",
        "method_faithful_sd35_adaptation",
    ),
    BaselineSpec(
        "t2smark",
        "T2SMark",
        "https://github.com/0xD009/T2SMark",
        "official_run_sd35_native_path",
    ),
)


def baseline_by_id(baseline_id: str) -> BaselineSpec:
    """Resolve exactly one in-scope primary baseline."""

    for baseline in PRIMARY_BASELINES:
        if baseline.baseline_id == baseline_id:
            return baseline
    raise ValueError(f"unknown or out-of-scope baseline: {baseline_id}")
