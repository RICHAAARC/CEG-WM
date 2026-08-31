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
    source_exact: str | None = None
    adapter_exact: str | None = None
    source_artifact_digest: str | None = None
    adapter_artifact_digest: str | None = None
    threshold_provenance: str | None = None
    threshold_artifact_digest: str | None = None
    source_license: str | None = None
    official_entrypoint: str | None = None
    detector_input: str | None = None
    native_score_name: str | None = None
    key_semantics: str | None = None
    wrong_key_semantics: str | None = None


PRIMARY_BASELINES: tuple[BaselineSpec, ...] = (
    BaselineSpec(
        "tree_ring",
        "Tree-Ring",
        "https://github.com/YuxinWenRick/tree-ring-watermark",
        "method_faithful_sd35_adaptation",
        score_direction="lower_is_watermarked",
        source_status="qualified",
        source_exact="3015283d9cf82e90b628f02ad2121bd37408ca9a",
        source_artifact_digest="sha256:beeba17215ad6c77e0f560ef08fa95569acdb9cbcde500e786e960f3246d439b",
        source_license="MIT",
        official_entrypoint="run_tree_ring_watermark.py + src/tree_ring_watermark/_detect.py:detect",
        detector_input="ordinary RGB image, diffusion pipeline, and public key dataset",
        native_score_name="fourier_key_l1_distance",
        key_semantics="Fourier ring array indexed by watermarked channel and radius",
        wrong_key_semantics="official detector searches any available key; no single wrong-key score is exposed",
    ),
    BaselineSpec(
        "gaussian_shading",
        "Gaussian Shading",
        "https://github.com/bsmhmmlf/Gaussian-Shading",
        "method_faithful_sd35_adaptation",
        score_direction="higher_is_watermarked",
        source_status="qualified",
        source_exact="09c678fadc7545acf7be12647ddf2a5e66f6a9dc",
        source_artifact_digest="sha256:9fc0a4e40785de0085d33745c984c13399c64c3f647ab76d5046b9f2b414c1b4",
        source_license="MIT",
        official_entrypoint="run_gaussian_shading.py + watermark.py:eval_watermark",
        detector_input="ordinary RGB image re-encoded and DDIM-inverted to a latent",
        native_score_name="watermark_bit_accuracy",
        key_semantics="watermark bits with either ChaCha20 key/nonce or random XOR key state",
        wrong_key_semantics="official evaluation holds one embedding-side key state; no independent wrong-key path is exposed",
    ),
    BaselineSpec(
        "shallow_diffuse",
        "Shallow Diffuse",
        "https://github.com/liwd190019/Shallow-Diffuse",
        "method_faithful_sd35_adaptation",
        score_direction="higher_is_watermarked",
        source_status="blocked_license_missing",
        source_exact="c80c553fdf66fda8db735d77a9d56538b7a0ade8",
        source_artifact_digest="sha256:15372b6439d4cdc2f14e50d5e5d026e41a3efd95774bcc0187d07a64b9ac78e3",
        official_entrypoint="run_shallow_diffuse_t2i.py + optim_utils.py:get_metrics",
        detector_input="ordinary RGB image, null prompt embedding, and inversion pipeline",
        native_score_name="negative_mask_l1diff_mean_or_negative_p_value",
        key_semantics="watermark mask, target patch, measurement mode, and channel",
        wrong_key_semantics="official evaluation has no independent wrong-key detector path",
    ),
    BaselineSpec(
        "t2smark",
        "T2SMark",
        "https://github.com/0xD009/T2SMark",
        "official_run_sd35_native_path",
        score_direction="higher_is_watermarked",
        source_status="qualified",
        source_exact="0c1fbfd50fcd1fba135477a2c016e284d5d7914d",
        source_artifact_digest="sha256:f8fddf5d6783bf8738daad33df3f3112c81decccc65ee1087c8ddb30446752b4",
        source_license="Apache-2.0",
        official_entrypoint="run_sd35.py",
        detector_input="ordinary RGB image re-encoded and naive-forward-diffused to a latent",
        native_score_name="norm1_w_master_key",
        key_semantics="master key controls PRNG support/signs; per-image key encodes the watermark",
        wrong_key_semantics="norm1_no_w uses fake_key = 1 - master_key on the same watermarked sample; it is not an unwatermarked arm",
    ),
)


def baseline_by_id(baseline_id: str) -> BaselineSpec:
    """Resolve exactly one in-scope primary baseline."""

    for baseline in PRIMARY_BASELINES:
        if baseline.baseline_id == baseline_id:
            return baseline
    raise ValueError(f"unknown or out-of-scope baseline: {baseline_id}")
