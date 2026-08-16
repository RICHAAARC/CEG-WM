"""LF clean-null whitening public asset validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from math import isfinite
from struct import unpack
from typing import Mapping, Sequence

from main.shared.key_schedule import stable_json_utf8


LF_NULL_WHITENED_MATCHED_SCORE_CANDIDATE_ID = (
    "lf_null_whitened_matched_score"
)
LF_NULL_WHITENING_ARTIFACT_ROLE = "lf_clean_null_whitening_operator"
LF_NULL_WHITENING_LATENT_SHAPE = (1, 16, 64, 64)
LF_NULL_WHITENING_FIT_SOURCE_CLUSTER_COUNT = 32
LF_NULL_WHITENING_DETREND_IDENTITY = (
    "per_channel_affine_plane_normalized_coordinates"
)
LF_NULL_WHITENING_TRANSFORM_IDENTITY = "orthonormal_dct_ii"
LF_NULL_WHITENING_BAND_IDENTITY = (
    "six_dyadic_chebyshev_frequency_rings_without_dc"
)
LF_NULL_WHITENING_REGULARIZATION_RATIO = "0x1.0000000000000p-10"
LF_NULL_WHITENING_WEIGHT_COUNT = 96
LF_NULL_WHITENING_OBSERVATION_PROTOCOL = "final_image_vae_posterior_mode"
SEMANTIC_TEXTURE_LF_WHITENED_CANDIDATE_ID = (
    "lf_semantic_texture_soft_whitened_matched_score"
)
SEMANTIC_TEXTURE_LF_WHITENING_ARTIFACT_ROLE = (
    "lf_semantic_texture_soft_clean_null_whitening_operator"
)


class LfNullWhiteningAssetError(ValueError):
    """The frozen public LF whitening asset is absent or invalid."""


def _sha256_text(value: object, role: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise LfNullWhiteningAssetError(
            f"{role} must be a lowercase SHA-256 digest"
        )
    return value


def _weight_words(value: object) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise LfNullWhiteningAssetError(
            "LF whitening weights must be a sequence"
        )
    words = tuple(value)
    if len(words) != LF_NULL_WHITENING_WEIGHT_COUNT:
        raise LfNullWhiteningAssetError(
            "LF whitening asset must contain exactly 96 weights"
        )
    for word in words:
        if (
            type(word) is not str
            or len(word) != 8
            or any(character not in "0123456789abcdef" for character in word)
        ):
            raise LfNullWhiteningAssetError(
                "LF whitening weight must be an eight-digit lowercase hex word"
            )
        weight = unpack(">f", bytes.fromhex(word))[0]
        if not isfinite(weight) or weight <= 0.0:
            raise LfNullWhiteningAssetError(
                "LF whitening weights must be finite and strictly positive"
            )
    return words


def _canonical_payload(
    *,
    fit_manifest_sha256: str,
    weights_binary32_be_hex: tuple[str, ...],
) -> dict[str, object]:
    return {
        "artifact_role": LF_NULL_WHITENING_ARTIFACT_ROLE,
        "band_identity": LF_NULL_WHITENING_BAND_IDENTITY,
        "candidate_id": LF_NULL_WHITENED_MATCHED_SCORE_CANDIDATE_ID,
        "detrend_identity": LF_NULL_WHITENING_DETREND_IDENTITY,
        "fit_manifest_sha256": fit_manifest_sha256,
        "fit_source_cluster_count": LF_NULL_WHITENING_FIT_SOURCE_CLUSTER_COUNT,
        "latent_shape": list(LF_NULL_WHITENING_LATENT_SHAPE),
        "observation_protocol": LF_NULL_WHITENING_OBSERVATION_PROTOCOL,
        "regularization_ratio": LF_NULL_WHITENING_REGULARIZATION_RATIO,
        "transform_identity": LF_NULL_WHITENING_TRANSFORM_IDENTITY,
        "weights_binary32_be_hex": list(weights_binary32_be_hex),
    }


@dataclass(frozen=True, slots=True)
class LfNullWhiteningAsset:
    """Digest-bound public 16-channel by 6-band whitening operator."""

    fit_manifest_sha256: str
    weights_binary32_be_hex: tuple[str, ...]
    whitening_asset_digest: str
    weights: tuple[float, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        fit_manifest_sha256 = _sha256_text(
            self.fit_manifest_sha256,
            "LF whitening fit manifest digest",
        )
        words = _weight_words(self.weights_binary32_be_hex)
        declared_digest = _sha256_text(
            self.whitening_asset_digest,
            "LF whitening asset digest",
        )
        payload = _canonical_payload(
            fit_manifest_sha256=fit_manifest_sha256,
            weights_binary32_be_hex=words,
        )
        computed_digest = sha256(stable_json_utf8(payload)).hexdigest()
        if computed_digest != declared_digest:
            raise LfNullWhiteningAssetError(
                "LF whitening asset digest does not match its canonical payload"
            )
        object.__setattr__(self, "fit_manifest_sha256", fit_manifest_sha256)
        object.__setattr__(self, "weights_binary32_be_hex", words)
        object.__setattr__(
            self,
            "weights",
            tuple(unpack(">f", bytes.fromhex(word))[0] for word in words),
        )

    @property
    def canonical_payload(self) -> dict[str, object]:
        """Return the exact stable-json payload bound by the asset digest."""

        return _canonical_payload(
            fit_manifest_sha256=self.fit_manifest_sha256,
            weights_binary32_be_hex=self.weights_binary32_be_hex,
        )

    def validate(self) -> None:
        """Revalidate bytes and digest before every detector consumption."""

        revalidated = type(self)(
            fit_manifest_sha256=self.fit_manifest_sha256,
            weights_binary32_be_hex=self.weights_binary32_be_hex,
            whitening_asset_digest=self.whitening_asset_digest,
        )
        if revalidated.weights != self.weights:
            raise LfNullWhiteningAssetError(
                "LF whitening decoded weights drifted from the frozen payload"
            )

    @classmethod
    def from_canonical_payload(
        cls,
        payload: Mapping[str, object],
        *,
        whitening_asset_digest: str,
    ) -> LfNullWhiteningAsset:
        """Load the exact registered payload without fitting or fallback."""

        if type(payload) is not dict:
            raise LfNullWhiteningAssetError(
                "LF whitening payload must be a plain mapping"
            )
        expected_keys = set(
            _canonical_payload(
                fit_manifest_sha256="0" * 64,
                weights_binary32_be_hex=("3f800000",) * 96,
            )
        )
        if set(payload) != expected_keys:
            raise LfNullWhiteningAssetError(
                "LF whitening payload fields drifted from the candidate contract"
            )
        words = _weight_words(payload["weights_binary32_be_hex"])
        fit_manifest_sha256 = _sha256_text(
            payload["fit_manifest_sha256"],
            "LF whitening fit manifest digest",
        )
        expected_payload = _canonical_payload(
            fit_manifest_sha256=fit_manifest_sha256,
            weights_binary32_be_hex=words,
        )
        if payload != expected_payload:
            raise LfNullWhiteningAssetError(
                "LF whitening payload identities drifted from the candidate contract"
            )
        return cls(
            fit_manifest_sha256=fit_manifest_sha256,
            weights_binary32_be_hex=words,
            whitening_asset_digest=whitening_asset_digest,
        )


def _semantic_texture_canonical_payload(
    *,
    fit_manifest_sha256: str,
    weights_binary32_be_hex: tuple[str, ...],
) -> dict[str, object]:
    return {
        "artifact_role": SEMANTIC_TEXTURE_LF_WHITENING_ARTIFACT_ROLE,
        "band_identity": LF_NULL_WHITENING_BAND_IDENTITY,
        "candidate_id": SEMANTIC_TEXTURE_LF_WHITENED_CANDIDATE_ID,
        "detrend_identity": LF_NULL_WHITENING_DETREND_IDENTITY,
        "fit_manifest_sha256": fit_manifest_sha256,
        "fit_source_cluster_count": LF_NULL_WHITENING_FIT_SOURCE_CLUSTER_COUNT,
        "latent_shape": list(LF_NULL_WHITENING_LATENT_SHAPE),
        "observation_protocol": LF_NULL_WHITENING_OBSERVATION_PROTOCOL,
        "regularization_ratio": LF_NULL_WHITENING_REGULARIZATION_RATIO,
        "route_candidate_id": "routing_semantic_texture_soft",
        "transform_identity": LF_NULL_WHITENING_TRANSFORM_IDENTITY,
        "weights_binary32_be_hex": list(weights_binary32_be_hex),
    }


@dataclass(frozen=True, slots=True)
class SemanticTextureLfWhiteningAsset:
    """Dedicated soft-route W identity; never aliases an older LF asset."""

    fit_manifest_sha256: str
    weights_binary32_be_hex: tuple[str, ...]
    whitening_asset_digest: str
    weights: tuple[float, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        manifest = _sha256_text(
            self.fit_manifest_sha256,
            "semantic-texture LF fit manifest digest",
        )
        words = _weight_words(self.weights_binary32_be_hex)
        declared = _sha256_text(
            self.whitening_asset_digest,
            "semantic-texture LF whitening asset digest",
        )
        payload = _semantic_texture_canonical_payload(
            fit_manifest_sha256=manifest,
            weights_binary32_be_hex=words,
        )
        if sha256(stable_json_utf8(payload)).hexdigest() != declared:
            raise LfNullWhiteningAssetError(
                "semantic-texture LF asset digest does not match its payload"
            )
        object.__setattr__(self, "fit_manifest_sha256", manifest)
        object.__setattr__(self, "weights_binary32_be_hex", words)
        object.__setattr__(
            self,
            "weights",
            tuple(unpack(">f", bytes.fromhex(word))[0] for word in words),
        )

    @property
    def canonical_payload(self) -> dict[str, object]:
        return _semantic_texture_canonical_payload(
            fit_manifest_sha256=self.fit_manifest_sha256,
            weights_binary32_be_hex=self.weights_binary32_be_hex,
        )

    def validate(self) -> None:
        replay = type(self)(
            fit_manifest_sha256=self.fit_manifest_sha256,
            weights_binary32_be_hex=self.weights_binary32_be_hex,
            whitening_asset_digest=self.whitening_asset_digest,
        )
        if replay.weights != self.weights:
            raise LfNullWhiteningAssetError(
                "semantic-texture LF decoded weights drifted"
            )

    @classmethod
    def from_canonical_payload(
        cls,
        payload: Mapping[str, object],
        *,
        whitening_asset_digest: str,
    ) -> SemanticTextureLfWhiteningAsset:
        if type(payload) is not dict:
            raise LfNullWhiteningAssetError(
                "semantic-texture LF payload must be a plain mapping"
            )
        expected_keys = set(
            _semantic_texture_canonical_payload(
                fit_manifest_sha256="0" * 64,
                weights_binary32_be_hex=("3f800000",) * 96,
            )
        )
        if set(payload) != expected_keys:
            raise LfNullWhiteningAssetError(
                "semantic-texture LF payload fields drifted"
            )
        words = _weight_words(payload["weights_binary32_be_hex"])
        manifest = _sha256_text(
            payload["fit_manifest_sha256"],
            "semantic-texture LF fit manifest digest",
        )
        expected = _semantic_texture_canonical_payload(
            fit_manifest_sha256=manifest,
            weights_binary32_be_hex=words,
        )
        if payload != expected:
            raise LfNullWhiteningAssetError(
                "semantic-texture LF payload identity drifted"
            )
        return cls(
            fit_manifest_sha256=manifest,
            weights_binary32_be_hex=words,
            whitening_asset_digest=whitening_asset_digest,
        )


__all__ = [
    "LF_NULL_WHITENED_MATCHED_SCORE_CANDIDATE_ID",
    "LF_NULL_WHITENING_ARTIFACT_ROLE",
    "LF_NULL_WHITENING_BAND_IDENTITY",
    "LF_NULL_WHITENING_DETREND_IDENTITY",
    "LF_NULL_WHITENING_FIT_SOURCE_CLUSTER_COUNT",
    "LF_NULL_WHITENING_LATENT_SHAPE",
    "LF_NULL_WHITENING_OBSERVATION_PROTOCOL",
    "LF_NULL_WHITENING_REGULARIZATION_RATIO",
    "LF_NULL_WHITENING_TRANSFORM_IDENTITY",
    "LF_NULL_WHITENING_WEIGHT_COUNT",
    "LfNullWhiteningAsset",
    "LfNullWhiteningAssetError",
    "SEMANTIC_TEXTURE_LF_WHITENED_CANDIDATE_ID",
    "SEMANTIC_TEXTURE_LF_WHITENING_ARTIFACT_ROLE",
    "SemanticTextureLfWhiteningAsset",
]
