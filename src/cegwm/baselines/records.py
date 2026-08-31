"""Method-neutral, evidence-preserving records for baseline observations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Mapping

from cegwm.baselines.registry import baseline_by_id


@dataclass(frozen=True)
class BaselineObservation:
    """One observed detector score; not a scientific result by itself.

    Source and adapter exacts are retained per record.  A method may decide only
    with its own calibrated threshold provenance; no shared numeric threshold is
    represented by this schema.
    """

    baseline_id: str
    source_exact: str | None
    adapter_exact: str | None
    prompt_id: str
    seed: int
    base_latent_commitment: str
    split: str
    sample_role: str
    attack_id: str
    continuous_score: float | None
    score_direction: str | None
    threshold_provenance: str | None
    decision: bool | None
    quality: Mapping[str, float]
    runtime_seconds: float | None
    status: str
    failure_reason: str | None
    artifact_digests: Mapping[str, str]

    def as_dict(self) -> dict[str, object]:
        """Validate before returning data suitable for an append-only JSONL writer."""

        return asdict(validate_observation(self))


def validate_observation(observation: BaselineObservation) -> BaselineObservation:
    """Fail closed on identity, decision, and artifact-boundary violations."""

    baseline = baseline_by_id(observation.baseline_id)
    if not observation.prompt_id or not observation.base_latent_commitment:
        raise ValueError("prompt_id and base_latent_commitment are required")
    if observation.seed < 0:
        raise ValueError("seed must be non-negative")
    if observation.status not in {"observed", "failed", "not_available"}:
        raise ValueError("status is not recognized")
    if observation.status == "observed":
        required = (observation.source_exact, observation.adapter_exact, observation.threshold_provenance)
        if any(not value for value in required):
            raise ValueError("observed records require source, adapter, and threshold identities")
        if observation.continuous_score is None or not isinstance(observation.decision, bool):
            raise ValueError("observed records require a continuous score and decision")
        if baseline.source_status != "validated" or baseline.adapter_status != "validated":
            raise ValueError("observed records require validated source and adapter registry entries")
        registry_identities = (
            baseline.source_exact,
            baseline.adapter_exact,
            baseline.source_artifact_digest,
            baseline.adapter_artifact_digest,
            baseline.threshold_provenance,
            baseline.threshold_artifact_digest,
        )
        if any(value is None for value in registry_identities):
            raise ValueError("observed records require registry-bound source, adapter, and threshold identities")
        if baseline.score_direction is None:
            raise ValueError("method detector score direction is unresolved")
        if observation.score_direction != baseline.score_direction:
            raise ValueError("score_direction must equal the method-declared direction")
        expected_prefix = f"{observation.baseline_id}:calibration:"
        if not observation.threshold_provenance.startswith(expected_prefix):
            raise ValueError("threshold provenance must be bound to the baseline")
        if not re.fullmatch(r"[0-9a-f]{40}", observation.source_exact):
            raise ValueError("source_exact must be a lowercase 40-character git exact")
        if not re.fullmatch(r"[0-9a-f]{40}", observation.adapter_exact):
            raise ValueError("adapter_exact must be a lowercase 40-character git exact")
        if observation.source_exact != baseline.source_exact:
            raise ValueError("source_exact must match the registry")
        if observation.adapter_exact != baseline.adapter_exact:
            raise ValueError("adapter_exact must match the registry")
        if observation.threshold_provenance != baseline.threshold_provenance:
            raise ValueError("threshold provenance must match the registry")
    elif observation.status in {"failed", "not_available"}:
        if any(value is not None for value in (observation.continuous_score, observation.score_direction,
                                                observation.threshold_provenance, observation.decision)):
            raise ValueError("failed or unavailable records cannot carry detection evidence")
    if observation.status == "failed" and not observation.failure_reason:
        raise ValueError("failed records require failure_reason")
    if not observation.artifact_digests:
        raise ValueError("artifact_digests are required, including for failed units")
    if any(not name or not re.fullmatch(r"sha256:[0-9a-f]{64}", digest)
           for name, digest in observation.artifact_digests.items()):
        raise ValueError("artifact digests must be named sha256 values")
    if observation.status == "observed" and not {"source", "adapter", "threshold"}.issubset(observation.artifact_digests):
        raise ValueError("observed records require source, adapter, and threshold artifact digests")
    if observation.status == "observed" and (
        observation.artifact_digests["source"] != baseline.source_artifact_digest
        or observation.artifact_digests["adapter"] != baseline.adapter_artifact_digest
        or observation.artifact_digests["threshold"] != baseline.threshold_artifact_digest
    ):
        raise ValueError("observed artifact digests must match the registry")
    return observation
