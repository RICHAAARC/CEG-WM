"""Create-only bounded delivery for untouched soft-route confirmation."""

from __future__ import annotations

from scripts.experiment_execution.semantic_texture_soft_route_candidate_selection_server import (
    finalize_soft_route_mechanism_failure_delivery,
    finalize_soft_route_mechanism_candidate_selection_delivery,
)


def finalize_soft_route_mechanism_untouched_confirmation_delivery(*args: object, **kwargs: object) -> tuple[int, dict[str, object]]:
    """Use the same create-only result-only delivery contract for confirmation."""

    return finalize_soft_route_mechanism_candidate_selection_delivery(
        *args,
        **kwargs,
        expected_role="semantic_texture_soft_route_untouched_confirmation",
        result_filename="semantic_texture_soft_route_untouched_confirmation_result.json",
        receipt_filename="semantic_texture_soft_route_untouched_confirmation_receipt.json",
        artifact_filename="semantic_texture_soft_route_confirmation_artifact.json",
        archive_prefix="semantic_texture_soft_route_untouched_confirmation",
    )


def finalize_soft_route_mechanism_untouched_confirmation_failure_delivery(
    **kwargs: object,
) -> tuple[int, dict[str, object]]:
    return finalize_soft_route_mechanism_failure_delivery(
        **kwargs,
        result_filename="semantic_texture_soft_route_untouched_confirmation_result.json",
        receipt_filename="semantic_texture_soft_route_untouched_confirmation_receipt.json",
        archive_prefix="semantic_texture_soft_route_untouched_confirmation",
    )
