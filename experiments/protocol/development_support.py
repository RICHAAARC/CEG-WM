"""Neutral study-unit support shared by active diagnostic protocols."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DevelopmentStudyUnit:
    unit_index: int
    phase: str
    responsibility_id: str
    source_cluster_ordinal: int
    content_branch_id: str
    geometry_case_id: str
    maximum_record_attempts: int
    maximum_duration_seconds: int
