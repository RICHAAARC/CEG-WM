"""Shared external-baseline canary transaction.

The T2SMark implementation is intentionally reused unchanged: its transaction
is method-neutral and its identity is fully supplied by the caller's config.
"""
from .t2smark_canary import *  # noqa: F401,F403

RUN_SCHEMA = "cegwm.external_baseline_sd35_canary.v1"
RUN_ID_DEFAULTS = {"tree_ring": "tree_ring_sd35_one_unit_v1", "gaussian_shading": "gaussian_shading_sd35_one_unit_v1", "shallow_diffuse": "shallow_diffuse_sd35_one_unit_v1"}
