"""Development-only paired geometry from one detector observation."""
from cegwm.geometry_v7.contracts import (
    GeometryEstimate, estimate_geometry, syncseal_raw_to_public_continuous,
)


def estimate_pair(backend, image):
    rounded=backend.detect_geometry(image)
    try:
        if rounded.raw_syncseal_corners is None:
            raise ValueError('raw geometry unavailable')
        continuous=estimate_geometry(rounded.uncalibrated_sync_logit,
            syncseal_raw_to_public_continuous(rounded.raw_syncseal_corners),
            raw_syncseal_corners=rounded.raw_syncseal_corners)
    except Exception as error:
        continuous=GeometryEstimate.error_record(error)
    return {'rounded':rounded,'continuous':continuous}
