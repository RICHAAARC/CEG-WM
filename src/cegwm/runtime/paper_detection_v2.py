"""Paper V2: early return, one continuous predicted H, identical content scores."""
from cegwm.runtime.blind_detection import BlindProductionAssets, _detect_core
from cegwm.runtime.blind_scoring_v2 import score_statistic_v2


def score_current_rgb(image, key, assets):
    return score_statistic_v2(image, key, assets, reuse_observation=True)


def detect_paper_v2(image, key, assets, calibrated_tau):
    if type(assets) is not BlindProductionAssets:
        raise TypeError('paper detection requires production assets')
    return _detect_core(image, key, assets, calibrated_tau,
                        continuous_corners=True, reuse_observation=True)
