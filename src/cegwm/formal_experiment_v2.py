"""V2 paper protocol; reuse V1 denominators, summaries and failure retention."""
import json
import math
from pathlib import Path
from dataclasses import replace
from PIL import Image
from cegwm.formal_experiment import *
from cegwm import formal_experiment as v1
from cegwm.geometry_v7.r1a import _pixel_output_to_source

EXPERIMENT_ID = 'paper-main-reconstruction-v2'
ROTATION = 'rotation_10_bilinear_black_fixed_canvas_v2'
FORMAL_CONDITIONS = (*v1.FORMAL_CONDITIONS[:-1], ROTATION)


def load_formal_config(path):
    config = json.loads(Path(path).read_text())
    if config['experiment_id'] != EXPERIMENT_ID:
        raise ValueError('different paper experiment')
    if tuple(config['conditions']) != FORMAL_CONDITIONS:
        raise ValueError('different six-condition protocol')
    return config


def expand_rosters(repo_root, config):
    return {name: tuple(replace(u, unit_id='v2-'+u.unit_id) for u in units)
            for name, units in v1.expand_rosters(repo_root, config).items()}


def apply_attack(image, condition):
    if condition != ROTATION:
        return v1.apply_attack(image, condition)
    rgb = image.convert('RGB')
    if rgb.size != (512, 512):
        raise ValueError('paper rotation requires the 512 canvas')
    angle = math.radians(10.)
    c, s = math.cos(angle), math.sin(angle)
    matrix = ((c, -s, 0.), (s, c, 0.), (0., 0., 1.))
    return rgb.transform((512, 512), Image.Transform.PERSPECTIVE,
        _pixel_output_to_source(matrix), resample=Image.Resampling.BILINEAR,
        fillcolor=(0, 0, 0))


class FormalRunStore(v1.FormalRunStore):
    def __init__(self, root, identity, unit_ids):
        # Code revision is descriptive. Keep the original run identity when resuming;
        # method/stage/job and fixed unit list still prevent mixed output directories.
        path = Path(root) / 'run_config.json'
        if path.exists():
            previous = json.loads(path.read_text())
            old = previous['identity']
            if ({k: v for k, v in old.items() if k != 'expected_exact'} ==
                {k: v for k, v in identity.items() if k != 'expected_exact'} and
                    previous['unit_ids'] == list(unit_ids)):
                identity = old
        super().__init__(root, identity, unit_ids)
