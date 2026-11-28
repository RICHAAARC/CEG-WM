"""Checkpoint compatibility copied from the already used discrete_pair_v1 loader.

Only a temporary config view is changed; original weights/cache are immutable.
"""
import contextlib
import json
from pathlib import Path
import signal
import tempfile


def ensure_vq_component_compatibility():
    import diffusers.pipelines as pipelines
    if not hasattr(pipelines,'vq_diffusion'):
        from diffusers.pipelines.deprecated import vq_diffusion
        pipelines.vq_diffusion = vq_diffusion


@contextlib.contextmanager
def compatible_checkpoint(snapshot):
    snapshot = Path(snapshot).resolve()
    config = json.loads((snapshot/'transformer'/'config.json').read_text())
    if 'norm_type' not in config:
        if (config.get('num_embeds_ada_norm') is None or config.get('num_vector_embeds') is None
                or config.get('in_channels') is not None):
            raise ValueError('Unknown legacy config: refusing to infer norm_type')
        config['norm_type'] = 'ada_norm'
    with tempfile.TemporaryDirectory(prefix='gds-vq-compat-') as directory:
        view = Path(directory)
        for original in snapshot.iterdir():
            if original.name != 'transformer':
                (view/original.name).symlink_to(original.resolve(),target_is_directory=original.is_dir())
        (view/'transformer').mkdir()
        for original in (snapshot/'transformer').iterdir():
            if original.name != 'config.json':
                (view/'transformer'/original.name).symlink_to(original.resolve(),target_is_directory=original.is_dir())
        (view/'transformer'/'config.json').write_text(json.dumps(config,indent=2))
        yield view


@contextlib.contextmanager
def source_timeout(seconds):
    if not hasattr(signal,'SIGALRM'):
        raise RuntimeError('Linux/Colab required')
    def expired(signum,frame):
        raise TimeoutError(f'G0 generation exceeded {seconds} seconds')
    old = signal.signal(signal.SIGALRM,expired)
    signal.alarm(int(seconds))
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM,old)
