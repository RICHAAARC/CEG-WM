from types import SimpleNamespace

import pytest
import torch

from cegwm.baselines.sd35_runtime import InversionStableDiffusion3PipelineMixin


class _Pipe(InversionStableDiffusion3PipelineMixin):
    _execution_device = "cpu"
    def __init__(self, stochastic: bool = False) -> None:
        self.scheduler = SimpleNamespace(config=SimpleNamespace(use_dynamic_shifting=False, stochastic_sampling=stochastic))
        self.scheduler.set_timesteps = self._set
    def _set(self, steps, device, **kwargs):
        self.scheduler.timesteps = torch.arange(steps)
        self.scheduler.sigmas = torch.tensor([4.0, 3.0, 2.0, 1.0, 0.0])
    def _conditioning(self, prompt, guidance): return torch.zeros(1), torch.zeros(1), False
    def _velocity(self, *args, **kwargs): return torch.ones_like(args[0])


def test_denoise_uses_forward_euler_sigma_next_minus_current() -> None:
    pipe = _Pipe(); result = pipe.denoise_segment(torch.zeros((1,1,1,1)), prompt="x", guidance=4.5, steps=4, start=0, end=2)
    assert result.item() == -2.0
    post = pipe.denoise_segment(torch.zeros((1,1,1,1)), prompt="x", guidance=1.0, steps=4, start=2, end=4)
    assert post.item() == -2.0


def test_schedule_rejects_stochastic_configuration() -> None:
    with pytest.raises(RuntimeError, match="deterministic"):
        _Pipe(stochastic=True)._schedule(4, (1,16,64,64))
