"""CPU-only chain/RNG/cast tests: no pipeline weights or model downloads."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from cegwm.runtime import channel_precompensation_sd35 as runtime


pytestmark = pytest.mark.integration


class Processor:
    def preprocess(self, image):
        rgb = image if isinstance(image, torch.Tensor) else runtime.image_tensor(image)
        return rgb * 2 - 1

    def postprocess(self, decoded, output_type):
        rgb = (decoded / 2 + 0.5).clamp(0, 1)
        if output_type == "pt":
            return rgb
        return [runtime._to_image(runtime.quantized_rgb_proxy(rgb))]


class VAE(torch.nn.Module):
    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.8, dtype=dtype), requires_grad=False)
        self.config = SimpleNamespace(scaling_factor=1.3, shift_factor=0.04)
        self.eval()

    def encode(self, pixels):
        value = torch.tanh(pixels * self.weight)
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: value))

    def decode(self, latent, return_dict=False):
        return (torch.tanh(latent * self.weight),)


class Transformer(torch.nn.Module):
    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.2, dtype=dtype), requires_grad=False)
        self.config = SimpleNamespace(in_channels=3)
        self.eval()

    def forward(self, hidden_states, timestep, encoder_hidden_states, pooled_projections,
                joint_attention_kwargs, return_dict):
        return (hidden_states * self.weight + pooled_projections[:, :1, None, None] * 0.01,)


class Scheduler:
    order = 1

    def __init__(self):
        self.config = {"stochastic_sampling": False}
        self.begin_index = None
        self.step_index = 0
        self.extra_state = {"scalar": 17}

    def set_timesteps(self, steps):
        self.timesteps = torch.arange(steps, 0, -1).float()
        self.sigmas = torch.linspace(1, 0, steps + 1)
        self.step_index = 0

    def step(self, prediction, timestep, latent, return_dict=False):
        assert timestep == self.timesteps[self.step_index]
        sigma_step = self.sigmas[self.step_index + 1] - self.sigmas[self.step_index]
        self.step_index += 1
        return ((latent.float() + prediction.float() * sigma_step).to(latent.dtype),)


class Source:
    def __init__(self, dtype=torch.float32):
        self.transformer, self.vae = Transformer(dtype), VAE(dtype)
        self.scheduler, self.image_processor = Scheduler(), Processor()
        self.dtype = dtype
        self.joint_attention_kwargs = self.clip_skip = None
        self.mismatch = False

    @torch.no_grad()
    def __call__(self, **kwargs):
        self.guidance_scale = kwargs["guidance_scale"]
        self.do_classifier_free_guidance = self.guidance_scale > 1
        self.scheduler.set_timesteps(kwargs["num_inference_steps"])
        latent = (torch.randn(1, 3, 2, 2, generator=kwargs["generator"]) * 0.08).to(self.dtype)
        prompt = torch.ones(2, 2, 2, dtype=self.dtype)
        pooled = torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=self.dtype)
        for index, t in enumerate(self.scheduler.timesteps):
            inp = torch.cat([latent] * 2)
            prediction = self.transformer(inp, t.expand(2), prompt, pooled, None, False)[0]
            a, b = prediction.chunk(2)
            latent = self.scheduler.step(a + self.guidance_scale * (b - a), t, latent)[0]
            if self.mismatch and index == 18:
                pooled = pooled + 0.2
            state = kwargs["callback_on_step_end"](self, index, t,
                {"latents": latent, "prompt_embeds": prompt, "pooled_prompt_embeds": pooled})
            latent, prompt, pooled = state["latents"], state["prompt_embeds"], state["pooled_prompt_embeds"]
        decoded = self.vae.decode(latent / self.vae.config.scaling_factor + self.vae.config.shift_factor)[0]
        return SimpleNamespace(images=self.image_processor.postprocess(decoded, kwargs["output_type"]))


class Reconstruction:
    """Official-style no-grad wrapper with stochastic input encode + 3 layers.

    The noise variance depends on input, so detaching posterior std would also
    fail the independent analytic sensitivity assertion.
    """
    def __init__(self, detach=False, fail_at=None):
        class ReconstructionVAE(VAE):
            def encode(self, pixels):
                mean, std = 0.3*pixels, 0.02+0.01*pixels

                def sample(generator):
                    self.last_noise = torch.randn(mean.shape, generator=generator)
                    return mean + std*self.last_noise

                return SimpleNamespace(latent_dist=SimpleNamespace(sample=sample))

            def decode(self, latent, return_dict=False):
                return (0.4*latent,)

        class Denoiser(Transformer):
            def forward(self, latent):
                return 0.7*latent + 0.05*latent.square()

        self.unet, self.vae = Denoiser(), ReconstructionVAE()
        self.image_processor = Processor()
        self.watermark = None
        self.detach, self.fail_at = detach, fail_at
        self.last_noise = None

    @torch.no_grad()
    def __call__(self, *, image, generator, output_type, callback_on_step_end=None, **kwargs):
        x = self.image_processor.preprocess(image)
        posterior_sample = self.vae.encode(x).latent_dist.sample(generator)
        e = self.vae.last_noise
        n = torch.randn(x.shape, generator=generator)
        self.last_noise = (e, n)
        # Input posterior mean and std, fixed seeded reparameterization, add_noise.
        latent = posterior_sample + 0.02*n
        for step in range(3):
            if step == self.fail_at:
                raise RuntimeError("fake denoising failure")
            latent = self.unet(latent)
            if callback_on_step_end is not None:
                callback_on_step_end(self, step, torch.tensor(3-step), {"latents": latent})
        rgb = self.image_processor.postprocess(self.vae.decode(latent)[0], "pt")
        if self.detach:
            rgb = rgb.detach()
        if output_type == "pil":
            return SimpleNamespace(images=[runtime._to_image(runtime.quantized_rgb_proxy(rgb))])
        return SimpleNamespace(images=rgb)


def settings(dtype=None):
    return runtime.GenerationSettings("example", 23, 18, 0.01, 7, height=2, width=2, device="cpu")


def recon_settings():
    return runtime.ReconstructionSettings(0.3, 1, 20, device="cpu")


def image():
    return Image.fromarray(np.arange(12, dtype=np.uint8).reshape(2, 2, 3) * 6 + 80)


def vector():
    return torch.linspace(-0.2, 0.3, 12)


def test_clean_public_score_and_multilayer_reader_vjp_are_consistent():
    vae, processor, counters = VAE(), Processor(), {}
    observed = runtime.rgb_reader_vjp(image(), vae=vae, image_processor=processor,
                                     reader_vector=vector(), counters=counters)
    x = runtime.image_tensor(image())
    expected = 2 * 0.8 * 1.3 * (1 - torch.tanh((2*x-1)*0.8).square()) * vector().reshape_as(x)
    torch.testing.assert_close(observed.gradient_rgb, expected)
    assert observed.score == float(torch.dot(observed.feature.double(), vector().double()))
    assert counters == {"reader_proxy_forward_calls": 1, "reader_proxy_forward_returned": 1,
                        "reader_ordinary_forward_calls": 1, "reader_ordinary_forward_returned": 1,
                        "reader_vjp_calls": 1, "reader_vjp_returned": 1}


def test_reconstruction_vjp_traverses_reparameterization_all_steps_decode_and_reader():
    pipe, vae, processor, counters = Reconstruction(), VAE(), Processor(), {}
    out = runtime.reconstruction_reader_vjp(pipe, image(), recon_settings(), 17,
        observation_vae=vae, observation_processor=processor, reader_vector=vector(), counters=counters)
    x = runtime.image_tensor(image()) * 2 - 1
    e, n = pipe.last_noise
    current = 0.3*x + (0.02+0.01*x)*e + 0.02*n
    derivative = 2*(0.3+0.01*e)
    for _ in range(3):
        derivative = derivative * (0.7+0.1*current)
        current = 0.7*current + 0.05*current.square()
    q = runtime.image_tensor(out.image)
    expected = derivative*0.2*2*0.8*1.3*(1-torch.tanh((2*q-1)*0.8).square())*vector().reshape_as(q)
    torch.testing.assert_close(out.gradient_rgb, expected)
    assert counters["sdxl_denoising_steps"] == 3
    assert counters["sdxl_reconstruction_vjp_calls"] == 1
    assert counters["sdxl_input_vae_encode_attempted"] == counters["sdxl_input_vae_encode_returned"] == 1
    assert counters["sdxl_output_vae_decode_attempted"] == counters["sdxl_output_vae_decode_returned"] == 1
    assert counters["sdxl_unet_forward_attempted"] == counters["sdxl_unet_forward_returned"] == 3
    assert not torch.allclose(out.gradient_rgb, vector().reshape_as(q))


def test_unwrapped_gradient_forward_matches_public_no_grad_and_seeded_rng():
    pipe, vae = Reconstruction(), VAE()
    arguments = dict(observation_vae=vae, observation_processor=Processor(), reader_vector=vector())
    first = runtime.reconstruction_reader_vjp(pipe, image(), recon_settings(), 17, **arguments)
    noise = tuple(x.clone() for x in pipe.last_noise)
    second = runtime.reconstruction_reader_vjp(pipe, image(), recon_settings(), 17, **arguments)
    ordinary = pipe(image=image(), generator=torch.Generator().manual_seed(17), output_type="pil").images[0]
    assert np.array_equal(np.asarray(first.image), np.asarray(ordinary))
    assert all(torch.equal(a, b) for a, b in zip(noise, pipe.last_noise))
    assert torch.equal(first.gradient_rgb, second.gradient_rgb)
    assert torch.equal(first.diagnostics["generator_state_after"], second.diagnostics["generator_state_after"])
    third = runtime.reconstruction_reader_vjp(pipe, image(), recon_settings(), 18, **arguments)
    assert not torch.equal(first.gradient_rgb, third.gradient_rgb)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_captured_suffix_reproduces_zero_rgb_and_two_vjps_share_one_forward(dtype):
    pipe, counters = Source(dtype), {}
    capture = runtime.capture_source(pipe, settings(), counters=counters)
    assert capture.remaining_steps == (19,)
    assert capture.before_state["next_step_index"] == 19
    assert capture.scheduler_state["extra_state"] == {"scalar": 17}
    cotangents = torch.stack([torch.ones(1,3,2,2), torch.full((1,3,2,2), 0.3)])
    gradients = runtime.source_suffix_vjps(pipe, capture, cotangents, counters=counters)
    assert gradients.shape == (2,1,3,2,2)
    torch.testing.assert_close(gradients[1], gradients[0]*0.3, rtol=0.003, atol=1e-4)
    assert counters["sd35_suffix_forward_calls"] == 1
    assert counters["sd35_suffix_vjp_calls"] == 2
    assert counters["sd35_full_generation_calls"] == 1
    assert counters["sd35_full_denoising_steps"] == 20
    assert pipe.scheduler.step_index == 20  # clone, never mutate live pipeline


def test_write_is_full_replay_and_records_cast_delta_without_budget_repair():
    pipe, counters = Source(torch.float16), {}
    capture = runtime.capture_source(pipe, settings(), counters=counters)
    request = torch.full_like(capture.before_state["latents"], 1e-10, dtype=torch.float32)
    out = runtime.replay_write(pipe, settings(), capture, request, counters=counters)
    assert torch.count_nonzero(out.actual_delta) == 0
    assert out.after_latents.dtype == torch.float16
    assert torch.equal(out.actual_delta, out.after_latents.double()-capture.before_state["latents"].double())
    assert counters["sd35_full_generation_calls"] == 2
    assert counters["sd35_full_denoising_steps"] == 40


def test_write_mismatch_retains_capture_and_consumes_attempt():
    pipe, counters = Source(), {}
    capture = runtime.capture_source(pipe, settings(), counters=counters)
    pipe.mismatch = True
    with pytest.raises(runtime.RuntimeStageError) as caught:
        runtime.replay_write(pipe, settings(), capture, torch.zeros_like(capture.before_state["latents"]), counters=counters)
    assert caught.value.cause_type == "ReplayStateMismatch"
    assert caught.value.partial["before_state"]
    assert counters["sd35_full_generation_calls"] == 2


def test_suffix_forward_mismatch_is_not_silently_used():
    pipe = Source()
    capture = runtime.capture_source(pipe, settings())
    capture.image = Image.new("RGB", (2,2), (0,0,0))
    counts = {}
    with pytest.raises(runtime.RuntimeStageError, match="exactly reproduce") as caught:
        runtime.source_suffix_vjps(pipe, capture, torch.ones(2,1,3,2,2), counters=counts)
    assert "image" in caught.value.partial
    assert counts["sd35_suffix_forward_calls"] == 1
    assert counts.get("sd35_suffix_vjp_calls", 0) == 0


@pytest.mark.parametrize("mode", ["detached", "failure", "hook", "unfrozen"])
def test_invalid_reconstruction_graph_fails_with_partial_and_physical_counts(mode):
    pipe, vae, counts = Reconstruction(detach=mode=="detached", fail_at=1 if mode=="failure" else None), VAE(), {}
    if mode == "hook":
        pipe.unet._hf_hook = object()
    if mode == "unfrozen":
        vae.weight.requires_grad_(True)
    with pytest.raises(runtime.RuntimeStageError) as caught:
        runtime.reconstruction_reader_vjp(pipe, image(), recon_settings(), 17,
            observation_vae=vae, observation_processor=Processor(), reader_vector=vector(), counters=counts)
    assert caught.value.partial["input_image"].mode == "RGB"
    assert counts.get("sdxl_reconstruction_forward_calls", 0) == (mode in ("detached", "failure"))
    if mode == "failure":
        assert counts["sdxl_denoising_steps"] == 1


def test_round_boundary_exact_values_and_proxy_only_derivative():
    x = torch.tensor([0.1, 0.5, 0.9]).reshape(1,3,1,1).requires_grad_(True)
    q = runtime.quantized_rgb_proxy(x)
    assert torch.equal(q, torch.round(x.detach()*255)/255)
    gradient, = torch.autograd.grad(q.sum(), x)
    assert torch.equal(gradient, torch.ones_like(x))


def test_cpu_stashed_weights_survive_dtype_mutation_before_backward():
    linear = torch.nn.Linear(2,2,bias=False).requires_grad_(False).eval()
    original = linear.weight.detach().clone()
    x = torch.ones(1,2,requires_grad=True)
    with runtime._cpu_saved_tensors():
        y = linear(x)
        linear.to(torch.float16)
        gradient, = torch.autograd.grad(y.sum(), x)
    torch.testing.assert_close(gradient, original.sum(0).reshape(1,2))


def test_feature_allclose_does_not_hide_scalar_projection_error(monkeypatch):
    # A small aligned error in every coordinate can cross the QIM inner margin.
    actual = torch.full((65536,), 1 / 512, dtype=torch.float16)
    proxy = torch.zeros_like(actual)
    reader = torch.full((65536,), 1 / 256, dtype=torch.float64)
    monkeypatch.setattr(runtime, "encode_final_rgb_image", lambda *args: actual)
    _, score, diagnostics = runtime._actual_read(image(), proxy, VAE(), Processor(), reader, {})
    assert score == 0.5
    assert diagnostics["reader_proxy_score"] == 0
    assert diagnostics["reader_proxy_score_abs_error"] == 0.5
    assert diagnostics["reader_proxy_score_difference"] == -0.5
    assert diagnostics["reader_forward_check"] == "feature_allclose_only_not_scalar_equivalence"


def test_reused_frozen_weight_is_copied_once_with_correct_backward():
    linear = torch.nn.Linear(2, 2, bias=False).requires_grad_(False).eval()
    x = torch.ones(1, 2, requires_grad=True)
    with runtime._cpu_saved_tensors(stable_modules=(linear,)) as stats:
        y = sum(linear(x).sum() for _ in range(3))
        gradient, = torch.autograd.grad(y, x)
    torch.testing.assert_close(gradient, 3 * linear.weight.sum(0).reshape(1, 2))
    assert stats["stable_parameter_cpu_copies"] == 1
    assert stats["stable_parameter_cache_hits"] == 2
    assert stats["uncached_cpu_copies"] == 0


def test_saved_parameter_versions_do_not_reuse_pre_update_values():
    linear = torch.nn.Linear(2, 2, bias=False).requires_grad_(False).eval()
    old = linear.weight.detach().clone()
    x = torch.ones(1, 2, requires_grad=True)
    with runtime._cpu_saved_tensors(stable_modules=(linear,)) as stats:
        before = linear(x)
        linear.weight.add_(2)
        after = linear(x)
        gradient, = torch.autograd.grad((before + after).sum(), x)
    torch.testing.assert_close(gradient, (old + linear.weight).sum(0).reshape(1, 2))
    assert stats["stable_parameter_cpu_copies"] == 2
    assert stats["stable_parameter_cache_hits"] == 0


def test_saved_parameter_views_keep_distinct_shape_stride_and_gradient():
    module = torch.nn.Linear(2, 2, bias=False).requires_grad_(False).eval()
    with torch.no_grad():
        module.weight.copy_(torch.tensor([[1., 2.], [3., 4.]]))
    x = torch.ones(1, 2, requires_grad=True)
    with runtime._cpu_saved_tensors(stable_modules=(module,)) as stats:
        y = x @ module.weight + x @ module.weight.T
        gradient, = torch.autograd.grad(y.sum(), x)
    torch.testing.assert_close(gradient, torch.tensor([[7., 13.]]))
    assert stats["stable_parameter_cpu_copies"] == 2


def test_activations_are_not_cached_even_when_same_tensor_is_saved_twice():
    x = torch.tensor([0.2, 0.6], requires_grad=True)
    with runtime._cpu_saved_tensors() as stats:
        y = torch.sin(x) + torch.cos(x)
        gradient, = torch.autograd.grad(y.sum(), x)
    torch.testing.assert_close(gradient, torch.cos(x) - torch.sin(x))
    assert stats["uncached_cpu_copies"] == 2
    assert stats["stable_parameter_cache_hits"] == 0


def test_mixed_stable_weight_and_temporary_dtype_weight_keep_correct_gradients():
    stable = torch.nn.Linear(2, 2, bias=False).requires_grad_(False).eval()
    temporary = torch.nn.Linear(2, 2, bias=False).requires_grad_(False).eval()
    original = temporary.weight.detach().clone()
    x = torch.ones(1, 2, requires_grad=True)
    with runtime._cpu_saved_tensors(stable_modules=(stable,)) as stats:
        y = stable(x) + stable(x) + temporary(x)
        temporary.to(torch.float16)
        gradient, = torch.autograd.grad(y.sum(), x)
    torch.testing.assert_close(gradient, (2 * stable.weight + original).sum(0).reshape(1, 2))
    assert stats["stable_parameter_cpu_copies"] == 1
    assert stats["stable_parameter_cache_hits"] == 1
    assert stats["uncached_cpu_copies"] == 1


def test_half_precision_zero_vjp_is_reported_as_numerical_proxy_limitation():
    pipe = Source(torch.float16)
    capture = runtime.capture_source(pipe, settings())
    diagnostics = {}
    gradient = runtime.source_suffix_vjps(pipe, capture, torch.full((1, 1, 3, 2, 2), 1e-12),
                                          diagnostics=diagnostics)
    assert torch.count_nonzero(gradient) == 0
    assert diagnostics["latent_leaf_dtype"] == "torch.float16"
    assert diagnostics["gradient_rows"][0]["backward_gradient_dtype"] == "torch.float16"
    assert diagnostics["gradient_rows"][0]["all_zero"]
    assert "not proof of zero channel sensitivity" in diagnostics["zero_gradient_interpretation"]
    full_precision = Source(torch.float32)
    full_capture = runtime.capture_source(full_precision, settings())
    full_gradient = runtime.source_suffix_vjps(full_precision, full_capture,
                                              torch.full((1, 1, 3, 2, 2), 1e-12))
    assert torch.count_nonzero(full_gradient) == full_gradient.numel()
