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
    partial = caught.value.partial
    assert torch.equal(partial["suffix_before_latents"], capture.before_state["latents"])
    assert len(partial["scheduler_step_outputs"]) == 1
    assert partial["scheduler_step_outputs"][0]["index"] == 19
    assert partial["pre_vae_latents"].dtype == capture.before_state["latents"].dtype
    diagnostics = partial["source_suffix_diagnostics"]
    assert diagnostics["schedule_restore"]["captured_latent_device"] == "cpu"
    assert set(diagnostics["schedule_restore"]["tensors"]) == {"timesteps", "sigmas"}
    assert not diagnostics["zero_rgb_comparison"]["exact_equal"]
    assert diagnostics["zero_rgb_comparison"]["max_abs_uint8_difference"] > 0
    assert counts["sd35_suffix_forward_calls"] == 1
    assert counts.get("sd35_suffix_vjp_calls", 0) == 0


def test_runtime_suffix_basepoint_is_explicit_independent_and_shares_two_vjps():
    pipe = Source()
    capture = runtime.capture_source(pipe, settings())
    historical = Image.new("RGB", (2, 2), (0, 0, 0))
    capture.image = historical
    zero_counts, zero_diagnostics = {}, {}
    zero = runtime.replay_suffix_write(pipe, capture, counters=zero_counts, diagnostics=zero_diagnostics)
    assert capture.image is historical
    assert not zero_diagnostics["zero_rgb_comparison"]["exact_equal"]
    assert torch.equal(zero.after_latents, capture.before_state["latents"])
    assert zero.actual_delta.dtype == torch.float64
    assert torch.count_nonzero(zero.actual_delta) == 0
    assert zero.injection_metrics["requested_l2"] == zero.injection_metrics["actual_l2"] == 0
    assert zero_counts["sd35_suffix_write_calls"] == zero_counts["sd35_suffix_write_returned"] == 1
    assert zero_counts["sd35_suffix_forward_calls"] == 1
    assert zero_counts.get("sd35_full_generation_calls", 0) == 0

    cotangents = torch.stack([torch.ones(1, 3, 2, 2), torch.full((1, 3, 2, 2), 0.3)])
    counts, diagnostics = {}, {}
    gradients = runtime.source_suffix_vjps(pipe, capture, cotangents, counters=counts,
        diagnostics=diagnostics, basepoint_mode="runtime_suffix", reference_image=zero.image)
    # Analytic derivative of this fake last Euler step and tanh decoder;
    # uint8 rounding alone has the adopted straight-through derivative.
    x = capture.before_state["latents"]
    predicted = x * 0.2 + (0.1 + 7 * (0.3 - 0.1)) * 0.01
    final = x - predicted * 0.05
    expected = 0.5 * 0.8 / 1.3 * (1 - torch.tanh((final / 1.3 + 0.04) * 0.8).square()) * 0.99
    torch.testing.assert_close(gradients[0], expected)
    torch.testing.assert_close(gradients[1], expected * 0.3)
    assert diagnostics["basepoint_mode"] == "runtime_suffix"
    assert diagnostics["basepoint_rgb_comparison"]["exact_equal"]
    assert not diagnostics["zero_rgb_comparison"]["exact_equal"]
    assert counts["sd35_suffix_forward_calls"] == counts["sd35_suffix_decode_calls"] == 1
    assert counts["sd35_suffix_vjp_calls"] == counts["sd35_suffix_vjp_returned"] == 2
    with pytest.raises(runtime.RuntimeStageError, match="captured zero RGB"):
        runtime.source_suffix_vjps(pipe, capture, cotangents)


@pytest.mark.parametrize("case", ["unknown_mode", "missing_reference", "legacy_replacement", "wrong_reference"])
def test_suffix_basepoint_errors_cannot_silently_bypass_reference(case):
    pipe = Source()
    capture = runtime.capture_source(pipe, settings())
    kwargs = {"basepoint_mode": "runtime_suffix"}
    if case == "unknown_mode":
        kwargs["basepoint_mode"] = "skip_check"
    elif case == "legacy_replacement":
        kwargs = {"reference_image": capture.image}
    elif case == "wrong_reference":
        kwargs["reference_image"] = Image.new("RGB", (2, 2), (0, 0, 0))
    counts, diagnostics = {}, {}
    with pytest.raises(runtime.RuntimeStageError) as caught:
        runtime.source_suffix_vjps(pipe, capture, torch.ones(2, 1, 3, 2, 2),
                                  counters=counts, diagnostics=diagnostics, **kwargs)
    assert counts.get("sd35_suffix_vjp_calls", 0) == 0
    assert counts.get("sd35_suffix_forward_calls", 0) == (case == "wrong_reference")
    if case == "wrong_reference":
        assert "image" in caught.value.partial
        assert not diagnostics["basepoint_rgb_comparison"]["exact_equal"]


@pytest.mark.parametrize("amplitude", [1e-10, 0.015])
def test_suffix_write_matches_same_cast_full_replay_without_full_call_or_backward(monkeypatch, amplitude):
    pipe = Source(torch.float16)
    capture = runtime.capture_source(pipe, settings())
    before = deepcopy(capture.before_state)
    saved_scheduler = deepcopy(capture.scheduler_state)
    request = torch.full_like(before["latents"], amplitude, dtype=torch.float32)
    full = runtime.replay_write(pipe, settings(), capture, request)
    transform = pipe.transformer.forward
    calls = []

    def forward(*args, **kwargs):
        calls.append(torch.is_grad_enabled())
        return transform(*args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("suffix writer must not use full generation or backward")

    monkeypatch.setattr(pipe.transformer, "forward", forward)
    monkeypatch.setattr(Source, "__call__", forbidden)
    monkeypatch.setattr(torch.autograd, "grad", forbidden)
    counts, diagnostics = {}, {}
    written = runtime.replay_suffix_write(pipe, capture, request, counters=counts, diagnostics=diagnostics)
    assert calls == [False]
    assert np.array_equal(np.asarray(written.image), np.asarray(full.image))
    assert torch.equal(written.actual_delta, full.actual_delta)
    assert torch.equal(written.after_latents, full.after_latents)
    assert written.after_latents.dtype == torch.float16
    assert written.actual_delta.dtype == torch.float64
    assert written.before_state is capture.before_state
    assert runtime._different_paths(before, capture.before_state) == []
    assert runtime._different_paths(saved_scheduler, capture.scheduler_state) == []
    assert pipe.scheduler.step_index == 20
    assert written.remaining_steps == (19,)
    assert counts["sd35_suffix_denoising_steps"] == 1
    assert counts["sd35_suffix_write_calls"] == counts["sd35_suffix_write_returned"] == 1
    assert counts.get("sd35_full_generation_calls", 0) == counts.get("sd35_suffix_vjp_calls", 0) == 0
    assert diagnostics["write_path"] == "fixed_capture_suffix"
    if amplitude == 1e-10:
        assert written.injection_metrics["requested_l2"] > 0
        assert written.injection_metrics["actual_l2"] == 0
    else:
        assert written.injection_metrics["actual_l2"] > 0


def test_suffix_write_failure_retains_actual_cast_without_retry(monkeypatch):
    pipe = Source(torch.float16)
    capture = runtime.capture_source(pipe, settings())
    request = torch.full_like(capture.before_state["latents"], 0.015, dtype=torch.float32)

    def broken(*args, **kwargs):
        raise RuntimeError("one failing transformer")

    monkeypatch.setattr(pipe.transformer, "forward", broken)
    counts = {}
    with pytest.raises(runtime.RuntimeStageError, match="one failing transformer") as caught:
        runtime.replay_suffix_write(pipe, capture, request, counters=counts)
    assert caught.value.stage == "source_suffix_write"
    partial = caught.value.partial
    expected = (capture.before_state["latents"].float() + request).half()
    assert torch.equal(partial["after_latents"], expected)
    assert torch.equal(partial["actual_delta"], expected.double() - capture.before_state["latents"].double())
    assert counts["sd35_suffix_write_calls"] == counts["sd35_suffix_forward_calls"] == 1
    assert counts.get("sd35_suffix_write_returned", 0) == 0


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


def real_processor():
    # This initializes only the actual preprocessing class, never a model.
    module = pytest.importorskip("diffusers.image_processor")
    return module.VaeImageProcessor(vae_scale_factor=1)


def test_real_pil_tensor_layout_canonicalization_preserves_values_and_derivative():
    processor = real_processor()
    picture = Image.fromarray(np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3))
    rgb = runtime.image_tensor(picture).requires_grad_(True)
    ordinary = processor.preprocess(picture)
    processed = processor.preprocess(rgb)
    diagnostics = {}
    canonical = runtime._canonical_preprocessed_pixels(processed, ordinary, diagnostics=diagnostics)
    assert ordinary.is_contiguous(memory_format=torch.channels_last)
    assert canonical.stride() == ordinary.stride()
    assert torch.equal(canonical.detach(), ordinary)
    assert diagnostics["proxy_before"]["stride"] == list(processed.stride())
    weights = torch.linspace(-1, 1, rgb.numel()).reshape_as(rgb)
    derivative, = torch.autograd.grad((canonical * weights).sum(), rgb)
    torch.testing.assert_close(derivative, 2 * weights, rtol=0, atol=0)


@pytest.mark.parametrize("change", ["values", "dtype", "shape"])
def test_layout_canonicalization_never_repairs_wrong_preprocessed_values(change):
    processor = real_processor()
    ordinary = processor.preprocess(image())
    proxy = processor.preprocess(runtime.image_tensor(image()).requires_grad_(True))
    if change == "values":
        proxy = proxy + 0.01
    elif change == "dtype":
        proxy = proxy.half()
    else:
        proxy = proxy[:, :, :1, :]
    diagnostics = {}
    with pytest.raises(ValueError, match="preprocessing"):
        runtime._canonical_preprocessed_pixels(proxy, ordinary, diagnostics=diagnostics)
    assert "ordinary" in diagnostics and "proxy_before" in diagnostics
    if change == "values":
        assert not diagnostics["values_equal"]
        assert diagnostics["max_abs_error"] > 0.009


def test_real_processor_clean_reader_uses_identical_pil_layout_without_changing_public_read():
    class RecordingVAE(VAE):
        def __init__(self):
            super().__init__()
            self.input_records = []

        def encode(self, pixels):
            self.input_records.append((pixels.stride(), pixels.dtype, torch.is_grad_enabled()))
            return super().encode(pixels)

    vae, processor = RecordingVAE(), real_processor()
    observed = runtime.rgb_reader_vjp(image(), vae=vae, image_processor=processor, reader_vector=vector())
    expected_stride = processor.preprocess(image()).stride()
    assert vae.input_records == [(expected_stride, torch.float32, True),
                                 (expected_stride, torch.float32, False)]
    assert observed.diagnostics["reader_vae_input"]["stride"] == list(expected_stride)
    assert observed.diagnostics["reader_proxy_score_abs_error"] == 0
    assert observed.diagnostics["compatibility_version"] == runtime.RGB_PROXY_COMPATIBILITY_VERSION
    assert observed.diagnostics["reader_ordinary_execution"]["grad_enabled"] is False
    assert torch.count_nonzero(observed.gradient_rgb) == observed.gradient_rgb.numel()


@pytest.mark.parametrize("fail", [False, True])
def test_real_processor_sdxl_wrapper_canonicalizes_input_and_restores_method(fail):
    pipe = Reconstruction(fail_at=1 if fail else None)
    pipe.image_processor = real_processor()
    original_preprocess = pipe.image_processor.preprocess
    expected = original_preprocess(image())
    original_encode = pipe.vae.encode
    seen = []

    def recording_encode(pixels):
        seen.append((pixels.stride(), pixels.detach().clone(), pixels.requires_grad))
        return original_encode(pixels)

    pipe.vae.encode = recording_encode
    kwargs = dict(observation_vae=VAE(), observation_processor=real_processor(), reader_vector=vector())
    if fail:
        with pytest.raises(runtime.RuntimeStageError, match="fake denoising failure") as caught:
            runtime.reconstruction_reader_vjp(pipe, image(), recon_settings(), 17, **kwargs)
        diagnostics = caught.value.partial["reader_diagnostics"]
    else:
        result = runtime.reconstruction_reader_vjp(pipe, image(), recon_settings(), 17, **kwargs)
        diagnostics = result.diagnostics
        assert torch.count_nonzero(result.gradient_rgb) > 0
    assert seen[0][0] == expected.stride()
    assert torch.equal(seen[0][1], expected)
    assert seen[0][2]
    assert diagnostics["sdxl_input_preprocess"]["proxy_after"]["stride"] == list(expected.stride())
    assert diagnostics["sdxl_vae_encode_inputs"][0]["tensor"]["stride"] == list(expected.stride())
    assert diagnostics["sdxl_vae_encode_inputs"][0]["execution"]["grad_enabled"]
    assert pipe.image_processor.preprocess == original_preprocess
    assert "preprocess" not in pipe.image_processor.__dict__


def test_sdxl_wrong_tensor_preprocessing_still_fails_and_restores_existing_method():
    pipe = Reconstruction()
    processor = real_processor()
    original = processor.preprocess

    def wrong_preprocess(value, *args, **kwargs):
        result = original(value, *args, **kwargs)
        return result + 0.01 if isinstance(value, torch.Tensor) else result

    processor.preprocess = wrong_preprocess
    pipe.image_processor = processor
    counters = {}
    with pytest.raises(runtime.RuntimeStageError, match="does not agree exactly") as caught:
        runtime.reconstruction_reader_vjp(pipe, image(), recon_settings(), 17,
            observation_vae=VAE(), observation_processor=real_processor(), reader_vector=vector(), counters=counters)
    assert processor.preprocess is wrong_preprocess
    assert counters.get("sdxl_input_vae_encode_attempted", 0) == 0
    assert caught.value.partial["reader_diagnostics"]["sdxl_input_preprocess"]["values_equal"] is False


def test_reader_allclose_failure_preserves_features_score_error_and_input_metadata():
    class ExecutionDriftVAE(VAE):
        def encode(self, pixels):
            value = super().encode(pixels).latent_dist.mode()
            if torch.is_grad_enabled():
                value = value + 0.01
            return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: value))

    counters = {}
    with pytest.raises(runtime.RuntimeStageError, match="reader forward disagree") as caught:
        runtime.rgb_reader_vjp(image(), vae=ExecutionDriftVAE(), image_processor=real_processor(),
                               reader_vector=vector(), counters=counters)
    partial = caught.value.partial
    assert partial["reader_proxy_feature"].shape == partial["reader_ordinary_feature"].shape
    diagnostics = partial["reader_diagnostics"]
    assert diagnostics["reader_proxy_max_abs_error"] > 0.01
    assert diagnostics["reader_proxy_score_abs_error"] > 0
    assert not diagnostics["reader_feature_allclose"]
    assert diagnostics["reader_preprocess"]["values_equal"]
    assert "sdpa_enabled_policies_not_selected_kernel" in diagnostics["reader_proxy_execution"]
    assert counters.get("reader_vjp_calls", 0) == 0


def test_obtained_gradient_is_saved_even_if_nonfinite_validation_fails(monkeypatch):
    def invalid_gradient(output, inputs, **kwargs):
        return (torch.full_like(inputs, float("nan")),)

    monkeypatch.setattr(torch.autograd, "grad", invalid_gradient)
    with pytest.raises(runtime.RuntimeStageError, match="reader RGB sensitivity") as caught:
        runtime.rgb_reader_vjp(image(), vae=VAE(), image_processor=real_processor(), reader_vector=vector())
    assert torch.isnan(caught.value.partial["gradient_rgb"]).all()
    assert caught.value.partial["diagnostics"]["reader_feature_allclose"]


def test_source_schedule_restores_both_tensor_devices_without_dtype_changes():
    # Meta transfers exercise actual .to(device) dispatch without a GPU/model.
    # A regression restoring only timesteps leaves sigmas on CPU and fails here.
    timesteps = torch.tensor([3.0, 2.0, 1.0], dtype=torch.float32)
    sigmas = torch.tensor([0.3, 0.2, 0.1, 0.0], dtype=torch.float32)
    scheduler = SimpleNamespace(timesteps=timesteps, sigmas=sigmas)
    diagnostics = {}
    state = {"latent_device": "meta", "scheduler_class": "test.schedule"}
    runtime._restore_source_schedule_devices(scheduler, state, diagnostics)
    assert scheduler.timesteps.device.type == scheduler.sigmas.device.type == "meta"
    assert scheduler.timesteps.dtype == timesteps.dtype and scheduler.sigmas.dtype == sigmas.dtype
    assert scheduler.timesteps.stride() == timesteps.stride() and scheduler.sigmas.stride() == sigmas.stride()
    assert timesteps.device.type == sigmas.device.type == "cpu"
    assert diagnostics["tensors"]["sigmas"]["saved"]["device"] == "cpu"
    assert diagnostics["tensors"]["sigmas"]["restored"]["device"] == "meta"
    assert diagnostics["inferred_for_legacy_capture"] is True


def test_actual_flowmatch_schedule_keeps_requested_device_through_set_timesteps():
    # Actual official scheduler only: no pipeline weights or denoising model.
    module = pytest.importorskip("diffusers.schedulers.scheduling_flow_match_euler_discrete")
    scheduler = module.FlowMatchEulerDiscreteScheduler()
    scheduler.set_timesteps(20, device="meta")
    assert scheduler.timesteps.device.type == scheduler.sigmas.device.type == "meta"


def test_source_diagnostics_preserve_original_capture_schema_and_dtype():
    pipe = Source(torch.float16)
    capture = runtime.capture_source(pipe, settings())
    state_keys = tuple(capture.before_state)
    scheduler_keys = tuple(capture.scheduler_state)
    saved_sigmas = capture.scheduler_state["sigmas"].clone()
    diagnostics = {}
    result = runtime.source_suffix_vjps(pipe, capture, torch.ones(2, 1, 3, 2, 2), diagnostics=diagnostics)
    assert result.shape[0] == 2
    assert diagnostics["before_latents"]["dtype"] == "torch.float16"
    assert diagnostics["pre_vae_latents"]["dtype"] == "torch.float16"
    assert diagnostics["steps"][0]["scheduler_output"]["dtype"] == "torch.float16"
    assert diagnostics["zero_rgb_comparison"]["exact_equal"]
    assert tuple(capture.before_state) == state_keys and tuple(capture.scheduler_state) == scheduler_keys
    assert torch.equal(capture.scheduler_state["sigmas"], saved_sigmas)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_trace_ng_ng_g_preserves_inputs_layers_and_never_calls_backward(dtype, monkeypatch):
    pipe = Source(dtype)
    capture = runtime.capture_source(pipe, settings())
    original_latents = capture.before_state["latents"].clone()
    original_sigma = capture.scheduler_state["sigmas"].clone()
    original_live_step = pipe.scheduler.step_index
    original_grad_mode = torch.is_grad_enabled()
    counters = {}
    monkeypatch.setattr(torch.autograd, "grad", lambda *args, **kwargs: pytest.fail("trace performed a VJP"))
    results = [runtime.trace_source_suffix(pipe, capture, grad_enabled=enabled, counters=counters)
               for enabled in (False, False, True)]
    for result, enabled in zip(results, (False, False, True)):
        assert result["metadata"]["leaf_requires_grad"] is True
        assert result["metadata"]["grad_enabled"] is enabled
        assert result["metadata"]["execution"]["grad_enabled"] is enabled
        assert result["metadata"]["decoded_requires_grad"] is enabled
        assert result["metadata"]["steps"][0]["guided_prediction_requires_grad"] is enabled
        assert result["metadata"]["scheduler_step_index_before"] == 19
        assert result["metadata"]["scheduler_step_index_after"] == 20
        assert result["metadata"]["zero_rgb_comparison"]["exact_equal"]
        assert result["metadata"]["module_modes"] == {
            "transformer_training": False, "vae_training": False,
            "transformer_parameters_require_grad": False, "vae_parameters_require_grad": False}
        assert torch.equal(result["before_latents"], original_latents)
        assert result["before_latents"].stride() == original_latents.stride()
        for name in ("before_latents", "pre_vae_latents", "decoded", "postprocessed_rgb"):
            tensor = result[name]
            assert tensor.device.type == "cpu" and tensor.dtype == dtype and not tensor.requires_grad
            assert torch.equal(tensor, results[0][name])
        for name in ("guided_prediction", "scheduler_output"):
            tensor = result["steps"][0][name]
            assert tensor.device.type == "cpu" and tensor.dtype == dtype and not tensor.requires_grad
            assert torch.equal(tensor, results[0]["steps"][0][name])
        assert result["steps"][0]["timestep"].dtype == capture.before_state["timesteps"].dtype
        assert result["uint8_rgb"].dtype == torch.uint8 and result["uint8_rgb"].shape == (2, 2, 3)
        assert np.array_equal(result["uint8_rgb"].numpy(), np.asarray(result["image"]))
    assert results[0]["metadata"]["saved_tensors"]["stable_parameter_cpu_copies"] == 0
    assert results[2]["metadata"]["saved_tensors"]["stable_parameter_cpu_copies"] > 0
    assert pipe.scheduler.step_index == original_live_step
    assert torch.equal(capture.before_state["latents"], original_latents)
    assert torch.equal(capture.scheduler_state["sigmas"], original_sigma)
    assert torch.is_grad_enabled() is original_grad_mode
    assert counters == {"sd35_suffix_forward_calls": 3, "sd35_suffix_denoising_steps": 3,
                        "sd35_suffix_decode_calls": 3, "sd35_suffix_decode_returned": 3,
                        "sd35_suffix_forward_returned": 3}


def test_trace_keeps_mismatched_rgb_while_production_uses_same_forward_and_still_rejects():
    pipe = Source(torch.float16)
    capture = runtime.capture_source(pipe, settings())
    capture.image = Image.new("RGB", (2, 2), (0, 0, 0))
    trace = runtime.trace_source_suffix(pipe, capture, grad_enabled=True)
    assert not trace["metadata"]["zero_rgb_comparison"]["exact_equal"]
    assert "decoded" in trace and "uint8_rgb" in trace
    counters = {}
    with pytest.raises(runtime.RuntimeStageError, match="exactly reproduce") as caught:
        runtime.source_suffix_vjps(pipe, capture, torch.ones(2, 1, 3, 2, 2), counters=counters)
    partial = caught.value.partial
    for name in ("before_latents", "pre_vae_latents", "decoded", "postprocessed_rgb", "uint8_rgb"):
        assert torch.equal(trace[name], partial[name])
    assert torch.equal(trace["steps"][0]["guided_prediction"], partial["steps"][0]["guided_prediction"])
    assert counters.get("sd35_suffix_vjp_calls", 0) == 0


def test_trace_failure_preserves_prediction_before_failed_scheduler_step(monkeypatch):
    pipe = Source()
    capture = runtime.capture_source(pipe, settings())
    prior_mode = torch.is_grad_enabled()

    def fail_step(*args, **kwargs):
        raise RuntimeError("fake scheduler execution failure")

    monkeypatch.setattr(Scheduler, "step", fail_step)
    counters = {}
    with pytest.raises(runtime.RuntimeStageError, match="fake scheduler execution failure") as caught:
        runtime.trace_source_suffix(pipe, capture, grad_enabled=False, counters=counters)
    assert caught.value.stage == "source_suffix_trace"
    partial = caught.value.partial
    assert partial["before_latents"].device.type == "cpu"
    assert partial["steps"][0]["guided_prediction"].device.type == "cpu"
    assert "scheduler_output" not in partial["steps"][0]
    assert "decoded" not in partial
    assert partial["metadata"]["grad_enabled"] is False
    assert torch.is_grad_enabled() is prior_mode
    assert counters == {"sd35_suffix_forward_calls": 1, "sd35_suffix_denoising_steps": 1}


def test_trace_failure_preserves_pre_vae_latents_before_decode_error(monkeypatch):
    pipe = Source()
    capture = runtime.capture_source(pipe, settings())

    def fail_decode(*args, **kwargs):
        raise RuntimeError("fake VAE decode failure")

    monkeypatch.setattr(pipe.vae, "decode", fail_decode)
    counters = {}
    with pytest.raises(runtime.RuntimeStageError, match="fake VAE decode failure") as caught:
        runtime.trace_source_suffix(pipe, capture, grad_enabled=True, counters=counters)
    assert "scheduler_output" in caught.value.partial["steps"][0]
    assert "pre_vae_latents" in caught.value.partial and "decoded" not in caught.value.partial
    assert counters["sd35_suffix_decode_calls"] == 1
    assert counters.get("sd35_suffix_decode_returned", 0) == 0
    assert counters.get("sd35_suffix_vjp_calls", 0) == 0


@pytest.mark.parametrize("invalid", ["mode", "steps"])
def test_trace_rejects_unplanned_mode_or_suffix_before_forward(invalid):
    pipe = Source()
    capture = runtime.capture_source(pipe, settings())
    if invalid == "steps":
        capture.remaining_steps = (18, 19)
    counters = {}
    with pytest.raises(runtime.RuntimeStageError):
        runtime.trace_source_suffix(pipe, capture, grad_enabled=0 if invalid == "mode" else True, counters=counters)
    assert counters == {}
