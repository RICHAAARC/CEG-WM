"""Guarded source-weight references: values, layouts and mutation boundaries."""
import pytest
import torch

from cegwm.runtime.channel_precompensation_sd35 import _cpu_saved_tensors

pytestmark = pytest.mark.unit


def fixed():
    module = torch.nn.Linear(4, 4, bias=False, dtype=torch.float64).requires_grad_(False).eval()
    module.weight.copy_(torch.arange(16, dtype=torch.float64).reshape(4, 4) / 20)
    return module


def test_reference_and_copy_forward_gradient_equal_without_weight_cpu_copy():
    module = fixed()
    outputs = []
    for use_reference in (False, True):
        x = torch.ones(2, 4, dtype=torch.float64, requires_grad=True)
        with _cpu_saved_tensors(stable_modules=(module,), retain_stable_storage=use_reference) as stats:
            y = sum(torch.sin(x @ module.weight.T) + x @ module.weight for _ in range(3))
            gradient, = torch.autograd.grad(y.sum(), x)
        outputs.append((y.detach(), gradient, stats))
    assert torch.equal(outputs[0][0], outputs[1][0])
    assert torch.equal(outputs[0][1], outputs[1][1])
    old, new = outputs[0][2], outputs[1][2]
    assert new["stable_parameter_cpu_copies"] == 0
    assert new["stable_parameter_reference_saves"] == 6
    assert new["stable_parameter_cpu_bytes_avoided"] == old["stable_parameter_cpu_copy_bytes"] > 0
    assert new["activation_cpu_copy_bytes"] == old["activation_cpu_copy_bytes"] > 0


@pytest.mark.parametrize("layout,reference", [
    ("whole", True), ("transpose", True), ("prefix", True),
    ("slice_holes", False), ("offset", False), ("overlap", False)])
def test_saved_view_layout_matches_original_copy_semantics(layout, reference):
    module = fixed()
    views = dict(whole=module.weight, transpose=module.weight.T, prefix=module.weight[:2],
                 slice_holes=module.weight[:, ::2], offset=module.weight[1:],
                 overlap=module.weight.as_strided((4, 4), (0, 1)))
    view = views[layout]
    observed = []
    class UseSaved(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, weight):
            ctx.save_for_backward(weight)
            return x @ weight
        @staticmethod
        def backward(ctx, dy):
            weight, = ctx.saved_tensors
            observed.append((weight.stride(), weight.untyped_storage().data_ptr()))
            return dy @ weight.T, None
    results = []
    for flag in (False, True):
        x = torch.ones(2, view.shape[0], dtype=torch.float64, requires_grad=True)
        with _cpu_saved_tensors(stable_modules=(module,), retain_stable_storage=flag) as stats:
            y = UseSaved.apply(x, view)
            gradient, = torch.autograd.grad(y.sum(), x)
        results.append((y.detach(), gradient, stats))
    assert observed[0][0] == observed[1][0] == view.to("cpu", copy=True).stride()
    assert (observed[1][1] == view.untyped_storage().data_ptr()) is reference
    assert torch.equal(results[0][0], results[1][0]) and torch.equal(results[0][1], results[1][1])
    assert bool(results[1][2]["layout_copy_fallbacks"]) is not reference


@pytest.mark.parametrize("mutation", ["inplace", "data_replace", "parameter_replace", "dtype"])
def test_reference_rejects_weight_mutation_instead_of_silent_gradient_change(mutation):
    module = fixed()
    with pytest.raises(RuntimeError, match="stable saved weight|stable weight"):
        with _cpu_saved_tensors(stable_modules=(module,), retain_stable_storage=True):
            x = torch.ones(1, 4, dtype=torch.float64, requires_grad=True)
            y = module(x)
            if mutation == "inplace":
                module.weight.add_(1)
            elif mutation == "data_replace":
                module.weight.data = module.weight.detach().clone()
            elif mutation == "parameter_replace":
                module.weight = torch.nn.Parameter(module.weight.clone(), requires_grad=False)
            else:
                module.to(torch.float32)
            torch.autograd.grad(y.sum(), x)


def test_reference_must_not_escape_guarded_backward_lifetime():
    module = fixed()
    x = torch.ones(1, 4, dtype=torch.float64, requires_grad=True)
    with _cpu_saved_tensors(stable_modules=(module,), retain_stable_storage=True):
        y = module(x)
    with pytest.raises(RuntimeError, match="outlived"):
        torch.autograd.grad(y.sum(), x)
