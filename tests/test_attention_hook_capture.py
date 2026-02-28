"""
File purpose: attention hook 运行时捕获回归测试。
Module type: General module
"""

import pytest

from main.diffusion.sd3.hooks import register_attention_hooks, remove_attention_hooks


def test_attention_hook_capture_collects_runtime_maps() -> None:
    torch = pytest.importorskip("torch")
    nn = pytest.importorskip("torch.nn")

    class FakeAttentionBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.to_q = nn.Identity()
            self.to_k = nn.Identity()

        def forward(self, hidden_states):
            _ = self.to_q(hidden_states)
            _ = self.to_k(hidden_states)
            return hidden_states

    class FakeTransformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.block = FakeAttentionBlock()

        def forward(self, hidden_states):
            return self.block(hidden_states)

    class FakePipeline:
        def __init__(self) -> None:
            self.transformer = FakeTransformer()

    pipeline = FakePipeline()
    cfg = {"detect": {"geometry": {"attention_capture_max_layers": 1}}}

    hook_handle = register_attention_hooks(pipeline, cfg)
    hidden_states = torch.randn(1, 4, 8)
    _ = pipeline.transformer(hidden_states)

    runtime_maps = hook_handle.collect()
    remove_attention_hooks(hook_handle)

    assert runtime_maps is not None
    assert hasattr(runtime_maps, "shape")
    assert len(runtime_maps.shape) >= 3
    assert hook_handle._hook_handles == []
