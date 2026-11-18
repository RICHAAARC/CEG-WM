"""Engineering execution identity, separate from the frozen scientific budget."""
from dataclasses import asdict, dataclass
import hashlib
import json
import os


@dataclass(frozen=True)
class ExecutionProfile:
    name: str = "colab-float32-micro1-checkpoint-v1"
    microbatch: int = 1
    checkpoint_vae: bool = True
    recovery_every: int = 100
    cublas_workspace: str = ":4096:8"
    deterministic_algorithms: bool = True
    cudnn_benchmark: bool = False
    allow_tf32: bool = False
    float32_matmul_precision: str = "highest"

    @property
    def digest(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()

    def validate(self, config):
        if self.microbatch < 1 or config.batch_size % self.microbatch or self.recovery_every < 1:
            raise ValueError("execution profile must divide effective batch and have positive recovery interval")


def legacy_profile(config):
    return ExecutionProfile("batch-v1", config.batch_size, False, config.checkpoint_every)


def configure_torch(device="cpu", torch_module=None):
    """Shared numerical settings for clean creation, train and every observation."""
    if str(device).split(":")[0] == "cuda" and os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise ValueError("CUBLAS_WORKSPACE_CONFIG=:4096:8 must be set before Python/CUDA startup")
    if torch_module is None:
        import torch as torch_module
    torch_module.use_deterministic_algorithms(True)
    torch_module.set_float32_matmul_precision("highest")
    torch_module.backends.cudnn.benchmark = False
    torch_module.backends.cudnn.deterministic = True
    torch_module.backends.cudnn.allow_tf32 = False
    torch_module.backends.cuda.matmul.allow_tf32 = False
