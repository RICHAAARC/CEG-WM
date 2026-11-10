"""One bounded 192-MiB CPU saved-graph comparison; diagnostic, not a model run."""
import argparse
from contextlib import nullcontext
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def measure(mode):
    root = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(root / "src"), str(root)]
    import torch
    from cegwm.runtime.channel_precompensation_sd35 import _cpu_saved_tensors
    from cegwm.runtime.stage_protection_disk import DiskSavedTensors
    from cegwm.runtime.stage_protection_process import memory_snapshot
    torch.set_num_threads(1)
    class Save(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, buffer):
            ctx.save_for_backward(buffer)
            return x + buffer[0]
        @staticmethod
        def backward(ctx, dy):
            buffer, = ctx.saved_tensors
            return dy * (buffer[0] + 1), None
    with tempfile.TemporaryDirectory(prefix="ceg-saved-graph-") as temporary:
        context = DiskSavedTensors(temporary) if mode == "disk" else nullcontext(None)
        with context as disk:
            before = memory_snapshot()
            x = torch.ones((), dtype=torch.float64, requires_grad=True)
            y = x
            with _cpu_saved_tensors(disk_storage=disk) as counts:
                for index in range(12):
                    buffer = torch.full((4 * 1024**2,), float(index), dtype=torch.float32)
                    y = Save.apply(y, buffer)
                    del buffer
                gc.collect()
                saved = memory_snapshot()
                y.backward()
            return dict(mode=mode, pid=os.getpid(), total_saved_bytes=192 * 1024**2,
                tensor_bytes=16 * 1024**2, steps=12, before=before, forward_saved=saved,
                forward_rss_increment_bytes=saved["rss_bytes"] - before["rss_bytes"],
                after_backward=memory_snapshot(), value=float(y.detach()), gradient=float(x.grad),
                cpu_copy_counts=counts, disk_counts=dict(disk.stats) if disk else None,
                scope="CPU diagnostic; RSS excludes kernel file cache, does not prove CUDA feasibility")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("cpu", "disk"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.mode:
        print(json.dumps(measure(args.mode)))
        return
    rows = []
    for mode in ("cpu", "disk"):
        child = subprocess.run([sys.executable, "-m", "scripts.measure_stage_saved_tensor_memory", "--mode", mode],
                               capture_output=True, text=True)
        if child.returncode:
            raise RuntimeError(child.stderr)
        rows.append(json.loads(child.stdout))
    report = dict(observations=rows, gradient_equal=rows[0]["gradient"] == rows[1]["gradient"],
                  value_equal=rows[0]["value"] == rows[1]["value"], threshold_gate=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
