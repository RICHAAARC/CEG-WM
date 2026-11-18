"""Safe-by-default CLI; execution is always explicit and local-only."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path

from .config import ARMS, Config, DCT_ORDER, SHIFTS, load_config


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    for name in ("plan", "validate", "train", "select", "evaluate", "diagnose"):
        command = sub.add_parser(name)
        command.add_argument("--execute", action="store_true")
        command.add_argument("--config")
        command.add_argument("--manifest")
        command.add_argument("--device", default="cpu")
        command.add_argument("--output")
        if name in ("train", "select"):
            command.add_argument("--run-dir")
        if name == "train":
            command.add_argument("--arm", choices=[*ARMS, "all"], default="all")
            command.add_argument("--resume")
            command.add_argument("--stop-after", type=int)
        if name in ("evaluate", "diagnose"):
            command.add_argument("--selection")
        if name == "evaluate":
            command.add_argument("--split", choices=["dev", "test"], default="dev")
            command.add_argument("--dev-report")
            command.add_argument("--lpips-weights")
    return root


def plan(config):
    from .networks import parameter_counts
    return {"mode": "plan_only", "config": asdict(config), "config_hash": config.digest,
            "DCT_order": DCT_ORDER, "attacks_dx_dy": SHIFTS, "parameters": parameter_counts(config),
            "architecture": "3 Conv3x3+SiLU, same-zero padding, radius3; Conv1x1 heads; width64",
            "split_roles": {"train": "optimizer updates", "dev": "A investment gate and optional B pilot",
                            "model_selection": "checkpoint ranking only; top1, bit accuracy, earliest update",
                            "calibration": "reserved, unused; no FPR claim", "test": "locked final report, never tuning"},
            "suggested_source_counts": {"train": 256, "dev": 64, "model_selection": 64, "calibration": 128, "test": 128},
            "execution": "No model loaded. Supply --manifest and --execute for actions.",
            "status": "engineering candidate only; no scientific validation"}


def required(args, *names):
    for name in names:
        if not getattr(args, name, None):
            raise ValueError("--" + name.replace("_", "-") + " is required with --execute")


def main(argv=None):
    args = parser().parse_args(argv)
    config = load_config(args.config)
    if args.command == "plan" or not args.execute:
        result = plan(config)
    else:
        required(args, "manifest")
        if args.command == "train" and args.device.split(":")[0] == "cuda" and os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
            raise ValueError("deterministic CUDA training requires CUBLAS_WORKSPACE_CONFIG=:4096:8 before starting Python; no model loaded")
        from .data import Cache, file_hash, json_write
        cache = Cache(args.manifest, config)
        if args.command == "validate":
            result = cache.report()
        else:
            from .adapter import load_adapter, load_lpips
            from .engine import train, select_checkpoints, evaluate, four_write_diagnostic
            if args.command in ("train", "select"):
                required(args, "run_dir")
            if args.command in ("select", "evaluate", "diagnose"):
                required(args, "output")
            if args.command in ("evaluate", "diagnose"):
                required(args, "selection")
            adapter = load_adapter(cache.model_dir, config, args.device)
            if args.command == "train":
                if args.resume and args.arm == "all":
                    raise ValueError("resume one explicit arm at a time")
                result = {}
                for arm in ARMS if args.arm == "all" else (args.arm,):
                    path = train(cache, config, adapter, arm, Path(args.run_dir)/arm, device=args.device,
                                 resume=args.resume, stop_after=args.stop_after)
                    result[arm] = {"checkpoint": str(path.resolve()), "sha256": file_hash(path)}
            elif args.command == "select":
                result = select_checkpoints(cache, config, adapter, args.run_dir, args.output, device=args.device)
            elif args.command == "evaluate":
                metric = None
                metric_info = {"status": "unavailable", "sha256": None}
                if args.lpips_weights:
                    try:
                        metric_info["sha256"] = file_hash(args.lpips_weights)
                        metric = load_lpips(args.lpips_weights, args.device)
                        metric_info["status"] = "loaded_local_full_alexnet_state"
                    except (ImportError, OSError, RuntimeError, ValueError) as error:
                        metric_info.update({"status": "unavailable", "error": f"{type(error).__name__}: {error}"})
                result = evaluate(cache, config, adapter, args.selection, args.split, args.output, device=args.device,
                                  metric=metric, dev_report=args.dev_report, metric_info=metric_info)
            else:
                result = four_write_diagnostic(cache, config, adapter, args.selection, args.output, device=args.device)
    print(json.dumps(result, indent=2, allow_nan=False))
    return result


if __name__ == "__main__":
    main()
